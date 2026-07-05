import ast
import os
import traceback
import torch
from torch.utils.data import DataLoader, TensorDataset

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from src.shared.models.factory import get_model
from src.shared.network.compression import compress_weights
from src.shared.data.data_loader import DATASET_METADATA, get_dataset, non_iid_partition
from src.shared.utils.logger_setup import setup_logger
from src.shared.security.backdoor_math import poison_partition

GLOBAL_DATA_CACHE = {}


class Client(NumPyClient):
    """Executes the ground-level hustle for the federation.
    
    Serves as the primary operational node that runs local epochs on the raw telemetry,
    injects adversarial static when acting as a rogue agent, and fires the final quantized
    gradient payloads back up the chain.
    """
    
    def __init__(self, logger, node_type: str, log_prefix: str, fog_num: int, edge_num: int, model=None, train_loader=None, device="cpu", dataset_metadata=None, train_config=None):
        """Initializes the Federated Learning Client with explicit node identity, logging prefixes, and local configurations."""
        self.logger = logger
        self.node_type = node_type
        self.log_prefix = log_prefix
        self.fog_num = fog_num
        self.edge_num = edge_num
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_metadata = dataset_metadata or {}
        self.train_config = train_config or {}
        
        self.broker_ip = self.train_config.get("broker_ip", "127.0.0.1")
        self.socket_timeout = self.train_config.get("socket_timeout", 600.0)
        
        if self.node_type == "fog_client":
            fog_ipc_base = int(self.train_config.get("fog_ipc_base", 10000))
            self.ipc_port = fog_ipc_base + self.fog_num
        elif self.node_type == "edge":
            from src.tier_edge.local_ids import EdgeTrainer
            from src.tier_edge.byzantine_simulator import AdversaryManager

            self.adversary_manager = AdversaryManager(
                fog_num=self.fog_num, 
                edge_num=self.edge_num, 
                logger=self.logger
            )

            self.train_loader = self.adversary_manager.corrupt_data_if_needed(self.train_loader)

            self.trainer = EdgeTrainer(
                logger=self.logger, log_prefix=self.log_prefix, model=self.model,
                train_loader=self.train_loader, device=self.device,
                train_config=self.train_config, dataset_metadata=self.dataset_metadata
            )

    def get_parameters(self, config: dict) -> list:
        """Extracts and fully compresses the local model parameters for network transmission to the server."""
        if self.node_type == "edge":
            return self.trainer.get_parameters()
            
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        bits = int(self.train_config.get("quantization_bits", 32))
        return compress_weights(weights, bits)

    def fit(self, parameters: list, config: dict):
        """Executes the local training round and returns the sequentially updated, optionally corrupted, network parameters."""
        try:
            current_round = config.get("server_round", 0)
            strategy = config.get("strategy", "fedavg")
            
            if self.node_type == "fog_client":
                from src.tier_edge.fog_bridge_client import FogBridgeClient
                bridge = FogBridgeClient(self.logger, self.log_prefix, self.ipc_port, self.socket_timeout)
                return bridge.execute_round(current_round)
            elif self.node_type == "edge":                
                self.adversary_manager.current_round = current_round
                
                res_params, num_examples, metrics = self.trainer.execute_training(parameters, current_round, strategy, config)
                
                metrics["log_prefix"] = self.log_prefix
                res_params, metrics = self.adversary_manager.corrupt_payload_if_needed(res_params, metrics)
                return res_params, num_examples, metrics                
        except Exception as e:
            self.logger.error(f"{self.log_prefix} CRITICAL SILENT CRASH: {e}\n{traceback.format_exc()}", extra={"round": current_round})
            return [], 0, {"status": "crashed"}

    def evaluate(self, parameters: list, config: dict):
        """Bypasses local evaluation to preserve computational resources on constrained edge environments."""
        return 0.0, 1, {"accuracy": 1.0}


def _build_fog_client(run_config: dict, node_config: dict):
    """Constructs the configuration, network IPC, and logging environment specifically for a Fog Client node."""
    raw_fog_val = str(node_config.get("fog_id", "0"))
    fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
    node_type = "fog_client" 
    log_prefix = f"[FOG {fog_num} CLIENT]"

    train_config = {
        "broker_ip": str(run_config.get("broker_ip", "127.0.0.1")),
        "socket_timeout": float(run_config.get("socket_timeout", 600.0)),
        "fog_ipc_base": int(run_config.get("fog_ipc_base", 10000))
    }
    
    logger = setup_logger(log_prefix)
    return logger, node_type, log_prefix, fog_num, train_config


def client_fn(context: Context):
    """Initializes and securely constructs the complete Client instance based on the provided framework context."""
    from src.shared.utils.config_loader import get_merged_config
    run_config = get_merged_config(context.run_config)
    node_config = context.node_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader = None
    dataset_metadata = None
    train_config = None

    strategy = str(run_config.get("strategy", "zta"))
    dataset_name = str(run_config.get("dataset", "edge_iiotset")).lower()
    
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset requested: {dataset_name}")
        
    dataset_path = str(run_config.get("dataset_path", "data/edge_iiotset/raw/network_traffic_samples.csv"))
    dataset_fraction = float(run_config.get("dataset_fraction", 1.0))
    num_classes = DATASET_METADATA[dataset_name]["classes"]
    n_features = DATASET_METADATA[dataset_name]["features"]
    model_arch = str(run_config.get("model_architecture", "cnnlstm"))
    random_seed = int(run_config.get("random_seed", 42))
    test_split = float(run_config.get("test_split", 0.30))
    val_split = float(run_config.get("val_split", 0.50))

    edge_num = 0
    
    if "partition-id" in node_config:
        partition_id = int(node_config["partition-id"]) 
        edge_num = partition_id
        internal_id = partition_id - 1 
        
        raw_fog_val = str(node_config.get("fog_num", "0"))
        fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
        
        node_type = "edge"
        log_prefix = f"[EDGE {fog_num}_{partition_id}]" 
        logger = setup_logger(log_prefix)

        try: 
            custom_top_str = str(run_config.get("custom_fog_topology", "[]"))
            custom_topology = ast.literal_eval(custom_top_str) if custom_top_str.strip() else []
            num_fogs = int(run_config.get("num_fogs", 2))
            uniform_edges = int(run_config.get("uniform_edges_per_fog", 2))
            master_seed = int(run_config.get("random_seed", 42))

            if custom_topology and len(custom_topology) > 0:
                topology = custom_topology
            else:
                topology = [uniform_edges] * num_fogs

            total_edges = sum(topology) 
            edges_before_me = sum(topology[:fog_num-1]) if fog_num > 1 else 0
            global_index = edges_before_me + internal_id 

            from src.shared.security.threat_profiler import assign_edge_roles
            role = assign_edge_roles(run_config, total_edges, global_index, master_seed, logger)
                
            train_config = {
                "role": role,
                "learning_rate": float(run_config.get("learning_rate", 0.001)),
                "quantization_bits": int(run_config.get("quantization_bits", 32)),
                "broker_ip": str(run_config.get("broker_ip", "127.0.0.1")),
                "socket_timeout": float(run_config.get("socket_timeout", 600.0)),
                "fog_ipc_base": int(run_config.get("fog_ipc_base", 10000)),
                "n_features": n_features,
                "num_classes": num_classes,
                "local_epochs": int(run_config.get("local_epochs", 1)),
                "clip_min": float(run_config.get("clip_min", 0.0)),
                "clip_max": float(run_config.get("clip_max", 1.0)),
                "clip_norm": float(run_config.get("clip_norm", 1.0)),
                "fedprox_mu": float(run_config.get("fedprox_mu", 0.01)),
                "shap_threshold": float(run_config.get("shap_threshold", 0.15)),
                "shap_aware_base_attack": str(run_config.get("shap_aware_base_attack", "label_flip")),
                "robustness_eval_attack": str(run_config.get("robustness_eval_attack", "pgd")),
                "pgd_n_iter": int(run_config.get("pgd_n_iter", 7)),
                "shap_explain_count": int(run_config.get("shap_explain_count", 15)),
                "shap_val_samples": int(run_config.get("shap_val_samples", 100)),
            }
            
            if role in ["pgd", "fgsm"]:
                train_config["adv_ratio"] = float(run_config.get(f"{role}_adv_ratio", 0.3))
                train_config["eps"] = float(run_config.get(f"{role}_eps", 0.1))
                train_config["alpha"] = float(run_config.get(f"pgd_alpha", 0.01))
                train_config["n_iter"] = int(run_config.get(f"pgd_n_iter", 7))
            elif role == "benign":
                train_config["adv_ratio"] = float(run_config.get("benign_adv_ratio", 0.3))
                train_config["eps"] = float(run_config.get("benign_eps", 0.05))
                train_config["alpha"] = float(run_config.get("benign_alpha", 0.01))
                train_config["n_iter"] = int(run_config.get("benign_n_iter", 3))
            
            if role == "label_flip":
                train_config["p_flip"] = float(run_config.get("p_flip", 1.0))
            if role == "gradient_manip":
                train_config["alpha"] = float(run_config.get("gradient_alpha", 5.0))
            if role == "shap_aware":
                train_config["p_flip"] = float(run_config.get("p_flip", 1.0))
                train_config["alpha"] = float(run_config.get("gradient_alpha", 5.0))
            
            global GLOBAL_DATA_CACHE
            
            apply_smote = run_config.get("apply_smote", False)
            simulate_leakage = run_config.get("simulate_global_leakage", False)
            n_classes_per = int(run_config.get("n_classes_per", 3))

            cache_key = f"train_{dataset_name}_{dataset_fraction}_{simulate_leakage}_{apply_smote}"
            
            logger.debug(f"[CONFIG USAGE] get_dataset | simulate_global_leakage: {simulate_leakage}, apply_smote: {apply_smote}, n_classes_per: {n_classes_per}")
            
            if cache_key not in GLOBAL_DATA_CACHE:
                X_full, y_full, n_classes_eval = get_dataset(
                    dataset_name, dataset_path, num_classes, random_seed, 
                    simulate_global_leakage=simulate_leakage, 
                    apply_smote=apply_smote, split="train", 
                    test_split=test_split, val_split=val_split
                )
                    
                generator = torch.Generator().manual_seed(random_seed)
                indices = torch.randperm(len(X_full), generator=generator)
                X_full = X_full[indices]
                y_full = y_full[indices]

                if dataset_fraction < 1.0:
                    subset_size = int(len(X_full) * dataset_fraction)
                    X_full = X_full[:subset_size]
                    y_full = y_full[:subset_size]
                    
                GLOBAL_DATA_CACHE[cache_key] = (X_full, y_full, n_classes_eval)
            else:
                X_full, y_full, n_classes_eval = GLOBAL_DATA_CACHE[cache_key]

            power_law_a = float(run_config.get("power_law_a", 0.4))
            partitions = non_iid_partition(X=X_full, y=y_full, n_agents=total_edges, n_classes_per=n_classes_per, power_law_a=power_law_a, seed=master_seed)
            X_part, y_part = partitions[global_index % len(partitions)]
            
            if role == "backdoor":
                poison_fraction = float(run_config.get("backdoor_poison_fraction", 0.5))
                target_class = int(run_config.get("backdoor_target_class", 0))
                trigger_value = float(run_config.get("backdoor_trigger_value", 1.5))
                raw_features = run_config.get("backdoor_trigger_features", "[-3, -2, -1]")
                
                trigger_features = ast.literal_eval(raw_features) if isinstance(raw_features, str) else raw_features
                    
                logger.info(f"Performing Backdoor attack! Backdoor poison fraction: {poison_fraction}, Target class: {target_class}, Backdoor Trigger value: {trigger_value}, Backdoor Trigger features: {trigger_features}")
                
                X_part, y_part = poison_partition(
                    X=X_part, y=y_part, poison_fraction=poison_fraction, target_class=target_class,
                    trigger_features=tuple(trigger_features), trigger_value=trigger_value, seed=(master_seed + global_index) 
                )
            
            if not isinstance(X_part, torch.Tensor):
                X_part = torch.tensor(X_part, dtype=torch.float32)

            dataset = TensorDataset(X_part, y_part)
            batch_size = int(run_config.get("batch_size", 32))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            classes_present, counts = torch.unique(y_part, return_counts=True)
            dataset_metadata = {
                "dataset_name": dataset_name,
                "role": role,
                "total_samples": len(y_part),
                "unique_classes": len(classes_present),
                "distribution": str(dict(zip(classes_present.tolist(), counts.tolist()))) 
            }
            logger.info(f"STATIC ROLE: {role.upper()} | Global Index: {global_index}/{total_edges-1} | Network Topology: {topology}")
            
        except Exception as e:
            logger.error(f"FATAL CRASH DURING BOOT: {e}\n{traceback.format_exc()}")
            raise e
        
    else:
        logger, node_type, log_prefix, fog_num, train_config = _build_fog_client(run_config, node_config)

    model = get_model(model_arch, n_features, num_classes)
    
    return Client(
        logger=logger, node_type=node_type, log_prefix=log_prefix, fog_num=fog_num, 
        model=model, train_loader=train_loader, device=device, 
        dataset_metadata=dataset_metadata, train_config=train_config, edge_num=edge_num,
    ).to_client()


app = ClientApp(client_fn=client_fn)