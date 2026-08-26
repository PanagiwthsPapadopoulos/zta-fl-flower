import ast
import os
import time
import traceback
import torch
import json
from torch.utils.data import DataLoader, TensorDataset

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from src.shared.models.factory import get_model
from src.shared.network.compression import compress_weights
from src.shared.data.data_loader import DATASET_METADATA, get_dataset, non_iid_partition
from src.shared.utils.logger_setup import setup_logger
from src.shared.security.backdoor_math import poison_partition



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
        
        self.broker_ip = self.train_config["broker_ip"]
        self.socket_timeout = self.train_config["socket_timeout"]
        
        if self.node_type == "fog_client":
            fog_ipc_base = int(self.train_config["fog_ipc_base"])
            self.ipc_port = fog_ipc_base + self.fog_num
        elif self.node_type == "edge":
            from src.tier_edge.local_ids import EdgeTrainer
            from src.tier_edge.byzantine_simulator import AdversaryManager

            # Init AdversaryManager for injecting threats configured in config/threat.yaml
            self.adversary_manager = AdversaryManager(
                fog_num=self.fog_num, 
                edge_num=self.edge_num, 
                logger=self.logger
            )

            self.train_loader = self.adversary_manager.corrupt_data_if_needed(self.train_loader)

            # Init EdgeTrainer for training on edge devices
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
        bits = int(self.train_config["quantization_bits"])
        return compress_weights(weights, bits)

    def fit(self, parameters: list, config: dict):
        """Executes the local training round and returns the sequentially updated, optionally corrupted, network parameters."""
        try:
            current_round = config["server_round"]

            # Per-round variable dump
            round_dump = {
                "round": current_round,
                "node_type": self.node_type,
                "log_prefix": self.log_prefix,
                "server_provided_config": config,
                "local_train_config": self.train_config,
                "dataset_metadata": self.dataset_metadata,
                "payload_params_count": len(parameters) if parameters else 0
            }
            self.logger.debug(f"Variable Dump (ROUND {current_round}): {json.dumps(round_dump, indent=2, default=str)}")
            
            if self.node_type == "fog_client":
                from src.tier_edge.fog_bridge_client import FogBridgeClient
                bridge = FogBridgeClient(self.logger, self.log_prefix, self.ipc_port, self.socket_timeout)
                return bridge.execute_round(current_round)
            elif self.node_type == "edge":
                self.adversary_manager.current_round = current_round

                # Track Local Execution Timestamps & Latency
                train_start_unix = time.time()
                res_params, num_examples, metrics = self.trainer.execute_training(parameters, current_round, config)
                train_end_unix = time.time()
                
                metrics["local_adversarial_training_start_unix"] = train_start_unix
                metrics["local_adversarial_training_end_unix"] = train_end_unix
                metrics["latency_adv_training_sec"] = train_end_unix - train_start_unix

                metrics["log_prefix"] = self.log_prefix
                res_params, metrics = self.adversary_manager.corrupt_payload_if_needed(res_params, metrics)
                
                # Track Dynamic Payload Size
                payload_size_mb = sum(arr.nbytes for arr in res_params) / (1024 * 1024) if res_params else 0.0
                metrics["payload_size_mb"] = payload_size_mb
                
                return res_params, num_examples, metrics                
        except Exception as e:
            self.logger.error(f"{self.log_prefix} CRITICAL SILENT CRASH: {e}\n{traceback.format_exc()}", extra={"round": current_round})
            return [], 0, {"status": "crashed"}

    def evaluate(self, parameters: list, config: dict):
        """Bypasses local evaluation to preserve computational resources on constrained edge environments."""
        return 0.0, 1, {"accuracy": 1.0}


def _build_fog_client(run_config: dict, node_config: dict):
    """Constructs the configuration, network IPC, and logging environment specifically for a Fog Client node."""
    raw_fog_val = str(node_config["fog_id"])
    fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
    node_type = "fog_client" 
    log_prefix = f"[FOG {fog_num} CLIENT]"

    train_config = {
        "broker_ip": str(run_config["broker_ip"]),
        "socket_timeout": float(run_config["socket_timeout"]),
        "fog_ipc_base": int(run_config["fog_ipc_base"]),
        "quantization_bits": int(run_config["quantization_bits"])
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

    dataset_name = str(run_config["dataset"]).lower()
    
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset requested: {dataset_name}")
        
    dataset_path = str(run_config["dataset_path"])
    dataset_fraction = float(run_config["dataset_fraction"])
    num_classes = DATASET_METADATA[dataset_name]["classes"]
    n_features = DATASET_METADATA[dataset_name]["features"]
    model_arch = str(run_config["model_architecture"])
    random_seed = int(run_config["random_seed"])
    test_split = float(run_config["test_split"])
    val_split = float(run_config["val_split"])

    edge_num = 0
    
    if "partition-id" in node_config:
        partition_id = int(node_config["partition-id"]) 
        edge_num = partition_id
        internal_id = partition_id - 1 
        
        raw_fog_val = str(node_config["fog_num"])
        fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
        
        node_type = "edge"
        log_prefix = f"[EDGE {fog_num}_{partition_id}]" 
        logger = setup_logger(log_prefix)

        # Edge Node Logic
        try: 
            # Extract variables
            custom_top_str = str(run_config["custom_fog_topology"])
            custom_topology = ast.literal_eval(custom_top_str) if custom_top_str.strip() else []
            num_fogs = int(run_config["num_fogs"])
            uniform_edges = int(run_config["uniform_edges_per_fog"])
            master_seed = int(run_config["random_seed"])

            if custom_topology and len(custom_topology) > 0:
                topology = custom_topology
            else:
                topology = [uniform_edges] * num_fogs

            total_edges = sum(topology) 
            edges_before_me = sum(topology[:fog_num-1]) if fog_num > 1 else 0
            global_index = edges_before_me + internal_id 

            # Assign Roles
            from src.shared.security.threat_profiler import assign_edge_roles
            role = assign_edge_roles(run_config, total_edges, global_index, master_seed, logger)
                
            train_config = {
                "role": role,
                "learning_rate": float(run_config["learning_rate"]),
                "quantization_bits": int(run_config["quantization_bits"]),
                "broker_ip": str(run_config["broker_ip"]),
                "socket_timeout": float(run_config["socket_timeout"]),
                "fog_ipc_base": int(run_config["fog_ipc_base"]),
                "n_features": n_features,
                "num_classes": num_classes,
                "local_epochs": int(run_config["local_epochs"]),
                "clip_min": float(run_config["clip_min"]),
                "clip_max": float(run_config["clip_max"]),
                "clip_norm": float(run_config["clip_norm"]),
                "shap_threshold": float(run_config["shap_threshold"]),
                "shap_aware_base_attack": str(run_config["shap_aware_base_attack"]),
                "robustness_eval_attack": str(run_config["robustness_eval_attack"]),
                "shap_explain_count": int(run_config["shap_explain_count"]),
                "shap_val_samples": int(run_config["shap_val_samples"]),
            }
            
            # Load BENIGN role variables
            if role == "benign":
                train_config["adv_ratio"] = float(run_config["benign_adv_ratio"])
                train_config["eps"] = float(run_config["benign_eps"])
                train_config["alpha"] = float(run_config["benign_alpha"])
                train_config["n_iter"] = int(run_config["benign_n_iter"])
            
            # Load EXTRA roles variables
            if role == "label_flip":
                train_config["p_flip"] = float(run_config["p_flip"])
            if role == "gradient_manip":
                train_config["alpha"] = float(run_config["gradient_alpha"])
            if role == "shap_aware":
                train_config["p_flip"] = float(run_config["p_flip"])
                train_config["alpha"] = float(run_config["gradient_alpha"])
            
            apply_smote = run_config["apply_smote"]
            simulate_leakage = run_config["simulate_global_leakage"]
            n_classes_per = int(run_config["n_classes_per"])

            # Fetch instantly from disk artifact
            X_full, y_full, n_classes_eval = get_dataset(
                dataset_name, dataset_path, num_classes, random_seed, 
                simulate_global_leakage=simulate_leakage, 
                apply_smote=apply_smote, split="train", 
                test_split=test_split, val_split=val_split
            )
                        
            # Shuffle globally before partitioning
            generator = torch.Generator().manual_seed(random_seed)
            indices = torch.randperm(len(X_full), generator=generator)
            X_full = X_full[indices]
            y_full = y_full[indices]
            
            # Fraction the dataset
            if dataset_fraction < 1.0:
                subset_size = int(len(X_full) * dataset_fraction)
                X_full = X_full[:subset_size]
                y_full = y_full[:subset_size]

            # Split the data across the edge devices to simulate real world data traffic from different devices
            power_law_a = float(run_config["power_law_a"])
            partitions = non_iid_partition(X=X_full, y=y_full, n_agents=total_edges, n_classes_per=n_classes_per, power_law_a=power_law_a, seed=master_seed)
            X_part, y_part = partitions[global_index % len(partitions)]
            
            # After preparing the dataset, check if the node must perform a backdoor 
            # If so, poison the data
            if role == "backdoor":
                poison_fraction = float(run_config["backdoor_poison_fraction"])
                target_class = int(run_config["backdoor_target_class"])
                trigger_value = float(run_config["backdoor_trigger_value"])
                raw_features = run_config["backdoor_trigger_features"]
                
                trigger_features = ast.literal_eval(raw_features) if isinstance(raw_features, str) else raw_features
                    
                logger.info(f"Performing Backdoor attack! Backdoor poison fraction: {poison_fraction}, Target class: {target_class}, Backdoor Trigger value: {trigger_value}, Backdoor Trigger features: {trigger_features}")
                
                X_part, y_part = poison_partition(
                    X=X_part, y=y_part, poison_fraction=poison_fraction, target_class=target_class,
                    trigger_features=tuple(trigger_features), trigger_value=trigger_value, seed=(master_seed + global_index) 
                )
            
            if not isinstance(X_part, torch.Tensor):
                X_part = torch.tensor(X_part, dtype=torch.float32)

            dataset = TensorDataset(X_part, y_part)
            batch_size = int(run_config["batch_size"])
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
        
    # Fog Node Logic
    else:
        logger, node_type, log_prefix, fog_num, train_config = _build_fog_client(run_config, node_config)

    model = get_model(model_arch, n_features, num_classes)
    
    return Client(
        logger=logger, node_type=node_type, log_prefix=log_prefix, fog_num=fog_num, 
        model=model, train_loader=train_loader, device=device, 
        dataset_metadata=dataset_metadata, train_config=train_config, edge_num=edge_num,
    ).to_client()


app = ClientApp(client_fn=client_fn)