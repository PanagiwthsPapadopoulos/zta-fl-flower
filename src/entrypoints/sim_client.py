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
from src.shared.utils.hardware import get_device, get_dataloader_kwargs
from src.tier_edge.local_ids import EdgeTrainer
from src.tier_edge.byzantine_simulator import AdversaryManager


class Client(NumPyClient):
    """Executes the ground-level hustle for the federation."""
    
    def __init__(self, logger, log_prefix: str, fog_num: int, edge_num: int, model=None, train_loader=None, device="cpu", dataset_metadata=None, train_config=None):
        self.logger = logger
        self.log_prefix = log_prefix
        self.fog_num = fog_num
        self.edge_num = edge_num
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_metadata = dataset_metadata or {}
        self.train_config = train_config or {}
        
        # Init AdversaryManager for injecting threats
        self.adversary_manager = AdversaryManager(
            fog_num=self.fog_num, 
            edge_num=self.edge_num, 
            logger=self.logger
        )
        self.train_loader = self.adversary_manager.corrupt_data_if_needed(self.train_loader)
        
        # Init EdgeTrainer
        self.trainer = EdgeTrainer(
            logger=self.logger, log_prefix=self.log_prefix, model=self.model,
            train_loader=self.train_loader, device=self.device,
            train_config=self.train_config, dataset_metadata=self.dataset_metadata
        )

    def get_parameters(self, config: dict) -> list:
        return self.trainer.get_parameters()

    def fit(self, parameters: list, config: dict):
        try:
            current_round = config["server_round"]
            round_dump = {
                "round": current_round,
                "log_prefix": self.log_prefix,
                "server_provided_config": config,
                "local_train_config": self.train_config,
                "dataset_metadata": self.dataset_metadata,
                "payload_params_count": len(parameters) if parameters else 0
            }
            self.logger.debug(f"Variable Dump (ROUND {current_round}): {json.dumps(round_dump, indent=2, default=str)}", extra={"round": current_round})
            
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
        return 0.0, 1, {"accuracy": 1.0}


def client_fn(context: Context):
    from src.shared.utils.config_loader import get_merged_config
    run_config = get_merged_config(context.run_config)
    node_config = context.node_config
    device = get_device()
    
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

    # Ray gives 0-indexed partition ID. 
    # Mathematically determine virtual fog cluster and edge id based on topology array.
    partition_id = int(node_config["partition-id"])
    
    custom_top_str = str(run_config.get("custom_fog_topology", ""))
    custom_topology = ast.literal_eval(custom_top_str) if custom_top_str.strip() else []
    num_fogs = int(run_config["num_fogs"])
    uniform_edges = int(run_config["uniform_edges_per_fog"])
    
    if custom_topology and len(custom_topology) > 0:
        topology = custom_topology
    else:
        topology = [uniform_edges] * num_fogs
        
    current_count = 0
    fog_num = 1
    edge_num = 1
    for f_idx, e_count in enumerate(topology):
        if partition_id < current_count + e_count:
            fog_num = f_idx + 1
            edge_num = (partition_id - current_count) + 1
            break
        current_count += e_count

    global_index = partition_id
    total_edges = sum(topology)

    # Point at the correct hardware emulator sidecar
    os.environ["TPM2TOOLS_TCTI"] = f"swtpm:host=tpm-{fog_num}-{edge_num},port=2321"

    log_prefix = f"[EDGE {fog_num}_{edge_num}]"
    logger = setup_logger(log_prefix)

    try:
        from src.shared.security.threat_profiler import assign_edge_roles
        role = assign_edge_roles(run_config, total_edges, global_index, random_seed, logger)
        
        train_config = {
            "role": role,
            "learning_rate": float(run_config["learning_rate"]),
            "quantization_bits": int(run_config["quantization_bits"]),
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
            "shap_val_samples": int(run_config["shap_val_samples"])
        }
        
        if role == "benign":
            train_config["adv_ratio"] = float(run_config["benign_adv_ratio"])
            train_config["eps"] = float(run_config["benign_eps"])
            train_config["alpha"] = float(run_config["benign_alpha"])
            train_config["n_iter"] = int(run_config["benign_n_iter"])

        if role == "label_flip":
            train_config["p_flip"] = float(run_config["p_flip"])
        if role == "gradient_manip":
            train_config["alpha"] = float(run_config["gradient_alpha"])
        if role == "shap_aware":
            train_config["p_flip"] = float(run_config["p_flip"])
            train_config["alpha"] = float(run_config["gradient_alpha"])
            
        apply_smote = run_config["apply_smote"]
        simulate_leakage = run_config["simulate_global_leakage"]
        
        # Fetch pre-partitioned edge artifact instantly from disk
        X_part, y_part, n_classes_eval = get_dataset(
            dataset_name, dataset_path, num_classes, random_seed, 
            simulate_global_leakage=simulate_leakage, 
            apply_smote=apply_smote, split=f"edge_{global_index}", 
            test_split=test_split, val_split=val_split
        )
        
        if role == "backdoor":
            poison_fraction = float(run_config["backdoor_poison_fraction"])
            target_class = int(run_config["backdoor_target_class"])
            trigger_value = float(run_config["backdoor_trigger_value"])
            raw_features = run_config["backdoor_trigger_features"]
            trigger_features = ast.literal_eval(raw_features) if isinstance(raw_features, str) else raw_features
            
            logger.info(f"Performing Backdoor attack! Backdoor poison fraction: {poison_fraction}, Target class: {target_class}, Backdoor Trigger value: {trigger_value}, Backdoor Trigger features: {trigger_features}")
            X_part, y_part = poison_partition(
                X=X_part, y=y_part, poison_fraction=poison_fraction, target_class=target_class,
                trigger_features=tuple(trigger_features), trigger_value=trigger_value, seed=(random_seed + global_index) 
            )

        if not isinstance(X_part, torch.Tensor):
            X_part = torch.tensor(X_part, dtype=torch.float32)
            
        dataset = TensorDataset(X_part, y_part)
        batch_size = int(run_config["batch_size"])
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **get_dataloader_kwargs())
        
        classes_present, counts = torch.unique(y_part, return_counts=True)
        dataset_metadata = {
            "dataset_name": dataset_name,
            "role": role,
            "total_samples": len(y_part),
            "unique_classes": len(classes_present),
            "distribution": str(dict(zip(classes_present.tolist(), counts.tolist()))),
            "hardware_device": device
        }
        logger.info(f"STATIC ROLE: {role.upper()} | Global Index: {global_index}/{total_edges-1} | Network Topology: {topology}")
        
    except Exception as e:
        logger.error(f"FATAL CRASH DURING BOOT: {e}\n{traceback.format_exc()}")
        raise e

    model = get_model(model_arch, n_features, num_classes)
    
    return Client(
        logger=logger, log_prefix=log_prefix, fog_num=fog_num, edge_num=edge_num,
        model=model, train_loader=train_loader, device=device, 
        dataset_metadata=dataset_metadata, train_config=train_config,
    ).to_client()


app = ClientApp(client_fn=client_fn)