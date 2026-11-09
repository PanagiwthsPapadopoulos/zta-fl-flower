from src.shared.security.shap_aware import attack_validation_data
from src.tier_edge.inference import evaluation_loader
from src.shared.utils.config_loader import optional_float
import ast
import os
import time
import traceback
import gc
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from src.shared.models.factory import get_model
from src.shared.utils.reproducibility import seed_training
from src.shared.network.compression import compress_weights
from src.shared.data.data_loader import DATASET_METADATA, get_dataset, non_iid_partition
from src.shared.utils.logger_setup import setup_logger
from src.shared.security.backdoor_math import poison_partition
from src.shared.utils.hardware import get_device, get_dataloader_kwargs
from src.tier_edge.local_ids import EdgeTrainer


class Client(NumPyClient):
    """Executes the ground-level hustle for the federation."""
    
    def __init__(self, logger, log_prefix: str, fog_num: int, edge_num: int, model=None, train_loader=None, device="cpu", dataset_metadata=None, train_config=None, eval_loader=None, shap_data=None):
        self.logger = logger
        self.log_prefix = log_prefix
        self.fog_num = fog_num
        self.edge_num = edge_num
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_metadata = dataset_metadata or {}
        self.train_config = train_config or {}
        
        # Init EdgeTrainer
        self.trainer = EdgeTrainer(
            logger=self.logger, log_prefix=self.log_prefix, model=self.model,
            train_loader=self.train_loader, device=self.device,
            train_config=self.train_config, dataset_metadata=self.dataset_metadata, eval_loader=eval_loader, shap_data=shap_data
        )

    def get_properties(self, config: dict) -> dict:
        if not hasattr(self, "trainer"):
            return {}
        return self.trainer.attestation_metrics(int(config["server_round"]), config)

    def get_parameters(self, config: dict) -> list:
        return self.trainer.get_parameters()

    def fit(self, parameters: list, config: dict):
        current_round = config.get("server_round", 0)
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
            
            # Track Local Execution Timestamps & Latency
            train_start_unix = time.time()
            retried_on_cpu = False
            try:
                result = self.trainer.execute_training(parameters, current_round, config)
            except torch.OutOfMemoryError:
                if self.device != "cuda":
                    raise
                self.logger.warning("CUDA memory exhausted; retrying this client's complete update on CPU",
                                    extra={"round": current_round})
                retried_on_cpu = True
            if retried_on_cpu:
                # Leave the except block first so its traceback releases GPU tensors.
                self.model.to("cpu")
                self.device = self.trainer.device = "cpu"
                gc.collect()
                torch.cuda.empty_cache()
                # execute_training reloads the original global parameters, resets the
                # training seed, and generates the normal post-training TPM quote.
                result = self.trainer.execute_training(parameters, current_round, config)
            res_params, num_examples, metrics = result
            metrics["simulation_cpu_retry"] = retried_on_cpu
            train_end_unix = time.time()
            
            metrics.setdefault("local_adversarial_training_start_unix", train_start_unix)
            metrics.setdefault("local_adversarial_training_end_unix", train_end_unix)
            metrics.setdefault("latency_adv_training_sec", train_end_unix - train_start_unix)
            metrics["log_prefix"] = self.log_prefix
            
            # Track Dynamic Payload Size
            payload_size_mb = sum(arr.nbytes for arr in res_params) / (1024 * 1024) if res_params else 0.0
            metrics["payload_size_mb"] = payload_size_mb
            
            return res_params, num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"{self.log_prefix} Simulation client failed: {e}\n{traceback.format_exc()}", extra={"round": current_round})
            raise

    def evaluate(self, parameters: list, config: dict):
        if not hasattr(self, "trainer"):
            return 0.0, 0, {}  # A Fog bridge owns no edge inference model.
        return self.trainer.evaluate(parameters)


def client_fn(context: Context):
    from src.shared.utils.simulation_resources import simulation_config
    run_config = simulation_config(context.run_config)
    node_config = context.node_config
    device = get_device(tier="edge")
    
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

    # Point at the correct hardware emulator sidecar
    tcti_str = f"swtpm:host=tpm-{fog_num}-{edge_num},port=2321"
    os.environ["TPM2TOOLS_TCTI"] = tcti_str
    log_prefix = f"[EDGE {fog_num}_{edge_num}]"
    logger = setup_logger(log_prefix)

    try:
        from src.shared.security.threat_profiler import assign_edge_roles
        role = assign_edge_roles(run_config, total_edges, global_index, random_seed, logger)
        
        train_config = {
            "random_seed": random_seed,
            "client_index": global_index,
            "tpm_tcti": tcti_str,
            "role": role,
            "learning_rate": float(run_config["learning_rate"]),
            "quantization_bits": int(run_config["quantization_bits"]),
            "n_features": n_features,
            "num_classes": num_classes,
            "local_epochs": int(run_config["local_epochs"]),
            "clip_min": optional_float(run_config["clip_min"]),
            "clip_max": optional_float(run_config["clip_max"]),
            "clip_norm": float(run_config["clip_norm"]),
            "edge_inference_int8": bool(run_config["edge_inference_int8"]),
            "shap_aware_base_attack": str(run_config["shap_aware_base_attack"]),
            "robustness_eval_attack": str(run_config["robustness_eval_attack"]),
            "shap_explain_count": int(run_config["shap_explain_count"]),
            "shap_val_samples": int(run_config["shap_val_samples"])
        }
        
        # Shared adversarial-training settings for benign and label-flip paths.
        train_config["adv_ratio"] = float(run_config["benign_adv_ratio"])
        train_config["eps"] = float(run_config["benign_eps"])
        train_config["alpha"] = float(run_config["benign_alpha"])
        train_config["n_iter"] = int(run_config["benign_n_iter"])

        if role == "label_flip":
            train_config["p_flip"] = float(run_config["p_flip"])
        if role == "gradient_manip":
            train_config["gradient_alpha"] = float(run_config["gradient_alpha"])
        if role == "shap_aware":
            train_config.update(
                p_flip=float(run_config["p_flip"]), fog_num=fog_num,
                shap_aware_tau=optional_float(run_config.get("shap_aware_tau")),
                shap_aware_constrained=run_config.get("shap_aware_constrained", True),
                shap_aware_max_backtracks=int(run_config.get("shap_aware_max_backtracks", 10)),
            )
            
        apply_smote = run_config["apply_smote"]
        simulate_leakage = run_config["simulate_global_leakage"]
        
        # Fetch pre-partitioned edge artifact instantly from disk
        X_part, y_part, n_classes_eval = get_dataset(
            dataset_name, dataset_path, num_classes, random_seed, 
            simulate_global_leakage=simulate_leakage, 
            apply_smote=apply_smote, split=f"edge_{global_index}", 
            test_split=test_split, val_split=val_split, partition_config=run_config
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
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **get_dataloader_kwargs(device=device))
        
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

    seed_training(random_seed, global_index, 0)
    model = get_model(model_arch, n_features, num_classes)
    
    return Client(
        logger=logger, log_prefix=log_prefix, fog_num=fog_num, edge_num=edge_num,
        model=model, train_loader=train_loader, device=device, 
        dataset_metadata=dataset_metadata, train_config=train_config,
        eval_loader=evaluation_loader(run_config),
        shap_data=attack_validation_data(run_config, fog_num) if role == "shap_aware" else None,
    ).to_client()


app = ClientApp(client_fn=client_fn)
