import os
from datetime import datetime

import torch
import numpy as np
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.shared.utils.logger_setup import setup_logger
from src.shared.data.data_loader import get_dataset, DATASET_METADATA
from src.tier_cloud.global_evaluator import GlobalEvaluator


def fit_config(server_round: int) -> dict:
    """Provides dynamic fit instructions on a per-round basis."""
    return {"server_round": server_round}


def _build_run_metadata(run_config: dict) -> dict:
    """Extracts and standardizes the entire run_config object into a unified dictionary."""
    experiment_name = str(run_config.get("run_name", "")).strip()
    if not experiment_name:
        experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "dataset": str(run_config.get("dataset", "edge_iiotset")),
        "dataset_fraction": float(run_config.get("dataset_fraction", 1.0)),
        "model_architecture": str(run_config.get("model_architecture", "cnnlstm")),
        "num_rounds": int(run_config.get("num_rounds", 3)),
        "min_clients": int(run_config.get("min-clients", 1)),
        "learning_rate": float(run_config.get("learning_rate", 0.001)),
        "batch_size": int(run_config.get("batch_size", 32)),
        "random_seed": int(run_config.get("random_seed", 42)),
        "n_classes_per": int(run_config.get("n_classes_per", 3)),
        "shap_threshold": float(run_config.get("shap_threshold", 0.5)),
        "shap_val_samples": int(run_config.get("shap_val_samples", 100)),
        "shap_explain_count": int(run_config.get("shap_explain_count", 10)),
        "backdoor_ratio": float(run_config.get("backdoor_ratio", 0.0)),
        "backdoor_poison_fraction": float(run_config.get("backdoor_poison_fraction", 0.5)),
        "backdoor_target_class": int(run_config.get("backdoor_target_class", 0)),
        "backdoor_trigger_features": str(run_config.get("backdoor_trigger_features", "[-3, -2, -1]")),
        "backdoor_trigger_value": float(run_config.get("backdoor_trigger_value", 1.5)),
        "benign_adv_ratio": float(run_config.get("benign_adv_ratio", 0.3)),
        "benign_eps": float(run_config.get("benign_eps", 0.05)),
        "benign_alpha": float(run_config.get("benign_alpha", 0.2)),
        "benign_n_iter": float(run_config.get("benign_n_iter", 3)),
        "rollback_threshold": float(run_config.get("rollback_threshold", 0.80)),
        "quantization_bits": int(run_config.get("quantization_bits", 32)),
        "robustness_eval_attack": str(run_config.get("robustness_eval_attack", "pgd")),
        "clip_min": float(run_config.get("clip_min", 0.0)),
        "clip_max": float(run_config.get("clip_max", 1.0)),
        "simulate_global_leakage": bool(run_config.get("simulate_global_leakage", False)),
        "tpm_freshness_window": int(run_config.get("tpm_freshness_window", 300)),
        "snapshot_rounds": list(run_config.get("snapshot_rounds", [])),
        "snapshot_interval": int(run_config.get("snapshot_interval", 9999))
    }


def server_fn(context: Context) -> ServerAppComponents:
    """Spins up the core server process parsing configurations."""
    from src.shared.utils.config_loader import get_merged_config
    run_config = get_merged_config(context.run_config)
    
    run_metadata = _build_run_metadata(run_config)

    tier = str(run_config.get("tier", "unknown"))
    raw_fog_id = str(run_config.get("fog_id", "cloud"))
    fog_num = int(''.join(filter(str.isdigit, raw_fog_id))) if any(c.isdigit() for c in raw_fog_id) else 0

    log_prefix = "[CLOUD SERVER]" if tier == "cloud" else f"[FOG {fog_num} SERVER]"
    logger = setup_logger(log_prefix)
    
    logger.info("Starting up... Expecting clients.")
        
    dataset_path = str(run_config.get("dataset_path", "data/edge_iiotset/raw/network_traffic_samples.csv"))
    broker_ip = str(run_config.get("broker_ip", "127.0.0.1"))
    fog_ipc_base = int(run_config.get("fog_ipc_base", 10000))
    socket_timeout = float(run_config.get("socket_timeout", 600.0))

    num_classes = DATASET_METADATA[run_metadata["dataset"]]["classes"]
    n_features = DATASET_METADATA[run_metadata["dataset"]]["features"]
    val_data = None

    random_seed = int(run_config.get("random_seed", 42))
    test_split = float(run_config.get("test_split", 0.30))
    val_split = float(run_config.get("val_split", 0.50))
    
    if tier == "fog":
        try:
            simulate_global_leakage = run_config.get("simulate_global_leakage", False)

            # Fetch shuffled and prepared data
            dataset_returns = get_dataset(run_metadata["dataset"], dataset_path, num_classes, random_seed, simulate_global_leakage, False, "val", test_split, val_split)
            
            X_full = dataset_returns[0]
            y_full = dataset_returns[1]

            # Keep a fraction of the dataset
            if run_metadata["dataset_fraction"] < 1.0:
                # Have a hard floor of shap_val_samples, in order to perform the shap calculations accurately
                subset_size = max(run_metadata["shap_val_samples"], int(len(X_full) * run_metadata["dataset_fraction"]))
                X_full = X_full[:subset_size]
                y_full = y_full[:subset_size]

            # Build the evaluation subset
            val_data = (X_full[:run_metadata["shap_val_samples"]], y_full[:run_metadata["shap_val_samples"]])
            
            # Extract the feature count
            n_features = X_full.shape[1]
        except FileNotFoundError:
            logger.warning(f"Dataset not found at {dataset_path}. SHAP checks will be bypassed.")
    
    
    evaluator = GlobalEvaluator(
        dataset=run_metadata["dataset"],
        dataset_path=dataset_path, 
        num_classes=num_classes, 
        n_features=n_features, 
        device="cpu", 
        random_seed=run_metadata["random_seed"], 
        run_metadata=run_metadata,
        tier=tier,
        logger=logger
    )
    evaluate_fn = evaluator.evaluate
    
    if tier == "cloud":
        from src.tier_cloud.cloud_aggregator import CloudAggregator
        strategy = CloudAggregator(
            n_features=n_features,
            num_classes=num_classes,
            broker_ip=broker_ip,
            fog_ipc_base=fog_ipc_base,
            socket_timeout=socket_timeout,
            shap_explain_count=run_metadata["shap_explain_count"],
            logger=logger,
            log_prefix=log_prefix,
            tier=tier,
            fog_num=fog_num,
            val_data=val_data,
            shap_threshold=run_metadata["shap_threshold"],
            min_available_clients=run_metadata["min_clients"],
            min_fit_clients=run_metadata["min_clients"],
            min_evaluate_clients=run_metadata["min_clients"],
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,
            run_metadata=run_metadata,
            model_architecture=run_metadata["model_architecture"]
        )
    else:
        from src.tier_fog.fog_aggregator import FogAggregator
        strategy = FogAggregator(
            n_features=n_features,
            num_classes=num_classes,
            broker_ip=broker_ip,
            fog_ipc_base=fog_ipc_base,
            socket_timeout=socket_timeout,
            shap_explain_count=run_metadata["shap_explain_count"],
            logger=logger,
            log_prefix=log_prefix,
            tier=tier,
            fog_num=fog_num,
            val_data=val_data,
            shap_threshold=run_metadata["shap_threshold"],
            min_available_clients=run_metadata["min_clients"],
            min_fit_clients=run_metadata["min_clients"],
            min_evaluate_clients=run_metadata["min_clients"],
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,
            run_metadata=run_metadata,
            model_architecture=run_metadata["model_architecture"]
        )

    config = ServerConfig(num_rounds=run_metadata["num_rounds"])
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)