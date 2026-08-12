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
    import os
    from datetime import datetime
    
    experiment_name = str(run_config["run_name"]).strip()
    
    # If the YAML config was left blank, fetch the frozen timestamp from the host-mounted file
    if not experiment_name or experiment_name.lower() in ["none", "null"]:
        try:
            # os.path.getmtime guarantees that a node booting 5 minutes late reads the exact 
            # same millisecond timestamp as the node that booted first.
            mtime = os.path.getmtime("config/training.yaml")
            experiment_name = datetime.fromtimestamp(mtime).strftime("run_%Y%m%d_%H%M%S")
        except Exception:
            # Absolute fallback if the file is somehow inaccessible
            experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "dataset": str(run_config["dataset"]),
        "dataset_fraction": float(run_config["dataset_fraction"]),
        "model_architecture": str(run_config["model_architecture"]),
        "num_rounds": int(run_config["num_rounds"]),
        "min_clients": int(run_config.get("min-clients", 1)),
        "learning_rate": float(run_config["learning_rate"]),
        "batch_size": int(run_config["batch_size"]),
        "random_seed": int(run_config["random_seed"]),
        "n_classes_per": int(run_config["n_classes_per"]),
        "shap_threshold": float(run_config["shap_threshold"]),
        "shap_val_samples": int(run_config["shap_val_samples"]),
        "shap_explain_count": int(run_config["shap_explain_count"]),
        "backdoor_ratio": float(run_config["backdoor_ratio"]),
        "backdoor_poison_fraction": float(run_config["backdoor_poison_fraction"]),
        "backdoor_target_class": int(run_config["backdoor_target_class"]),
        "backdoor_trigger_features": str(run_config["backdoor_trigger_features"]),
        "backdoor_trigger_value": float(run_config["backdoor_trigger_value"]),
        "benign_adv_ratio": float(run_config["benign_adv_ratio"]),
        "benign_eps": float(run_config["benign_eps"]),
        "benign_alpha": float(run_config["benign_alpha"]),
        "benign_n_iter": float(run_config["benign_n_iter"]),
        "rollback_threshold": float(run_config["rollback_threshold"]),
        "quantization_bits": int(run_config["quantization_bits"]),
        "robustness_eval_attack": str(run_config["robustness_eval_attack"]),
        "clip_min": float(run_config["clip_min"]),
        "clip_max": float(run_config["clip_max"]),
        "simulate_global_leakage": bool(run_config["simulate_global_leakage"]),
        "tpm_freshness_window": int(run_config["tpm_freshness_window"]),
        "snapshot_rounds": list(run_config["snapshot_rounds"]),
        "snapshot_interval": int(run_config["snapshot_interval"])
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
        
    dataset_path = str(run_config["dataset_path"])
    broker_ip = str(run_config["broker_ip"])
    fog_ipc_base = int(run_config["fog_ipc_base"])
    socket_timeout = float(run_config["socket_timeout"])

    num_classes = DATASET_METADATA[run_metadata["dataset"]]["classes"]
    n_features = DATASET_METADATA[run_metadata["dataset"]]["features"]
    val_data = None

    random_seed = int(run_config["random_seed"])
    test_split = float(run_config["test_split"])
    val_split = float(run_config["val_split"])
    
    if tier == "fog":
        try:
            simulate_global_leakage = run_config["simulate_global_leakage"]

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