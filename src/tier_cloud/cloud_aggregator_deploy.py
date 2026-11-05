import os
import json
import time
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict

import torch
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

from src.tier_cloud.model_state import GlobalModelState
from src.shared.network.model_payload import decode_model, positive_trust, validate_region_result


class CloudAggregator(GlobalModelState, FedAvg):
    """The central orchestration engine routing the federation at Cloud Layer.
    Acts as the network dispatcher bridging cross-tier TCP flows and managing the global weights array.
    """

    def __init__(
        self,
        logger: logging.Logger,
        log_prefix: str,
        tier: str,
        fog_num: int,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_features: int = 40,
        num_classes: int = 15,
        run_metadata: Optional[dict] = None,
        *args,
        **kwargs
    ):
        self.broker_ip = kwargs.pop("broker_ip", "127.0.0.1")
        self.ipc_port = kwargs.pop("fog_ipc_base", 10000) + fog_num
        self.socket_timeout = kwargs.pop("socket_timeout", 600.0)
        self.shap_explain_count = kwargs.pop("shap_explain_count", 10)
        self.model_architecture = kwargs.pop("model_architecture", "cnnlstm")
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.log_prefix = log_prefix
        self.tier = tier
        self.fog_num = fog_num
        self.current_bridged_round = 0
        self.val_data = val_data
        self.n_features = n_features
        self.num_classes = num_classes
        self.run_metadata = run_metadata or {}
        self.initialize_global_state()
        self.experiment_name = self.run_metadata.get("experiment_name", "default_run")
        self.snapshot_rounds = self.run_metadata["snapshot_rounds"]
        self.snapshot_interval = self.run_metadata["snapshot_interval"]
        self.results_dict = {
            "metadata": self.run_metadata,
            "performance": []
        }
        self.active_nonces = {}

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        """Configures the next training round and logs the broadcast."""
        self.sync_global_state(parameters)
        self.active_nonces = {}
        self.round_start_time = time.time()
        self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []  # Centralized evaluation supplies the authoritative metrics.

    def aggregate_fit(self, server_round, results, failures):
        models, weights, rejected = [], [], []
        empty_regions, failed_regions = 0, len(failures)
        for proxy, result in results:
            try:
                if result.metrics.get("status") == "crashed" and not result.parameters.tensors and result.num_examples == 0:
                    failed_regions += 1
                    continue
                status = validate_region_result(result.num_examples, bool(result.parameters.tensors), result.metrics)
                if status == "error":
                    failed_regions += 1
                    continue
                if status != "aggregated":
                    empty_regions += 1
                    continue
                trust = positive_trust(result.metrics.get("total_regional_trust"))
                model = decode_model(self.global_model, result.parameters, self.quantization_bits)
                if trust > 0:
                    models.append(model)
                    weights.append(trust)
            except (ValueError, TypeError, OverflowError, IndexError, KeyError, EOFError) as exc:
                rejected.append(proxy.cid)
                self.logger.warning(f"Rejected Fog {proxy.cid}: {exc}", extra={"round": server_round})
        parameters, metrics = self.aggregate_regions(models, weights, server_round)
        metrics["rejected_regions"] = len(rejected)
        metrics["empty_regions"] = empty_regions
        metrics["failed_regions"] = failed_regions
        return parameters, metrics

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        """Evaluates the global model, records metrics, and persists the state to disk."""
        self.logger.info(f"{self.log_prefix}: Calling Cloud Evaluate! Saving model weights every {self.snapshot_interval} rounds and specifically at rounds: {self.snapshot_rounds}", extra={"round": server_round})
        self.sync_global_state(parameters)
        eval_res = super().evaluate(server_round, parameters)

        if eval_res is None:
            return None

        round_latency_total_sec = time.time() - getattr(self, 'round_start_time', time.time())
        payload_size_mb = 0.0
        if parameters:
            ndarrays = parameters_to_ndarrays(parameters)
            payload_size_mb = sum(arr.nbytes for arr in ndarrays) / (1024 * 1024)

        loss, metrics = eval_res
        round_data = {
            "round": server_round,
            "global_loss": loss,
            "global_accuracy": metrics.get("accuracy"),
            "global_macro_f1": metrics.get("macro_f1"),
            "global_auc": metrics.get("auc"),
            "round_latency_total_sec": round_latency_total_sec,
            "payload_size_mb": payload_size_mb,
            "timestamp": datetime.now().isoformat()
        }
        self.results_dict["performance"].append(round_data)

        run_dir = f"results/{self.experiment_name}"
        os.makedirs(run_dir, exist_ok=True)
        filepath = f"{run_dir}/{self.experiment_name}.json"
        with open(filepath, "w") as f:
            json.dump(self.results_dict, f, indent=4)

        model_filepath = os.path.join(run_dir, "global_model.pt")
        torch.save(self.global_model.state_dict(), model_filepath)

        if server_round in self.snapshot_rounds or (self.snapshot_interval > 0 and server_round % self.snapshot_interval == 0 and server_round != 0):
            model_filepath = os.path.join(run_dir, f"global_model_round_{server_round}.pt")
            torch.save(self.global_model.state_dict(), model_filepath)

        self.logger.info(f"{self.log_prefix} Saved metrics and model to {run_dir}/ with accuracy: {round_data['global_accuracy']}%", extra={"round": server_round})
        return loss, metrics
