import os
import json
import copy
import traceback
import logging
from datetime import datetime
from collections import OrderedDict
from typing import Tuple, List, Optional, Any, Dict

import torch
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from src.shared.models.factory import get_model
from src.shared.network.compression import compress_weights, decompress_weights
from src.shared.utils.metrics import federated_averaging, krum_select, trimmed_mean_aggregate, flame_aggregate, fltrust_aggregate


class CloudAggregator(FedAvg):
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
        shap_threshold: float = 0.5,
        run_metadata: Optional[dict] = None,
        *args,
        **kwargs
    ):
        self.broker_ip = kwargs.pop("broker_ip", "127.0.0.1")
        self.ipc_port = kwargs.pop("fog_ipc_base", 10000) + fog_num
        self.socket_timeout = kwargs.pop("socket_timeout", 600.0)
        self.shap_explain_count = kwargs.pop("shap_explain_count", 10)
        self.strategy = kwargs.pop("strategy", "ztafl")
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
        self.shap_threshold = shap_threshold

        self.global_model = get_model(self.model_architecture, self.n_features, self.num_classes)
        self.cached_global_state = None
        self.previous_val_acc = None

        self.run_metadata = run_metadata or {}
        self.experiment_name = self.run_metadata.get("experiment_name", "default_run")
        self.results_dict = {
            "metadata": self.run_metadata,
            "performance": []
        }
        self.active_nonces = {}

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        if not results:
            return None, {}

        round_display = server_round
        temp_gatekeeper = None
        
        # Cloud verification route bypasses gatekeeper filtering directly
        trusted_results = []
        for client_proxy, fit_res in results:
            if fit_res.num_examples > 0:
                trusted_results.append((client_proxy, fit_res))

        if not trusted_results:
            return None, {}

        self.logger.info(f"{self.log_prefix} 🧮 Executing {self.strategy.upper()} PyTorch aggregation...", extra={"round": round_display})

        local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)

        try:
            aggregated_model, saboteurs, rewarded_nodes = self._apply_aggregation_strategy(local_models, sizes, trust_weights, tpm_ids, display_names)
            
            self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())

            self.global_model.load_state_dict(aggregated_model.state_dict())
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]

            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)

            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            aggregated_metrics = {}

        except Exception as e:
            self.logger.error(f"{self.log_prefix} ❌ Math failed for {self.strategy}: {e}\n{traceback.format_exc()}", extra={"round": round_display})
            return None, {} # Force the round to abort gracefully

        return aggregated_parameters, aggregated_metrics

    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float], List[str], List[str]]:
        local_models, sizes, trust_weights, tpm_ids, display_names = [], [], [], [], []
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32))

        for client_proxy, fit_res in trusted_results:
            tpm_id = fit_res.metrics.get("tpm_id", f"CID-{client_proxy.cid}")
            display_identity = fit_res.metrics.get("display_identity", f"{tpm_id} (Unknown)")
            
            model = get_model(self.model_architecture, self.n_features, self.num_classes)
            raw_params = parameters_to_ndarrays(fit_res.parameters)
            decompressed_params = decompress_weights(raw_params, quantization_bits)

            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), decompressed_params)})
            model.load_state_dict(state_dict, strict=True)

            local_models.append(model)
            sizes.append(fit_res.num_examples)
            trust_weights.append(float(fit_res.metrics.get("total_regional_trust", 1.0)))
            tpm_ids.append(tpm_id)
            display_names.append(display_identity)

        return local_models, sizes, trust_weights, tpm_ids, display_names

    def _apply_aggregation_strategy(self, local_models: List[torch.nn.Module], sizes: List[int], trust_weights: List[float], tpm_ids: List[str], display_names: List[str]) -> Tuple[torch.nn.Module, List[Tuple[str, str]], List[Tuple[str, str]]]:
        if self.strategy in ["zta", "ztafl"]:
            total_trust = sum(trust_weights)
            weights = [w / total_trust for w in trust_weights] if total_trust > 0 else None
            return federated_averaging(local_models, weights=weights), [], []
        elif self.strategy in ["fedavg", "fedprox"]:
            total_samples = sum(sizes)
            weights = [s / total_samples for s in sizes] if total_samples > 0 else None
            return federated_averaging(local_models, weights=weights), [], []
        elif self.strategy == "krum":
            default_f = max(1, int(len(local_models) * 0.3))
            krum_f = int(self.run_metadata.get("krum_f", default_f))
            return krum_select(local_models, f=krum_f), [], []
        elif self.strategy == "trimmed_mean":
            trimmed_mean_beta = float(self.run_metadata.get("trimmed_mean_beta", 0.1))
            return trimmed_mean_aggregate(local_models, beta=trimmed_mean_beta), [], []
        elif self.strategy == "flame":
            flame_target_frac = float(self.run_metadata.get("flame_target_frac", 0.5))
            return flame_aggregate(local_models, self.global_model, target_frac=flame_target_frac), [], []
        elif self.strategy == "fltrust":
            server_model = get_model(self.model_architecture, self.n_features, self.num_classes)
            server_model.load_state_dict(self.global_model.state_dict())
            return fltrust_aggregate(local_models, server_model, self.global_model), [], []
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is None:
            return None

        loss, metrics = eval_res
        round_data = {
            "round": server_round,
            "global_loss": loss,
            "global_accuracy": metrics.get("accuracy"),
            "global_macro_f1": metrics.get("macro_f1"),
            "global_asr": metrics.get("asr", 0.0),
            "global_robustness_pgd": metrics.get("robustness_pgd", {}),
            "global_robustness_fgsm": metrics.get("robustness_fgsm", {}),
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
        self.logger.info(f"{self.log_prefix} 💾 Saved metrics and model to {run_dir}/ with accuracy: {round_data["global_accuracy"]}%", extra={"round": server_round})

        return loss, metrics