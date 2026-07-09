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
from src.shared.utils.metrics import federated_averaging


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
        """Configures the next training round and logs the broadcast."""
        self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        """Aggregates trusted regional models from the Fog tier into a global model.

        Bypasses strict gatekeeper filtering under the assumption that Fog nodes 
        have already sanitized their respective regions. Applies standard aggregation, 
        caches the state, and quantizes the final global weights.
        """
        # Check if ANY results came from the fog nodes
        if not results:
            return None, {}

        round_display = server_round
        temp_gatekeeper = None
        
        # Cloud verification route bypasses gatekeeper filtering directly
        # Only mark as trusted_result if the model actually trained on any data
        trusted_results = []
        for client_proxy, fit_res in results:
            if fit_res.num_examples > 0:
                trusted_results.append((client_proxy, fit_res))

        if not trusted_results:
            return None, {}

        local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)

        try:
            # Call aggregation strategy
            aggregated_model, saboteurs, rewarded_nodes = self._apply_aggregation_strategy(local_models, sizes, trust_weights, tpm_ids, display_names)
            
            # Cache a deep copy of the current model
            self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())

            # The global model adopts the newly calculated blended weights
            self.global_model.load_state_dict(aggregated_model.state_dict())

            # Convert into NumPy arrays and compress
            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)

            # Convert into a seriliazed network format to send over the network
            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            aggregated_metrics = {}

        except Exception as e:
            self.logger.error(f"{self.log_prefix} ❌ Math failed: {e}\n{traceback.format_exc()}", extra={"round": round_display})
            return None, {} # Force the round to abort gracefully

        return aggregated_parameters, aggregated_metrics

    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float], List[str], List[str]]:
        """Unpacks and decompresses Fog-tier network payloads into native PyTorch models."""
        local_models, sizes, trust_weights, tpm_ids, display_names = [], [], [], [], []
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32))

        for client_proxy, fit_res in trusted_results:
            # Extract TPM ID and display identity
            tpm_id = fit_res.metrics.get("tpm_id", f"CID-{client_proxy.cid}")
            display_identity = fit_res.metrics.get("display_identity", f"{tpm_id} (Unknown)")
            
            # Extract the received data
            model = get_model(self.model_architecture, self.n_features, self.num_classes)
            raw_params = parameters_to_ndarrays(fit_res.parameters)
            decompressed_params = decompress_weights(raw_params, quantization_bits)

            # Convert to native PyTorch tensors for the engine
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), decompressed_params)})
            model.load_state_dict(state_dict, strict=True)

            # Store the extracted data
            local_models.append(model)
            sizes.append(fit_res.num_examples)
            trust_weights.append(float(fit_res.metrics.get("total_regional_trust", 1.0)))
            tpm_ids.append(tpm_id)
            display_names.append(display_identity)

        return local_models, sizes, trust_weights, tpm_ids, display_names

    def _apply_aggregation_strategy(self, local_models: List[torch.nn.Module], sizes: List[int], trust_weights: List[float], tpm_ids: List[str], display_names: List[str]) -> Tuple[torch.nn.Module, List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Applies a trust-weighted Federated Averaging strategy at the Cloud layer."""
        total_trust = sum(trust_weights)
        weights = [w / total_trust for w in trust_weights] if total_trust > 0 else None
        return federated_averaging(local_models, weights=weights), [], []

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        """Evaluates the global model, records metrics, and persists the state to disk.

        Serializes the training history into a JSON payload and saves the master 
        PyTorch weights (.pt) to the experiment directory.
        """
        # Call the built-in testing process
        eval_res = super().evaluate(server_round, parameters)
        
        # Exit early if it failed
        if eval_res is None:
            return None

        # Unbox the results it it was successful
        loss, metrics = eval_res
        round_data = {
            "round": server_round,
            "global_loss": loss,
            "global_accuracy": metrics.get("accuracy"),
            "global_macro_f1": metrics.get("macro_f1"),
            "timestamp": datetime.now().isoformat()
        }

        self.results_dict["performance"].append(round_data)

        # Path for saving 
        run_dir = f"results/{self.experiment_name}"
        os.makedirs(run_dir, exist_ok=True)
        filepath = f"{run_dir}/{self.experiment_name}.json"
        with open(filepath, "w") as f:
            json.dump(self.results_dict, f, indent=4)

        # Dump the model weights
        model_filepath = os.path.join(run_dir, "global_model.pt")
        torch.save(self.global_model.state_dict(), model_filepath)
        self.logger.info(f"{self.log_prefix} 💾 Saved metrics and model to {run_dir}/ with accuracy: {round_data['global_accuracy']}%", extra={"round": server_round})

        return loss, metrics