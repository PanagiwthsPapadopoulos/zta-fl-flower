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
from torch.utils.data import DataLoader, TensorDataset

from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from src.models.factory import get_model
from src.network.compression import compress_weights, decompress_weights
from src.federation.strategies.aggregation import (
    federated_averaging,
    shap_weighted_aggregate,
    krum_select,
    trimmed_mean_aggregate,
    flame_aggregate,
    fltrust_aggregate
)

class ZTAStrategy(FedAvg):
    """
    The central orchestration engine routing the federation.
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
        if self.tier == "fog":
            from src.security.policy.gatekeeper import ZeroTrustGatekeeper
            self.gatekeeper = ZeroTrustGatekeeper(logger=self.logger, log_prefix=self.log_prefix, run_metadata=self.run_metadata)

            from src.network.fog_bridge import FogBridgeServer
            self.fog_bridge = FogBridgeServer(
                logger=self.logger,
                log_prefix=self.log_prefix,
                ipc_port=self.ipc_port,
                socket_timeout=self.socket_timeout
            )

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        from flwr.common import FitIns
        import secrets

        if self.tier == "cloud":
            self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        elif self.tier == "fog":
            bridged_round = self.fog_bridge.wait_for_start()
            if bridged_round > 0:
                self.current_bridged_round = bridged_round
            else:
                self.logger.error(f"{self.log_prefix} [IPC SERVER] Failed to establish valid round sync.", extra={"round": server_round})
                raise ConnectionError("Fog Bridge failed to synchronize.")

        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_client_instructions = []

        if self.tier == "fog":
            for client_proxy, shared_fit_ins in shared_instructions:
                client_config = shared_fit_ins.config.copy()
                client_config["server_round"] = self.current_bridged_round
                client_config["strategy"] = self.strategy

                if self.strategy in ["zta", "ztafl"]:
                    nonce = secrets.token_hex(16)
                    client_config["nonce"] = nonce
                    self.active_nonces[client_proxy.cid] = nonce
                
                isolated_fit_ins = FitIns(parameters=shared_fit_ins.parameters, config=client_config)
                new_client_instructions.append((client_proxy, isolated_fit_ins))
                
            return new_client_instructions
            
        return shared_instructions

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        if not results:
            self._relay_ipc_fog_bridge(None)
            return None, {}

        trusted_results = self._filter_results(server_round, results)

        if not trusted_results:
            self.logger.warning(f"{self.log_prefix} ⚠️ No trusted results to aggregate! Releasing IPC bridge with empty payload.")
            self._relay_ipc_fog_bridge(None)
            return None, {}

        round_display = self.current_bridged_round if self.tier == 'fog' else server_round
        self.logger.info(f"{self.log_prefix} 🧮 Executing {self.strategy.upper()} PyTorch aggregation...", extra={"round": round_display})

        # 🚨 FIX: Properly extracting the hardware IDs alongside names
        local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)

        try:
            aggregated_model, saboteurs = self._apply_aggregation_strategy(local_models, sizes, trust_weights, tpm_ids, display_names)
            
            if saboteurs and self.tier == "fog" and hasattr(self, 'gatekeeper') and self.gatekeeper.trust_db:
                # 🚨 FIX: Correctly unpack the tuple and supply the round_num to prevent TypeError!
                for tpm_id, display_identity in saboteurs:
                    self.logger.error(f"{self.log_prefix} ☠️ SHAP SABOTAGE DETECTED: {display_identity} submitted statistically toxic weights. Retroactively slashing TrustDB score!", extra={"round": round_display})
                    self.gatekeeper.trust_db.process_attestation(node_id=tpm_id, display_name=display_identity, is_valid=False, round_num=round_display)

            self._evaluate_rollback_sanity_check(aggregated_model, round_display)

            self.global_model.load_state_dict(aggregated_model.state_dict())
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]

            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            self.logger.debug(f"[CONFIG USAGE] compress_weights | quantization_bits: {quantization_bits}", extra={"round": round_display})
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)

            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            aggregated_metrics = {}

            if hasattr(self, 'current_regional_trust'):
                aggregated_metrics["total_regional_trust"] = self.current_regional_trust

        except Exception as e:
            self.logger.error(f"{self.log_prefix} ❌ Math failed for {self.strategy}: {e}\n{traceback.format_exc()}", extra={"round": round_display})
            self.logger.warning(f"{self.log_prefix} ⚠️ Falling back to Flower's default C++ FedAvg engine.", extra={"round": round_display})
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, trusted_results, failures)

        self._relay_ipc_fog_bridge(aggregated_parameters)
        self.logger.info(f"{self.log_prefix} ✅ Aggregation successfully finished! Handing off to Evaluator.", extra={"round": round_display})

        return aggregated_parameters, aggregated_metrics

    def _filter_results(self, server_round: int, results: list) -> list:
        round_display = self.current_bridged_round if self.tier == "fog" else server_round
        if self.tier == "fog":
            return self.gatekeeper.filter_node_updates(self.tier, self.strategy, round_display, results, self.active_nonces)
        else:
            from src.security.policy.gatekeeper import ZeroTrustGatekeeper
            temp_gatekeeper = ZeroTrustGatekeeper(logger=self.logger, log_prefix=self.log_prefix, run_metadata=self.run_metadata)
            return temp_gatekeeper.filter_node_updates(self.tier, self.strategy, round_display, results, self.active_nonces)

    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float], List[str], List[str]]:
        local_models, sizes, trust_weights, tpm_ids, display_names = [], [], [], [], []
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32))

        for client_proxy, fit_res in trusted_results:
            # 🚨 FIX: Extract hardware ID specifically for the saboteur list
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

    def _apply_aggregation_strategy(self, local_models: List[torch.nn.Module], sizes: List[int], trust_weights: List[float], tpm_ids: List[str], display_names: List[str]) -> Tuple[torch.nn.Module, List[Tuple[str, str]]]:
        saboteurs = []
        if self.strategy in ["zta", "ztafl"] and self.tier == "fog":
            X_val, y_val = self.val_data
            agg_model, regional_trust, passed_flags = shap_weighted_aggregate(
                local_models=local_models, ref_model=self.global_model,
                X_val=X_val, y_val=y_val, sizes=sizes, n_classes=self.num_classes, n_explain=self.shap_explain_count
            )
            self.current_regional_trust = regional_trust
            saboteurs = [(tpm_ids[i], display_names[i]) for i, passed in enumerate(passed_flags) if not passed]
            return agg_model, saboteurs
            
        elif self.strategy in ["zta", "ztafl"] and self.tier == "cloud":
            total_trust = sum(trust_weights)
            weights = [w / total_trust for w in trust_weights] if total_trust > 0 else None
            return federated_averaging(local_models, weights=weights), []
        elif self.strategy in ["fedavg", "fedprox"]:
            total_samples = sum(sizes)
            weights = [s / total_samples for s in sizes] if total_samples > 0 else None
            return federated_averaging(local_models, weights=weights), []
        elif self.strategy == "krum":
            default_f = max(1, int(len(local_models) * 0.3))
            krum_f = int(self.run_metadata.get("krum_f", default_f))
            return krum_select(local_models, f=krum_f), []
        elif self.strategy == "trimmed_mean":
            trimmed_mean_beta = float(self.run_metadata.get("trimmed_mean_beta", 0.1))
            return trimmed_mean_aggregate(local_models, beta=trimmed_mean_beta), []
        elif self.strategy == "flame":
            flame_target_frac = float(self.run_metadata.get("flame_target_frac", 0.5))
            return flame_aggregate(local_models, self.global_model, target_frac=flame_target_frac), []
        elif self.strategy == "fltrust":
            server_model = get_model(self.model_architecture, self.n_features, self.num_classes)
            server_model.load_state_dict(self.global_model.state_dict())
            if self.val_data is not None:
                X_val, y_val = self.val_data
                learning_rate = float(self.run_metadata.get("learning_rate", 0.001))
                batch_size = int(self.run_metadata.get("batch_size", 32))

                optimizer = torch.optim.Adam(server_model.parameters(), lr=learning_rate)
                criterion = torch.nn.CrossEntropyLoss()
                server_model.train()

                dataset = TensorDataset(X_val, y_val)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for X_batch, y_batch in loader:
                    if X_batch.size(0) < 2:
                        continue
                    optimizer.zero_grad()
                    loss = criterion(server_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

            return fltrust_aggregate(local_models, server_model, self.global_model), []
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _evaluate_rollback_sanity_check(self, aggregated_model: torch.nn.Module, round_display: int) -> None:
        if self.tier == "fog" and self.val_data is not None:
            aggregated_model.eval()
            with torch.no_grad():
                X_val, y_val = self.val_data
                preds = aggregated_model(X_val).argmax(dim=-1)
                val_acc = (preds == y_val).float().mean().item()

            self.logger.info(f"{self.log_prefix} 🔍 Post-Aggregation Validation Accuracy: {val_acc:.4f}", extra={"round": round_display})
            
            if self.previous_val_acc is None:
                self.logger.info(f"{self.log_prefix} ✅ Initializing baseline state.")
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                self.previous_val_acc = val_acc
                return
            
            rollback_fraction = float(self.run_metadata.get("rollback_threshold", 0.80))
            dynamic_threshold = self.previous_val_acc * rollback_fraction

            if val_acc < dynamic_threshold and self.cached_global_state is not None:
                self.logger.warning(f"{self.log_prefix} ⚠️ CRITICAL: Accuracy ({val_acc:.4f}) dropped below dynamic threshold ({dynamic_threshold:.4f}). Rolling back to previous round weights!", extra={"round": round_display})
                aggregated_model.load_state_dict(self.cached_global_state)
            else:
                self.logger.info(f"{self.log_prefix} ✅ Aggregation passed sanity check. Caching state.", extra={"round": round_display})
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                self.previous_val_acc = val_acc
        elif self.tier == "cloud":
            self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())

    def _relay_ipc_fog_bridge(self, aggregated_parameters: Optional[list]) -> None:
        if self.tier == "fog":
            ndarrays_to_send = parameters_to_ndarrays(aggregated_parameters) if aggregated_parameters is not None else []
            self.fog_bridge.relay_weights(ndarrays_to_send)

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        if self.tier == "fog":
            self.logger.info(f"{self.log_prefix} Bypassing global evaluation on Fog tier to save resources.", extra={"round": server_round})
            return None

        eval_res = super().evaluate(server_round, parameters)

        if eval_res is None:
            self.logger.warning(f"{self.log_prefix} Evaluation failed or did not run!", extra={"round": server_round})
            return None

        loss, metrics = eval_res

        if self.tier == "cloud":
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

            self.logger.info(f"{self.log_prefix} 💾 Saved metrics and model to {run_dir}/", extra={"round": server_round})

        return loss, metrics