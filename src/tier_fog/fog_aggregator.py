import copy
import logging
import traceback
from collections import OrderedDict
from typing import Tuple, List, Optional, Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns

from src.shared.models.factory import get_model
from src.shared.network.compression import compress_weights, decompress_weights
from src.shared.utils.metrics import federated_averaging, krum_select, trimmed_mean_aggregate, flame_aggregate, fltrust_aggregate
from src.tier_fog.shap_verifier import shap_weighted_aggregate
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.tier_fog.fog_bridge_server import FogBridgeServer


class FogAggregator(FedAvg):
    """Orchestration aggregation node managing regional verification flows at the Fog tier."""
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
        self.active_nonces = {}

        self.gatekeeper = ZeroTrustGatekeeper(logger=self.logger, log_prefix=self.log_prefix, run_metadata=self.run_metadata)
        from src.shared.utils.admin_console import AdminConsole
        self.admin_console = AdminConsole(self.gatekeeper, self.logger)
        self.fog_bridge = FogBridgeServer(logger=self.logger, log_prefix=self.log_prefix, ipc_port=self.ipc_port, socket_timeout=self.socket_timeout)

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        import secrets
        bridged_round = self.fog_bridge.wait_for_start()
        if bridged_round > 0:
            self.current_bridged_round = bridged_round
        else:
            self.logger.error(f"{self.log_prefix} [IPC SERVER] Failed to establish valid round sync.", extra={"round": server_round})
            raise ConnectionError("Fog Bridge failed to synchronize.")

        round_display = self.current_bridged_round
        self.admin_console.execute_scheduled_updates(round_display, self.fog_num)

        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_client_instructions = []

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

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        if not results:
            self._relay_ipc_fog_bridge(None)
            return None, {}

        round_display = self.current_bridged_round
        trusted_results = self.gatekeeper.filter_node_updates(self.tier, self.strategy, round_display, results, self.active_nonces)

        if not trusted_results:
            self.logger.warning(f"{self.log_prefix} ⚠️ No trusted results to aggregate! Releasing IPC bridge with empty payload.")
            self._relay_ipc_fog_bridge(None)
            return None, {}

        self.logger.info(f"{self.log_prefix} 🧮 Executing {self.strategy.upper()} PyTorch aggregation...", extra={"round": round_display})

        local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)

        try:
            aggregated_model, saboteurs, rewarded_nodes = self._apply_aggregation_strategy(local_models, sizes, trust_weights, tpm_ids, display_names)
            
            if self.gatekeeper.trust_db:
                for tpm_id, display_identity in saboteurs:
                    self.logger.error(f"{self.log_prefix} ☠️ SHAP SABOTAGE DETECTED: {display_identity} submitted statistically toxic weights. Retroactively slashing TrustDB score!", extra={"round": round_display})
                    self.gatekeeper.trust_db.process_attestation(node_id=tpm_id, display_name=display_identity, is_valid=False, round_num=round_display)
                    
                for tpm_id, display_identity in rewarded_nodes:
                    self.logger.info(f"{self.log_prefix} 🌟 SHAP EXCELLENCE: {display_identity} strictly exceeded median stability. Granting behavioral trust reward!", extra={"round": round_display})
                    self.gatekeeper.trust_db.apply_behavioral_reward(node_id=tpm_id, display_name=display_identity, round_num=round_display)

            self._evaluate_rollback_sanity_check(aggregated_model, round_display)

            self.global_model.load_state_dict(aggregated_model.state_dict())
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]

            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)

            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            aggregated_metrics = {}

            if hasattr(self, 'current_regional_trust'):
                aggregated_metrics["total_regional_trust"] = self.current_regional_trust

        except Exception as e:
            self.logger.error(f"{self.log_prefix} ❌ Math failed for {self.strategy}: {e}\n{traceback.format_exc()}", extra={"round": round_display})
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, trusted_results, failures)

        self._relay_ipc_fog_bridge(aggregated_parameters)
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
            if self.val_data is None:
                raise ValueError("ZTA/ZTAFL strategy requires val_data for SHAP verification, but it was None.")
            X_val, y_val = self.val_data
            agg_model, regional_trust, passed_flags, reward_flags = shap_weighted_aggregate(
                local_models=local_models, ref_model=self.global_model,
                X_val=X_val, y_val=y_val, sizes=sizes, n_classes=self.num_classes, n_explain=self.shap_explain_count
            )
            self.current_regional_trust = regional_trust
            saboteurs = [(tpm_ids[i], display_names[i]) for i, passed in enumerate(passed_flags) if not passed]
            rewarded_nodes = [(tpm_ids[i], display_names[i]) for i, rewarded in enumerate(reward_flags) if rewarded]
            return agg_model, saboteurs, rewarded_nodes
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

            return fltrust_aggregate(local_models, server_model, self.global_model), [], []
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _evaluate_rollback_sanity_check(self, aggregated_model: torch.nn.Module, round_display: int) -> None:
        if self.val_data is not None:
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

    def _relay_ipc_fog_bridge(self, aggregated_parameters: Optional[list]) -> None:
        ndarrays_to_send = parameters_to_ndarrays(aggregated_parameters) if aggregated_parameters is not None else []
        self.fog_bridge.relay_weights(ndarrays_to_send)

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        self.logger.info(f"{self.log_prefix} Bypassing global evaluation on Fog tier to save resources.", extra={"round": server_round})
        return None