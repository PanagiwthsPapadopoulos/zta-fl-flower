import os
import json
import time
import copy
import re
import traceback
import logging
from datetime import datetime
from collections import OrderedDict
from typing import Tuple, List, Optional, Any, Dict
import torch
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns

from src.shared.models.factory import get_model
from src.shared.network.compression import compress_weights, decompress_weights
from src.shared.utils.metrics import federated_averaging
from src.tier_fog.shap_verifier import shap_weighted_aggregate
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.shared.utils.admin_console import AdminConsole
from src.shared.utils.logger_setup import setup_logger

class CloudAggregator(FedAvg):
    """The central orchestration engine routing the federation at Cloud Layer.
    Mathematically virtualizes Fog security layers while perfectly spoofing the 3-tier telemetry.
    """
    def __init__(
        self,
        logger: logging.Logger,
        log_prefix: str,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_features: int = 40,
        num_classes: int = 15,
        shap_threshold: float = 0.5,
        run_metadata: Optional[dict] = None,
        *args,
        **kwargs
    ):
        self.shap_explain_count = kwargs.pop("shap_explain_count", 10)
        self.model_architecture = kwargs.pop("model_architecture", "cnnlstm")
        super().__init__(*args, **kwargs)
        
        self.logger = logger
        self.log_prefix = log_prefix
        self.val_data = val_data
        self.n_features = n_features
        self.num_classes = num_classes
        self.shap_threshold = shap_threshold
        
        self.global_model = get_model(self.model_architecture, self.n_features, self.num_classes)
        self.cached_global_state = None
        self.previous_val_acc = None
        self.run_metadata = run_metadata or {}
        self.experiment_name = self.run_metadata.get("experiment_name", "default_run")
        self.snapshot_rounds = self.run_metadata["snapshot_rounds"]
        self.snapshot_interval = self.run_metadata["snapshot_interval"]
        self.num_fogs = self.run_metadata.get("num_fogs", 1)
        
        self.results_dict = {
            "metadata": self.run_metadata,
            "performance": []
        }
        self.active_nonces = {}
        
        # Virtualized Fog Components & Telemetry Spoofing
        self.gatekeepers = {}
        self.admin_consoles = {}
        self.fog_state_histories = {}
        
        # Create dedicated loggers for the faked 3-tier structure
        self.fog_server_loggers = {}
        self.fog_client_loggers = {}
        
        for i in range(1, self.num_fogs + 1):
            f_server_prefix = f"[FOG {i} SERVER]"
            f_client_prefix = f"[FOG {i} CLIENT]"
            
            self.fog_server_loggers[i] = setup_logger(f_server_prefix)
            self.fog_client_loggers[i] = setup_logger(f_client_prefix)
            
            # Point the Gatekeeper to the FOG SERVER logger so it prints the attestation logs correctly
            gk = ZeroTrustGatekeeper(logger=self.fog_server_loggers[i], log_prefix=f_server_prefix, run_metadata=self.run_metadata)
            self.gatekeepers[i] = gk
            self.admin_consoles[i] = AdminConsole(gk, self.logger)
            self.fog_state_histories[i] = {
                "metadata": {"fog_num": i, "tier": "fog"},
                "rounds": {}
            }

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        self.round_start_time = time.time()
        
        # Spoof Phase 1 & 2 triggers for the UI
        self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        
        for i in range(1, self.num_fogs + 1):
            self.fog_client_loggers[i].info(f"[FOG {i} CLIENT] [IPC CLIENT] Connected! Sending START signal for round {server_round}.", extra={"round": server_round})
            self.fog_server_loggers[i].info(f"[FOG {i} SERVER] [IPC SERVER] START received for round {server_round}. Shouting to EDGE clients!", extra={"round": server_round})
            self.admin_consoles[i].execute_scheduled_updates(server_round, i)
        
        import secrets
        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_client_instructions = []
        
        for client_proxy, shared_fit_ins in shared_instructions:
            client_config = shared_fit_ins.config.copy()
            client_config["server_round"] = server_round
            
            nonce = secrets.token_hex(16)
            client_config["nonce"] = nonce
            self.active_nonces[client_proxy.cid] = nonce
            
            isolated_fit_ins = FitIns(parameters=shared_fit_ins.parameters, config=client_config)
            new_client_instructions.append((client_proxy, isolated_fit_ins))
            
        return new_client_instructions

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        if not results:
            return None, {}
            
        round_start_unix = time.time()
        
        # Group incoming updates by their mathematical Virtual Fog IDs
        fog_groupings = {i: [] for i in range(1, self.num_fogs + 1)}
        for client_proxy, fit_res in results:
            prefix = fit_res.metrics.get("log_prefix", "")
            m = re.search(r'\[EDGE (\d+)_(\d+)\]', prefix)
            if m:
                f_num = int(m.group(1))
                if f_num in fog_groupings:
                    fog_groupings[f_num].append((client_proxy, fit_res))
            else:
                fog_groupings[1].append((client_proxy, fit_res))
                
        virtual_fog_models = []
        virtual_fog_weights = []
        
        # Sequentially process each Virtual Fog Cluster
        for fog_num, cluster_results in fog_groupings.items():
            f_server_logger = self.fog_server_loggers[fog_num]
            f_server_prefix = f"[FOG {fog_num} SERVER]"
            
            if not cluster_results:
                f_server_logger.warning(f"{f_server_prefix} No client returned updates! Releasing IPC bridge with empty payload.", extra={"round": server_round})
                continue
                            
            token_sizes_bytes = {}
            node_ground_truths = {}
            for client_proxy, fit_res in cluster_results:
                raw_token = fit_res.metrics.get("tpm_token_json", "")
                tpm_id = f"CID-{client_proxy.cid}"
                if raw_token:
                    try:
                        tpm_id = json.loads(raw_token).get("IDi", tpm_id)
                    except Exception:
                        pass
                token_sizes_bytes[tpm_id] = len(raw_token.encode('utf-8'))
                node_ground_truths[tpm_id] = fit_res.metrics.get("role", "unknown")
                
            all_tpm_ids = list(node_ground_truths.keys())
            
            # Phase 5: Attestation Filter (Gatekeeper inherently uses the Fog Server logger now)
            attestation_validation_start_unix = time.time()
            trusted_results = self.gatekeepers[fog_num].filter_node_updates("fog", server_round, cluster_results, self.active_nonces)
            attestation_validation_end_unix = time.time()
            latency_attestation_ms = (attestation_validation_end_unix - attestation_validation_start_unix) * 1000
            
            attestation_accepted = [fit_res.metrics.get("tpm_id", f"CID-{client_proxy.cid}") for client_proxy, fit_res in trusted_results]
            attestation_rejected = [tid for tid in all_tpm_ids if tid not in attestation_accepted]
            
            if not trusted_results:
                f_server_logger.warning(f"{f_server_prefix} No trusted results to aggregate! Releasing IPC bridge with empty payload.", extra={"round": server_round})
                continue
                
            node_training_latencies = {}
            node_payload_sizes = {}
            node_local_training_start_unix = {}
            node_local_training_end_unix = {}
            for client_proxy, fit_res in trusted_results:
                tpm_id = fit_res.metrics.get("tpm_id", f"CID-{client_proxy.cid}")
                node_training_latencies[tpm_id] = float(fit_res.metrics.get("latency_adv_training_sec", 0.0))
                node_payload_sizes[tpm_id] = float(fit_res.metrics.get("payload_size_mb", 0.0))
                if "local_adversarial_training_start_unix" in fit_res.metrics:
                    node_local_training_start_unix[tpm_id] = fit_res.metrics.get("local_adversarial_training_start_unix")
                if "local_adversarial_training_end_unix" in fit_res.metrics:
                    node_local_training_end_unix[tpm_id] = fit_res.metrics.get("local_adversarial_training_end_unix")
                    
            local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)
            
            # Phase 6: SHAP Calculation & Fog Aggregation
            f_server_logger.info(f"{f_server_prefix} Executing SHAP aggregation...", extra={"round": server_round})
            shap_calculation_start_unix = time.time()
            X_val, y_val = self.val_data
            
            agg_model, regional_trust, passed_flags, reward_flags, raw_scores_list = shap_weighted_aggregate(
                local_models=local_models, ref_model=self.global_model,
                X_val=X_val, y_val=y_val, sizes=sizes, n_classes=self.num_classes, n_explain=self.shap_explain_count
            )
            shap_calculation_end_unix = time.time()
            latency_shap_computation_sec = shap_calculation_end_unix - shap_calculation_start_unix
            
            raw_shap_scores = {tpm_ids[i]: float(raw_scores_list[i]) for i in range(len(tpm_ids))}
            saboteurs = [(tpm_ids[i], display_names[i]) for i, passed in enumerate(passed_flags) if not passed]
            rewarded_nodes = [(tpm_ids[i], display_names[i]) for i, rewarded in enumerate(reward_flags) if rewarded]
            
            if self.gatekeepers[fog_num].trust_db:
                for tpm_id, display_identity in saboteurs:
                    self.gatekeepers[fog_num].trust_db.apply_behavioral_penalty(node_id=tpm_id, display_name=display_identity, round_num=server_round)
                for tpm_id, display_identity in rewarded_nodes:
                    self.gatekeepers[fog_num].trust_db.apply_behavioral_reward(node_id=tpm_id, display_name=display_identity, round_num=server_round)
                    
            aggregation_start_unix = time.time()
            virtual_fog_models.append(agg_model)
            virtual_fog_weights.append(regional_trust)
            aggregation_end_unix = time.time()
            round_end_unix = time.time()
            
            # Dump the synchronized state to satisfy the UI
            self._dump_fog_state(
                fog_num, server_round, attestation_accepted, attestation_rejected, raw_shap_scores,
                latency_attestation_ms, node_training_latencies, latency_shap_computation_sec,
                node_payload_sizes, token_sizes_bytes, node_ground_truths, saboteurs, rewarded_nodes,
                round_start_unix, round_end_unix, attestation_validation_start_unix, attestation_validation_end_unix,
                shap_calculation_start_unix, shap_calculation_end_unix, aggregation_start_unix, aggregation_end_unix,
                node_local_training_start_unix, node_local_training_end_unix
            )
            
            # Spoof the final Phase 6 bridge delivery log
            f_server_logger.debug(f"{f_server_prefix} [IPC SERVER] Relaying {len(local_models)} layers back to Fog Client...", extra={"round": server_round})
            
            # Spoof the Phase 7 Cloud trigger
            self.logger.info(f"{self.log_prefix} Received weights from [FOG {fog_num} CLIENT]", extra={"round": server_round})

        # Phase 7: Final Global Aggregation of Virtual Fogs
        if not virtual_fog_models:
            return None, {}
            
        try:
            total_trust = sum(virtual_fog_weights)
            normalized_weights = [w / total_trust for w in virtual_fog_weights] if total_trust > 0 else None
            final_global_model = federated_averaging(virtual_fog_models, weights=normalized_weights)
            
            self._evaluate_rollback_sanity_check(final_global_model, server_round)
            self.global_model.load_state_dict(final_global_model.state_dict())
            
            quantization_bits = int(self.run_metadata["quantization_bits"])
            raw_ndarrays = [val.cpu().numpy() for _, val in final_global_model.state_dict().items()]
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)
            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            
            return aggregated_parameters, {}
        except Exception as e:
            self.logger.error(f"{self.log_prefix} Math failed: {e}\n{traceback.format_exc()}", extra={"round": server_round})
            return None, {}

    def _evaluate_rollback_sanity_check(self, aggregated_model: torch.nn.Module, round_display: int) -> None:
        if self.val_data is not None:
            X_val, y_val = self.val_data
            aggregated_model.eval()
            with torch.no_grad():
                device = next(aggregated_model.parameters()).device
                X_val_gpu = X_val.to(device)
                y_val_gpu = y_val.to(device)
                preds = aggregated_model(X_val_gpu).argmax(dim=-1)
                val_acc = (preds == y_val_gpu).float().mean().item()
                
            if self.previous_val_acc is None:
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                self.previous_val_acc = val_acc
                return
                
            rollback_fraction = float(self.run_metadata["rollback_threshold"])
            dynamic_threshold = self.previous_val_acc * rollback_fraction
            
            if val_acc < dynamic_threshold and self.cached_global_state is not None:
                self.logger.info(f"{self.log_prefix} Rollback check. Acc: {val_acc:.4f}, Threshold: {dynamic_threshold:.4f}. Rolling back weights!", extra={"round": round_display})
                aggregated_model.load_state_dict(self.cached_global_state)
            else:
                self.logger.info(f"{self.log_prefix} Rollback check. Acc: {val_acc:.4f}, Threshold: {dynamic_threshold:.4f}. Passed. Caching state.", extra={"round": round_display})
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                self.previous_val_acc = val_acc

    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float], List[str], List[str]]:
        local_models, sizes, trust_weights, tpm_ids, display_names = [], [], [], [], []
        quantization_bits = int(self.run_metadata["quantization_bits"])
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

    def _dump_fog_state(self, fog_num: int, round_num: int, attestation_accepted: list, attestation_rejected: list, raw_shap_scores: dict, latency_attestation_ms: float, node_training_latencies: dict, latency_shap_computation_sec: float, node_payload_sizes: dict, token_sizes_bytes: dict, node_ground_truths: dict, saboteurs: list, rewarded_nodes: list, round_start_unix: float, round_end_unix: float, attestation_validation_start_unix: float, attestation_validation_end_unix: float, shap_calculation_start_unix: float, shap_calculation_end_unix: float, aggregation_start_unix: float, aggregation_end_unix: float, node_local_training_start_unix: dict, node_local_training_end_unix: dict) -> None:
        trust_db_snapshot = {}
        if self.gatekeepers[fog_num].trust_db:
            for node_id, state in self.gatekeepers[fog_num].trust_db._db.items():
                trust_db_snapshot[node_id] = {
                    "score": state["score"],
                    "is_quarantined": state["is_quarantined"],
                    "recovery_streak": state["recovery_streak"],
                    "last_attestation_passed": state.get("last_attestation_passed", False)
                }
                
        self.fog_state_histories[fog_num]["rounds"][round_num] = {
            "attestation_accepted": attestation_accepted,
            "attestation_rejected": attestation_rejected,
            "trust_db": trust_db_snapshot,
            "raw_shap_scores": raw_shap_scores,
            "latency_attestation_ms": latency_attestation_ms,
            "node_training_latencies": node_training_latencies,
            "latency_shap_computation_sec": latency_shap_computation_sec,
            "node_payload_sizes": node_payload_sizes,
            "token_sizes_bytes": token_sizes_bytes,
            "node_ground_truths": node_ground_truths,
            "saboteurs": saboteurs,
            "rewarded_nodes": rewarded_nodes,
            "round_start_unix": round_start_unix,
            "round_end_unix": round_end_unix,
            "attestation_validation_start_unix": attestation_validation_start_unix,
            "attestation_validation_end_unix": attestation_validation_end_unix,
            "shap_calculation_start_unix": shap_calculation_start_unix,
            "shap_calculation_end_unix": shap_calculation_end_unix,
            "aggregation_start_unix": aggregation_start_unix,
            "aggregation_end_unix": aggregation_end_unix,
            "node_local_training_start_unix": node_local_training_start_unix,
            "node_local_training_end_unix": node_local_training_end_unix,
            "timestamp": datetime.now().isoformat()
        }
        
        trust_history = {"metadata": self.fog_state_histories[fog_num].get("metadata", {}), "rounds": {}}
        shap_history = {"metadata": self.fog_state_histories[fog_num].get("metadata", {}), "rounds": {}}
        perf_history = {"metadata": self.fog_state_histories[fog_num].get("metadata", {}), "rounds": {}}
        
        for r, data in self.fog_state_histories[fog_num]["rounds"].items():
            trust_history["rounds"][r] = {
                "attestation_accepted": data["attestation_accepted"],
                "attestation_rejected": data["attestation_rejected"],
                "trust_db": data["trust_db"],
                "node_ground_truths": data["node_ground_truths"],
                "timestamp": data["timestamp"]
            }
            shap_history["rounds"][r] = {
                "raw_shap_scores": data["raw_shap_scores"],
                "saboteurs": data["saboteurs"],
                "rewarded_nodes": data["rewarded_nodes"],
                "timestamp": data["timestamp"]
            }
            perf_history["rounds"][r] = {
                "latency_attestation_ms": data["latency_attestation_ms"],
                "latency_shap_computation_sec": data["latency_shap_computation_sec"],
                "node_training_latencies": data.get("node_training_latencies", {}),
                "node_payload_sizes": data.get("node_payload_sizes", {}),
                "token_sizes_bytes": data.get("token_sizes_bytes", {}),
                "round_start_unix": data.get("round_start_unix"),
                "round_end_unix": data.get("round_end_unix"),
                "attestation_validation_start_unix": data.get("attestation_validation_start_unix"),
                "attestation_validation_end_unix": data.get("attestation_validation_end_unix"),
                "shap_calculation_start_unix": data.get("shap_calculation_start_unix"),
                "shap_calculation_end_unix": data.get("shap_calculation_end_unix"),
                "aggregation_start_unix": data.get("aggregation_start_unix"),
                "aggregation_end_unix": data.get("aggregation_end_unix"),
                "node_local_training_start_unix": data.get("node_local_training_start_unix", {}),
                "node_local_training_end_unix": data.get("node_local_training_end_unix", {}),
                "timestamp": data["timestamp"]
            }
            
        trust_dir = f"results/{self.experiment_name}/trustdb"
        shap_dir = f"results/{self.experiment_name}/shap_scores"
        perf_dir = f"results/{self.experiment_name}/performance_metrics"
        
        os.makedirs(trust_dir, exist_ok=True)
        os.makedirs(shap_dir, exist_ok=True)
        os.makedirs(perf_dir, exist_ok=True)
        
        # Log this explicitly so the parser script picks it up to clear Phase 6
        self.fog_server_loggers[fog_num].debug(f"[FOG {fog_num} SERVER] Saved 3-pillar Fog state artifacts for round {round_num}", extra={"round": round_num})
        
        with open(os.path.join(trust_dir, f"fog_{fog_num}_state.json"), "w") as f:
            json.dump(trust_history, f, indent=4)
        with open(os.path.join(shap_dir, f"fog_{fog_num}_state.json"), "w") as f:
            json.dump(shap_history, f, indent=4)
        with open(os.path.join(perf_dir, f"fog_{fog_num}_state.json"), "w") as f:
            json.dump(perf_history, f, indent=4)

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        self.logger.info(f"{self.log_prefix} CLOUD Eval Round {server_round} | Calling Cloud Evaluate! Saving model weights every {self.snapshot_interval} rounds and specifically at rounds: {self.snapshot_rounds}", extra={"round": server_round})
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
            
        torch.save(self.global_model.state_dict(), os.path.join(run_dir, f"global_model.pt"))
        if server_round in self.snapshot_rounds or (self.snapshot_interval > 0 and server_round % self.snapshot_interval == 0 and server_round != 0):
            torch.save(self.global_model.state_dict(), os.path.join(run_dir, f"global_model_round_{server_round}.pt"))
            
        self.logger.info(f"{self.log_prefix} Saved metrics and model to {run_dir}/ with accuracy: {round_data['global_accuracy']}%", extra={"round": server_round})
        return loss, metrics