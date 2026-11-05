import os
import json
import time
import re
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
import torch
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, FitIns

from src.tier_cloud.model_state import GlobalModelState
from src.shared.network.model_payload import decode_model, positive_trust, validate_sample_count
from src.tier_fog.round_processing import process_region
from src.tier_fog.client_instructions import configure_edge_fit
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.shared.utils.logger_setup import setup_logger


class CloudAggregator(GlobalModelState, FedAvg):
    """The central orchestration engine routing the federation at Cloud Layer.
    Runs shared Fog verification locally; telemetry explicitly identifies simulated transport.
    """
    def __init__(
        self,
        logger: logging.Logger,
        log_prefix: str,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fog_val_data: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        n_features: int = 40,
        num_classes: int = 15,
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
        self.fog_val_data = fog_val_data or {}
        self.n_features = n_features
        self.num_classes = num_classes
        
        self.run_metadata = run_metadata or {}
        self.initialize_global_state()
        self.experiment_name = self.run_metadata.get("experiment_name", "default_run")
        self.snapshot_rounds = self.run_metadata["snapshot_rounds"]
        self.snapshot_interval = self.run_metadata["snapshot_interval"]
        self.num_fogs = self.run_metadata.get("num_fogs", 1)
        
        self.run_metadata["telemetry_mode"] = "simulated"
        self.results_dict = {
            "metadata": self.run_metadata,
            "performance": []
        }
        self.active_nonces = {}
        
        # Virtual Fog verification and explicitly simulated transport telemetry
        self.gatekeepers = {}
        self.fog_state_histories = {}
        
        # Dedicated loggers for the virtual three-tier structure
        self.fog_server_loggers = {}
        self.fog_client_loggers = {}
        
        for i in range(1, self.num_fogs + 1):
            f_server_prefix = f"[FOG {i} SERVER]"
            f_client_prefix = f"[FOG {i} CLIENT]"
            
            self.fog_server_loggers[i] = setup_logger(f_server_prefix)
            self.fog_client_loggers[i] = setup_logger(f_client_prefix)
            
            # Point the Gatekeeper to the FOG SERVER logger so it prints the attestation logs correctly
            gk = ZeroTrustGatekeeper(logger=self.fog_server_loggers[i], log_prefix=f_server_prefix, run_metadata=self.run_metadata, fog_num=i)
            self.gatekeepers[i] = gk
            self.fog_state_histories[i] = {
                "metadata": {"fog_num": i, "tier": "fog"},
                "rounds": {}
            }

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        self.sync_global_state(parameters)
        self.active_nonces = {}
        self.round_start_time = time.time()
        
        # Emit simulation phase events for the UI
        self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
        
        for i in range(1, self.num_fogs + 1):
            self.fog_client_loggers[i].info(f"[FOG {i} CLIENT] [SIMULATED IPC CLIENT] Connected! Sending START signal for round {server_round}.", extra={"round": server_round})
            self.fog_server_loggers[i].info(f"[FOG {i} SERVER] [SIMULATED IPC SERVER] START received for round {server_round}. Shouting to EDGE clients!", extra={"round": server_round})
        
        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        return configure_edge_fit(shared_instructions, list(self.gatekeepers.values()), server_round,
                                  self.active_nonces)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []  # GlobalEvaluator evaluates centrally without broadcasting to edges.

    def aggregate_fit(self, server_round, results, failures):
        groupings = {i: [] for i in self.gatekeepers}
        assignments = {cid: i for i, gatekeeper in self.gatekeepers.items()
                       for cid in gatekeeper.client_identities}
        assignments.update({cid: i for i, gatekeeper in self.gatekeepers.items()
                            for cid in gatekeeper.admission_outcomes})
        for proxy, result in results:
            fog = assignments.get(proxy.cid)
            if fog is None:
                label = result.metrics.get("log_prefix", "")
                match = re.fullmatch(r"\[EDGE ([1-9]\d*)_([1-9]\d*)\]", label) if isinstance(label, str) else None
                fog = int(match[1]) if match and int(match[1]) in groupings else 1
            groupings[fog].append((proxy, result))
            assignments[proxy.cid] = fog
        # Unknown clients with no response have no authenticated Fog assignment yet.
        for cid in self.active_nonces:
            assignments.setdefault(cid, 1)
        models, weights = [], []
        empty_regions, failed_regions = 0, 0
        for fog, cluster in groupings.items():
            parameters, _, metrics = process_region(
                gatekeeper=self.gatekeepers[fog], reference=self.global_model,
                val_data=self.fog_val_data.get(fog, self.val_data), results=cluster,
                nonces=self.active_nonces,
                expected_clients=[cid for cid in self.active_nonces if assignments[cid] == fog],
                metadata=self.run_metadata, fog_num=fog, round_num=server_round,
                history=self.fog_state_histories[fog], n_explain=self.shap_explain_count,
                num_classes=self.num_classes, simulated=True,
            )
            if parameters is not None:
                # The same Fog-to-Cloud encode/decode boundary as deployment.
                models.append(decode_model(self.global_model, parameters, self.quantization_bits))
                weights.append(positive_trust(metrics["total_regional_trust"]))
            elif metrics["aggregation_status"] == "error":
                failed_regions += 1
            else:
                empty_regions += 1
        parameters, metrics = self.aggregate_regions(models, weights, server_round)
        metrics.update(empty_regions=empty_regions, failed_regions=failed_regions, rejected_regions=0)
        return parameters, metrics

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        self.logger.info(f"{self.log_prefix} CLOUD Eval Round {server_round} | Calling Cloud Evaluate! Saving model weights every {self.snapshot_interval} rounds and specifically at rounds: {self.snapshot_rounds}", extra={"round": server_round})
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
            
        torch.save(self.global_model.state_dict(), os.path.join(run_dir, f"global_model.pt"))
        if server_round in self.snapshot_rounds or (self.snapshot_interval > 0 and server_round % self.snapshot_interval == 0 and server_round != 0):
            torch.save(self.global_model.state_dict(), os.path.join(run_dir, f"global_model_round_{server_round}.pt"))
            
        self.logger.info(f"{self.log_prefix} Saved metrics and model to {run_dir}/ with accuracy: {round_data['global_accuracy']}%", extra={"round": server_round})
        from src.shared.utils.simulation_lifecycle import record_saved_round
        record_saved_round(getattr(self, "run_metadata", {}), server_round)
        return loss, metrics
