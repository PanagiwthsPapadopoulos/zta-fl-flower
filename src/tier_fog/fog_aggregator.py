import logging
from typing import Tuple, List, Optional, Any
import torch
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns

from src.shared.models.factory import get_seeded_model
from src.shared.network.model_payload import encode_model, load_model_arrays
from src.tier_fog.round_processing import process_region
from src.tier_fog.client_instructions import configure_edge_fit
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
        self.global_model = get_seeded_model(self.model_architecture, self.n_features, self.num_classes,
                                            int(self.run_metadata.get("random_seed", 42)))
        self.initial_parameters = encode_model(self.global_model, int(self.run_metadata.get("quantization_bits", 32)))
        self.active_nonces = {}
        self.experiment_name = self.run_metadata.get("experiment_name", "default_run")
        self.fog_state_history = {
            "metadata": {"fog_num": self.fog_num, "tier": self.tier},
            "rounds": {}
        }
        self.gatekeeper = ZeroTrustGatekeeper(logger=self.logger, log_prefix=self.log_prefix, run_metadata=self.run_metadata, fog_num=self.fog_num)
        self.fog_bridge = FogBridgeServer(logger=self.logger, log_prefix=self.log_prefix, ipc_port=self.ipc_port, socket_timeout=self.socket_timeout)

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        """Synchronizes the training round and Cloud global model with the fog bridge, samples clients, and attaches nonces."""
        bridged_round, cloud_ndarrays = self.fog_bridge.wait_for_start()
        if bridged_round > 0:
            self.current_bridged_round = bridged_round
        else:
            self.logger.error(f"{self.log_prefix} [IPC SERVER] Failed to establish valid round sync.", extra={"round": server_round})
            raise ConnectionError("Fog Bridge failed to synchronize.")

        if not cloud_ndarrays:
            raise ValueError("Cloud supplied an empty model")
        load_model_arrays(self.global_model, cloud_ndarrays, int(self.run_metadata.get("quantization_bits", 32)))
        parameters = ndarrays_to_parameters(cloud_ndarrays)
        self.active_nonces = {}

        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        return configure_edge_fit(shared_instructions, [self.gatekeeper], self.current_bridged_round,
                                  self.active_nonces, min(float(self.socket_timeout), 60.0))

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Evaluation is centralized; do not redistribute weights to quarantined edges.
        return []

    def aggregate_fit(self, server_round, results, failures):
        parameters, samples, metrics = process_region(
            gatekeeper=self.gatekeeper, reference=self.global_model, val_data=self.val_data,
            results=results, nonces=self.active_nonces, expected_clients=self.active_nonces,
            metadata=self.run_metadata, fog_num=self.fog_num, round_num=self.current_bridged_round,
            history=self.fog_state_history, n_explain=self.shap_explain_count,
            num_classes=self.num_classes,
        )
        self._relay_ipc_fog_bridge(parameters, samples, metrics)
        return parameters, metrics

    def _relay_ipc_fog_bridge(self, aggregated_parameters: Optional[list], num_examples: int = 1, metrics: Optional[dict] = None) -> None:
        """Translates and relays the aggregated model weights and regional trust metrics to the external IPC bridge."""
        ndarrays_to_send = parameters_to_ndarrays(aggregated_parameters) if aggregated_parameters is not None else []
        self.fog_bridge.relay_weights(ndarrays_to_send, num_examples=num_examples, metrics=metrics)

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        """Overrides centralized global evaluation to conserve computational resources. Global model evaluation happens in the Cloud."""
        self.logger.info(f"{self.log_prefix} Bypassing global evaluation on Fog tier to save resources.", extra={"round": server_round})
        return None
