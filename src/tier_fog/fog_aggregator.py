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
        """Synchronizes the training round with the fog bridge, samples clients, and customizes 
        training configurations with strategy metadata and cryptographic nonces for security.
        
        This function is called before training."""
        
        # Communicate with Fog Client on which round is about to start.
        # If the round number is agreed upon, continue.
        import secrets
        bridged_round = self.fog_bridge.wait_for_start()
        if bridged_round > 0:
            self.current_bridged_round = bridged_round
        else:
            self.logger.error(f"{self.log_prefix} [IPC SERVER] Failed to establish valid round sync.", extra={"round": server_round})
            raise ConnectionError("Fog Bridge failed to synchronize.")

        # Execute scheduled admin actions, such as valid PCR ledger updates from a reliable source
        round_display = self.current_bridged_round
        self.admin_console.execute_scheduled_updates(round_display, self.fog_num)

        # Call the parent strategy's built-in logic
        shared_instructions = super().configure_fit(server_round, parameters, client_manager)
        new_client_instructions = []

        # Loop through every sampled client
        for client_proxy, shared_fit_ins in shared_instructions:
            client_config = shared_fit_ins.config.copy()
            client_config["server_round"] = self.current_bridged_round

            # Create and store the nonce for the attestation token
            nonce = secrets.token_hex(16)
            client_config["nonce"] = nonce
            self.active_nonces[client_proxy.cid] = nonce
            
            isolated_fit_ins = FitIns(parameters=shared_fit_ins.parameters, config=client_config)
            new_client_instructions.append((client_proxy, isolated_fit_ins))
            
        return new_client_instructions

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        """Filters, evaluates, and aggregates client weight updates while maintaining 
        a zero-trust node behavioral database and compressing the resulting model.
        
        This function is called after receiving the model updates from clients."""
        # Check if no clients returned updates to return early
        self.logger.info(f"{self.log_prefix} No client returned updates! Releasing IPC bridge with empty payload.")
        if not results:
            self._relay_ipc_fog_bridge(None)
            return None, {}

        # Filter the incoming nodes based on their token validation
        round_display = self.current_bridged_round
        trusted_results = self.gatekeeper.filter_node_updates(self.tier, round_display, results, self.active_nonces)

        # Alert if no nodes pass the authentication check
        # This only checks the identity of the node, not its behavior
        if not trusted_results:
            self.logger.warning(f"{self.log_prefix} No trusted results to aggregate! Releasing IPC bridge with empty payload.")
            self._relay_ipc_fog_bridge(None)
            return None, {}

        # From here and below, all nodes are authenticated 
        self.logger.info(f"{self.log_prefix} Executing SHAP aggregation...", extra={"round": round_display})

        # Unpack the data
        local_models, sizes, trust_weights, tpm_ids, display_names = self._extract_models_from_results(trusted_results)

        try:
            # Apply aggregation strategy
            aggregated_model, saboteurs, rewarded_nodes = self._apply_aggregation_strategy(local_models, sizes, trust_weights, tpm_ids, display_names)
            
            # Deal with the Saboteur and Rewarded Nodes accordingly
            if self.gatekeeper.trust_db:
                for tpm_id, display_identity in saboteurs:
                    self.logger.error(f"{self.log_prefix} SHAP SABOTAGE DETECTED: {display_identity} submitted statistically toxic weights. Retroactively slashing TrustDB score!", extra={"round": round_display})
                    self.gatekeeper.trust_db.process_attestation(node_id=tpm_id, display_name=display_identity, is_valid=False, round_num=round_display)
                    
                for tpm_id, display_identity in rewarded_nodes:
                    self.logger.info(f"{self.log_prefix} SHAP EXCELLENCE: {display_identity} strictly exceeded median stability. Granting behavioral trust reward!", extra={"round": round_display})
                    self.gatekeeper.trust_db.apply_behavioral_reward(node_id=tpm_id, display_name=display_identity, round_num=round_display)

            # Apply rollback if needed
            self._evaluate_rollback_sanity_check(aggregated_model, round_display)

            # Update the fog's model in memory
            self.global_model.load_state_dict(aggregated_model.state_dict())
            
            # Convert the tensors into raw NumPy arrays and compress them 
            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]
            compressed_ndarrays = compress_weights(raw_ndarrays, quantization_bits)

            # Convert the NumPy arrays into Flower's specific Parameters object
            # so that they can be transmitted over the network.
            aggregated_parameters = ndarrays_to_parameters(compressed_ndarrays)
            aggregated_metrics = {}

            # Create a dictionary to piggyback custom metrics alongside model weights
            if hasattr(self, 'current_regional_trust'):
                aggregated_metrics["total_regional_trust"] = self.current_regional_trust

        except Exception as e:
            self.logger.error(f"{self.log_prefix} Math failed: {e}\n{traceback.format_exc()}", extra={"round": round_display})
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, trusted_results, failures)

        # Relay aggregated parameters to fog client
        self._relay_ipc_fog_bridge(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float], List[str], List[str]]:
        """Unpacks, decompresses, and translates client network payloads into native PyTorch models.

        This utility iterates through verified client results, reversing network quantization 
        and translating Flower parameter objects back into state dictionaries. It maps these 
        dictionaries to PyTorch architectures while extracting relevant metadata for weighted aggregation."""
        local_models, sizes, trust_weights, tpm_ids, display_names = [], [], [], [], []
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32))

        for client_proxy, fit_res in trusted_results:
            # Extract the TPM ID and display identity
            tpm_id = fit_res.metrics.get("tpm_id", f"CID-{client_proxy.cid}")
            display_identity = fit_res.metrics.get("display_identity", f"{tpm_id} (Unknown)")
            
            # Create a fresh, blank PyTorch model skeleton matching the expected architecture
            model = get_model(self.model_architecture, self.n_features, self.num_classes)
            
            # Decompress the arrays back to 32-bit precision floats
            raw_params = parameters_to_ndarrays(fit_res.parameters)
            decompressed_params = decompress_weights(raw_params, quantization_bits)

            # Convert the NumPy arrays into PyTorch Tensors and zip them into an OrderedDict matching the model's layer names
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), decompressed_params)})
           
            # Inject the tensors into the blank PyTorch model.
            # If they don't perfectly match the server's expected architecture, intentionally crash.
            model.load_state_dict(state_dict, strict=True)

            # Grab the number of training examples the client used and their current trust scores
            # to use them later to weight their contribution during aggregation.
            local_models.append(model)
            sizes.append(fit_res.num_examples)
            trust_weights.append(float(fit_res.metrics.get("total_regional_trust", 1.0)))
            tpm_ids.append(tpm_id)
            display_names.append(display_identity)

        return local_models, sizes, trust_weights, tpm_ids, display_names

    def _apply_aggregation_strategy(self, local_models: List[torch.nn.Module], sizes: List[int], trust_weights: List[float], tpm_ids: List[str], display_names: List[str]) -> Tuple[torch.nn.Module, List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Evaluates local models using SHAP validation data and delegates aggregation."""
        # Check if the server has a local validation dataset.
        # THIS IS A CRUCIAL STEP! This aggregation strategy needs
        # a clean, trusted, "ground truth" dataset to test against client updates
        if self.val_data is None:
            raise ValueError("ZTA strategy requires val_data for SHAP verification, but it was None.")

        # Unpack the validation features and labels
        X_val, y_val = self.val_data

        # Call the aggregation strategy
        agg_model, regional_trust, passed_flags, reward_flags = shap_weighted_aggregate(
            local_models=local_models, ref_model=self.global_model,
            X_val=X_val, y_val=y_val, sizes=sizes, n_classes=self.num_classes, n_explain=self.shap_explain_count
        )
        self.current_regional_trust = regional_trust

        # Map the returned True/Flase arrays back to the actual hardware IDs and display names
        saboteurs = [(tpm_ids[i], display_names[i]) for i, passed in enumerate(passed_flags) if not passed]
        rewarded_nodes = [(tpm_ids[i], display_names[i]) for i, rewarded in enumerate(reward_flags) if rewarded]
        
        return agg_model, saboteurs, rewarded_nodes

    def _evaluate_rollback_sanity_check(self, aggregated_model: torch.nn.Module, round_display: int) -> None:
        """Evaluates the aggregated model and rolls back to a previous state if performance collapses."""
        if self.val_data is not None:
            aggregated_model.eval()
            # Disable gradient tracking to save memory
            with torch.no_grad():
                X_val, y_val = self.val_data
                # Calculate current accuracy of the model
                preds = aggregated_model(X_val).argmax(dim=-1)
                val_acc = (preds == y_val).float().mean().item()
            
            # If this is the first time the check is running
            # Take a deep copy of the model's weights and save the accuracy
            if self.previous_val_acc is None:
                self.logger.info(f"{self.log_prefix} ✅ Initializing baseline state.")
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                self.previous_val_acc = val_acc
                return
            
            # Calculate the dynamic minimum acceptable accuracy
            rollback_fraction = float(self.run_metadata.get("rollback_threshold", 0.80))
            dynamic_threshold = self.previous_val_acc * rollback_fraction

            # Check the current accuracy against the previous round's accuracy
            if val_acc < dynamic_threshold and self.cached_global_state is not None:
                self.logger.info(f"{self.log_prefix} Checking accuracy for rollback with fraction={rollback_fraction}. Accuracy calculated: {{val_acc.4f}} and dynamic threshold: {{dynamic_threshold:.4f}}. Rolling back to previous round weights!", extra={"round": round_display})
                aggregated_model.load_state_dict(self.cached_global_state)
            else:
                self.logger.info(f"{self.log_prefix} Checking accuracy for rollback with fraction={rollback_fraction}. Accuracy calculated: {{val_acc.4f}} and dynamic threshold: {{dynamic_threshold:.4f}}. Aggregation passed sanity check. Caching state.", extra={"round": round_display})
                self.cached_global_state = copy.deepcopy(aggregated_model.state_dict())
                # Save the accuracy for the next round
                self.previous_val_acc = val_acc

    def _relay_ipc_fog_bridge(self, aggregated_parameters: Optional[list]) -> None:
        """Translates and relays the aggregated model weights to the external IPC bridge."""
        ndarrays_to_send = parameters_to_ndarrays(aggregated_parameters) if aggregated_parameters is not None else []
        self.fog_bridge.relay_weights(ndarrays_to_send)

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        """Overrides centralized global evaluation to conserve computational resources.
           Global model evaluation happens in the Cloud."""
        self.logger.info(f"{self.log_prefix} Bypassing global evaluation on Fog tier to save resources.", extra={"round": server_round})
        return None