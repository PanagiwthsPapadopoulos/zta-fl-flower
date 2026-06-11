import os
import ast
import json
import copy
import socket
import logging
import traceback
import gc
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

# Limits thread usage to prevent OpenMP CPU deadlocks during heavy Server evaluation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset

from flwr.common import ndarrays_to_parameters, Context, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Local security and model imports
from src.security.backdoor import compute_backdoor_asr
from src.security.adversarial import evaluate_robustness
from src.federation.aggregation import (
    federated_averaging, 
    shap_weighted_aggregate,
    krum_select, 
    trimmed_mean_aggregate, 
    flame_aggregate, 
    fltrust_aggregate
)
from src.network.ipc import send_msg, recv_msg
from src.utils.logger_setup import setup_logger
from src.models.factory import get_model
from src.utils.metrics import accuracy, macro_f1
from src.utils.data_loader import get_dataset, DATASET_METADATA
from src.utils.compression import compress_weights, decompress_weights


GLOBAL_DATA_CACHE = {}


def get_evaluate_fn(
    dataset: str, 
    dataset_path: str, 
    num_classes: int, 
    n_features: int, 
    device: str, 
    random_seed: int, 
    run_metadata: dict, 
    tier: str,
    logger: logging.Logger
) -> Callable:
    """
    Builds the central evaluation callback routine that pushes the global model through its paces.
    Hooks into the designated test arrays to crunch core metrics (Accuracy, Macro-F1) alongside
    heavy-hitting security metrics like the Backdoor Attack Success Rate (ASR) and the
    robustness drops under raw PGD/FGSM adversarial fire.
    """
    
    def evaluate(server_round: int, parameters: list, config: dict) -> Tuple[float, dict]:

        # Employs a lazy-loading strategy, delaying massive memory allocations until the combat rounds actually start.
        if "eval_data" not in GLOBAL_DATA_CACHE:
            logger.info("[EVALUATOR] Lazy loading centralized test set...")

        # Dynamically unpack and ensure SMOTE is OFF for evaluation
        simulate_leakage = run_metadata.get("simulate_global_leakage", False)
        eval_split = "val" if tier == "fog" else "test"
        eval_fraction = float(run_metadata.get("dataset_fraction", 1.0))

        cache_key = f"eval_{dataset}_{eval_split}_{eval_fraction}_{simulate_leakage}"

        logger.debug(f"[CONFIG USAGE] get_dataset | simulate_global_leakage: {simulate_leakage}")

        dataset_returns = get_dataset(
            dataset_name=dataset, 
            dataset_path=dataset_path, 
            num_classes=num_classes, 
            random_seed=random_seed,
            simulate_global_leakage=simulate_leakage, 
            apply_smote=False,
            split=eval_split,
            test_split=float(run_metadata.get("test_split", 0.30)),
            val_split=float(run_metadata.get("val_split", 0.50))
        )
        
        X_full = dataset_returns[0]
        y_full = dataset_returns[1]
        
        # Applies isolated data transformations to the server pool if operating under strict, non-leaking bounds.
        if len(dataset_returns) > 3:
            server_scaler = dataset_returns[3]
            server_pca = dataset_returns[4]
            
            X_full_np = X_full.numpy() if isinstance(X_full, torch.Tensor) else X_full
            X_full_np = server_scaler.transform(X_full_np)
            X_full_np = server_pca.transform(X_full_np)
            X_full = torch.tensor(X_full_np, dtype=torch.float32)

        # Scrambles the array with an anchored seed before slicing to prevent localized class starvation.
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(len(X_full), generator=generator)
        X_full = X_full[indices]
        y_full = y_full[indices]

        # 2. Then apply the dataset fraction cutoff
        if eval_fraction < 1.0:
            subset_size = int(len(X_full) * eval_fraction)
            X_full = X_full[:subset_size]
            y_full = y_full[:subset_size]
                
        GLOBAL_DATA_CACHE[cache_key] = (X_full, y_full)
        logger.info(f"[EVALUATOR] Test set ready! Dimensions restricted to: {X_full.shape}")
            
        X_test, y_test = GLOBAL_DATA_CACHE[cache_key]
        batch_size = int(run_metadata.get("batch_size", 256))
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"[EVALUATOR] Started Global Evaluation for Round {server_round} on {len(X_test)} samples!")

        model_architecture = str(run_metadata.get("model_architecture", "cnnlstm"))
        model = get_model(model_architecture, n_features, num_classes)
        
        quantization_bits = int(run_metadata.get("quantization_bits", 32))
        logger.debug(f"[CONFIG USAGE] decompress_weights | quantization_bits: {quantization_bits}")
        decompressed_params = decompress_weights(parameters, quantization_bits)
        
        params_dict = zip(model.state_dict().keys(), decompressed_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                
                all_preds.append(logits.argmax(dim=-1).cpu())

        preds_tensor = torch.cat(all_preds)
        avg_loss = total_loss / len(test_dataset)

        acc = accuracy(y_test.cpu(), preds_tensor)
        f1 = macro_f1(y_test.cpu(), preds_tensor, n_classes=num_classes)

        # Triggers the backdoor assessment loop: patches the trigger pattern onto clean records
        # and tests if the model blindly maps them into the malicious target space.
        backdoor_target_class = int(run_metadata.get("backdoor_target_class", 0))
        backdoor_trigger_value = float(run_metadata.get("backdoor_trigger_value", 1.5))
        backdoor_trigger_features_str = run_metadata.get("backdoor_trigger_features", "[-3, -2, -1]")
        
        if isinstance(backdoor_trigger_features_str, str):
            try:
                backdoor_trigger_features = ast.literal_eval(backdoor_trigger_features_str)
            except Exception:
                backdoor_trigger_features = [-3, -2, -1]
        else:
            backdoor_trigger_features = backdoor_trigger_features_str

        logger.debug(f"[CONFIG USAGE] compute_backdoor_asr | backdoor_target_class: {backdoor_target_class}, backdoor_trigger_features: {backdoor_trigger_features_str}, backdoor_trigger_value: {backdoor_trigger_value}, batch_size: {batch_size}")
        asr = compute_backdoor_asr(
            model=model, X_test=X_test, y_test=y_test,
            target_class=backdoor_target_class, trigger_features=tuple(backdoor_trigger_features),
            trigger_value=backdoor_trigger_value, device=device, batch_size=batch_size
        )
        
        # Puts the parameters under real fire: calculates the performance degradation delta
        # under continuous Projected Gradient Descent and Fast Gradient Sign Method attacks.
        pgd_eps = float(run_metadata.get("pgd_eps", 0.1))
        pgd_alpha = float(run_metadata.get("pgd_alpha", 0.01))
        pgd_n_iter = int(run_metadata.get("pgd_n_iter", 7))
        fgsm_eps = float(run_metadata.get("fgsm_eps", 0.2))
        fgsm_alpha = float(run_metadata.get("fgsm_alpha", 0.02))
        clip_min = float(run_metadata.get("clip_min", 0.0))
        clip_max = float(run_metadata.get("clip_max", 1.0))
        
        logger.debug(f"[CONFIG USAGE] evaluate_robustness (pgd) | pgd_eps: {pgd_eps}, pgd_alpha: {pgd_alpha}, pgd_n_iter: {pgd_n_iter}, batch_size: {batch_size}, clip_min: {clip_min}, clip_max: {clip_max}")
        robustness_pgd = evaluate_robustness(
            model=model, X=X_test, y=y_test, attack="pgd",
            eps=pgd_eps, alpha=pgd_alpha, n_iter=pgd_n_iter,
            batch_size=batch_size, device=device, clip_min=clip_min, clip_max=clip_max
        )
        
        logger.debug(f"[CONFIG USAGE] evaluate_robustness (fgsm) | fgsm_eps: {fgsm_eps}, fgsm_alpha: {fgsm_alpha}, n_iter: 1, batch_size: {batch_size}, clip_min: {clip_min}, clip_max: {clip_max}")
        robustness_fgsm = evaluate_robustness(
            model=model, X=X_test, y=y_test, attack="fgsm",
            eps=fgsm_eps, alpha=fgsm_alpha, n_iter=1,
            batch_size=batch_size, device=device, clip_min=clip_min, clip_max=clip_max
        )

        return avg_loss, {
            "accuracy": acc, "macro_f1": f1, "asr": asr, 
            "robustness_pgd": robustness_pgd, "robustness_fgsm": robustness_fgsm
        }
        
    return evaluate


class Strategy(FedAvg):
    """
    The central orchestration engine routing the federation.
    
    Acts as the network dispatcher bridging cross-tier TCP flows and managing the global
    weights array. Intercepts incoming parameters, handles the raw math, and triggers
    system rollbacks if the network gets poisoned.
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
        self.ipc_conn = None
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

        # --- Zero-Trust Attestation Engine & Policy Engine Provisioning ---
        if self.tier == "fog":
            from src.security.tpm_core import TPMEngine
            from src.security.trust_db import TrustDB
            
            insecure_flag = self.run_metadata.get("insecure", False) or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
            self.tpm_engine = TPMEngine(logger=self.logger,insecure_mode=insecure_flag)
            self.trust_db = TrustDB(logger=self.logger,tau_min=float(self.run_metadata.get("tau_min", 0.6)))
            self.active_nonces = {}

            # Boots up the local IPC listener if the process is anchoring a fog subnet
            self.ipc_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ipc_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Boots up the local IPC listener if the process is anchoring a fog subnet
        if self.tier == "fog":
            self.ipc_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ipc_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.ipc_server_sock.settimeout(self.socket_timeout) 
            self.ipc_server_sock.bind(("0.0.0.0", self.ipc_port))
            self.ipc_server_sock.listen(1)
            self.logger.info(f"{self.log_prefix} [IPC] Listening on port {self.ipc_port}. Awaiting Cloud signal...")

    def configure_fit(self, server_round: int, parameters: list, client_manager: Any) -> list:
        """
        Dispatches fit instructions down the hierarchical topology.
        Locks operations on the intermediate nodes until internal handshake protocols sync up the crew.
        """
        if self.tier == "cloud":
            self.logger.info(f"{self.log_prefix} Shouting to all FOG clients!", extra={"round": server_round})
            
        elif self.tier == "fog":
            self.logger.info(f"{self.log_prefix} [IPC] Blocking until local Fog Client bridges...")
            try:
                self.ipc_conn, _ = self.ipc_server_sock.accept()
                self.ipc_conn.settimeout(self.socket_timeout)
                msg = recv_msg(self.ipc_conn)
                
                if isinstance(msg, dict) and msg.get("cmd") == "START":
                    self.current_bridged_round = msg.get("round", 0)
                    self.logger.info(f"{self.log_prefix} START received. Shouting to EDGE clients!", extra={"round": self.current_bridged_round})
            except socket.timeout:
                self.logger.error(f"{self.log_prefix} [IPC] Timeout waiting for Fog Client!", extra={"round": server_round})
                raise

        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Injects synchronization rounds and current strategies to the downstream instructions
        if self.tier == "fog":
            import secrets
            for client_proxy, fit_ins in client_instructions:
                fit_ins.config["server_round"] = self.current_bridged_round
                fit_ins.config["strategy"] = self.strategy

            # Dynamic Nonce Dispersion for Anti-Replay Verification
                if self.strategy in ["zta", "ztafl"]:
                    nonce = secrets.token_hex(16)
                    fit_ins.config["nonce"] = nonce
                    self.active_nonces[client_proxy.cid] = nonce

        return client_instructions

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> Tuple[Optional[list], dict]:
        """
        Processes the aggregated results received from the downstream layer.
        Catches math faults and routes them securely to prevent cascading infrastructure collapse.
        """
        if not results:
            return None, {}

        trusted_results = self._filter_results(server_round, results)
        
        if not trusted_results:
            return None, {}

        round_display = self.current_bridged_round if self.tier == 'fog' else server_round
        self.logger.info(f"{self.log_prefix} 🧮 Executing {self.strategy.upper()} PyTorch aggregation...", extra={"round": round_display})

        local_models, sizes, trust_weights = self._extract_models_from_results(trusted_results)

        try:
            aggregated_model = self._apply_aggregation_strategy(local_models, sizes, trust_weights)
            self._evaluate_rollback_sanity_check(aggregated_model, round_display)

            # Saves the definitive global state and packages parameters for distribution
            self.global_model.load_state_dict(aggregated_model.state_dict())
            raw_ndarrays = [val.cpu().numpy() for _, val in aggregated_model.state_dict().items()]
            
            quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
            self.logger.debug(f"[CONFIG USAGE] compress_weights | quantization_bits: {quantization_bits}")
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

        self.logger.info(f"{self.log_prefix} ✅ Aggregation successfully finished! Handing off to Evaluator.")

        return aggregated_parameters, aggregated_metrics

    # --- Internal Handlers ---

    def _filter_results(self, server_round: int, results: list) -> list:
        """Filters out zero-weight reports and logs incoming client parameters."""
        trusted_results = []
        if self.tier == "cloud":
            for client_proxy, fit_res in results:
                node_name = fit_res.metrics.get("node_name", f"Unknown CID: {client_proxy.cid}")
                self.logger.info(f"{self.log_prefix} 📥 Received weights from {node_name}", extra={"round": server_round})
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
        else:
            for client_proxy, fit_res in results:
                node_name = fit_res.metrics.get("node_name", f"Unknown CID: {client_proxy.cid}")
                round_display = self.current_bridged_round
                
                # --- Zero-Trust Application Layer Gatekeeping ---
                if self.strategy in ["zta", "ztafl"]:
                    tpm_token_json = fit_res.metrics.get("tpm_token_json", "")
                    
                    # Check Quarantine Status
                    if self.trust_db.is_quarantined(node_name):
                        self.logger.warning(f"{self.log_prefix} 🛑 REJECTED: Agent {node_name} is currently quarantined in TrustDB!", extra={"round": round_display})
                        continue
                    
                    # Verify Crytographic Attestation Token Structure
                    authenticated = False
                    if tpm_token_json:
                        try:
                            token = json.loads(tpm_token_json)
                            expected_nonce = self.active_nonces.get(client_proxy.cid, "")
                            pubkey_path = f"/app/data/tpm_state/{node_name.lower().replace('[', '').replace(']', '').replace(' ', '_')}/ak.pub"
                            authenticated = self.tpm_engine.verify_attestation_token(token, expected_nonce, pubkey_path)
                        except Exception as e:
                            self.logger.error(f"{self.log_prefix} Error extracting TPM data payload for {node_name}: {str(e)}")
                    
                    if not authenticated:
                        self.logger.warning(f"{self.log_prefix} 🔒 ATTESTATION FAILED: Agent {node_name} failed software integrity check!", extra={"round": round_display})
                        self.trust_db.penalize_agent(node_name, "Attestation Fault")
                        continue
                        
                    # Reward persistent behavior following successful authorization checkpoint
                    self.trust_db.reward_agent(node_name)
                
                self.logger.info(f"{self.log_prefix} Received weights from {node_name}", extra={"round": round_display})
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
                
        return trusted_results
                
    def _extract_models_from_results(self, trusted_results: list) -> Tuple[List[torch.nn.Module], List[int], List[float]]:
        """Decompresses the inbound payloads back into functional PyTorch architectures."""
        local_models, sizes, trust_weights = [], [], []
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32)) 
        
        for client_proxy, fit_res in trusted_results:
            model = get_model(self.model_architecture, self.n_features, self.num_classes)
            raw_params = parameters_to_ndarrays(fit_res.parameters)
            decompressed_params = decompress_weights(raw_params, quantization_bits)
            
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), decompressed_params)})
            model.load_state_dict(state_dict, strict=True)
            
            local_models.append(model)
            sizes.append(fit_res.num_examples)
            # Extracts the regional trust metric broadcasted up from the fog layers.
            trust_weights.append(float(fit_res.metrics.get("total_regional_trust", 1.0)))
            
        return local_models, sizes, trust_weights

    def _apply_aggregation_strategy(self, local_models: List[torch.nn.Module], sizes: List[int], trust_weights: List[float]) -> torch.nn.Module:
        """Routes the parameters through the specific mathematical defense mechanisms currently armed in the configuration."""
        if self.strategy in ["zta", "ztafl"] and self.tier == "fog":
            X_val, y_val = self.val_data
           
            agg_model, regional_trust = shap_weighted_aggregate(
                local_models=local_models, ref_model=self.global_model,
                X_val=X_val, y_val=y_val, sizes=sizes, n_classes=self.num_classes, n_explain=self.shap_explain_count
            )
            self.current_regional_trust = regional_trust 
            return agg_model
            
        elif self.strategy in ["zta", "ztafl"] and self.tier == "cloud":
            # Implements the strict ZTA Cloud formula: ignores sample volumes and aggregates using purely structural trust variables.
            total_trust = sum(trust_weights)
            weights = [w / total_trust for w in trust_weights] if total_trust > 0 else None
            return federated_averaging(local_models, weights=weights)
            
        elif self.strategy in ["fedavg", "fedprox"]:
            total_samples = sum(sizes)
            weights = [s / total_samples for s in sizes] if total_samples > 0 else None
            return federated_averaging(local_models, weights=weights)
            
        elif self.strategy == "krum":
            default_f = max(1, int(len(local_models) * 0.3))
            krum_f = int(self.run_metadata.get("krum_f", default_f))
            
            self.logger.debug(f"[CONFIG USAGE] krum_select | krum_f: {krum_f}")
            return krum_select(local_models, f=krum_f)
            
        elif self.strategy == "trimmed_mean":
            trimmed_mean_beta = float(self.run_metadata.get("trimmed_mean_beta", 0.1))
            
            self.logger.debug(f"[CONFIG USAGE] trimmed_mean_aggregate | trimmed_mean_beta: {trimmed_mean_beta}")
            return trimmed_mean_aggregate(local_models, beta=trimmed_mean_beta)
            
        elif self.strategy == "flame":
            flame_target_frac = float(self.run_metadata.get("flame_target_frac", 0.5))
            
            self.logger.debug(f"[CONFIG USAGE] flame_aggregate | flame_target_frac: {flame_target_frac}")
            return flame_aggregate(local_models, self.global_model, target_frac=flame_target_frac)    
                
        elif self.strategy == "fltrust":
            server_model = get_model(self.model_architecture, self.n_features, self.num_classes)
            server_model.load_state_dict(self.global_model.state_dict())
            if self.val_data is not None:
                X_val, y_val = self.val_data
                learning_rate = float(self.run_metadata.get("learning_rate", 0.001))
                batch_size = int(self.run_metadata.get("batch_size", 32))
                
                self.logger.debug(f"[CONFIG USAGE] fltrust_aggregate | learning_rate: {learning_rate}, batch_size: {batch_size}")
                
                optimizer = torch.optim.Adam(server_model.parameters(), lr=learning_rate)
                criterion = torch.nn.CrossEntropyLoss()
                server_model.train()

                # Safely batch the server's clean dataset to prevent OOM
                dataset = TensorDataset(X_val, y_val)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                for X_batch, y_batch in loader:
                    if X_batch.size(0) < 2:
                        continue # BatchNorm protection
                    optimizer.zero_grad()
                    loss = criterion(server_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                    
            return fltrust_aggregate(local_models, server_model, self.global_model)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _evaluate_rollback_sanity_check(self, aggregated_model: torch.nn.Module, round_display: int) -> None:
        """
        Operates as the final safety net preventing a compromised aggregation from corrupting the master state.
        
        Evaluates the new weights on the clean validation set. If the accuracy tanks below the
        dynamic threshold boundary (e.g. $Acc_t < Acc_{t-1} \times \tau_{rollback}$), the system
        purges the update and reverts to the previous operational state.
        """
        if self.tier == "fog" and self.val_data is not None:
            aggregated_model.eval()
            with torch.no_grad():
                X_val, y_val = self.val_data
                preds = aggregated_model(X_val).argmax(dim=-1)
                val_acc = (preds == y_val).float().mean().item()
                
            self.logger.info(f"{self.log_prefix} 🔍 Post-Aggregation Validation Accuracy: {val_acc:.4f}", extra={"round": round_display})
            
            rollback_fraction = float(self.run_metadata.get("rollback_threshold", 0.80))

            self.logger.info(f"[CONFIG USAGE] _evaluate_rollback_sanity_check | rollback_threshold: {rollback_fraction}")
            
            if self.previous_val_acc is not None:
                dynamic_threshold = self.previous_val_acc * rollback_fraction
            else:
                dynamic_threshold = 0.0 # First round has no previous benchmark to drop from
                
            self.logger.debug(f"[CONFIG USAGE] evaluate_rollback_sanity_check | prev_acc: {self.previous_val_acc}, dynamic_threshold: {dynamic_threshold:.4f}")
            
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
        """Passes the newly aggregated parameters back over the socket connection."""
        if self.tier == "fog":
            if aggregated_parameters is not None:
                send_msg(self.ipc_conn, parameters_to_ndarrays(aggregated_parameters))
            else:
                send_msg(self.ipc_conn, [])
            self.ipc_conn.close()
            self.ipc_conn = None

    def evaluate(self, server_round: int, parameters: list) -> Optional[Tuple[float, dict]]:
        """
        Triggers the global model evaluation protocol managed natively by Flower.
        Bypassed by intermediate fog nodes to consolidate processing overhead entirely onto the cloud instance.
        """
        if self.tier == "fog":
            self.logger.info(f"{self.log_prefix} Bypassing global evaluation on Fog tier to save resources.")
            return None
        
        eval_res = super().evaluate(server_round, parameters)
        
        if eval_res is None:
            self.logger.warning(f"{self.log_prefix} Evaluation failed or did not run!")
            return None
            
        loss, metrics = eval_res

        # Locally saves JSON metrics exclusively on the overarching cloud tier
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
            
            # Create a dedicated folder for this specific run
            run_dir = f"results/{self.experiment_name}"
            os.makedirs(run_dir, exist_ok=True)

            # Save the metrics JSON into this folder
            filepath = f"{run_dir}/{self.experiment_name}.json"
            with open(filepath, "w") as f:
                json.dump(self.results_dict, f, indent=4) 

            # Save the global model weights into this folder
            model_filepath = os.path.join(run_dir, "global_model.pt")
            torch.save(self.global_model.state_dict(), model_filepath)
                
            self.logger.info(f"{self.log_prefix} 💾 Saved metrics and model to {run_dir}/", extra={"round": server_round})
            
        return loss, metrics


class ZTACloudStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """
        Implements the specialized ZTA-FL Cloud Aggregation formula.
        Bypasses default client dataset sizing logic. Averages the parameters exclusively
        through the regional trust multipliers $w_f$ pushed upstream by the fog layers:
        r"$\theta^{t+1} = \sum_{f=1}^M \left(\frac{w_f}{\sum w_f}\right) \theta_f^t$".
        """
        if not results:
            return None, {}

        # Extract the models and the custom Fog Trust Weights (w_f)
        fog_updates = []
        total_cloud_weight = 0.0
        
        for client_proxy, fit_res in results:
            fog_model_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            # Extract the regional trust weight sent by the Fog node (default to 1.0 if missing)
            w_f = float(fit_res.metrics.get("total_regional_trust", 1.0))
            
            fog_updates.append((fog_model_ndarrays, w_f))
            total_cloud_weight += w_f

        # Normalize the Fog weights
        normalized_updates = []
        for fog_model, w_f in fog_updates:
            normalized_weight = w_f / total_cloud_weight if total_cloud_weight > 0 else 1.0 / len(fog_updates)
            normalized_updates.append((fog_model, normalized_weight))

        # Execute the exact formula: theta^{t+1} = sum(w_f * theta_f^t)
        aggregated_ndarrays = []
        for i in range(len(normalized_updates[0][0])):
            layer_sum = sum(weight * model[i] for model, weight in normalized_updates)
            aggregated_ndarrays.append(layer_sum)

        return ndarrays_to_parameters(aggregated_ndarrays), {}

# --- Factory Functions ---

def fit_config(server_round: int) -> dict:
    """Provides dynamic fit instructions on a per-round basis."""
    return {"server_round": server_round}


def _build_run_metadata(run_config: dict) -> dict:
    """Extracts and standardizes the entire run_config object into a unified dictionary."""
    experiment_name = str(run_config.get("run_name", "")).strip()
    if not experiment_name:
        experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return {
        "experiment_name": experiment_name,
        "strategy": str(run_config.get("strategy", "zta")),
        "timestamp": datetime.now().isoformat(),
        "dataset": str(run_config.get("dataset", "edge_iiotset")),
        "dataset_fraction": float(run_config.get("dataset_fraction", 1.0)),
        "model_architecture": str(run_config.get("model_architecture", "cnnlstm")),
        "num_rounds": int(run_config.get("num_rounds", 3)),
        "min_clients": int(run_config.get("min-clients", 1)),
        "learning_rate": float(run_config.get("learning_rate", 0.001)),
        "batch_size": int(run_config.get("batch_size", 32)),
        "random_seed": int(run_config.get("random_seed", 42)),
        "n_classes_per": int(run_config.get("n_classes_per", 3)),
        "shap_threshold": float(run_config.get("shap_threshold", 0.5)),
        "shap_val_samples": int(run_config.get("shap_val_samples", 100)),
        "shap_explain_count": int(run_config.get("shap_explain_count", 10)),
        "pgd_ratio": float(run_config.get("pgd_ratio", 0.0)),
        "fgsm_ratio": float(run_config.get("fgsm_ratio", 0.0)),
        "backdoor_ratio": float(run_config.get("backdoor_ratio", 0.0)),
        "pgd_adv_ratio": float(run_config.get("pgd_adv_ratio", 0.3)),
        "pgd_eps": float(run_config.get("pgd_eps", 0.1)),
        "pgd_alpha": float(run_config.get("pgd_alpha", 0.01)),
        "pgd_n_iter": int(run_config.get("pgd_n_iter", 7)),
        "fgsm_adv_ratio": float(run_config.get("fgsm_adv_ratio", 0.5)),
        "fgsm_eps": float(run_config.get("fgsm_eps", 0.2)),
        "fgsm_alpha": float(run_config.get("fgsm_alpha", 0.02)),
        "backdoor_poison_fraction": float(run_config.get("backdoor_poison_fraction", 0.5)),
        "backdoor_target_class": int(run_config.get("backdoor_target_class", 0)),
        "backdoor_trigger_features": str(run_config.get("backdoor_trigger_features", "[-3, -2, -1]")),
        "backdoor_trigger_value": float(run_config.get("backdoor_trigger_value", 1.5)),
        "benign_adv_ratio": float(run_config.get("benign_adv_ratio", 0.3)),
        "rollback_threshold": float(run_config.get("rollback_threshold", 0.80)),
        "quantization_bits": int(run_config.get("quantization_bits", 32)),
        "krum_f": int(run_config.get("krum_f", 1)),
        "trimmed_mean_beta": float(run_config.get("trimmed_mean_beta", 0.1)),
        "flame_target_frac": float(run_config.get("flame_target_frac", 0.5)),
        "robustness_eval_attack": str(run_config.get("robustness_eval_attack", "pgd")),
        "clip_min": float(run_config.get("clip_min", 0.0)),
        "clip_max": float(run_config.get("clip_max", 1.0)),
        "simulate_global_leakage": bool(run_config.get("simulate_global_leakage", False)),
    }


def server_fn(context: Context) -> ServerAppComponents:
    """
    Spins up the core server process parsing configurations and arming the selected federation strategy.
    """
    run_config = context.run_config
    
    run_metadata = _build_run_metadata(run_config)

    tier = str(run_config.get("tier", "unknown"))
    raw_fog_id = str(run_config.get("fog_id", "cloud"))
    fog_num = int(''.join(filter(str.isdigit, raw_fog_id))) if any(c.isdigit() for c in raw_fog_id) else 0

    log_prefix = "[CLOUD SERVER]" if tier == "cloud" else f"[FOG {fog_num} SERVER]"
    logger = setup_logger(log_prefix)
    
    logger.info("Starting up... Expecting clients.")

    SUPPORTED_STRATEGIES = ["zta", "ztafl", "fedavg", "fedprox", "krum", "trimmed_mean", "flame", "fltrust"]  
    if run_metadata["strategy"].lower() not in SUPPORTED_STRATEGIES:
        logger.error(f"{log_prefix} ❌ Unknown strategy '{run_metadata['strategy']}'. Defaulting to 'zta'.")
        run_metadata["strategy"] = "zta"
        
    logger.info(f"{log_prefix} ⚔️ ACTIVE AGGREGATION STRATEGY: {run_metadata['strategy'].upper()}")

    # Initializes overarching network routing and IO parameters
    dataset_path = str(run_config.get("dataset_path", "data/edge_iiotset/raw/network_traffic_samples.csv"))
    broker_ip = str(run_config.get("broker_ip", "127.0.0.1"))
    fog_ipc_base = int(run_config.get("fog_ipc_base", 10000))
    socket_timeout = float(run_config.get("socket_timeout", 600.0))

    num_classes = DATASET_METADATA[run_metadata["dataset"]]["classes"]
    n_features = DATASET_METADATA[run_metadata["dataset"]]["features"]
    val_data = None

    # Extract dataset variables
    random_seed = int(run_config.get("random_seed", 42))
    test_split=float(run_config.get("test_split", 0.30))
    val_split=float(run_config.get("val_split", 0.50))
    
    # Prepares local validation arrays strictly on the Fog tier for SHAP calculations
    if tier == "fog":
        try:
            logger.debug(f"[CONFIG USAGE] get_dataset | dataset: {run_metadata['dataset']}, dataset_path: {dataset_path}, num_classes: {num_classes}, dataset_fraction: {run_metadata['dataset_fraction']}, shap_val_samples: {run_metadata['shap_val_samples']}")            
            
            simulate_global_leakage = run_config.get("simulate_global_leakage", False)

            # Safely unpack variable-length returns and apply Server scaling
            dataset_returns = get_dataset(run_metadata["dataset"], dataset_path, num_classes, random_seed, simulate_global_leakage, False, "val", test_split, val_split)
            
            X_full = dataset_returns[0]
            y_full = dataset_returns[1]
            
            # If the loader returned 5 items, we are in Secure Mode. The Server MUST scale its evaluation data!
            if len(dataset_returns) > 3:
                server_scaler = dataset_returns[3]
                server_pca = dataset_returns[4]
                
                X_full_np = X_full.numpy() if isinstance(X_full, torch.Tensor) else X_full
                X_full_np = server_scaler.transform(X_full_np)
                X_full_np = server_pca.transform(X_full_np)
                X_full = torch.tensor(X_full_np, dtype=torch.float32)
            
            # Shuffle first to guarantee class distribution for SHAP baseline
            generator = torch.Generator().manual_seed(random_seed)
            indices = torch.randperm(len(X_full), generator=generator)
            X_full = X_full[indices]
            y_full = y_full[indices]

            if run_metadata["dataset_fraction"] < 1.0:
                subset_size = max(run_metadata["shap_val_samples"], int(len(X_full) * run_metadata["dataset_fraction"]))
                X_full = X_full[:subset_size]
                y_full = y_full[:subset_size]

            val_data = (X_full[:run_metadata["shap_val_samples"]], y_full[:run_metadata["shap_val_samples"]])
            n_features = X_full.shape[1]
        except FileNotFoundError:
            logger.warning(f"Dataset not found at {dataset_path}. SHAP checks will be bypassed.")
    
    logger.debug(f"[CONFIG USAGE] get_evaluate_fn | dataset: {run_metadata['dataset']}, dataset_path: {dataset_path}, num_classes: {num_classes}, n_features: {n_features}, random_seed: {run_metadata['random_seed']}")
    evaluate_fn = get_evaluate_fn(
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

    logger.info(f"[CONFIG USAGE] Strategy | num_classes: {num_classes}, shap_explain_count: {run_metadata['shap_explain_count']}, tier: {tier}, fog_num: {fog_num}, n_features: {n_features}, shap_threshold: {run_metadata['shap_threshold']}, min_clients: {run_metadata['min_clients']}, strategy: {run_metadata['strategy']}, model_architecture: {run_metadata['model_architecture']}, rollback_threshold: {run_metadata['rollback_threshold']}")
    strategy = Strategy(
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
        strategy=run_metadata["strategy"],
        evaluate_fn=evaluate_fn,
        run_metadata=run_metadata,
        model_architecture=run_metadata["model_architecture"]
    )

    config = ServerConfig(num_rounds=run_metadata["num_rounds"])
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)