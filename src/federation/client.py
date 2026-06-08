import os
import ast
import copy
import time
import socket
import random
import traceback
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from src.security.adversarial import local_train_byzantine, local_train_honest
from src.security.backdoor import poison_partition
from src.utils.data_loader import get_dataset, DATASET_METADATA, non_iid_partition
from src.utils.logger_setup import setup_logger
from src.network.ipc import send_msg, recv_msg
from src.models.factory import get_model
from src.utils.compression import compress_weights, decompress_weights

# Limits thread usage to prevent OpenMP CPU deadlocks during heavy Server evaluation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

GLOBAL_DATA_CACHE = {}

class Client(NumPyClient):
    """
    Executes the ground-level hustle for the federation.
    
    Serves as the primary operational node that runs local epochs on the raw telemetry,
    injects adversarial static when acting as a rogue agent, and fires the final quantized
    gradient payloads back up the chain.
    """
    
    def __init__(self, logger, node_type: str, log_prefix: str, fog_num: int, model=None, train_loader=None, device="cpu", dataset_metadata=None, train_config=None):
        self.logger = logger
        self.node_type = node_type
        self.log_prefix = log_prefix
        self.fog_num = fog_num
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_metadata = dataset_metadata or {}
        self.train_config = train_config or {}
        
        self.broker_ip = self.train_config.get("broker_ip", "127.0.0.1")
        self.socket_timeout = self.train_config.get("socket_timeout", 600.0)
        
        if self.node_type == "fog_client":
            fog_ipc_base = int(self.train_config.get("fog_ipc_base", 10000))
            self.ipc_port = fog_ipc_base + self.fog_num

    def get_parameters(self, config: dict) -> list:
        """
        Extracts the internal architectural weights and squeezes them through a
        quantization protocol to dramatically reduce network payload overhead.
        """
        if self.model is None:
            raise RuntimeError(f"{self.log_prefix} Model is uninitialized.")
        
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        bits = int(self.train_config.get("quantization_bits", 32))
        return compress_weights(weights, bits)

    def set_parameters(self, parameters: list):
        """
        Catches the outbound payload, decompresses the tensors, and securely
        snaps the parameters straight back into the local PyTorch architecture.
        """
        if self.model is not None and parameters:
            bits = int(self.train_config.get("quantization_bits", 32))
            decompressed_params = decompress_weights(parameters, bits)
            params_dict = zip(self.model.state_dict().keys(), decompressed_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list, config: dict):
        """
        Triggers the primary local processing loops. Maps directly to the assigned
        hardware type—bridging TCP sockets for intermediate fog nodes or grinding out
        the backpropagation for standard edge devices.
        """
        try:
            current_round = config.get("server_round", 0)
            active_strategy = config.get("active_strategy", "fedavg")
            
            if self.node_type == "fog_client":
                return self._execute_fog_bridge(current_round)
            elif self.node_type == "edge":
                return self._execute_edge_training(parameters, current_round, active_strategy)
                
        except Exception as e:
            self.logger.error(f"{self.log_prefix} CRITICAL SILENT CRASH: {e}\n{traceback.format_exc()}", extra={"round": 0})
            return [], 0, {"status": "crashed"}

    def evaluate(self, parameters: list, config: dict):
        """Forces localized evaluation returns into a neutral state to preserve computation bounds."""
        return 0.0, 1, {"accuracy": 1.0}

    def _execute_fog_bridge(self, current_round: int):
        """
        Punches through the network layers connecting the split fog tier architecture.
        Listens for the signal to sync the models and hauls the data across the subnets.
        """
        self.logger.info(f"{self.log_prefix} [IPC BRIDGE] Connecting to Fog Server on port {self.ipc_port}...", extra={"round": current_round})
        
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.socket_timeout)
            try:
                target_host = os.getenv("FOG_SERVER_HOST", "127.0.0.1")
                sock.connect((target_host, self.ipc_port))
                break
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                sock.close()
                time.sleep(0.5)

        self.logger.info(f"{self.log_prefix} [IPC BRIDGE] Connected! Sending START signal.", extra={"round": current_round})
        send_msg(sock, {"cmd": "START", "round": current_round})
        
        try:
            aggregated_ndarrays = recv_msg(sock)
        except socket.timeout:
            self.logger.error(f"{self.log_prefix} [IPC BRIDGE] Timeout waiting for weights!", extra={"round": current_round})
            raise
        finally:
            sock.close()
        
        if not aggregated_ndarrays:
            return [], 0, {"status": "neutralized_attack"}
        
        return aggregated_ndarrays, 1, {"node_name": self.log_prefix}

    def _apply_static_adversarial_split(self, active_loader: DataLoader, role: str) -> DataLoader:
        """
        Strictly implements the paper's 70/30 static clean/adversarial split.
        Subsets the arrays and forces the poison chunk directly
        through continuous optimization bypass attacks.
        """
        adv_ratio = float(self.train_config.get("adv_ratio", 0.3))
        if adv_ratio <= 0.0:
            return active_loader

        self.logger.info(f"{self.log_prefix} Applying static {adv_ratio*100}% adversarial split for {role.upper()}...")
        
        # Unpack the dataset
        X_all, y_all = [], []
        for X, y in active_loader:
            X_all.append(X)
            y_all.append(y)
        X_all = torch.cat(X_all).to(self.device)
        y_all = torch.cat(y_all).to(self.device)

        # Perform the 70/30 split
        split_idx = int(len(X_all) * (1 - adv_ratio))
        X_clean, y_clean = X_all[:split_idx], y_all[:split_idx]
        X_to_poison, y_to_poison = X_all[split_idx:], y_all[split_idx:]

        self.model.eval()
        eps = float(self.train_config.get("eps", 0.1))
        alpha = float(self.train_config.get("alpha", 0.01))
        clip_min = float(self.train_config.get("clip_min", 0.0))
        clip_max = float(self.train_config.get("clip_max", 1.0))
        
        X_adv_list = []
        batch_size = active_loader.batch_size

        for start in range(0, X_to_poison.size(0), batch_size):
            end = start + batch_size
            X_chunk = X_to_poison[start:end]
            y_chunk = y_to_poison[start:end]
            
            if X_chunk.size(0) < 2:
                X_adv_list.append(X_chunk.cpu())
                continue
                
            if role == "pgd":
                from src.security.adversarial import pgd_attack
                chunk_adv = pgd_attack(
                    model=self.model, x=X_chunk, y=y_chunk, 
                    eps=eps, alpha=alpha,
                    clip_min=clip_min, clip_max=clip_max
                )
            else: 
                from src.security.adversarial import fgsm_attack
                chunk_adv = fgsm_attack(
                    model=self.model, x=X_chunk, y=y_chunk, 
                    alpha=eps, clip_min=clip_min, clip_max=clip_max
                )
            X_adv_list.append(chunk_adv.cpu())

        X_adv = torch.cat(X_adv_list)

        # Recombine into a new static DataLoader
        X_combined = torch.cat([X_clean, X_adv.detach()]).cpu()
        y_combined = torch.cat([y_clean, y_to_poison]).cpu()
        
        return DataLoader(TensorDataset(X_combined, y_combined), batch_size=active_loader.batch_size, shuffle=True)

    def _execute_edge_training(self, parameters: list, current_round: int, active_strategy: str):
        """
        Executes the core training loop for standard edge devices.
        Manages backpropagation, initiates static data splits if assigned a hostile role,
        and calculates loss profiles sequentially across defined optimization epochs.
        """
        self.set_parameters(parameters)
        
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_model.to(self.device)
        
        lr = self.train_config.get("learning_rate", 0.001)
        role = self.train_config.get("role", "benign")
        epochs = self.train_config.get("local_epochs", 1)
        
        active_loader = self.train_loader
        if role in ["pgd", "fgsm", "benign"] and float(self.train_config.get("adv_ratio", 0.0)) > 0:
            active_loader = self._apply_static_adversarial_split(active_loader, role)
            
        loss = 0.0

        self.logger.debug(f"[CONFIG USAGE] _execute_edge_training | learning_rate: {lr}, local_epochs: {epochs}")

        for epoch in range(epochs):
            if active_strategy == "fedprox" and role not in ["label_flip", "backdoor", "gradient_manip"]:
                loss = self._train_standard_or_poison("fedprox", lr, current_round, active_loader, active_strategy, global_model)
            elif role in ["backdoor", "label_flip", "gradient_manip"]:
                loss = self._train_standard_or_poison(role, lr, current_round, active_loader, active_strategy, global_model)
            else:
                loss = self._train_standard_or_poison(role, lr, current_round, active_loader, active_strategy, global_model)
                
            self.logger.info(f"{self.log_prefix} Epoch {epoch + 1}/{epochs} complete. Loss: {loss:.4f}", extra={"round": current_round})
        
        metadata = {
            "node_name": self.log_prefix,
            "loss": loss,
            **self.dataset_metadata 
        }
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), metadata

    def _train_standard_or_poison(self, role: str, lr: float, current_round: int, active_loader: DataLoader, active_strategy: str, global_model: torch.nn.Module):
        """
        Computes the gradients based strictly on assigned logic tracks. Executes honest descents
        or completely hijacks the parameters via artificial scaling, label flipping, or
        SHAP-aware optimization maneuvers depending on the active threat profile.
        """
        clip_norm = float(self.train_config.get("clip_norm", 1.0))
        clip_min = float(self.train_config.get("clip_min", 0.0))
        clip_max = float(self.train_config.get("clip_max", 1.0))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.logger.debug(f"[CONFIG USAGE] _train_standard_or_poison | clip_norm: {clip_norm}")
        
        if active_strategy == "fedprox" and role not in ["label_flip", "backdoor", "gradient_manip"]:
            from src.federation.aggregation import fedprox_update
            fedprox_mu = float(self.train_config.get("fedprox_mu", 0.01))
            return fedprox_update(
                model=self.model, global_model=global_model, loader=active_loader,
                optimizer=optimizer, mu=fedprox_mu, device=self.device
            )
            
        elif role in ["backdoor", "label_flip", "gradient_manip", "shap_aware"]:
            
            if role == "shap_aware":
                from src.security.adversarial import local_train_shap_aware
                shap_tau = float(self.train_config.get("shap_tau", 0.15))
                shap_aware_base_attack = self.train_config.get("shap_aware_base_attack", "label_flip")

                self.logger.debug(f"[CONFIG USAGE] local_train_shap_aware | shap_tau: {shap_tau}, shap_aware_base_attack: {shap_aware_base_attack}")

                return local_train_shap_aware(
                    model=self.model, global_model=global_model, loader=active_loader, attack=shap_aware_base_attack,
                    n_classes=self.train_config.get("num_classes", 15), 
                    shap_threshold=shap_tau, device=self.device,
                    lr=lr, epochs=1, clip_norm=clip_norm
                )
            
            attack_type = "gradient_manipulation" if role == "gradient_manip" else role
            alpha_scale = float(self.train_config.get("alpha", 5.0))
            num_classes = self.train_config.get("num_classes", 15)
            return local_train_byzantine(
                model=self.model, loader=active_loader, attack=attack_type,
                n_classes=num_classes, scale=alpha_scale, device=self.device, lr=lr,
                epochs=1, clip_norm=clip_norm
            )
            
        else:
            return local_train_honest(
                model=self.model, loader=active_loader, device=self.device, lr=lr,
                epochs=1, clip_norm=clip_norm
            )


def _build_fog_client(run_config: dict, node_config: dict):
    """
    Spins up the fog client structures binding the intermediate agents to their assigned TCP communication boundaries.
    """
    raw_fog_val = str(node_config.get("fog_id", "0"))
    fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
    node_type = "fog_client" 
    log_prefix = f"[FOG {fog_num} CLIENT]"

    train_config = {
        "broker_ip": str(run_config.get("broker_ip", "127.0.0.1")),
        "socket_timeout": float(run_config.get("socket_timeout", 600.0)),
        "fog_ipc_base": int(run_config.get("fog_ipc_base", 10000))
    }
    
    logger = setup_logger(log_prefix)
    return logger, node_type, log_prefix, fog_num, train_config

def _assign_edge_roles(run_config: dict, total_edges: int, global_index: int, master_seed: int, logger):
    """
    Computes strict assignments distributing adversarial identities across the edge grid
    based entirely on fixed mathematical ratios drawn from the primary configuration parameters.
    """
    pgd_ratio = float(run_config.get("pgd_ratio", 0.0))
    fgsm_ratio = float(run_config.get("fgsm_ratio", 0.0))
    backdoor_ratio = float(run_config.get("backdoor_ratio", 0.0))
    label_flip_ratio = float(run_config.get("label_flip_ratio", 0.0))
    grad_manip_ratio = float(run_config.get("grad_manip_ratio", 0.0))
    shap_aware_ratio = float(run_config.get("shap_aware_ratio", 0.0))

    num_pgd = round(total_edges * pgd_ratio)
    num_fgsm = round(total_edges * fgsm_ratio)
    num_backdoor = round(total_edges * backdoor_ratio)
    num_label_flip = round(total_edges * label_flip_ratio)
    num_grad_manip = round(total_edges * grad_manip_ratio)
    num_shap_aware = round(total_edges * shap_aware_ratio)

    logger.debug(f"[CONFIG USAGE] _assign_edge_roles | pgd_ratio: {pgd_ratio}, fgsm_ratio: {fgsm_ratio}, backdoor_ratio: {backdoor_ratio}, label_flip_ratio: {label_flip_ratio}, grad_manip_ratio: {grad_manip_ratio}")
    
    total_attackers = num_pgd + num_fgsm + num_backdoor + num_label_flip + num_grad_manip + num_shap_aware
    num_benign = max(0, total_edges - total_attackers)
    
    role_list = ["pgd"] * num_pgd + ["fgsm"] * num_fgsm + ["backdoor"] * num_backdoor
    role_list += ["label_flip"] * num_label_flip + ["gradient_manip"] * num_grad_manip
    role_list += ["shap_aware"] * num_shap_aware
    role_list += ["benign"] * num_benign
    
    random.Random(master_seed).shuffle(role_list) 
    role = role_list[global_index % total_edges] if len(role_list) > 0 else "benign"
    return role

def client_fn(context: Context):
    """
    Generates the local execution arrays matching the provided execution parameters.
    Handles topology routing and hard-binds designated operational states.
    """
    run_config = context.run_config
    node_config = context.node_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader = None
    dataset_metadata = None
    train_config = None

    active_strategy = str(run_config.get("strategy", "zta"))
    dataset_name = str(run_config.get("dataset", "edge_iiotset")).lower()
    
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset requested: {dataset_name}")
        
    dataset_path = str(run_config.get("dataset_path", "data/edge_iiotset/raw/network_traffic_samples.csv"))
    dataset_fraction = float(run_config.get("dataset_fraction", 1.0))
    num_classes = DATASET_METADATA[dataset_name]["classes"]
    n_features = DATASET_METADATA[dataset_name]["features"]
    model_arch = str(run_config.get("model_architecture", "cnnlstm"))
    random_seed = int(run_config.get("random_seed", 42))
    test_split=float(run_config.get("test_split", 0.30))
    val_split=float(run_config.get("val_split", 0.50))
    
    if "partition-id" in node_config:
        partition_id = int(node_config["partition-id"]) 
        internal_id = partition_id - 1 
        
        raw_fog_val = str(node_config.get("fog_num", "0"))
        fog_num = int(''.join(filter(str.isdigit, raw_fog_val))) if any(c.isdigit() for c in raw_fog_val) else 0
        
        node_type = "edge"
        log_prefix = f"[EDGE {fog_num}_{partition_id}]" 
        logger = setup_logger(log_prefix)

        try: 
            custom_top_str = str(run_config.get("custom_fog_topology", "[]"))
            custom_topology = ast.literal_eval(custom_top_str) if custom_top_str.strip() else []
            num_fogs = int(run_config.get("num_fogs", 2))
            uniform_edges = int(run_config.get("uniform_edges_per_fog", 2))
            master_seed = int(run_config.get("random_seed", 42))

            if custom_topology and len(custom_topology) > 0:
                topology = custom_topology
            else:
                topology = [uniform_edges] * num_fogs

            total_edges = sum(topology) 
            edges_before_me = sum(topology[:fog_num-1]) if fog_num > 1 else 0
            global_index = edges_before_me + internal_id 

            role = _assign_edge_roles(run_config, total_edges, global_index, master_seed, logger)
                
            train_config = {
                "role": role,
                "learning_rate": float(run_config.get("learning_rate", 0.001)),
                "quantization_bits": int(run_config.get("quantization_bits", 32)),
                "broker_ip": str(run_config.get("broker_ip", "127.0.0.1")),
                "socket_timeout": float(run_config.get("socket_timeout", 600.0)),
                "fog_ipc_base": int(run_config.get("fog_ipc_base", 10000)),
                "n_features": n_features,
                "num_classes": num_classes,
                "local_epochs": int(run_config.get("local_epochs", 1)),
                "clip_min": float(run_config.get("clip_min", 0.0)),
                "clip_max": float(run_config.get("clip_max", 1.0)),
                "clip_norm": float(run_config.get("clip_norm", 1.0)),
                "fedprox_mu": float(run_config.get("fedprox_mu", 0.01)),
                "shap_tau": float(run_config.get("shap_tau", 0.15)),
                "shap_aware_base_attack": str(run_config.get("shap_aware_base_attack", "label_flip")),
            }
            
            if role in ["pgd", "fgsm"]:
                train_config["adv_ratio"] = float(run_config.get(f"{role}_adv_ratio", 0.3))
                train_config["eps"] = float(run_config.get(f"{role}_eps", 0.1))
                train_config["alpha"] = float(run_config.get(f"{role}_alpha", 0.01))
                train_config["n_iter"] = int(run_config.get(f"{role}_n_iter", 7))
            elif role == "benign":
                train_config["adv_ratio"] = float(run_config.get("benign_adv_ratio", 0.3))
                train_config["eps"] = float(run_config.get("benign_eps", 0.05))
                train_config["alpha"] = float(run_config.get("benign_alpha", 0.01))
                train_config["n_iter"] = int(run_config.get("benign_n_iter", 3))
            
            if role == "label_flip":
                train_config["p_flip"] = float(run_config.get("p_flip", 1.0))
            if role == "gradient_manip":
                train_config["alpha"] = float(run_config.get("gradient_alpha", 5.0))
            
            global GLOBAL_DATA_CACHE
            
            apply_smote = run_config.get("apply_smote", False)
            simulate_leakage = run_config.get("simulate_global_leakage", False)
            n_classes_per = int(run_config.get("n_classes_per", 3))

            cache_key = f"train_{dataset_name}_{dataset_fraction}_{simulate_leakage}_{apply_smote}"
            
            logger.debug(f"[CONFIG USAGE] get_dataset | simulate_global_leakage: {simulate_leakage}, apply_smote: {apply_smote}, n_classes_per: {n_classes_per}")
            
            if cache_key not in GLOBAL_DATA_CACHE:
                
                if simulate_leakage:
                    X_full, y_full, n_classes_eval = get_dataset(dataset_name, dataset_path, num_classes, random_seed, simulate_global_leakage=True, apply_smote=apply_smote, split="train", test_split=test_split, val_split=val_split)
                    server_scaler, server_pca = None, None
                else:
                    X_full, y_full, n_classes_eval, server_scaler, server_pca = get_dataset(dataset_name, dataset_path, num_classes, random_seed, simulate_global_leakage=False, apply_smote=apply_smote, split="train", test_split=test_split, val_split=val_split)
                    
                generator = torch.Generator().manual_seed(random_seed)
                indices = torch.randperm(len(X_full), generator=generator)
                X_full = X_full[indices]
                y_full = y_full[indices]

                if dataset_fraction < 1.0:
                    subset_size = int(len(X_full) * dataset_fraction)
                    X_full = X_full[:subset_size]
                    y_full = y_full[:subset_size]
                    
                GLOBAL_DATA_CACHE[cache_key] = (X_full, y_full, n_classes_eval, server_scaler, server_pca)
            else:
                X_full, y_full, n_classes_eval, server_scaler, server_pca = GLOBAL_DATA_CACHE[cache_key]

            power_law_a = float(run_config.get("power_law_a", 0.4))
            partitions = non_iid_partition(X=X_full, y=y_full, n_agents=total_edges, n_classes_per=n_classes_per, power_law_a=power_law_a, seed=master_seed)
            X_part, y_part = partitions[global_index % len(partitions)]
            
            if role == "backdoor":
                poison_fraction = float(run_config.get("backdoor_poison_fraction", 0.5))
                target_class = int(run_config.get("backdoor_target_class", 0))
                trigger_value = float(run_config.get("backdoor_trigger_value", 1.5))
                raw_features = run_config.get("backdoor_trigger_features", "[-3, -2, -1]")
                
                trigger_features = ast.literal_eval(raw_features) if isinstance(raw_features, str) else raw_features
                    
                X_part, y_part = poison_partition(
                    X=X_part, y=y_part, poison_fraction=poison_fraction, target_class=target_class,
                    trigger_features=tuple(trigger_features), trigger_value=trigger_value, seed=(master_seed + global_index) 
                )
            
            if not simulate_leakage:
                X_part_np = X_part.numpy() if isinstance(X_part, torch.Tensor) else X_part
                X_part_np = server_scaler.transform(X_part_np)
                X_part_np = server_pca.transform(X_part_np)
                X_part = torch.tensor(X_part_np, dtype=torch.float32)
            else:
                if not isinstance(X_part, torch.Tensor):
                    X_part = torch.tensor(X_part, dtype=torch.float32)

            dataset = TensorDataset(X_part, y_part)
            batch_size = int(run_config.get("batch_size", 32))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            classes_present, counts = torch.unique(y_part, return_counts=True)
            dataset_metadata = {
                "dataset_name": dataset_name,
                "role": role,
                "total_samples": len(y_part),
                "unique_classes": len(classes_present),
                "distribution": str(dict(zip(classes_present.tolist(), counts.tolist()))) 
            }
            logger.info(f"STATIC ROLE: {role.upper()} | Global Index: {global_index}/{total_edges-1} | Network Topology: {topology}")
            
        except Exception as e:
            logger.error(f"FATAL CRASH DURING BOOT: {e}\n{traceback.format_exc()}")
            raise e
        
    else:
        logger, node_type, log_prefix, fog_num, train_config = _build_fog_client(run_config, node_config)

    model = get_model(model_arch, n_features, num_classes)
    
    return Client(
        logger=logger, node_type=node_type, log_prefix=log_prefix, fog_num=fog_num, 
        model=model, train_loader=train_loader, device=device, 
        dataset_metadata=dataset_metadata, train_config=train_config
    ).to_client()

app = ClientApp(client_fn=client_fn)