import copy
import json
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from src.shared.security.adversarial_math import local_train_honest, local_train_byzantine, adversarial_train_epoch
from src.shared.network.compression import compress_weights, decompress_weights

class EdgeTrainer:
    """Encapsulates the PyTorch machine learning loop for standard edge devices.
    Handles decompression, local epochs, adversarial data splits, and quantization.
    """
    def __init__(self, logger, log_prefix, model, train_loader, device, train_config, dataset_metadata):
        """Initializes the localized Edge Trainer responsible for executing isolated model optimizations."""
        self.logger = logger
        self.log_prefix = log_prefix
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.train_config = train_config
        self.dataset_metadata = dataset_metadata

    def get_parameters(self) -> list:
        """Extracts and compresses weights for network transmission."""
        if self.model is None:
            raise RuntimeError(f"{self.log_prefix} Model is uninitialized.")
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        bits = int(self.train_config.get("quantization_bits", 32))
        return compress_weights(weights, bits)

    def set_parameters(self, parameters: list):
        """Decompresses inbound payloads and loads them into PyTorch."""
        if self.model is not None and parameters:
            bits = int(self.train_config.get("quantization_bits", 32))
            decompressed_params = decompress_weights(parameters, bits)
            params_dict = zip(self.model.state_dict().keys(), decompressed_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def execute_training(self, parameters: list, current_round: int, strategy: str, config: dict):
        """Executes the core training loop, managing epochs and adversarial logic."""
        self.set_parameters(parameters)
        
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_model.to(self.device)
        
        lr = self.train_config.get("learning_rate", 0.001)
        role = self.train_config.get("role", "benign")
        epochs = self.train_config.get("local_epochs", 1)
        
        active_loader = self.train_loader
        
        # OFFENSIVE ACTION: Only active attackers apply static adversarial splits to poison their data.
        if role in ["pgd", "fgsm"] and float(self.train_config.get("adv_ratio", 0.0)) > 0:
            active_loader = self._apply_static_adversarial_split(current_round, active_loader, role)
            
        loss = 0.0
        self.logger.debug(f"[CONFIG USAGE] execute_training | learning_rate: {lr}, local_epochs: {epochs}")
        for epoch in range(epochs):
            if strategy == "fedprox" and role not in ["label_flip", "backdoor", "gradient_manip"]:
                loss = self._train_standard_or_poison("fedprox", lr, current_round, active_loader, strategy, global_model)
            else:
                loss = self._train_standard_or_poison(role, lr, current_round, active_loader, strategy, global_model)
                
            self.logger.info(f"{self.log_prefix} Epoch {epoch + 1}/{epochs} complete. Loss: {loss:.4f}", extra={"round": current_round})
        
        metadata = {
            "node_name": self.log_prefix,
            "loss": loss,
            **self.dataset_metadata 
        }

        if strategy in ["zta", "ztafl"]:
            from src.tier_edge.tpm_attestation import TPMAttestation
            tpm_engine = TPMAttestation(logger=self.logger)
            nonce = config.get("nonce", f"round_{current_round}_default")
            tpm_token = tpm_engine.generate_attestation_token(nonce=nonce, software_label=self.log_prefix, round_num=current_round)
            metadata["tpm_token_json"] = json.dumps(tpm_token)
            
        return self.get_parameters(), len(self.train_loader.dataset), metadata

    def _apply_static_adversarial_split(self, current_round: int, active_loader: DataLoader, role: str) -> DataLoader:
        """Splits the active dataset and iteratively generates adversarial injection examples based on the assigned threat role."""
        adv_ratio = float(self.train_config.get("adv_ratio", 0.3))
        if adv_ratio <= 0.0:
            return active_loader
        
        X_all, y_all = [], []
        for X, y in active_loader:
            X_all.append(X)
            y_all.append(y)
        X_all = torch.cat(X_all).to(self.device)
        y_all = torch.cat(y_all).to(self.device)

        split_idx = int(len(X_all) * (1 - adv_ratio))
        X_clean, y_clean = X_all[:split_idx], y_all[:split_idx]
        X_to_poison, y_to_poison = X_all[split_idx:], y_all[split_idx:]

        self.model.eval()
        eps = float(self.train_config.get("eps", 0.1))
        alpha = float(self.train_config.get("alpha", 0.01))
        clip_min = float(self.train_config.get("clip_min", 0.0))
        clip_max = float(self.train_config.get("clip_max", 1.0))
        pgd_n_iter = int(self.train_config.get("pgd_n_iter", 7))
        
        if role == "pgd":
            self.logger.info(f"{self.log_prefix} Performing PGD Attack! PGD adv ratio: {adv_ratio}, PGD number of iter: {pgd_n_iter}, PGD eps: {eps}, PGD alpha: {alpha}, PGD clip_min: {clip_min}, PGD clip_max: {clip_max}", extra={"round": current_round})
        else: 
            self.logger.info(f"{self.log_prefix} Performing FGSM Attack! FGSM adv ratio: {adv_ratio}, FGSM eps: {eps}, FGSM clip_min: {clip_min}, FGSM clip_max: {clip_max}", extra={"round": current_round})

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
                from src.shared.security.adversarial_math import pgd_attack
                # Use adv_ratio = 1.0 because the split has already happened
                chunk_adv = pgd_attack(model=self.model, x=X_chunk, y=y_chunk, eps=eps, adv_ratio=1.0, alpha=alpha, n_iter=pgd_n_iter, clip_min=clip_min, clip_max=clip_max)
            else: 
                from src.shared.security.adversarial_math import fgsm_attack
                # Use adv_ratio = 1.0 because the split has already happened
                chunk_adv = fgsm_attack(model=self.model, x=X_chunk, y=y_chunk, epsilon=eps, adv_ratio=1.0, clip_min=clip_min, clip_max=clip_max)

            X_adv_list.append(chunk_adv.cpu())

        X_adv = torch.cat(X_adv_list)
        X_combined = torch.cat([X_clean, X_adv.detach()]).cpu()
        y_combined = torch.cat([y_clean, y_to_poison]).cpu()
        
        return DataLoader(TensorDataset(X_combined, y_combined), batch_size=active_loader.batch_size, shuffle=True)

    def _train_standard_or_poison(self, role: str, lr: float, current_round: int, active_loader: DataLoader, strategy: str, global_model: torch.nn.Module):
        """Executes the specific PyTorch optimization loop variant strictly dictated by the node's assigned profile role."""
        clip_norm = float(self.train_config.get("clip_norm", 1.0))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.logger.debug(f"[CONFIG USAGE] _train_standard_or_poison | clip_norm: {clip_norm}")
        
        if strategy == "fedprox" and role not in ["label_flip", "backdoor", "gradient_manip"]:
            from src.shared.utils.metrics import fedprox_update
            fedprox_mu = float(self.train_config.get("fedprox_mu", 0.01))
            return fedprox_update(model=self.model, global_model=global_model, loader=active_loader, optimizer=optimizer, mu=fedprox_mu, device=self.device)
            
        elif role in ["backdoor", "label_flip", "gradient_manip", "shap_aware"]:
            if role == "shap_aware":
                from src.shared.security.adversarial_math import local_train_shap_aware
                shap_threshold = float(self.train_config.get("shap_threshold", 0.15))
                shap_aware_base_attack = self.train_config.get("shap_aware_base_attack", "label_flip")
                num_classes = self.train_config.get("num_classes", 10)
                shap_val_samples = self.train_config.get("shap_val_samples", 100)
                shap_explain_count = self.train_config.get("shap_explain_count", 15)
                alpha_scale = float(self.train_config.get("alpha", 5.0))
                p_flip = self.train_config.get("p_flip", 0.2)
                epochs = 1

                self.logger.info(f"{self.log_prefix} Performing SHAP AWARE Attack! Attack type: {shap_aware_base_attack}, Number of classes: {num_classes}, Shap Threshold: {shap_threshold}, Shap explain count: {shap_explain_count}, Shap val samples: {shap_val_samples}, Alpha scale for gradient manip: {alpha_scale}, Flip probability: {p_flip}, Learning Rate: {lr}, Epochs: {epochs}, Clip norm: {clip_norm}", extra={"round": current_round})
                return local_train_shap_aware(
                    model=self.model, global_model=global_model, loader=active_loader, attack=shap_aware_base_attack,
                    n_classes=num_classes, shap_threshold=shap_threshold, p_flip=p_flip, scale=alpha_scale, device=self.device, lr=lr, epochs=epochs, clip_norm=clip_norm
                )
            
            attack_type = "gradient_manipulation" if role == "gradient_manip" else role
            alpha_scale = float(self.train_config.get("alpha", 5.0))
            num_classes = self.train_config.get("num_classes", 15)
            p_flip = self.train_config.get("p_flip", 0.2)
            epochs = 1

            self.logger.info(f"{self.log_prefix} Performing {attack_type} Attack! Number of classes: {num_classes}, Alpha scale: {alpha_scale}, p_flip: {p_flip}, Learning Rate: {lr}, Epochs: {epochs}, Clip norm: {clip_norm}", extra={"round": current_round})
            return local_train_byzantine(
                model=self.model, loader=active_loader, attack=attack_type,
                n_classes=num_classes, scale=alpha_scale, p_flip=p_flip, device=self.device, lr=lr, epochs=epochs, clip_norm=clip_norm
            )
        else:
            # DEFENSIVE ACTION: Native dynamic adversarial training integration for strictly benign nodes
            adv_ratio = float(self.train_config.get("adv_ratio", 0.0))
            eps = float(self.train_config.get("eps", 0.1))
            alpha = float(self.train_config.get("alpha", 0.01))
            n_iter = int(self.train_config.get("n_iter", 7))
            clip_min = float(self.train_config.get("clip_min", 0.0))
            clip_max = float(self.train_config.get("clip_max", 1.0))
            use_pgd = bool(self.train_config.get("robustness_eval_attack", "pgd") == "pgd" )

            if role == "benign" and adv_ratio > 0:
                self.logger.info(f"{self.log_prefix} Performing Adversarial Training on Benign Node! Adversary Ratio: {adv_ratio}, Epsilon: {eps}, Alpha: {alpha}, Number of Iter: {n_iter}, Use pgd (if false, use fgsm): {use_pgd}, clip_min: {clip_min}, clip_max: {clip_max}, Clip norm: {clip_norm}", extra={"round": current_round})
                return adversarial_train_epoch(
                    model=self.model, 
                    loader=active_loader, 
                    optimizer=optimizer,
                    adv_ratio=adv_ratio,
                    eps=eps,
                    alpha=alpha,
                    n_iter=n_iter,
                    device=self.device,
                    use_pgd=use_pgd,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    clip_norm=clip_norm
                )

            return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, epochs=1, clip_norm=clip_norm)