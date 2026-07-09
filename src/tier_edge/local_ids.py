import copy
import json
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from src.shared.security.adversarial_math import local_train_honest, local_train_byzantine, adversarial_train_epoch, local_train_proximal, local_train_shap_aware
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

    def execute_training(self, parameters: list, current_round: int, config: dict):
        """Executes the core training loop, managing epochs and adversarial logic."""
        self.set_parameters(parameters)
        
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_model.to(self.device)
        
        lr = self.train_config.get("learning_rate", 0.001)
        role = self.train_config.get("role", "benign")
        epochs = self.train_config.get("local_epochs", 1)
        
        active_loader = self.train_loader
        
        self.logger.info(f"{self.log_prefix} Starting Training with learning rate: {lr}, epochs: {epochs}, role: {role}")
                    
        # Start the training for each role
        loss = 0.0
        for epoch in range(epochs):
            loss = self._train_based_on_role(role, lr, current_round, active_loader, global_model)
            self.logger.info(f"{self.log_prefix} Epoch {epoch + 1}/{epochs} complete. Loss: {loss:.4f}", extra={"round": current_round})
        
        metadata = {
            "node_name": self.log_prefix,
            "loss": loss,
            **self.dataset_metadata 
        }

        # Generate TPM token
        from src.tier_edge.tpm_attestation import TPMAttestation
        tpm_engine = TPMAttestation(logger=self.logger)
        nonce = config.get("nonce", f"round_{current_round}_default")
        tpm_token = tpm_engine.generate_attestation_token(nonce=nonce, software_label=self.log_prefix, round_num=current_round)
        metadata["tpm_token_json"] = json.dumps(tpm_token)
            
        return self.get_parameters(), len(self.train_loader.dataset), metadata

    def _train_based_on_role(self, role: str, lr: float, current_round: int, active_loader: DataLoader, global_model: torch.nn.Module):
        """Executes the specific PyTorch optimization loop variant strictly dictated by the node's assigned profile role."""
        clip_norm = float(self.train_config.get("clip_norm", 1.0))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Shap Aware attacker (label_flip or gradient_manipulation)
        if role in ["shap_aware"]:
            shap_threshold = float(self.train_config.get("shap_threshold", 0.15))
            shap_aware_base_attack = self.train_config.get("shap_aware_base_attack", "label_flip")
            num_classes = self.train_config.get("num_classes", 10)
            shap_val_samples = self.train_config.get("shap_val_samples", 100)
            shap_explain_count = self.train_config.get("shap_explain_count", 15)
            alpha_scale = float(self.train_config.get("alpha", 5.0))
            p_flip = self.train_config.get("p_flip", 0.2)
    
            # Check if the shap aware attack is valid
            if shap_aware_base_attack in ["label_flip", "gradient_manipulation"]:
                self.logger.info(f"{self.log_prefix} Performing SHAP AWARE Attack! Attack type: {shap_aware_base_attack}, Number of classes: {num_classes}, Shap Threshold: {shap_threshold}, Shap explain count: {shap_explain_count}, Shap val samples: {shap_val_samples}, Alpha scale for gradient manip: {alpha_scale}, Flip probability: {p_flip}, Learning Rate: {lr}, Clip norm: {clip_norm}", extra={"round": current_round})
                return local_train_shap_aware(
                    model=self.model, global_model=global_model, loader=active_loader, attack=shap_aware_base_attack,
                    n_classes=num_classes, shap_threshold=shap_threshold, p_flip=p_flip, scale=alpha_scale, device=self.device, lr=lr, clip_norm=clip_norm
                )

            # If the the shap aware attack is not valid, execute honest training
            else:
                self.logger.info(f"{self.log_prefix} Invalid SHAP AWARE Attack type! Performing honest training!", extra={"round": current_round})
                return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm)

        # Label flip or Gradient manipulation attacker
        elif role in ["label_flip", "gradient_manip"]:
            alpha_scale = float(self.train_config.get("alpha", 5.0))
            num_classes = self.train_config.get("num_classes", 15)
            p_flip = self.train_config.get("p_flip", 0.2)

            self.logger.info(f"{self.log_prefix} Performing {role} Attack! Number of classes: {num_classes}, Alpha scale: {alpha_scale}, p_flip: {p_flip}, Learning Rate: {lr}, Clip norm: {clip_norm}", extra={"round": current_round})
            return local_train_byzantine(
                model=self.model, loader=active_loader, attack=role,
                n_classes=num_classes, scale=alpha_scale, p_flip=p_flip, device=self.device, lr=lr, clip_norm=clip_norm
            )   

        elif role in ["backdoor"]:
            self.logger.info(f"{self.log_prefix} Backdoor attack detected! Dataset already poisoned, performing honest training!", extra={"round": current_round})
            return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm)
            
        # Benign node
        elif role in ["benign"]:
            adv_ratio = float(self.train_config.get("adv_ratio", 0.0))
            eps = float(self.train_config.get("eps", 0.1))
            alpha = float(self.train_config.get("alpha", 0.01))
            n_iter = int(self.train_config.get("n_iter", 7))
            clip_min = float(self.train_config.get("clip_min", 0.0))
            clip_max = float(self.train_config.get("clip_max", 1.0))
            use_pgd = bool(self.train_config.get("robustness_eval_attack", "pgd") == "pgd")

            # Check ratio of adversarial examples. If <= 0, just execute honest training
            if adv_ratio > 0:
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
            else:
                self.logger.info(f"{self.log_prefix} Could not perform adversarial training! Adversary ratio is set to 0! Executing honest training with learning rate: {lr} and clip_norm: {clip_norm}.", extra={"round": current_round})
                return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm)
        
        # If no valid role is detected, execute honest training
        else:
            self.logger.info(f"{self.log_prefix} No valid role detected! Detected role: {role}. Executing honest training with learning rate: {lr} and clip_norm: {clip_norm}.", extra={"round": current_round})
            return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm)