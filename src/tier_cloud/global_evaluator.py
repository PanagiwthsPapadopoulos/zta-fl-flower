import ast
import logging
from typing import Tuple, Dict, Any
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.shared.security.backdoor_math import compute_backdoor_asr
from src.shared.security.adversarial_math import evaluate_robustness
from src.shared.utils.metrics import accuracy, macro_f1
from src.shared.data.data_loader import get_dataset
from src.shared.models.factory import get_model
from src.shared.network.compression import decompress_weights


class GlobalEvaluator:
    """Handles centralized evaluation of the global model.
    Crunches core metrics (Accuracy, Macro-F1) alongside heavy-hitting 
    security metrics like Backdoor ASR and PGD/FGSM robustness.
    """
    def __init__(
        self, 
        dataset: str, 
        dataset_path: str, 
        num_classes: int, 
        n_features: int, 
        device: str, 
        random_seed: int, 
        run_metadata: dict, 
        tier: str,
        logger: logging.Logger
    ):
        """Initializes the Global Evaluator responsible for centralized model performance and security integrity assessments."""
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.n_features = n_features
        self.device = device
        self.random_seed = random_seed
        self.run_metadata = run_metadata
        self.tier = tier
        self.logger = logger
        
        self._data_cache = {}

    def evaluate(self, server_round: int, parameters: list, config: dict) -> Tuple[float, dict]:
        """The core callback invoked by Flower to assess the global parameters."""
        if "eval_data" not in self._data_cache:
            self.logger.info("[EVALUATOR] Lazy loading centralized test set...")

        simulate_leakage = self.run_metadata.get("simulate_global_leakage", False)
        eval_split = "val" if self.tier == "fog" else "test"
        eval_fraction = float(self.run_metadata.get("dataset_fraction", 1.0))
        apply_smote = bool(self.run_metadata.get("apply_smote", True))

        cache_key = f"eval_{self.dataset}_{eval_split}_{eval_fraction}_{simulate_leakage}"

        self.logger.debug(f"[CONFIG USAGE] get_dataset | simulate_global_leakage: {simulate_leakage}")

        if cache_key not in self._data_cache:
            dataset_returns = get_dataset(
                dataset_name=self.dataset, 
                dataset_path=self.dataset_path, 
                num_classes=self.num_classes, 
                random_seed=self.random_seed,
                simulate_global_leakage=simulate_leakage, 
                apply_smote=apply_smote,
                split=eval_split,
                test_split=float(self.run_metadata.get("test_split", 0.30)),
                val_split=float(self.run_metadata.get("val_split", 0.50))
            )
            
            X_full = dataset_returns[0]
            y_full = dataset_returns[1]
            
            if len(dataset_returns) > 3:
                server_scaler = dataset_returns[3]
                server_pca = dataset_returns[4]
                
                X_full_np = X_full.numpy() if isinstance(X_full, torch.Tensor) else X_full
                X_full_np = server_scaler.transform(X_full_np)
                X_full_np = server_pca.transform(X_full_np)
                X_full = torch.tensor(X_full_np, dtype=torch.float32)

            generator = torch.Generator().manual_seed(self.random_seed)
            indices = torch.randperm(len(X_full), generator=generator)
            X_full = X_full[indices]
            y_full = y_full[indices]

            if eval_fraction < 1.0:
                subset_size = int(len(X_full) * eval_fraction)
                X_full = X_full[:subset_size]
                y_full = y_full[:subset_size]
                    
            self._data_cache[cache_key] = (X_full, y_full)
            self.logger.info(f"[EVALUATOR] Test set ready! Dimensions restricted to: {X_full.shape}")
            
        X_test, y_test = self._data_cache[cache_key]
        batch_size = int(self.run_metadata.get("batch_size", 256))
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.logger.info(f"[EVALUATOR] Started Global Evaluation for Round {server_round} on {len(X_test)} samples!")

        model_architecture = str(self.run_metadata.get("model_architecture", "cnnlstm"))
        model = get_model(model_architecture, self.n_features, self.num_classes)
        
        quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
        self.logger.debug(f"[CONFIG USAGE] decompress_weights | quantization_bits: {quantization_bits}")
        decompressed_params = decompress_weights(parameters, quantization_bits)
        
        params_dict = zip(model.state_dict().keys(), decompressed_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.to(self.device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                
                all_preds.append(logits.argmax(dim=-1).cpu())

        preds_tensor = torch.cat(all_preds)
        avg_loss = total_loss / len(test_dataset)

        acc = accuracy(y_test.cpu(), preds_tensor)
        f1 = macro_f1(y_test.cpu(), preds_tensor, n_classes=self.num_classes)

        backdoor_target_class = int(self.run_metadata.get("backdoor_target_class", 0))
        backdoor_trigger_value = float(self.run_metadata.get("backdoor_trigger_value", 1.5))
        backdoor_trigger_features_str = self.run_metadata.get("backdoor_trigger_features", "[-3, -2, -1]")
        
        if isinstance(backdoor_trigger_features_str, str):
            try:
                backdoor_trigger_features = ast.literal_eval(backdoor_trigger_features_str)
            except Exception:
                backdoor_trigger_features = [-3, -2, -1]
        else:
            backdoor_trigger_features = backdoor_trigger_features_str

        self.logger.debug(f"[CONFIG USAGE] compute_backdoor_asr | backdoor_target_class: {backdoor_target_class}, backdoor_trigger_features: {backdoor_trigger_features_str}, backdoor_trigger_value: {backdoor_trigger_value}, batch_size: {batch_size}")
        asr = compute_backdoor_asr(
            model=model, X_test=X_test, y_test=y_test,
            target_class=backdoor_target_class, trigger_features=tuple(backdoor_trigger_features),
            trigger_value=backdoor_trigger_value, device=self.device, batch_size=batch_size
        )
        
        pgd_eps = float(self.run_metadata.get("pgd_eps", 0.1))
        pgd_alpha = float(self.run_metadata.get("pgd_alpha", 0.01))
        pgd_n_iter = int(self.run_metadata.get("pgd_n_iter", 7))
        fgsm_eps = float(self.run_metadata.get("fgsm_eps", 0.2))
        fgsm_alpha = float(self.run_metadata.get("fgsm_alpha", 0.02))
        clip_min = float(self.run_metadata.get("clip_min", 0.0))
        clip_max = float(self.run_metadata.get("clip_max", 1.0))
        
        self.logger.debug(f"[CONFIG USAGE] evaluate_robustness (pgd) | pgd_eps: {pgd_eps}, pgd_alpha: {pgd_alpha}, pgd_n_iter: {pgd_n_iter}, batch_size: {batch_size}, clip_min: {clip_min}, clip_max: {clip_max}")
        robustness_pgd = evaluate_robustness(
            model=model, X=X_test, y=y_test, attack="pgd",
            eps=pgd_eps, alpha=pgd_alpha, n_iter=pgd_n_iter,
            batch_size=batch_size, device=self.device, clip_min=clip_min, clip_max=clip_max
        )
        
        self.logger.debug(f"[CONFIG USAGE] evaluate_robustness (fgsm) | fgsm_eps: {fgsm_eps}, fgsm_alpha: {fgsm_alpha}, n_iter: 1, batch_size: {batch_size}, clip_min: {clip_min}, clip_max: {clip_max}")
        robustness_fgsm = evaluate_robustness(
            model=model, X=X_test, y=y_test, attack="fgsm",
            eps=fgsm_eps, alpha=fgsm_alpha, n_iter=1,
            batch_size=batch_size, device=self.device, clip_min=clip_min, clip_max=clip_max
        )

        return avg_loss, {
            "accuracy": acc, "macro_f1": f1, "asr": asr, 
            "robustness_pgd": robustness_pgd, "robustness_fgsm": robustness_fgsm
        }