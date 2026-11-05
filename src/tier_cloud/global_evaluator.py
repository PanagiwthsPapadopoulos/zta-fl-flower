import logging
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.utils.metrics import accuracy, macro_f1, roc_auc_multiclass
from src.shared.data.data_loader import get_dataset
from src.shared.models.factory import get_model
from src.shared.network.model_payload import load_model_arrays


class GlobalEvaluator:
    """Handles centralized evaluation of the global model.
    Strictly crunches core standard metrics (Loss, Accuracy, Macro-F1) 
    and returns them to the aggregation layer. Does NOT handle disk I/O.
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
        """Initializes the Global Evaluator responsible for centralized standard model performance."""
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.n_features = n_features
        self.device = device
        self.random_seed = random_seed
        self.run_metadata = run_metadata
        self.tier = tier
        self.logger = logger
        
        # Initialize the model blueprint
        model_arch = self.run_metadata["model_architecture"]
        self.model = get_model(model_arch, self.n_features, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Load the prepared evaluation tensors and class count.
        simulate_leakage = self.run_metadata["simulate_global_leakage"]
        apply_smote = self.run_metadata["apply_smote"]
        self.test_data = get_dataset(
            dataset_name=self.dataset, 
            dataset_path=self.dataset_path, 
            num_classes=self.num_classes,
            random_seed=self.random_seed,
            simulate_global_leakage=simulate_leakage,
            apply_smote=apply_smote,
            split="test",
            test_split=self.run_metadata["test_split"],
            val_split=self.run_metadata["val_split"]
        )
        
        # get_dataset returns (X_tensor, y_tensor, num_classes).
        # Only the input and label tensors belong in the TensorDataset.
        X_test, y_test, _ = self.test_data
        
        # Create a TensorDataset to make it compatible with DataLoader
        from torch.utils.data import TensorDataset
        dataset_tensor = TensorDataset(X_test, y_test)
        
        self.test_loader = DataLoader(dataset_tensor, batch_size=64, shuffle=False)

    def evaluate(self, server_round: int, parameters: list, config: dict) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Reconstructs the global model and evaluates it against the centralized test set."""
        
        # 1. Decompress and load the weights into the model
        quantization_bits = int(self.run_metadata["quantization_bits"])
        load_model_arrays(self.model, parameters, quantization_bits)
        self.model.eval()
        if len(self.test_loader.dataset) == 0:
            raise ValueError("Global evaluation requires a nonempty test split")

        # 2. Run the evaluation loop
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                if not torch.isfinite(outputs).all():
                    raise ValueError("Nonfinite global evaluation predictions")
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                # Apply softmax to get probability distributions for AUC
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        # 3. Calculate actual metrics
        avg_loss = total_loss / len(self.test_loader.dataset)
        all_preds, all_targets, all_probs = torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_probs)
        acc = accuracy(all_targets, all_preds)
        f1 = macro_f1(all_targets, all_preds, self.num_classes)
        auc = roc_auc_multiclass(all_targets, all_probs, self.num_classes)

        metrics = {
            "accuracy": acc,
            "macro_f1": f1,
            "auc_defined": auc is not None,
        }
        if auc is not None:
            metrics["auc"] = auc

        self.logger.info(
            f"{self.tier.upper()} Eval Round {server_round} | "
            f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
        )

        self.logger.info(
            f"Preds: {all_preds[:10]} | Targets: {all_targets[:10]}"
        )

        # Return cleanly so CloudAggregator can handle the saving
        return avg_loss, metrics
