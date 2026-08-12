import logging
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.utils.metrics import accuracy, macro_f1
from src.shared.data.data_loader import get_dataset
from src.shared.models.factory import get_model
from src.shared.network.compression import decompress_weights


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

        # Load the evaluation dataset using the updated function signature
        self.test_data = get_dataset(
            dataset_name=self.dataset, 
            dataset_path=self.dataset_path, 
            num_classes=self.num_classes,
            random_seed=self.random_seed,
            split="test"
        )
        
        # Note: get_dataset now returns a tuple of (X_tensor, y_tensor, num_classes)
        # We need to unpack just the tensors for the DataLoader
        X_test, y_test, _ = self.test_data
        
        # Create a TensorDataset to make it compatible with DataLoader
        from torch.utils.data import TensorDataset
        dataset_tensor = TensorDataset(X_test, y_test)
        
        self.test_loader = DataLoader(dataset_tensor, batch_size=64, shuffle=False)

    def evaluate(self, server_round: int, parameters: list, config: dict) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Reconstructs the global model and evaluates it against the centralized test set."""
        
        # 1. Decompress and load the weights into the model
        quantization_bits = int(self.run_metadata["quantization_bits"])
        decompressed_params = decompress_weights(parameters, quantization_bits)

        state_dict = {}
        for k, v in zip(self.model.state_dict().keys(), decompressed_params):
            state_dict[k] = torch.tensor(v)
        
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # 2. Run the evaluation loop
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 3. Calculate actual metrics
        avg_loss = total_loss / len(self.test_loader.dataset)
        acc = accuracy(torch.tensor(all_targets), torch.tensor(all_preds))
        f1 = macro_f1(torch.tensor(all_targets), torch.tensor(all_preds))

        metrics = {
            "accuracy": acc,
            "macro_f1": f1
        }

        self.logger.info(
            f"{self.tier.upper()} Eval Round {server_round} | "
            f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
        )

        self.logger.info(
            f"Preds: {preds[:10]} | Targets: {targets[:10]}"
        )

        # Return cleanly so CloudAggregator can handle the saving
        return avg_loss, metrics