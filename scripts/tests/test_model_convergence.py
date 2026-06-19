import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.utils.config_loader import load_yaml_configs
from src.core.edge_trainer import EdgeTrainer

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("Convergence_Test")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Minimal structural representation of the CNN-LSTM
class MinimalCNNLSTM(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        # Simulated CNN feature extractor
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Simulated LSTM temporal sequence analyzer
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # Reshape for Conv1D: (batch, channels, features)
        x = x.unsqueeze(1) 
        x = torch.relu(self.conv(x))
        # Reshape for LSTM: (batch, sequence_length, features)
        x = x.permute(0, 2, 1) 
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def run_micro_batch_overfit_test():
    """
    Executes a high-speed convergence validation by intentionally overfitting 
    the model on a microscopic dataset. This mathematically proves the architecture 
    is capable of gradient descent and learning without requiring full dataset evaluation.
    """
    print("\n" + "="*80)
    print("🚀 TEST: Rapid Convergence via Micro-Batch Overfitting")
    print("="*80)

    # 1. Configuration matching live architecture
    N_CLASSES = 15
    INPUT_DIM = 40
    
    # 2. Generate a microscopic, fixed dataset (exactly 10 rows)
    torch.manual_seed(42)
    X_micro = torch.randn((10, INPUT_DIM))
    y_micro = torch.randint(0, N_CLASSES, (10,))
    
    # Batch size equals the dataset size so it trains entirely in one step
    micro_loader = DataLoader(TensorDataset(X_micro, y_micro), batch_size=10)
    model = MinimalCNNLSTM(INPUT_DIM, N_CLASSES)

    # 3. Aggressive training configuration to force memorization
    train_config = {
        "quantization_bits": 32,
        "learning_rate": 0.05, # High learning rate to accelerate memorization
        "local_epochs": 50,    # High epoch count to force the loss to zero
        "clip_norm": 5.0,
        "num_classes": N_CLASSES,
        "role": "benign",
        "adv_ratio": 0.0
    }

    trainer = EdgeTrainer(
        logger=logger, log_prefix="[CONVERGENCE_TEST]", model=model, 
        train_loader=micro_loader, device="cpu", 
        train_config=train_config, dataset_metadata={}
    )

    # Extract initial weights to simulate the start of a round
    initial_params = trainer.get_parameters()

    # Execute the aggressive training loop
    logger.info("Executing 50 local epochs on 10 samples to force gradient convergence...")
    _, _, metadata = trainer.execute_training(
        parameters=initial_params, current_round=1, strategy="fedavg", config={}
    )

    # 4. Mathematical Assertion: The loss must approach zero.
    final_loss = metadata.get("loss", 999.0)
    
    if final_loss < 0.1:
        logger.info(f"✅ Convergence verified! The model successfully learned and overfit the micro-batch. Final Loss: {final_loss:.4f}")
    else:
        logger.error(f"❌ Convergence failure. The model failed to learn. Final Loss: {final_loss:.4f}")
        sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 MODEL CONVERGENCE TEST COMPLETE. SYSTEM IS CAPABLE OF LEARNING. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_micro_batch_overfit_test()