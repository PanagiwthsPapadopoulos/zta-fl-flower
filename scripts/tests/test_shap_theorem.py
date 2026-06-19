import os
import sys
import logging
import math
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.utils.metrics import compute_shap_stability
from src.utils.config_loader import load_yaml_configs
from src.data.data_loader import DATASET_METADATA

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("ZTA_Theorem1_Test")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

def print_separator(title: str):
    print("\n" + "="*80)
    print(f"🚀 TEST: {title}")
    print("="*80)

# =====================================================================
# Differentiable PyTorch Model (Required for real SHAP gradients)
# =====================================================================
class MinimalDifferentiableModel(nn.Module):
    """
    A minimal, fully differentiable model. 
    Unlike DummyModel, this allows Captum/GradientSHAP to trace gradients 
    all the way from the output back to the input features.
    """
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def inject_targeted_poison(model: MinimalDifferentiableModel):
    """
    Simulates a targeted Byzantine attack (e.g., label flipping).
    Reverses the signs of the weights in the final layer to drastically 
    alter the gradient path and semantic behavior of the model, without 
    necessarily changing the raw magnitude of the weights.
    """
    with torch.no_grad():
        model.fc2.weight.data = -model.fc2.weight.data
        model.fc1.weight.data += torch.randn_like(model.fc1.weight.data) * 0.5
    return model

def inject_honest_drift(model: MinimalDifferentiableModel, noise_scale: float = 0.05):
    """
    Simulates normal, benign non-IID drift.
    Adds a small amount of Gaussian noise to the weights.
    """
    with torch.no_grad():
        model.fc2.weight.data += torch.randn_like(model.fc2.weight.data) * noise_scale
        model.fc1.weight.data += torch.randn_like(model.fc1.weight.data) * noise_scale
    return model

def run_theorem_1_diagnostics():
    """
    Validates Theorem 1 mathematically using real feature importance vectors.
    Proves that a Byzantine attacker attempting a semantic attack (label flip) 
    will induce a feature importance shift that pushes their SHAP stability 
    score below the expected mu - 2*sigma threshold.
    """
    print("\n" + "#"*80)
    print("### STARTING THEOREM 1: REAL SHAP COMPUTATION TEST ###")
    print("#"*80)

    # -----------------------------------------------------------------
    # Configuration & Data Setup
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    dataset_name = run_metadata.get("dataset", "edge_iiotset").lower()
    
    N_CLASSES = DATASET_METADATA.get(dataset_name, {}).get("classes", 15)
    INPUT_DIM = DATASET_METADATA.get(dataset_name, {}).get("features", 40)
    N_EXPLAIN = run_metadata.get("shap_explain_count", 100)

    # Generate a fixed, deterministic validation background set
    torch.manual_seed(42)
    X_val = torch.randn((N_EXPLAIN, INPUT_DIM), requires_grad=True)
    y_val = torch.randint(0, N_CLASSES, (N_EXPLAIN,))

    # Initialize the Global Reference Model
    ref_model = MinimalDifferentiableModel(INPUT_DIM, N_CLASSES)
    
    # Initialize 10 Honest Agents with normal data heterogeneity (drift)
    honest_models = []
    for _ in range(10):
        agent_model = MinimalDifferentiableModel(INPUT_DIM, N_CLASSES)
        agent_model.load_state_dict(ref_model.state_dict())
        honest_models.append(inject_honest_drift(agent_model, noise_scale=0.02))

    # Initialize 1 Byzantine Agent executing a severe semantic attack
    byzantine_model = MinimalDifferentiableModel(INPUT_DIM, N_CLASSES)
    byzantine_model.load_state_dict(ref_model.state_dict())
    byzantine_model = inject_targeted_poison(byzantine_model)

    # =====================================================================
    # TEST EXECUTION: Real SHAP Tensor Math
    # =====================================================================
    print_separator("Computing Native GradientSHAP Stability Vectors")
    
    honest_scores = []
    
    # 1. Compute stability for all honest agents
    for i, model in enumerate(honest_models):
        score = compute_shap_stability(
            model=model, 
            ref_model=ref_model, 
            X_val=X_val, 
            y_val=y_val, 
            n_explain=N_EXPLAIN
        )
        honest_scores.append(score)
        logger.debug(f"Honest Agent {i+1} Stability: {score:.4f}")

    # 2. Calculate Statistical Boundaries
    mu_s = np.mean(honest_scores)
    sigma_s = np.std(honest_scores)
    detection_threshold = mu_s - (2 * sigma_s)
    
    logger.info(f"Honest Distribution -> Mean (μ): {mu_s:.4f}, Std (σ): {sigma_s:.4f}")
    logger.info(f"Calculated Theorem 1 Detection Boundary (μ - 2σ): {detection_threshold:.4f}")

    # 3. Compute stability for the Byzantine agent
    byzantine_score = compute_shap_stability(
            model=byzantine_model, 
            ref_model=ref_model, 
            X_val=X_val, 
            y_val=y_val, 
            n_explain=N_EXPLAIN
        )
    logger.info(f"Byzantine Agent Stability Score: {byzantine_score:.4f}")

    # =====================================================================
    # ASSERTIONS (Theorem 1 Mathematical Proof)
    # =====================================================================
    if byzantine_score < detection_threshold:
        logger.info(f"✅ THEOREM 1 VERIFIED: Byzantine score ({byzantine_score:.4f}) successfully fell below the rejection threshold ({detection_threshold:.4f}).")
    else:
        logger.error(f"❌ THEOREM 1 FAILURE: Byzantine score ({byzantine_score:.4f}) evaded the threshold ({detection_threshold:.4f}). Feature importance shift was not detected.")
        sys.exit(1)
        
    print("\n" + "#"*80)
    print("🎉 REAL SHAP COMPUTATION & THEOREM 1 TESTS PASSED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_theorem_1_diagnostics()