import os
import sys
import logging
import math
from unittest.mock import patch
import torch
import torch.nn as nn

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.federation.strategies.aggregation import shap_weighted_aggregate
from src.utils.config_loader import load_yaml_configs

# IMPORTANT: Adjust the import path if data_loader.py lives in a different directory (e.g., src.data.data_loader)
from src.data.data_loader import DATASET_METADATA

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("ZTA_SHAP_Test")
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
# Deterministic Testing Utilities
# =====================================================================
class DummyModel(nn.Module):
    """A minimal model with a single parameter to track weight inheritance."""
    def __init__(self, value: float, n_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.n_classes = n_classes

    def forward(self, x):
        batch_size = x.shape[0]
        # Dynamically size the output tensor to match the requested dataset classes from config
        out = torch.zeros((batch_size, self.n_classes))
        out[:, 0] = 10.0 # Strongly predict Class 0 so accuracy evaluates to 100%
        return out

def run_shap_aggregation_diagnostics():
    """
    Validates the SHAP-Weighted Robust Aggregation mechanism natively.
    Proves mathematical adherence to the MAD filter and the composite weighting 
    formula: w_i = s_i * acc_i * sqrt(|D_i|).
    """
    print("\n" + "#"*80)
    print("### STARTING ZTA SHAP-AGGREGATION MATH TEST ###")
    print("#"*80)

    # -----------------------------------------------------------------
    # Configuration Loading & Dynamic Parameters (Strict Adherence)
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    
    # 1. Pull the dataset name directly from the user's YAML config
    dataset_name = run_metadata.get("dataset_name", "edge_iiotset").lower()
    
    if dataset_name not in DATASET_METADATA:
        logger.error(f"❌ FATAL: Dataset '{dataset_name}' defined in YAML is missing from data_loader.py DATASET_METADATA.")
        sys.exit(1)
        
    # 2. Extract structural dimensions safely from the data_loader's source of truth
    N_CLASSES = DATASET_METADATA[dataset_name]["classes"]
    INPUT_DIM = DATASET_METADATA[dataset_name]["features"]
    
    # 3. Pull the explainer count directly from the user's YAML config
    N_EXPLAIN = run_metadata.get("shap_explain_count")
    if N_EXPLAIN is None:
        logger.error("❌ FATAL: 'shap_explain_count' is missing from your YAML configuration files.")
        sys.exit(1)
    
    logger.debug(f"Loaded Config [{dataset_name}] -> Classes: {N_CLASSES}, Features: {INPUT_DIM}, SHAP Explainer: {N_EXPLAIN}")

    # Initialize dynamic tensors that will shape-shift based on your YAML
    ref_model = DummyModel(0.0, n_classes=N_CLASSES) 
    X_val = torch.zeros((100, INPUT_DIM))
    y_val = torch.zeros((100,), dtype=torch.long)

    # =====================================================================
    # TEST 1: The Nominal Case (Weight Scaling)
    # =====================================================================
    print_separator("Nominal Environment: Size & Stability Scaling")
    logger.debug("Simulating 3 honest agents with varying dataset sizes.")
    
    # Dummy weights (10, 20, 30) are mathematical constants used strictly for the assertion checks.
    local_models = [
        DummyModel(10.0, n_classes=N_CLASSES), 
        DummyModel(20.0, n_classes=N_CLASSES), 
        DummyModel(30.0, n_classes=N_CLASSES)
    ]
    dataset_sizes = [100, 400, 100]

    with patch('src.utils.metrics.compute_shap_stability', return_value=0.95):
        agg_model, regional_trust = shap_weighted_aggregate(
            local_models=local_models,
            ref_model=ref_model,
            X_val=X_val,
            y_val=y_val,
            sizes=dataset_sizes,
            n_classes=N_CLASSES,
            n_explain=N_EXPLAIN
        )

    result_val = agg_model.weight.item()
    if math.isclose(result_val, 20.0, rel_tol=1e-5):
        logger.info(f"✅ Weight scaling successful. Aggregated value is exactly {result_val:.1f}.")
    else:
        logger.error(f"❌ Scaling failure. Expected 20.0, got {result_val}")
        sys.exit(1)

    # =====================================================================
    # TEST 2: The Byzantine Rejection (Outlier Filter)
    # =====================================================================
    print_separator("Byzantine Attack: MAD Threshold Rejection")
    logger.debug("Simulating 5 agents: 4 Honest, 1 Poisoned.")

    local_models = [
        DummyModel(10.0, n_classes=N_CLASSES), DummyModel(10.0, n_classes=N_CLASSES), 
        DummyModel(10.0, n_classes=N_CLASSES), DummyModel(10.0, n_classes=N_CLASSES), 
        DummyModel(999.0, n_classes=N_CLASSES) # The Attack
    ]
    sizes = [100, 100, 100, 100, 100]

    def side_effect_shap(model, ref, x, y, **kwargs):
        val = model.weight.item()
        if val == 999.0:
            return 0.3
        return 0.9

    with patch('src.utils.metrics.compute_shap_stability', side_effect=side_effect_shap):
        agg_model, regional_trust = shap_weighted_aggregate(
            local_models=local_models,
            ref_model=ref_model,
            X_val=X_val,
            y_val=y_val,
            sizes=sizes,
            n_classes=N_CLASSES,
            n_explain=N_EXPLAIN
        )

    result_val = agg_model.weight.item()
    if math.isclose(result_val, 10.0, rel_tol=1e-5):
        logger.info(f"✅ Byzantine rejection successful. Malicious weight purged. Global state: {result_val:.1f}.")
    else:
        logger.error(f"❌ Rejection failure. Poison leaked into model. Result: {result_val}")
        sys.exit(1)

    # =====================================================================
    # TEST 3: The Total Collapse (Fallback Routing)
    # =====================================================================
    print_separator("Isolation Edge Case: 1 Agent Topology")
    with patch('src.utils.metrics.compute_shap_stability', return_value=0.5):
        agg_model, regional_trust = shap_weighted_aggregate(
            local_models=[DummyModel(42.0, n_classes=N_CLASSES)],
            ref_model=ref_model,
            X_val=X_val,
            y_val=y_val,
            sizes=[100],
            n_classes=N_CLASSES,
            n_explain=N_EXPLAIN
        )
        
    result_val = agg_model.weight.item()
    if math.isclose(result_val, 42.0, rel_tol=1e-5):
        logger.info(f"✅ Isolation fallback successful. Handled single-agent topology cleanly: {result_val:.1f}.")
    else:
        logger.error(f"❌ Isolation failure. Expected 42.0, got {result_val}")
        sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 ZTA SHAP AGGREGATION MATH COMPLETE. ALL TESTS PASSED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_shap_aggregation_diagnostics()