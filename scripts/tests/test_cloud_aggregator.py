import os
import sys
import logging
import math
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.utils.config_loader import load_yaml_configs
from src.federation.strategies.zta_strategy import ZTAStrategy
from src.data.data_loader import DATASET_METADATA

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("Cloud_Aggregator_Test")
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
    """A minimal model to track exactly how parameters are merged at the Cloud tier."""
    def __init__(self, value: float):
        super().__init__()
        # Use a single, easily traceable parameter
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))

def run_cloud_aggregator_diagnostics():
    """
    Validates the Cloud Server's Global Aggregation math.
    Proves that the equation θ^{t+1} = sum(w_f * θ_f^t) accurately uses 
    the regional trust metrics provided by the Fog nodes.
    """
    print("\n" + "#"*80)
    print("### STARTING CLOUD GLOBAL AGGREGATION TEST ###")
    print("#"*80)

    # -----------------------------------------------------------------
    # Configuration Loading
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    dataset_name = run_metadata.get("dataset", "edge_iiotset").lower()
    
    N_CLASSES = DATASET_METADATA.get(dataset_name, {}).get("classes", 15)
    INPUT_DIM = DATASET_METADATA.get(dataset_name, {}).get("features", 40)

    # Initialize the Cloud Strategy
    strategy = ZTAStrategy(
        logger=logger,
        log_prefix="[TEST CLOUD]",
        tier="cloud",  # Explicitly setting tier to Cloud
        fog_num=0,
        n_features=INPUT_DIM,
        num_classes=N_CLASSES,
        run_metadata=run_metadata
    )

    # Replace the internal complex global model with our traceable DummyModel
    strategy.global_model = DummyModel(0.0)

    # =====================================================================
    # TEST 1: ZTA Global Aggregation (Weighted by Regional Trust)
    # =====================================================================
    print_separator("Global Math Verification: Regional Trust Weighting")
    logger.debug("Simulating the Cloud receiving verified updates from 2 different Fog nodes.")
    
    # Simulating 2 Fog Node Models
    # Fog 1 sends weights valued at 10.0
    # Fog 2 sends weights valued at 50.0
    fog_models = [DummyModel(10.0), DummyModel(50.0)]
    
    # Fog 1 filtered out attacks and had 100 'trust points' worth of data survive
    # Fog 2 had a massive clean region with 300 'trust points' worth of data
    regional_trust_weights = [100.0, 300.0] 
    
    # The actual dataset sizes shouldn't matter for ZTA; the trust metric overrides it.
    raw_sizes = [9999, 9999] 
    
    strategy.strategy = "zta"
    
    # Execute the Cloud's mathematical routing
    agg_model = strategy._apply_aggregation_strategy(
        local_models=fog_models, 
        sizes=raw_sizes, 
        trust_weights=regional_trust_weights
    )
    
    result = agg_model.weight.item()
    
    # Mathematical Expectation: (10.0 * 100/400) + (50.0 * 300/400) = 2.5 + 37.5 = 40.0
    if math.isclose(result, 40.0, rel_tol=1e-5):
        logger.info(f"✅ Cloud ZTA Aggregation mathematically perfect. Weighted global state is {result:.1f}.")
    else:
        logger.error(f"❌ Cloud Math Failure. Expected 40.0, got {result}")
        sys.exit(1)

    # =====================================================================
    # TEST 2: FedAvg Fallback Routing (Weighted by Dataset Size)
    # =====================================================================
    print_separator("Standard Fallback Routing (Size-Proportional)")
    logger.debug("Verifying Cloud correctly switches to raw dataset sizes if configured for FedAvg.")
    
    fog_models_fedavg = [DummyModel(10.0), DummyModel(50.0)]
    raw_sizes_fedavg = [1000, 3000] 
    bogus_trust = [999.0, 999.0] # This should be completely ignored by the FedAvg engine
    
    strategy.strategy = "fedavg"
    
    agg_model_fedavg = strategy._apply_aggregation_strategy(
        local_models=fog_models_fedavg, 
        sizes=raw_sizes_fedavg, 
        trust_weights=bogus_trust
    )
    
    result_fedavg = agg_model_fedavg.weight.item()
    
    # Mathematical Expectation: (10.0 * 1000/4000) + (50.0 * 3000/4000) = 2.5 + 37.5 = 40.0
    if math.isclose(result_fedavg, 40.0, rel_tol=1e-5):
        logger.info(f"✅ Fallback routing successful. Cloud successfully utilized volume scalars. Global state: {result_fedavg:.1f}.")
    else:
        logger.error(f"❌ Fallback Routing Failure. Expected 40.0, got {result_fedavg}")
        sys.exit(1)

    # =====================================================================
    # TEST 3: Pipeline Integration (Metrics Extraction)
    # =====================================================================
    print_separator("Cloud Pipeline Unpacking & Re-Serialization")
    logger.debug("Proving the Cloud strategy successfully unpacks the 'total_regional_trust' metric from incoming Flower payloads.")
    
    # We use MagicMock to simulate the exact payload structure sent by Flower over the network
    class MockFitRes:
        def __init__(self, trust_val, size):
            self.parameters = MagicMock()
            self.num_examples = size
            self.metrics = {"total_regional_trust": trust_val}
            
    mock_results = [
        (MagicMock(), MockFitRes(trust_val=100.0, size=500)),
        (MagicMock(), MockFitRes(trust_val=300.0, size=500))
    ]
    
    # Patch out the heavy ML unpacking so we can trace the pure Python routing
    with patch('src.federation.strategies.zta_strategy.get_model', return_value=DummyModel(0.0)), \
         patch('src.federation.strategies.zta_strategy.parameters_to_ndarrays', return_value=[]), \
         patch('src.federation.strategies.zta_strategy.decompress_weights', side_effect=[[np.array([10.0])], [np.array([50.0])]]):
        
        extracted_models, extracted_sizes, extracted_trust = strategy._extract_models_from_results(mock_results)
        
        if extracted_trust == [100.0, 300.0] and extracted_sizes == [500, 500]:
            logger.info("✅ Pipeline extraction successful. Regional trust metrics correctly parsed from Flower transport objects.")
            logger.info(f"   -> Extracted Trust: {extracted_trust}")
            logger.info(f"   -> Extracted Model Weights: {[m.weight.item() for m in extracted_models]}")
        else:
            logger.error("❌ Pipeline extraction failed to parse metrics.")
            sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 CLOUD AGGREGATOR TESTS COMPLETE. GLOBAL ORCHESTRATION VERIFIED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_cloud_aggregator_diagnostics()