import os
import sys
import logging
import torch
import torch.nn as nn
from src.utils.config_loader import load_yaml_configs
import math

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.federation.strategies.zta_strategy import ZTAStrategy

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("ZTA_Rollback_Test")
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
# Deterministic PyTorch Model
# =====================================================================
class DeterministicModel(nn.Module):
    """
    A real PyTorch module designed to bypass training and output exact accuracy 
    percentages. It registers a single parameter matrix that dictates the exact 
    logits returned, allowing us to perfectly control the validation accuracy.
    """
    def __init__(self, num_samples: int = 100):
        super().__init__()
        # 100 samples, 2 classes. Registered as a real parameter so state_dict captures it.
        self.logits = nn.Parameter(torch.zeros(num_samples, 2))

    def forward(self, x):
        return self.logits

def set_model_accuracy(model: DeterministicModel, target_acc_percent: int):
    """
    Modifies the model's weights in-place to guarantee it achieves exactly 
    `target_acc_percent` on an all-zero target dataset.
    """
    # Reset all weights
    model.logits.data.zero_()
    
    # Correct predictions (Class 0 is higher)
    model.logits.data[:target_acc_percent, 0] = 1.0
    model.logits.data[:target_acc_percent, 1] = 0.0
    
    # Incorrect predictions (Class 1 is higher)
    model.logits.data[target_acc_percent:, 0] = 0.0
    model.logits.data[target_acc_percent:, 1] = 1.0


def run_rollback_diagnostics():
    """
    Validates the pure mathematical implementation of the ZTAStrategy Rollback Mechanism.
    Verifies that state dictionaries are correctly cached, thresholds are 
    dynamically calculated from configuration, and poisoned weights are purged.
    """
    print("\n" + "#"*80)
    print("### STARTING ZTA STRATEGY PURE MATH ROLLBACK TEST ###")
    print("#"*80)

    logger.info("Initializing Standard ZTAStrategy for testing...")
    
    X_val = torch.zeros(100, 1)
    y_val = torch.zeros(100, dtype=torch.long)
    val_data = (X_val, y_val)
    
    # -----------------------------------------------------------------
    # Configuration Loading & Dynamic Test Parameters
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    
    # Safely extract the threshold
    current_threshold = run_metadata.get("rollback_threshold", 0.8)
    
    # -----------------------------------------------------------------
    # FULLY DYNAMIC TEST PARAMETERS
    # -----------------------------------------------------------------
    # Round 1: Start at a perfect 100% to set the highest possible ceiling.
    target_1 = 100
    
    # Round 2: Must PASS the sanity check (target_2 >= target_1 * threshold).
    # We calculate the exact mathematical boundary, and set the accuracy safely above it.
    boundary_2 = target_1 * current_threshold
    target_2 = int(boundary_2 + ((100 - boundary_2) / 2)) # Halfway between boundary and 100
    
    # Round 3: Must FAIL the sanity check (target_3 < target_2 * threshold).
    # We calculate the exact boundary and drop it 5% below the trigger line.
    boundary_3 = target_2 * current_threshold
    target_3 = max(0, int(boundary_3 - 5))

    TEST_PARAMS = {
        "round_1_acc": target_1,
        "round_2_acc": target_2,
        "round_3_poison_acc": target_3
    }

    logger.debug(f"Loaded Rollback Threshold: {current_threshold}")
    logger.debug(f"Generated R1: {target_1}%, R2: {target_2}%, R3 (Poison): {target_3}%")

    strategy = ZTAStrategy(
        logger=logger,
        log_prefix="[TEST FOG 1]",
        tier="fog",
        fog_num=1, 
        val_data=val_data,
        run_metadata=run_metadata
    )

    model = DeterministicModel(num_samples=100)

    # =====================================================================
    # TEST 1: The Initial Baseline (Round 1)
    # =====================================================================
    print_separator("Round 1: Establishing the Baseline State")
    target_1 = TEST_PARAMS["round_1_acc"]
    
    logger.debug(f"Simulating Round 1 completion with {target_1}% Accuracy.")
    
    set_model_accuracy(model, target_1)
    strategy._evaluate_rollback_sanity_check(aggregated_model=model, round_display=1)

    if math.isclose(strategy.previous_val_acc, target_1 / 100.0, rel_tol=1e-5) and "logits" in strategy.cached_global_state:
        logger.info("✅ Baseline established correctly.")
    else:
        logger.error(f"❌ Baseline failure. Acc: {strategy.previous_val_acc}, State Keys: {list(strategy.cached_global_state.keys())}")
        sys.exit(1)

    # =====================================================================
    # TEST 2: Nominal System Improvement (Round 2)
    # =====================================================================
    print_separator("Round 2: Nominal Improvement & Threshold Update")
    target_2 = TEST_PARAMS["round_2_acc"]
    logger.debug(f"Simulating Round 2 completion with {target_2}% Accuracy (Valid Update).")
    
    set_model_accuracy(model, target_2)
    strategy._evaluate_rollback_sanity_check(aggregated_model=model, round_display=2)
    
    if math.isclose(strategy.previous_val_acc, target_2 / 100.0, rel_tol=1e-5):
        logger.info("✅ Valid update accepted. Dynamic threshold boundary moved.")
    else:
        logger.error(f"❌ Valid update rejected. Strategy.previous_val_acc is {strategy.previous_val_acc}")
        sys.exit(1)

    # =====================================================================
    # TEST 3: Byzantine Poisoning & Defensive Rollback (Round 3)
    # =====================================================================
    print_separator("Round 3: Critical Poisoning & State Rollback")
    target_3 = TEST_PARAMS["round_3_poison_acc"]
    logger.debug(f"Simulating catastrophic Byzantine aggregation yielding {target_3}% Accuracy.")
    logger.debug(f"Expected Behavior: System must intercept the update, purge parameters, and revert to {target_2}%.")
    
    # Corrupt model using dynamic attack parameter
    set_model_accuracy(model, target_3)

    # Trigger safety check
    strategy._evaluate_rollback_sanity_check(aggregated_model=model, round_display=3)
    
    # Verification: Rollback Success checks against Round 2 state
    reverted_acc = (model(X_val).argmax(dim=-1) == y_val).float().mean().item()
    if math.isclose(reverted_acc, target_2 / 100.0, rel_tol=1e-5):
        logger.info(f"✅ ROLLBACK SUCCESSFUL! Poisoned parameters purged. Memory state reverted to {reverted_acc * 100:.1f}%.")
    else:
        logger.error(f"❌ ROLLBACK FAILED! Model memory state is currently {reverted_acc * 100:.1f}%.")
        sys.exit(1)

    # Cleanup
    if hasattr(strategy, "fog_bridge"):
        strategy.fog_bridge.close()

    print("\n" + "#"*80)
    print("🎉 ZTA PURE MATH ROLLBACK TEST COMPLETE. ALL TESTS PASSED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_rollback_diagnostics()