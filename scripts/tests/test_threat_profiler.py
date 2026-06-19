import os
import sys
import logging
from collections import Counter
from unittest.mock import patch

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.utils.config_loader import load_yaml_configs
from security.threat_engine.threat_profiler import assign_edge_roles

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("Threat_Profiler_Test")
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

def run_threat_profiler_diagnostics():
    """
    Validates the role assignment logic directly against the active YAML configuration.
    Proves that the ratios defined in security.yaml mathematically translate to the 
    exact expected number of nodes, and verifies cryptographic seed determinism.
    """
    print("\n" + "#"*80)
    print("### STARTING THREAT PROFILER DISTRIBUTION TEST ###")
    print("#"*80)

    # -----------------------------------------------------------------
    # Configuration Loading (Strict Adherence)
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    
    MASTER_SEED = run_metadata.get("random_seed")
    if MASTER_SEED is None:
        logger.error("❌ FATAL: 'random_seed' is missing from your YAML configurations.")
        sys.exit(1)

    TOTAL_EDGES = 100 
    logger.debug(f"Loaded Config -> Seed: {MASTER_SEED}, Testing Grid Size: {TOTAL_EDGES}")

    # =====================================================================
    # TEST 1: The Live Distribution Math Test
    # =====================================================================
    print_separator("Active Configuration Math & Distribution")
    
    # 1. Dynamically calculate mathematical expectations based strictly on the active YAML
    pgd_ratio = float(run_metadata.get("pgd_ratio", 0.0))
    fgsm_ratio = float(run_metadata.get("fgsm_ratio", 0.0))
    backdoor_ratio = float(run_metadata.get("backdoor_ratio", 0.0))
    label_flip_ratio = float(run_metadata.get("label_flip_ratio", 0.0))
    grad_manip_ratio = float(run_metadata.get("grad_manip_ratio", 0.0))
    shap_aware_ratio = float(run_metadata.get("shap_aware_ratio", 0.0))

    exp_pgd = round(TOTAL_EDGES * pgd_ratio)
    exp_fgsm = round(TOTAL_EDGES * fgsm_ratio)
    exp_backdoor = round(TOTAL_EDGES * backdoor_ratio)
    exp_label_flip = round(TOTAL_EDGES * label_flip_ratio)
    exp_grad_manip = round(TOTAL_EDGES * grad_manip_ratio)
    exp_shap_aware = round(TOTAL_EDGES * shap_aware_ratio)

    total_attackers = exp_pgd + exp_fgsm + exp_backdoor + exp_label_flip + exp_grad_manip + exp_shap_aware
    exp_benign = max(0, TOTAL_EDGES - total_attackers)

    # Map the expected counts to the string outputs of assign_edge_roles
    expected_counts = {
        "pgd": exp_pgd,
        "fgsm": exp_fgsm,
        "backdoor": exp_backdoor,
        "label_flip": exp_label_flip,
        "gradient_manip": exp_grad_manip,
        "shap_aware": exp_shap_aware,
        "benign": exp_benign
    }

    # Filter out 0-counts for cleaner logging of the active configuration
    active_expectations = {k: v for k, v in expected_counts.items() if v > 0}
    logger.debug(f"Mathematical Expectations from YAML: {active_expectations}")

    # 2. Run the actual distribution logic
    live_roles = []
    with patch.object(logger, 'debug'):
        for global_index in range(TOTAL_EDGES):
            role = assign_edge_roles(run_metadata, TOTAL_EDGES, global_index, MASTER_SEED, logger)
            live_roles.append(role)
        
    actual_counts = Counter(live_roles)
    logger.info(f"Actual Assigned Distribution: {dict(actual_counts)}")
    
    # 3. Assert the actual output matches the YAML math exactly
    passed_math = True
    for attack_role, expected_num in expected_counts.items():
        actual_num = actual_counts.get(attack_role, 0)
        if actual_num != expected_num:
            logger.error(f"❌ {attack_role} math failed! Expected {expected_num}, Got {actual_num}")
            passed_math = False
            
    if passed_math:
        logger.info("✅ Live configuration successfully mapped to exact node counts.")
    else:
        sys.exit(1)

    # =====================================================================
    # TEST 2: Deterministic Seed Integrity
    # =====================================================================
    print_separator("Cryptographic Determinism (Seed Locking)")
    logger.debug(f"Verifying that master_seed ({MASTER_SEED}) generates the exact same map twice.")
    
    second_pass_roles = []
    with patch.object(logger, 'debug'):
        for global_index in range(TOTAL_EDGES):
            role = assign_edge_roles(run_metadata, TOTAL_EDGES, global_index, MASTER_SEED, logger)
            second_pass_roles.append(role)
        
    if live_roles == second_pass_roles:
        logger.info("✅ Seed determinism successful. The threat distribution map is locked and reproducible.")
    else:
        logger.error("❌ Determinism failure. The same seed produced different network distributions.")
        sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 THREAT PROFILER TESTS COMPLETE. ALL ACTIVE CONFIGURATIONS VALIDATED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_threat_profiler_diagnostics()