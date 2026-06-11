import os
import sys
import logging

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.security.trust_db import TrustDatabase
from src.security.tpm_core import TPMEngine

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("TrustDB_Hardware_Test")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

PUBLIC_KEY_PATH = "/app/tpm_state/ak.pub"
SOFTWARE_LABEL = "EDGE_1_1"

def print_separator(title: str):
    print("\n" + "="*80)
    print(f"🚀 TEST: {title}")
    print("="*80)

def run_hardware_integration_diagnostics():
    """
    Integration diagnostic suite for the Trust Database and TPM hardware.
    Validates the mathematical boundaries of the state machine, including reward scaling,
    quarantine enforcement, and rehabilitation interruption loops over live hardware context.
    """
    print("\n" + "#"*80)
    print("### STARTING TRUST-DB + HARDWARE TPM INTEGRATION TEST ###")
    print("#"*80)

    # 1. Instantiate the DB Logic
    db = TrustDatabase(logger=logger)
    
    # 2. Boot the REAL Hardware TPM Engine (insecure_mode=False)
    logger.info("Booting physical TPM Engine...")
    tpm = TPMEngine(logger=logger, insecure_mode=False)
    
    # Extract the true hardware ID to register in the DB
    hardware_id = tpm._get_hardware_identity()
    db.register_node(hardware_id)

    # =====================================================================
    # TEST 1: The Golden Path (Real Hardware Generation & Reward)
    # =====================================================================
    print_separator("Initialization & Reward Scaling (Live Hardware)")
    logger.debug("Generating 5 consecutive valid hardware attestations...")
    
    for i in range(1, 6):
        live_nonce = f"golden_nonce_{i}"
        
        # Real hardware generation
        token = tpm.generate_attestation_token(nonce=live_nonce, software_label=SOFTWARE_LABEL)
        
        # Real mathematical verification
        is_valid = tpm.verify_attestation_token(token, expected_nonce=live_nonce, public_key_path=PUBLIC_KEY_PATH)
        
        # Process through DB
        db.process_attestation(hardware_id, is_valid=is_valid)
        
    score = db.get_score(hardware_id)
    if score == 0.8: # 0.7 + (5 * 0.02)
        logger.info("✅ Reward scaling nominal. Hardware tokens accepted. Score: 0.8.")
    else:
        logger.error(f"❌ Reward scaling failure. Score: {score}")
        sys.exit(1)

    # =====================================================================
    # TEST 2: Quarantine Trigger & Persistent Rejection
    # =====================================================================
    print_separator("Quarantine Enforcement & Persistent Isolation")
    logger.debug("Generating a real hardware token, but forging the signature in transit.")
    
    live_nonce = "forgery_nonce_001"
    forged_token = tpm.generate_attestation_token(nonce=live_nonce, software_label=SOFTWARE_LABEL)
    
    # Corrupt the signature bytes
    original_sig = forged_token["signature"]
    corrupted_char = 'B' if original_sig[-1] == 'A' else 'A'
    forged_token["signature"] = original_sig[:-1] + corrupted_char
    
    # The mathematical verification WILL fail
    is_valid = tpm.verify_attestation_token(forged_token, expected_nonce=live_nonce, public_key_path=PUBLIC_KEY_PATH)
    
    # 0.8 * 0.5 = 0.4 (Forces quarantine)
    db.process_attestation(hardware_id, is_valid=is_valid)
    
    if db.is_quarantined(hardware_id):
        logger.info("✅ Quarantine successfully engaged due to bad signature.")
    else:
        logger.error("❌ State machine failed to engage quarantine.")
        sys.exit(1)

    logger.debug("--- Simulating repeated malicious communication while quarantined ---")
    for i in range(1, 4):
        logger.debug(f"  [Rejection Loop {i}/3] Generating and forging new payload...")
        temp_nonce = f"persistent_forgery_{i}"
        temp_token = tpm.generate_attestation_token(nonce=temp_nonce, software_label=SOFTWARE_LABEL)
        temp_token["signature"] = "tampered_garbage_signature" # Forge it
        
        is_valid = tpm.verify_attestation_token(temp_token, expected_nonce=temp_nonce, public_key_path=PUBLIC_KEY_PATH)
        db.process_attestation(hardware_id, is_valid=is_valid)
        
    final_score = db.get_score(hardware_id)
    # 0.4 -> 0.2 -> 0.1 -> 0.05
    if final_score == 0.05 and db.is_quarantined(hardware_id):
        logger.info(f"✅ Persistent isolation nominal. Score eroded to {final_score:.2f} and quarantine maintained.")
    else:
        logger.error("❌ Isolation boundary failure.")
        sys.exit(1)

    # =====================================================================
    # TEST 3: The Failed Rehabilitation
    # =====================================================================
    print_separator("Failed Rehabilitation (Interrupted Recovery)")
    logger.debug("Node starts sending valid hardware tokens, but fails before 5th round.")
    
    logger.debug("  -> Generating 3 REAL, valid tokens (needs 5 to recover)...")
    for i in range(1, 4):
        rec_nonce = f"recovery_nonce_{i}"
        valid_token = tpm.generate_attestation_token(nonce=rec_nonce, software_label=SOFTWARE_LABEL)
        is_valid = tpm.verify_attestation_token(valid_token, expected_nonce=rec_nonce, public_key_path=PUBLIC_KEY_PATH)
        db.process_attestation(hardware_id, is_valid=is_valid)
        
    logger.debug("  -> Node suddenly sends 1 forged token. Recovery must reset.")
    fail_nonce = "recovery_fail_nonce"
    fail_token = tpm.generate_attestation_token(nonce=fail_nonce, software_label=SOFTWARE_LABEL)
    fail_token["signature"] = "garbage"
    is_valid = tpm.verify_attestation_token(fail_token, expected_nonce=fail_nonce, public_key_path=PUBLIC_KEY_PATH)
    db.process_attestation(hardware_id, is_valid=is_valid)
    
    if db._db[hardware_id]["recovery_streak"] == 0 and db.is_quarantined(hardware_id):
        logger.info("✅ Rehabilitation interruption nominal. Streak reset to 0; quarantine maintained.")
    else:
        logger.error("❌ Rehabilitation logic failure. Streak was not reset.")
        sys.exit(1)

    # =====================================================================
    # TEST 4: The Terminal State (Unrecoverable Node)
    # =====================================================================
    print_separator("Terminal State Exhaustion (Never Recovers)")
    logger.debug("Simulating a node that never completes the 5-step recovery.")
    
    logger.debug("  -> Executing pattern: 4 valid, 1 invalid, 4 valid, 1 invalid...")
    for cycle in range(3):
        for i in range(4):
            t_nonce = f"term_valid_{cycle}_{i}"
            t_token = tpm.generate_attestation_token(nonce=t_nonce, software_label=SOFTWARE_LABEL)
            is_valid = tpm.verify_attestation_token(t_token, expected_nonce=t_nonce, public_key_path=PUBLIC_KEY_PATH)
            db.process_attestation(hardware_id, is_valid=is_valid)
            
        t_fail_nonce = f"term_fail_{cycle}"
        t_fail_token = tpm.generate_attestation_token(nonce=t_fail_nonce, software_label=SOFTWARE_LABEL)
        t_fail_token["signature"] = "garbage"
        is_valid = tpm.verify_attestation_token(t_fail_token, expected_nonce=t_fail_nonce, public_key_path=PUBLIC_KEY_PATH)
        db.process_attestation(hardware_id, is_valid=is_valid)
        
    terminal_score = db.get_score(hardware_id)
    
    if db.is_quarantined(hardware_id) and terminal_score < 0.01:
        logger.info(f"✅ Terminal state verified. Score crashed to {terminal_score:.5f}.")
    else:
        logger.error("❌ Terminal state failure.")
        sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 TRUST DATABASE + HARDWARE INTEGRATION COMPLETE. ALL TESTS PASSED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_hardware_integration_diagnostics()