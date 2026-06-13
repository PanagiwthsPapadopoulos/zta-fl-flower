import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from security.attestation.tpm_core import TPMEngine

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("TPM_Diagnostics")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

PUBLIC_KEY_PATH = "/app/tpm_state/ak.pub"

def print_separator(title: str):
    print("\n" + "="*80)
    print(f"TEST: {title}")
    print("="*80)

def run_diagnostics():
    print("\n" + "#"*80)
    print("--- TPM 2.0 DIAGNOSTIC SUITE ---")
    print("#"*80)

    # =====================================================================
    # TEST 1: Engine Initialization & Hardware Detection
    # =====================================================================
    print_separator("TPMEngine Initialization & Configuration")
    logger.debug("Initializing TPMEngine instance.")
    
    start_time = time.time()
    tpm = TPMEngine(logger=logger, insecure_mode=False)
    boot_time = time.time() - start_time
    
    logger.info(f"Engine initialization completed in {boot_time:.2f} seconds.")
    if not os.path.exists(PUBLIC_KEY_PATH):
        logger.error(f"Initialization failure: Public key missing at {PUBLIC_KEY_PATH}.")
        sys.exit(1)
    else:
        logger.info(f"Key export verified at {PUBLIC_KEY_PATH}.")

    # =====================================================================
    # TEST 2: Standard Token Generation
    # =====================================================================
    print_separator("Standard Attestation Token Generation")
    valid_nonce = "diagnostic_nonce_8899aabbccddeeff"
    software_label = "EDGE_1_1"
    
    logger.debug("Simulating attestation request from verifier.")
    logger.debug(f"Input Software Label: {software_label}")
    logger.debug(f"Input Nonce (t): {valid_nonce}")
    
    token = tpm.generate_attestation_token(nonce=valid_nonce, software_label=software_label)
    
    if token.get("status") != "attested":
        logger.error(f"Generation failure. Returned status: {token.get('status')}")
        sys.exit(1)

    logger.info("Token generation successful. Payload structure:")
    logger.info(f"  [1] IDi (Hardware Name) : {token['IDi']}")
    logger.info(f"  [2] Software Route      : {token['software_label']}")
    logger.info(f"  [3] PCR Data (Base64)   : {token['pcr_data'][:40]}... (truncated)")
    logger.info(f"  [4] Signature (Base64)  : {token['signature'][:40]}... (truncated)")
    
    golden_token = token

    # =====================================================================
    # TEST 3: Token Verification (Valid Data)
    # =====================================================================
    print_separator("Cryptographic Verification of Valid Token")
    logger.debug("Simulating verifier processing of incoming token.")
    logger.debug(f"Expected Nonce parameter: {valid_nonce}")
    
    is_valid = tpm.verify_attestation_token(
        token=golden_token, 
        expected_nonce=valid_nonce, 
        public_key_path=PUBLIC_KEY_PATH
    )
    
    if is_valid:
        logger.info("Verification passed. Signature and constraints validated.")
    else:
        logger.error("Verification failed. Cryptographic constraints were not met.")
        sys.exit(1)

    # =====================================================================
    # TEST 4: Replay Attack Simulation (Mismatched Nonce)
    # =====================================================================
    print_separator("Replay Attack Prevention (Invalid Nonce)")
    logger.debug("Simulating submission of valid token against a stale nonce constraint.")
    stale_nonce = "diagnostic_stale_nonce_112233"
    logger.debug(f"Token bound nonce: {valid_nonce}")
    logger.debug(f"Verifier expected nonce: {stale_nonce}")
    
    is_valid = tpm.verify_attestation_token(
        token=golden_token, 
        expected_nonce=stale_nonce, 
        public_key_path=PUBLIC_KEY_PATH
    )
    
    if not is_valid:
        logger.info("Replay attack successfully mitigated. Nonce mismatch detected.")
    else:
        logger.error("Replay mitigation failed. Validation bypassed the nonce constraint.")
        sys.exit(1)

    # =====================================================================
    # TEST 5: Signature Forgery Simulation
    # =====================================================================
    print_separator("Signature Forgery Detection")
    logger.debug("Simulating verification against tampered payload data.")
    
    forged_token = golden_token.copy()
    original_sig = forged_token["signature"]
    corrupted_char = 'B' if original_sig[-1] == 'A' else 'A'
    forged_token["signature"] = original_sig[:-1] + corrupted_char
    
    logger.debug(f"Original signature suffix: ...{original_sig[-5:]}")
    logger.debug(f"Modified signature suffix: ...{forged_token['signature'][-5:]}")
    
    is_valid = tpm.verify_attestation_token(
        token=forged_token, 
        expected_nonce=valid_nonce, 
        public_key_path=PUBLIC_KEY_PATH
    )
    
    if not is_valid:
        logger.info("Forgery successfully detected. Cryptographic signature rejected.")
    else:
        logger.error("Forgery detection failed. Tampered signature was accepted.")
        sys.exit(1)

    # =====================================================================
    # TEST 6: Memory Management Stress Test
    # =====================================================================
    print_separator("Transient Memory Management (Stress Test)")
    logger.debug("Executing sequential requests to validate transient resource cleanup.")
    
    success_count = 0
    for i in range(1, 6):
        stress_nonce = f"stress_nonce_{i}"
        logger.debug(f"  [Cycle {i}/5] Executing generation with nonce: {stress_nonce}")
        
        try:
            temp_token = tpm.generate_attestation_token(nonce=stress_nonce, software_label=software_label)
            if temp_token.get("status") == "attested":
                success_count += 1
                logger.debug(f"  [Cycle {i}/5] Execution nominal.")
            else:
                logger.warning(f"  [Cycle {i}/5] Execution failed. Status: {temp_token.get('status')}")
        except Exception as e:
            logger.error(f"  [Cycle {i}/5] Exception encountered: {str(e)}")
            
    if success_count == 5:
        logger.info("Stress test completed successfully without memory exhaustion.")
    else:
        logger.error(f"Memory management failure. Completed {success_count}/5 cycles.")
        sys.exit(1)

    print("\n" + "#"*80)
    print("--- DIAGNOSTIC SUITE EXECUTION COMPLETE ---")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_diagnostics()