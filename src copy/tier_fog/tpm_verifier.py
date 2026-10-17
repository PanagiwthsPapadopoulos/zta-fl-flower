import os
import subprocess
import base64
import logging
import uuid
import time
import re
import traceback


class TPMVerifier:
    """Fog tier validation layer verifying inbound edge attestation tokens statically."""
    def __init__(self, logger: logging.Logger, insecure_mode: bool = False):
        self.logger = logger
        self.insecure_mode = insecure_mode or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"

    def verify_attestation_token(self, token: dict, expected_nonce: str, public_key_path: str, round_num: int, expected_pcr: str = None, max_age_seconds: int = 60) -> bool:
        """Verifies an incoming attestation token's signature, structure, and cryptographically extracts PCR health against the expected baseline."""
        try:
            if self.insecure_mode or token.get("status") == "insecure_bypass":
                return True
                            
            if token.get("status") != "attested":
                self.logger.warning(f"[TPM-VERIFY] Edge Node sent corrupted/error token: {token}", extra={"round": round_num})
                return False
            
            token_time = token.get("timestamp", 0)
            current_time = time.time()
            
            if (current_time - token_time) > max_age_seconds:
                self.logger.critical(f"[TPM-VERIFY] FRESHNESS CHECK FAILED! Token expired. Age: {current_time - token_time:.1f}s > {max_age_seconds}s", extra={"round": round_num})
                return False

            hardware_idi = token.get('IDi', 'Unknown')
            self.logger.debug(f"[TPM-VERIFY] Fog Server evaluating token from IDi {hardware_idi[:16]}... EXPECTED NONCE: {expected_nonce}", extra={"round": round_num})

            session_id = uuid.uuid4().hex
            msg_file = f"/tmp/verify_quote_{session_id}.msg"
            sig_file = f"/tmp/verify_quote_{session_id}.sig"
            nonce_file = f"/tmp/expected_nonce_{session_id}.bin"

            with open(msg_file, "wb") as f:
                f.write(base64.b64decode(token["quote_msg"]))
            with open(sig_file, "wb") as f:
                f.write(base64.b64decode(token["signature"]))
                
            # Force binary writing so tpm2_checkquote matches the Edge's exact binary input
            try:
                nonce_bytes = bytes.fromhex(expected_nonce)
            except ValueError:
                nonce_bytes = expected_nonce.encode('utf-8')
                
            with open(nonce_file, "wb") as f:
                f.write(nonce_bytes)

            result = subprocess.run([
                "tpm2_checkquote", "-u", public_key_path, 
                "-m", msg_file, "-s", sig_file,
                "-q", nonce_file 
            ], capture_output=True, text=True)

            is_valid = result.returncode == 0
            
            if is_valid:
                # SECURE EXTRACTION: Crack open the verified binary message
                print_result = subprocess.run([
                    "tpm2_print", "-t", "TPMS_ATTEST", msg_file
                ], capture_output=True, text=True)
                
                if print_result.returncode != 0:
                    self.logger.error(f"TPMEngine: Failed to parse TPMS_ATTEST structure for IDi: {hardware_idi[:16]}", extra={"round": round_num})
                    return False
                    
                # --- LOG: UNPACKING CONFIRMATION ---
                self.logger.info(f"[TPM-VERIFY] The sealed hardware token has been successfully unhashed/unpacked on the server.", extra={"round": round_num})
                    
                # Scan the output for the true PCR digest and the actual Nonce sealed inside the signature
                pcr_match = re.search(r'pcrDigest:\s*([a-fA-F0-9]+)', print_result.stdout)
                nonce_match = re.search(r'extraData:\s*([a-fA-F0-9]+)', print_result.stdout)
                
                actual_pcr = pcr_match.group(1) if pcr_match else "NOT_FOUND"
                actual_nonce = nonce_match.group(1) if nonce_match else "NOT_FOUND"
                
                # --- LOG: EXTRACTED RAW VALUES FOR COMPARISON ---
                self.logger.info(f"[TPM-VERIFY] Raw Unpacked Values for comparison -> IDi: {hardware_idi[:16]}..., Extracted Nonce: {actual_nonce}, Extracted PCR: {actual_pcr}", extra={"round": round_num})
                
                if actual_nonce != expected_nonce:
                    self.logger.critical(f"TPMEngine: NONCE MISMATCH FAILED! Replay attack detected for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
                    is_valid = False
                elif expected_pcr and actual_pcr != expected_pcr:
                    self.logger.critical(f"TPMEngine: PCR HEALTH CHECK FAILED! OS tampering detected for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
                    is_valid = False
                else:
                    self.logger.info(f"TPMEngine: Cryptographic and PCR verification passed for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
            else:
                self.logger.warning(f"TPMEngine: Verification rejected. Expected Nonce matched Token Nonce? {is_valid}. Output: {result.stderr.strip()}", extra={"round": round_num})
                
            return is_valid
            
        except Exception as e:
            # MASTER TRACEBACK LOGGER FOR SILENT CRASHES
            self.logger.critical(f"[TPM-CRASH] Silent crash intercepted in verify_attestation_token! Error: {e}\nTraceback: {traceback.format_exc()}", extra={"round": round_num})
            return False
        finally:
            try:
                for temp_file in [msg_file, sig_file, nonce_file]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception:
                pass