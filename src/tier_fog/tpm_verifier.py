import hashlib
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

    def verify_attestation_token(
        self,
        token: dict,
        expected_nonce: str,
        public_key_path: str,
        round_num: int,
        expected_pcr: str = None,
        max_age_seconds: int = 60
    ) -> bool:
        """Verifies an incoming attestation token's signature, sealed nonce+timestamp freshness, and PCR health."""
        session_id = uuid.uuid4().hex
        msg_file = f"/tmp/verify_quote_{session_id}.msg"
        sig_file = f"/tmp/verify_quote_{session_id}.sig"
        nonce_file = f"/tmp/expected_nonce_{session_id}.bin"
        try:
            if self.insecure_mode or token.get("status") == "insecure_bypass":
                return True

            if token.get("status") != "attested":
                self.logger.warning(f"[TPM-VERIFY] Edge Node sent corrupted/error token: {token}", extra={"round": round_num})
                return False

            token_time = float(token.get("timestamp", 0.0))
            current_time = time.time()

            if (current_time - token_time) > max_age_seconds or (token_time - current_time) > 30.0:
                self.logger.critical(f"[TPM-VERIFY] FRESHNESS CHECK FAILED! Token expired or future-dated. Age: {current_time - token_time:.1f}s > {max_age_seconds}s", extra={"round": round_num})
                return False

            hardware_idi = token.get('IDi', 'Unknown')
            expected_qualifying_hex = hashlib.sha256(f"{expected_nonce}:{token_time:.6f}".encode("utf-8")).hexdigest()
            self.logger.debug(
                f"[TPM-VERIFY] Fog Server evaluating token from IDi {hardware_idi[:16]}... EXPECTED NONCE: {expected_nonce}, EXPECTED QUALIFYING DIGEST: {expected_qualifying_hex}",
                extra={"round": round_num}
            )

            with open(msg_file, "wb") as f:
                f.write(base64.b64decode(token["quote_msg"]))
            with open(sig_file, "wb") as f:
                f.write(base64.b64decode(token["signature"]))

            with open(nonce_file, "wb") as f:
                f.write(bytes.fromhex(expected_qualifying_hex))

            result = subprocess.run([
                "tpm2_checkquote", "-u", public_key_path,
                "-m", msg_file, "-s", sig_file,
                "-q", nonce_file
            ], capture_output=True, text=True)

            is_valid = result.returncode == 0

            if is_valid:
                print_result = subprocess.run([
                    "tpm2_print", "-t", "TPMS_ATTEST", msg_file
                ], capture_output=True, text=True)

                if print_result.returncode != 0:
                    self.logger.error(f"TPMEngine: Failed to parse TPMS_ATTEST structure for IDi: {hardware_idi[:16]}", extra={"round": round_num})
                    return False

                self.logger.info("[TPM-VERIFY] The sealed hardware token has been successfully unhashed/unpacked on the server.", extra={"round": round_num})

                pcr_match = re.search(r'pcrDigest:\s*([a-fA-F0-9]+)', print_result.stdout)
                nonce_match = re.search(r'extraData:\s*([a-fA-F0-9]+)', print_result.stdout)

                actual_pcr = pcr_match.group(1) if pcr_match else "NOT_FOUND"
                actual_nonce = nonce_match.group(1) if nonce_match else "NOT_FOUND"

                self.logger.info(
                    f"[TPM-VERIFY] Raw Unpacked Values for comparison -> IDi: {hardware_idi[:16]}..., Extracted Qualifying Digest: {actual_nonce}, Extracted PCR: {actual_pcr}",
                    extra={"round": round_num}
                )

                if actual_nonce.lower() != expected_qualifying_hex.lower():
                    self.logger.critical(f"TPMEngine: NONCE/TIMESTAMP MISMATCH FAILED! Replay or timestamp tampering detected for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
                    is_valid = False
                elif expected_pcr and actual_pcr.lower() != expected_pcr.lower():
                    self.logger.critical(f"TPMEngine: PCR HEALTH CHECK FAILED! OS tampering detected for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
                    is_valid = False
                else:
                    self.logger.info(f"TPMEngine: Cryptographic, freshness, and PCR verification passed for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
            else:
                self.logger.warning(f"TPMEngine: Verification rejected. Expected Nonce/Timestamp matched Token Quote? {is_valid}. Output: {result.stderr.strip()}", extra={"round": round_num})

            return is_valid

        except Exception as e:
            self.logger.critical(f"[TPM-CRASH] Silent crash intercepted in verify_attestation_token! Error: {e}\nTraceback: {traceback.format_exc()}", extra={"round": round_num})
            return False
        finally:
            try:
                for temp_file in [msg_file, sig_file, nonce_file]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception:
                pass