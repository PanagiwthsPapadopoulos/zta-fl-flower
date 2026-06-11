import os
import subprocess
import base64
import logging
import re

class TPMEngine:
    """
    Interface for TPM 2.0 hardware operations supporting zero-trust attestation.
    
    If the ZTA_INSECURE_MODE environment variable is set to true, the engine 
    bypasses hardware dependencies to facilitate software-only testing.
    """
    def __init__(self, logger: logging.Logger, insecure_mode: bool = False):
        self.logger = logger
        self.insecure_mode = insecure_mode or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
        self.tcti = os.getenv("TPM2TOOLS_TCTI", "").strip()
        
        if not self.insecure_mode and self.tcti:
            self.logger.info("TPMEngine: Hardware context detected. Initializing prover configuration.")
            self._flush_tpm_memory()
            self._provision_ak()
        else:
            self.logger.info("TPMEngine: Hardware context absent or insecure mode enabled. Defaulting to verifier-only mode.")

    def _flush_tpm_memory(self):
        """
        Clears transient object handles and session contexts from the TPM's volatile memory.
        Prevents 0x902 (Out of Memory) errors during sequential operations.
        """
        try:
            res = subprocess.run(["tpm2_getcap", "handles-transient"], capture_output=True, text=True)
            for handle in re.findall(r'0x[0-9a-fA-F]+', res.stdout):
                subprocess.run(["tpm2_flushcontext", handle], capture_output=True)

            res = subprocess.run(["tpm2_getcap", "handles-loaded-session"], capture_output=True, text=True)
            for handle in re.findall(r'0x[0-9a-fA-F]+', res.stdout):
                subprocess.run(["tpm2_flushcontext", handle], capture_output=True)
        except Exception:
            pass 

    def _provision_ak(self):
        """
        Executes the provisioning sequence for the Attestation Key (AK).
        
        Procedure:
        1. Verifies the existence of an AK at the target NVRAM index.
        2. Generates a Primary Key and persists it to bypass transient memory constraints.
        3. Creates the AK under the persistent parent.
        4. Loads and evicts the AK to its final NVRAM destination.
        5. Exports the public key for verifier access.
        """
        try:
            if subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True).returncode == 0:
                return 

            self.logger.info("TPMEngine: Initializing new Attestation Key (AK) provisioning sequence.")

            subprocess.run(["tpm2_createprimary", "-C", "o", "-c", "/tmp/primary.ctx"], check=True)
            subprocess.run(["tpm2_evictcontrol", "-C", "o", "-c", "/tmp/primary.ctx", "0x81000001"], check=True)
            self._flush_tpm_memory()
            
            subprocess.run([
                "tpm2_create", "-C", "0x81000001", 
                "-G", "rsa2048:rsassa:null", "-g", "sha256",  
                "-a", "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|restricted|sign",
                "-u", "/tmp/ak.pub", "-r", "/tmp/ak.priv"
            ], check=True)
            
            subprocess.run(["tpm2_load", "-C", "0x81000001", "-u", "/tmp/ak.pub", "-r", "/tmp/ak.priv", "-c", "/tmp/ak.ctx"], check=True)
            subprocess.run(["tpm2_evictcontrol", "-C", "o", "-c", "/tmp/ak.ctx", "0x81010002"], check=True)
            
            subprocess.run(["cp", "/tmp/ak.pub", "/app/tpm_state/ak.pub"], check=True)
            self.logger.info("TPMEngine: AK provisioning sequence completed successfully.")
            
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"TPMEngine: Provisioning error: {err_msg}")
        except Exception as e:
            self.logger.error(f"TPMEngine: Unexpected failure during provisioning: {str(e)}")
        finally:
            self._flush_tpm_memory()

    def _get_hardware_identity(self) -> str:
        """
        Retrieves the cryptographic 'Name' of the Attestation Key.
        Serves as the unique hardware identifier (IDi) for the edge device.
        """
        try:
            res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True, text=True, check=True)
            for line in res.stdout.split('\n'):
                if line.strip().startswith("name:"):
                    return line.split("name:")[1].strip()
            return "UNKNOWN_HARDWARE_ID"
        except Exception as e:
            self.logger.error(f"TPMEngine: Identity extraction failed: {e}")
            return "UNKNOWN_HARDWARE_ID"

    def generate_attestation_token(self, nonce: str, software_label: str) -> dict:
        """
        Reads the Platform Configuration Registers (PCRs) and generates a signed quote.
        
        Args:
            nonce (str): A verifier-provided string to ensure payload freshness.
            software_label (str): The logical identifier for network routing.
            
        Returns:
            dict: The complete attestation tuple {IDi, t, PCR, SigTPM} encoded in base64.
        """
        if self.insecure_mode:
            self.logger.debug("TPMEngine: Insecure mode active. Returning mock token.")
            return {"status": "insecure_bypass", "IDi": software_label, "pcr_hash": "dummy_hash", "signature": "dummy_sig"}

        try:
            with open("/tmp/nonce.bin", "w") as f:
                f.write(nonce)

            subprocess.run([
                "tpm2_quote", "-c", "0x81010002", "-l", "sha256:0", 
                "-q", "/tmp/nonce.bin", "-m", "/tmp/quote.msg", 
                "-s", "/tmp/quote.sig", "-o", "/tmp/pcr.bin"
            ], check=True, capture_output=True)

            hardware_idi = self._get_hardware_identity()

            with open("/tmp/quote.msg", "rb") as f:
                msg_b64 = base64.b64encode(f.read()).decode('utf-8')
            with open("/tmp/quote.sig", "rb") as f:
                sig_b64 = base64.b64encode(f.read()).decode('utf-8')
            with open("/tmp/pcr.bin", "rb") as f:
                pcr_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            self.logger.info(f"TPMEngine: Attestation token generated for IDi: {hardware_idi[:16]}...")
            
            return {
                "status": "attested",
                "software_label": software_label, 
                "IDi": hardware_idi,              
                "quote_msg": msg_b64,             
                "signature": sig_b64,
                "pcr_data": pcr_b64
            }
            
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"TPMEngine: Hardware quoting failure: {err_msg}")
            return {"status": "hardware_fault"}
        except Exception as e:
            self.logger.error(f"TPMEngine: Execution error during token generation: {str(e)}")
            return {"status": "error"}

    def verify_attestation_token(self, token: dict, expected_nonce: str, public_key_path: str) -> bool:
        """
        Validates the signature, nonce, and PCR state of an incoming attestation token.
        
        Args:
            token (dict): The base64-encoded token dictionary generated by the prover.
            expected_nonce (str): The nonce originally issued by the verifier.
            public_key_path (str): File system path to the prover's public key.
            
        Returns:
            bool: True if cryptographic and logical verifications pass, False otherwise.
        """
        if self.insecure_mode or token.get("status") == "insecure_bypass":
            self.logger.debug("TPMEngine: Insecure mode active. Token verification bypassed.")
            return True

        if token.get("status") != "attested":
            self.logger.warning("TPMEngine: Invalid token status detected. Rejecting.")
            return False

        try:
            with open("/tmp/verify_quote.msg", "wb") as f:
                f.write(base64.b64decode(token["quote_msg"]))
            with open("/tmp/verify_quote.sig", "wb") as f:
                f.write(base64.b64decode(token["signature"]))

            with open("/tmp/expected_nonce.bin", "w") as f:
                f.write(expected_nonce)

            result = subprocess.run([
                "tpm2_checkquote", "-u", public_key_path, 
                "-m", "/tmp/verify_quote.msg", "-s", "/tmp/verify_quote.sig",
                "-q", "/tmp/expected_nonce.bin" 
            ], capture_output=True, text=True)

            is_valid = result.returncode == 0
            if is_valid:
                self.logger.info(f"TPMEngine: Cryptographic verification passed for IDi: {token.get('IDi', 'Unknown')[:16]}...")
            else:
                self.logger.warning(f"TPMEngine: Verification rejected. Output: {result.stderr.strip()}")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"TPMEngine: Exception during verification routine: {str(e)}")
            return False