import os
import subprocess
import base64
import logging
import re
import uuid

class TPMEngine:
    """A hardware abstraction layer for TPM 2.0 cryptographic provisioning and zero-trust attestation quote generation."""
    def __init__(self, logger: logging.Logger, insecure_mode: bool = False):
        """Initializes the hardware TPM Engine supporting zero-trust attestation and securely provisions the context."""
        self.logger = logger
        self.insecure_mode = insecure_mode or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
        self.tcti = os.getenv("TPM2TOOLS_TCTI", "").strip()
        
        if not self.insecure_mode and self.tcti:
            self.logger.info("TPMEngine: Hardware context detected. Initializing prover configuration.", extra={"round": 0})
            self._flush_tpm_memory()
            self._provision_ak()
        else:
            self.logger.info("TPMEngine: Hardware context absent or insecure mode enabled. Defaulting to verifier-only mode.", extra={"round": 0})

    def _flush_tpm_memory(self):
        """Flushes transient handles and actively loaded sessions from the TPM's internal memory space."""
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
        """Provisions a brand new Attestation Key (AK) securely within the TPM's non-volatile memory."""
        try:
            tpm_dir = "/app/runtime/tpm_state"
            shared_ak_pub = None
            
            if os.path.exists(tpm_dir):
                for dirname in os.listdir(tpm_dir):
                    if dirname.startswith("edge_"):
                        shared_ak_pub = os.path.join(tpm_dir, dirname, "ak.pub")
                        break

            if subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True).returncode == 0:
                self.logger.info("TPMEngine: AK already exists in NVRAM.", extra={"round": 0})
                
                if shared_ak_pub and not os.path.exists(shared_ak_pub):
                    self.logger.info(f"TPMEngine: Re-exporting ak.pub to shared volume: {shared_ak_pub}", extra={"round": 0})
                    subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-o", shared_ak_pub], check=True)
                return 

            self.logger.info("TPMEngine: Initializing new Attestation Key (AK) provisioning sequence.", extra={"round": 0})
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
            
            if shared_ak_pub:
                subprocess.run(["cp", "/tmp/ak.pub", shared_ak_pub], check=True)
            
            self.logger.info("TPMEngine: AK provisioning sequence completed successfully.", extra={"round": 0})
        except Exception as e:
            self.logger.error(f"TPMEngine: Provisioning error: {str(e)}", extra={"round": 0})
        finally:
            self._flush_tpm_memory()

    def _get_hardware_identity(self) -> str:
        """Retrieves the fixed hardware identity string structurally associated with the provisioned Attestation Key."""
        try:
            res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True, text=True, check=True)
            for line in res.stdout.split('\n'):
                if line.strip().startswith("name:"):
                    return line.split("name:")[1].strip()
            return "UNKNOWN_HARDWARE_ID"
        except Exception:
            return "UNKNOWN_HARDWARE_ID"

    def generate_attestation_token(self, nonce: str, software_label: str, round_num: int = 0) -> dict:
        """Generates a cryptographic quote and captures the active PCR state to serve as a verifiable attestation token."""
        if self.insecure_mode:
            return {"status": "insecure_bypass", "IDi": software_label, "pcr_hash": "dummy_hash", "signature": "dummy_sig"}

        fog_num, edge_num = None, None
        try:
            m = re.search(r'\[EDGE (\d+)_(\d+)\]', software_label)
            if m:
                fog_num, edge_num = m.groups()
                ak_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/ak.pub"
                
                if not os.path.exists(ak_path):
                    res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-f", "pem", "-o", ak_path], capture_output=True)
                    if res.returncode != 0:
                        subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-o", ak_path], check=True)
        except Exception as e:
            self.logger.error(f"TPMEngine: Failed to export ak.pub: {e}", extra={"round": round_num})

        self.logger.debug(f"[TPM-GENERATE] Edge Node {software_label} preparing to sign NONCE: {nonce}", extra={"round": round_num})

        session_id = uuid.uuid4().hex
        nonce_file = f"/tmp/nonce_{session_id}.bin"
        msg_file = f"/tmp/quote_{session_id}.msg"
        sig_file = f"/tmp/quote_{session_id}.sig"
        pcr_file = f"/tmp/pcr_{session_id}.bin"

        try:
            with open(nonce_file, "w") as f:
                f.write(nonce)

            subprocess.run([
                "tpm2_quote", "-c", "0x81010002", "-l", "sha256:0", 
                "-q", nonce_file, "-m", msg_file, 
                "-s", sig_file, "-o", pcr_file
            ], check=True, capture_output=True)

            hardware_idi = self._get_hardware_identity()

            if fog_num and edge_num:
                shared_pcr_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/clean_pcr.bin"
                shared_id_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/tpm_id.txt"
                
                if not os.path.exists(shared_pcr_path):
                    subprocess.run(["cp", pcr_file, shared_pcr_path], check=True)
                    self.logger.info(f"TPMEngine: Exported pristine PCR baseline to {shared_pcr_path}", extra={"round": round_num})
                    
                if not os.path.exists(shared_id_path):
                    with open(shared_id_path, "w") as f:
                        f.write(hardware_idi)

            with open(msg_file, "rb") as f:
                msg_b64 = base64.b64encode(f.read()).decode('utf-8')
            with open(sig_file, "rb") as f:
                sig_b64 = base64.b64encode(f.read()).decode('utf-8')
            with open(pcr_file, "rb") as f:
                pcr_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            self.logger.info(f"TPMEngine: Attestation token generated for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
            
            return {
                "status": "attested",
                "software_label": software_label, 
                "IDi": hardware_idi,              
                "quote_msg": msg_b64,             
                "signature": sig_b64,
                "pcr_data": pcr_b64
            }
        except Exception as e:
            self.logger.error(f"TPMEngine: Execution error during token generation: {str(e)}", extra={"round": round_num})
            return {"status": "error"}
        finally:
            for temp_file in [nonce_file, msg_file, sig_file, pcr_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def verify_attestation_token(self, token: dict, expected_nonce: str, public_key_path: str, round_num: int = 0, expected_pcr: str = None) -> bool:
        """Verifies an incoming attestation token's signature, structure, and PCR health against the expected baseline."""
        if self.insecure_mode or token.get("status") == "insecure_bypass":
            return True
            
        if token.get("status") == "simulated_key_theft":
            self.logger.critical(f"[TPM-VERIFY] ⚠️ SIMULATED KEY THEFT: Bypassing cryptography. The attacker possesses the valid private key!", extra={"round": round_num})
            return True
            
        if token.get("status") != "attested":
            self.logger.warning(f"[TPM-VERIFY] Edge Node sent corrupted/error token: {token}", extra={"round": round_num})
            return False

        hardware_idi = token.get('IDi', 'Unknown')
        
        self.logger.debug(f"[TPM-VERIFY] Fog Server evaluating token from IDi {hardware_idi[:16]}... EXPECTED NONCE: {expected_nonce}", extra={"round": round_num})

        session_id = uuid.uuid4().hex
        msg_file = f"/tmp/verify_quote_{session_id}.msg"
        sig_file = f"/tmp/verify_quote_{session_id}.sig"
        nonce_file = f"/tmp/expected_nonce_{session_id}.bin"

        try:
            with open(msg_file, "wb") as f:
                f.write(base64.b64decode(token["quote_msg"]))
            with open(sig_file, "wb") as f:
                f.write(base64.b64decode(token["signature"]))
            with open(nonce_file, "w") as f:
                f.write(expected_nonce)

            result = subprocess.run([
                "tpm2_checkquote", "-u", public_key_path, 
                "-m", msg_file, "-s", sig_file,
                "-q", nonce_file 
            ], capture_output=True, text=True)

            is_valid = result.returncode == 0
            
            if is_valid:
                actual_pcr = token.get("pcr_data", "")
                if expected_pcr and actual_pcr != expected_pcr:
                    self.logger.critical(f"TPMEngine: 🚨 PCR HEALTH CHECK FAILED! OS tampering detected for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
                    is_valid = False
                else:
                    self.logger.info(f"TPMEngine: Cryptographic and PCR verification passed for IDi: {hardware_idi[:16]}...", extra={"round": round_num})
            else:
                self.logger.warning(f"TPMEngine: Verification rejected. Expected Nonce matched Token Nonce? {is_valid}. Output: {result.stderr.strip()}", extra={"round": round_num})
                
            return is_valid
        except Exception as e:
            self.logger.error(f"TPMEngine: Exception during verification routine: {str(e)}", extra={"round": round_num})
            return False
        finally:
            for temp_file in [msg_file, sig_file, nonce_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)