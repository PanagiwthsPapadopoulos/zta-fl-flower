import os
import subprocess
import base64
import logging
import re
import uuid
import time
import traceback


class TPMAttestation:
    """A hardware abstraction layer for TPM 2.0 cryptographic provisioning and zero-trust attestation quote generation."""
    def __init__(self, logger: logging.Logger, current_round: int=0, insecure_mode: bool = False):
        self.logger = logger
        self.insecure_mode = insecure_mode or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
        self.tcti = os.getenv("TPM2TOOLS_TCTI", "").strip()
        self.current_round = current_round
        
        if not self.insecure_mode and self.tcti:
            self.logger.info("TPMEngine: Hardware context detected. Initializing prover configuration.", extra={"round": self.current_round})
            self._flush_tpm_memory()
            self._provision_ak()
        else:
            self.logger.info("TPMEngine: Hardware context absent or insecure mode enabled. Defaulting to verifier-only mode.", extra={"round": self.current_round})

    def _flush_tpm_memory(self):
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
        try:
            tpm_dir = "/app/runtime/tpm_state"
            shared_ak_pub = None
            
            if os.path.exists(tpm_dir):
                for dirname in os.listdir(tpm_dir):
                    if dirname.startswith("edge_"):
                        shared_ak_pub = os.path.join(tpm_dir, dirname, "ak.pub")
                        break

            if subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True).returncode == 0:
                self.logger.info("TPMEngine: AK already exists in NVRAM.", extra={"round": self.current_round})
                if shared_ak_pub and not os.path.exists(shared_ak_pub):
                    subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-o", shared_ak_pub], check=True)
                return 

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
        except Exception as e:
            self.logger.error(f"TPMEngine: Provisioning error: {str(e)}", extra={"round": self.current_round})
        finally:
            self._flush_tpm_memory()

    def _get_hardware_identity(self) -> str:
        try:
            res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True, text=True, check=True)
            for line in res.stdout.split('\n'):
                if line.strip().startswith("name:"):
                    return line.split("name:")[1].strip()
            return "UNKNOWN_HARDWARE_ID"
        except Exception:
            return "UNKNOWN_HARDWARE_ID"

    def generate_attestation_token(self, nonce: str, software_label: str, round_num: int) -> dict:
        """Generates a cryptographic quote and captures the active PCR state to serve as a verifiable attestation token."""
        try:
            if self.insecure_mode:
                return {"status": "insecure_bypass", "IDi": software_label, "pcr_hash": "dummy_hash", "signature": "dummy_sig"}

            fog_num, edge_num = None, None
            m = re.search(r'\[EDGE (\d+)_(\d+)\]', software_label)
            if m:
                fog_num, edge_num = m.groups()
                ak_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/ak.pub"
                
                if not os.path.exists(ak_path):
                    res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-f", "pem", "-o", ak_path], capture_output=True)
                    if res.returncode != 0:
                        subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-o", ak_path], check=True)

            self.logger.debug(f"[TPM-GENERATE] Edge Node {software_label} preparing to sign NONCE: {nonce}", extra={"round": round_num})

            session_id = uuid.uuid4().hex
            nonce_file = f"/tmp/nonce_{session_id}.bin"
            msg_file = f"/tmp/quote_{session_id}.msg"
            sig_file = f"/tmp/quote_{session_id}.sig"
            pcr_file = f"/tmp/pcr_{session_id}.bin"

            hardware_idi = self._get_hardware_identity()
            
            # --- LOG: PLAINTEXT VALUES BEFORE HASHING ---
            self.logger.info(f"[TPM-GENERATE] Plaintext inputs mapped for hardware operation -> IDi: {hardware_idi}, Nonce: {nonce}", extra={"round": round_num})

            # Force binary writing so the TPM hashes true bytes, not ASCII characters
            try:
                nonce_bytes = bytes.fromhex(nonce)
            except ValueError:
                nonce_bytes = nonce.encode('utf-8')
                
            with open(nonce_file, "wb") as f:
                f.write(nonce_bytes)

            subprocess.run([
                "tpm2_quote", "-c", "0x81010002", "-l", "sha256:0", 
                "-q", nonce_file, "-m", msg_file, 
                "-s", sig_file, "-o", pcr_file
            ], check=True, capture_output=True)
            
            # --- LOG: HARDWARE HASHING/SIGNING CONFIRMATION ---
            self.logger.info(f"[TPM-GENERATE] The nonce and PCR value have been successfully hashed and signed together into the hardware quote structure.", extra={"round": round_num})

            if fog_num and edge_num:
                shared_pcr_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/clean_pcr.bin"
                shared_id_path = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}/tpm_id.txt"
                
                if not os.path.exists(shared_pcr_path):
                    subprocess.run(["cp", pcr_file, shared_pcr_path], check=True)
                    
                if not os.path.exists(shared_id_path):
                    with open(shared_id_path, "w") as f:
                        f.write(hardware_idi)

            with open(msg_file, "rb") as f:
                msg_b64 = base64.b64encode(f.read()).decode('utf-8')
            with open(sig_file, "rb") as f:
                sig_b64 = base64.b64encode(f.read()).decode('utf-8')

            token = {
                "status": "attested",
                "software_label": software_label, 
                "IDi": hardware_idi,              
                "quote_msg": msg_b64,             
                "signature": sig_b64,
                "timestamp": time.time(),
            }

            # --- LOG: FINAL PLAINTEXT TOKEN ---
            self.logger.info(f"[TPM-GENERATE] Final Plaintext JSON Token prepared for transit: {token}", extra={"round": round_num})

            return token
            
        except Exception as e:
            # MASTER TRACEBACK LOGGER FOR SILENT CRASHES
            self.logger.critical(f"[TPM-CRASH] Silent crash intercepted in generate_attestation_token! Error: {e}\nTraceback: {traceback.format_exc()}", extra={"round": round_num})
            return {"status": "error"}
        finally:
            try:
                for temp_file in [nonce_file, msg_file, sig_file, pcr_file]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception:
                pass