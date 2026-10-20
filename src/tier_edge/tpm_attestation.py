import hashlib
import os
import subprocess
import base64
import logging
import re
import uuid
import time
import traceback
from typing import Optional


class TPMAttestation:
    """A hardware abstraction layer for TPM 2.0 cryptographic provisioning and zero-trust attestation quote generation."""

    def __init__(self, logger: logging.Logger, current_round: int = 0, insecure_mode: bool = False, tcti: Optional[str] = None):
        self.logger = logger
        self.insecure_mode = insecure_mode or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
        self.tcti = (tcti or os.getenv("TPM2TOOLS_TCTI", "")).strip()
        self.current_round = current_round

        if not self.insecure_mode and self.tcti:
            self.logger.info("TPMEngine: Hardware context detected. Initializing prover configuration.", extra={"round": self.current_round})
            self._flush_tpm_memory()
            self._provision_ak()
        else:
            self.logger.info("TPMEngine: Hardware context absent or insecure mode enabled. Defaulting to verifier-only mode.", extra={"round": self.current_round})

    def _get_tpm_env(self) -> dict:
        """Returns a process-isolated environment dictionary bound to this node's specific TPM TCTI socket."""
        env = os.environ.copy()
        if self.tcti:
            env["TPM2TOOLS_TCTI"] = self.tcti
        return env

    def _flush_tpm_memory(self):
        env = self._get_tpm_env()
        try:
            res = subprocess.run(["tpm2_getcap", "handles-transient"], capture_output=True, text=True, env=env)
            for handle in re.findall(r'0x[0-9a-fA-F]+', res.stdout):
                subprocess.run(["tpm2_flushcontext", handle], capture_output=True, env=env)
            res = subprocess.run(["tpm2_getcap", "handles-loaded-session"], capture_output=True, text=True, env=env)
            for handle in re.findall(r'0x[0-9a-fA-F]+', res.stdout):
                subprocess.run(["tpm2_flushcontext", handle], capture_output=True, env=env)
        except Exception:
            pass

    def _provision_ak(self):
        env = self._get_tpm_env()
        session_id = uuid.uuid4().hex
        primary_ctx = f"/tmp/primary_{session_id}.ctx"
        ak_pub = f"/tmp/ak_{session_id}.pub"
        ak_priv = f"/tmp/ak_{session_id}.priv"
        ak_ctx = f"/tmp/ak_{session_id}.ctx"
        try:
            if subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True, env=env).returncode == 0:
                self.logger.info("TPMEngine: AK already exists in NVRAM.", extra={"round": self.current_round})
                return

            subprocess.run(["tpm2_createprimary", "-C", "o", "-c", primary_ctx], check=True, capture_output=True, env=env)
            subprocess.run(["tpm2_evictcontrol", "-C", "o", "-c", primary_ctx, "0x81000001"], check=True, capture_output=True, env=env)
            self._flush_tpm_memory()

            # Provision ECDSA-256 (NIST P-256) Attestation Key per paper specification
            subprocess.run([
                "tpm2_create", "-C", "0x81000001",
                "-G", "ecc256:ecdsa:null", "-g", "sha256",
                "-a", "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|restricted|sign",
                "-u", ak_pub, "-r", ak_priv
            ], check=True, capture_output=True, env=env)

            subprocess.run(["tpm2_load", "-C", "0x81000001", "-u", ak_pub, "-r", ak_priv, "-c", ak_ctx], check=True, capture_output=True, env=env)
            subprocess.run(["tpm2_evictcontrol", "-C", "o", "-c", ak_ctx, "0x81010002"], check=True, capture_output=True, env=env)

        except Exception as e:
            self.logger.error(f"TPMEngine: Provisioning error: {str(e)}", extra={"round": self.current_round})
        finally:
            self._flush_tpm_memory()
            for temp_file in [primary_ctx, ak_pub, ak_priv, ak_ctx]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass

    def _get_hardware_identity(self) -> str:
        env = self._get_tpm_env()
        try:
            res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002"], capture_output=True, text=True, check=True, env=env)
            for line in res.stdout.split('\n'):
                if line.strip().startswith("name:"):
                    return line.split("name:")[1].strip()
            return "UNKNOWN_HARDWARE_ID"
        except Exception:
            return "UNKNOWN_HARDWARE_ID"

    def generate_attestation_token(self, nonce: str, software_label: str, round_num: int) -> dict:
        """Generates a cryptographic quote sealing both the nonce and timestamp alongside the active PCR state."""
        env = self._get_tpm_env()
        session_id = uuid.uuid4().hex
        nonce_file = f"/tmp/nonce_{session_id}.bin"
        msg_file = f"/tmp/quote_{session_id}.msg"
        sig_file = f"/tmp/quote_{session_id}.sig"
        pcr_file = f"/tmp/pcr_{session_id}.bin"
        try:
            if self.insecure_mode:
                return {"status": "insecure_bypass", "IDi": software_label, "pcr_hash": "dummy_hash", "signature": "dummy_sig"}

            fog_num, edge_num = None, None
            m = re.search(r'\[EDGE (\d+)_(\d+)\]', software_label)
            if m:
                fog_num, edge_num = m.groups()
                edge_dir = f"/app/runtime/tpm_state/edge_{fog_num}_{edge_num}"
                os.makedirs(edge_dir, exist_ok=True)
                ak_path = f"{edge_dir}/ak.pub"

                if not os.path.exists(ak_path):
                    res = subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-f", "pem", "-o", ak_path], capture_output=True, env=env)
                    if res.returncode != 0:
                        subprocess.run(["tpm2_readpublic", "-c", "0x81010002", "-o", ak_path], check=True, env=env)

            self.logger.debug(f"[TPM-GENERATE] Edge Node {software_label} preparing to sign NONCE: {nonce}", extra={"round": round_num})

            hardware_idi = self._get_hardware_identity()
            token_timestamp = round(time.time(), 6)

            # Cryptographically bind the timestamp to the server nonce inside TPMS_ATTEST.extraData
            qualifying_hex = hashlib.sha256(f"{nonce}:{token_timestamp:.6f}".encode("utf-8")).hexdigest()
            qualifying_bytes = bytes.fromhex(qualifying_hex)

            self.logger.info(
                f"[TPM-GENERATE] Plaintext inputs mapped for hardware operation -> IDi: {hardware_idi}, Nonce: {nonce}, Timestamp: {token_timestamp:.6f}",
                extra={"round": round_num}
            )

            with open(nonce_file, "wb") as f:
                f.write(qualifying_bytes)

            subprocess.run([
                "tpm2_quote", "-c", "0x81010002", "-l", "sha256:0",
                "-q", nonce_file, "-m", msg_file,
                "-s", sig_file, "-o", pcr_file
            ], check=True, capture_output=True, env=env)

            self.logger.info("[TPM-GENERATE] The nonce, timestamp, and PCR value have been successfully hashed and signed together into the hardware quote structure.", extra={"round": round_num})

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
                "timestamp": token_timestamp,
            }
            self.logger.info(f"[TPM-GENERATE] Final Plaintext JSON Token prepared for transit: {token}", extra={"round": round_num})
            return token

        except Exception as e:
            self.logger.critical(f"[TPM-CRASH] Silent crash intercepted in generate_attestation_token! Error: {e}\nTraceback: {traceback.format_exc()}", extra={"round": round_num})
            return {"status": "error"}
        finally:
            try:
                for temp_file in [nonce_file, msg_file, sig_file, pcr_file]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except Exception:
                pass