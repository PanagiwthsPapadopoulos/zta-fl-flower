import os
import json
import logging
from typing import Tuple

from src.tier_fog.tpm_verifier import TPMVerifier
from src.tier_fog.trust_db import TrustDatabase


class ZeroTrustGatekeeper:
    """A security enforcement gateway for validating incoming update signatures and hardware attestation states against the ledger."""
    def __init__(self, logger: logging.Logger, log_prefix: str, run_metadata: dict):
        """Initializes the Zero Trust Gatekeeper, configuring the underlying TPM engine and trust database."""
        self.logger = logger
        self.log_prefix = log_prefix
        self.run_metadata = run_metadata
        self.tpm_state_root = "/app/runtime/tpm_state"
        
        try:
            insecure_flag = self.run_metadata.get("insecure", False) or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
            self.tpm_engine = TPMVerifier(logger=self.logger, insecure_mode=insecure_flag)
            self.trust_db = TrustDatabase(logger=self.logger) 
        except Exception as e:
            self.logger.error(f"{self.log_prefix} [GATEKEEPER] Initialization failed: {e}")
            self.tpm_engine = None
            self.trust_db = None

    def get_live_ledger(self) -> dict:
        """STRICT READ-ONLY SSOT: Reads the unified Admin ledger directly from the mounted Docker volume."""
        ledger_path = os.path.join(self.tpm_state_root, "pcr_ledger.json")
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"{self.log_prefix} [GATEKEEPER] SSOT Ledger is corrupted or empty!")
        else:
            self.logger.warning(f"{self.log_prefix} [GATEKEEPER] SSOT Ledger missing at {ledger_path}! Network locked down.")
        
        return {}

    def _get_folder_path(self, untrusted_log_prefix: str) -> str:
        """Constructs the expected directory path mapped to a specific edge node's localized TPM state."""
        edge_folder = untrusted_log_prefix.lower().replace('[', '').replace(']', '').replace(' ', '_')
        return os.path.join(self.tpm_state_root, edge_folder)

    def filter_node_updates(self, tier: str, server_round: int, results: list, active_nonces: dict) -> list:
        """Filters incoming client model updates based on real-time cryptographic attestation and ledger verification."""
        trusted_results = []
        current_ledger = self.get_live_ledger()
        
        for client_proxy, fit_res in results:
            if tier == "cloud":
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
                continue

            is_valid, tpm_id, display_identity = self._verify_single_node(
                client_proxy, fit_res, server_round, active_nonces, current_ledger
            )
            
            if is_valid:
                self.logger.info(f"{self.log_prefix} Received verified weights from {display_identity}", extra={"round": server_round})
                fit_res.metrics["tpm_id"] = tpm_id
                fit_res.metrics["display_identity"] = display_identity
                trusted_results.append((client_proxy, fit_res))
            else:
                self.logger.warning(f"{self.log_prefix} 🛑 REJECTED: Attestation/PCR mismatch for {display_identity}!", extra={"round": server_round})
                
        return trusted_results

    def _verify_single_node(self, client_proxy, fit_res, server_round, active_nonces, current_ledger: dict) -> Tuple[bool, str, str]:
        """Verifies a single node's cryptographic attestation token against the active nonce and shared read-only ledger."""
        untrusted_log_prefix = fit_res.metrics.get("log_prefix", f"CID_{client_proxy.cid}")
        tpm_token_json = fit_res.metrics.get("tpm_token_json", "")
        folder_path = self._get_folder_path(untrusted_log_prefix)
        
        if not tpm_token_json:
            return False, f"CID-{client_proxy.cid}", "Missing_Token"

        try:
            token = json.loads(tpm_token_json)
            
            id_file = os.path.join(folder_path, "tpm_id.txt")
            with open(id_file, "r") as f:
                expected_tpm_id = f.read().strip()
            
            tpm_id = token.get("IDi", str(client_proxy.cid))
            expected_pcr = current_ledger.get(expected_tpm_id)
            
            if not expected_pcr:
                self.logger.error(f"{self.log_prefix} 🛑 Unauthorized Device! ID {expected_tpm_id} is missing from the Admin ledger.")
                return False, tpm_id, untrusted_log_prefix
            
            pubkey_path = os.path.join(folder_path, "ak.pub")
            max_age = int(self.run_metadata.get("tpm_freshness_window", 300))
            
            authenticated = self.tpm_engine.verify_attestation_token(
                token=token, 
                expected_nonce=active_nonces.get(client_proxy.cid, ""), 
                public_key_path=pubkey_path, 
                expected_pcr=expected_pcr,
                max_age_seconds=max_age,
                round_num=server_round,
            )
            
            display_identity = f"{tpm_id} ({untrusted_log_prefix})"
            if self.trust_db:
                self.trust_db.process_attestation(tpm_id, display_identity, is_valid=authenticated, round_num=server_round)
            
            return authenticated, tpm_id, display_identity

        except Exception as e:
            self.logger.error(f"Stateless verification error for {untrusted_log_prefix}: {e}")
            return False, "UNKNOWN", untrusted_log_prefix