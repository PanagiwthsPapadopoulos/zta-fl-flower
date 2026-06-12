import os
import json
import logging
from typing import List, Tuple, Dict, Any

class ZeroTrustGatekeeper:
    """
    Handles hardware attestation and identity verification for federated edge nodes.
    Owns the TPM cryptographic engine and the stateful Trust Database.
    """
    def __init__(self, logger: logging.Logger, log_prefix: str, run_metadata: dict):
        self.logger = logger
        self.log_prefix = log_prefix
        self.run_metadata = run_metadata

        try:
            from src.security.attestation.tpm_core import TPMEngine
            from src.security.policy.trust_db import TrustDatabase
            
            insecure_flag = self.run_metadata.get("insecure", False) or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
            self.logger.debug(f"{self.log_prefix} [GATEKEEPER] Booting TPM Engine & Trust Database...")
            
            self.tpm_engine = TPMEngine(logger=self.logger, insecure_mode=insecure_flag)
            self.trust_db = TrustDatabase(logger=self.logger) 
            
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [GATEKEEPER] Non-fatal initialization crash: {e}")
            self.tpm_engine = None
            self.trust_db = None

    def filter_node_updates(self, tier: str, strategy_name: str, server_round: int, results: list, active_nonces: dict) -> list:
        """
        Interrogates inbound payloads. Applies hardware verification and quarantine 
        checks at the Fog tier, or acts as a passthrough for the Cloud tier.
        """
        trusted_results = []
        
        for client_proxy, fit_res in results:
            node_name = fit_res.metrics.get("node_name", f"Unknown CID: {client_proxy.cid}")
            
            # Cloud tier and non-ZTA strategies bypass hardware attestation
            if tier == "cloud" or strategy_name not in ["zta", "ztafl"]:
                self.logger.info(f"{self.log_prefix} 📥 Received weights from {node_name}", extra={"round": server_round})
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
                continue

            # --- Fog Tier Zero-Trust Enforcement ---
            if not self.trust_db or not self.tpm_engine:
                self.logger.warning(f"{self.log_prefix} [GATEKEEPER] Security modules offline. Bypassing checks for {node_name}.")
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
                continue

            # 1. Check Quarantine Status
            if self.trust_db.is_quarantined(node_name):
                self.logger.warning(f"{self.log_prefix} 🛑 REJECTED: Agent {node_name} is currently quarantined in TrustDB!", extra={"round": server_round})
                continue
            
            # 2. Verify Cryptographic Attestation Token Structure
            tpm_token_json = fit_res.metrics.get("tpm_token_json", "")
            authenticated = False
            
            if tpm_token_json:
                try:
                    token = json.loads(tpm_token_json)
                    expected_nonce = active_nonces.get(client_proxy.cid, "")
                    pubkey_path = f"/app/data/tpm_state/{node_name.lower().replace('[', '').replace(']', '').replace(' ', '_')}/ak.pub"
                    authenticated = self.tpm_engine.verify_attestation_token(token, expected_nonce, pubkey_path)
                except Exception as e:
                    self.logger.error(f"{self.log_prefix} Error extracting TPM data payload for {node_name}: {str(e)}")
            
            # 3. Apply TrustDB State Updates (Fixed to use process_attestation)
            if not authenticated:
                self.logger.warning(f"{self.log_prefix} 🔒 ATTESTATION FAILED: Agent {node_name} failed software integrity check!", extra={"round": server_round})
                self.trust_db.process_attestation(node_name, is_valid=False)
                continue
                
            self.trust_db.process_attestation(node_name, is_valid=True)
            self.logger.info(f"{self.log_prefix} Received weights from {node_name}", extra={"round": server_round})
            
            if fit_res.num_examples > 0:
                trusted_results.append((client_proxy, fit_res))
                
        return trusted_results