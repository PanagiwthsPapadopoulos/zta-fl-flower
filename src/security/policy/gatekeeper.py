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
        self.identity_map = {} 
        self.pcr_baselines: Dict[str, str] = {} # 🚨 The Authoritative PCR Registry

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

    def _get_or_set_baseline(self, tpm_id: str, received_pcr: str) -> str:
        """
        Implements deterministic baseline registration. 
        In production, this would be loaded from a signed manufacturer manifest.
        """
        if tpm_id not in self.pcr_baselines:
            self.logger.info(f"{self.log_prefix} [GATEKEEPER] Establishing initial PCR baseline for {tpm_id[:16]}...")
            self.pcr_baselines[tpm_id] = received_pcr
        return self.pcr_baselines[tpm_id]

    def filter_node_updates(self, tier: str, strategy_name: str, server_round: int, results: list, active_nonces: dict) -> list:
        trusted_results = []
        
        for client_proxy, fit_res in results:
            untrusted_log_prefix = fit_res.metrics.get("log_prefix", f"CID_{client_proxy.cid}")
            
            if tier == "cloud" or strategy_name not in ["zta", "ztafl"]:
                if fit_res.num_examples > 0:
                    trusted_results.append((client_proxy, fit_res))
                continue

            tpm_token_json = fit_res.metrics.get("tpm_token_json", "")
            authenticated = False
            tpm_id = f"CID-{client_proxy.cid}" 
            
            if tpm_token_json:
                try:
                    token = json.loads(tpm_token_json)
                    
                    # Handle simulated key theft vs normal attestation
                    if token.get("status") == "simulated_key_theft":
                        target_victim = token.get("target_victim")
                        tpm_id = self.identity_map.get(target_victim, f"STOLEN_KEY_{target_victim}")
                        expected_pcr = self.pcr_baselines.get(tpm_id, "BASELINE_ERROR")
                        authenticated = self.tpm_engine.verify_attestation_token(token, "", "", server_round, expected_pcr)
                    else:
                        tpm_id = token.get("IDi", tpm_id)
                        self.identity_map[untrusted_log_prefix] = tpm_id
                        
                        # 🚨 GATEKEEPER: Perform PCR comparison
                        actual_pcr = token.get("pcr_data", "")
                        expected_pcr = self._get_or_set_baseline(tpm_id, actual_pcr)
                        
                        expected_nonce = active_nonces.get(client_proxy.cid, "")
                        pubkey_path = f"/app/runtime/tpm_state/{untrusted_log_prefix.lower().replace('[', '').replace(']', '').replace(' ', '_')}/ak.pub"
                        
                        authenticated = self.tpm_engine.verify_attestation_token(
                            token=token, 
                            expected_nonce=expected_nonce, 
                            public_key_path=pubkey_path, 
                            round_num=server_round, 
                            expected_pcr=expected_pcr
                        )
                except Exception as e:
                    self.logger.error(f"{self.log_prefix} PCR Enforcement error: {e}", extra={"round": server_round})
            
            display_identity = f"{tpm_id} ({untrusted_log_prefix})"
            self.trust_db.process_attestation(tpm_id, display_identity, is_valid=authenticated, round_num=server_round)
            
            if authenticated:
                self.logger.info(f"{self.log_prefix} Received verified weights from {display_identity}", extra={"round": server_round})
                fit_res.metrics["tpm_id"] = tpm_id
                fit_res.metrics["display_identity"] = display_identity
                trusted_results.append((client_proxy, fit_res))
            else:
                self.logger.warning(f"{self.log_prefix} 🛑 REJECTED: Attestation/PCR mismatch for {display_identity}!", extra={"round": server_round})
                
        return trusted_results