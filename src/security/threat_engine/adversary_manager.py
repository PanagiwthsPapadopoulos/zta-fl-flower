import os
import json
import logging
import numpy as np
import torch

class AdversaryManager:
    """
    Autonomous threat orchestration engine using integer coordinate mapping.
    """
    def __init__(self, fog_num: int, edge_num: int, logger: logging.Logger):
        self.fog_num = fog_num
        self.edge_num = edge_num
        self.log_prefix = f"[EDGE {fog_num}_{edge_num}]" 
        
        self.logger = logger
        self.attack_type = None
        self.activate_on_round = 9999
        self.current_round = 0
        self.target_victim = None 
        
        self.cache_file = f"/tmp/stolen_token_{self.fog_num}_{self.edge_num}.json"
        self._load_threat_profile()

    def _load_threat_profile(self):
        try:
            from src.utils.config_loader import load_yaml_configs
            config = load_yaml_configs()
            adversaries = config.get("adversaries", [])
            
            for adv in adversaries:
                if adv.get("fog_num") == self.fog_num and adv.get("edge_num") == self.edge_num:
                    if not adv.get("enabled", True): 
                        self.logger.info(f"{self.log_prefix} Threat profile found but DISABLED in config.")
                        continue
                        
                    self.attack_type = adv.get("attack_type")
                    self.activate_on_round = adv.get("activate_on_round", 1)
                    
                    if self.attack_type == "identity_theft":
                        v_fog = adv.get("target_victim_fog", self.fog_num)
                        v_edge = adv.get("target_victim_edge", 1)
                        self.target_victim = f"[EDGE {v_fog}_{v_edge}]"

                    self.logger.warning(f"🚨 ADVERSARY PROFILE LOADED: {self.attack_type.upper()} scheduled for Round {self.activate_on_round}")
                    break
        except Exception as e:
            self.logger.error(f"AdversaryManager failed to load profile: {e}")

    def corrupt_data_if_needed(self, trainloader):
        if self.attack_type == "label_flipping":
            manager = self 
            class LabelFlippingLoader:
                def __init__(self, loader): self.loader = loader
                def __iter__(self):
                    import torch
                    for x, y in self.loader:
                        if manager.current_round >= manager.activate_on_round:
                            manager.logger.error(f"{manager.log_prefix} ☠️ EXECUTING DATA POISONING: Flipping labels to 0")
                            yield x, torch.zeros_like(y)
                        else:
                            yield x, y
                def __len__(self): return len(self.loader)
            return LabelFlippingLoader(trainloader)
        return trainloader

    def corrupt_payload_if_needed(self, parameters: list, metrics: dict) -> tuple:
        if not self.attack_type:
            return parameters, metrics

        # ---------------------------------------------------------
        # ATTACK: TPM Replay
        # ---------------------------------------------------------
        if self.attack_type == "tpm_replay":
            if self.current_round < self.activate_on_round:
                self.logger.debug(f"{self.log_prefix} Hoarding valid token to disk for future replay attack...")
                token = metrics.get("tpm_token_json")
                if token:
                    with open(self.cache_file, "w") as f:
                        f.write(token)
            else:
                self.logger.error(f"{self.log_prefix} ☠️ EXECUTING REPLAY ATTACK: Injecting stale token from disk.")
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "r") as f:
                        metrics["tpm_token_json"] = f.read()

        # ---------------------------------------------------------
        # ATTACK: TPM Forgery (Man-in-the-Middle)
        # ---------------------------------------------------------
        elif self.attack_type == "tpm_forgery" and self.current_round >= self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING TPM FORGERY: Corrupting RSA signature payload.")
            token_str = metrics.get("tpm_token_json")
            if token_str:
                try:
                    token = json.loads(token_str)
                    sig = token.get("signature", "")
                    # Break the RSA math
                    if len(sig) > 5:
                        token["signature"] = sig[:-5] + "XXXXX"
                    metrics["tpm_token_json"] = json.dumps(token)
                except Exception as e: pass

        # ---------------------------------------------------------
        # 🚨 NEW ATTACK: OS Malware / PCR Alteration
        # ---------------------------------------------------------
        elif self.attack_type == "pcr_alteration" and self.current_round >= self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING PCR ALTERATION: Simulating Rootkit/Malware Infection in OS hashes.")
            token_str = metrics.get("tpm_token_json")
            if token_str:
                try:
                    token = json.loads(token_str)
                    pcr = token.get("pcr_data", "")
                    # Alter the OS state hash. The RSA signature will STILL be valid!
                    if len(pcr) > 7:
                        token["pcr_data"] = pcr[:-7] + "MALWARE"
                    metrics["tpm_token_json"] = json.dumps(token)
                except Exception as e: pass

        # ---------------------------------------------------------
        # ATTACK: Model Poisoning (Software Hijacking)
        # ---------------------------------------------------------
        elif self.attack_type == "model_poisoning" and self.current_round >= self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING MODEL POISONING: Injecting randomized Gaussian weights.")
            parameters = [np.random.normal(0, 5, size=p.shape).astype(p.dtype) for p in parameters]

        # ---------------------------------------------------------
        # ATTACK: Identity Theft
        # ---------------------------------------------------------
        elif self.attack_type == "identity_theft" and self.current_round >= self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING IDENTITY THEFT: Spoofing {self.target_victim}'s keys and injecting poisoned weights.")
            parameters = [np.random.normal(0, 5, size=p.shape).astype(p.dtype) for p in parameters]
            token = {
                "status": "simulated_key_theft",
                "target_victim": self.target_victim
            }
            metrics["tpm_token_json"] = json.dumps(token)
            metrics["log_prefix"] = self.target_victim

        return parameters, metrics