import os
import json
import logging
import numpy as np


class AdversaryManager:
    """
    An autonomous orchestration controller for mapping and triggering scheduled client-side adversarial attacks. 
    This is not part of the original article, it is part of an extensive testing suite.
    """
    def __init__(self, fog_num: int, edge_num: int, logger: logging.Logger):
        """Initializes the Adversary Manager and preemptively loads the designated threat profile for the current node."""
        self.fog_num = fog_num
        self.edge_num = edge_num
        self.log_prefix = f"[EDGE {fog_num}_{edge_num}]" 
        
        self.logger = logger
        self.attack_type = None
        self.activate_on_round = 9999
        # This gets overwritten in client.py at the start of each round
        self.current_round = 0
        self.target_victim = None 
        self.cache_file = f"/tmp/tpm_cache_{fog_num}_{edge_num}.json"
        
        self._load_threat_profile()

    def _load_threat_profile(self):
        """Loads the configured threat profile parameters and activation schedules from the application configuration."""
        try:
            from src.shared.utils.config_loader import load_yaml_configs
            config = load_yaml_configs()
            adversaries = config.get("adversaries", [])
            
            for adv in adversaries:
                if adv.get("fog_num") == self.fog_num and adv.get("edge_num") == self.edge_num:
                    if not adv.get("enabled", True): 
                        # self.logger.info(f"{self.log_prefix} Threat profile found but DISABLED in config.", extra={"round": self.current_round})
                        continue
                        
                    self.attack_type = adv.get("attack_type")
                    self.activate_on_round = adv["activate_on_round"]
                    
                    v_fog = adv.get("target_victim_fog", self.fog_num)
                    v_edge = adv.get("target_victim_edge", 1)
                    self.target_victim = f"[EDGE {v_fog}_{v_edge}]"

                    self.logger.warning(f"🚨 ADVERSARY PROFILE LOADED: {self.attack_type.upper()} scheduled for Round {self.activate_on_round}", extra={"round": self.current_round})
                    break
        except Exception as e:
            self.logger.error(f"AdversaryManager failed to load profile: {e}", extra={"round": self.current_round})

    def corrupt_data_if_needed(self, trainloader):
        """Wraps the honest local data loader with a malicious transformation loader if data poisoning is actively scheduled."""
        if self.attack_type == "label_flipping":
            manager = self 
            class LabelFlippingLoader:
                def __init__(self, loader): self.loader = loader
                def __iter__(self):
                    import torch
                    for x, y in self.loader:
                        if manager.current_round == manager.activate_on_round:
                            manager.logger.error(f"{manager.log_prefix} ☠️ EXECUTING DATA POISONING: Flipping labels to 0", extra={"round": self.current_round})
                            yield x, torch.zeros_like(y)
                        else:
                            yield x, y
                def __len__(self): return len(self.loader)
            return LabelFlippingLoader(trainloader)
        return trainloader

    def corrupt_payload_if_needed(self, parameters: list, metrics: dict) -> tuple:
        """Injects malicious modifications into the outgoing model parameters or TPM token payload depending on active attacks."""
        if not self.attack_type:
            return parameters, metrics

        if self.attack_type == "tpm_replay":
            if self.current_round == self.activate_on_round:
                self.logger.debug(f"{self.log_prefix} Hoarding valid token to disk for future replay attack...", extra={"round": self.current_round})
                token = metrics.get("tpm_token_json")
                if token:
                    with open(self.cache_file, "w") as f:
                        f.write(token)
            else:
                self.logger.error(f"{self.log_prefix} ☠️ EXECUTING REPLAY ATTACK: Injecting stale token from disk.", extra={"round": self.current_round})
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "r") as f:
                        metrics["tpm_token_json"] = f.read()

        elif self.attack_type == "tpm_forgery" and self.current_round == self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING TPM FORGERY: Corrupting RSA signature payload.", extra={"round": self.current_round})
            token_str = metrics.get("tpm_token_json")
            if token_str:
                try:
                    token = json.loads(token_str)
                    sig = token.get("signature", "")
                    if len(sig) > 5:
                        token["signature"] = sig[:-5] + "XXXXX"
                    metrics["tpm_token_json"] = json.dumps(token)
                except Exception: pass

        elif self.attack_type == "pcr_alteration" and self.current_round == self.activate_on_round:
            token_str = metrics.get("tpm_token_json")
            if token_str:
                try:
                    token = json.loads(token_str)
                    token["pcr_data"] = "INJECTED PCR DURING TRAINING"

                    if self.current_round == self.activate_on_round:
                        self.logger.info(f"{self.log_prefix} Executing PCR alteration. New PCR value: \"INJECTED PCR DURING TRAINING\". ", extra={"round": self.current_round})
                        
                    metrics["tpm_token_json"] = json.dumps(token)
                except Exception: pass

        elif self.attack_type == "model_poisoning" and self.current_round == self.activate_on_round:
            self.logger.error(f"{self.log_prefix} ☠️ EXECUTING MODEL POISONING: Injecting randomized Gaussian weights.", extra={"round": self.current_round})
            parameters = [np.random.normal(0, 5, size=p.shape).astype(p.dtype) for p in parameters]

        return parameters, metrics