import json
import os
import logging


class AdminConsole:
    """An administrative interface for managing targeted operational parameters and simulating hardware state updates."""
    def __init__(self, gatekeeper, logger: logging.Logger):
        self.gatekeeper = gatekeeper
        self.logger = logger
        try:
            from src.shared.utils.config_loader import load_yaml_configs
            self.scheduled_actions = load_yaml_configs().get("admin_actions", [])
        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] Failed to load admin config: {e}")
            self.scheduled_actions = []

    def execute_scheduled_updates(self, current_round: int, fog_num: int):
        for action in self.scheduled_actions:
            if action.get("activate_on_round") == current_round:
                if action.get("enabled", True) is False: 
                    continue
                atype = action.get("action_type")
                if atype in ["pcr_legit_update", "pcr_fake_update"] and action.get("fog_num") == fog_num:
                    if atype == "pcr_fake_update":
                        self.logger.warning(f"[ADMIN CONSOLE] Injecting malicious firmware to physical environment.")
                    else:
                        self.logger.info(f"[ADMIN CONSOLE] Deploying legitimate firmware update to physical environment.")
                    self._process_physical_update(action.get("target_edge"), fog_num)

    def _process_physical_update(self, edge_num: int, fog_num: int):
        dirname = f"edge_{fog_num}_{edge_num}"
        ledger_path = f"/app/runtime/tpm_state/pcr_ledger.json"
        id_path = f"/app/runtime/tpm_state/{dirname}/tpm_id.txt"
        
        try:
            with open(id_path, 'r', encoding='utf-8') as file:
                tpm_id = file.read().strip() 
            with open(ledger_path, 'r', encoding='utf-8') as file:
                pcr_ledger_contents = json.load(file) 
            
            if tpm_id not in pcr_ledger_contents:
                raise KeyError(f"Target key '{tpm_id}' was not found in the JSON ledger.")
            
            pcr_ledger_contents[tpm_id] = "INJECTED PCR DURING TRAINING"
            with open(ledger_path, 'w', encoding='utf-8') as ledger_file:
                json.dump(pcr_ledger_contents, ledger_file, indent=4)
            self.logger.info(f"[ADMIN CONSOLE] 💉 Pushed new PCR for {tpm_id[:16]}... to the PCR ledger.")
        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] 🛑 Failed to physically update environment: {e}")