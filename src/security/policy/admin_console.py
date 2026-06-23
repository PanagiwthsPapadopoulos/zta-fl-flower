import json
import os
import logging

class AdminConsole:
    """An administrative interface for managing targeted operational parameters and simulating hardware state updates."""
    def __init__(self, gatekeeper, logger: logging.Logger):
        """Initializes the Admin Console and loads any scheduled administrative actions from configurations."""
        self.gatekeeper = gatekeeper
        self.logger = logger
        
        try:
            from src.utils.config_loader import load_yaml_configs
            config = load_yaml_configs()
            self.scheduled_actions = config.get("admin_actions", [])
        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] Failed to load admin config: {e}")
            self.scheduled_actions = []

    def execute_scheduled_updates(self, current_round: int, fog_num: int):
        """Executes scheduled firmware alterations or updates based on the current synchronized network round."""
        for action in self.scheduled_actions:
            if action.get("activate_on_round") == current_round:
                if action.get("enabled", True) is False: continue
                
                atype = action.get("action_type")
                    
                if atype == "pcr_legit_update" and action.get("fog_num") == fog_num:
                    self.logger.info(f"[ADMIN CONSOLE] Deploying legitimate firmware update to physical environment.")
                    self._process_physical_update(action.get("target_edge"), fog_num)
                    
                elif atype == "pcr_fake_update" and action.get("fog_num") == fog_num:
                    self.logger.warning(f"[ADMIN CONSOLE] Injecting malicious firmware to physical environment. Node will fail attestation.")
                    self._process_physical_update(action.get("target_edge"), fog_num)

    def _process_physical_update(self, edge_num: int, fog_num: int):
        """
        Simulates pushing a firmware update to the physical edge node directory.
        CRITICAL: This alters the ledger. The file remains read-only for FOG nodes.
        """
        dirname = f"edge_{fog_num}_{edge_num}"
        ledger_path = f"/app/runtime/tpm_state/pcr_ledger.json"
        id_path = f"/app/runtime/tpm_state/{dirname}/tpm_id.txt"
        
        # Read the TPM id
        try:
            with open(id_path, 'r', encoding='utf-8') as file:
                tpm_id = file.read().strip() 
        except FileNotFoundError:
            self.logger.error(f"[ADMIN CONSOLE] 🛑 Error: '{id_path}' was not found.")
            return 

        # Read the PCR_ledger
        try:
            with open(ledger_path, 'r', encoding='utf-8') as file:
                pcr_ledger_contents = json.load(file) 
        except FileNotFoundError:
            self.logger.error(f"[ADMIN CONSOLE] 🛑 Error: '{ledger_path}' was not found.")
            return 
        except json.JSONDecodeError:
            self.logger.error(f"[ADMIN CONSOLE] 🛑 Error: '{ledger_path}' contains invalid JSON.")
            return 

        try:
            # Check if the specific key exists at this level
            if tpm_id not in pcr_ledger_contents:
                raise KeyError(f"Target key '{tpm_id}' was not found in the JSON ledger.")
            
            # Modify the value of the targeted key
            pcr_ledger_contents[tpm_id] = "INJECTED PCR DURING TRAINING"

            # Dump the dictionary back to the file properly formatted
            with open(ledger_path, 'w', encoding='utf-8') as ledger_file:
                json.dump(pcr_ledger_contents, ledger_file, indent=4)

            self.logger.info(f"[ADMIN CONSOLE] 💉 Pushed new PCR for {tpm_id[:16]}... to the PCR ledger.")
            
        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] 🛑 Failed to physically update environment: {e}")

   