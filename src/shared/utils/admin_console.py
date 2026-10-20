import hashlib
import json
import os
import logging
import subprocess


class AdminConsole:
    """An administrative interface for managing targeted operational parameters and simulating hardware state updates."""

    def __init__(self, gatekeeper, logger: logging.Logger):
        self.gatekeeper = gatekeeper
        self.logger = logger
        try:
            from src.shared.utils.config_loader import load_yaml_configs
            self.scheduled_actions = load_yaml_configs().get("admin_actions", [])
        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] Failed to load admin config: {e}", extra={"round": 0})
            self.scheduled_actions = []

    def execute_scheduled_updates(self, current_round: int, fog_num: int):
        for action in self.scheduled_actions:
            if action.get("activate_on_round") == current_round:
                if action.get("enabled", True) is False:
                    continue
                atype = action.get("action_type")
                if atype in ["pcr_legit_update", "pcr_fake_update"] and action.get("fog_num") == fog_num:
                    is_legit = (atype == "pcr_legit_update")
                    if not is_legit:
                        self.logger.warning("[ADMIN CONSOLE] Injecting malicious firmware to physical environment.", extra={"round": current_round})
                    else:
                        self.logger.info("[ADMIN CONSOLE] Deploying legitimate signed firmware update to physical environment.", extra={"round": current_round})
                    self._process_physical_update(action.get("target_edge"), fog_num, current_round, update_ledger=is_legit)

    def _process_physical_update(self, edge_num: int, fog_num: int, current_round: int = 0, update_ledger: bool = True):
        dirname = f"edge_{fog_num}_{edge_num}"
        ledger_path = "/app/runtime/tpm_state/pcr_ledger.json"
        id_path = f"/app/runtime/tpm_state/{dirname}/tpm_id.txt"
        pcr_bin_path = f"/app/runtime/tpm_state/{dirname}/clean_pcr.bin"

        try:
            with open(id_path, 'r', encoding='utf-8') as file:
                tpm_id = file.read().strip()

            with open(ledger_path, 'r', encoding='utf-8') as file:
                pcr_ledger_contents = json.load(file)

            if tpm_id not in pcr_ledger_contents:
                raise KeyError(f"Target key '{tpm_id}' was not found in the JSON ledger.")

            # Extend the target edge emulator's PCR 0 with a firmware measurement
            firmware_digest = "11111111111111111111111111111111111111111111111111111111cafebabe"
            env = os.environ.copy()
            env["TPM2TOOLS_TCTI"] = f"swtpm:host=tpm-{fog_num}-{edge_num},port=2321"
            ext_res = subprocess.run(
                ["tpm2_pcrextend", f"0:sha256={firmware_digest}"],
                capture_output=True, text=True, env=env
            )

            # Read the updated PCR 0 value to compute the new TPMS_ATTEST pcrDigest (SHA-256 of selected PCRs)
            new_pcr_digest_hex = None
            if ext_res.returncode == 0:
                read_res = subprocess.run(
                    ["tpm2_pcrread", "sha256:0", "-o", pcr_bin_path],
                    capture_output=True, text=True, env=env
                )
                if read_res.returncode == 0 and os.path.exists(pcr_bin_path):
                    with open(pcr_bin_path, "rb") as bf:
                        new_pcr_digest_hex = hashlib.sha256(bf.read()).hexdigest()

            if update_ledger:
                if new_pcr_digest_hex:
                    pcr_ledger_contents[tpm_id] = new_pcr_digest_hex
                with open(ledger_path, 'w', encoding='utf-8') as ledger_file:
                    json.dump(pcr_ledger_contents, ledger_file, indent=4)
                self.logger.info(f"[ADMIN CONSOLE] Verified manifest and pushed updated PCR digest for {tpm_id[:16]}... to the PCR ledger.", extra={"round": current_round})
            else:
                if not new_pcr_digest_hex:
                    # Fallback if swtpm is on a remote host: corrupt ledger reference with a mismatched valid hex digest
                    pcr_ledger_contents[tpm_id] = hashlib.sha256(b"UNAUTHORIZED_FIRMWARE_DRIFT").hexdigest()
                    with open(ledger_path, 'w', encoding='utf-8') as ledger_file:
                        json.dump(pcr_ledger_contents, ledger_file, indent=4)
                self.logger.warning(f"[ADMIN CONSOLE] Unverified firmware drift applied to {tpm_id[:16]}... without ledger authorization.", extra={"round": current_round})

        except Exception as e:
            self.logger.error(f"[ADMIN CONSOLE] Failed to physically update environment: {e}", extra={"round": current_round})