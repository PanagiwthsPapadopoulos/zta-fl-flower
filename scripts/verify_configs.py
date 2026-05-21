import os
import json
import ast
import re
import sys
import tomllib
from collections import defaultdict

# Enumerable set detailing parameters unrelated to training mechanics.
IGNORE_VARS = {
    "broker_ip", "socket_timeout", "cloud_sa_port", "cloud_fl_port", 
    "cloud_ctrl_port", "fog_sa_base", "fog_fl_base", "fog_ctrl_base", 
    "fog_client_io_base", "fog_ipc_base", "edge_client_io_base",
    "tier", "fog_id", "min-clients", "num_fogs", "uniform_edges_per_fog", 
    "custom_fog_topology", "run_name", "num_rounds", "robustness_eval_attack",
    "power_law_a"
}

# Enumerable set isolating conditional parameters tied exclusively to explicit execution paths.
CONDITIONAL_VARS = {
    "fedprox_mu", "krum_f", "trimmed_mean_beta", "flame_target_frac",
    "shap_val_samples", "shap_explain_count", "p_flip", "gradient_alpha",
    "backdoor_poison_fraction", "backdoor_target_class", "backdoor_trigger_features",
    "backdoor_trigger_value", "pgd_adv_ratio", "pgd_eps", "pgd_alpha", "pgd_n_iter",
    "fgsm_adv_ratio", "fgsm_eps", "fgsm_alpha", "benign_adv_ratio", "benign_eps",
    "benign_alpha", "benign_n_iter", "shap_aware_base_attack", "shap_tau"
}

def load_toml_config(toml_path: str = "pyproject.toml") -> dict:
    """
    Parses the primary project configuration definition translating block syntax into internal python maps.
    """
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
    except Exception as e:
        print(f"❌ Error loading {toml_path}: {e}")
        return {}

def crawl_network_logs(log_dir: str = "logs/nodes"):
    """
    Investigates output node activity tracing declared environmental usage to ensure system parity.
    Extracts defined telemetry markers and builds a tracking record array.
    """
    global_vars = defaultdict(set)
    
    if not os.path.exists(log_dir):
        print(f"❌ Error: Log directory '{log_dir}' does not exist. Run the network first.")
        return {}

    for filename in os.listdir(log_dir):
        if not filename.endswith(".jsonl"):
            continue
            
        with open(os.path.join(log_dir, filename), "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", "")
                    
                    if "[CONFIG USAGE]" in msg and "|" in msg:
                        kv_string = msg.split("|")[1].strip()
                        
                        # Bypasses structural splits enclosed inside native collection lists.
                        pairs = re.split(r',\s*(?![^\[]*\])', kv_string)
                        
                        for pair in pairs:
                            if ":" not in pair: 
                                continue
                            
                            k, v = pair.split(":", 1)
                            k, v = k.strip(), v.strip()
                            
                            # Standardizes variable assignments reverting string cast types.
                            try:
                                val = float(v) if '.' in v else int(v)
                            except ValueError:
                                if v == "True": val = True
                                elif v == "False": val = False
                                elif v == "None": continue
                                elif v.startswith("[") and v.endswith("]"):
                                    try: val = tuple(ast.literal_eval(v))
                                    except: val = v
                                else:
                                    val = v
                                    
                            global_vars[k].add(val)
                except Exception:
                    continue
                    
    return global_vars

def audit_network(toml_config: dict, log_config: dict):
    """
    Iterates compiled operational states verifying adherence directly alongside structural source directives.
    Yields output blocks matching discrepancies across defined deployment stages.
    """
    print("\n==================================================")
    print(" 🌐 GLOBAL NETWORK CONFIGURATION AUDIT")
    print("==================================================\n")
    
    errors = 0
    success = 0
    
    audit_targets = {k: v for k, v in toml_config.items() if k not in IGNORE_VARS}
    
    for key, toml_val in audit_targets.items():
        if key not in log_config:
            if key in CONDITIONAL_VARS:
                print(f"ℹ️  SKIPPED: '{key}' (Unused by the active strategy/role)")
                continue
            else:
                print(f"⚠️  MISSING: '{key}' is in pyproject.toml but was never loaded by the network.")
                errors += 1
                continue
            
        log_values = log_config[key]
        
        # Conforms list mapping logic directly interpreting standard notation variants.
        if isinstance(toml_val, str) and toml_val.startswith("[") and toml_val.endswith("]"):
            try:
                toml_val = tuple(ast.literal_eval(toml_val))
            except Exception:
                pass
        elif isinstance(toml_val, list):
            toml_val = tuple(toml_val)
            
        if len(log_values) > 1 and key not in {"random_seed"}:
            print(f"❌ CONFLICT: Nodes loaded conflicting values for '{key}': {log_values}")
            errors += 1
            continue
            
        # Coordinates comparative bounds checking ensuring absolute configuration adherence.
        match_found = False
        for log_val in log_values:
            if type(toml_val) in [int, float] and type(log_val) in [int, float]:
                if float(toml_val) == float(log_val):
                    match_found = True
                    break
            elif str(toml_val) == str(log_val):
                match_found = True
                break
                
        if not match_found:
            print(f"❌ MISMATCH: '{key}' | TOML wants {toml_val} -> Network used {log_values}")
            errors += 1
        else:
            success += 1

    print("\n==================================================")
    if errors == 0:
        print(f"✅ PASS: All {success} active ML hyperparameters perfectly match pyproject.toml!")
    else:
        print(f"⚠️  FAIL: Found {errors} configuration discrepancies.")
    print("==================================================\n")

if __name__ == "__main__":
    toml_dict = load_toml_config()
    if toml_dict:
        network_dict = crawl_network_logs()
        if network_dict:
            audit_network(toml_dict, network_dict)