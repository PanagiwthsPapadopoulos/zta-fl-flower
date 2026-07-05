import os
import json
import glob
import re
import copy

LOG_DIR = "logs/nodes" # Adapt to your actual path

def parse_logs():
    """
    Chronologically scans the JSONL logs and rebuilds a complete snapshot of the 
    ZTA-FL network state for every single round to enable 'Time Travel' viewing.
    """
    if not os.path.exists(LOG_DIR):
        return {}

    all_logs = []
    log_files = glob.glob(os.path.join(LOG_DIR, "*.jsonl"))
    
    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    # Sort by Round FIRST, then timestamp to neutralize timezone drift across nodes
    all_logs.sort(key=lambda x: (x.get("round", 0), x.get("timestamp", "")))

    state_history = {}
    current_state = {
        "cloud_status": "Idle",
        "cloud_accuracy": None,
        "nodes": {},
        "trust_db": {},
        "tokens": [],
        "params": {
            "network": {},
            "cloud": {},
            "fog": {},
            "edge": {}
        }
    }

    for entry in all_logs:
        raw_node_id = entry.get("node", "UNKNOWN")
        msg = entry.get("message", "")
        rnd = entry.get("round", 0)
        
        # --- SPLIT-BRAIN UNIFICATION FIX ---
        if "FOG" in raw_node_id:
            m = re.search(r"FOG (\d+)", raw_node_id)
            node_id = f"[FOG {m.group(1)}]" if m else raw_node_id
        elif "CLOUD" in raw_node_id:
            node_id = "[CLOUD]"
        else:
            node_id = raw_node_id
        
        # Detect Round Boundary & Reset Phase Statuses
        if rnd not in state_history:
            new_state = copy.deepcopy(current_state)
            new_state["cloud_status"] = "Idle"
            new_state["cloud_accuracy"] = None
            for n in new_state["nodes"].values():
                n["Status"] = "Idle / Awaiting Cloud" if n["Type"] == "fog" else "Idle"
            state_history[rnd] = new_state

        active_state = state_history[rnd]

        # Initialize node in state if not exists
        if node_id not in active_state["nodes"]:
            node_type = "cloud" if "CLOUD" in node_id else "fog" if "FOG" in node_id else "edge"
            active_state["nodes"][node_id] = {
                "ID": node_id,
                "Type": node_type,
                "Status": "Idle",
                "Role": "Honest", 
                "Logs": [],
                "runtime_params": {}
            }

        # Keep last 5 logs for live trackers
        active_state["nodes"][node_id]["Logs"].append(f"[{entry['timestamp']}] {raw_node_id} {msg}")
        if len(active_state["nodes"][node_id]["Logs"]) > 5:
            active_state["nodes"][node_id]["Logs"].pop(0)

        # --- DYNAMIC PARAMETER EXTRACTION (Global Configs) ---
        if "[CONFIG USAGE]" in msg and "|" in msg:
            params_str = msg.split("|", 1)[1]
            matches = re.findall(r'([a-zA-Z0-9_]+):\s*(\[.*?\]|[^,]+)', params_str)
            
            for k, v in matches:
                k, v = k.strip(), v.strip()
                if k in ["dataset", "dataset_path", "dataset_fraction", "model_architecture", "quantization_bits", "random_seed", "num_classes", "n_features", "simulate_global_leakage"]:
                    active_state["params"]["network"][k] = v
                elif k in ["strategy", "rollback_threshold", "min_clients", "tier"]:
                    active_state["params"]["cloud"][k] = v
                elif k in ["shap_explain_count", "shap_threshold", "shap_val_samples", "fog_num"]:
                    active_state["params"]["fog"][k] = v
                else:
                    active_state["params"]["edge"][k] = v

        # --- CLOUD PARSING ---
        if "CLOUD" in node_id:
            if "Shouting to all FOG clients" in msg:
                active_state["cloud_status"] = "Broadcasting"
            elif "Executing" in msg and "aggregation" in msg:
                active_state["cloud_status"] = "Aggregating"
            elif "Saved metrics and model" in msg:
                active_state["cloud_status"] = "Round Complete"
                
            acc_match = re.search(r"acc(?:uracy)?.*?([0-9]*\.[0-9]+)", msg, re.IGNORECASE)
            if acc_match:
                active_state["cloud_accuracy"] = float(acc_match.group(1))

        # --- FOG PARSING ---
        elif "FOG" in node_id:
            if "Expecting clients" in msg:
                active_state["nodes"][node_id]["Status"] = "Idle / Awaiting Cloud"
            elif "START received" in msg:
                active_state["nodes"][node_id]["Status"] = "Propagating to Edges"
            elif "evaluating token" in msg:
                active_state["nodes"][node_id]["Status"] = "Receiving Tokens"
            elif "Received verified weights from" in msg:
                active_state["nodes"][node_id]["Status"] = "Verifying & TrustDB Check"
                edge_match = re.search(r"\[(EDGE \d+_\d+)\]", msg)
                if edge_match:
                    e_id = f"[{edge_match.group(1)}]"
                    if e_id in active_state["nodes"]:
                        active_state["nodes"][e_id]["Status"] = "Round Complete"
            elif "REJECTED: Attestation/PCR mismatch" in msg:
                active_state["nodes"][node_id]["Status"] = "Verifying & TrustDB Check"
                edge_match = re.search(r"\[(EDGE \d+_\d+)\]", msg)
                if edge_match:
                    e_id = f"[{edge_match.group(1)}]"
                    if e_id in active_state["nodes"]:
                        active_state["nodes"][e_id]["Status"] = "Rejected"
            elif "Executing" in msg and "aggregation" in msg:
                active_state["nodes"][node_id]["Status"] = "SHAP Aggregation"
            elif "Relay successful" in msg:
                active_state["nodes"][node_id]["Status"] = "Round Complete"
                
            if "TrustDB: Agent" in msg and "rewarded" in msg:
                match = re.search(r"Agent (.+?) rewarded.* (\d+\.\d+) -> (\d+\.\d+)", msg)
                if match:
                    agent, old_score, new_score = match.groups()
                    active_state["trust_db"][agent] = {"score": float(new_score), "status": "Rewarded"}

        # --- EDGE PARSING ---
        elif "EDGE" in node_id:
            if "STATIC ROLE:" in msg:
                role_match = re.search(r"STATIC ROLE: ([A-Z_]+)", msg)
                if role_match:
                    role = role_match.group(1)
                    active_state["nodes"][node_id]["Role"] = "Honest" if role == "BENIGN" else f"Byzantine ({role})"
            elif "Threat profile found but DISABLED" in msg:
                active_state["nodes"][node_id]["Role"] = "Honest"
                active_state["nodes"][node_id]["Status"] = "Idle"
            elif "Applying static" in msg and "adversarial split" in msg:
                active_state["nodes"][node_id]["Status"] = "Data Preparation"
            elif "execute_training" in msg:
                active_state["nodes"][node_id]["Status"] = "Training (In Progress)"
            elif "Epoch" in msg and "complete" in msg:
                match = re.search(r"Epoch (\d+)/(\d+) complete", msg)
                if match:
                    if match.group(1) == match.group(2):
                        active_state["nodes"][node_id]["Status"] = "Training Complete & Uploading"
                    else:
                        active_state["nodes"][node_id]["Status"] = f"Training (Epoch {match.group(1)}/{match.group(2)})"
                        
            # --- TOKEN GENERATION & EXTRACTION ---
            elif "preparing to sign NONCE" in msg:
                active_state["nodes"][node_id]["Status"] = "Generating Token & Uploading"
                # Initialize an empty framework token based on the specified parameters
                active_state["tokens"].append({
                    "Edge_ID": node_id, 
                    "Round": rnd,
                    "ID_i": "Pending...",
                    "t (Timestamp)": entry.get("timestamp"),
                    "PCR": "Pending...",
                    "Sig_TPM": "Pending..."
                })
                
            elif "Token:" in msg and "[TPM-GENERATE]" in msg:
                # Dynamically extract full unredacted cryptographic values using Regex
                pcr_match = re.search(r"'pcr_data':\s*'([^']+)'", msg)
                sig_match = re.search(r"'signature':\s*'([^']+)'", msg)
                idi_match = re.search(r"'IDi':\s*'([^']+)'", msg)
                
                # Locate the specific token mapping to the active node and round to update it
                for t in reversed(active_state["tokens"]):
                    if t["Edge_ID"] == node_id and t["Round"] == rnd:
                        if idi_match: t["ID_i"] = idi_match.group(1)
                        if pcr_match: t["PCR"] = pcr_match.group(1)
                        if sig_match: t["Sig_TPM"] = sig_match.group(1)
                        break
            
            # --- Live Extraction from specific Attack/Defense Logs ---
            elif "Performing" in msg:
                rp = active_state["nodes"][node_id]["runtime_params"]
                
                if "SHAP AWARE Attack!" in msg:
                    rp["type"] = "SHAP AWARE"
                    m = re.search(r"Attack type: (.*?), Number of classes: (\d+), Shap Threshold: ([\d.]+), Shap explain count: (\d+), Shap val samples: (\d+), Alpha scale for gradient manip: ([\d.]+), Flip probability: ([\d.]+), Learning Rate: ([\d.]+), Epochs: (\d+), Clip norm: ([\d.]+)", msg)
                    if m:
                        rp["shap_aware_base_attack"], rp["num_classes"], rp["shap_threshold"], rp["shap_explain_count"], rp["shap_val_samples"], rp["alpha_scale"], rp["p_flip"], rp["lr"], rp["epochs"], rp["clip_norm"] = m.groups()
                        
                elif "Backdoor attack!" in msg:
                    rp["type"] = "BACKDOOR"
                    m = re.search(r"Backdoor poison fraction: ([\d.]+), Target class: (\d+), Backdoor Trigger value: ([\d.]+), Backdoor Trigger features: (\[.*?\])", msg)
                    if m:
                        rp["backdoor_poison_fraction"], rp["target_class"], rp["trigger_value"], rp["trigger_features"] = m.groups()
                        
                elif "PGD Attack!" in msg:
                    rp["type"] = "PGD"
                    m = re.search(r"PGD adv ratio: ([\d.]+), PGD number of iter: (\d+), PGD eps: ([\d.]+), PGD alpha: ([\d.]+), PGD clip_min: ([\w.-]+), PGD clip_max: ([\w.-]+)", msg)
                    if m:
                        rp["adv_ratio"], rp["pgd_n_iter"], rp["eps"], rp["alpha"], rp["clip_min"], rp["clip_max"] = m.groups()
                        
                elif "FGSM Attack!" in msg:
                    rp["type"] = "FGSM"
                    m = re.search(r"FGSM adv ratio: ([\d.]+), FGSM eps: ([\d.]+), FGSM clip_min: ([\w.-]+), FGSM clip_max: ([\w.-]+)", msg)
                    if m:
                        rp["adv_ratio"], rp["eps"], rp["clip_min"], rp["clip_max"] = m.groups()
                        
                elif "Adversarial Training on Benign Node!" in msg:
                    rp["type"] = "BENIGN"
                    m = re.search(r"Adversary Ratio: ([\d.]+), Epsilon: ([\d.]+), Alpha: ([\d.]+), Number of Iter: (\d+), Use pgd \(if false, use fgsm\): (.*?), clip_min: ([\w.-]+), clip_max: ([\w.-]+), Clip norm: ([\d.]+)", msg)
                    if m:
                        rp["adv_ratio"], rp["eps"], rp["alpha"], rp["n_iter"], rp["use_pgd"], rp["clip_min"], rp["clip_max"], rp["clip_norm"] = m.groups()
                        
                elif "Attack!" in msg:
                    # Fallback for Generic Attacks like Label Flip & Gradient Manipulation
                    m_type = re.search(r"Performing (.*?) Attack!", msg)
                    if m_type:
                        rp["type"] = m_type.group(1).upper()
                    m = re.search(r"Number of classes: (\d+), Alpha scale: ([\d.]+), p_flip: ([\d.]+), Learning Rate: ([\d.]+), Epochs: (\d+), Clip norm: ([\d.]+)", msg)
                    if m:
                        rp["num_classes"], rp["alpha_scale"], rp["p_flip"], rp["lr"], rp["epochs"], rp["clip_norm"] = m.groups()

        # Sync the global baseline for the chronological transition
        current_state = copy.deepcopy(active_state)

    return state_history