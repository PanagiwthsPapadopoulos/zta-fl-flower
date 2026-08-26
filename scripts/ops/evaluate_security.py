"""
Standalone Security Evaluator for ZTA-FL models

How to Use:
    Run this script from your terminal, passing the name of the specific 
    experiment run you want to evaluate.

Example:
    python scripts/ops/evaluate_security.py test_run

Prerequisites:
    Ensure your project directory contains the necessary artifacts from your 
    federated learning run:
    
    1. Directory Structure: A `results/<experiment_name>` directory must exist.
    2. Required Files (inside the run directory):
       - `<experiment_name>.json`: The configuration artifact for the run.
       - `global_model.pt`: The saved PyTorch global model weights.
    3. Optional Directory (for Attestation Metrics): 
       - `trustdb/`: A subdirectory containing JSON files from the attestation 
         rounds. If missing, the script skips calculating FAR, FRR, and FPR.

Output:
    Generates a `heavy_metrics.json` artifact inside `results/<experiment_name>` containing:
    - Adversarial Robustness results
    - Backdoor ASR (Attack Success Rate)
    - Attestation Gatekeeper Metrics (FAR, FRR, FPR)
    - Execution timing stats

"""
import os
import sys
import json
import time
import argparse
import ast
import torch

# Dynamically anchor to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Force the correct config path into the environment BEFORE importing src modules
os.environ["CONFIG_PATH"] = os.path.join(project_root, 'config')

# Ensure the script can find the 'src' module
sys.path.append(project_root)

from src.shared.models.factory import get_model
from src.shared.data.data_loader import get_dataset, DATASET_METADATA
from src.shared.security.adversarial_math import evaluate_robustness
from src.shared.security.backdoor_math import compute_backdoor_asr
from src.shared.utils.config_loader import load_yaml_configs


def calculate_attestation_rates(run_dir: str) -> dict:
    """Parses the trustdb JSON artifacts to calculate FAR, FRR, and FPR."""
    trustdb_dir = os.path.join(run_dir, "trustdb")
    total_benign, total_malicious = 0, 0
    false_rejections, false_acceptances = 0, 0

    if not os.path.exists(trustdb_dir):
        return {}

    for fname in os.listdir(trustdb_dir):
        if fname.endswith(".json"):
            with open(os.path.join(trustdb_dir, fname), "r") as f:
                fog_data = json.load(f)
            
            for r_num, r_data in fog_data.get("rounds", {}).items():
                accepted = r_data.get("attestation_accepted", [])
                rejected = r_data.get("attestation_rejected", [])
                ground_truths = r_data.get("node_ground_truths", {})
                
                for node_id in accepted + rejected:
                    role = ground_truths.get(node_id, "benign")
                    is_malicious = role != "benign"
                    
                    if is_malicious:
                        total_malicious += 1
                        if node_id in accepted:
                            false_acceptances += 1
                    else:
                        total_benign += 1
                        if node_id in rejected:
                            false_rejections += 1

    # In security/biometrics: False Positive Rate (FPR) = False Rejection Rate (FRR) for benign nodes flagged as attacks.
    frr = (false_rejections / total_benign) if total_benign > 0 else 0.0
    far = (false_acceptances / total_malicious) if total_malicious > 0 else 0.0

    return {
        "total_benign_attempts": total_benign,
        "total_malicious_attempts": total_malicious,
        "false_rejections_count": false_rejections,
        "false_acceptances_count": false_acceptances,
        "FRR_percentage": round(frr * 100, 4),
        "FAR_percentage": round(far * 100, 4),
        "FPR_percentage": round(frr * 100, 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone Security Evaluator for Federated Models")
    parser.add_argument("experiment_name", type=str, help="The name of the experiment run (e.g., run_20260823_163527)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.abspath(f"results/{args.experiment_name}")
    config_path = os.path.join(run_dir, f"{args.experiment_name}.json")
    model_path = os.path.join(run_dir, "global_model.pt")
    heavy_metrics_path = os.path.join(run_dir, "heavy_metrics.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration artifact not found at {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Global model weights not found at {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)
    
    metadata = run_data.get("metadata", {})
    dataset_name = str(metadata["dataset"]).lower()
    model_arch = str(metadata.get("model_architecture")).lower()
    random_seed = int(metadata.get("random_seed"))
    
    # Dynamically resolve dataset path
    yaml_configs = load_yaml_configs()
    dataset_path = metadata.get("dataset_path") or yaml_configs["dataset_path"]

    if not os.path.exists(dataset_path):
        print(f"\n[!] CRITICAL: Dataset not found at '{dataset_path}'")
        print(f"[!] Please ensure your YAML configurations contain the correct absolute or relative 'dataset_path'.")
        sys.exit(1)

    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_METADATA registry.")
        
    n_features = DATASET_METADATA[dataset_name]["features"]
    num_classes = DATASET_METADATA[dataset_name]["classes"]

    print(f"\n[*] Initializing Security Evaluation for: {args.experiment_name}")
    print(f"[*] Device: {device.upper()} | Model: {model_arch.upper()} | Dataset: {dataset_name.upper()}\n")

    # Load Model
    print("[*] Reconstructing and loading global model... ", end="", flush=True)
    model = get_model(model_arch, n_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("success")

    # Load Data
    print(f"[*] Loading test data split from {dataset_path}... ", end="", flush=True)
    X_test, y_test, _ = get_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, 
        num_classes=num_classes, random_seed=random_seed, split="test",
        simulate_global_leakage=bool(metadata.get("simulate_global_leakage", False)),
        apply_smote=bool(metadata.get("apply_smote", True))
    )
    print("success\n")

    payload = {
        "run_name": args.experiment_name,
        "metrics": {},
        "execution_stats": {},
        "timestamp": time.time(),
        "human_readable_time": time.ctime()
    }

    # --- METRIC 1: ADVERSARIAL ROBUSTNESS ---
    print("[*] Generating Adversarial Robustness... ", end="", flush=True)
    t0 = time.perf_counter()
    rob_results = evaluate_robustness(
        model=model, X=X_test, y=y_test,
        attack=str(metadata.get("robustness_eval_attack")),
        eps=float(metadata.get("benign_eps")),
        alpha=float(metadata.get("benign_alpha")),
        n_iter=int(metadata.get("benign_n_iter")),
        device=device,
        clip_min=float(metadata.get("clip_min")),
        clip_max=float(metadata.get("clip_max"))
    )
    rob_time = time.perf_counter() - t0
    print(f"success ({rob_time:.2f}s)")
    payload["metrics"]["adversarial_robustness"] = rob_results
    payload["execution_stats"]["robustness_calculation_time_sec"] = round(rob_time, 4)

    # --- METRIC 2: BACKDOOR ASR ---
    print("[*] Generating ASR... ", end="", flush=True)
    t0 = time.perf_counter()
    raw_feats = metadata["backdoor_trigger_features"]
    trig_feats = tuple(ast.literal_eval(raw_feats)) if isinstance(raw_feats, str) else tuple(raw_feats)
    
    asr_score = compute_backdoor_asr(
        model=model, X_test=X_test, y_test=y_test,
        target_class=int(metadata["backdoor_target_class"]),
        trigger_features=trig_feats,
        trigger_value=float(metadata.get("backdoor_trigger_value")),
        device=device
    )
    asr_time = time.perf_counter() - t0
    print(f"success ({asr_time:.2f}s)")
    payload["metrics"]["backdoor_asr_percentage"] = asr_score
    payload["execution_stats"]["asr_calculation_time_sec"] = round(asr_time, 4)

    # --- METRIC 3: ATTESTATION RATES (FAR/FRR/FPR) ---
    print("[*] Calculating FAR, FRR, and FPR... ", end="", flush=True)
    t0 = time.perf_counter()
    attestation_metrics = calculate_attestation_rates(run_dir)
    att_time = time.perf_counter() - t0
    print(f"success ({att_time:.2f}s)")
    payload["metrics"]["attestation_gatekeeper"] = attestation_metrics
    payload["execution_stats"]["attestation_parsing_time_sec"] = round(att_time, 4)

    # --- SAVE ARTIFACT ---
    with open(heavy_metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    print(f"\n[+] Security evaluation complete! Metrics securely written to:\n    -> {heavy_metrics_path}")

if __name__ == "__main__":
    main()