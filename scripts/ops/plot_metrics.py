import os
import json
import argparse
import matplotlib.pyplot as plt

def get_latest_run_dir(base_dir="results"):
    """Finds the most recently created folder in the results directory."""
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)

def get_nested_metric(entry, main_key, sub_key):
    """Safely extracts a nested value, handling cases where the entry might be a float."""
    val = entry.get(main_key, {})
    if isinstance(val, dict):
        return val.get(sub_key)
    return None


def merge_offline_metrics(performance, heavy_metrics):
    """Attach offline measurements only to the checkpoint round actually evaluated."""
    rows = [dict(entry) for entry in performance]
    evaluated_round = heavy_metrics.get("evaluated_round")
    if evaluated_round is None:
        return rows
    metrics = heavy_metrics.get("metrics", {})
    for entry in rows:
        if entry.get("round") != evaluated_round:
            continue
        if "backdoor_asr_percentage" in metrics:
            entry["global_asr"] = metrics["backdoor_asr_percentage"]
        robustness = metrics.get("adversarial_robustness", {})
        if robustness.get("attack") in ("pgd", "fgsm"):
            entry[f"global_robustness_{robustness['attack']}"] = robustness
    return rows

def plot_experiment_metrics(run_dir: str, panels: list):
    """Dynamically generates side-by-side subplots based on user requested panels."""
    
    # Normalize the path to remove trailing slashes, then get the folder name
    clean_dir = os.path.normpath(run_dir)
    folder_name = os.path.basename(clean_dir)
    
    # Look for {foldername}.json
    json_path = os.path.join(run_dir, f"{folder_name}.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Error: Could not find {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    performance = data.get("performance", [])
    heavy_path = os.path.join(run_dir, "heavy_metrics.json")
    if os.path.exists(heavy_path):
        with open(heavy_path) as handle:
            heavy = json.load(handle)
        if heavy.get("evaluated_round") is None:
            print("Offline metrics have no checkpoint round; rerun evaluate_security.py to plot them.")
        performance = merge_offline_metrics(performance, heavy)
    if not performance:
        print("❌ Error: No performance data found in JSON.")
        return

    # Extract metric series from the saved run schema.
    rounds = [entry.get("round", i+1) for i, entry in enumerate(performance)]
    accuracy = [entry.get("global_accuracy") for entry in performance]
    macro_f1 = [entry.get("global_macro_f1") for entry in performance]
    loss = [entry.get("global_loss") for entry in performance]
    asr = [entry.get("global_asr") for entry in performance]
    
    # Safely extract nested robustness metrics using the helper function
    pgd_clean = [get_nested_metric(entry, "global_robustness_pgd", "clean_acc") for entry in performance]
    pgd_adv = [get_nested_metric(entry, "global_robustness_pgd", "adv_acc") for entry in performance]
    fgsm_clean = [get_nested_metric(entry, "global_robustness_fgsm", "clean_acc") for entry in performance]
    fgsm_adv = [get_nested_metric(entry, "global_robustness_fgsm", "adv_acc") for entry in performance]

    experiment_name = data.get("metadata", {}).get("experiment_name", folder_name)

    # --- DYNAMIC GRID LAYOUT ---
    num_panels = len(panels)
    # 6 inches of width per panel so they don't squish together
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6))
    
    # Normalize the single-panel result to an iterable.
    if num_panels == 1:
        axes = [axes]

    # --- DRAW EACH REQUESTED PANEL ---
    for ax, panel in zip(axes, panels):
        if panel == "accuracy_f1":
            ax.set_title("Global Accuracy & F1")
            ax.plot(rounds, accuracy, marker='o', label='Accuracy', color='tab:blue')
            ax.plot(rounds, macro_f1, marker='s', linestyle='--', label='Macro F1', color='tab:cyan')
            ax.set_ylim(0, 1.05)
            
        elif panel == "loss":
            ax.set_title("Global Loss")
            ax.plot(rounds, loss, marker='x', label='Loss', color='tab:red')
            
        elif panel == "asr":
            ax.set_title("Backdoor ASR")
            ax.plot(rounds, asr, marker='v', label='Attack Success Rate', color='tab:purple')
            ax.set_ylim(0, 100)
            
        elif panel == "pgd":
            ax.set_title("PGD Robustness")
            ax.plot(rounds, pgd_clean, marker='o', label='Clean Acc (PGD)', color='tab:green')
            ax.plot(rounds, pgd_adv, marker='x', linestyle='--', label='Adv Acc (PGD)', color='tab:red')
            ax.set_ylim(0, 1.05)
            
        elif panel == "fgsm":
            ax.set_title("FGSM Robustness")
            ax.plot(rounds, fgsm_clean, marker='o', label='Clean Acc (FGSM)', color='tab:green')
            ax.plot(rounds, fgsm_adv, marker='x', linestyle='--', label='Adv Acc (FGSM)', color='tab:orange')
            ax.set_ylim(0, 1.05)

        # Apply standard formatting to whatever panel this is
        ax.set_xlabel('Global Round')
        ax.set_ylabel('ASR (%)' if panel == 'asr' else 'Score / Value')
        series = {"accuracy_f1": accuracy + macro_f1, "loss": loss, "asr": asr,
                  "pgd": pgd_clean + pgd_adv, "fgsm": fgsm_clean + fgsm_adv}[panel]
        if all(value is None for value in series):
            ax.text(0.5, 0.5, "Not evaluated", transform=ax.transAxes, ha="center", va="center")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Master title and layout adjustments
    fig.suptitle(f"ZTA-FL Results: {experiment_name}", fontsize=16)
    fig.tight_layout()

    # Name the file based on the panels the user chose (e.g., "accuracy_f1_loss_curves.png")
    output_filename = "_".join(panels) + "_curves.png"
    output_path = os.path.join(run_dir, output_filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated combined plot at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ZTA-FL Results Dynamically")
    
    # Optional experiment directory.
    parser.add_argument("run_dir", type=str, nargs='?', default=None, 
                        help="Path to the experiment folder. If left blank, it grabs the newest one.")
    
    # Select one or more plot panels.
    parser.add_argument("--panels", nargs='+', type=str, 
                        default=["accuracy_f1", "loss"], 
                        choices=["accuracy_f1", "loss", "asr", "pgd", "fgsm"],
                        help="Choose one or more panels to display side-by-side.")
    
    args = parser.parse_args()
    
    # Auto-detect folder if none is provided
    target_dir = args.run_dir
    if target_dir is None:
        print("🔍 No folder specified. Auto-detecting the most recent experiment...")
        target_dir = get_latest_run_dir()
        if target_dir is None:
            print("❌ No results folders found.")
            exit(1)
        print(f"📂 Found latest run: {target_dir}")
    
    plot_experiment_metrics(target_dir, args.panels)
