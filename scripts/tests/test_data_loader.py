"""
================================================================================
DATA LOADER CONSUMER TEST
================================================================================

HOW TO RUN:
Ensure your terminal is at the absolute root of the project (the `zta-fl-flower` directory).
Execute the following command to spin up an isolated Docker container for testing:

docker run --rm \
    --entrypoint python3 \
    -v "$PWD/data:/app/data" \
    -v "$PWD/src:/app/src:ro" \
    -v "$PWD/scripts:/app/scripts:ro" \
    panagiotispapadopoulos/zta-cloud-node:latest \
    /app/scripts/tests/test_data_loader.py

--------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES:
This script acts as a simulation of a single Flower node (Edge, Fog, or Cloud) booting up. 
It bypasses the entire Federated Learning network and tests only the final step of the 
data pipeline: reading the pre-compiled PyTorch binary tensors (.pt files) from the hard drive.

WHEN TO USE THIS:
- After running the offline builder (`build_artifacts.py`) to verify the artifacts generated successfully.
- If your Edge or Fog nodes crash on boot with a 'FileNotFoundError'.
- If you are tweaking dataset settings in YAML and want a quick sanity check that the 
  data maps into memory correctly before deploying a full Federated Learning round.

HOW IT WORKS:
In our "Compile Once, Run Anywhere" architecture, datasets are not mathematically processed 
at runtime. Instead, they are pre-compiled into binary artifacts. This script calls `get_dataset()`, 
which dynamically generates a unique filename prefix based on the parameters provided below 
(e.g., `edge_iiotset_leakage_False_smote_True_train.pt`). It then attempts to instantly load 
that specific binary file from the mounted `/data/` volume into RAM.

THE PARAMETERS:
* dataset_name: The internal identifier of the dataset (e.g., "edge_iiotset").
* dataset_path: The path to the raw CSV. The loader uses this path to figure out where 
                the `/artifacts/` directory is located.
* num_classes: Expected number of classification targets.
* random_seed: Used to ensure deterministic behavior; must match your builder script.
* simulate_global_leakage: Boolean flag (True/False). If changed, it expects a totally different file!
* apply_smote: Boolean flag (True/False). If changed, it expects a totally different file!
* split: Which partition to load. "train" (Edge), "val" (Fog), or "test" (Cloud).

EXPECTED OUTPUT:
- SUCCESS: The script will print the exact memory dimensions (shape) and data types of the loaded X and y tensors.
- FAILURE: A loud error (usually FileNotFoundError) indicating the parameters below do not match 
           what was actually compiled, or the builder script was never run.
================================================================================
"""

import sys
import torch
from src.shared.utils.config_loader import load_yaml_configs
from src.shared.data.data_loader import get_dataset, DATASET_METADATA
def test_consumer():
    print("🧪 Testing Consumer (get_dataset)...")
    try:
        # Dynamically read the active SSOT configuration
        config = load_yaml_configs()
        
        dataset_name = str(config["dataset"]).lower()
        dataset_path = str(config["dataset_path"])
        random_seed = int(config["random_seed"])
        simulate_leakage = bool(config["simulate_global_leakage"])
        apply_smote = bool(config["apply_smote"])
        test_split = float(config["test_split"])
        val_split = float(config["val_split"])
        
        num_classes = DATASET_METADATA[dataset_name]["classes"]
        
        print(f"Detected Config: {dataset_name} | Leakage: {simulate_leakage} | SMOTE: {apply_smote}")

        # Request the 'train' split using the exact live parameters
        X, y, classes = get_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_classes=num_classes,
            random_seed=random_seed,
            simulate_global_leakage=simulate_leakage,
            apply_smote=apply_smote,
            split="train",
            test_split=test_split,
            val_split=val_split
        )
        
        print("✅ SUCCESS! Artifact loaded instantly.")
        print(f"X Tensor Shape: {X.shape} | Dtype: {X.dtype}")
        print(f"y Tensor Shape: {y.shape} | Dtype: {y.dtype}")
        print(f"Total Classes:  {classes}")
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_consumer()