"""
Run using the following command:

docker run --rm \
    --entrypoint python3 \
    -v "$PWD/data:/app/data" \
    -v "$PWD/config:/app/config:ro" \
    -v "$PWD/src:/app/src:ro" \
    -v "$PWD/scripts:/app/scripts:ro" \
    panagiotispapadopoulos/zta-cloud-node:latest \
    /app/scripts/setup/build_artifacts.py

"""

import ast
import os
import sys
import traceback
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.shared.utils.config_loader import load_yaml_configs
from src.shared.data.data_loader import (
    load_edge_iiotset, load_cic_ids2017, load_unsw_nb15,
    MinMaxScaler, PCA, DATASET_METADATA, non_iid_partition
)

def build_artifacts():
    try:
        config = load_yaml_configs()
            
        dataset_name = str(config["dataset"]).lower()
        dataset_path = str(config["dataset_path"])
        random_seed = int(config["random_seed"])
        simulate_leakage = bool(config["simulate_global_leakage"])
        apply_smote = bool(config["apply_smote"])
        test_split = float(config["test_split"])
        val_split = float(config["val_split"])
        
        num_classes = DATASET_METADATA[dataset_name]["classes"]
        target_features = DATASET_METADATA[dataset_name]["features"]

        # 1. Self-Explanatory Naming
        artifact_prefix = f"{dataset_name}_leakage_{simulate_leakage}_smote_{apply_smote}"
        artifact_dir = f"/app/data/{dataset_name}/artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        train_path = os.path.join(artifact_dir, f"{artifact_prefix}_train.pt")
        val_path = os.path.join(artifact_dir, f"{artifact_prefix}_val.pt")
        test_path = os.path.join(artifact_dir, f"{artifact_prefix}_test.pt")
        
        # =========================================================
        # STAGE A: BASE DATASET ARTIFACTS (CACHED)
        # =========================================================
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            print(f"Base artifacts for '{artifact_prefix}' already exist. Skipping base compilation.")
            X_train_tensor, y_train_tensor, _ = torch.load(train_path, weights_only=True)
        else:
            print(f"Compiling new base artifacts for '{artifact_prefix}'...")
            
            loaders = {
                "edge_iiotset": load_edge_iiotset, "edge": load_edge_iiotset,
                "cic_ids2017": load_cic_ids2017, "cic": load_cic_ids2017,
                "unsw_nb15": load_unsw_nb15, "unsw": load_unsw_nb15
            }
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Raw dataset CSV not found at: {dataset_path}")
                
            X_raw, y_raw = loaders[dataset_name](dataset_path)

            X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=test_split, stratify=y_raw, random_state=random_seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, stratify=y_temp, random_state=random_seed)

            X_fit = X_raw if simulate_leakage else X_train

            scaler_pre = MinMaxScaler()
            pca = PCA(n_components=target_features)
            scaler_post = MinMaxScaler() 

            scaler_pre.fit(X_fit)
            X_train_scaled = scaler_pre.transform(X_train)
            X_val_scaled = scaler_pre.transform(X_val)
            X_test_scaled = scaler_pre.transform(X_test)

            apply_pca = X_fit.shape[1] >= target_features
            if apply_pca:
                X_fit_scaled = scaler_pre.transform(X_fit)
                pca.fit(X_fit_scaled)
                X_train_pca = pca.transform(X_train_scaled)
                X_val_pca = pca.transform(X_val_scaled)
                X_test_pca = pca.transform(X_test_scaled)
            else:
                X_train_pca, X_val_pca, X_test_pca = X_train_scaled, X_val_scaled, X_test_scaled

            if apply_pca:
                X_fit_pca = pca.transform(X_fit_scaled)
            else:
                X_fit_pca = scaler_pre.transform(X_fit)
                
            scaler_post.fit(X_fit_pca)
            X_train_final = scaler_post.transform(X_train_pca).astype(np.float32)
            X_val_final = scaler_post.transform(X_val_pca).astype(np.float32)
            X_test_final = scaler_post.transform(X_test_pca).astype(np.float32)

            if apply_smote:
                smote = SMOTE(random_state=random_seed)
                X_train_final, y_train = smote.fit_resample(X_train_final, y_train)

            def pad_tensor(X):
                if X.shape[1] < target_features:
                    pad = np.zeros((X.shape[0], target_features - X.shape[1]), dtype=np.float32)
                    return np.concatenate([X, pad], axis=1)
                return X

            splits_data = {
                "train": (pad_tensor(X_train_final), y_train),
                "val": (pad_tensor(X_val_final), y_val),
                "test": (pad_tensor(X_test_final), y_test)
            }
            
            for s_name, (X_data, y_data) in splits_data.items():
                X_tensor = torch.tensor(X_data, dtype=torch.float32)
                y_tensor = torch.tensor(y_data, dtype=torch.long)
                target_file = os.path.join(artifact_dir, f"{artifact_prefix}_{s_name}.pt")
                torch.save((X_tensor, y_tensor, num_classes), target_file)
                if s_name == "train":
                    X_train_tensor, y_train_tensor = X_tensor, y_tensor

            print("Base compilation complete and saved to disk.")

        # =========================================================
        # STAGE B: NON-IID EDGE PARTITIONS (OVERWRITTEN EACH RUN)
        # =========================================================
        dataset_fraction = float(config["dataset_fraction"])
        n_classes_per = int(config["n_classes_per"])
        power_law_a = float(config["power_law_a"])

        custom_top_str = str(config.get("custom_fog_topology", ""))
        custom_topology = ast.literal_eval(custom_top_str) if custom_top_str.strip() else []
        num_fogs = int(config["num_fogs"])
        uniform_edges = int(config["uniform_edges_per_fog"])

        if custom_topology and len(custom_topology) > 0:
            topology = custom_topology
        else:
            topology = [uniform_edges] * num_fogs

        total_edges = sum(topology)

        # Clean up any previous edge partition files to avoid filesystem clutter
        for fname in os.listdir(artifact_dir):
            if "_edge_" in fname and fname.endswith(".pt"):
                os.remove(os.path.join(artifact_dir, fname))

        print(f"Generating Non-IID partitions for {total_edges} edge nodes...")

        # Shuffle globally before partitioning
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(len(X_train_tensor), generator=generator)
        X_full = X_train_tensor[indices]
        y_full = y_train_tensor[indices]

        # Fraction the dataset
        if dataset_fraction < 1.0:
            subset_size = int(len(X_full) * dataset_fraction)
            X_full = X_full[:subset_size]
            y_full = y_full[:subset_size]

        # Split the data across the edge devices
        partitions = non_iid_partition(
            X=X_full,
            y=y_full,
            n_agents=total_edges,
            n_classes_per=n_classes_per,
            power_law_a=power_law_a,
            seed=random_seed
        )

        for idx, (X_part, y_part) in enumerate(partitions):
            edge_file = os.path.join(artifact_dir, f"{artifact_prefix}_edge_{idx}.pt")
            torch.save((X_part, y_part, num_classes), edge_file)

        print(f"Saved {len(partitions)} edge partitions to disk.")

    except Exception as e:
        print(f"\n❌ FATAL ARTIFACT BUILD ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)  # Strictly forces Docker to return a failure code

if __name__ == "__main__":
    build_artifacts()