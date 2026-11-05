from __future__ import annotations
import csv
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.shared.data.dataset_schema import (
    CIC_IDS2017_LABELS, EDGE_FULL_FEATURES, EDGE_FULL_LABELS, UNSW_NB15_LABELS,
)

_MASTER_DATA_CACHE = {}

# The shipped snippet has its own taxonomy. Do not interpret its Scanning or
# MITM_ARP labels as the full dataset's DDoS_HTTP or DDoS_ICMP classes.
EDGE_IIOT_LABELS = {
    "Normal": 0, "DoS_TCP": 1, "DoS_UDP": 2, "Scanning": 3,
    "MITM_Attack": 4, "Fingerprinting": 5, "Password": 6,
    "Port_Scanning": 7, "Ransomware": 8, "Backdoor": 9,
    "Vulnerability_Scanner": 10, "Upload": 11, "SQL_Injection": 12,
    "XSS": 13, "MITM_ARP": 14,
}

_CIC_DROP_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}
_UNSW_DROP_COLS = {"srcip", "dstip", "proto", "state", "service", "attack_cat"}

DATASET_METADATA = {
    "edge_iiotset": {"classes": 15, "features": 40},
    "edge":         {"classes": 15, "features": 40}, 
    "cic_ids2017":  {"classes": 15, "features": 40},
    "cic":          {"classes": 15, "features": 40},
    "unsw_nb15":    {"classes": 10, "features": 40},
    "unsw":         {"classes": 10, "features": 40}
}


class MinMaxScaler:
    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PCA:
    def __init__(self, n_components: int = 40) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, : self.n_components].T 
        self.explained_variance_ = eigenvalues[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _load_numeric_csv(path: str, label_col: str, labels, drop_columns=()) -> Tuple[np.ndarray, np.ndarray]:
    """Reject schema/data errors explicitly; never turn unknown attacks into benign rows."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            headers = [name.strip() for name in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"{path}: CSV is empty") from exc
        if len(set(headers)) != len(headers):
            raise ValueError(f"{path}: duplicate column names")
        matches = [i for i, name in enumerate(headers) if name.lower() == label_col.lower()]
        if len(matches) != 1:
            raise ValueError(f"{path}: required label column {label_col!r} is missing or ambiguous")
        label_idx = matches[0]
        feature_idxs = [i for i, name in enumerate(headers) if i != label_idx and name not in drop_columns]
        if not feature_idxs:
            raise ValueError(f"{path}: no numeric feature columns")
        if callable(labels):
            labels = labels([headers[i] for i in feature_idxs])
        X_raw, y_raw = [], []
        for line, row in enumerate(reader, 2):
            if len(row) != len(headers):
                raise ValueError(f"{path}:{line}: expected {len(headers)} columns, received {len(row)}")
            raw_label = row[label_idx].strip()
            if raw_label not in labels:
                raise ValueError(f"{path}:{line}: unknown label {raw_label!r}; update the explicit label mapping")
            values = []
            for i in feature_idxs:
                try:
                    value = float(row[i])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line}: invalid numeric value in {headers[i]!r}: {row[i]!r}") from exc
                if not np.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                    raise ValueError(f"{path}:{line}: nonfinite or overflowing value in {headers[i]!r}")
                values.append(value)
            X_raw.append(values)
            y_raw.append(labels[raw_label])
        if not X_raw:
            raise ValueError(f"{path}: CSV contains no observations")
    return np.array(X_raw, dtype=np.float32), np.array(y_raw, dtype=np.int64)


def load_edge_iiotset(path: str, label_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    # The cleaner writes this complete 42-column schema. Legacy snippets and
    # their tests retain their original mapping; no cross-taxonomy aliases.
    def labels_for_features(features):
        return EDGE_FULL_LABELS if set(features) == set(EDGE_FULL_FEATURES) else EDGE_IIOT_LABELS
    return _load_numeric_csv(path, label_col, labels_for_features)


def load_cic_ids2017(path: str, label_col: str = "Label") -> Tuple[np.ndarray, np.ndarray]:
    return _load_numeric_csv(path, label_col, CIC_IDS2017_LABELS, _CIC_DROP_COLS)


def load_unsw_nb15(path: str, label_col: str = "attack_cat") -> Tuple[np.ndarray, np.ndarray]:
    return _load_numeric_csv(path, label_col, UNSW_NB15_LABELS, _UNSW_DROP_COLS | {"label"})


def class_counts(y, num_classes):
    """Include absent classes so sample coverage is visible in artifact manifests."""
    return {str(i): int((y == i).sum()) for i in range(num_classes)}


def resample_training_data(X, y, random_seed):
    """SMOTE with an explicit small-sample policy and no synthetic missing classes."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("SMOTE requires at least two observed training classes")
    minority = counts[counts < counts.max()]
    if len(minority) == 0:
        return X, y, None  # Already balanced; no synthetic observations needed.
    if minority.min() < 2:
        raise ValueError("SMOTE cannot interpolate a singleton minority class; add training rows or disable SMOTE")
    k_neighbors = min(5, int(minority.min()) - 1)
    X_resampled, y_resampled = SMOTE(random_state=random_seed, k_neighbors=k_neighbors).fit_resample(X, y)
    return X_resampled, y_resampled, k_neighbors


def get_dataset(
    dataset_name: str, dataset_path: str, num_classes: int, random_seed: int,
    simulate_global_leakage: bool = False, apply_smote: bool = True, split: str = "train",
    test_split: float = 0.15, val_split: float = 0.15, partition_config: Optional[dict] = None
):
    """
    Consumer function: STRICTLY reads pre-compiled artifacts from disk.
    Performs zero runtime mathematical processing.
    """
    import os
    import torch
    
    # Resolve the artifact filename from the preprocessing settings.
    artifact_prefix = f"{dataset_name}_leakage_{simulate_global_leakage}_smote_{apply_smote}"
    
    # Resolves to /app/data/{dataset_name}/artifacts/
    base_dir = os.path.dirname(os.path.dirname(dataset_path)) 
    artifact_path = os.path.join(base_dir, "artifacts", f"{artifact_prefix}_{split}.pt")
    
    from src.shared.data.artifact_manifest import base_provenance, partition_provenance, validate_artifact, REBUILD
    try:
        expected = base_provenance(dataset_name, dataset_path, num_classes, random_seed,
                                   simulate_global_leakage, apply_smote, test_split, val_split)
    except OSError as exc:
        raise ValueError(f"Cannot validate artifact source: {exc}. {REBUILD}") from exc
    if split.startswith("edge_") and partition_config is None:
        raise ValueError("Edge artifact loading requires partition_config. " + REBUILD)
    partition = partition_provenance(partition_config) if split.startswith("edge_") else None
    manifest_path = os.path.join(base_dir, "artifacts", f"{artifact_prefix}_manifest.json")
    validate_artifact(artifact_path, manifest_path, expected, partition)
    return torch.load(artifact_path, weights_only=True)


def non_iid_partition(X: torch.Tensor, y: torch.Tensor, n_agents: int, n_classes_per: int = 3, power_law_a: float = 0.4, seed: int = 42) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if n_agents < 1 or n_classes_per < 1 or power_law_a <= 0 or len(y) == 0 or len(X) != len(y):
        raise ValueError("Partitioning requires nonempty matching tensors and positive parameters")
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    classes = np.unique(y.numpy()).tolist()
    n_classes = len(classes)

    class_indices: dict[int, list[int]] = {c: [] for c in classes}
    for idx, label in enumerate(y.tolist()):
        class_indices[label].append(idx)

    for c in classes:
        np_rng.shuffle(class_indices[c])

    raw_sizes = np_rng.power(a=power_law_a, size=n_agents)
    agent_target_sizes = (raw_sizes * (5000 - 500) + 500).astype(int)

    partitions: list[tuple[torch.Tensor, torch.Tensor]] = []
    class_pointers = {c: 0 for c in classes}

    for agent_id in range(n_agents):
        n_cls = min(n_classes_per, n_classes)
        assigned = rng.sample(classes, k=n_cls)
        target_size = agent_target_sizes[agent_id]
        share_per_class = max(1, target_size // n_cls)

        agent_idxs: list[int] = []
        for c in assigned:
            indices = class_indices[c]
            if not indices:
                continue
            start = class_pointers[c]
            selected = []
            unique_share = min(share_per_class, len(indices))
            while len(selected) < unique_share:
                remaining = unique_share - len(selected)
                chunk = indices[start : start + remaining]
                selected.extend(chunk)
                start = (start + len(chunk)) % len(indices)
            class_pointers[c] = start
            agent_idxs.extend(selected)

        np_rng.shuffle(agent_idxs)
        if len(agent_idxs) == 0:
            agent_idxs = [class_indices[c][0] for c in assigned if class_indices[c]]

        idx_tensor = torch.tensor(agent_idxs, dtype=torch.long)
        partitions.append((X[idx_tensor], y[idx_tensor]))
    return partitions
