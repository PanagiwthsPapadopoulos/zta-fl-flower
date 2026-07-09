from __future__ import annotations
import csv
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

_MASTER_DATA_CACHE = {}

EDGE_IIOT_LABELS = {
    "Normal": 0, "DoS_TCP": 1, "DoS_UDP": 2, "Scanning": 3,
    "MITM_Attack": 4, "Fingerprinting": 5, "Password": 6,
    "Port_Scanning": 7, "Ransomware": 8, "Backdoor": 9,
    "Vulnerability_Scanner": 10, "Upload": 11, "SQL_Injection": 12,
    "XSS": 13, "MITM_ARP": 14,
}

CIC_IDS2017_LABELS = {
    "BENIGN": 0, "DoS Hulk": 1, "DDoS": 2, "PortScan": 3,
    "Bot": 4, "Web Attack – Brute Force": 5, "Web Attack – XSS": 6,
    "Web Attack – Sql Injection": 7, "Infiltration": 8, "Heartbleed": 9,
}

UNSW_NB15_LABELS = {
    "Normal": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3,
    "DoS": 4, "Reconnaissance": 5, "Backdoor": 6, "Analysis": 7,
    "Shellcode": 8, "Worms": 9,
}

_CIC_DROP_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}
_UNSW_DROP_COLS = {"srcip", "dstip", "proto", "state", "service", "attack_cat"}

DATASET_METADATA = {
    "edge_iiotset": {"classes": 15, "features": 40},
    "edge":         {"classes": 15, "features": 40}, 
    "cic_ids2017":  {"classes": 10, "features": 40},
    "cic":          {"classes": 10, "features": 40},
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


def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows

def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def load_edge_iiotset(path: str, label_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    headers, rows = _read_csv(path)
    label_idx = headers.index(label_col)
    feature_idxs = [i for i in range(len(headers)) if i != label_idx]

    X_raw, y_raw = [], []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = EDGE_IIOT_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)
    return np.array(X_raw, dtype=np.float32), np.array(y_raw, dtype=np.int64)

def load_cic_ids2017(path: str, label_col: str = "Label") -> Tuple[np.ndarray, np.ndarray]:
    headers, rows = _read_csv(path)
    label_idx = next((i for i, h in enumerate(headers) if h.strip().lower() == label_col.lower()), len(headers) - 1)
    drop_set = {i for i, h in enumerate(headers) if h.strip() in _CIC_DROP_COLS}
    feature_idxs = [i for i in range(len(headers)) if i != label_idx and i not in drop_set]

    X_raw, y_raw = [], []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = CIC_IDS2017_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)

    X = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(y_raw, dtype=np.int64)

def load_unsw_nb15(path: str, label_col: str = "attack_cat") -> Tuple[np.ndarray, np.ndarray]:
    headers, rows = _read_csv(path)
    label_idx = next((i for i, h in enumerate(headers) if h.strip() == label_col), len(headers) - 1)
    binary_label_idx = next((i for i, h in enumerate(headers) if h.strip() == "label"), None)

    drop_set = {i for i, h in enumerate(headers) if h.strip() in _UNSW_DROP_COLS}
    if binary_label_idx is not None:
        drop_set.add(binary_label_idx)

    feature_idxs = [i for i in range(len(headers)) if i != label_idx and i not in drop_set]

    X_raw, y_raw = [], []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = UNSW_NB15_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)

    X = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(y_raw, dtype=np.int64)


def get_dataset(
    dataset_name: str, dataset_path: str, num_classes: int, random_seed: int,
    simulate_global_leakage: bool = False, apply_smote: bool = True, split: str = "train",
    test_split: float = 0.30, val_split: float = 0.50   
):
    global _MASTER_DATA_CACHE
    cache_key = f"{dataset_name}_{simulate_global_leakage}_{apply_smote}_{split}"
    if cache_key in _MASTER_DATA_CACHE:
        return _MASTER_DATA_CACHE[cache_key]
        
    loaders = {
        "edge_iiotset": load_edge_iiotset, "edge": load_edge_iiotset,
        "cic_ids2017": load_cic_ids2017, "cic": load_cic_ids2017,
        "unsw_nb15": load_unsw_nb15, "unsw": load_unsw_nb15
    }
    X_raw, y_raw = loaders[dataset_name](dataset_path)
    target_features = DATASET_METADATA[dataset_name]["features"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=test_split, stratify=y_raw, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, stratify=y_temp, random_state=random_seed)    
    scaler = MinMaxScaler()
    pca = PCA(n_components=target_features)
    
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if X_train_scaled.shape[1] >= target_features:
        pca.fit(X_train_scaled)
        
    if split == "train":
        X_target, y_target = X_train_scaled, y_train
    elif split == "val":
        X_target, y_target = scaler.transform(X_val), y_val
    elif split == "test":
        X_target, y_target = scaler.transform(X_test), y_test

    if X_target.shape[1] > target_features:
        X_target = pca.transform(X_target).astype(np.float32)
    elif X_target.shape[1] < target_features:
        pad = np.zeros((X_target.shape[0], target_features - X_target.shape[1]), dtype=np.float32)
        X_target = np.concatenate([X_target, pad], axis=1)

    if apply_smote and split == "train":
        smote = SMOTE(random_state=random_seed)
        X_resampled, y_resampled = smote.fit_resample(X_target, y_target)
    else:
        X_resampled, y_resampled = X_target, y_target
        
    X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_tensor = torch.tensor(y_resampled, dtype=torch.long)
    result = (X_tensor, y_tensor, num_classes)
    _MASTER_DATA_CACHE[cache_key] = result
    return result


def non_iid_partition(X: torch.Tensor, y: torch.Tensor, n_agents: int, n_classes_per: int = 3, power_law_a: float = 0.4, seed: int = 42) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
            while len(selected) < share_per_class:
                remaining = share_per_class - len(selected)
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