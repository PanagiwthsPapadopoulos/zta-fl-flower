from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

TRIGGER_FEATURES: tuple = (-3, -2, -1)
TRIGGER_VALUE: float = 1.5
TARGET_CLASS: int = 0


def apply_trigger(X: torch.Tensor, trigger_features: tuple, trigger_value: float) -> torch.Tensor:
    """
    Applies the trigger to the selected features of the input tensor.
    """
    X_triggered = X.clone()
    for feat_idx in trigger_features:
        # Assuming X is 2D (samples, features); adjust dimensions for images (e.g., NCHW)
        X_triggered[:, feat_idx] = trigger_value
    return X_triggered


def poison_partition(
    X: torch.Tensor, 
    y: torch.Tensor, 
    poison_fraction: float = 0.5, 
    target_class: int = TARGET_CLASS, 
    trigger_features: tuple = TRIGGER_FEATURES, 
    trigger_value: float = TRIGGER_VALUE, 
    seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Poisons a dataset for a standard dirty-label backdoor attack, 
    ensuring only non-target class samples are poisoned.
    """
    n = int(X.shape[0])
    n_poison = int(poison_fraction * n)

    if n_poison == 0:
        return X.clone(), y.clone()

    g = torch.Generator().manual_seed(seed)

    # Isolate eligible samples (do not poison samples already in the target class)
    eligible_indices = torch.where(y != target_class)[0]

    # Safety check: ensure we don't try to sample more than what's available
    actual_n_poison = min(n_poison, len(eligible_indices))
    if actual_n_poison == 0:
        return X.clone(), y.clone()

    # Randomly select from the eligible indices pool
    permuted_idx = torch.randperm(len(eligible_indices), generator=g)[:actual_n_poison]
    poison_idx = eligible_indices[permuted_idx]

    X_out, y_out = X.clone(), y.clone()

    # Apply trigger and flip the label
    X_out[poison_idx] = apply_trigger(X_out[poison_idx], trigger_features, trigger_value)
    y_out[poison_idx] = target_class
    
    return X_out, y_out


@torch.no_grad()
def compute_backdoor_asr(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, target_class: int = TARGET_CLASS, trigger_features: tuple = TRIGGER_FEATURES, trigger_value: float = TRIGGER_VALUE, device: str = "cpu", batch_size: int = 256) -> float:
    model.train()
    model.to(device)
    X_eval = X_test[y_test != target_class].to(device)
    if X_eval.shape[0] == 0:
        return 0.0

    X_trig = apply_trigger(X_eval, trigger_features, trigger_value)
    n, hits = X_trig.shape[0], 0
    for start in range(0, n, batch_size):
        chunk = X_trig[start:min(start + batch_size, n)]
        if chunk.shape[0] < 2:
            continue
        hits += int((model(chunk).argmax(dim=-1) == target_class).sum().item())
    return float(100.0 * hits / max(n, 1))