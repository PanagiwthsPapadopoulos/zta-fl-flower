"""
Implements backdoor attack and evaluation utilities designed to analyze 
federated learning robustness.

Constructs a standard BadNet-style trigger pattern attack sequence. The process
involves poisoning a localized subset of training samples by enforcing a constant 
feature-level trigger and modifying the labels to a target class. The global model 
is subsequently evaluated to measure the Attack Success Rate on clean data patched 
with the identical trigger pattern.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# Defines the deterministic default trigger configuration for reproducibility.
# Imposes a fixed additive shift on the trailing features of the poisoned inputs.
TRIGGER_FEATURES: tuple = (-3, -2, -1)
TRIGGER_VALUE: float = 1.5
TARGET_CLASS: int = 0


def apply_trigger(
    X: torch.Tensor,
    trigger_features: tuple = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
) -> torch.Tensor:
    """
    Overwrites designated indices within an input feature tensor with a predefined trigger value.

    Parameters
    ----------
    X : torch.Tensor
        Input feature matrix.
    trigger_features : tuple of int
        Target feature indices. Negative indices denote positions from the array end.
    trigger_value : float
        Scalar value applied to the designated feature positions.

    Returns
    -------
    torch.Tensor
        A cloned tensor containing the stamped trigger pattern.
    """
    X_trig = X.clone()
    for f in trigger_features:
        X_trig[:, f] = trigger_value
    return X_trig


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
    Subsets a fraction of a client dataset, injects the trigger pattern, and 
    modifies the corresponding labels to the target class to execute a BadNet attack.

    Parameters
    ----------
    X : torch.Tensor
        Local feature matrix.
    y : torch.Tensor
        Corresponding local labels.
    poison_fraction : float
        Proportion of samples selected for poisoning.
    target_class : int
        The intended adversarial label.
    trigger_features : tuple
        Target indices for the trigger.
    trigger_value : float
        Scalar magnitude of the trigger.
    seed : int
        Random seed initializing the sample selection generator.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The modified feature matrix and associated label tensor.
    """
    n = int(X.shape[0])
    n_poison = int(poison_fraction * n)
    if n_poison == 0:
        return X.clone(), y.clone()

    g = torch.Generator().manual_seed(seed)
    idx_perm = torch.randperm(n, generator=g)
    poison_idx = idx_perm[:n_poison]

    X_out = X.clone()
    y_out = y.clone()

    X_out[poison_idx] = apply_trigger(
        X_out[poison_idx],
        trigger_features=trigger_features,
        trigger_value=trigger_value,
    )
    y_out[poison_idx] = target_class
    return X_out, y_out


@torch.no_grad()
def compute_backdoor_asr(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    target_class: int = TARGET_CLASS,
    trigger_features: tuple = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
    device: str = "cpu",
    batch_size: int = 256,
) -> float:
    """
    Calculates the empirical Attack Success Rate of a trained model by applying 
    the trigger pattern to all non-target validation samples.

    Parameters
    ----------
    model : nn.Module
        The global model subjected to evaluation.
    X_test : torch.Tensor
        Unmodified evaluation features.
    y_test : torch.Tensor
        Ground truth evaluation labels.
    target_class : int
        The adversarial label class.
    trigger_features : tuple
        Target indices matching the poisoning parameters.
    trigger_value : float
        Scalar magnitude matching the poisoning parameters.
    device : str
        Target execution hardware for the tensors.
    batch_size : int
        Number of samples processed per iteration.

    Returns
    -------
    float
        The calculated Attack Success Rate represented as a percentage.
    """
    model.train()
    model.to(device)

    non_target_mask = (y_test != target_class)
    X_eval = X_test[non_target_mask].to(device)
    if X_eval.shape[0] == 0:
        return 0.0

    X_trig = apply_trigger(
        X_eval,
        trigger_features=trigger_features,
        trigger_value=trigger_value,
    )

    n = X_trig.shape[0]
    hits = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X_trig[start:end]
        if chunk.shape[0] < 2:
            continue
        preds = model(chunk).argmax(dim=-1)
        hits += int((preds == target_class).sum().item())

    asr = 100.0 * hits / max(n, 1)
    return float(asr)