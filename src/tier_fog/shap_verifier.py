import copy
import math
from typing import List, Tuple

import torch
import torch.nn as nn

from src.shared.network.model_payload import validate_model, validate_sample_count
from src.shared.utils.metrics import federated_averaging, compute_shap_stability, compute_shap_attributions
from src.shared.utils.hardware import get_device


def shap_weighted_aggregate(
    local_models: List[nn.Module], ref_model: nn.Module,
    X_val: torch.Tensor, y_val: torch.Tensor, sizes: List[int],
    n_classes: int = 15, n_explain: int = 10, seed: int = 42, diagnostics: dict | None = None,
) -> Tuple[nn.Module, float, List[bool], List[bool], List[float]]:
    """Matched-sample SHAP with the authors' uniform fallback over survivors.

    The returned regional weight is the raw weight sum for ordinary aggregation
    and 1.0 (the sum of uniform weights) for fallback, so Cloud can use it.
    """
    if len(local_models) != len(sizes):
        raise ValueError("One sample count is required per model")
    if diagnostics is not None:
        diagnostics.update(submitted_updates=len(local_models), valid_population_size=0,
                           mean=None, std=None, threshold=None,
                           raw_weight_total=0.0, uniform_fallback=False)
    if not local_models:
        return copy.deepcopy(ref_model), 0.0, [], [], []
    device = get_device()
    ref_model = ref_model.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    validate_model(ref_model)
    attrs_ref = compute_shap_attributions(ref_model, X_val, y_val, n_explain, device, seed)
    scores = [0.0] * len(local_models)
    accuracies = [0.0] * len(local_models)
    valid = [False] * len(local_models)
    for i, model in enumerate(local_models):
        try:
            validate_sample_count(sizes[i])
            validate_model(model)
            model.to(device).eval()
            score = compute_shap_stability(model, ref_model, X_val, y_val, n_explain, n_classes,
                                           device, precomputed_attrs_ref=attrs_ref, seed=seed)
            if not math.isfinite(score):
                raise ValueError("Nonfinite stability score")
            with torch.no_grad():
                logits = model(X_val)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite validation predictions")
                accuracies[i] = (logits.argmax(dim=-1) == y_val).float().mean().item()
            scores[i] = score
            valid[i] = True
        except (ValueError, FloatingPointError):
            # These are attributable invalid updates; infrastructure errors propagate.
            continue
    population = torch.tensor([s for s, ok in zip(scores, valid) if ok], dtype=torch.float64)
    if diagnostics is not None:
        diagnostics["valid_population_size"] = population.numel()
    if population.numel() == 0:
        return copy.deepcopy(ref_model), 0.0, valid, [False] * len(scores), scores
    # Paper V-B: population mean and standard deviation, including singleton
    # cohorts (std=0). Small cohorts are smoke tests, not robust detection evidence.
    mu = population.mean().item()
    sigma = population.std(unbiased=False).item()
    if population.min().item() == population.max().item():
        # Floating-point reduction can put the mean just below identical scores
        # (e.g. three 0.7s), incorrectly rewarding every tied participant.
        mu, sigma = population[0].item(), 0.0
    threshold = mu - 2 * sigma
    if diagnostics is not None:
        diagnostics.update(mean=mu, std=sigma, threshold=threshold)
    passed = [ok and s >= threshold for ok, s in zip(valid, scores)]
    # The paper's unbounded-below score cannot be used as a negative averaging
    # weight. Such updates keep their raw score for filtering and have zero
    # ordinary weight; surviving updates remain eligible for uniform fallback.
    weights = [max(0.0, s) * acc * math.sqrt(size) if ok else 0.0
               for s, acc, size, ok in zip(scores, accuracies, sizes, passed)]
    total = math.fsum(weights)
    if not math.isfinite(total):
        raise ValueError("Nonfinite total regional trust")
    if diagnostics is not None:
        diagnostics["raw_weight_total"] = total
    rewards = [ok and s > mu for ok, s in zip(passed, scores)]
    if total < 1e-12:
        survivors = [m for m, ok in zip(local_models, passed) if ok]
        if not survivors:
            return copy.deepcopy(ref_model), 0.0, passed, [False] * len(scores), scores
        if diagnostics is not None:
            diagnostics["uniform_fallback"] = True
        return federated_averaging(survivors), 1.0, passed, rewards, scores
    contributing = [(m, w) for m, w in zip(local_models, weights) if w > 0]
    model = federated_averaging([m for m, _ in contributing], [w / total for _, w in contributing])
    return model, total, passed, rewards, scores
