import copy
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import concurrent.futures

from src.shared.utils.metrics import federated_averaging, compute_shap_stability, disable_dropout, restore_dropout
from src.shared.utils.hardware import get_device, get_dynamic_thread_limit


def shap_weighted_aggregate(
    local_models: List[nn.Module],
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    sizes: List[int],
    n_classes: int = 15,
    n_explain: int = 10,
) -> Tuple[nn.Module, float, List[bool], List[bool], List[float]]:
    """Runs parallel SHAP stability checks to weight and aggregate updates based on structural trust.
    Returns the aggregated model, regional trust, flags, AND the raw SHAP scores for logging.
    """
    stability_scores: list[float] = [0.0] * len(local_models)
    device = get_device()

    ref_model = ref_model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Precompute reference model attributions ONCE on the main thread to avoid
    # concurrent autograd/dropout race conditions on ref_model across threads
    from captum.attr import GradientShap
    n_exp = min(n_explain, X_val.shape[0])
    X_sub, y_sub = X_val[:n_exp], y_val[:n_exp]
    orig_probs_ref = disable_dropout(ref_model)
    ref_model.eval()
    try:
        gs_ref = GradientShap(ref_model)
        if device == "cuda":
            with torch.backends.cudnn.flags(enabled=False):
                precomputed_attrs_ref = gs_ref.attribute(X_sub, baselines=X_val, target=y_sub).detach()
        else:
            precomputed_attrs_ref = gs_ref.attribute(X_sub, baselines=X_val, target=y_sub).detach()
    finally:
        restore_dropout(ref_model, orig_probs_ref)

    def _compute_single_shap(idx: int, local_m: nn.Module) -> tuple[int, float]:
        local_m = local_m.to(device)
        score = compute_shap_stability(
            local_m, ref_model, X_val, y_val,
            n_explain=n_explain, n_classes=n_classes, device=device,
            precomputed_attrs_ref=precomputed_attrs_ref
        )
        return idx, score

    max_threads = min(get_dynamic_thread_limit(), len(local_models))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(_compute_single_shap, i, m) for i, m in enumerate(local_models)]
        for future in concurrent.futures.as_completed(futures):
            i, score = future.result()
            stability_scores[i] = max(0.0, min(1.0, float(score)))

    if len(stability_scores) > 1:
        stability_tensor = torch.tensor(stability_scores, dtype=torch.float32)
        mu_s = torch.median(stability_tensor).item()
        mad = torch.median(torch.abs(stability_tensor - mu_s)).item()
        sigma_s = max(mad * 1.4826, 1e-5)
    else:
        mu_s = stability_scores[0]
        sigma_s = 0.0

    filter_threshold = mu_s - (2 * sigma_s)

    surviving_models, surviving_weights = [], []
    passed_flags = []
    reward_flags = []

    for i, local_m in enumerate(local_models):
        if stability_scores[i] < filter_threshold:
            passed_flags.append(False)
            reward_flags.append(False)
            continue

        passed_flags.append(True)
        reward_flags.append(stability_scores[i] >= mu_s)

        local_m.eval()
        with torch.no_grad():
            preds = local_m(X_val).argmax(dim=-1)
            acc_i = (preds == y_val).float().mean().item()

        w_i = stability_scores[i] * acc_i * math.sqrt(sizes[i])
        surviving_models.append(local_m)
        surviving_weights.append(w_i)

    total_regional_trust = sum(surviving_weights)

    # Safe fallback: if all models were filtered out as Byzantine, retain ref_model
    if not surviving_models:
        return copy.deepcopy(ref_model), 0.0, passed_flags, reward_flags, stability_scores

    normalized_weights = [w / total_regional_trust for w in surviving_weights] if total_regional_trust > 1e-12 else None
    agg_model = federated_averaging(surviving_models, weights=normalized_weights)

    return agg_model, total_regional_trust, passed_flags, reward_flags, stability_scores