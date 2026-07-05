import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import concurrent.futures

from src.shared.utils.metrics import federated_averaging, compute_shap_stability


def shap_weighted_aggregate(
    local_models: List[nn.Module],
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    sizes: List[int],
    n_classes: int = 15,
    n_explain: int = 10,
) -> Tuple[nn.Module, float, List[bool], List[bool]]: 
    """Runs parallel SHAP stability checks to weight and aggregate updates based on structural trust.
    Returns the aggregated model AND the total regional trust weight for Cloud use.
    """
    stability_scores: list[float] = [0.0] * len(local_models)
    
    def _compute_single_shap(idx: int, local_m: nn.Module) -> tuple[int, float]:
        score = compute_shap_stability(
            local_m, ref_model, X_val, y_val,
            n_explain=n_explain, n_classes=n_classes, device="cpu"
        )
        return idx, score

    max_threads = min(10, len(local_models))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(_compute_single_shap, i, m) for i, m in enumerate(local_models)]
        for future in concurrent.futures.as_completed(futures):
            i, score = future.result()
            stability_scores[i] = score

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

    if not surviving_models:
        return federated_averaging(local_models, weights=None), 0.0, passed_flags, reward_flags

    normalized_weights = [w / total_regional_trust for w in surviving_weights] if total_regional_trust > 1e-12 else None
    agg_model = federated_averaging(surviving_models, weights=normalized_weights)
    
    return agg_model, total_regional_trust, passed_flags, reward_flags