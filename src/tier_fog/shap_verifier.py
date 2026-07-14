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
) -> Tuple[nn.Module, float, List[bool], List[bool], List[float]]: 
    """Runs parallel SHAP stability checks to weight and aggregate updates based on structural trust.
    Returns the aggregated model, regional trust, flags, AND the raw SHAP scores for logging.
    """
    # Initialize a list to hold the SHAP stability score (s_i) for each agent's model
    stability_scores: list[float] = [0.0] * len(local_models)
    
    # Helper function to compute SHAP stability for a single model asynchronously
    def _compute_single_shap(idx: int, local_m: nn.Module) -> tuple[int, float]:
        # Computes s_i = 1 - (||phi_i - phi_ref||_2 / (||phi_ref||_2 + epsilon))
        score = compute_shap_stability(
            local_m, ref_model, X_val, y_val,
            n_explain=n_explain, n_classes=n_classes, device="cpu"
        )
        return idx, score

    # Cap the maximum threads to 10 to prevent resource exhaustion at the Fog Node
    max_threads = min(10, len(local_models))

    # Parallelize SHAP computation across available threads to reduce per-round latency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Dispatch a SHAP calculation task for every model submitted by edge agents
        futures = [executor.submit(_compute_single_shap, i, m) for i, m in enumerate(local_models)]
        
        # As tasks complete, collect the results
        for future in concurrent.futures.as_completed(futures):
            i, score = future.result()
            stability_scores[i] = score

    # If there is more than one model, calculate statistical boundaries for outlier detection
    if len(stability_scores) > 1:
        # Convert list to tensor
        stability_tensor = torch.tensor(stability_scores, dtype=torch.float32)

        # Calculate mu_s using the Median (more robust to extreme Byzantine poisoning than Mean)
        mu_s = torch.median(stability_tensor).item()

        # Calculate Median Absolute Deviation (MAD) to estimate standard deviation robustly
        mad = torch.median(torch.abs(stability_tensor - mu_s)).item()

        # Approximate sigma_s (standard deviation) from MAD (1.4826 is the scaling factor for normal distributions)
        # Ensure sigma_s is at least 1e-5 to avoid division by zero or overly aggressive filtering
        sigma_s = max(mad * 1.4826, 1e-5)
    else:
        # Fallback if only one agent is present
        mu_s = stability_scores[0]
        sigma_s = 0.0
        
    # Calculate the Byzantine filtering threshold: mu_s - 2*sigma_s
    filter_threshold = mu_s - (2 * sigma_s)

    # Initialize lists to store models and weights that pass the security filter
    surviving_models, surviving_weights = [], []

    # Flags to report back to TrustDB for filter passing and rewarding
    passed_flags = []
    reward_flags = []

    # Iterate through all agents to filter and calculate aggregation weights
    for i, local_m in enumerate(local_models):
        # Check if agent's stability score falls below the mu_s - 2*sigma_s threshold
        if stability_scores[i] < filter_threshold:
            passed_flags.append(False)
            reward_flags.append(False)
            continue 
            
        # Agent passed the Byzantine filter
        passed_flags.append(True) 

        # Agent gets a trust boost if their score is above average (s_i >= mu_s)
        reward_flags.append(stability_scores[i] >= mu_s)

        # Set the model to evaluation mode to calculate validation accuracy
        local_m.eval()

        # Disable gradient tracking since we are just doing inference
        with torch.no_grad():
            # Get model predictions on the shared validation set
            preds = local_m(X_val).argmax(dim=-1)

            # Calculate acc_i (percentage of correct predictions)
            acc_i = (preds == y_val).float().mean().item()
            
        # Compute the final aggregation weight w_i based on the formula: s_i * acc_i * sqrt(|D_i|)
        w_i = stability_scores[i] * acc_i * math.sqrt(sizes[i])

        # Add the validated model and its computed weight to the survival lists
        surviving_models.append(local_m)
        surviving_weights.append(w_i)

    # Calculate the total weight of all surviving models to use for normalization
    total_regional_trust = sum(surviving_weights)

    # Edge case: If all models were identified as Byzantine and filtered out
    if not surviving_models:
        # Fallback: average the original models without weights 
        # Passed and reward flags still report total failure to TrustDB
        return federated_averaging(local_models, weights=None), 0.0, passed_flags, reward_flags, stability_scores

    # Normalize weights so they sum to 1.0 (standard requirement for Federated Averaging)
    # Prevent division by zero if total_regional_trust is effectively zero
    normalized_weights = [w / total_regional_trust for w in surviving_weights] if total_regional_trust > 1e-12 else None
    
    # Perform the final aggregation step using the normalized SHAP/Accuracy/Size weights
    agg_model = federated_averaging(surviving_models, weights=normalized_weights)

    # Return the new Fog-aggregated model, the total regional trust, and TrustDB signals
    return agg_model, total_regional_trust, passed_flags, reward_flags, stability_scores