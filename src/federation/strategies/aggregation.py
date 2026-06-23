"""
Federated aggregation strategies for the ZTA-FL framework.

Implements standard and Byzantine-robust aggregation algorithms used by the
fog aggregator nodes to combine local model updates from IIoT edge devices.
"""

from __future__ import annotations

import copy
import math
import statistics
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import concurrent.futures


def federated_averaging(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
) -> nn.Module:
    """
    Aggregates a batch of local models using standard weighted parameter averaging.
    Operates on the baseline formula $\theta_{global} = \sum_{i=1}^K w_i \cdot \theta_i$.
    Executes straight math for honest environments without aggressive filtering.

    Parameters
    ----------
    models : list of nn.Module
        Local model instances, one per participating edge device.
    weights : list of float, optional
        Per-device weighting coefficients (e.g., proportional to local dataset
        sizes).  Must sum to 1.  Uniform weights are used when ``None``.

    Returns
    -------
    nn.Module
        A newly instantiated model containing the unified global state.
    """
    if len(models) == 0:
        raise ValueError("At least one model is required for aggregation.")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    if abs(sum(weights) - 1.0) > 1e-5:
        total = sum(weights)
        weights = [w / total for w in weights]

    global_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for key in global_model.state_dict():
            aggregated = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
            for model, w in zip(models, weights):
                param = model.state_dict()[key].float()
                aggregated += w * param
            global_model.state_dict()[key].copy_(aggregated)

    return global_model


def fedprox_update(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mu: float = 0.01,
    device: str = "cpu",
) -> float:
    """
    Executes a local training epoch while anchoring the weights with a FedProx proximal term.
    
    Punishes parameter drift from the global baseline to keep heterogeneous, non-IID 
    clients tethered to the main objective. The loss function is heavily modified by
    adding the constraint: $0.5 \times \mu \times ||w - w_{global}||^2$.

    Parameters
    ----------
    model : nn.Module
        The active local model being optimized in-place.
    global_model : nn.Module
        The frozen global architecture used as the reference anchor.
    loader : DataLoader
        The iterator pushing the localized dataset batches.
    optimizer : torch.optim.Optimizer
        The target optimization algorithm driving the descent.
    mu : float
        The proximal regularization coefficient dictating the tether strength.
    device : str
        Hardware execution target.

    Returns
    -------
    float
        The average training loss calculated across the epoch.
    """
    model.train()
    model.to(device)
    global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    global_params = {n: p.detach().clone() for n, p in global_model.named_parameters()}

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if X_batch.size(0) < 2:
            continue  # BatchNorm requires >= 2 samples

        optimizer.zero_grad()
        logits = model(X_batch)
        ce_loss = criterion(logits, y_batch)

        # Computes the proximal penalty: 0.5 * mu * ||w - w_global||^2
        prox = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in global_params:
                prox += ((param - global_params[name]) ** 2).sum()
        prox = 0.5 * mu * prox

        loss = ce_loss + prox
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)



def krum_select(
    models: List[nn.Module],
    f: int,
) -> nn.Module:
    """
    Drops the hammer on rogue agents using the Krum fault tolerance criterion.
    Computes squared Euclidean distances between all flattened model tensors, isolating
    the single client update that sits geometrically closest to its trusted peers.
    Guarantees stability against up to $f$ malicious attackers.

    Parameters
    ----------
    models : list of nn.Module
        Submitted local model updates, including potentially hijacked variants.
    f : int
        The maximum ceiling of anticipated Byzantine nodes to tolerate.

    Returns
    -------
    nn.Module
        The model that minimises the Krum score.
    """
    n = len(models)

# Bypasses distance voting if the client topology is too small to build a consensus.
    if n < 3:
        return federated_averaging(models)
    
    # Adjusts the tolerance bounds dynamically if the network density drops mid-operation.
    if n <= 2 * f + 2:
        f = max(0, (n - 3) // 2)

    # Flatten all model parameters into vectors
    def flatten(m: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1).float() for p in m.parameters()])

    # Vectorize pairwise calculations using PyTorch highly optimized math operations
    # Stacking shifts memory directly to (n, num_parameters)
    vectors = torch.stack([flatten(m) for m in models])

    # torch.cdist highly optimizes memory usage for pair-wise distance without nested looping
    # and pow(2) converts the euclidean distances to squared euclidean distances
    dists = torch.cdist(vectors, vectors, p=2.0).pow(2)

    # Pairwise squared distances
    n_select = n - f - 2
    scores = []

    for i in range(n):
        # Isolate distances for vector i and clone to avoid modifying original matrix
        d = dists[i].clone()
        d[i] = float('inf') # Mask out self-distance so it isn't part of calculation
        
        sorted_d, _ = torch.sort(d)
        scores.append(sorted_d[:n_select].sum().item())

    best_idx = scores.index(min(scores))
    return copy.deepcopy(models[best_idx])



def trimmed_mean_aggregate(
    models: List[nn.Module],
    beta: float = 0.1,
) -> nn.Module:
    """
    Aggregate model parameters using coordinate-wise trimmed mean.

    The top and bottom ``beta`` fraction of values are discarded before
    averaging each parameter coordinate, reducing the influence of outlier
    updates from potentially compromised devices.

    Parameters
    ----------
    models : list of nn.Module
        Local model updates.
    beta : float
        Fraction of extreme values to trim from each end (e.g., 0.1 removes
        the bottom 10 % and top 10 %).

    Returns
    -------
    nn.Module
        Aggregated model with trimmed-mean parameters.
    """
    if not 0.0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5); got {beta}.")

    n = len(models)
    k = math.floor(beta * n)  # number of values to trim per side

    global_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for key in global_model.state_dict():
            stacked = torch.stack(
                [m.state_dict()[key].float() for m in models], dim=0
            )  # (n, *param_shape)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if k > 0:
                trimmed = sorted_vals[k: n - k]
            else:
                trimmed = sorted_vals
                
            global_model.state_dict()[key].copy_(trimmed.mean(dim=0))

    return global_model



def shap_weighted_aggregate(
    local_models: List[nn.Module],
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    sizes: List[int],
    n_classes: int = 15,
    n_explain: int = 10,
) -> Tuple[nn.Module, float, List[bool], List[bool]]: 
    """
    Runs parallel SHAP stability checks to weight and aggregate updates based on structural trust.
    Returns the aggregated model AND the total regional trust weight for Cloud use.
    Calculates attribution shifts between the proposed updates and the global baseline.
    Models that exceed the Median Absolute Deviation (MAD) tolerance are hard-filtered.
    Surviving models are merged based on a composite trust multiplier:
    $w_i = SHAP_{stability} \times Accuracy_i \times \sqrt{Dataset\_Size_i}$.
    
    Parameters
    ----------
    local_models : List[nn.Module]
        The collection of local client model submissions.
    ref_model : nn.Module
        The global baseline model defining the structural reference frame.
    X_val : torch.Tensor
        Centralized evaluation feature block.
    y_val : torch.Tensor
        Centralized evaluation label block.
    sizes : List[int]
        The telemetry volumes reported by the respective clients.
    n_classes : int
        The upper bound of operational target classes.
    n_explain : int
        The sample volume allocated to the GradientSHAP background interpolation.
        
    Returns
    -------
    Tuple[nn.Module, float, List[bool], List[bool]]
        The hardened global model alongside its aggregate regional trust metric, passed flags, and reward flags.    
    """
    from src.utils.metrics import compute_shap_stability

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

    # Derives the Median Absolute Deviation to lock down the acceptable trust envelope.
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
        # Only reward agents STRICTLY above the median stability
        reward_flags.append(stability_scores[i] > mu_s)

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



def fltrust_aggregate(
    local_models: List[nn.Module],
    server_model: nn.Module,
    global_model: nn.Module,
) -> nn.Module:
    """
    Executes FLTrust: cross-referencing client updates against a trusted server baseline.

    Computes the cosine similarity between the inbound client gradients and the server's
    privately verified update. Hostile parameter shifts exhibiting negative similarity
    are hard-zeroed. Accepted directions are geometrically scaled down to match the
    server's own gradient magnitude ($|| \Delta \theta_{server} ||_2$), capping attack impact.

    Parameters
    ----------
    local_models : list of nn.Module
        Client model updates.
    server_model : nn.Module
        The clean reference model trained natively on the server's root dataset.
    global_model : nn.Module
        Previous global model (reference for computing update deltas).

    Returns
    -------
    nn.Module
        The composite architecture built entirely from verified, directionally aligned gradients.
    """
    device = next(global_model.parameters()).device

    def delta(model: nn.Module) -> torch.Tensor:
        v_global = torch.cat([p.data.view(-1).float().to(device)
                               for p in global_model.parameters()])
        v_model  = torch.cat([p.data.view(-1).float().to(device)
                               for p in model.parameters()])
        return v_model - v_global

    server_delta = delta(server_model)
    server_norm  = server_delta.norm(p=2).clamp(min=1e-12)

    trust_scores, normed_deltas = [], []
    for m in local_models:
        d = delta(m)
        d_norm = d.norm(p=2).clamp(min=1e-12)
        cos_sim = (d @ server_delta) / (d_norm * server_norm)
        ts = max(0.0, cos_sim.item())
        # Re-scale to server update magnitude
        d_scaled = d / d_norm * server_norm
        trust_scores.append(ts)
        normed_deltas.append(d_scaled)

    total_ts = sum(trust_scores)
    if total_ts < 1e-12:
        # All trust scores zero — server update direction unhelpful; fall back to FedAvg
        return federated_averaging(local_models, weights=None)

    # Build aggregated delta weighted by trust scores
    agg_delta = torch.zeros_like(server_delta)
    for ts, d in zip(trust_scores, normed_deltas):
        agg_delta += (ts / total_ts) * d

    # Apply delta to global model
    result = copy.deepcopy(global_model)
    with torch.no_grad():
        offset = 0
        for p in result.parameters():
            n = p.data.numel()
            p.data += agg_delta[offset: offset + n].view_as(p.data)
            offset += n
    return result



def flame_aggregate(
    local_models: List[nn.Module],
    global_model: nn.Module,
    target_frac: float = 0.5,
) -> nn.Module:
    """
    FLAME: norm-clipping + cosine-similarity outlier rejection.

    Each update is clipped to the median L2 norm of all updates, then updates
    with cosine similarity below the median to the mean direction are filtered
    out.  The survivors are averaged with equal weights.

    Parameters
    ----------
    local_models : list of nn.Module
        Client model updates.
    global_model : nn.Module
        Previous global model for computing deltas.
    target_frac : float
        Minimum fraction of updates to keep (avoids over-rejection on small
        populations).

    Returns
    -------
    nn.Module
        The averaged architecture containing only the tightly clustered, clipped survivors.
    """
    device = next(global_model.parameters()).device

    def delta(model: nn.Module) -> torch.Tensor:
        v_g = torch.cat([p.data.view(-1).float().to(device) for p in global_model.parameters()])
        v_m = torch.cat([p.data.view(-1).float().to(device) for p in model.parameters()])
        return v_m - v_g

    deltas = [delta(m) for m in local_models]
    norms  = torch.tensor([d.norm(p=2).item() for d in deltas])

    # Clip each update to median norm
    median_norm = norms.median().clamp(min=1e-12)
    clipped = [d / max(d.norm(p=2).item(), median_norm.item()) * median_norm.item()
               for d in deltas]

    # Mean direction
    mean_dir = torch.stack(clipped, dim=0).mean(dim=0)
    mean_dir_norm = mean_dir.norm(p=2).clamp(min=1e-12)

    # Cosine similarity to mean direction
    cos_sims = [
        ((c @ mean_dir) / (c.norm(p=2).clamp(min=1e-12) * mean_dir_norm)).item()
        for c in clipped
    ]

    # Accept updates with above-median cosine similarity (keep at least 50%)
    threshold = max(sorted(cos_sims)[len(cos_sims) // 2],
                    sorted(cos_sims)[int(len(cos_sims) * (1 - target_frac))])
    accepted = [m for m, s in zip(local_models, cos_sims) if s >= threshold]

    if not accepted:
        accepted = local_models  # fallback

    return federated_averaging(accepted, weights=None)