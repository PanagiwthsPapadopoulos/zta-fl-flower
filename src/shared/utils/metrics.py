from __future__ import annotations
import copy
import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn


def federated_averaging(models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
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
                aggregated += w * model.state_dict()[key].float()
            global_model.state_dict()[key].copy_(aggregated)
    return global_model


def krum_select(models: List[nn.Module], f: int) -> nn.Module:
    n = len(models)
    if n < 3:
        return federated_averaging(models)
    if n <= 2 * f + 2:
        f = max(0, (n - 3) // 2)

    def flatten(m: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1).float() for p in m.parameters()])

    vectors = torch.stack([flatten(m) for m in models])
    dists = torch.cdist(vectors, vectors, p=2.0).pow(2)
    scores = []

    for i in range(n):
        d = dists[i].clone()
        d[i] = float('inf')
        sorted_d, _ = torch.sort(d)
        scores.append(sorted_d[:n - f - 2].sum().item())

    return copy.deepcopy(models[scores.index(min(scores))])


def trimmed_mean_aggregate(models: List[nn.Module], beta: float = 0.1) -> nn.Module:
    n, k = len(models), math.floor(beta * len(models))
    global_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for key in global_model.state_dict():
            stacked = torch.stack([m.state_dict()[key].float() for m in models], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k: n - k] if k > 0 else sorted_vals
            global_model.state_dict()[key].copy_(trimmed.mean(dim=0))
    return global_model


def fltrust_aggregate(local_models: List[nn.Module], server_model: nn.Module, global_model: nn.Module) -> nn.Module:
    device = next(global_model.parameters()).device
    def delta(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1).float().to(device) for p in model.parameters()]) - torch.cat([p.data.view(-1).float().to(device) for p in global_model.parameters()])

    server_delta = delta(server_model)
    server_norm = server_delta.norm(p=2).clamp(min=1e-12)
    trust_scores, normed_deltas = [], []

    for m in local_models:
        d = delta(m)
        d_norm = d.norm(p=2).clamp(min=1e-12)
        ts = max(0.0, ((d @ server_delta) / (d_norm * server_norm)).item())
        trust_scores.append(ts)
        normed_deltas.append(d / d_norm * server_norm)

    total_ts = sum(trust_scores)
    if total_ts < 1e-12:
        return federated_averaging(local_models, weights=None)

    agg_delta = torch.zeros_like(server_delta)
    for ts, d in zip(trust_scores, normed_deltas):
        agg_delta += (ts / total_ts) * d

    result = copy.deepcopy(global_model)
    with torch.no_grad():
        offset = 0
        for p in result.parameters():
            n = p.data.numel()
            p.data += agg_delta[offset: offset + n].view_as(p.data)
            offset += n
    return result


def flame_aggregate(local_models: List[nn.Module], global_model: nn.Module, target_frac: float = 0.5) -> nn.Module:
    device = next(global_model.parameters()).device
    def delta(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1).float().to(device) for p in model.parameters()]) - torch.cat([p.data.view(-1).float().to(device) for p in global_model.parameters()])

    deltas = [delta(m) for m in local_models]
    norms = torch.tensor([d.norm(p=2).item() for d in deltas])
    median_norm = norms.median().clamp(min=1e-12).item()
    clipped = [d / max(d.norm(p=2).item(), median_norm) * median_norm for d in deltas]

    mean_dir = torch.stack(clipped, dim=0).mean(dim=0)
    mean_dir_norm = mean_dir.norm(p=2).clamp(min=1e-12)
    cos_sims = [((c @ mean_dir) / (c.norm(p=2).clamp(min=1e-12) * mean_dir_norm)).item() for c in clipped]

    threshold = max(sorted(cos_sims)[len(cos_sims) // 2], sorted(cos_sims)[int(len(cos_sims) * (1 - target_frac))])
    accepted = [m for m, s in zip(local_models, cos_sims) if s >= threshold] or local_models
    return federated_averaging(accepted, weights=None)


def disable_dropout(model: nn.Module) -> dict:
    original_probs = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_probs[name] = module.p
            module.p = 0.0  
    return original_probs


def restore_dropout(model: nn.Module, original_probs: dict):
    for name, module in model.named_modules():
        if name in original_probs:
            module.p = original_probs[name]


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    import torchmetrics.functional as tmf
    
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)
    if y_true.dim() > 1:
        y_true = y_true.argmax(dim=-1)
        
    y_true = y_true.long().flatten()
    y_pred = y_pred.long().flatten()
    
    num_classes = max(y_true.max(), y_pred.max()).item() + 1
    num_classes = max(2, num_classes)
    
    return float(tmf.accuracy(y_pred, y_true, task="multiclass", num_classes=num_classes))


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: Optional[int] = None) -> float:
    import torchmetrics.functional as tmf
    
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)
    if y_true.dim() > 1:
        y_true = y_true.argmax(dim=-1)
        
    y_true = y_true.long().flatten()
    y_pred = y_pred.long().flatten()
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()).item() + 1
    n_classes = max(2, n_classes)
    
    return float(tmf.f1_score(y_pred, y_true, task="multiclass", num_classes=n_classes, average="macro"))


def compute_shap_stability(model: nn.Module, ref_model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, n_explain: int = 50, n_classes: int = 15, device: str = "cpu") -> float:
    from captum.attr import GradientShap

    n_explain = min(n_explain, X_val.shape[0])
    X_sub, y_sub, background_set = X_val[:n_explain].to(device), y_val[:n_explain].to(device), X_val.to(device)

    # Disable dropout to ensure deterministic calculations, though eval mode often covers this.
    orig_probs_model, orig_probs_ref = disable_dropout(model), disable_dropout(ref_model)
    
    # Ensure models are in evaluation mode to freeze batch normalization layers
    model.eval()
    ref_model.eval()

    try:
        gs_model = GradientShap(model)
        gs_ref = GradientShap(ref_model)
        
        # Calculate attributions utilizing Captum directly
        attrs_model = gs_model.attribute(X_sub, baselines=background_set, target=y_sub)
        attrs_ref = gs_ref.attribute(X_sub, baselines=background_set, target=y_sub)
    finally:
        restore_dropout(model, orig_probs_model)
        restore_dropout(ref_model, orig_probs_ref)

    # Flatten the multi-dimensional attributions per sample before calculating the L2 norm
    attrs_model_flat = attrs_model.view(attrs_model.size(0), -1)
    attrs_ref_flat = attrs_ref.view(attrs_ref.size(0), -1)
    
    # Custom implementation calculation along the flattened feature dimension
    stability_scores = 1.0 - ((attrs_model_flat - attrs_ref_flat).norm(p=2, dim=1) / (attrs_ref_flat.norm(p=2, dim=1) + 1e-8))
    
    return float(stability_scores.mean().item())

def roc_auc_multiclass(y_true: torch.Tensor, y_pred_probs: torch.Tensor, n_classes: int) -> float:
    import torchmetrics.functional as tmf
    # y_pred_probs must be raw probabilities (post-softmax), not argmax indices
    y_true = y_true.long().flatten()
    # Using 'macro' average and 'ovr' (One-vs-Rest) strategy
    return float(tmf.auroc(y_pred_probs, y_true, task="multiclass", num_classes=n_classes, average="macro")) 