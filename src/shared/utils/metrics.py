from __future__ import annotations
import copy
from contextlib import nullcontext
import math
from typing import Optional, List
import numpy as np
import torch
import threading
import torch.nn as nn
from src.shared.utils.hardware import get_device

_SHAP_RNG_LOCK = threading.Lock()

def federated_averaging(models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
    if len(models) == 0:
        raise ValueError("At least one model is required for aggregation.")
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    if len(weights) != len(models) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Aggregation weights must be finite, nonnegative, and match models")
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Aggregation requires positive total weight")
    weights = [w / total for w in weights]
    from src.shared.network.model_payload import validate_model
    for model in models:
        validate_model(model)

    global_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for key in global_model.state_dict():
            aggregated = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
            for model, w in zip(models, weights):
                aggregated += w * model.state_dict()[key].float()
            global_model.state_dict()[key].copy_(aggregated)
    validate_model(global_model)
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
    """Author convention: omit classes absent from both targets and predictions."""
    import torchmetrics.functional as tmf
    
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)
    if y_true.dim() > 1:
        y_true = y_true.argmax(dim=-1)
        
    y_true = y_true.long().flatten()
    y_pred = y_pred.long().flatten()
    
    if y_true.numel() == 0 and y_pred.numel() == 0:
        return 0.0
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()).item() + 1
    n_classes = max(2, n_classes)
    
    # Keep prediction-only classes: their false positives must count in macro-F1.
    per_class = tmf.f1_score(y_pred, y_true, task="multiclass", num_classes=n_classes,
                            average=None)
    support = (torch.bincount(y_true, minlength=n_classes)
               + torch.bincount(y_pred, minlength=n_classes)) > 0
    return float(per_class[support].mean())


def compute_shap_attributions(model, X_val, y_val, n_explain=50, device="cpu", seed=42):
    """Serialize Captum, replay both RNG streams, and restore caller RNG/model state."""
    from captum.attr import GradientShap
    n_explain = min(n_explain, X_val.shape[0])
    if n_explain < 1:
        raise ValueError("SHAP requires nonempty validation inputs")
    X_sub = X_val[:n_explain].to(device)
    y_sub = y_val[:n_explain].to(device)
    background = X_val.to(device)
    devices = [X_sub.device.index] if X_sub.is_cuda else []
    with _SHAP_RNG_LOCK:
        numpy_state = np.random.get_state()
        mps_state = torch.mps.get_rng_state() if X_sub.device.type == "mps" else None
        training = model.training
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if X_sub.is_cuda:
                    torch.cuda.default_generators[X_sub.device.index].manual_seed(seed)
                if mps_state is not None:
                    torch.mps.manual_seed(seed)
                np.random.seed(seed)
                guard = torch.backends.cudnn.flags(enabled=False) if X_sub.is_cuda else nullcontext()
                with guard:
                    attrs = GradientShap(model).attribute(X_sub, baselines=background, target=y_sub).detach()
            if not torch.isfinite(attrs).all():
                raise ValueError("Nonfinite SHAP attributions")
            return attrs
        finally:
            np.random.set_state(numpy_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
            model.train(training)


def compute_shap_stability(
    model: nn.Module,
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_explain: int = 50,
    n_classes: int = 15,
    device: str = "cpu",
    precomputed_attrs_ref: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> float:
    attrs_model = compute_shap_attributions(model, X_val, y_val, n_explain, device, seed)
    attrs_ref = (compute_shap_attributions(ref_model, X_val, y_val, n_explain, device, seed)
                 if precomputed_attrs_ref is None else precomputed_attrs_ref.to(device))
    if attrs_model.shape != attrs_ref.shape or not torch.isfinite(attrs_ref).all():
        raise ValueError("Invalid reference SHAP attributions")
    # Paper V-B compares one feature-importance vector per model. Mean absolute
    # attribution over the matched examples defines that vector; the paper does
    # not specify this reduction. Preserve negative scores for the population test.
    phi_model = attrs_model.reshape(attrs_model.size(0), -1).abs().mean(dim=0)
    phi_ref = attrs_ref.reshape(attrs_ref.size(0), -1).abs().mean(dim=0)
    ratio = (phi_model - phi_ref).norm(p=2) / (phi_ref.norm(p=2) + 1e-6)
    if not torch.isfinite(ratio):
        raise ValueError("Nonfinite SHAP stability ratio")
    return float((1.0 - ratio).item())


def roc_auc_multiclass(y_true: torch.Tensor, y_pred_probs: torch.Tensor, n_classes: int) -> Optional[float]:
    """Macro OVR ROC-AUC over classes with positives and negatives in the test set."""
    return roc_auc_multiclass_details(y_true, y_pred_probs, n_classes)[0]


def roc_auc_multiclass_details(
    y_true: torch.Tensor, y_pred_probs: torch.Tensor, n_classes: int,
) -> tuple[Optional[float], dict]:
    """Return observed-class macro AUC and JSON-safe coverage/per-class diagnostics.

    Scores must be softmax probabilities across ALL configured outputs. Do not
    renormalize the observed columns: that changes the model's rankings.
    """
    import torchmetrics.functional as tmf
    if n_classes < 2:
        raise ValueError("ROC-AUC requires at least two configured classes")
    y_true = y_true.detach().flatten()
    probs = y_pred_probs.detach()
    if probs.ndim != 2 or probs.shape != (len(y_true), n_classes):
        raise ValueError("ROC-AUC requires one probability column per configured class")
    if not torch.isfinite(y_true).all() or (y_true != y_true.long()).any():
        raise ValueError("ROC-AUC targets must be finite integer class IDs")
    y_true = y_true.long().to(probs.device)
    if ((y_true < 0) | (y_true >= n_classes)).any():
        raise ValueError("ROC-AUC target class is outside the configured range")
    if not torch.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("ROC-AUC probabilities must be finite and within [0, 1]")
    if not torch.allclose(probs.sum(dim=1), torch.ones(len(y_true), device=probs.device, dtype=probs.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError("ROC-AUC probabilities must sum to one across all configured outputs")
    counts = torch.bincount(y_true, minlength=n_classes)
    eligible = torch.where((counts > 0) & (counts < len(y_true)))[0].tolist()
    details = {
        "method": "macro_ovr_observed_classes",
        "status": "defined" if eligible else ("empty_test_split" if len(y_true) == 0 else "insufficient_test_classes"),
        "configured_classes": n_classes,
        "evaluated_classes": eligible,
        "class_counts": counts.tolist(),
        "per_class_auc": [None] * n_classes,
    }
    if not eligible:
        return None, details
    for class_id in eligible:
        score = float(tmf.auroc(probs[:, class_id], (y_true == class_id).long(), task="binary"))
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite ROC-AUC calculation for class {class_id}")
        details["per_class_auc"][class_id] = score
    return math.fsum(details["per_class_auc"][c] for c in eligible) / len(eligible), details
