"""
Defines the empirical evaluation calculations applied during experimentation routines.

Facilitates the generation of classification accuracy arrays, macro-averaged F1 aggregates, 
and proprietary SHAP-aligned explanation stability statistics utilized by filtering aggregators.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def disable_dropout(model: nn.Module) -> dict:
    """
    Suppresses dropout application while retaining the network layer configuration within the active evaluation state.
    Records and returns the antecedent probabilities required for eventual restoration.
    """
    original_probs = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_probs[name] = module.p
            module.p = 0.0  
    return original_probs


def restore_dropout(model: nn.Module, original_probs: dict):
    """
    Reinstates historical dropout parameters to modules tracked within the configuration dictionary.
    """
    for name, module in model.named_modules():
        if name in original_probs:
            module.p = original_probs[name]


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates the proportional relationship of matching predictions to true categorical elements.
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)

    y_true_np = y_true.cpu().numpy().astype(int)
    y_pred_np = y_pred.cpu().numpy().astype(int)

    return float((y_true_np == y_pred_np).mean())


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: Optional[int] = None) -> float:
    """
    Aggregates a generalized F1 composite score by averaging the individual harmonic mean across all target classes.
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)

    y_true_np = y_true.cpu().numpy().astype(int)
    y_pred_np = y_pred.cpu().numpy().astype(int)

    if n_classes is None:
        n_classes = max(y_true_np.max(), y_pred_np.max()) + 1

    f1_scores = []
    for c in range(n_classes):
        tp = int(((y_pred_np == c) & (y_true_np == c)).sum())
        fp = int(((y_pred_np == c) & (y_true_np != c)).sum())
        fn = int(((y_pred_np != c) & (y_true_np == c)).sum())

        if tp + fp == 0 and tp + fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_shap_stability(
    model: nn.Module,
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_explain: int = 50,
    n_classes: int = 15,
    device: str = "cpu",
) -> float:
    """
    Derives an interpretability comparison metric measuring structural consistency between a localized variant and the global baseline.
    Computes differences bounded by GradientSHAP interpolations utilizing standardized background sets.
    """
    
    n_explain = min(n_explain, X_val.shape[0])
    X_sub = X_val[:n_explain].to(device)
    y_sub = y_val[:n_explain].to(device)
    background_set = X_val.to(device)  

    def compute_gradientshap(m: nn.Module, x: torch.Tensor, y: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes an integrated attribution array mapping model inputs toward generated output trajectories via baseline sampling.
        """
        m.train()  
        n_steps = 10
        n_samples = 100  
        
        final_attributions = torch.zeros_like(x)

        for _ in range(n_samples):
            # Selects an arbitrary background representation from the provided context stack.
            idx = torch.randint(0, background.size(0), (1,))
            baseline = background[idx].expand_as(x)
            
            sample_attributions = torch.zeros_like(x)
            for step in range(1, n_steps + 1):
                interp = baseline + (step / n_steps) * (x - baseline)
                interp = interp.detach().requires_grad_(True)
                logits = m(interp)
                
                scores = logits.gather(1, y.view(-1, 1)).squeeze()
                grad = torch.autograd.grad(scores.sum(), interp)[0]
                sample_attributions += grad.detach()

            sample_attributions = sample_attributions / n_steps
            sample_attributions = sample_attributions * (x - baseline)
            final_attributions += sample_attributions

        return final_attributions / n_samples 

    # Configures strict algorithmic execution state bypassing randomized architectural dropouts.
    orig_probs_model = disable_dropout(model)
    orig_probs_ref = disable_dropout(ref_model)
    
    try:
        attrs_model = compute_gradientshap(model, X_sub, y_sub, background_set)
        attrs_ref = compute_gradientshap(ref_model, X_sub, y_sub, background_set)
    finally:
        # Ensures reversion to original state parameters regardless of computational exceptions.
        restore_dropout(model, orig_probs_model)
        restore_dropout(ref_model, orig_probs_ref)

    diff_norm = (attrs_model - attrs_ref).norm(p=2, dim=-1)
    ref_norm = attrs_ref.norm(p=2, dim=-1)
    
    epsilon = 1e-8
    stability_scores = 1.0 - (diff_norm / (ref_norm + epsilon))
    
    return float(stability_scores.mean().item())