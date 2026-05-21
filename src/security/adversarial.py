"""
Implements adversarial training protocols and robustness assessment routines.

Provides programmatic definitions for Fast Gradient Sign Method and Projected Gradient 
Descent attacks mapped specifically to bypass evaluation mode restrictions associated 
with cuDNN-backed recurrent layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.01,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Executes a single-step Fast Gradient Sign Method perturbation derived from 
    the cross-entropy loss gradient of the provided input batch.

    Parameters
    ----------
    model : nn.Module
        The neural network targeted by the attack.
    x : torch.Tensor
        Original feature inputs.
    y : torch.Tensor
        True class labels corresponding to the inputs.
    alpha : float
        Perturbation amplitude defining the L-infinity boundaries.
    clip_min : float
        Lower threshold for valid feature representation.
    clip_max : float
        Upper threshold for valid feature representation.

    Returns
    -------
    torch.Tensor
        The adversarially modified input tensor.
    """
    original_mode = model.training
    model.train()

    x_adv = x.detach().clone().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    perturbation = alpha * x_adv.grad.sign()
    x_adv = (x_adv + perturbation).detach()
    
    # Enforces global boundary clipping rather than localized batch clipping 
    # to maintain an accurate adversarial representation context.
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    model.train(original_mode)
    return x_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.1,
    alpha: Optional[float] = None,
    n_iter: int = 7,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Executes a multi-step Projected Gradient Descent attack that recursively projects 
    the adversarial tensor onto an L-infinity ball around the authentic input data.

    Parameters
    ----------
    model : nn.Module
        The neural network targeted by the attack.
    x : torch.Tensor
        Original feature inputs.
    y : torch.Tensor
        True class labels corresponding to the inputs.
    eps : float
        The maximum L-infinity perturbation radius permitted.
    alpha : float, optional
        Iterative step scale. Defaults automatically based on epsilon and iteration count.
    n_iter : int
        The total number of gradient descent loops executed.
    clip_min : float
        Lower boundary limit for feature bounds.
    clip_max : float
        Upper boundary limit for feature bounds.

    Returns
    -------
    torch.Tensor
        The resulting adversarial feature tensor following iterative manipulation.
    """
    if alpha is None:
        alpha = 2.0 * eps / n_iter

    original_mode = model.training
    model.train()

    x_adv = x.detach().clone()
    
    # Distributes a randomized starting offset bounded by the epsilon threshold.
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            step = alpha * x_adv.grad.sign()
            x_adv = x_adv + step
            
            # Constrains the running modification back onto the L-infinity ball origin.
            x_adv = torch.max(x - eps, torch.min(x + eps, x_adv))
            
            # Reapplies global boundary checks to prevent domain exhaustion.
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

    model.train(original_mode)
    return x_adv.detach()


def adversarial_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adv_ratio: float = 0.3,
    eps: float = 0.1,
    alpha: float = 0.01,
    n_iter: int = 7,
    device: str = "cpu",
    use_pgd: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    clip_norm: float = 1.0,
) -> float:
    """
    Iterates through a training epoch while substituting a defined fraction of the 
    clean batch data with generated adversarial examples.

    Parameters
    ----------
    model : nn.Module
        The active neural network executing the training pass.
    loader : DataLoader
        The iterator supplying the batch data.
    optimizer : torch.optim.Optimizer
        The target optimization algorithm bound to the neural network.
    adv_ratio : float
        The decimal fraction defining the distribution of adversarial to clean samples.
    eps : float
        The overall perturbation capacity limit.
    alpha : float
        The step increment scalar for the perturbation algorithm.
    n_iter : int
        The cycle count implemented specifically for projected attacks.
    device : str
        Hardware environment string mapping.
    use_pgd : bool
        Determines the attack methodology applied (PGD or FGSM).
    clip_min : float
        Absolute minimum input constraint.
    clip_max : float
        Absolute maximum input constraint.
    clip_norm : float
        Gradient norm clipping parameter utilized prior to stepping the optimizer.

    Returns
    -------
    float
        The calculated mean training loss measured over the entire sequence.
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if X_batch.size(0) < 2:
            continue

        batch_size = X_batch.size(0)
        n_adv = int(batch_size * adv_ratio)

        if n_adv >= 2:
            X_clean = X_batch[n_adv:]
            y_clean = y_batch[n_adv:]
            X_sub = X_batch[:n_adv]
            y_sub = y_batch[:n_adv]

            if use_pgd:
                X_adv = pgd_attack(
                    model=model, x=X_sub, y=y_sub, 
                    eps=eps, alpha=alpha, n_iter=n_iter, 
                    clip_min=clip_min, clip_max=clip_max
                )
            else:
                X_adv = fgsm_attack(
                    model=model, x=X_sub, y=y_sub, 
                    alpha=alpha, clip_min=clip_min, clip_max=clip_max
                )

            X_combined = torch.cat([X_adv, X_clean], dim=0)
            y_combined = torch.cat([y_sub, y_clean], dim=0)
        else:
            X_combined = X_batch
            y_combined = y_batch

        if X_combined.size(0) < 2:
            continue

        model.train()
        optimizer.zero_grad()
        logits = model(X_combined)
        loss = criterion(logits, y_combined)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_robustness(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    attack: str = "fgsm",
    eps: float = 0.1,
    alpha: Optional[float] = None,
    n_iter: int = 7,
    batch_size: int = 256,
    device: str = "cpu",
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> dict[str, float]:
    """
    Measures the classification capability of a model against synthetically manufactured 
    adversarial variants. Bypasses strict evaluation mode standards to accommodate 
    recurrent hardware acceleration constraints.

    Parameters
    ----------
    model : nn.Module
        The target model analyzed for resilience.
    X : torch.Tensor
        Clean evaluation inputs.
    y : torch.Tensor
        Authentic evaluation labels.
    attack : str
        Algorithm utilized to formulate perturbations.
    eps : float
        Maximum deviation budget allocated to the attack.
    alpha : Optional[float]
        Iteration increment scalar.
    n_iter : int
        Descent repetition count applicable for multi-step approaches.
    batch_size : int
        Sample density compiled per sequential chunk.
    device : str
        Execution backend identifier.
    clip_min : float
        Minimum bounded data envelope.
    clip_max : float
        Maximum bounded data envelope.

    Returns
    -------
    dict
        A mapping containing clean accuracy, compromised accuracy, and the total degradation value.
    """
    model.train()
    model.to(device)

    X = X.to(device)
    y = y.to(device)

    clean_correct = 0
    adv_correct = 0
    total = 0

    for start in range(0, X.size(0), batch_size):
        X_batch = X[start: start + batch_size]
        y_batch = y[start: start + batch_size]

        if X_batch.size(0) < 2:
            continue

        with torch.no_grad():
            clean_logits = model(X_batch)
        clean_preds = clean_logits.argmax(dim=-1)
        clean_correct += (clean_preds == y_batch).sum().item()

        if attack == "pgd":
            X_adv = pgd_attack(
                model=model, x=X_batch, y=y_batch, 
                eps=eps, alpha=alpha, n_iter=n_iter, 
                clip_min=clip_min, clip_max=clip_max
            )
        else:
            X_adv = fgsm_attack(
                model=model, x=X_batch, y=y_batch, 
                alpha=eps, clip_min=clip_min, clip_max=clip_max
            )

        with torch.no_grad():
            adv_logits = model(X_adv)
        adv_preds = adv_logits.argmax(dim=-1)
        adv_correct += (adv_preds == y_batch).sum().item()

        total += y_batch.size(0)

    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "acc_drop": clean_acc - adv_acc,
        "attack": attack,
        "eps": eps,
    }


import torch
import random


def label_flip_attack(
    y: torch.Tensor,
    n_classes: int,
    p_flip: float = 1.0,
) -> torch.Tensor:
    """
    Simulates a data poisoning attack sequence by mapping authentic labels to alternative 
    incorrect classifications via stochastic sampling.

    Parameters
    ----------
    y : torch.Tensor
        The authentic label distribution array.
    n_classes : int
        The theoretical limit of possible unique class states.
    p_flip : float
        The likelihood ratio dictating translation execution per element.

    Returns
    -------
    torch.Tensor
        A reconstructed label tensor demonstrating the applied corruption.
    """
    y_flipped = y.clone()
    
    for i in range(len(y)):
        if torch.rand(1).item() < p_flip:
            possible_flips = [c for c in range(n_classes) if c != y[i].item()]
            if possible_flips:
                y_flipped[i] = random.choice(possible_flips)
                
    return y_flipped


import torch.nn as nn


def gradient_manipulation_attack(
    model: nn.Module,
    scale: float = 10.0,
) -> None:
    """
    Modifies local parameters directly during a backward pass to intentionally overload 
    standard weight aggregation operations. Executed directly prior to the optimization step.

    Parameters
    ----------
    model : nn.Module
        The active neural model storing the populated gradient buffers.
    scale : float
        The arithmetic scalar applied uniformly across the stored gradients.
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)
     

def local_train_byzantine(
    model: nn.Module,
    loader: DataLoader,
    attack: str,
    n_classes: int,
    scale: float = 10.0,
    device: str = "cpu",
    epochs: int = 1,
    lr: float = 0.001,
    clip_norm: float = 1.0,
) -> None:
    """
    Simulates a compromised client node compiling a hostile update payload using a 
    provided injection or manipulation algorithm.

    Parameters
    ----------
    model : nn.Module
        The local network architecture instance.
    loader : DataLoader
        The iterator processing the compromised local datasets.
    attack : str
        Algorithm string identifying the malicious strategy.
    n_classes : int
        Limit of possible outcome classifications.
    scale : float
        Gradient multiplier applied during manipulation strikes.
    device : str
        Target execution hardware for processing.
    epochs : int
        Local sequence repetition count prior to dispatch.
    lr : float
        Learning rate regulating the convergence stepping.
    clip_norm : float
        Normalization capacity limit enacted to preserve general update integrity.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    if attack == "label_flip":
        loader = label_flip_loader(loader, n_classes, loader.batch_size)

    for _ in range(epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if len(X_b) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            if attack == "gradient_manipulation":
                gradient_manipulation_attack(model=model, scale=scale)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            
    return total_loss / max(1, n_batches)


def local_train_honest(
    model: nn.Module, 
    loader: DataLoader, 
    device: str = "cpu", 
    epochs: int = 1, 
    lr: float = 0.001, 
    clip_norm: float = 1.0, 
) -> float:
    """
    Executes a standard client learning loop absent of adversarial injections or alterations.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    for _ in range(epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if len(X_b) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            
    return total_loss / max(1, n_batches)


def label_flip_loader(loader: DataLoader, n_classes: int, batch_size: int) -> DataLoader:
    """
    Restructures an existing DataLoader output to replace all legitimate labels with 
    randomized incorrect classifications.
    """
    X_all, y_all = [], []
    for X_b, y_b in loader:
        X_all.append(X_b)
        y_all.append(y_b)
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    
    # Executes the total corruption sequence assigning random bounds across the arrays.
    y_flipped = torch.randint(0, n_classes, y_all.shape)
    return DataLoader(TensorDataset(X_all, y_flipped), batch_size=batch_size, shuffle=True)


def local_train_shap_aware(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    attack: str,
    n_classes: int,
    shap_threshold: float = 0.1,
    device: str = "cpu",
    epochs: int = 1,
    lr: float = 0.001,
    clip_norm: float = 1.0,
) -> float:
    """
    Performs a localized model training phase governed by a constrained optimization 
    strategy. Assesses whether generated hostile updates fall within an acceptable 
    SHAP profile limit; if not, retracts parameters incrementally toward the reference frame.

    Parameters
    ----------
    model : nn.Module
        The client model subjected to targeted adjustment.
    global_model : nn.Module
        The frozen reference standard distributed during initialization.
    loader : DataLoader
        The iterator processing the compromised local datasets.
    attack : str
        Algorithm string identifying the malicious strategy.
    n_classes : int
        Limit of possible outcome classifications.
    shap_threshold : float
        Tolerance fraction restricting deviation of explanation attributions.
    device : str
        Target execution hardware for processing.
    epochs : int
        Local sequence repetition count prior to constraint processing.
    lr : float
        Learning rate regulating the convergence stepping.
    clip_norm : float
        Normalization capacity limit enacted to preserve general update integrity.
        
    Returns
    -------
    float
        The calculated mean training loss evaluated during the modification cycle.
    """
    from src.utils.metrics import compute_shap_stability
    import copy

    # Compiles a minor sub-batch array utilized to calculate local feature attributions.
    X_bg, y_bg = next(iter(loader))
    X_bg, y_bg = X_bg[:50].to(device), y_bg[:50].to(device) 

    global_model.eval()
    global_model.to(device)

    # Implements the standard base adversarial strike routine.
    loss = local_train_byzantine(
        model=model, loader=loader, attack=attack, 
        n_classes=n_classes, device=device, epochs=epochs, 
        lr=lr, clip_norm=clip_norm
    )

    # Commences a multi-step verification sequence projecting extreme parameter modifications back into tolerance ranges.
    model.eval()
    max_projections = 10
    
    for _ in range(max_projections):
        score = compute_shap_stability(
            model, global_model, X_bg, y_bg, 
            n_explain=10, n_classes=n_classes, device=device
        )
        
        deviation = 1.0 - score
        
        # Validates whether the current model state adheres to the security constraint envelope.
        if deviation < shap_threshold:
            break 
            
        # Realigns the weight tensors toward the reference threshold.
        with torch.no_grad():
            for p_new, p_global in zip(model.parameters(), global_model.parameters()):
                p_new.data = 0.8 * p_new.data + 0.2 * p_global.data

    return loss