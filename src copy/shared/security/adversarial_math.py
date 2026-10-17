from __future__ import annotations
from typing import Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def fgsm_attack(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    epsilon: float = 0.1,     # Perturbation budget
    adv_ratio: float = 0.3,   # Percentage of the attacker's batch to poison 
    clip_min: float = 0.0, 
    clip_max: float = 1.0
) -> torch.Tensor:
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).
    """
    # Calculate how many samples in the batch to replace based on the adv_ratio
    batch_size = x.size(0)
    num_adv = int(batch_size * adv_ratio)
    
    # If the ratio is 0, return the original untouched batch
    if num_adv <= 1:
        return x.clone()
        
    original_mode = model.training
    model.eval()
    
    # Split the batch into the portion to attack and the portion to keep clean
    x_to_attack = x[:num_adv].detach().clone().requires_grad_(True)
    y_to_attack = y[:num_adv]
    x_clean = x[num_adv:]
    
    # Generate logits and calculate loss only for the targeted subset
    logits = model(x_to_attack)
    loss = F.cross_entropy(logits, y_to_attack)
    
    model.zero_grad()
    loss.backward()
    
    # Apply the FGSM perturbation using the designated epsilon budget
    perturbation = epsilon * x_to_attack.grad.sign()
    
    # Apply the Clip function to the adversarial examples
    x_adv = torch.clamp((x_to_attack + perturbation).detach(), clip_min, clip_max)
    
    model.train(original_mode)
    
    # Recombine the adversarial examples with the clean examples
    if num_adv < batch_size:
        x_combined = torch.cat([x_adv, x_clean], dim=0)
    else:
        x_combined = x_adv
        
    return x_combined


def pgd_attack(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    eps: float = 0.1,             # Target perturbation budgets tested: 0.05, 0.1, 0.15, 0.2
    adv_ratio: float = 0.3,       # 30% adversarial subset as specified in the paper
    alpha: Optional[float] = None, 
    n_iter: int = 7,              # Default matches "PGD-7"; change to 20 to test "PGD-20"
    clip_min: float = 0.0, 
    clip_max: float = 1.0
) -> torch.Tensor:
    """
    Generates adversarial examples using Projected Gradient Descent (PGD).
    """
    # Calculate how many samples in the batch to attack based on the ratio
    batch_size = x.size(0)
    num_adv = int(batch_size * adv_ratio)
    
    # If the ratio is 0%, return the original untouched batch
    if num_adv <= 1:
        return x.clone()
        
    if alpha is None:
        alpha = 2.0 * eps / n_iter
        
    original_mode = model.training
    model.eval()
    
    # Split the batch into the portion to attack and the portion to keep clean
    x_to_attack = x[:num_adv].detach().clone()
    y_to_attack = y[:num_adv]
    x_clean = x[num_adv:]
    
    # Initialize the adversarial examples with random uniform noise
    x_adv = x_to_attack.clone()
    x_adv = torch.clamp(x_adv + torch.empty_like(x_adv).uniform_(-eps, eps), clip_min, clip_max)

    # Perform the PGD steps
    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y_to_attack)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            step = alpha * x_adv.grad.sign()
            # Apply the step and project back into the L-infinity ball around the original image
            x_adv = torch.clamp(torch.max(x_to_attack - eps, torch.min(x_to_attack + eps, x_adv + step)), clip_min, clip_max)
            
    model.train(original_mode)
    
    # Recombine the adversarial examples with the clean examples (if any)
    if num_adv < batch_size:
        x_combined = torch.cat([x_adv.detach(), x_clean], dim=0)
    else:
        x_combined = x_adv.detach()
        
    return x_combined


def adversarial_train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, adv_ratio: float = 0.3, eps: float = 0.1, alpha: float = 0.01, n_iter: int = 7, device: str = "cpu", use_pgd: bool = True, clip_min: float = 0.0, clip_max: float = 1.0, clip_norm: float = 1.0) -> float:
    """
    Executes one epoch of adversarial training using either FGSM or PGD attacks.
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if X_batch.size(0) < 2:
            continue
        
                    
        if use_pgd:
            X_combined = pgd_attack(model, X_batch, y_batch, eps, adv_ratio, alpha, n_iter, clip_min, clip_max)
        else:
            X_combined = fgsm_attack(model, X_batch, y_batch, eps, adv_ratio, clip_min, clip_max)

        optimizer.zero_grad()
        logits = model(X_combined)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_robustness(model: nn.Module, X: torch.Tensor, y: torch.Tensor, attack: str = "fgsm", eps: float = 0.1, alpha: Optional[float] = None, n_iter: int = 7, batch_size: int = 256, device: str = "cpu", clip_min: float = 0.0, clip_max: float = 1.0) -> dict[str, float]:
    """
    Evaluates model robustness by comparing accuracy on clean data versus adversarial data.
    """
    model.eval()
    model.to(device)
    X, y = X.to(device), y.to(device)
    clean_correct, adv_correct, total = 0, 0, 0

    for start in range(0, X.size(0), batch_size):
        X_batch, y_batch = X[start: start + batch_size], y[start: start + batch_size]
        if X_batch.size(0) < 2:
            continue
        with torch.no_grad():
            clean_preds = model(X_batch).argmax(dim=-1)
        clean_correct += (clean_preds == y_batch).sum().item()

        X_adv = pgd_attack(model, X_batch, y_batch, eps, 1.0, alpha, n_iter, clip_min, clip_max) if attack == "pgd" else fgsm_attack(model, X_batch, y_batch, eps, 1.0, clip_min, clip_max)
        with torch.no_grad():
            adv_preds = model(X_adv).argmax(dim=-1)
        adv_correct += (adv_preds == y_batch).sum().item()
        total += y_batch.size(0)

    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)
    return {"clean_acc": clean_acc, "adv_acc": adv_acc, "acc_drop": clean_acc - adv_acc, "attack": attack, "eps": eps}


def label_flip_attack(y: torch.Tensor, n_classes: int, p_flip: float = 1.0) -> torch.Tensor:
    """
    Performs a label flipping attack by changing target labels to incorrect classes.
    """
    y_flipped = y.clone()
    for i in range(len(y)):
        if torch.rand(1).item() < p_flip:
            possible_flips = [c for c in range(n_classes) if c != y[i].item()]
            if possible_flips:
                y_flipped[i] = random.choice(possible_flips)
    return y_flipped


def gradient_manipulation_attack(model: nn.Module, scale: float = 10.0) -> None:
    """
    Manipulates model gradients by scaling them, typically to simulate a Byzantine attack.
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def local_train_byzantine(model: nn.Module, loader: DataLoader, attack: str, n_classes: int, scale: float = 10.0, p_flip: float = 0.1, device: str = "cpu", lr: float = 0.001, clip_norm: float = 1.0) -> float:
    """
    Simulates local training for a Byzantine (malicious) client applying data or gradient poisoning.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    if attack == "label_flip":
        loader = label_flip_loader(loader, n_classes, loader.batch_size, p_flip)

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        if len(X_b) < 2:
            continue
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        if attack in ["gradient_manip", "gradient_manipulation"]:
            gradient_manipulation_attack(model, scale)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def local_train_honest(model: nn.Module, loader: DataLoader, device: str = "cpu", lr: float = 0.001, clip_norm: float = 1.0) -> float:
    """
    Performs standard, honest local training for a client in a federated learning setup.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

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


def label_flip_loader(loader: DataLoader, n_classes: int, batch_size: int, p_flip: float = 1.0) -> DataLoader:
    """
    Creates a new DataLoader with flipped labels generated by the label flip attack.
    """
    X_all, y_all = [], []
    for X_b, y_b in loader:
        X_all.append(X_b)
        y_all.append(y_b)
        
    y_all_cat = torch.cat(y_all, dim=0)
    y_flipped = label_flip_attack(y_all_cat, n_classes, p_flip)
    
    return DataLoader(TensorDataset(torch.cat(X_all, dim=0), y_flipped), batch_size=batch_size, shuffle=True)


def local_train_shap_aware(
    model: nn.Module, 
    global_model: nn.Module, 
    loader: DataLoader, 
    attack: str, 
    n_classes: int, 
    shap_threshold: float = 0.1, 
    shap_explain_count: int = 10, 
    shap_val_samples: int = 100, 
    scale: float = 10.0, 
    p_flip: float = 1.0, 
    stealth_iterations: int = 10,
    blend_ratio: float = 0.8,
    device: str = "cpu",  
    lr: float = 0.001, 
    clip_norm: float = 1.0
) -> float:
    """
    Executes a Byzantine local training step that iteratively attempts to evade SHAP-based anomaly detection.
    """
    from src.shared.utils.metrics import compute_shap_stability
    X_bg, y_bg = next(iter(loader))
    
    # Cap the slice cleanly to prevent indexing out of bounds on tiny batches
    actual_samples = min(shap_val_samples, X_bg.size(0))
    X_bg, y_bg = X_bg[:actual_samples].to(device), y_bg[:actual_samples].to(device)
    global_model.eval().to(device)

    # Use explicit keyword arguments to completely avoid positional mismatches
    loss = local_train_byzantine(
        model=model, 
        loader=loader, 
        attack=attack, 
        n_classes=n_classes, 
        scale=scale, 
        p_flip=p_flip, 
        device=device, 
        lr=lr, 
        clip_norm=clip_norm
    )
    model.eval()
    
    # Use the dedicated stealth iteration count
    for _ in range(stealth_iterations):
        score = compute_shap_stability(model, global_model, X_bg, y_bg, n_explain=shap_explain_count, n_classes=n_classes, device=device)
        if (1.0 - score) < shap_threshold:
            break
        with torch.no_grad():
            for p_new, p_global in zip(model.parameters(), global_model.parameters()):
                # Dynamically apply the blend_ratio
                p_new.data = blend_ratio * p_new.data + (1.0 - blend_ratio) * p_global.data
    return loss

def local_train_proximal(model: nn.Module, global_model: nn.Module, loader: DataLoader, device: str = "cpu", lr: float = 0.001, clip_norm: float = 1.0, mu: float = 0.01) -> float:
    """
    Performs FedProx training, adding an L2 proximal term to penalize weights 
    drifting too far from the global model.
    """
    model.train()
    model.to(device)
    global_model.eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        if len(X_b) < 2:
            continue
            
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        
        # FedProx: Add the L2 proximal penalty
        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss += (mu / 2) * proximal_term
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        
    return total_loss / max(1, n_batches)