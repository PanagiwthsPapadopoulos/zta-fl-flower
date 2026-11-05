from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _apply_static_adversarial_split(x, y, adv_ratio):
    if not math.isfinite(adv_ratio) or not 0 <= adv_ratio <= 1:
        raise ValueError("adv_ratio must be in [0, 1]")
    count = int(x.size(0) * adv_ratio)
    return count, x[:count], y[:count], x[count:]


@contextmanager
def _attack_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        # cuDNN LSTM backward requires training-mode reserves. Native autograd
        # supports eval-mode gradients without changing dropout or BN behavior.
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            yield
    finally:
        for module, training in modes:
            module.training = training


def _project_attack(candidate, original, eps, clip_min, clip_max):
    lower, upper = original - eps, original + eps
    # Expand feature bounds only enough to contain the original observation.
    # Held-out PCA features outside the training range must still obey epsilon.
    if clip_min is not None:
        lower = torch.maximum(lower, torch.minimum(original, torch.as_tensor(clip_min, device=original.device, dtype=original.dtype)))
    if clip_max is not None:
        upper = torch.minimum(upper, torch.maximum(original, torch.as_tensor(clip_max, device=original.device, dtype=original.dtype)))
    return torch.maximum(lower, torch.minimum(upper, candidate)).detach()


def _validate_attack(eps, clip_min, clip_max):
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("Attack epsilon must be finite and nonnegative")
    for bound in (clip_min, clip_max):
        if bound is not None and not math.isfinite(bound):
            raise ValueError("Attack bounds must be finite or None")
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clip_min must not exceed clip_max")


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1,
                adv_ratio: float = 0.3, clip_min: Optional[float] = None,
                clip_max: Optional[float] = None) -> torch.Tensor:
    """Input-only FGSM gradients; preserve parameter gradients and module modes."""
    _validate_attack(epsilon, clip_min, clip_max)
    count, original, targets, clean = _apply_static_adversarial_split(x, y, adv_ratio)
    if count < 1 or epsilon == 0:
        return x.detach().clone()
    original = original.detach()
    with _attack_mode(model):
        attacked = original.clone().requires_grad_(True)
        loss = F.cross_entropy(model(attacked), targets)
        gradient, = torch.autograd.grad(loss, attacked)
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite FGSM input gradient")
        attacked = _project_attack(attacked + epsilon * gradient.sign(), original, epsilon, clip_min, clip_max)
    return torch.cat([attacked, clean.detach()], dim=0)


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float = 0.1,
               adv_ratio: float = 0.3, alpha: Optional[float] = None, n_iter: int = 7,
               clip_min: Optional[float] = None, clip_max: Optional[float] = None) -> torch.Tensor:
    """PGD projected onto epsilon bounds on every step, including random start."""
    _validate_attack(eps, clip_min, clip_max)
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("PGD n_iter must be a positive integer")
    alpha = 2.0 * eps / n_iter if alpha is None else alpha
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("PGD alpha must be finite and nonnegative")
    count, original, targets, clean = _apply_static_adversarial_split(x, y, adv_ratio)
    if count < 1 or eps == 0:
        return x.detach().clone()
    original = original.detach()
    # Author convention: clamp only the random start to this attack batch's
    # range. Subsequent steps project onto the epsilon ball, not [0, 1].
    start = (original + torch.empty_like(original).uniform_(-eps, eps)).clamp(original.min(), original.max())
    attacked = _project_attack(start, original, eps, clip_min, clip_max)
    with _attack_mode(model):
        for _ in range(n_iter):
            attacked = attacked.detach().requires_grad_(True)
            loss = F.cross_entropy(model(attacked), targets)
            gradient, = torch.autograd.grad(loss, attacked)
            if not torch.isfinite(gradient).all():
                raise ValueError("Nonfinite PGD input gradient")
            attacked = _project_attack(attacked + alpha * gradient.sign(), original, eps, clip_min, clip_max)
    return torch.cat([attacked, clean.detach()], dim=0)


def adversarial_train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                            adv_ratio: float = 0.3, eps: float = 0.1, alpha: float = 0.01,
                            n_iter: int = 7, device: str = "cpu", use_pgd: bool = True,
                            clip_min: Optional[float] = None, clip_max: Optional[float] = None, clip_norm: float = 1.0) -> float:
    """Train on one mixed batch, matching the author's BatchNorm/loss policy.

    Attack generation uses evaluation-mode input gradients without changing
    model state; the combined training pass updates BatchNorm once per batch.
    """
    model.train().to(device)
    total_loss, n_batches = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if len(inputs) < 2:
            continue
        combined = (pgd_attack(model, inputs, labels, eps, adv_ratio, alpha, n_iter, clip_min, clip_max)
                    if use_pgd else fgsm_attack(model, inputs, labels, eps, adv_ratio, clip_min, clip_max))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(combined), labels)
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
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None
) -> dict[str, float]:
    """
    Evaluates model robustness by comparing accuracy on clean data versus adversarial data.
    """
    if attack not in ("fgsm", "pgd"):
        raise ValueError("Robustness attack must be fgsm or pgd")
    model.eval()
    model.to(device)
    X, y = X.to(device), y.to(device)
    clean_correct, adv_correct, total = 0, 0, 0
    for start in range(0, X.size(0), batch_size):
        X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
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
    """Author sampling: selected labels are drawn uniformly from all classes."""
    if n_classes < 1 or not math.isfinite(p_flip) or not 0 <= p_flip <= 1:
        raise ValueError("Require positive n_classes and p_flip in [0, 1]")
    y_flipped = y.clone()
    mask = torch.rand(len(y), device=y.device) < p_flip
    count = int(mask.sum().item())
    if count:
        y_flipped[mask] = torch.randint(0, n_classes, (count,), device=y.device)
    return y_flipped


def gradient_manipulation_attack(model: nn.Module, scale: float = 10.0) -> None:
    """
    Manipulates model gradients by scaling them, typically to simulate a Byzantine attack.
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
    p_flip: float = 0.1,
    device: str = "cpu",
    lr: float = 0.001,
    clip_norm: float = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_model: Optional[nn.Module] = None,
    apply_post_attack: bool = True,
    labels_prepared: bool = False,
    adv_ratio: float = 0.0,
    eps: float = 0.1,
    alpha: float = 0.01,
    n_iter: int = 7,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    use_pgd: bool = False,
) -> float:
    """
    Simulates local training for a Byzantine (malicious) client applying data or gradient poisoning.
    When running multiple local epochs, pass apply_post_attack=(epoch == epochs - 1) and global_model
    so gradient manipulation scales the cumulative multi-epoch delta once instead of compounding per epoch.
    """
    model.train()
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0
    if global_model is not None:
        ref_params = [p.data.detach().clone().to(device) for p in global_model.parameters()]
    else:
        ref_params = [p.data.detach().clone() for p in model.parameters()]

    if attack == "label_flip":
        if not labels_prepared:
            loader = label_flip_loader(loader, n_classes, loader.batch_size, p_flip)
        if adv_ratio == 0:
            return local_train_honest(model, loader, device, lr, clip_norm, optimizer)
        return adversarial_train_epoch(
            model, loader, optimizer, adv_ratio=adv_ratio, eps=eps, alpha=alpha,
            n_iter=n_iter, device=device, use_pgd=use_pgd, clip_min=clip_min,
            clip_max=clip_max, clip_norm=clip_norm,
        )
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

    if apply_post_attack and attack in ["gradient_manip", "gradient_manipulation"]:
        with torch.no_grad():
            for p, p_ref in zip(model.parameters(), ref_params):
                p.data.copy_(p_ref + scale * (p.data - p_ref))
    return total_loss / max(1, n_batches)


def local_train_honest(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    lr: float = 0.001,
    clip_norm: float = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> float:
    """
    Performs standard, honest local training for a client in a federated learning setup.
    """
    model.train()
    model.to(device)
    if optimizer is None:
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
    # Select from the partition before shuffling/batch dropping, as in EdgeAgent.
    if isinstance(loader.dataset, TensorDataset):
        X_all, y_all = loader.dataset.tensors
    else:
        rows = list(loader.dataset)
        X_all = torch.stack([row[0] for row in rows])
        y_all = torch.stack([row[1] for row in rows])
    y_flipped = label_flip_attack(y_all, n_classes, p_flip)
    return DataLoader(TensorDataset(X_all, y_flipped), batch_size=batch_size,
                      shuffle=True, drop_last=loader.drop_last)


def local_train_proximal(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    lr: float = 0.001,
    clip_norm: float = 1.0,
    mu: float = 0.01,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> float:
    """
    Performs FedProx training, adding an L2 proximal term to penalize weights
    drifting too far from the global model.
    """
    model.train()
    model.to(device)
    global_model.eval().to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        if len(X_b) < 2:
            continue

        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)

        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t.detach()).square().sum()
        loss += (mu / 2) * proximal_term

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)
