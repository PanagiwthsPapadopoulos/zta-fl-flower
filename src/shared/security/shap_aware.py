"""Paper VIII-B poisoning objective with an explicit raw SHAP-distance bound.

The paper does not specify L_poison, tau, optimizer or attacker's data access.
Here L_poison is poisoned-label CE or negative clean-label CE. Feasible
backtracking SGD checks the actual encoded update after every proposed step.
This is a local numerical solver, not the unavailable authors' experiment runner.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.shared.data.data_loader import DATASET_METADATA, get_dataset
from src.shared.network.compression import compress_weights
from src.shared.network.model_payload import load_model_arrays, validate_model
from src.shared.security.adversarial_math import _attack_mode, label_flip_attack
from src.shared.utils.metrics import compute_shap_attributions


def attack_validation_data(config, fog_num):
    """White-box assumption: attacker knows its Fog's validation sample/seed.

    Match both server adapters' selection, without reading the test split or
    using poisoned training labels to calculate the SHAP constraint.
    """
    name = str(config['dataset']).lower()
    seed = int(config['random_seed'])
    X, y, _ = get_dataset(
        name, config['dataset_path'], DATASET_METADATA[name]['classes'], seed,
        bool(config['simulate_global_leakage']), bool(config['apply_smote']), 'val',
        float(config['test_split']), float(config['val_split']),
    )
    count = int(config['shap_val_samples'])
    if count < 1 or len(y) == 0:
        raise ValueError('SHAP attack requires nonempty Fog validation data')
    fraction = float(config['dataset_fraction'])
    if fraction < 1:
        limit = max(count, int(len(X) * fraction))
        X, y = X[:limit], y[:limit]
    indices = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + fog_num))[:count]
    return X[indices], y[indices]


def train_shap_constrained(
    model, reference, loader, validation_data, *, attack, n_classes,
    constrained, tau, epochs, lr, clip_norm, n_explain, seed,
    quantization_bits, p_flip=1., max_backtracks=10, reference_arrays=None,
):
    """Minimize poisoning loss with strict ||phi(candidate)-phi(reference)|| < tau.

    Each minibatch proposes a clipped gradient step. Backtracking halves its
    length until the decoded candidate improves that minibatch's poison loss
    and (if constrained) satisfies the raw distance bound. Otherwise the last
    feasible iterate is retained. BN buffers are frozen; no post-hoc blending,
    optimizer momentum, normalized stability target or Fog acceptance oracle.

    Both experiment modes use identical objective/data/gradient budgets. The
    unconstrained control removes only the SHAP feasibility test. Neither mode
    promises a global optimum. Returned arrays are exactly the checked payload.
    """
    if not isinstance(constrained, bool):
        raise ValueError('shap_aware_constrained must be a boolean')
    if constrained and (tau is None or not math.isfinite(tau) or tau <= 0):
        raise ValueError('Set shap_aware_tau to a finite positive raw SHAP-distance bound; the paper supplies no value')
    if attack not in ('label_flip', 'gradient_manip', 'gradient_manipulation'):
        raise ValueError('Unsupported SHAP poisoning objective')
    if epochs < 1 or n_explain < 1 or max_backtracks < 0:
        raise ValueError('Require positive epochs/explained count and nonnegative backtracking budget')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError('SHAP attack requires finite positive learning rate and gradient clip')
    if validation_data is None:
        raise ValueError('SHAP attack requires the owning Fog validation sample')
    if not isinstance(loader.dataset, TensorDataset) or len(loader.dataset.tensors) != 2:
        raise ValueError('SHAP attack requires a tensor partition with inputs and labels')
    X, clean_y = loader.dataset.tensors
    if len(clean_y) == 0:
        raise ValueError('SHAP attack requires a nonempty training partition')
    device = next(model.parameters()).device
    X_val, y_val = validation_data
    if len(y_val) == 0 or len(X_val) != len(y_val):
        raise ValueError('SHAP attack requires nonempty matching validation tensors')
    # Poison once per round, preserving clean validation targets. The negative-CE
    # objective is an explicit untargeted alternative, not the separate delta-
    # scaling attack and does not use gradient_alpha or feature augmentation.
    targets = label_flip_attack(clean_y, n_classes, p_flip) if attack == 'label_flip' else clean_y.clone()
    batches = DataLoader(TensorDataset(X, targets), batch_size=loader.batch_size,
                         shuffle=True, drop_last=False,
                         generator=torch.Generator().manual_seed(seed))
    sign = 1. if attack == 'label_flip' else -1.

    def objective(candidate, x, y):
        value = sign * F.cross_entropy(candidate(x), y)
        if not torch.isfinite(value):
            raise ValueError('Nonfinite SHAP poisoning loss')
        return value

    def arrays_for(candidate):
        validate_model(candidate)
        arrays = [v.detach().cpu().numpy().copy() for v in candidate.state_dict().values()]
        return compress_weights(arrays, quantization_bits)

    # Work on copies for scoring; reference state/modes never change. Preserve
    # the exact incoming payload as the feasible no-progress fallback.
    reference_copy = copy.deepcopy(reference).to(device).eval()
    wire_model = copy.deepcopy(reference_copy)
    payload = (arrays_for(reference_copy) if reference_arrays is None
               else [a.copy() for a in reference_arrays])
    load_model_arrays(wire_model, payload, quantization_bits)
    if any(not torch.equal(v, wire_model.state_dict()[k]) for k, v in reference_copy.state_dict().items()):
        raise ValueError('SHAP reference must match the decoded incoming payload')
    model.load_state_dict(reference_copy.state_dict())
    ref_attrs = compute_shap_attributions(reference_copy, X_val, y_val, n_explain, device, seed)
    phi_ref = ref_attrs.reshape(len(ref_attrs), -1).abs().mean(0)

    def distance(candidate):
        attrs = compute_shap_attributions(candidate, X_val, y_val, n_explain, device, seed)
        phi = attrs.reshape(len(attrs), -1).abs().mean(0)
        value = float((phi - phi_ref).norm(p=2))
        if not math.isfinite(value):
            raise ValueError('Nonfinite SHAP constraint distance')
        return value

    def partition_loss(candidate):
        total = 0.
        with torch.no_grad():
            for start in range(0, len(targets), loader.batch_size):
                x, y = X[start:start + loader.batch_size].to(device), targets[start:start + loader.batch_size].to(device)
                total += float(objective(candidate, x, y)) * len(y)
        return total / len(targets)

    stats = dict(
        shap_attack_constrained=constrained,
        shap_attack_objective='poisoned_label_ce' if sign > 0 else 'negative_clean_ce',
        shap_attack_solver='feasible_backtracking_sgd',
        shap_attack_data_access='owning_fog_validation_and_seed',
        shap_attack_seed=seed, shap_attack_epochs=epochs, shap_attack_learning_rate=lr,
        shap_attack_clip_norm=clip_norm, shap_attack_max_backtracks=max_backtracks,
        shap_attack_validation_samples=len(y_val), shap_attack_explained_samples=min(n_explain, len(y_val)),
        shap_attack_quantization_bits=quantization_bits, shap_attack_p_flip=float(p_flip),
        shap_attack_steps=0, shap_attack_accepted_steps=0, shap_attack_rejected_steps=0,
        shap_attack_candidate_evaluations=0, shap_attack_initial_loss=partition_loss(wire_model),
    )
    if constrained:
        stats['shap_attack_tau'] = float(tau)
    params = list(model.parameters())
    with _attack_mode(model):
        for _ in range(epochs):
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                loss = objective(model, x, y)
                gradients = torch.autograd.grad(loss, params)
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError('Nonfinite SHAP poisoning gradient')
                norm = torch.stack([g.norm() for g in gradients]).norm()
                factor = min(1., clip_norm / (float(norm) + 1e-12))
                previous = [p.detach().clone() for p in params]
                with torch.no_grad():
                    previous_loss = float(objective(wire_model, x, y))
                stats['shap_attack_steps'] += 1
                accepted = False
                for backtrack in range(max_backtracks + 1):
                    with torch.no_grad():
                        for p, old, grad in zip(params, previous, gradients):
                            p.copy_(old - lr * (0.5 ** backtrack) * factor * grad)
                    proposal = arrays_for(model)
                    load_model_arrays(wire_model, proposal, quantization_bits)
                    stats['shap_attack_candidate_evaluations'] += 1
                    with torch.no_grad():
                        improved = float(objective(wire_model, x, y)) < previous_loss
                    if improved and (not constrained or distance(wire_model) < tau):
                        payload = proposal
                        stats['shap_attack_accepted_steps'] += 1
                        accepted = True
                        break
                if not accepted:
                    with torch.no_grad():
                        for p, old in zip(params, previous):
                            p.copy_(old)
                    load_model_arrays(wire_model, payload, quantization_bits)
                    stats['shap_attack_rejected_steps'] += 1
    # Recheck the exact payload, including when no step was feasible. Failure
    # aborts the update; it is never silently labelled a successful constraint.
    load_model_arrays(wire_model, payload, quantization_bits)
    final_distance = distance(wire_model)
    stats['shap_attack_distance'] = final_distance
    if constrained:
        stats['shap_attack_feasible'] = final_distance < tau
        if not stats['shap_attack_feasible']:
            raise ValueError('Final encoded SHAP attack violates the distance bound')
    stats['shap_attack_final_loss'] = partition_loss(wire_model)
    stats['shap_attack_status'] = 'optimized' if stats['shap_attack_accepted_steps'] else 'no_progress'
    model.load_state_dict(wire_model.state_dict())
    return stats['shap_attack_final_loss'], payload, stats
