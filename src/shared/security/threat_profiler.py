"""Seeded global attacker assignment under the paper's beta < 0.5 assumption.

Fog membership does not constrain placement. A region may have a malicious
majority even though the complete federation has an honest majority.
"""
import ast
import math
import random


def assign_edge_roles(run_config: dict, total_edges: int, global_index: int, master_seed: int, logger) -> str:
    custom = run_config.get("custom_fog_topology", "")
    custom = ast.literal_eval(custom) if isinstance(custom, str) and custom.strip() else custom
    topology = list(custom) if custom else [int(run_config["uniform_edges_per_fog"])] * int(run_config["num_fogs"])
    if sum(topology) != total_edges or any(n <= 0 for n in topology) or not 0 <= global_index < total_edges:
        raise ValueError("Invalid topology or global client index for role assignment")
    role_keys = (("backdoor", "backdoor_ratio"), ("label_flip", "label_flip_ratio"),
                 ("gradient_manip", "grad_manip_ratio"), ("shap_aware", "shap_aware_ratio"))
    ratios = [float(run_config[key]) for _, key in role_keys]
    if any(not math.isfinite(r) or r < 0 for r in ratios) or math.fsum(ratios) >= 0.5:
        raise ValueError("Attacker ratios must be finite, nonnegative, and sum to less than 0.5")
    counts = [round(total_edges * ratio) for ratio in ratios]
    # Preserve the configured per-role counts rather than silently clipping
    # them when rounding would violate the global experimental assumption.
    if 2 * sum(counts) >= total_edges:
        raise ValueError("Rounded attacker counts must remain below half of all edges. "
                         "Reduce attacker ratios or increase the global population.")
    rng = random.Random(master_seed)
    attackers = [role for (role, _), count in zip(role_keys, counts) for _ in range(count)]
    roles = attackers + ["benign"] * (total_edges - len(attackers))
    rng.shuffle(roles)
    return roles[global_index]


def experiment_ground_truth(run_config: dict) -> dict:
    """Server-owned labels for metrics only; never used to admit or reject edges.

    The retained paper poisoning attacks use legitimate enrolled identities.
    None is an invalid-attestation scenario.
    """
    custom = run_config.get("custom_fog_topology", "")
    custom = ast.literal_eval(custom) if isinstance(custom, str) and custom.strip() else custom
    topology = list(custom) if custom else [int(run_config["uniform_edges_per_fog"])] * int(run_config["num_fogs"])
    truth, index = {}, 0
    for fog, size in enumerate(topology, 1):
        for edge in range(1, size + 1):
            role = assign_edge_roles(run_config, sum(topology), index, int(run_config["random_seed"]), None)
            truth[f"[EDGE {fog}_{edge}]"] = {"role": role, "attestation_valid": True}
            index += 1
    return truth
