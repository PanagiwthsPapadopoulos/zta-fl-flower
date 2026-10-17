import random


def assign_edge_roles(run_config: dict, total_edges: int, global_index: int, master_seed: int, logger) -> str:
    """Computes strict assignments distributing adversarial identities across the edge grid."""
    backdoor_ratio = float(run_config["backdoor_ratio"])
    label_flip_ratio = float(run_config["label_flip_ratio"])
    grad_manip_ratio = float(run_config["grad_manip_ratio"])
    shap_aware_ratio = float(run_config["shap_aware_ratio"])

    num_backdoor = round(total_edges * backdoor_ratio)
    num_label_flip = round(total_edges * label_flip_ratio)
    num_grad_manip = round(total_edges * grad_manip_ratio)
    num_shap_aware = round(total_edges * shap_aware_ratio)

    total_attackers = num_backdoor + num_label_flip + num_grad_manip + num_shap_aware
    num_benign = max(0, total_edges - total_attackers)
    
    role_list = ["backdoor"] * num_backdoor + ["label_flip"] * num_label_flip + ["gradient_manip"] * num_grad_manip + ["shap_aware"] * num_shap_aware + ["benign"] * num_benign
    
    random.Random(master_seed).shuffle(role_list) 
    return role_list[global_index % total_edges] if len(role_list) > 0 else "benign"