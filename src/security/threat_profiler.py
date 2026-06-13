import random

def assign_edge_roles(run_config: dict, total_edges: int, global_index: int, master_seed: int, logger) -> str:
    """
    Computes strict assignments distributing adversarial identities across the edge grid
    based entirely on fixed mathematical ratios drawn from the primary configuration parameters.
    """
    pgd_ratio = float(run_config.get("pgd_ratio", 0.0))
    fgsm_ratio = float(run_config.get("fgsm_ratio", 0.0))
    backdoor_ratio = float(run_config.get("backdoor_ratio", 0.0))
    label_flip_ratio = float(run_config.get("label_flip_ratio", 0.0))
    grad_manip_ratio = float(run_config.get("grad_manip_ratio", 0.0))
    shap_aware_ratio = float(run_config.get("shap_aware_ratio", 0.0))

    num_pgd = round(total_edges * pgd_ratio)
    num_fgsm = round(total_edges * fgsm_ratio)
    num_backdoor = round(total_edges * backdoor_ratio)
    num_label_flip = round(total_edges * label_flip_ratio)
    num_grad_manip = round(total_edges * grad_manip_ratio)
    num_shap_aware = round(total_edges * shap_aware_ratio)

    logger.debug(f"[CONFIG USAGE] assign_edge_roles | pgd_ratio: {pgd_ratio}, fgsm_ratio: {fgsm_ratio}, backdoor_ratio: {backdoor_ratio}, label_flip_ratio: {label_flip_ratio}, grad_manip_ratio: {grad_manip_ratio}")
    
    total_attackers = num_pgd + num_fgsm + num_backdoor + num_label_flip + num_grad_manip + num_shap_aware
    num_benign = max(0, total_edges - total_attackers)
    
    role_list = ["pgd"] * num_pgd + ["fgsm"] * num_fgsm + ["backdoor"] * num_backdoor
    role_list += ["label_flip"] * num_label_flip + ["gradient_manip"] * num_grad_manip
    role_list += ["shap_aware"] * num_shap_aware
    role_list += ["benign"] * num_benign
    
    random.Random(master_seed).shuffle(role_list) 
    role = role_list[global_index % total_edges] if len(role_list) > 0 else "benign"
    return role