import os
import sys
import torch

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.data.data_loader import non_iid_partition
from src.utils.config_loader import load_yaml_configs

def test_non_iid_skew_properties():
    """
    Validates that the partitioner correctly generates heterogeneity.
    Automatically scales agent counts, class limits, and power-law variance 
    expectations based on the project configuration.
    """
    print("Executing Dynamic Data Partition Math Test...")
    
    # Load configuration
    run_metadata = load_yaml_configs()
    
    # Calculate total agents dynamically from topology
    num_fogs = int(run_metadata.get("num_fogs", 10))
    edges_per_fog = int(run_metadata.get("uniform_edges_per_fog", 10))
    total_agents = num_fogs * edges_per_fog
    
    dynamic_classes_per = int(run_metadata.get("n_classes_per", 3))
    dynamic_power_law = float(run_metadata.get("power_law_a", 0.4))
    
    print(f"[CONFIG LOADED] Total Agents: {total_agents}")
    print(f"[CONFIG LOADED] Classes Per Agent: {dynamic_classes_per}")
    print(f"[CONFIG LOADED] Power Law Alpha: {dynamic_power_law}")

    X = torch.randn(total_agents * 50, 40) # Ensure enough dummy data
    y = torch.randint(0, 15, (total_agents * 50,))
    
    # Execute partitioning
    partitions = non_iid_partition(X, y, n_agents=total_agents, n_classes_per=dynamic_classes_per, power_law_a=dynamic_power_law, seed=42)
    
    # 1. Assert Dynamic Class Skew
    for idx, (_, y_p) in enumerate(partitions):
        unique_classes = len(torch.unique(y_p))
        assert unique_classes <= dynamic_classes_per, f"Agent {idx} exceeded {dynamic_classes_per} classes (Found {unique_classes})."
        
    # 2. Extract dataset counts
    counts = torch.tensor([len(y_p) for _, y_p in partitions], dtype=torch.float)
    std_dev = torch.std(counts).item()
    
    print(f"\nPartition Volumes Standard Deviation: {std_dev:.2f}")
    print(f"Min samples: {int(torch.min(counts).item())} | Max samples: {int(torch.max(counts).item())}")

    # =================================================================
    # VISUAL DISTRIBUTION PROOF (ASCII HISTOGRAM)
    # =================================================================
    print("\n=================================================")
    print(" DISTRIBUTION SHAPE (AGENT BUCKETS)")
    print("=================================================")
    thresholds = [1000, 2000, 3000, 4000, 5000]
    prev = 0
    for t in thresholds:
        in_bin = ((counts > prev) & (counts <= t)).sum().item()
        bar = "█" * int((in_bin / total_agents) * 50) # Visual bar
        print(f"[{prev:4d} - {t:4d} samples] : {in_bin:3d} agents | {bar}")
        prev = t
    print("=================================================\n")
    
    # Scale the expected variance based on the power law severity
    expected_min_std_dev = 50.0 if dynamic_power_law < 0.6 else 5.0
    assert std_dev >= expected_min_std_dev, f"Distribution too uniform for alpha={dynamic_power_law}. Variance is too low."
    print("✅ DATA PARTITION TEST PASSED")

if __name__ == "__main__":
    test_non_iid_skew_properties()