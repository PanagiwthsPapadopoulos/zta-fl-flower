import random
import os

def generate_random_toml():
    """
    Constructs an abstract structural file utilizing varied random elements generating valid execution definitions.
    Generates experimental permutations encompassing diverse hyperparameter arrays simulating disparate node activity limits.
    """
    strategies = ["zta", "fedavg", "fedprox", "krum", "trimmed_mean", "flame", "fltrust"]
    datasets = ["edge_iiotset", "cic_ids2017", "unsw_nb15"]
    
    num_fogs = random.randint(2, 4)
    uniform_edges = random.randint(2, 3)
    
    toml_content = f"""
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "zta-fl-flower"
version = "1.0.0"
description = "Automated ZTA-FL Federation Test"
dependencies = [
    "flwr[simulation]>=1.9.0",
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.3.0",
    "imbalanced-learn>=0.12.0",  
]

[tool.flwr.app]
publisher = "zta-team"

[tool.flwr.app.components]
serverapp = "src.federation.server:app"
clientapp = "src.federation.client:app"

[tool.flwr.app.config]
# ==============================================================================
# --- 1. CORE EXPERIMENT & TOPOLOGY ---
# ==============================================================================
strategy = "{random.choice(strategies)}"
tier = "cloud" 
fog_id = "fog_0" 
dataset = "{random.choice(datasets)}"
min-clients = 1
num_fogs = {num_fogs}
uniform_edges_per_fog = {uniform_edges}
dataset_fraction = 0.05
num_rounds = 2
local_epochs = {random.randint(1, 3)}
learning_rate = {random.choice([0.01, 0.005, 0.001, 0.0001])}
batch_size = {random.choice([32, 64, 128])}
quantization_bits = {random.choice([8, 16, 32])}

# Attack Ratios
pgd_ratio = {random.uniform(0.0, 0.15):.2f}
fgsm_ratio = {random.uniform(0.0, 0.15):.2f}
label_flip_ratio = {random.uniform(0.0, 0.1):.2f}

# Required Defaults for Auditor
model_architecture = "cnnlstm"
random_seed = 42
clip_min = 0.0
clip_max = 1.0
clip_norm = 1.0
rollback_threshold = 0.80

# ==============================================================================
# --- 7. INFRASTRUCTURE & NETWORKING (EXPLICIT PORTS) ---
# ==============================================================================
broker_ip = "127.0.0.1"
socket_timeout = 3600.0
cloud_sa_port = 9001
cloud_fl_port = 9002
cloud_ctrl_port = 9003
fog_sa_base = 9100
fog_fl_base = 9200
fog_ctrl_base = 9300
fog_client_io_base = 9400
fog_ipc_base = 9500
edge_client_io_base = 10000
"""
    
    with open("pyproject.toml", "w") as f:
        f.write(toml_content.strip())
    
    print("✅ Successfully generated a completely randomized pyproject.toml!")

if __name__ == "__main__":
    generate_random_toml()