# Zero-Trust Agentic Federated Learning (Flower Implementation)

A privacy-preserving, Byzantine-resilient federated learning framework for Industrial IoT (IIoT) intrusion detection. This repository is a custom implementation of the ZTA-FL(https://arxiv.org/abs/2512.23809) framework utilizing the [Flower (flwr)](https://flower.dev/) federated learning framework. 

The system integrates device attestation concepts, SHAP-weighted robust aggregation, and on-device adversarial training into a three-tier edge–fog–cloud architecture.

> **📖 Architecture Parity & Reproducibility Report**
>
> Curious how this codebase maps to the theoretical experiments in the original ZTA-FL paper? 
> Please read the **[ZTA-FL Architecture Parity Report](ZTA_FL_Architecture_Parity_Report.md)** before exploring the code. It explicitly details which components are mathematically faithful reproductions (e.g., MAD-based SHAP filtering, TrustDB state machines), which are intentional engineering trade-offs for bare-metal IIoT hardware, and the exact Zero-Trust NGINX mTLS routing used to secure this execution over public networks.

---

## Overview

Traditional centralized intrusion detection systems cannot scale to IIoT deployments spanning thousands of heterogeneous edge devices across multiple industrial sites. Federated learning allows devices to collaboratively train a shared intrusion-detection model without sharing raw traffic data. However, standard FL is vulnerable to Byzantine poisoning, adversarial evasion, and Sybil attacks.

This project implements a defense-in-depth architecture using Flower's SuperLink/SuperNode topology to mitigate these threats.

---

## System Architecture

```mermaid
flowchart TB
    subgraph CLOUD["☁️  Cloud Layer (Flower SuperLink/Server)"]
        direction TB
        GA["Global Aggregator\nθ⁽ᵗ⁺¹⁾ = Σ wᵢ · θᵢ⁽ᵗ⁾"]
    end

    subgraph FOG["🌫️  Fog Layer (Intermediary Flower Nodes)"]
        direction LR
        FN1["Fog Node 1\n──────────\n• SHAP Stability Score\n• Byzantine Filter\n• Selected Aggregation (Krum, etc.)"]
        FN2["Fog Node 2\n──────────\n• SHAP Stability Score\n• Byzantine Filter\n• Selected Aggregation"]
    end

    subgraph EDGE["🏭  Edge Layer (Flower Clients)"]
        direction LR
        E1["Agent 1\n─────────\nCNN-LSTM IDS\nAdv. Training"]
        E2["Agent 2\n─────────\nCNN-LSTM IDS\nAdv. Training"]
        E3["Agent 3\n─────────\nCNN-LSTM IDS\nAdv. Training"]
        EN["Agent N\n─────────\nCNN-LSTM IDS\nAdv. Training"]
    end

    subgraph ATTACK["⚠️  Threat Model Configured via TOML"]
        direction TB
        BYZ["Byzantine Agents\n(Label Flip / Grad Scale)"]
        ADV["Adversarial Inputs\n(FGSM / PGD)"]
        SYB["Sybil / Replay\nAttackers"]
    end

    %% Data flow — upward
    E1 -->|"Model update"| FN1
    E2 -->|"Model update"| FN1
    E3 -->|"Model update"| FN2
    EN -->|"Model update"| FN2

    FN1 -->|"Aggregated fog update"| GA
    FN2 -->|"Aggregated fog update"| GA

    %% Global model — downward
    GA -->|"Global model θ⁽ᵗ⁺¹⁾"| FN1
    GA -->|"Global model θ⁽ᵗ⁺¹⁾"| FN2
    FN1 -->|"Updated model"| E1
    FN1 -->|"Updated model"| E2
    FN2 -->|"Updated model"| E3
    FN2 -->|"Updated model"| EN

    %% Threats
    BYZ -. "Poisoned update" .-> FN1
    ADV -. "Evasion attempt" .-> E2
    SYB -. "Fake token" .-> FN2

    %% Styles
    classDef cloudStyle  fill:#1a6496,stroke:#0d3d5c,color:#fff
    classDef fogStyle    fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef edgeStyle   fill:#4a4a8a,stroke:#2c2c6a,color:#fff
    classDef threatStyle fill:#b71c1c,stroke:#7f0000,color:#fff,stroke-dasharray:4 4

    class GA cloudStyle
    class FN1,FN2 fogStyle
    class E1,E2,E3,EN edgeStyle
    class BYZ,ADV,SYB threatStyle
```

> **Note on Network Security:** The diagram above details the Machine Learning and Federated Data Flow. For the exact **Zero-Trust NGINX mTLS routing architecture** used to secure this execution over public networks, please see the [Architecture Parity Report](ZTA_FL_Architecture_Parity_Report.md).

---

## Project Structure

```
zta-fl-flower/
├── config/                    # Configuration files for architecture parameter tuning
├── data/                      # Raw datasets (cic_ids2017, edge_iiotset, unsw_nb15)
├── logs/                      # Live execution logs for all network nodes
├── results/                   # JSON outputs and global models
├── runtime/                   # Local state and TPM ledgers (e.g., tpm_state/)
├── src/                       # Core Node Runtime Code (Deployed to physical machines)
│   ├── entrypoints/           # Flower application initializers (server.py, client.py)
│   ├── shared/                # Shared logic utilized across multiple tiers
│   │   ├── data/              # Data loading, SMOTE, and non-IID partitioning
│   │   ├── models/            # ML architectures
│   │   ├── network/           # IPC routing and custom 8-bit quantization mechanics
│   │   ├── security/          # Threat injection (adversarial_math.py, backdoor_math.py)
│   │   └── utils/             # Evaluation metrics, configs, and admin console
│   ├── tier_cloud/            # Cloud Layer 
│   ├── tier_edge/             # Edge Layer 
│   └── tier_fog/              # Fog Layer
├── scripts/                   # Helpful scripts for operation and setup
├── tools/                     # Offline Local Utilities & Analytics 
├── verification/              # Pre-Flight Checks & Static Analysis
├── ZTA_FL_Architecture_Parity_Report.md # Architecture parity report
└── pyproject.toml             # Flower configuration file 
```
---

## System Prerequisites

This architecture heavily relies on **DOCKER** and **OPENSSL**. Before running the python environment or orchestration scripts, you must install the following:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install openssl
```

**macOS:**
```bash
brew install openssl
```

---

## Configuration & Customization

The entire behavior of the distributed network, machine learning models, and threat environment is controlled centrally via the `configs/` directory. 

Key sections you can modify:

* **Topology (`configs/network.yaml`):** Scale the network by adjusting `num_fogs` and `uniform_edges_per_fog`. You can even dictate custom distributions using `custom_fog_topology`.
* **Strategy (`configs/training.yaml`):** Change the aggregation strategy by editing the `strategy` variable (e.g., `"zta"`, `"fedavg"`, `"krum"`, `"trimmed_mean"`).
* **Data Dynamics (`configs/training.yaml`):** Control non-IID data distribution using `n_classes_per` (label skew) and `power_law_a` (quantity skew). 
* **Threat Model (`configs/security.yaml`):** Introduce malicious agents by adjusting the ratios under the `THREAT MODEL` section:
  * `label_flip_ratio = 0.2` (Turns 20% of edges into label-flippers)
  * `grad_manip_ratio = 0.1`
  * `backdoor_ratio = 0.0`
  * `shap_aware_ratio = 0.1`

> **Note:** Performance varies depending on your hardware specifications and the total number of nodes in the network.

---

## Datasets

Three publicly available network intrusion datasets are supported in the `data/` directory:

| Dataset | Description |
|---------|-------------|
| **Edge-IIoTset** | Traffic from IIoT devices (PLC, SCADA, Smart Sensors). |
| **CIC-IDS2017** | Network flow records from a general enterprise network. |
| **UNSW-NB15** | Network intrusion records from the UNSW cyber range. |

Configure which dataset to use by updating the `dataset` and `dataset_path` variables in `configs/training.yaml`.

---

## Deploying the Distributed Network

To run the full distributed Zero-Trust architecture, we use a dynamic orchestration script that spins up the Cloud SuperLink, Fog SuperNodes/SuperLinks, and Edge SuperNodes using Docker on a single machine.

> **Note:** Make sure the **DOCKER** daemon is running.

The following script builds the infrastructure that the code runs on top of. Use the `--insecure` flag to bypass mTLS and TLS.
```bash
git clone https://github.com/PanagiwthsPapadopoulos/zta-fl-flower.git
cd zta-fl-flower
chmod +x scripts/ops/boot_network_docker.sh
./scripts/ops/boot_network_docker.sh
```
While keeping the above command running, in a new terminal run the following, which ships the `FAB` to the network:
```bash
cd zta-fl-flower
chmod +x scripts/ops/deploy_code_docker.sh
./scripts/ops/deploy_code_docker.sh
```

> 🛑 **CRITICAL WARNING: GLOBAL CONFIGURATION OVERWRITE**
> 
> In current versions of Flower, routing is managed globally via the `~/.flwr/config.toml` file. According to **[Flower Issue #6824](https://github.com/flwrlabs/flower/issues/6824)**, there is currently no native support for isolated, per-project SuperLink configurations.
> 
> Because this deployment script requires specific, dynamic port routing to orchestrate the 3-tier architecture, it **will overwrite your global `~/.flwr/config.toml` file.**
> 
> **You MUST back up your existing `~/.flwr/config.toml` before running this script** if you have other active Flower endpoints saved on your machine. The script will prompt you for confirmation before making any destructive changes.

> **Note:** Windows users must use WSL (Windows Subsystem for Linux) to run the orchestration bash scripts


## Monitoring & Logs
The terminal displays the architecture map and then holds the process. The flow of the pipeline for the whole network written to the `logs/` directory:

```bash
# Watch the Cloud (Global convergence)
tail -f logs/nodes/cloud/cloud_server.jsonl

# Watch a specific Fog (Local aggregation & Security filtering)
tail -f logs/nodes/fog_1/fog_1_server.jsonl

# Watch an Edge Node (Local training & TPM Attestation)
tail -f logs/nodes/edge_1_1/edge_1_1.jsonl
```
Otherwise, you can watch the pipeline update live using the following command:

```python
python3 verification/verify_pipeline.py -w
```
> **Note:** The `logs/` directory is **wiped clean at the start of every run.** Ensure you export any critical log instances before restarting the network.

---

## Post-Run Analysis

Once the network finishes its communication rounds, the results, including aggregated metrics and layer weights, are saved as JSON files in the `results/` directory.

You can generate visualizations (like Accuracy vs. Communication Rounds, or SHAP stability distributions) using the provided script:

* **Create and Activate the virtual environment & Install matplotlib**
```
python3 -m venv .venv-zta
source .venv-zta/bin/activate
pip install "matplotlib>=3.10.9"
```

* **Auto-detect latest run:** 
```
python3 tools/plot_metrics.py
```
* **Plot specific run:** 
```
python3 tools/plot_metrics.py results/experiment_name
```
* **Custom panels:** 
```
python3 tools/plot_metrics.py --panels loss asr pgd
```

**Available Panels:** `accuracy_f1`, `loss`, `asr`, `pgd`, `fgsm`.

Generated plots will be saved to `results/experiment_name/`.

---

## Acknowledgments and References

This repository is built using the open-source **Flower** federated learning framework:
* [Flower (flwr) GitHub Repository](https://github.com/adap/flower)
* [Flower Official Documentation](https://flower.dev/)

The architecture, logic, and experimental design implemented here are derived from the original **ZTA-FL** paper. If this architecture and methodology helps your research, please refer to and cite the original authors:

```bibtex
@article{singh2025ztafl,
  title   = {Zero-Trust Agentic Federated Learning for Secure {IIoT} Defense Systems},
  author  = {Singh, Samaresh Kumar and Roy, Joyjit and So, Martin},
  journal = {arXiv preprint arXiv:2512.23809},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.23809},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
