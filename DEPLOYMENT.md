# 3-Tier Distributed Deployment Guide (`mode = "deploy"`)

**Baseline Requirement:** Flower `1.30.0` | Python `>=3.10,<4.0` (CPython `3.11.x` recommended)  
**Supported Container Engines:** Docker Compose & Apptainer (Rootless / HPC)

**Deployment Mode** orchestrates Flower's [Deployment Runtime](https://flower.ai/docs/framework/deploy.html) (`SuperLink`, `SuperNode`, and `SuperExec`) into a physical/containerized **3-tier hierarchical federated learning network** (Cloud $\leftrightarrow$ Fog $\leftrightarrow$ Edge). Unlike standard 2-tier Federated Learning, every tier runs in isolated containers or processes with real mTLS network routing, local TCP IPC bridges, and hardware-backed TPM 2.0 attestation.

⚠️ **Disclaimer:**  This mode is very resource intensive to run on a single machine. It serves only as a Proof of Concept and scaling has not been thoroughly tested. No deployment with more than 3 fog nodes and 3 edge nodes each (9 total) has been attempted on a single computer. 

📝 Tips depending on your deployment environment:
- When using **Docker**, make sure you have allocated sufficient RAM, else the containers get **silently** OOM killed.
- When using **Apptainer**, make sure first to:
	- load **Apptainer**'s module,
	- raise the process limit of the user if you are running this on an HPC, or else you will quickly hit the limit and silently fail.
	
---

## 1. How the 3-Tier Architecture Works Under the Hood

Because Flower natively orchestrates 2-tier (`ServerApp` $\leftrightarrow$ `ClientApp`) topologies, this architecture constructs a 3-tier hierarchy by running **`1 + NUM_FOGS` independent Flower federations simultaneously** (`cloud`, `fog1`, `fog2`, ...):

```text
[Tier 1: Cloud Federation ("cloud")]
  └── cloud-superlink + cloud-serverapp (CloudAggregator)
         ▲
         │  gRPC / NGINX TLS
         ▼
[Tier 2: Fog Node i (Dual-Stack Bridge)]
  ├── Upward Stack:   fog-i-supernode + fog-i-clientapp (FogBridgeClient)
  │                         ▲
  │                         │ Local TCP IPC Socket (fog_ipc_base + i)
  │                         ▼
  └── Downward Stack: fog-i-superlink + fog-i-serverapp (FogAggregator + FogBridgeServer)
         ▲
         │  gRPC / NGINX mTLS (Federation "fog{i}")
         ▼
[Tier 3: Edge Nodes (edge_i_1 ... edge_i_M)]
  └── edge-i-j-supernode + edge-i-j-clientapp (EdgeTrainer + TPMAttestation + swtpm)
````

### Runtime Lifecycle of a Single Global Round

1. **Cloud Broadcast (`src/tier_cloud/cloud_aggregator_deploy.py`):**
    
    - `CloudAggregator.configure_fit()` broadcasts the global compressed model parameters to all connected upward-facing **Fog ClientApps** (`min_clients = NUM_FOGS`).
        
2. **Fog IPC Synchronization (`src/tier_edge/fog_bridge_client.py` $\leftrightarrow$ `src/tier_fog/fog_bridge_server.py`):**
    
    - When `fit()` is triggered on `fog-i-clientapp` (`node_type="fog_client"` in `deploy_client.py`), `FogBridgeClient` opens a local TCP socket to `fog-i-serverapp` on port `fog_ipc_base + i`, sends `{"cmd": "START", "round": current_round}`, and **blocks**.
        
    - `FogAggregator.configure_fit()` (`src/tier_fog/fog_aggregator.py`), which was blocking on `FogBridgeServer.wait_for_start()`, unblocks, synchronizes its round counter, executes any scheduled `AdminConsole` firmware/PCR updates (`config/admin.yaml`), generates a unique 16-byte cryptographic `nonce` (`secrets.token_hex(16)`) per Edge client, and dispatches `FitIns` down to its regional Edge nodes.
        
3. **Edge Local Training, Threat Injection & TPM Signing (`src/entrypoints/deploy_client.py` & `src/tier_edge/`):**
    
    - Each Edge node (`node_type="edge"`) maps its `(fog_num, partition-id)` to a deterministic `global_index` and receives a static role via `assign_edge_roles()` (`benign`, `label_flip`, `gradient_manip`, `backdoor`, or `shap_aware`).
        
    - It loads its pre-compiled non-IID partition (`edge_{global_index}`) directly from `/app/data/<dataset>/artifacts/` (`get_dataset()`).
        
    - **Multi-Slot GPU Lock (`EdgeTrainer`):** When running on a shared host GPU (`use_gpu_lock=True`), the container acquires an exclusive POSIX file lock (`fcntl.flock`) on `/app/runtime/locks/gpu_slot_{slot}.lock` (bounded by `max_concurrent_gpus`) before initializing CUDA, and purges VRAM (`torch.cuda.empty_cache()`, `gc.collect()`) immediately after local epochs finish.
        
    - **Role-Based Optimization:**
        
        - `benign`: Executes on-device adversarial training (`adversarial_train_epoch` with PGD or FGSM on `benign_adv_ratio` of each batch).
            
        - `label_flip` / `gradient_manip`: Executes `local_train_byzantine`.
            
        - `backdoor`: Poisons non-target samples with feature triggers (`poison_partition`) and runs `local_train_honest`.
            
        - `shap_aware`: Executes `local_train_shap_aware`, iteratively blending malicious weights with the global reference model (`blend_ratio = 0.8`) until `1.0 - compute_shap_stability(...) < shap_threshold` to evade Fog detection.
            
    - **TPM 2.0 Quote Generation (`TPMAttestation`):** Connects to the local `swtpm` daemon (`TPM2TOOLS_TCTI`), provisions an RSA-2048 Attestation Key (`0x81010002`) in NVRAM, hashes the Fog's `nonce` together with `PCR 0` (`sha256:0`) via `tpm2_quote`, and packages the base64 `quote_msg`, `signature`, hardware `IDi`, and `timestamp` into `tpm_token_json`.
        
    - **Dynamic Round Attacks (`AdversaryManager`):** If configured in `config/threat.yaml`, intercepts outgoing payloads on `activate_on_round` to inject `tpm_replay`, `tpm_forgery`, `pcr_alteration`, or Gaussian `model_poisoning`.
        
    - Compresses model weights (`compress_weights` — 8-bit uint8 min-max quantization, 16-bit FP16, or 32-bit FP32) and returns them to the Fog Server.
        
4. **Fog Gatekeeper, Parallel GradientSHAP & IPC Relay (`src/tier_fog/`):**
    
    - **Phase 5 — Cryptographic Gatekeeper (`ZeroTrustGatekeeper` & `TPMVerifier`):** Checks token freshness (`tpm_freshness_window`), verifies RSA signatures via `tpm2_checkquote` against `/app/runtime/tpm_state/edge_{i}_{j}/ak.pub`, unpacks the binary `TPMS_ATTEST` structure via `tpm2_print` to verify `extraData` (nonce) and `pcrDigest` against the read-only Single Source of Truth (`pcr_ledger.json`), and updates the node's score in `TrustDatabase` (quarantining nodes that drop below `0.6`).
        
    - **Phase 6 — SHAP Stability Filtering (`shap_weighted_aggregate`):** Evaluates surviving models in parallel (`ThreadPoolExecutor`) using Captum's `GradientShap` against the Fog validation split (`X_val`). Calculates the **Median ($\mu_s$)** and **MAD-based standard deviation ($\sigma_s = \max(1.4826 \cdot \text{MAD}, 10^{-5})$)**. Models with $s_i < \mu_s - 2\sigma_s$ are rejected as **saboteurs** and penalized in `TrustDatabase`; models with $s_i \ge \mu_s$ receive a `+0.02` trust reward. Surviving models are aggregated using $w_i = s_i \cdot \text{acc}_i \cdot \sqrt{\vert{}D_i\vert{}}$.
        
    - **Rollback Sanity Check & 3-Pillar Telemetry:** Verifies regional accuracy against `previous_val_acc * rollback_threshold` (rolling back to `cached_global_state` on collapse), writes `trustdb`, `shap_scores`, and `performance_metrics` JSONs, and relays the compressed regional weights over the TCP IPC socket back to `FogBridgeClient`.
        
5. **Cloud Global Aggregation & Evaluation (`CloudAggregator` & `GlobalEvaluator`):**
    
    - `FogBridgeClient` forwards the regional weights up to the Cloud. `CloudAggregator` performs trust-weighted `federated_averaging` across all Fog regions, evaluates the global model on the held-out `test` split (`GlobalEvaluator` computing Loss, Accuracy, Macro-F1, and ROC-AUC), and saves `results/<experiment_name>/<experiment_name>.json` and `global_model.pt`.
        

## 2. Environment & Version Requirements (Critical)

> **⚠️ Strict Baseline Policy (`flwr==1.30.0`)**
> 
> For the native-CLI Docker workflow, **the terminal that runs `deploy_code_docker.sh` must resolve `flwr` to the project's `.venv/bin/flwr`, and that executable must report `Flower version: 1.30.0`**.
> 
> - **Python Version:** Python `>=3.10,<4.0` (CPython `3.11.x` recommended). Python `3.9` cannot install Flower `1.30.0`.
>     
> - **Why `1.30.0` is pinned:** Older CLIs (e.g., `1.23.0`) fail with `'federations'` / `No such command 'config'` because centralized `~/.flwr/config.toml` SuperLink connections were introduced in `1.26.0`. Newer releases (`1.35`–`1.40+`) transition the Runtime, Control, and Fleet APIs from gRPC to HTTP, which is incompatible with the `1.30.0` container images unless the entire infrastructure and port routing are migrated together.
>     

### One-Time Host Virtual Environment Setup

From the repository root:

```bash
# 1. Create a Python 3.11 virtual environment (macOS Homebrew example)
brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
source .venv/bin/activate
rehash

# 2. Install pinned dependencies into .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install "flwr[simulation]==1.30.0" PyYAML "rich>=13.9.4,<16"
rehash
```

### Mandatory Pre-Flight Verification (Run in Terminal B before deploying)

From the repository root:

```bash
source .venv/bin/activate
rehash

# Verify executable resolution and child-shell inheritance
command -v python
command -v python3
command -v flwr
/bin/bash -c 'command -v python3; command -v flwr; flwr --version'
python -m pip check
```

All commands must resolve inside `zta-fl-flower/.venv/bin/` and output `Flower version: 1.30.0` with `No broken requirements found.`

## 3. Running Deployment Mode with Docker Compose

> 🛑 **CRITICAL WARNING: GLOBAL `~/.flwr/config.toml` OVERWRITE**
> 
> Flower manages SuperLink routing globally via `~/.flwr/config.toml` ([Flower Issue #6824](https://www.google.com/search?q=https://github.com/flwrlabs/flower/issues/6824)) and there is currently no native support for isolated, per-project SuperLink configurations. Because this deployment script dynamically maps Control API ports for `cloud` and `fog1..fogN`, **running the boot script will overwrite your global `~/.flwr/config.toml`** file.
> 
> Back up your existing file before booting if you have other Flower endpoints saved.
### Terminal A — Boot the 3-Tier Infrastructure

Keep this terminal open for the duration of your experiments:

```bash
cd zta-fl-flower
source .venv/bin/activate
rehash

# Make scripts executable
chmod +x scripts/deployment/boot_network_docker.sh scripts/setup/*.sh

# Boot with full NGINX mTLS & hardware TPM attestation (Default)
./scripts/deployment/boot_network_docker.sh

# OR: Boot in plain-text mode (bypasses NGINX mTLS and TPM verification)
./scripts/deployment/boot_network_docker.sh --insecure
```

**What `boot_network_docker.sh` does:**

1. Parses `config/network.yaml` via `scripts/setup/parse_topology.py`.
    
2. Generates the Root CA, server/client certificates (`setup_security.sh`), and the NGINX mTLS reverse-proxy configuration (`setup_nginx.sh`) unless `--insecure` is passed.
    
3. Initializes host directories for each Edge node's software TPM (`setup_tpm.sh`).
    
4. Pulls `panagiotispapadopoulos/zta-cloud-node:latest` and `panagiotispapadopoulos/zta-edge-node:latest` and generates `docker/docker-compose.yml` (`generate_compose.sh`).
    
5. Boots the Docker Compose mesh, polls all Edge containers until their `swtpm` factory boot quotes complete, and compiles the Single Source of Truth `runtime/tpm_state/pcr_ledger.json` (`collect_ledgers.py`).
    
6. Writes the `cloud` and `fog{i}` connection profiles to `~/.flwr/config.toml`, creates `runtime/infra/.network_ready`, and prints:
    
    `ENGINE IS LIVE. RUN DEPLOY_CODE_DOCKER.SH`
    

### Terminal B — Compile Artifacts & Dispatch Application Bundles (FABs)

Once Terminal A prints `ENGINE IS LIVE`, open a **second terminal**:

``` bash
cd zta-fl-flower
source .venv/bin/activate
rehash

# Verify CLI and generated SuperLink connections
flwr --version
flwr config list

# Submit the FABs to all Fog federations and the Cloud federation
chmod +x scripts/deployment/deploy_code_docker.sh
./scripts/deployment/deploy_code_docker.sh
```

**What `deploy_code_docker.sh` does:**

1. Verifies `runtime/infra/.network_ready` and active Docker Compose containers.
    
2. Cleans previous node logs in `logs/nodes/` and `logs/system/run_*.log`, and touches `config/training.yaml` so all tiers synchronize on a single deterministic `run_YYYYMMDD_HHMMSS` directory name.
    
3. Prints the live ASCII topology map of your Cloud, Fog, and Edge containers.
    
4. Runs **The Big Crunch (`scripts/setup/build_artifacts.py`)** inside a temporary cloud container to compile and cache scaled, SMOTE-balanced, non-IID dataset partitions in `data/<dataset>/artifacts/` if they do not already exist.
    
5. Dispatches `flwr run . fog{i}` in the background for every Fog stack, waits for their Control ports, and dispatches `flwr run . cloud` to trigger global training.
    

## 4. Running Deployment Mode with Apptainer (HPC / Rootless)

For shared HPC clusters or environments without root Docker daemon access, the entire 3-tier deployment runs natively via **Apptainer** SIF images (`cloud_node.sif`, `edge_node.sif`, `nginx.sif`) and UNIX domain sockets (`swtpm.sock`).

Below are some useful commands to make sure that everything runs smoothly. They will need to be executed on every new terminal you start:

1. Load the module. On [AUTh's HPC](https://hpc.auth.gr) the commands are:
``` bash
module load gcc/13.2.0-nbog6z2
module load apptainer/1.1.9-xbwwi67
apptainer --version
```

2. Increase the process count for the specified user:
```
ulimit -u "$(ulimit -Hu)"
ulimit -n 65535
```

3. *Optional:* Track CPU, Memory and NVIDIA stats:
```
watch -n 1 -c "nvidia-smi; echo ''; echo '--- SYSTEM CPU & RAM ---'; top -b -n 1 | head -n 15"
```

4. *Optional:* Track current and peak process count:
```
peak=0; while true; do cur=$(ps -eLf -u $USER | wc -l); [ "$cur" -gt "$peak" ] && peak=$cur; printf "\rCurrent: %d | Peak: %d\e[K" "$cur" "$peak"; sleep 1; done
```

### Build .sif images (single use)

Once per machine, the images need to be built:

``` bash
cd zta-fl-flower
chmod +x scripts/ops/build_images_apptainer.sh
./scripts/ops/build_images_apptainer.sh
```

### Terminal A — Boot the Apptainer Federation

```bash
cd zta-fl-flower
source .venv/bin/activate
rehash

chmod +x scripts/deployment/boot_network_apptainer.sh
./scripts/deployment/boot_network_apptainer.sh          # Full TLS/mTLS + swtpm
# OR: ./scripts/ops/boot_network_apptainer.sh --insecure
```

### Terminal B — Dispatch FABs to Apptainer

Once Terminal A prints `ENGINE IS LIVE. RUN DEPLOY_CODE_APPTAINER.SH`:

```bash
cd zta-fl-flower
chmod +x scripts/deployment/deploy_code_apptainer.sh
./scripts/deployment/deploy_code_apptainer.sh
```

_(Note: `deploy_code_apptainer.sh` executes `/python/venv/bin/flwr` directly inside `images/cloud_node.sif`, so it is insulated from host `flwr` PATH mismatches.)_

## 5. Monitoring, Verification & Post-Run Acceptance

### Terminal C — Live Pipeline Monitoring

You can monitor the 3-tier training progress in real time using the terminal-safe Rich dashboard (`verification/verify_pipeline.py`) or raw JSONL streams:

``` bash
# Option 1: Scroll-safe watch mode (prints append-only snapshots when state changes)
python3 verification/verify_pipeline.py -w

# Option 2: Bounded full-screen live dashboard (uses alternate terminal screen)
# Does not work for Apptainer
python3 verification/verify_pipeline.py --live

# Option 3: Tail individual tier JSONL logs
tail -F logs/nodes/cloud/cloud_server.jsonl
tail -F logs/nodes/fog_1/fog_1_server.jsonl
tail -F logs/nodes/edge_1_1/edge_1_1.jsonl

# Option 4: Tail raw Flower CLI submission streams
tail -F logs/system/run_cloud.log logs/system/run_fog1.log logs/system/run_fog2.log
```

 If Rich is not installed on your system, you can run the script from inside the container:

```bash 
apptainer exec --bind "$PWD:/app" --pwd /app images/cloud_node.sif /python/venv/bin/python3 verification/verify_pipeline.py -w
```

> ⚠️ **Note:** The images have an older version of the `verify_pipeline.py` script, so the functionality of the `--live` flag can be referenced with `-w`.
### Output Artifacts & Verification

Every run outputs its artifacts under `results/<experiment_name>/`:

- `results/<experiment_name>/<experiment_name>.json` — Global metadata and per-round `global_loss`, `global_accuracy`, `global_macro_f1`, `round_latency_total_sec`, and `payload_size_mb`.
    
- `results/<experiment_name>/global_model.pt` (and periodic `global_model_round_<r>.pt` checkpoints).
    
- `results/<experiment_name>/trustdb/fog_<i>_state.json` — Per-round attestation pass/reject lists, ground-truth roles, and `TrustDatabase` scores/quarantine flags.
    
- `results/<experiment_name>/shap_scores/fog_<i>_state.json` — Raw per-node GradientSHAP stability scores, detected `saboteurs`, and `rewarded_nodes`.
    
- `results/<experiment_name>/performance_metrics/fog_<i>_state.json` — Granular timestamps and latencies for attestation, local adversarial training, SHAP verification, and aggregation.