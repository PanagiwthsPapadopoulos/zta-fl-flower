# High-Scale Hybrid Simulation Guide (`mode = "sim"`)

**Baseline Requirement:** Flower `1.30.0` (`flwr[simulation]==1.30.0`)  
**Supported Container Engine:** Docker Compose (`sim-cloud` + $N$ `sim-tpm-*` sidecars)

**Simulation Mode** is engineered to reproduce the large-scale experimental benchmarks of the ZTA-FL paper (scaling up to 100+ Edge clients) on a single machine without the memory and gRPC overhead of running hundreds of independent Flower `SuperNode` and `SuperLink` daemons.

It combines Flower's Ray-backed **[Simulation Runtime](https://flower.ai/docs/framework/how-to-run-simulations.html)** inside a single master container (`sim-cloud`) with **$N$ lightweight external hardware emulator containers (`sim-tpm-{fog}-{edge}`)** so that TPM 2.0 cryptographic attestation remains 100% genuine.

---

## 1. Hybrid Simulation Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ Docker Bridge Network (flwr-network)                                          │
│                                                                               │
│  ┌────────────────────────────────────────────────────────┐                   │
│  │ Container: sim-cloud (The Simulation Engine)           │                   │
│  │                                                        │                   │
│  │  • [CLOUD] Global Aggregator (Tier 1)                  │                   │
│  │  • [FOG 1..K] Virtual Aggregators in-memory (Tier 2)   │                   │
│  │      ├── ZeroTrustGatekeeper (tpm2_checkquote/print)   │                   │
│  │      ├── TrustDatabase & AdminConsole                  │                   │
│  │      └── Parallel GradientSHAP Verifier                │                   │
│  │  • [EDGE i_j] Ephemeral Ray Actors (Tier 3)            │                   │
│  │      └── EdgeTrainer + AdversaryManager + TPMAttestation                   │
│  └───────────────────────────┬────────────────────────────┘                   │
│                              │                                                │
│                              │ TCP Socket (TPM2TOOLS_TCTI=swtpm:host=...,2321)│
│                              ▼                                                │
│  ┌────────────────────┐  ┌────────────────────┐         ┌────────────────────┐│
│  │ sim-tpm-1-1 (:2321)│  │ sim-tpm-1-2 (:2321)│   ...   │ sim-tpm-K-M (:2321)││
│  │ (swtpm + NVRAM AK) │  │ (swtpm + NVRAM AK) │         │ (swtpm + NVRAM AK) ││
│  └────────────────────┘  └────────────────────┘         └────────────────────┘│
└───────────────────────────────────────────────────────────────────────────────┘
````

### How Simulation Mode Preserves Full Architectural Parity

1. **Dynamic Entrypoint Routing (`src/entrypoints/server.py` & `client.py`):**
    
    - `deploy_code_sim_docker.sh` invokes `flwr run . local` with `RUN_CFG='mode="sim" tier="cloud" min_clients=$TOTAL_EDGES'`.
        
    - Detecting `mode == "sim"`, `server.py` and `client.py` automatically route execution to `src/entrypoints/sim_server.py` and `src/entrypoints/sim_client.py`.
        
2. **Mathematical Topology Mapping & Remote TCP TPM Signing (`src/entrypoints/sim_client.py`):**
    
    - Flower's Ray engine spawns ephemeral `Client` actors indexed by a 0-based `partition-id` ($0 \dots N-1$).
        
    - `sim_client.py` reads `custom_fog_topology` (or `num_fogs` $\times$ `uniform_edges_per_fog`) from `config/network.yaml` and mathematically maps each `partition-id` to its exact hierarchical coordinate `[EDGE {fog_num}_{edge_num}]`.
        
    - It dynamically sets:
    
        ```python
        os.environ["TPM2TOOLS_TCTI"] = f"swtpm:host=tpm-{fog_num}-{edge_num},port=2321"
        ```
        
    - When the Ray actor finishes local training (`EdgeTrainer.execute_training()`), `TPMAttestation.generate_attestation_token()` connects over TCP to its dedicated `sim-tpm-{fog_num}-{edge_num}` container on port `2321` to execute a real hardware `tpm2_quote` over the round nonce and `PCR 0`.
        
3. **In-Memory Virtual Fog Clusters (`src/tier_cloud/cloud_aggregator_sim.py`):**
    
    - `sim_server.py` enforces **100% client participation** (`fraction_fit = 1.0`, `fraction_evaluate = 1.0`, `round_timeout = 180.0`) and initializes the simulation `CloudAggregator`.
        
    - During initialization, `CloudAggregator` creates $K$ (`num_fogs`) isolated virtual Fog instances in memory—each with its own `ZeroTrustGatekeeper`, `TrustDatabase`, `AdminConsole`, state history, and dedicated `[FOG i SERVER]` / `[FOG i CLIENT]` JSONL loggers.
        
    - In `aggregate_fit()`, incoming Edge updates are grouped by their `[EDGE {fog_num}_{edge_num}]` prefix into `fog_groupings[fog_num]`.
        
    - For each virtual Fog cluster $i \in \{1 \dots K\}$, `CloudAggregator` sequentially executes:
        
        1. **Scheduled Admin Actions:** `self.admin_consoles[fog_num].execute_scheduled_updates(server_round, fog_num)`.
            
        2. **Phase 5 (Cryptographic Gatekeeper):** Calls `self.gatekeepers[fog_num].filter_node_updates(...)`, invoking `tpm2_checkquote` and `tpm2_print` inside `sim-cloud` against `/app/runtime/tpm_state/edge_{i}_{j}/ak.pub` and `/app/runtime/tpm_state/pcr_ledger.json`.
            
        3. **Phase 6 (SHAP Verification & Regional Aggregation):** Runs `shap_weighted_aggregate()` (`GradientShap` + Median/MAD $\mu_s - 2\sigma_s$ filtering), updates the cluster's `TrustDatabase` for saboteurs and rewarded nodes, and writes the 3-pillar state files (`results/<experiment_name>/{trustdb,shap_scores,performance_metrics}/fog_{i}_state.json`).
            
    - **Phase 7 (Global Cloud Aggregation):** All surviving regional Fog models (`virtual_fog_models`) are combined via trust-weighted `federated_averaging` using `virtual_fog_weights` (`regional_trust`), checked against `_evaluate_rollback_sanity_check()`, quantized (`compress_weights`), and evaluated on the centralized test split via `GlobalEvaluator`.
        
4. **Full UI & Telemetry Spoofing:**
    
    - `CloudAggregator` emits the exact Phase 1, 2, 6, and 7 log triggers expected by `verification/verify_pipeline.py`, allowing the live Rich pipeline monitor to track virtual Fog and Edge progress seamlessly.
        

## 2. How to Run Simulation Mode

Unlike Deployment Mode, Simulation Mode runs `flwr run` **inside the `sim-cloud` container** (`/python/venv/bin/flwr`), which means it does **not** overwrite your host's `~/.flwr/config.toml` and cannot fail due to a mismatched `flwr` CLI on your host PATH.

### Step 1: Configure Topology & Training Parameters

Edit the YAML configuration files in `config/` before booting:

- **`config/network.yaml`:** Set `num_fogs` and `uniform_edges_per_fog` (or `custom_fog_topology`) to define the total number of Edge actors and `sim-tpm-*` containers to spawn.
    
- **`config/training.yaml`:** Configure `dataset`, `num_rounds`, `local_epochs`, `batch_size`, `quantization_bits`, and non-IID parameters (`n_classes_per`, `power_law_a`).
    
- **`config/security.yaml` & `config/threat.yaml`:** Set attacker ratios (`label_flip_ratio`, `grad_manip_ratio`, `backdoor_ratio`, `shap_aware_ratio`) and round-specific `adversaries`.
    

### Step 2: Terminal A — Boot the Simulation Network & TPM Array

Open Terminal A and run the simulation boot script:

``` bash
cd zta-fl-flower
chmod +x scripts/simulation/boot_network_sim_docker.sh scripts/simulation/generate_compose_sim.sh scripts/setup/*.sh

# Standard boot (reuses existing TPM state if present)
./scripts/simulation/boot_network_sim_docker.sh

# OR: Wipe and re-provision all TPM 2.0 NVRAM states from scratch
./scripts/simulation/boot_network_sim_docker.sh --new-tpm
```

**What `boot_network_sim_docker.sh` does:**

1. Validates that `docker` and `python3` are available and the Docker daemon is running.
    
2. If `--new-tpm` is passed, deletes `runtime/tpm_state` to force fresh Attestation Key (AK) and PCR provisioning.
    
3. Reads the topology from `config/network.yaml` (`parse_topology.py`) and provisions host directories under `runtime/tpm_state/edge_{i}_{j}` (`setup_tpm.sh`).
    
4. Pulls `panagiotispapadopoulos/zta-cloud-node:latest` and `panagiotispapadopoulos/zta-edge-node:latest` if not cached locally.
    
5. Runs `generate_compose_sim.sh` to generate `docker/docker-compose.yml` containing:
    
    - `sim-cloud` (automatically configured with NVIDIA GPU reservations if `nvidia-smi` is detected on the host).
        
    - $N$ `sim-tpm-{i}-{j}` containers, each launching `swtpm socket` on TCP port `2321` (control port `2322`), running `tpm2_startup -c`, and executing a factory bootstrap attestation (`FACTORY_BOOT_NONCE`) to export `ak.pub`, `clean_pcr.bin`, and `tpm_id.txt`.
        
6. Starts the containers (`docker compose up -d --wait`) and runs `scripts/setup/collect_ledgers.py`, polling until all $N$ `sim-tpm-*` containers finish generating their factory PCR state and compiling `runtime/tpm_state/pcr_ledger.json`.
    
7. Creates `runtime/infra/.network_ready` and prints:
    
    `ENGINE IS LIVE. RUN DEPLOY_CODE_SIM_DOCKER.SH`
    

Keep Terminal A running (pressing `Ctrl+C` in Terminal A will cleanly tear down the `sim-cloud` and `sim-tpm-*` containers).

### Step 3: Terminal B — Compile Artifacts & Dispatch the Simulation

Once Terminal A reports `ENGINE IS LIVE`, open **Terminal B** and run:

``` bash
cd zta-fl-flower
chmod +x scripts/simulation/deploy_code_sim_docker.sh
./scripts/simulation/deploy_code_sim_docker.sh
```

**What `deploy_code_sim_docker.sh` does:**

1. Verifies `runtime/infra/.network_ready` and confirms the Docker Compose containers are running.
    
2. Kills any stale Flower/Ray processes inside `sim-cloud`, clears old logs in `logs/nodes/`, and counts the active `sim-tpm-*` containers (`TOTAL_EDGES`).
    
3. Prints the visual architecture tree showing `sim-cloud`, the virtual Fog clusters, the ephemeral Ray Edge actors, and their TCP mappings to `sim-tpm-{i}-{j}:2321`.
    
4. Runs **The Big Crunch (`scripts/setup/build_artifacts.py`)** in a temporary container to verify or compile the partitioned dataset `.pt` files in `data/<dataset>/artifacts/` for all `TOTAL_EDGES` partitions.
    
5. Rewrites `options.num-supernodes = TOTAL_EDGES` in `/tmp/pyproject.cloud.toml`, copies it into `sim-cloud:/app/pyproject.toml`, and touches `config/training.yaml`.
    
6. Dispatches the simulation in the background inside `sim-cloud`:
    
    ``` bash
    docker exec -d -w /app \
      -e RUN_CFG="mode=\"sim\" tier=\"cloud\" min_clients=$TOTAL_EDGES" \
      -e N_SIM="$TOTAL_EDGES" \
      sim-cloud bash -c "/python/venv/bin/flwr run . local --run-config \"\$RUN_CFG\" > /app/logs/system/run_cloud.log 2>&1"
    ```
    
## 3. Running Deployment Mode with Apptainer (HPC / Rootless)

For shared HPC clusters or environments without root Docker daemon access, the entire simulation runs natively via **Apptainer** SIF images (`cloud_node.sif`, `edge_node.sif`).

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
chmod +x scripts/simulation/boot_network_sim_apptainer.sh
./scripts/ops/boot_network_sim_apptainer.sh
```

### Terminal B — Dispatch FABs to Apptainer

Once Terminal A prints `ENGINE IS LIVE. RUN DEPLOY_CODE_APPTAINER.SH`:

```bash
cd zta-fl-flower
chmod +x scripts/simulation/deploy_code_sim_apptainer.sh
./scripts/simulation/deploy_code_sim_apptainer.sh
```

## 4. Monitoring, Verification & Post-Run Acceptance

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
### Post-Run Plots & Security Evaluation

Simulation Mode produces the exact same 3-pillar and global JSON schema in `results/<experiment_name>/` as Deployment Mode, so all analysis tools work identically:

``` bash
# Generate accuracy, F1, loss, ASR, PGD, and FGSM plots
python3 tools/plot_metrics.py results/<experiment_name>

# Compute post-training security metrics (ASR, FAR, FRR, FPR, adversarial robustness)
python3 scripts/ops/evaluate_security.py '<experiment_name>'
```

## 5. Troubleshooting Simulation Mode

- `TPM emulator throws errors` or `collect_ledgers.py` hangs during boot:**
    
    - Run `./scripts/simulation/boot_network_sim_docker.sh --new-tpm` to wipe stale TPM state files in `runtime/tpm_state/` and force every `sim-tpm-*` container to regenerate its `ak.pub`, `clean_pcr.bin`, and `tpm_id.txt`.
        
- **`CRITICAL: Dataset artifact missing at ...` during training:**
    
    - If you changed `num_fogs` or `uniform_edges_per_fog` in `config/network.yaml`, ensure `build_artifacts.py` runs cleanly during `deploy_code_sim_docker.sh` so that `edge_0.pt` through `edge_{TOTAL_EDGES-1}.pt` exist in `data/<dataset>/artifacts/`.
        
- **Ray Out-of-Memory (OOM) with large client counts:**
    
    - In `sim_server.py`, `torch.set_num_threads(1)` is enforced on CPU and `EdgeTrainer` explicitly purges GPU VRAM (`torch.cuda.empty_cache()`, `gc.collect()`) at the end of every client `fit()` call before the Ray actor returns. If host RAM is constrained at 100+ clients, reduce `batch_size` or `shap_explain_count` in `config/training.yaml`.