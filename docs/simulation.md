# Hybrid Simulation Guide (`mode = "sim"`)

**Runtime:** Flower `1.30.0` with simulation dependencies.  
**Container engines:** Docker Compose and Apptainer.

Simulation runs Cloud and virtual Fog processing in a central Flower/Ray engine, with ephemeral Edge actors and one external software TPM per Edge. This avoids separate Flower services for every simulated node. Supported scale depends on the available resources and must be qualified on the execution host.

## 1. Simulation architecture

```text
Central Flower/Ray engine
  Cloud aggregation, validation and test reporting
  Virtual Fogs
    Per-Fog gatekeeper, TrustDatabase and outcome history
    Shared regional verification and serialized GradientSHAP
  Ephemeral Edge actors
    Prepared partitions, configured training and TPM quotes
      |
      +-- Docker: TCP to sim-tpm-<fog>-<edge>:2321
      +-- Apptainer: per-Edge swtpm UNIX socket
```

`mode="sim"` selects the simulation server/client adapters. Each Edge's zero-based `partition-id` maps to a Fog/Edge coordinate using the configured topology. Admission quotes are verified before model release; separate challenged quotes accompany returned updates. External `swtpm` processes execute TPM commands with ECDSA P-256 attestation keys. This is software emulation, not hardware-isolation evidence.

The Cloud strategy maintains a gatekeeper and trust/history state for each virtual Fog. It groups responses by authenticated Fog assignment and invokes the same regional processor used by deployment. That processor validates quotes and payloads, computes GradientSHAP scores serially, filters using population mean/std, applies trust updates and aggregates eligible models. Regional models cross the transport encode/decode boundary before Cloud aggregation and validation-based rollback. The Cloud reports test-set metrics separately.

Simulation emits explicitly marked phase events for the pipeline monitor. Virtual IPC events describe logical progress; they are not measurements of deployed IPC or network latency. Full client participation is requested for training, subject to admission and quarantine. Global evaluation is centralized rather than dispatched to Edges.

See the [architecture parity report](architecture-parity.md) for algorithm definitions and limitations, and the [deployment guide](deployment.md) for the deployed round lifecycle. Shared code does not establish equivalence of timing, physical isolation or security boundaries between the two modes.

## Local project files and runtime images

Docker mounts the local `src/` and `scripts/` read-only into the simulation engine and TPM containers, along with configuration. The engine also mounts the local `pyproject.toml`. Artifact compilation uses the local source and builder script. Apptainer binds the checkout for simulation and TPM provisioning, and mounts the local builder inputs explicitly. Python imports include `/app`; Apptainer simulation also retains `/tmp` for its runtime patch.

The images supply dependencies and system tools. Source and documentation changes do not require rebuilding or publishing Docker images or recreating SIF files. Stop an active run before editing source, then boot and dispatch again so processes and Flower bundles use the updated checkout. The boot scripts recreate Docker containers when mounts change. Changes to installed dependencies or system tools may still require new images.

## 2. How to Run Simulation Mode

Unlike Deployment Mode, Simulation Mode runs `flwr run` **inside the `sim-cloud` container** (`/python/venv/bin/flwr`), which means it does **not** overwrite your host's `~/.flwr/config.toml` and cannot fail due to a mismatched `flwr` CLI on your host PATH.

### Step 1: Configure Topology & Training Parameters

Edit the YAML configuration files in `config/` before booting:

- **`config/network.yaml`:** Set `num_fogs` and `uniform_edges_per_fog` (or `custom_fog_topology`) to define the total number of Edge actors and `sim-tpm-*` containers to spawn.
    
- **`config/training.yaml`:** Configure `dataset`, `num_rounds`, `local_epochs`, `batch_size`, and non-IID parameters (`n_classes_per`, `power_law_a`).
    
- **`config/security.yaml`:** Set attacker ratios (`label_flip_ratio`, `grad_manip_ratio`, `backdoor_ratio`, `shap_aware_ratio`).
    

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
    
4. Runs the dataset artifact builder (`scripts/setup/build_artifacts.py`) in a temporary container to verify or compile the partitioned dataset `.pt` files in `data/<dataset>/artifacts/` for all `TOTAL_EDGES` partitions.
    
5. Uses the local `pyproject.toml` mounted read-only at `/app/pyproject.toml`, passes the node count through Flower's `--federation-config` option, and touches `config/training.yaml`.
    
6. Dispatches the simulation in the background inside `sim-cloud`, streaming output to `logs/system/run_cloud.log`. The batch runner uses the same launcher in foreground mode so it can observe its exit status.

## 3. Running Simulation Mode with Apptainer (HPC / Rootless)

For shared HPC clusters or environments without root Docker daemon access, the entire simulation runs natively via **Apptainer** SIF images (`cloud_node.sif`, `edge_node.sif`).

Initialize the required modules and limits in each launch terminal. Module names and allowed limits depend on the cluster; the commands below are site-specific examples.

1. Load the module. Example module names for [AUTh HPC](https://hpc.auth.gr):
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

Obtain the runtime images once per machine. Existing SIF files are reused; repository-only changes do not require downloading or converting them again:

``` bash
cd zta-fl-flower
chmod +x scripts/ops/build_images_apptainer.sh
./scripts/ops/build_images_apptainer.sh
```

### Terminal A — Boot the Apptainer Federation

```bash
cd zta-fl-flower
chmod +x scripts/simulation/boot_network_sim_apptainer.sh
./scripts/simulation/boot_network_sim_apptainer.sh
```

### Terminal B — Dispatch FABs to Apptainer

Once Terminal A prints `ENGINE IS LIVE. RUN DEPLOY_CODE_SIM_APPTAINER.SH`:

```bash
cd zta-fl-flower
chmod +x scripts/simulation/deploy_code_sim_apptainer.sh
./scripts/simulation/deploy_code_sim_apptainer.sh
```

## 4. Monitoring, results and overnight runs

Use the [monitoring and results guide](monitoring-and-results.md) for the live pipeline display, logs, saved models and post-run analysis. For sequential unattended simulations with per-run configuration, time limits and diagnostic archives, follow the [overnight batch guide](batch-runs.md). The batch runner boots once, reuses the TPM services and one Flower coordinator across separate training runs, and tears down once after the queue. Use the [preflight run plan](preflight.md) for exact configurations and acceptance criteria before long experiments.

### Flower timeout

Set `round_timeout: 3600.0` in `config/network.yaml` to allow 3,600 seconds for Flower client responses. Both Docker and Apptainer read this setting through the normal shared configuration; Apptainer does not impose a separate minimum. It must be a finite positive number. Missing or invalid values fail server startup, and batch queue planning checks it before boot.

The server logs `Flower client-response timeout: ... seconds` at startup and records `metadata.round_timeout` in the run JSON. Managed batches also preserve the complete executed YAML and resolved configuration. Change the setting between runs and dispatch again; an already-running server does not reload it.

This is Flower's request timeout, not an end-to-end deadline covering artifact preparation, Fog aggregation and Cloud evaluation. `socket_timeout` still controls IPC waits, and the batch's `--stall-timeout` and `--run-timeout` still control progress and total duration. A larger Flower timeout does not extend either batch deadline.

## 5. Troubleshooting Simulation Mode

- **TPM emulator errors or stalled ledger collection during boot:**
    
    - Run `./scripts/simulation/boot_network_sim_docker.sh --new-tpm` to wipe stale TPM state files in `runtime/tpm_state/` and force every `sim-tpm-*` container to regenerate its `ak.pub`, `clean_pcr.bin`, and `tpm_id.txt`.
        
- **`CRITICAL: Dataset artifact missing at ...` during training:**
    
    - If you changed `num_fogs` or `uniform_edges_per_fog` in `config/network.yaml`, ensure `build_artifacts.py` runs cleanly during `deploy_code_sim_docker.sh` so that `edge_0.pt` through `edge_{TOTAL_EDGES-1}.pt` exist in `data/<dataset>/artifacts/`.
        
- **Ray Out-of-Memory (OOM) with large client counts:**
    
    - In `sim_server.py`, `torch.set_num_threads(1)` is enforced on CPU and `EdgeTrainer` explicitly purges GPU VRAM (`torch.cuda.empty_cache()`, `gc.collect()`) at the end of every client `fit()` call before the Ray actor returns. If host RAM is constrained at 100+ clients, reduce `batch_size` or `shap_explain_count` in `config/security.yaml`.
