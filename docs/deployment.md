# 3-Tier Distributed Deployment Guide (`mode = "deploy"`)

**Baseline Requirement:** Flower `1.30.0` | Python `>=3.10,<4.0` (CPython `3.11.x` recommended)  
**Supported Container Engines:** Docker Compose & Apptainer (Rootless / HPC)

Deployment mode runs separate Flower services for the Cloud, Fogs and Edges. Cloud–Fog communication uses TLS, Edge–Fog communication uses NGINX mTLS proxies, and each Fog's client/server processes exchange model updates through a TCP IPC bridge. Attestation uses software TPM 2.0 emulators (`swtpm`); it does not establish hardware-backed isolation.

Single-host deployment runs many concurrent services and requires sufficient memory and process limits. Docker memory exhaustion can terminate containers; inspect container exit status and OOM diagnostics. On HPC systems, load the required Apptainer module and check the allocation's process and file-descriptor limits. Qualify the intended topology using the [preflight plan](preflight.md).

## 1. Three-tier execution topology

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
```

### Round lifecycle

1. **Cloud dispatch and Fog synchronization.** The Cloud sends the current encoded global model to Fog ClientApps. Each `FogBridgeClient` forwards the model and round identifier to its local `FogBridgeServer` over TCP IPC and waits for the regional result.
2. **Edge admission.** Before releasing model parameters, the Fog requests a fresh quote through Flower's `get_properties` operation. The gatekeeper checks the enrolled identity, signature, nonce, timestamp, PCR state and quarantine status. Rejected clients receive attestation-only instructions without model parameters. Admission and returned updates use separate challenges.
3. **Local training and upload.** Admitted Edges load their prepared partitions and execute the configured benign, label-flip, gradient-manipulation, backdoor or SHAP-aware training path. Backdoor sampling includes all local rows, including rows already carrying the target label. Optional FP32/INT8 inference probes precede the upload quote. The software TPM uses an ECDSA P-256 attestation key; its quote binds the nonce and timestamp alongside the PCR state. The token is transported as JSON containing encoded quote/signature fields, with no separate application-layer encrypted envelope.
4. **Regional verification and aggregation.** The shared regional processor verifies upload quotes and model payloads, then evaluates eligible updates using serialized GradientSHAP attribution. Filtering uses the cohort population mean minus two standard deviations. Accepted scores strictly above the mean receive trust rewards. Ordinary weights use `max(0, stability) * validation_accuracy * sqrt(sample_count)`; eligible survivors receive uniform weights when the ordinary total is negligible. The Fog records trust, SHAP and timing outcomes and relays its encoded regional result over IPC.
5. **Cloud aggregation, rollback and reporting.** The Cloud combines regional models using their reported regional weights. It checks the encoded-and-decoded candidate against the previous accepted model on validation data. Accuracy at or below `rollback_threshold` times the previous accuracy causes rollback. Test-set evaluation and checkpoint/metric persistence follow separately.

The [architecture parity report](architecture-parity.md) describes the mathematical conventions, attack mechanisms and implementation limits. Both modes share the [regional processor](../src/tier_fog/round_processing.py), [admission logic](../src/tier_fog/client_instructions.py) and [Cloud model-state handling](../src/tier_cloud/model_state.py). Virtual Fogs in simulation do not reproduce deployed transport timing or process isolation.

### Configurable Flower timeout

`round_timeout` in `config/network.yaml` sets the Flower client-response deadline in seconds for both Cloud and Fog servers, in Docker and Apptainer. The default is `3600.0`; supply a finite positive number. The shared YAML travels through the existing configuration mounts and application bundle. Each server logs the resolved value at startup, and the Cloud run JSON records `metadata.round_timeout`. Changes take effect after dispatching a new run.

The Cloud waits for Fog responses, which include regional work; Fog servers wait for Edge responses. This setting does not impose a deadline on the entire experiment or on server-side aggregation/evaluation. `socket_timeout` remains an independent IPC limit. Set both to accommodate the measured workload. See the [preflight plan](preflight.md) for exact trial settings and evidence to check.

## 2. Environment and version requirements

The repository and runtime images use Flower `1.30.0`. The native-CLI Docker dispatcher must use the same version from the project environment. Python compatibility is declared in `pyproject.toml`; CPython 3.11 is the documented host setup.

Keep the host CLI and container runtime versions aligned. An upgrade requires validating application submission, connection configuration and port routing together; changing the host package alone does not update container services.

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

The Python and Flower executable paths must resolve inside the project environment. `flwr --version` must report `Flower version: 1.30.0`, and `python -m pip check` must report `No broken requirements found.`

## Local project files and runtime images

Docker's generated Compose file mounts the local `src/` and `scripts/` read-only into every project application service. The Edge startup script is launched from that mount. Configuration and writable data, logs, results and TPM state keep their existing mounts. Apptainer application services also bind the local source and scripts; setup and dispatch commands that already bind the whole checkout continue to do so. Python's import path includes `/app`, so locally supplied modules take precedence over the image's installed project copy.

Flower dispatch still packages the current checkout into application bundles. Stop the network before changing source during an experiment, then boot and dispatch again to start fresh processes and bundles. Changes to mount definitions require recreating the containers through the boot script.

Existing images provide dependencies and system tools. Moving scripts, editing source or reorganizing documentation does not require rebuilding or uploading images. A dependency or system-tool change may require an updated image. Running an image manually without these mounts still uses its bundled files. For separate hosts, each host must have the matching checkout and data at its mounted paths.

## 3. Running Deployment Mode with Docker Compose

**Connection configuration:** the repository's [Docker boot script](../scripts/deployment/boot_network_docker.sh) writes the `cloud` and `fog1..fogN` connection profiles to the host's `~/.flwr/config.toml`, replacing its contents. Back up that file before booting if it contains other saved endpoints. This describes the project's launcher behavior, not a limitation of every Flower configuration workflow.

### Terminal A — Boot the 3-Tier Infrastructure

Keep this terminal open for the duration of your experiments:

```bash
cd zta-fl-flower
source .venv/bin/activate
rehash

# Make scripts executable
chmod +x scripts/deployment/boot_network_docker.sh scripts/setup/*.sh

# Boot with NGINX mTLS and software TPM attestation (Default)
./scripts/deployment/boot_network_docker.sh

# OR: Boot in plain-text mode (bypasses NGINX mTLS and TPM verification)
./scripts/deployment/boot_network_docker.sh --insecure
```

**What `boot_network_docker.sh` does:**

1. Parses `config/network.yaml` via `scripts/setup/parse_topology.py`.
    
2. Generates the Root CA, server/client certificates (`setup_security.sh`), and the NGINX mTLS reverse-proxy configuration (`setup_nginx.sh`) unless `--insecure` is passed.
    
3. Initializes host directories for each Edge node's software TPM (`setup_tpm.sh`).
    
4. Reuses the cached cloud and edge images (pulling them only if absent) and generates `docker/docker-compose.yml` (`generate_compose.sh`).
    
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
    
4. Runs the dataset artifact builder (`scripts/setup/build_artifacts.py`) inside a temporary cloud container to compile and cache scaled, SMOTE-balanced, non-IID dataset partitions in `data/<dataset>/artifacts/` when missing or incompatible with the selected input and preprocessing settings. Edge partitions are regenerated for the current run.
    
5. Dispatches `flwr run . fog{i}` in the background for every Fog stack, waits for their Control ports, and dispatches `flwr run . cloud` to trigger global training.
    

## 4. Running Deployment Mode with Apptainer (HPC / Rootless)

For shared HPC clusters or environments without root Docker daemon access, the entire 3-tier deployment runs natively via **Apptainer** SIF images (`cloud_node.sif`, `edge_node.sif`, `nginx.sif`) and UNIX domain sockets (`swtpm.sock`).

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

Obtain the runtime images once per machine. This helper reuses existing SIF files; repository-only changes do not require converting or downloading them again:

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
# OR: ./scripts/deployment/boot_network_apptainer.sh --insecure
```

### Terminal B — Dispatch FABs to Apptainer

Once Terminal A prints `ENGINE IS LIVE. RUN DEPLOY_CODE_APPTAINER.SH`:

```bash
cd zta-fl-flower
chmod +x scripts/deployment/deploy_code_apptainer.sh
./scripts/deployment/deploy_code_apptainer.sh
```

_(Note: `deploy_code_apptainer.sh` executes `/python/venv/bin/flwr` directly inside `images/cloud_node.sif`, so it is insulated from host `flwr` PATH mismatches.)_

## 5. Monitoring and results

Use the [monitoring and results guide](monitoring-and-results.md) for the live pipeline display, node logs, saved models, metrics and post-run plots. Run the [source tests](testing.md) before live validation.
