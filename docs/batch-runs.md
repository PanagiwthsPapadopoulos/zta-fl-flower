# Overnight simulation batches

[batch_runs.py](../scripts/ops/batch_runs.py) supervises one simulation at a time using the existing Docker or Apptainer launchers. It does not run deployment mode. The host needs Python 3.9 or later and PyYAML, plus the selected container runtime and the usual simulation prerequisites. Machine-learning dependencies remain inside the simulation images.

## How to use

**Start only the batch runner. Do not start `boot_network_sim_*.sh` or `deploy_code_sim_*.sh` yourself.** The runner calls both scripts automatically for each queued run. For Docker, the Docker daemon must be running; the project's simulation network must be stopped. For Apptainer, use the execution host with Apptainer installed and stop any existing standalone simulation first.

1. Set `config/` to the baseline settings shared by the experiments, including the fixed network topology and intended dataset path.
2. Create one folder per experiment under `experiments_queue/`. Put its YAML overrides inside that folder, as shown in the next section. Folder names determine execution order.
3. From the repository root, validate the queue without starting anything:

   ```sh
   python3 scripts/ops/batch_runs.py --runtime docker --dry-run
   ```

4. Start the batch:

   ```sh
   python3 scripts/ops/batch_runs.py --runtime docker
   ```

   For Apptainer, replace `docker` with `apptainer` in both commands. Keep the session alive for the overnight run.

The terminal prints the batch archive location, each run's identity and its outcome. The archive's `batch.json` shows the queue status; each attempt's `boot.log`, `dispatch.log` and `status.json` provide startup diagnostics and the outcome reason. Ctrl+C stops the batch through its cleanup path. Use the recovery instructions below to continue pending entries later.

## Archived paper reproduction queue

[`experiments_archived/paper_80/`](../experiments_archived/paper_80/) contains the **80 executions (16 configurations × 5 seeds)** selected in [runs.md §2](runs.md). Each experiment directory contains only `training.yaml`, `security.yaml` and `network.yaml`. These are complete configuration snapshots, with descriptions and source references in YAML comments; no generator, manifest or extra configuration format is needed. Names encode configuration order, seed order, dataset, attack and attack parameters. `run_name` matches the folder; the supervisor adds its unique attempt identity at execution.

This queue is inactive by default. The existing `--queue` flag selects it directly, so **no rename or new flag is necessary**. It is a set of input configurations, separate from output archives under `results/batch_archives/`.

From the repository root:

```sh
# Validate all 80 entries without launching training:
python3 scripts/ops/batch_runs.py --runtime docker --queue experiments_archived/paper_80 --dry-run

# Execute all 80 sequentially:
python3 scripts/ops/batch_runs.py --runtime docker --queue experiments_archived/paper_80

# Or execute on the Apptainer host:
python3 scripts/ops/batch_runs.py --runtime apptainer --queue experiments_archived/paper_80
```

Use only one execution command. The baseline `config/network.yaml` must also describe 10 fogs with 10 edges each (`custom_fog_topology: "[]"`); the runner rejects a queue whose topology differs from the baseline. The checked-in queue pins this topology. Hardware settings are captured from the current config (`gpu_enabled_tiers: [cloud, fog, edge]`, `max_concurrent_gpus: 2`); adapt these YAML values to the execution host when necessary. Batch deadlines retain their normal defaults; full-data, 100-client runs may require longer limits as described below. A dry-run validates the queue contract, not memory capacity, dataset contents, runtime availability or expected runtime.

| Folder prefix | Configuration | Executions |
| --- | --- | ---: |
| `01`–`03` | Clean Edge-IIoTset, CIC-IDS2017, UNSW-NB15 | 15 |
| `04`–`06` | Edge label flip, β = 0.1 / 0.2 / 0.3 | 15 |
| `07`–`09` | Edge gradient manipulation, β = 0.1 / 0.2 / 0.3, α = 3 | 15 |
| `10` | Edge backdoor, β = 0.2, local poisoning = 10% | 5 |
| `11`–`16` | Edge FGSM / PGD-7 / PGD-20, each at ε = 0.05 / 0.10 | 30 |

Every configuration uses seeds `42, 123, 456, 789, 1011`. No exploratory, ablation or additional parameter-grid runs are included.

### Full dataset inputs

All YAMLs use `dataset_fraction: 1.0` and the following cleaned full-data inputs, verified present and nonempty in this checkout on 2026-11-05:

| Dataset | `dataset_path` |
| --- | --- |
| Edge-IIoTset | `data/edge_iiotset/full_dataset/network_traffic_samples.csv` |
| CIC-IDS2017 | `data/cic_ids2017/full_dataset/network_flows.csv` |
| UNSW-NB15 | `data/unsw_nb15/full_dataset/intrusion_records.csv` |

These are repository-relative paths, compatible with the launchers' `/app` mount on either runtime, rather than host-specific macOS absolute paths. Full-data files are Git-ignored: transfer or prepare them at these locations on another execution host using [dataset preparation](dataset-preparation.md). Selecting the full CSV does not mean every record trains on every client: the implementation still partitions client data into bounded subsets. Existing preparation reports record cleaning decisions; this queue creation did not redownload or re-clean the datasets.

### Article sanity check and explicit assumptions

Checked against the local nine-page article `notes/ZeroTrust_Agentic_Federated_Learning_for_Secure_IIoT Defense Systems.pdf`, especially pp. 3–5 (V.A–V.C, VI.A–VI.E, Tables II–IV), on 2026-11-05. The PDF is local and Git-ignored. This is a configuration sanity check, not a claim that the published metrics have been reproduced.

| Item | Queue value / finding | Article evidence |
| --- | --- | --- |
| Seeds | 42, 123, 456, 789, 1011 | VI.A, Reproducibility |
| Training | 100 rounds, 5 local epochs, batch 128, learning rate 0.001; CNN-LSTM; Adam implemented in source | VI.A, Implementation |
| Topology / label skew | 100 agents, 10 fogs, 3 classes per agent | VI.A |
| Preprocessing | SMOTE enabled, train/validation/test 70/15/15; 40-dimensional PCA implemented in source | VI.A |
| Attestation / rollback | Freshness 60 seconds; rollback ratio 0.80 | VI.A and V.B |
| Trust policy | Source implements initial 0.7, reward 0.02, penalty ×0.5, threshold 0.6, 5 attestations, reset 0.65; these are not YAML keys | V.A |
| SHAP | Background limit 100; existing GradientSHAP implementation | VI.A, SHAP Configuration |
| Quantization | 8-bit transport and existing INT8 Edge inference enabled | VI.A; exact 475KB size is not established |
| Defense augmentation | `benign_adv_ratio: 0.3` on every run, including clean-data runs | V.C specifies 70% clean / 30% adversarial training |
| Label flipping | β = 0.1 / 0.2 / 0.3 confirmed; **p_flip = 0.2 is an assumption** | Table III omits p_flip; VI.A lists 0.2 / 0.5 / 1.0 |
| Gradient manipulation | β = 0.1 / 0.2 / 0.3; α = 3 | Table III and VI.C |
| Backdoor | β = 0.2; 10% local poisoning | VI.A and VI.E |
| Evasion | FGSM / PGD-7 / PGD-20, ε = 0.05 / 0.10 | Table IV |

The following settings make the runnable protocol explicit where the paper is incomplete:

- **Label probability:** `p_flip: 0.2` retains the repository convention and is one of VI.A's listed values. It is not established as the Table III setting. The implementation can draw the original label during replacement, so the probability of an actual label change is `p_flip * (C - 1) / C`. The reporting gap in runs.md §7.4 remains unresolved.
- **Defense versus evaluation:** the implementation shares the attack selector, epsilon, step size and iteration count between local adversarial augmentation and robustness evaluation. The six evasion configurations therefore change both. The other ten configurations use FGSM with epsilon 0.05 as an explicit assumption. “Clean” means no malicious clients and clean evaluation, while ZTA-FL's adversarial defense stays enabled. The paper does not establish this exact training/evaluation coupling.
- **PGD step size:** `benign_alpha: 0.2` retains the repository default; the article does not specify it. It exceeds both epsilon budgets, but PGD projects back into the epsilon ball. This choice can affect attack strength. FGSM uses epsilon directly and ignores the PGD step size and iteration count.
- **Backdoor details:** target class 0, trigger indices `[-3, -2, -1]`, trigger value 1.5 are existing implementation conventions, not paper-specified values.
- **Other conventions:** power-law shape 0.4, 10 SHAP explanations, gradient clipping norm 1.0, train-only fitted transforms (`simulate_global_leakage: false`), and no global clipping bounds on PCA coordinates are pinned from the repository configuration, not established by the article.

YAML cannot supply missing protocol/layer feature skew, change the implementation's model parameter count to the reported 487K, guarantee a 475KB model or reproduce the paper's TensorFlow/hardware environment and 3.1-second SHAP timing. The quantity-skew code targets 500–5000 examples but actual allocations depend on available classes. See [architecture parity](architecture-parity.md) for these implementation limits. The Table III/VI discrepancies and deduplication remain the scope decisions in runs.md; identical reported means alone do not prove shared underlying executions.

## Runtime images and local files

The batch runner uses the same [simulation launchers](simulation.md) as individual runs. They mount the local source and scripts into the existing Docker or Apptainer runtime images. Repository-only changes do not require rebuilding or uploading images. Finish or stop a batch before editing its source; the saved fingerprint guards against continuing pending runs with a different implementation.

## Network lifecycle

**The simulation network is torn down after every attempt**, including failed attempts, and recreated for the next one using the same topology. This stops the simulation containers or processes; it does not shut down the host's Docker daemon or uninstall Apptainer.

Each attempt follows this sequence:

```text
Apply this run's configuration
  → boot_network_sim_docker.sh / boot_network_sim_apptainer.sh
  → wait for network health and complete TPM provisioning
  → deploy_code_sim_docker.sh / deploy_code_sim_apptainer.sh
      → prepare/reuse dataset artifacts
      → run the simulation
  → detect completion, crash or timeout
  → collect live diagnostics
  → stop this attempt's simulation processes and network
  → archive logs, runtime state and results
  → restore the original configuration
  → start the next queued run
```

Fresh TPM state is provisioned for each attempt; the previous state is retained in its archive. Existing container images and datasets remain available between runs. Artifact preparation follows the existing provenance checks rather than unconditionally rebuilding all training tensors. If shutdown or archival cannot be completed safely, the batch stops instead of starting the next run.

## Queue and configuration

Create directories under `experiments_queue/`, ordered by name:

```text
experiments_queue/
  01_clean/
    training.yaml
    security.yaml
  02_label_flip/
    training.yaml
    security.yaml
```

Each file is a mapping of keys to override in the corresponding baseline file. Complete files are also accepted. For example, `01_clean/training.yaml` could contain:

```yaml
dataset: unsw_nb15
dataset_path: data/unsw_nb15/full_dataset/intrusion_records.csv
random_seed: 42
num_rounds: 100
```

`02_label_flip/security.yaml` could contain:

```yaml
label_flip_ratio: 0.2
p_flip: 0.2
```

Every run starts from the same snapshot of the workspace's `config/`, with its own overrides applied. Values do not carry over from earlier runs. Specify the intended dataset in each run that needs it; an omitted setting inherits the baseline, not the previous queued experiment. Likewise, a clean run must explicitly disable any attack roles enabled in the baseline.

Only `network.yaml`, `training.yaml` and `security.yaml` are supported. Unknown keys, keys placed in the wrong file, invalid round counts, missing/invalid `round_timeout` and changed effective Fog/Edge topology fail validation before any run starts. Other network settings, such as transport quantization and `round_timeout`, may differ. This validation checks the queue contract; it does not replace application-level configuration validation.

Queue names use letters, digits, underscores and hyphens. Each attempt receives a unique generated `run_name`, also recorded as `batch_attempt`, so repeated batches cannot overwrite previous results. A queued `run_name` is retained in the override snapshot but replaced for execution. The complete plan is frozen at batch start; later queue edits do not change that batch.

Review the resolved plan first:

```sh
python3 scripts/ops/batch_runs.py --runtime docker --dry-run
```

Start an overnight batch using either runtime:

```sh
python3 scripts/ops/batch_runs.py --runtime docker
python3 scripts/ops/batch_runs.py --runtime apptainer
```

Run only one of those commands at a time. `--queue /path/to/queue` selects a different queue. Paths to scripts and the default queue are resolved from the repository, independently of the invoking shell's working directory. Keep the terminal session alive, for example with an existing terminal multiplexer on the execution host.

The runner owns this workspace's simulation environment while active. Stop any standalone simulation first, and do not run another launcher or edit active configuration/source files during a batch. The supervisor has an exclusive workspace lock, rejects already-active infrastructure, checks for source changes between attempts and restores the baseline configuration after each safely stopped attempt. A fresh simulation network and TPM state are created for each attempt with the same topology; previous runtime state is archived, never silently discarded.

## Completion, failures and time limits

Successful completion requires all of the following for the exact attempt ID:

1. The simulation `ServerApp` returned normally and wrote `batch_lifecycle.json` with `COMPLETED`.
2. Its own metrics file contains every requested round, with matching run identity and round count.
3. The final saved-round marker exists and the final model checkpoint is nonempty. The marker is written after metrics and checkpoint writes finish.
4. The foreground simulation launcher exited with code zero.

An arbitrary new JSON file, the appearance of a round number in a log, or successful background dispatch is insufficient. Operational completion does not assert that every update was accepted, every metric is defined, or paper results were reproduced. Those remain experiment results to inspect.

The batch distinguishes `SUCCESS`, `CRASHED`, `INCOMPLETE`, `BOOT_TIMEOUT`, `PREPARATION_TIMEOUT`, `STALLED`, `SHUTDOWN_TIMEOUT`, `TIMEOUT`, `CANCELLED` and recovered `INTERRUPTED` attempts. Infrastructure health and unexpected container restarts are checked alongside process exit and explicit server exceptions.

| Limit | Default | Override, in seconds |
| --- | --- | --- |
| Network boot and complete TPM provisioning | 30 minutes | `--boot-timeout 1800` |
| Dataset artifact preparation and dispatch | 2 hours | `--prepare-timeout 7200` |
| No newly saved round after training dispatch | 1 hour | `--stall-timeout 3600` |
| Total attempt, starting at network boot | 7 hours | `--run-timeout 25200` |
| Launcher exit after server completion; graceful cleanup before forced termination | 2 minutes | `--shutdown-timeout 120` |
| Monitoring interval | 5 seconds | `--poll-interval 5` |

For example, allow longer rounds without changing the other limits:

```sh
python3 scripts/ops/batch_runs.py --runtime apptainer --stall-timeout 7200
```

`STALLED` means a progress deadline expired; it is a suspected deadlock, not proof. A long first round or large evaluation can trigger it. Repeated “waiting” messages do not reset the timer. Runtime control and diagnostic commands have bounded waits; polling and cleanup may add time beyond a deadline.

`round_timeout` is a separate Flower setting in `network.yaml`, defaulting to `3600.0` seconds. A queued `network.yaml` can override it for an expensive experiment. The batch records the executed value along with other configuration, but raising it does not raise the batch limits in this table. Allow the stall deadline to cover initial model evaluation, client work, regional aggregation and saved-round evaluation; it should normally exceed the Flower client-response budget. The [preflight plan](preflight.md) includes live completion, crash and stalled-run checks.

After a failed attempt, the runner captures diagnostics, stops only processes/containers identified as belonging to that attempt, archives its outputs, restores baseline configuration and proceeds to the next queued run. There are no automatic retries. A backup, configuration-restoration or cleanup failure halts the batch. If processes cannot be confirmed stopped, the active configuration remains in place for recovery instead of changing underneath them.

## Preserved evidence

Each invocation creates a separate folder under `results/batch_archives/`:

```text
<batch-id>/
  batch.json                       frozen plan, limits and attempt outcomes
  baseline_config/                 original configuration, including comments
  source/                          source/scripts, documentation, Dockerfiles and dependency manifests
  git_state.json                   revision, dirty status and tracked diff
  preexisting/                     logs/runtime/.flwr present before the batch
  attempts/<attempt-id>/
    overrides/                     supplied per-run YAML text
    executed_config/               complete configuration actually installed
    effective_config.json          merged requested runtime configuration
    config_at_exit/                active config before restoration
    status.json                    outcome, reason, elapsed time, PIDs/exit codes
    events.jsonl                   phase/progress/outcome timestamps
    boot.log
    dispatch.log
    runtime_diagnostics.json       runtime/image, disk and available GPU details
    processes.json                 processes identified as owned by this attempt
    docker-*.log                   Docker container stdout/stderr, where applicable
    logs/                          all node and system logs
    runtime/                       TPM state, ledgers, locks, Ray/Flower runtime files
    .flwr/                         project-local Flower cache, if created
    results/                       complete copy of this attempt's result directory
    artifact_manifests/             manifests for the selected dataset's artifacts
    preprocessing_report.json      full-data cleaning report, when available
    dataset_reference.json         source location and size, when available
    docker-compose.yml             generated Compose definition, when available
```

The canonical `results/<attempt-id>/` also remains available for existing analysis tools. When the server starts, it records `resolved_runtime.json` there with the actual merged YAML/Flower settings, package versions, platform and selected execution environment values. It also records lifecycle status/tracebacks and saved-round progress. Failures before server startup cannot produce server-side metadata; their planned configuration and launcher diagnostics are still retained.

Live `logs/` and `runtime/` directories are moved into the archive only after the attempt stops. This preserves their files, including TPM material and diagnostic files owned by container users. Socket entries are historical filesystem metadata, not restorable running sockets. Docker inspection/logs are collected before container removal. Apptainer batch processes inherit a unique attempt marker, allowing Linux `/proc` ownership checks even for orphaned Ray workers. Batch launchers do not execute the standalone launchers' broad process-name cleanup or log deletion.

Dataset CSVs and compiled training tensors are **not copied**. They stay in `data/`; saved artifact manifests contain provenance hashes. Output models/checkpoints are copied with results. Archives are append-only across attempts and are not automatically pruned. If archival cannot complete, the runner leaves remaining live files in place and stops before another run can overwrite them.

## Interruption and recovery

Ctrl+C or SIGTERM cancels the current attempt, collects evidence, shuts it down, restores the baseline after confirmed shutdown and leaves the remaining queue pending. A hard kill or host failure may bypass cleanup; the batch journal and existing files remain available.

Resume the saved batch explicitly:

```sh
python3 scripts/ops/batch_runs.py --resume results/batch_archives/<batch-id>
```

Resume uses its saved backend, configuration plan and time limits. It recovers any unfinished attempt first and marks that attempt `INTERRUPTED`, then runs only `PENDING` entries. Successful, failed and cancelled attempts are not retried. To repeat a failed experiment, include it in a new batch. Recovery diagnostics are kept separately from earlier diagnostics.

Source changes prevent continuing pending entries with a different implementation; recovery of the old active attempt is still performed before stopping. Start a new batch for the new source revision. A new invocation will refuse to ignore an unfinished attempt identified by the workspace journal. The old script's name-only `batch_status.log` is not used to skip runs.

Exit codes: `0` means every queued attempt succeeded; `1` means at least one attempt failed/interrupted; `2` means validation or safe continuation failed; `130` means cancellation. Consult `batch.json` and per-attempt `status.json` for details.

## Verification

Host-only supervision and shell-dispatch tests:

```sh
python3 -B -m unittest scripts.tests.test_batch_runs -v
```

With the project's ML dependencies installed, include the simulation entry-point and checkpoint integration checks:

```sh
python3 -B scripts/tests/run_tests.py --pattern 'test_batch*.py' --report scripts/tests/reports/batch_runs_tests.json
```

Tests cover independent config overlays, exact completion requirements, crashes, deadlines, log noise, cancellation, archival failure, exclusive locking and interrupted-run recovery. Actual launcher scripts execute against recording Docker/Apptainer CLI stubs to check foreground exit propagation and preservation of diagnostics. Integration tests exercise the real server-call boundary and checkpoint saver with controlled training results. These tests do not launch live containers or constitute an overnight hardware/runtime validation.
