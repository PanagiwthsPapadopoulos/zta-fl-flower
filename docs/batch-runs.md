# Sequential simulation batches

[batch_runs.py](../scripts/ops/batch_runs.py) boots the simulation infrastructure **once**, runs the queue one experiment at a time, and shuts down once at the end. It supports Docker and Apptainer simulation, not deployment.

## Start the 80-experiment queue

From the repository root on the HPC, with the existing images and datasets:

```sh
PYTHONPATH="$PWD/.batch-python:$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/ops/batch_runs.py \
  --runtime apptainer \
  --queue experiments_archived/paper_80 \
  --prepare-timeout 43200 \
  --stall-timeout 7200 \
  --run-timeout 165600
```

The `.batch-python` path supports the previously copied PyYAML package on hosts without a virtual environment. It is harmless if that directory is absent and PyYAML is already installed. No packages are downloaded by the runner. The host needs Python 3.9+ and PyYAML; the existing image supplies Python 3.11+, Flower 1.30, Ray and ML dependencies. Apptainer supervision requires Linux pidfd support.

Add `--dry-run` to validate and print the complete plan without starting containers or changing files. Replace `apptainer` with `docker` for Docker. Keep the HPC allocation and terminal session alive for the batch's duration. Time limits apply per experiment, not to the entire queue.

**Start only this command.** The runner starts the TPM services and one persistent cloud environment automatically. It does not adopt an independently started federation; stop an existing standalone simulation first. Existing images are reused. Repository changes are mounted into the images, so this update does not require rebuilding them.

## What remains alive

```text
Boot all TPM services once
Start one cloud environment and private Flower SuperLink/SuperExec
  Experiment 1: prepare data → submit run → wait → save results → drain run workers
  Experiment 2: prepare data → submit run → wait → save results → drain run workers
  ...
Stop the coordinator and infrastructure once
```

Apptainer executes the queue worker in one long-lived `apptainer exec`; Docker uses the existing `sim-cloud` container. Each experiment is a separate Flower run with a new simulation process, Ray workers, model, optimizer, TrustDB and quarantine state. The same TPM services and identities are reused. Nonces, signature/PCR verification, SHAP filtering and trust policy remain enabled. The paper queue retains 100 clients, 100 rounds and the requested **180-second** TPM freshness window.

Flower's database and Ray working files live in a private `/tmp/zta-batch-*` directory inside the runtime, rather than the network home directory. After an experiment's workers exit, its Ray logs are archived and that Ray working directory is removed. The coordinator database is retained until the queue ends, then archived. The temporary directory is removed after confirmed shutdown; its location is recorded in `flower_runtime.json` if recovery is needed. A hard kill can leave it behind.

This removes repeated infrastructure teardown between experiments. It does not guarantee that the previously observed training interruption cannot recur.

## Queue and configuration

The configuration contract is unchanged:

```text
experiments_queue/
  01_clean/
    training.yaml
    security.yaml
  02_attack/
    training.yaml
    security.yaml
```

- Directory names determine order and must contain only letters, digits, underscores or hyphens.
- Every experiment starts from the **same baseline snapshot** of `config/network.yaml`, `config/training.yaml` and `config/security.yaml`.
- Queue files override only their corresponding baseline file. Partial overrides and complete files both work. Omitted values inherit the baseline, never the previous experiment.
- Unknown keys, misplaced keys, invalid round counts/timeouts and topology changes are rejected before boot. All experiments must have the baseline's effective Fog/Edge topology.
- Each experiment receives a unique execution `run_name` and matching `batch_attempt`; the queued name remains its human-readable label.
- The queue is frozen at batch creation. Editing the queue does not change an existing saved batch.

For example, an experiment's `training.yaml` can contain:

```yaml
dataset: unsw_nb15
dataset_path: data/unsw_nb15/full_dataset/intrusion_records.csv
random_seed: 42
num_rounds: 100
```

A clean experiment must explicitly disable attack roles that are enabled in its baseline.

The worker writes each resolved configuration to `runtime/batch/config/` and points the existing configuration loader there through `CONFIG_PATH`. **The workspace's `config/` files are not rewritten during training.** Both artifact preparation and the fresh Flower application use the same resolved configuration. Dataset artifacts are prepared/reused for every experiment using their existing provenance checks, including dataset, seed and partition changes.

Both batch backends read simulation worker CPU/GPU settings from the existing local federation section of `pyproject.toml`, and set the number of clients from the topology. `max_concurrent_gpus` remains a deployment setting; it does not control simulation scheduling. Standalone launchers retain their existing behavior. Do not edit source, hardware policy or the active configuration while a batch runs.

## Progress and output

The terminal prints the session directory and each experiment's phase/saved round. The persistent worker's log also records starts and final outcomes. The workspace pointer `results/batch_archives/active_batch.json` identifies the session; its `batch.json` holds the frozen plan and latest statuses.

```text
results/batch_archives/<session>/
  batch.json
  baseline_config/
  source/
  git_state.json
  preexisting/                         files from before this batch
  infra_<id>/                         one per invocation, including resume
    boot.log
    worker.log
    flower_runtime.json               coordinator PID, address, temporary path
    superlink-console.log
    flower-state.db                   after coordinator shutdown
    logs/ and runtime/                moved here only at final infrastructure shutdown
  attempts/<attempt>/
    executed_config/
    effective_config.json
    artifact_manifests/
    dataset_reference.json
    status.json
    events.jsonl
    prepare.log
    flower_submission.json
    flower_status.json                includes Flower's actual failure details
    run_cloud.log
    logs/                             that experiment's node JSONL files
    ray/                              that experiment's Ray diagnostic logs
results/<attempt>/                    metrics, model, lifecycle and progress markers
```

Models and results stay in their existing canonical `results/<attempt>/` directory; the runner does not duplicate large model files into the session archive. Raw datasets and training tensors remain in `data/`. During training, the current Flower stream is `logs/system/run_cloud.log`. Node logs rotate only after all experiment workers exit. Infrastructure logs remain open throughout the queue.

The internal [batch_worker.py](../scripts/ops/batch_worker.py) is launched by the container launcher; do not invoke it separately on the host.

## Completion and failures

Success requires Flower to report `finished:completed`, the matching application lifecycle to say `COMPLETED`, all requested metric rounds and the final saved-round marker, and a nonempty final model checkpoint. A zero CLI exit status alone is insufficient.

After a failed or timed-out run, the worker requests that run's stop, waits for its training processes to exit and, if necessary, terminates its own remaining run descendants. It preserves the persistent coordinator, executor and TPM services. Only after this succeeds does it archive the run's logs and apply the next experiment's configuration. A failed experiment remains recorded as failed; it is not silently retried or counted as a trained model.

An unconfirmed worker shutdown, unavailable coordinator, unreadable owned process, or archival failure stops the queue. This prevents a live old experiment from reading the next experiment's configuration. The host performs infrastructure teardown only once when the worker finishes, fails or is cancelled. Final Apptainer process inspection retries transient read failures within the existing shutdown allowance, but persistent uncertainty still blocks a new batch.

| Limit | Default | Option |
| --- | --- | --- |
| One-time network boot | 30 minutes | `--boot-timeout 1800` |
| Dataset preparation per experiment | 2 hours | `--prepare-timeout 7200` |
| No newly saved round | 1 hour | `--stall-timeout 3600` |
| Total experiment, including preparation | 7 hours | `--run-timeout 25200` |
| Each graceful/forced shutdown phase | 2 minutes | `--shutdown-timeout 120` |
| Progress polling | 5 seconds | `--poll-interval 5` |

Bounded Flower control calls and shutdown can add time beyond a deadline. The separate YAML `round_timeout` controls Flower's response budget. Neither that setting nor batch deadlines change TPM freshness. Long evaluations count toward the no-progress deadline.

## Interruption and resume

Ctrl+C requests cancellation, drains the active training processes and shuts down the infrastructure. Remaining experiments stay pending. Resume on the same execution host/workspace:

```sh
PYTHONPATH="$PWD/.batch-python:$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/ops/batch_runs.py \
  --resume results/batch_archives/<session> --retry-interrupted
```

Resume first recovers any owned infrastructure left by a terminated supervisor. It then boots one new environment for the remaining queue. `--retry-interrupted` restarts interrupted/cancelled experiments from round 1 under new attempt IDs, preserving their history. Without it (or with `--skip-interrupted`), only pending experiments run. Successful experiments are never repeated automatically. Other failed experiments require a new queue/batch if you want to repeat them.

Resume uses the **saved configuration and time limits**, not current queue edits. Source changes block continuation after recovery; start a new batch for new code or new settings. The workspace lock prevents two supervisors from running concurrently.

Old version-1 batches can be passed to `--resume` for cleanup only. After confirmed cleanup, the command reports that legacy recovery is complete and exits 2; run the normal new-batch command to use this implementation and the current 180-second queue settings. It does not silently translate an old batch into a new one. If old ownership evidence cannot establish shutdown, recovery remains blocked.

Exit codes: `0` all experiments succeeded; `1` the queue completed with failures; `2` validation/recovery or safe continuation failed; `130` cancellation. Check both experiment outcomes and the final infrastructure status.

## Verification

```sh
python3 -B -m unittest scripts.tests.test_batch_runs scripts.tests.test_apptainer_lifecycle -v
```

In an environment with the existing Flower/Ray/ML dependencies, also run:

```sh
python3 -B -m unittest scripts.tests.test_batch_worker scripts.tests.test_batch_integration -v
```

An optional local process test runs two tiny two-client Flower simulations on the same coordinator (requires localhost access and the installed runtime dependencies):

```sh
ZTA_FLOWER_SMOKE=1 python3 -B -m unittest scripts.tests.test_batch_worker.RealFlowerSmoke -v
```

These tests cover independent YAML overlays, exact completion, one coordinator per queue, configuration handover, cancellation, failure continuation, shutdown blocking, recovery and shell dispatch. They do not establish that 80 full HPC experiments will finish successfully.

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
| Attestation / rollback | Freshness **180 seconds** (HPC latency allowance; paper: 60); rollback ratio 0.80 | VI.A and V.B |
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
