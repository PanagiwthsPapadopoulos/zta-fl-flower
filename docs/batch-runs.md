# Sequential simulation batches

The batch runner boots one simulation environment and one Flower coordinator, then submits experiments one at a time. It uses Flower's run status to decide when to submit the next bundle, plus boot, preparation, stall, run and stop timeouts.

There are no batch process inventories, PID-name checks, TPM health gates, source-fingerprint gates, workspace locks, stale-session gates or checkpoint-completeness gates. TPM attestation, SHAP, trust filtering and the training application's security policies remain enabled. The paper queue still contains all 100 clients and uses the 180-second TPM freshness window.

## Run

From the repository root, using Python with the existing PyYAML dependency available:

```sh
python3 scripts/ops/batch_runs.py \
  --runtime apptainer \
  --queue experiments_archived/paper_80 \
  --prepare-timeout 43200 \
  --stall-timeout 10800 \
  --run-timeout 165600
```

Use `--runtime docker` for the Docker simulation. On HPC, use the existing `.batch-python` copy if host Python lacks PyYAML: prefix the command with `PYTHONPATH="$PWD/.batch-python:$PWD"`. No new packages are required. `--dry-run` prints the merged plan without launching anything; it is not a validation suite.

Start one batch in the workspace with the previous simulation stopped. The runner no longer detects competing supervisors or recovers old live processes. Keep the queue's topology consistent with `config/network.yaml`, which boots the TPM environment.

## Configuration and sequence

Experiment directories are read in sorted name order. Each starts from the same snapshot of `config/network.yaml`, `config/training.yaml` and `config/security.yaml`; corresponding files in that experiment directory override the baseline. Partial and complete YAML files both work. Extra files are ignored. Missing values never inherit from the previous experiment. The runner does not enforce a configuration schema.

The frozen plan is stored in `results/batch_archives/<session>/batch.json`. Each experiment gets a unique `run_name` and `batch_attempt`. Its resolved YAML files go to `runtime/batch/config/` and its attempt archive. `CONFIG_PATH` points artifact preparation and training at those files. Workspace `config/` files are not rewritten.

The worker prepares the dataset, submits a Flower application bundle, and polls its run ID. `finished:completed` is recorded as `SUCCESS`; other finished outcomes as `INCOMPLETE`. Either allows the next experiment. `SUCCESS` means Flower reported completion; the batch runner does not audit rounds or checkpoints. CPU/GPU scheduling settings come from the local federation in `pyproject.toml`; all configured clients are passed to Flower.

A preparation failure is recorded and the queue proceeds. An ambiguous submission error stops the worker rather than resubmitting a potentially accepted bundle. Transient status-query failures are retried within the existing run/stall timeouts.

## Timeouts and stopping

- `--boot-timeout`: maximum wait for network provisioning, and separately for Flower's control API (default 1800 seconds).
- `--prepare-timeout`: maximum dataset preparation time (7200).
- `--stall-timeout`: maximum time without a newly saved round during training (3600).
- `--run-timeout`: total experiment duration, including preparation (25200).
- `--shutdown-timeout`: wait for Flower to report a stopped run, and process termination allowance (120).
- `--poll-interval`: status polling interval (5).

On a run/stall timeout or cancellation, the worker requests `flwr stop` and waits for Flower's finished status. If that stop times out, the queue ends without submitting the next bundle. This is the handover condition, not a process inspection. Training's YAML round/RPC timeouts remain separate.

Ctrl+C requests cancellation. Final shutdown signals the process groups launched by the runner; Docker also tears down its Compose environment. There is no independent audit of detached processes. Do not start a second invocation while the first is stopping.

## Logs and resume

The terminal prints the batch archive and worker-log path. The worker log records experiment starts and outcomes. `results/batch_archives/active_batch.json` points to the latest session.

- `<session>/infra_<id>/boot.log`: network provisioning.
- `<session>/infra_<id>/worker.log`: queue execution and errors.
- `<session>/infra_<id>/superlink-console.log`: Flower coordinator output.
- `<session>/attempts/<attempt>/`: resolved configuration, preparation log, Flower submission/status, training log and available Ray logs.
- `results/<attempt>/`: training metrics and checkpoints.
- `logs/system/run_cloud.log`: current training stream.

Private Flower/Ray temporary files live under `/tmp/zta-batch-*` inside the runtime. Ray logs are archived after Flower reports a finished run; temporary files are removed during normal cleanup. Archival errors are reported without blocking the next bundle. Runtime/TPM files remain in the workspace and are reused; they are not moved into a full infrastructure snapshot. Hard termination can leave temporary files.

To resume pending entries after the previous invocation has stopped:

```sh
python3 scripts/ops/batch_runs.py --resume results/batch_archives/<session>
```

Add `--retry-interrupted` to restart interrupted/cancelled experiments from round 1. Previously successful or failed entries are not automatically rerun. Resume uses the saved configurations and timeouts with the current code; it performs no fingerprint comparison or old-process recovery.

## Development verification

```sh
python3 -B -m unittest scripts.tests.test_batch_runs scripts.tests.test_batch_worker scripts.tests.test_batch_integration -v
```

The tests cover configuration inheritance, bundle sequencing, timeout/stop handover and shell dispatch. The integration tests need the existing ML dependencies. An optional small real Flower test is available with `ZTA_FLOWER_SMOKE=1 python3 -B -m unittest scripts.tests.test_batch_worker.RealFlowerSmoke -v`. Local tests do not establish full HPC completion.

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
