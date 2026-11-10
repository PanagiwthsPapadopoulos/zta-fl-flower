# Preflight runs before long experiments

Run these stages in order. They qualify the implemented ZTA-FL path on the tested runtime, hardware and data; they cannot guarantee every future run or reproduce the paper's comparison tables by themselves. Baseline algorithms and ablation switches must be wired and validated separately before collecting those comparisons. Future-work proposals are outside this plan.

Run the four basic runtime/mode checks once. Then run the attack and scale matrix on the simulation runtime and hardware intended for the long experiments. Passing Docker on a laptop does not qualify Apptainer on an HPC node. Run one experiment at a time; do not combine different attack roles in the initial tests.

## 1. Source regression tests

From the repository root, with the project's dependencies installed:

```sh
python3 -B scripts/tests/run_tests.py --report scripts/tests/reports/preflight_source_tests.json
```

Pass: exit code zero, `successful: true`, no failures/errors/unexpected skips and `source_unchanged_during_run: true`. This suite checks components and controlled integration; it does not run a live federation.

## 2. Four live smoke tests

Back up the existing configuration before editing it. Set the following keys in their indicated files; retain the existing network addresses and port assignments. These are proposed test configurations, not changes to repository defaults.

`config/network.yaml`:

```yaml
num_fogs: 2
uniform_edges_per_fog: 3
custom_fog_topology: "[]"
round_timeout: 3600.0
socket_timeout: 3600.0
quantization_bits: 8
```

`config/training.yaml`:

```yaml
run_name: preflight_docker_deploy
num_rounds: 2
snapshot_rounds: [0, 1, 2]
snapshot_interval: 1000
gpu_enabled_tiers: []
max_concurrent_gpus: 1
dataset: edge_iiotset
dataset_path: data/edge_iiotset/raw/network_traffic_samples.csv
dataset_fraction: 1.0
apply_smote: true
test_split: 0.15
val_split: 0.15
model_architecture: cnnlstm
learning_rate: 0.001
batch_size: 64
local_epochs: 1
random_seed: 42
n_classes_per: 3
power_law_a: 0.4
clip_min: null
clip_max: null
clip_norm: 1.0
edge_inference_int8: true
edge_inference_samples: 256
```

`config/security.yaml`:

```yaml
simulate_global_leakage: false
shap_aware_base_attack: label_flip
shap_aware_constrained: true
shap_aware_tau: null
shap_aware_max_backtracks: 10
rollback_threshold: 0.80
tpm_freshness_window: 120
label_flip_ratio: 0.0
grad_manip_ratio: 0.0
backdoor_ratio: 0.0
shap_aware_ratio: 0.0
p_flip: 0.2
gradient_alpha: 0.0
backdoor_poison_fraction: 0.10
backdoor_target_class: 0
backdoor_trigger_features: "[-3, -2, -1]"
backdoor_trigger_value: 1.5
shap_val_samples: 100
shap_explain_count: 10
benign_adv_ratio: 0.0
benign_eps: 0.1
benign_alpha: 0.025
benign_n_iter: 7
robustness_eval_attack: fgsm
```

Use the setup prerequisites in the [deployment](deployment.md) and [simulation](simulation.md) guides. Keep security verification enabled; do not pass `--insecure`. For each row below, set `run_name` to the identifier, start the boot command in terminal A, wait for readiness, then run the dispatch command in terminal B. In terminal C, run `python3 scripts/ops/verify_pipeline.py --live`.

| Run identifier | Terminal A | Terminal B |
| --- | --- | --- |
| `preflight_docker_deploy` | `bash scripts/deployment/boot_network_docker.sh` | `bash scripts/deployment/deploy_code_docker.sh` |
| `preflight_docker_sim` | `bash scripts/simulation/boot_network_sim_docker.sh` | `bash scripts/simulation/deploy_code_sim_docker.sh` |
| `preflight_apptainer_deploy` | `bash scripts/deployment/boot_network_apptainer.sh` | `bash scripts/deployment/deploy_code_apptainer.sh` |
| `preflight_apptainer_sim` | `bash scripts/simulation/boot_network_sim_apptainer.sh` | `bash scripts/simulation/deploy_code_sim_apptainer.sh` |

Stop terminal A with Ctrl+C after the experiment has finished and you have captured any needed container stdout. Archive diagnostics before booting the next test: standalone launchers clear old logs. For example, set `RUN` to the just-finished identifier and use a new evidence directory:

```sh
RUN=preflight_docker_deploy
EVIDENCE="results/preflight/${RUN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVIDENCE"
cp -R config "$EVIDENCE/config"
cp -R logs "$EVIDENCE/logs"
cp -R runtime "$EVIDENCE/runtime"
cp -R "results/$RUN" "$EVIDENCE/results"
test ! -d .flwr || cp -R .flwr "$EVIDENCE/flower_cache"
```

Pass criteria for every ordinary live test in this guide:

- The requested rounds all exist in `results/<run>/<run>.json`; all Fog round records exist; the final checkpoint loads. Dispatch returning successfully is insufficient, since standalone dispatch can return before training finishes.
- The server startup log and `metadata.round_timeout` both show the requested value. For deployment, check Cloud and both Fog startup logs. Both modes must use the local checkout.
- Every expected Edge is accounted for in admission/attestation records. There are no unexplained missing responses, client training crashes, invalid model payloads, expired legitimate quotes, TPM bypasses, container restarts or memory kills. In these tests the poisoning nodes still have valid identities: a poisoned update may be rejected after authentication.
- Clean runs produce finite loss, accuracy and macro-F1, finite model parameters, and at least one accepted global model change compared with round 0. Repeated rollback, `no_positive_trust`, `no_results` or regional `error` requires investigation even if the run finishes. Individual SHAP rejections are not automatically bugs.
- CPU INT8 inference telemetry exists when enabled. Undefined AUC/FAR is acceptable only with its documented missing-class/zero-denominator explanation. A lower attack success rate or higher accuracy is a result, not a predetermined software-test requirement.

The shipped snippet only qualifies startup and the live path. Its class coverage is insufficient to qualify the full datasets.

## 3. Attack paths on real data

Use the same base configuration as stage 2, changing only these common settings:

```yaml
# network.yaml
num_fogs: 2
uniform_edges_per_fog: 10
# training.yaml
dataset_path: data/edge_iiotset/full_dataset/network_traffic_samples.csv
num_rounds: 3
batch_size: 128
local_epochs: 1
snapshot_rounds: [0, 1, 2, 3]
```

Each row starts from that clean baseline. Apply only the listed security overrides. With 20 Edges, ratios 0.1, 0.2 and 0.3 produce exactly 2, 4 and 6 attackers respectively. Verify those counts in `metadata.experiment_ground_truth`; assignment within individual Fogs can vary.

| Queue/run name | Overrides in `security.yaml` |
| --- | --- |
| `01_clean` | None: all four attack ratios and `benign_adv_ratio` remain zero |
| `02_label_low` | `label_flip_ratio: 0.1`, `p_flip: 0.2` |
| `03_label_high` | `label_flip_ratio: 0.3`, `p_flip: 1.0` |
| `04_gradient_negative` | `grad_manip_ratio: 0.2`, `gradient_alpha: -3.0` |
| `05_gradient_positive` | `grad_manip_ratio: 0.2`, `gradient_alpha: 5.0` |
| `06_backdoor` | `backdoor_ratio: 0.2` (the base poisoning fraction is 0.10) |
| `07_fgsm` | `benign_adv_ratio: 0.3`, `robustness_eval_attack: fgsm`, `benign_eps: 0.1` |
| `08_pgd7` | `benign_adv_ratio: 0.3`, `robustness_eval_attack: pgd`, `benign_eps: 0.1`, `benign_alpha: 0.025`, `benign_n_iter: 7` |
| `09_pgd20` | `benign_adv_ratio: 0.3`, `robustness_eval_attack: pgd`, `benign_eps: 0.2`, `benign_alpha: 0.025`, `benign_n_iter: 20` |
| `10_shap_label_control` | `shap_aware_ratio: 0.2`, `shap_aware_base_attack: label_flip`, `p_flip: 0.5`, `shap_aware_constrained: false`, `shap_aware_tau: 0.05` |
| `11_shap_label_constrained` | Same as row 10 with `shap_aware_constrained: true` |
| `12_shap_gradient_control` | `shap_aware_ratio: 0.2`, `shap_aware_base_attack: gradient_manipulation`, `shap_aware_constrained: false`, `shap_aware_tau: 0.05` |
| `13_shap_gradient_constrained` | Same as row 12 with `shap_aware_constrained: true` |

`shap_aware_tau: 0.05` and the PGD step size above are explicit preflight choices; the article does not prescribe these exact values. They are not established reproduction settings. The two SHAP objectives must each be checked against their matched control. For every constrained update, inspect `performance_metrics/fog_<i>_state.json` → `rounds` → `node_shap_attack_metrics`: the raw distance must be below tau and `shap_attack_feasible` must be true. An all-`no_progress` attack has exercised the constraint code but does not qualify an effective adaptive-attack experiment. Investigate before freezing its experimental bound. Feasibility does not guarantee Fog acceptance.

For automation, create one subdirectory per row under `experiments_queue/preflight-attacks/`. Copy the complete stage-3 baseline YAMLs into each, then apply that row's overrides. The workspace baseline must also have the same 2×10 topology. The batch runner substitutes a unique `run_name` and archives the requested overrides, executed configuration and results. Never copy one attack case to the next without resetting the other ratios.

Validate, then launch on the selected runtime (substitute `apptainer` for `docker` on that host):

```sh
python3 scripts/ops/batch_runs.py --runtime docker --queue experiments_queue/preflight-attacks --dry-run
python3 scripts/ops/batch_runs.py --runtime docker --queue experiments_queue/preflight-attacks \
  --boot-timeout 1800 --prepare-timeout 7200 --stall-timeout 7200 \
  --run-timeout 86400 --shutdown-timeout 120 --poll-interval 5
```

Pass: every ordinary attempt is `SUCCESS` and meets stage 2's data/telemetry checks. Examine the attack-specific telemetry and compare the saved models and metrics with the clean run. Accuracy need not improve monotonically; do not require a particular paper percentage after three rounds.

Run the offline security evaluator on the backdoor and FGSM/PGD runs as well. It uses each run's saved attack settings, not arbitrary new settings. For a batch attempt, select its exact ID from `batch.json` and bind its archived `executed_config` so dataset-path fallback cannot pick a different currently selected dataset:

```sh
# Replace these two values with the recorded IDs/paths.
RUN='<attempt-id>'
EXECUTED_CONFIG="$PWD/results/batch_archives/<batch-id>/attempts/$RUN/executed_config"
docker run --rm --entrypoint /python/venv/bin/python3 --workdir /app --env PYTHONPATH=/app \
  --mount "type=bind,source=$PWD,target=/app" \
  --mount "type=bind,source=$EXECUTED_CONFIG,target=/app/config,readonly" \
  panagiotispapadopoulos/zta-cloud-node:latest scripts/ops/evaluate_security.py "$RUN"

# Alternative on the Apptainer host:
apptainer exec --bind "$PWD:/app" --bind "$EXECUTED_CONFIG:/app/config:ro" \
  --pwd /app --env PYTHONPATH=/app images/cloud_node.sif \
  /python/venv/bin/python3 scripts/ops/evaluate_security.py "$RUN"
```

Pass: `heavy_metrics.json` is generated without a feature-space/checkpoint mismatch. ASR and robustness measurements are finite when their evaluated populations are nonempty; authentication errors and missing responses are inspected separately from model-update rejections. Copy this later-generated file into the attempt's archived `results/` as well; the completed batch archive is not updated automatically. If a later test has replaced artifacts for that dataset with a different seed or split, rebuild them using the saved executed configuration before offline evaluation.

## 4. Full topology and full datasets

Create a separate queue with the workspace and every queued configuration set to `num_fogs: 10`, `uniform_edges_per_fog: 10`, `custom_fog_topology: "[]"`. Topology cannot change within one batch.

Use `num_rounds: 3`, `local_epochs: 5`, `batch_size: 128`, `learning_rate: 0.001`, `dataset_fraction: 1.0`, `random_seed: 42`, and `snapshot_rounds: [0, 1, 2, 3]`. Keep the other stage-3 settings, including the 100-row Fog background and 10 explained samples. Repeat the following three cases for **each** dataset: clean, PGD-20 (row 09), and constrained SHAP label attack (row 11). That is nine runs. Include row 13 at this scale too if the gradient-objective SHAP attack will appear in the actual campaign.

| Dataset | `dataset` | `dataset_path` |
| --- | --- | --- |
| Edge-IIoTset | `edge_iiotset` | `data/edge_iiotset/full_dataset/network_traffic_samples.csv` |
| CIC-IDS2017 | `cic_ids2017` | `data/cic_ids2017/full_dataset/network_flows.csv` |
| UNSW-NB15 | `unsw_nb15` | `data/unsw_nb15/full_dataset/intrusion_records.csv` |

Use the hardware policy intended for the real campaign. Exact starting profiles are CPU: `gpu_enabled_tiers: []`, `max_concurrent_gpus: 1`; or CUDA: `gpu_enabled_tiers: [cloud, fog, edge]`, `max_concurrent_gpus: 2`. GPU whitelisting does not itself allocate GPU resources to Ray actors. For **Apptainer GPU simulation**, enable this existing section in `pyproject.toml` **before** freezing the batch:

```toml
[tool.flwr.federations.local]
options.num-supernodes = 100
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.5
```

The Apptainer launcher reads these resource values and overrides the node count from topology. The fractional GPU request lets Ray schedule two client workers per visible GPU; the existing lock policy remains separate. Confirm CUDA is actually visible inside the runtime and used in the node logs. Do not interpret a CPU fallback as a GPU qualification. CPU tests can explicitly set `num-gpus = 0.0`. Do not change the manifest or hardware policy while a batch runs.

Both batch backends now read the local federation resource settings from `pyproject.toml`. The standalone Docker launcher still derives its GPU request from the Edge tier's device policy. Confirm the actual device in node logs. GPU lock settings are deployment-only and do not limit simulation actors.

Run this queue with the stage-3 batch command, replacing its path with `experiments_queue/preflight-scale`. The launchers automatically build/reuse artifacts; they must not silently use snippet artifacts for a full-data path. Before accepting each dataset, inspect its archived artifact manifest: source provenance must match the full CSV, validation/test remain unresampled, all 15 Edge/CIC classes or 10 UNSW classes must have observations in the base splits, and the 100 Edge partitions must exist. Each partition intentionally contains a subset of classes. The source-test suite already checks transformation and split semantics on fixtures; these live runs qualify the actual artifacts and model interface.

Additional pass criteria: all 100 Edges are accounted for in round 1; no region errors, resource exhaustion or legitimate quote expiry; clean training updates the global model; expensive paths actually execute. Measure the longest interval between saved rounds, peak memory and disk growth from these runs. The one-hour Flower and IPC budgets and two-hour stall deadline are initial test limits, not predictions about the machine's speed. If healthy work exceeds them, record the measured duration, change the relevant budget explicitly and repeat that test. Record the selected TPM freshness window with the results; the paper queue now uses an explicit 180-second HPC latency allowance.

This optional smoke protocol uses 120 seconds; the paper queue uses the separately requested 180-second HPC allowance. The paper reports 60 seconds. If the campaign will claim that setting, repeat the scale cases with `tpm_freshness_window: 60` and require no legitimate quote-expiry failures. A passing test at 120 seconds does not establish readiness at 60 seconds.

## 5. Live batch recovery drills

These are optional destructive tests for a disposable smoke queue, not commands for an active paper campaign. Use two small entries and retain all security checks.

First verify that two successful experiments have different Flower run IDs and fresh model/trust state, while the coordinator PID and TPM identities remain unchanged. Confirm that the first run's training workers have exited before the second configuration is installed.

For a failure drill, stop only the first Flower run using its recorded `flower_run_id` and the private connection recorded in `infra_<id>/flower_runtime.json`. Do not kill the cloud container or signal the batch infrastructure marker: that now identifies the shared environment for the whole queue. The failed run should retain its actual Flower failure reason; after its workers exit, the second run should start on the same coordinator.

For cancellation/recovery, interrupt the host batch command with Ctrl+C. Expect the active run to become `CANCELLED`, the second to remain `PENDING`, and one final infrastructure shutdown. Resume with `--retry-interrupted`; expect a new environment and new attempt ID, with earlier evidence preserved. See [batch recovery](batch-runs.md#interruption-and-resume).

An unconfirmed process exit or failed archive must block the next configuration. Killing the entire cloud environment tests infrastructure failure and is expected to halt the queue, rather than automatically rebuild it between experiments.

## 6. Long-run qualification and campaign start

After the preceding gates pass, run a 100-round clean simulation on the production host with the full Edge-IIoTset dataset, 10×10 topology, five local epochs, batch 128 and seed 42; retain the qualified hardware/timeout/freshness settings. Use `snapshot_rounds: [0, 1, 10, 25, 50, 75, 100]`, `snapshot_interval: 1000`. Use the batch runner to retain the complete evidence.

Size the total batch deadline from the measurements, rather than retaining the seven-hour default: allow at least boot/preparation time plus 100 times the longest observed round interval, with a margin. For an initial generous cap use `--run-timeout 604800` (seven days), retaining `--stall-timeout 7200` only if the scale runs demonstrated enough headroom. A scheduler allocation must cover the same period. These deadlines do not force training to take that long; completion advances the batch immediately.

Pass: all 100 rounds finish with normal shutdown, usable final/checkpoint outputs, no growing resource leak, no unexplained security/communication failures, and complete diagnostics. This first run can count as a campaign run only if its scientific settings and preprocessing were frozen before it started.

The [paper's experimental setup](https://arxiv.org/html/2512.23809v1) specifies 100 rounds, five local epochs, batch 128, learning rate 0.001 and seeds 42, 123, 456, 789 and 1011. After qualification, expand to the intended dataset/attack/seed grid. Three-round preflights validate execution, not convergence, statistical confidence or equality with reported accuracy. Any new model, preprocessing revision, GPU policy, runtime, transport precision or previously untested attack path needs its corresponding qualification repeated before committing the whole campaign.
