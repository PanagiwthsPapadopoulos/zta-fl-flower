# Node-source tests: usage and evidence

This suite checks the software that runs on the Edge, Fog and Cloud before a live federation is launched. It uses small synthetic datasets, temporary directories and controlled examples so that expected outcomes can be inspected. A passing run is evidence for the assertions below, not proof of complete correctness, a fidelity percentage, attack-detection performance or published accuracy.

The policy is: explicit paper requirements first; authors' implementation where the paper is silent; disclosed correctness fixes and implementation conventions. The [architecture parity report](architecture-parity.md) records component behavior, evidence and limitations. The article's future-work proposals are outside implementation scope and are not missing requirements of this suite.

## Install and run

Run from the repository root with Python 3.11 and the project dependencies. A CPU-only environment is sufficient; no running Flower service, raw benchmark download, GPU or TPM is needed for the automated suite. The suite depends only on this project’s runtime, tests, configuration and installed Python packages; no external implementation checkout is required.

```sh
python3.11 -m venv .venv-test
. .venv-test/bin/activate
python -m pip install -e .
python -B scripts/tests/run_tests.py --report scripts/tests/reports/node_source_tests.json
```

The installer resolves the dependencies in `pyproject.toml`; they are not all pinned, so preserve the report's actual versions when citing a run. `requirements.txt` alone does not list all test dependencies (notably Captum and TorchMetrics). On hardware without a supported PyTorch dynamic INT8 CPU backend, quantization tests must fail visibly: do not silently skip them and claim quantization was verified.

`run_tests.py` uses standard `unittest` discovery, anchors paths to the repository root, selects the CPU test tier and repository YAML configuration, and disables the insecure TPM bypass. It does **not** start services or modify production configuration. Tests that require security results use explicit local mocks instead of the bypass. It exits nonzero for failures, import errors, unexpected successes, source changes during the run, or an empty test selection.

The optional JSON records tests run, failures, errors, skips, dependency versions, Python/platform information and hashes of source, tests, shell launchers, documentation, configuration, Dockerfiles and dependency manifests. It also rejects evidence if those files change during execution. No report is written without `--report`; an existing report at that path is overwritten. Store local evidence under `scripts/tests/reports/`. Raw run archives are ignored by Git. Only the selected `node_source_tests.json` summary and the historical `architecture_fidelity_probe.json` referenced by the parity report are eligible for publication; inspect them for machine-specific paths or sensitive fields before staging. A current test summary identifies the tested file hashes, including uncommitted changes; it is not evidence for a different commit. The file hashes identify the tested tree even if changes are uncommitted. A successful report with skipped tests must still be read with those omissions in mind; the current baseline expects no skips.

Useful commands:

```sh
# Full suite, verbose output, no report file:
python -B scripts/tests/run_tests.py

# Only the real two-round component flow:
python -B scripts/tests/run_tests.py --pattern test_node_flow.py

# Standard unittest also works from the repository root:
ZTA_TIER=test python -B -m unittest discover -s scripts/tests -p 'test_*.py' -v

# One test, useful when investigating a failure:
ZTA_TIER=test python -B -m unittest -v scripts.tests.test_paper_parity.PaperAggregationTests.test_negative_score_passes_filter_but_contributes_exactly_zero
```

The direct unittest commands respect the caller's `CONFIG_PATH`. Prefer the runner for consistent repository configuration. The defaults-contract test checks selected configuration defaults separately from algorithmic behavior. An intentional change to those defaults requires reviewing that test; do not change assertions just to make a run pass. Current configuration values select an experiment and do not establish architectural coverage.

## What executes, and what is replaced

**Real computation:** PyTorch training and input gradients, Adam, BatchNorm, Captum GradientSHAP, model averaging, dynamic INT8 inference, Flower tensor serialization, payload validation, data preparation, provenance checks, TrustDB transitions and metric-file generation.

**Controlled replacements:** selected tests substitute known SHAP scores or validation accuracies to isolate equations and thresholds. TPM tests replace external command responses or the verification function. In-process client proxies replace network RPC. Boot-wiring tests replace data loading and strategy construction. These substitutions are local to the tests and are described below.

The main integration test uses two Edges in two Fogs for two rounds, with two local epochs per round. It runs a benign and a label-flip client using the CNN-LSTM, actual PGD training, 8-bit uploads, real SHAP, Fog aggregation, Cloud validation/aggregation, INT8 inference, a subsequent broadcast and persisted telemetry. Quote generation/verification and RPC delivery are test doubles. Its reference model has dropout disabled and BatchNorm calibrated on the repeated synthetic training batch; both Edges reuse that controlled batch, and validation predictions are deliberately stable so the accepted-update path can be checked. Separate tests verify uniform fallback when eligible updates have effectively zero total weight, exclusion of invalid and rejected updates, forwarding to Cloud, and continued rollback enforcement. It does **not** demonstrate poisoning resistance, real signatures, TLS or network reliability. A one-client Fog is not a robust detector population.

A second two-round flow pairs the benign client with a SHAP-aware poisoner, exercising the constrained optimizer, actual transport and SHAP scoring, aggregation and persisted attack diagnostics under the same controlled integration boundaries.

The adapter boundary is explicit:

| Test path | Components actually invoked | What this does not establish |
| --- | --- | --- |
| `test_node_flow.py` | `sim_client.Client`, real shared `EdgeTrainer`, `configure_edge_fit`, `ZeroTrustGatekeeper`, `process_region` and `GlobalModelState` | Does not launch either full Cloud strategy, the deployment client, deployed Fog IPC, Flower services or sockets. |
| `test_shap_background.py` | Both `sim_server.server_fn` and `deploy_server.server_fn` sampling/wiring | Dataset and strategy constructors are replaced; this is not server startup validation. |
| `test_pipeline_integration.py` | Deployment Fog boot wiring, both Cloud adapters' evaluation/persistence, shared regional processing with simulated and deployed telemetry modes | Boot dependencies, base Flower evaluation, TPM and client responses are replaced where needed. A mode flag is not a deployed process. |
| `test_architecture_fidelity.py` | Both client adapters' evaluation delegation | The trainer is mocked for that delegation check; real inference is checked separately. |

Most filesystem outputs live in temporary directories and are removed at test completion. The suite does not rebuild the user's benchmark artifacts or overwrite experiment results. A requested evidence JSON is the only intentional lasting runner output. Python libraries may maintain normal cache directories.

## Requirement-to-test evidence map

| Requirement / basis | Tests | Expected evidence and replacement boundary |
| --- | --- | --- |
| Local runtime files override bundled project copies | `test_runtime_mounts.py`, `test_builder_imports.py` | Generated deployment/simulation Compose is parsed, application and TPM mounts are checked, and Apptainer boot/dispatch scripts run against recording CLI stubs, with socket readiness and ledger collection substituted and the global Flower profile write excluded. Import checks exercise mounted source precedence over a stale package. No live container execution. |
| Sequential simulation batches preserve evidence and distinguish completion from failure | `test_batch_runs.py`, `test_batch_integration.py` | Independent config overlays, fixed topology, identity-bound completion, saved rounds/checkpoints, crash and deadline outcomes, noisy-log stalls, cancellation, interrupted recovery, exclusive locking and archive/cleanup failure handling. Actual shell dispatch runs against recording runtime CLI stubs. Integration checks exercise the real simulation server-call boundary and saved-round marker placement after checkpoint writes. No live containers or overnight execution. See [batch usage](batch-runs.md). |
| Full benchmark CSV preparation preserves the numeric loader interface | `test_dataset_preparation.py` | Small fixtures with the complete source schemas verify exact-record deduplication before dropping fields or filling values, invalid numeric-row counts, separate Edge snippet/full taxonomies, all 15 CIC classes, headerless UNSW inputs, exclusion of its reference splits, duplicate-column checks and publication failure handling without permanent archives. UNSW numeric replacement is compared with the authors' float conversion, float32 and zero-replacement sequence, including overflow boundaries; tests check replacement counts, CLI reporting, preservation of distinct originals and strict schema/label validation. Cleaned Edge and zero-filled UNSW fixtures pass through the existing builder into 40-feature train/validation/test tensors. Further tests replace snippet artifacts when the selected input changes, reuse unchanged base tensors and regenerate deleted compiled artifacts. Downloads and full benchmark processing are not exercised. |
| one mixed clean/adversarial training pass; author training structure | `test_node_training.py`: `LocalTrainingTests` | A controlled attack tensor verifies exact BatchNorm mean/variance and one update. Separate real CNN-LSTM FGSM/PGD cases verify changed trainable weights, finite loss, singleton adversarial subsets and skipped singleton training tails. |
| seeded global attacker budget, without a per-Fog majority guarantee | `test_node_training.py`: `PlacementTests` | Exact role counts, repeatability, changed seed, regrouped Fog topology, server-owned role agreement, regional malicious majority, invalid requested/rounded budgets. No mocks. |
| train-fitted preprocessing without post-PCA scaling | `test_architecture_fidelity.py`: builder and attack-domain tests | Runs the real builder on a temporary 42-feature CSV and compares its training output with a separately fitted train-only pipeline using the same transform primitives. This checks orchestration, not an independent proof of PCA's numerical implementation. A fixed linear classifier gives analytic FGSM/PGD gradient directions and epsilon-bound expectations; additional checks cover state safety. |
| authors' attack sampling | `test_architecture_fidelity.py`: label/backdoor tests | Controlled label draws include unchanged selected labels and untouched unselected labels. Backdoor tests check row counts, trigger-only feature changes and selection of already-target rows. Epoch-wiring tests inspect fixed labels and the shared training settings. Real poisoned training also runs in `test_node_flow.py`. |
| final parameter-delta manipulation preserves trained buffers | `test_architecture_fidelity.py`: zero-scale manipulation test | With scale zero, parameters return to the reference while trained BatchNorm statistics remain changed. It does not claim unchanged predictions or exact author attack strength. |
| transport and inference are distinct mechanisms | `test_architecture_fidelity.py`: INT8 tests | Actual Flower encode/decode, rounding bound, integer-buffer preservation, smaller payload; real INT8 Linear/LSTM types, finite outputs and a stated numerical tolerance against the FP32 model on a seeded fixture; floating convolution; untouched trainable model still backpropagates. Serialized state size/positive timings are checked, not a speedup or 475KB. |
| Paper VI-A: Fog validation background, distinct from explained samples | `test_shap_background.py` | Boot adapters choose the same seeded 100 unique validation rows in both modes, keep labels aligned, preserve full Cloud validation and disclose fractional-pool behavior. A forwarding spy on real Captum verifies all 100 background rows versus 10 explained rows. A 37-row case explicitly records the shortfall rather than pretending 100 exist. |
| reference-norm stability, population filter, rewards, nonnegative weights | `test_paper_parity.py`: `PaperAggregationTests` | Controlled attributions/scores isolate the formula, raw negative scores and ties. Distinct local model states prove that, during ordinary weighted aggregation, a passing negative-score update contributes exactly zero and other weights follow score × accuracy × sqrt(sample count). Separate tests run real Captum on identical models. |
| Authors' uniform fallback at effectively zero usable weight | `test_paper_parity.py`, `test_pipeline_integration.py`: fallback tests | Distinct models verify equal averaging over eligible survivors, exclusion of invalid and rejected updates, and above-mean rewards. Shared regional processing forwards the fallback at 8/32 bits with regional weight 1, records its raw weight total and fallback flag, and reaches Cloud validation and rollback. Empty eligible sets produce no aggregate. |
| rollback at 80% or lower of previous accepted accuracy | `test_paper_parity.py`: `RollbackTests` | Mocked accuracy isolates the exact boundary and previous-versus-best reference. Real Cloud validation runs in `test_node_flow.py`. |
| authentication before model release; recovery once per round | `test_round_admission.py`, `test_security_state.py` | Enrolled poisoners can participate; invalid/missing admission fails closed; separate challenges prevent admission-quote reuse; five valid rounds recover; duplicates do not accelerate recovery or penalties. TPM/RPC results are substituted; trust and gatekeeper code execute. |
| Complete payload validation before model mutation | `test_payload_validation.py`, `test_paper_parity.py` | Invalid counts/shapes/dtypes, NaN/Inf/overflow, malformed INT8 metadata, negative variance, invalid counters and corrupt Flower blobs reject without changing the destination model. Empty regional statuses cannot conceal parameters, samples or positive trust. No successful cryptographic claim follows. |
| metrics describe the population and checkpoint actually evaluated | `test_security_metrics.py`, `test_paper_parity.py` | Separate authentication, model-update and quarantine outcomes; undefined denominators remain undefined; author-convention macro-F1 excluding classes absent from both targets and predictions while retaining prediction-only and missed classes; undefined AUC; final-singleton ASR; offline metrics attached only to their evaluated round. Synthetic inputs/logs give calculable expected results. |
| stale data and checkpoints cannot silently cross preprocessing versions | `test_artifact_provenance.py`, `test_architecture_fidelity.py` | Checksums, source/config/partition changes, missing manifests and legacy checkpoint metadata fail with actionable errors. Provenance-only tests use stand-in bytes; the builder test writes real tensors. |
| Shared Fog/Cloud wiring and observable outcomes | `test_pipeline_integration.py`, `test_node_flow.py` | Fog boot avoids Cloud test evaluation; both Cloud adapters persist defined/undefined AUC; shared regional processing runs at 8/32 bits. The malformed-update case proves payload rejection, not poisoning detection. The real-training flow covers two successive rounds and persisted per-edge inference/upload telemetry. |
| Shared attack selector and training budgets | `test_training_configuration.py` | Both real client boot functions forward shared augmentation settings for benign and label-flip roles and separate solver settings and validation data for SHAP-aware poisoning; external boot dependencies are mocked. Real training runs FGSM/PGD for both benign and label-flip roles. Zero augmentation invokes no feature attack and matches plain training on the supplied labels exactly. |
| SHAP-constrained poisoning and matched unconstrained control | `test_shap_aware_attack.py`, `test_node_flow.py` | Constrained-attack tests verify raw-distance feasibility, strict-boundary rejection, transmitted-model checks and the matched unconstrained control. Real GradientSHAP cases cover 8/16/32-bit transport, objective improvement, no-progress outcomes, frozen BatchNorm buffers and singleton batches. A separate benign/SHAP-aware two-round flow checks shared training, aggregation and persisted client-reported attack diagnostics; TPM and RPC remain test doubles. |
| Known defense limitations must remain visible | `test_paper_parity.py`, `test_architecture_fidelity.py` | Five-score versus six-score population example demonstrates small-cohort blindness. An additive output bias changes logits while raw GradientSHAP attributions stay unchanged. Passing these tests confirms limitations, not defenses against them. |

## Reading failures honestly

- `FAIL` means an asserted result differs; `ERROR` means the test could not complete, including missing dependencies. Both make the run unsuccessful.
- A regression test should fail when its claimed behavior is broken. Do not suppress errors, loosen tolerances without evidence, replace real computation with a mock, or remove a limitation test to obtain green output.
- A mock controls a dependency, not the answer to the assertion under test. Where a function is spied on, its original implementation is still called.
- Tests compare mathematical values with appropriate tolerances. They do not require cross-device bitwise agreement, fixed wall-clock timing or a universal latency advantage.
- Assertions use explicit expected behavior and mathematical invariants. They do not import another implementation as an oracle or claim that matching another implementation proves correctness.
- The runner's count refers to test methods. A method may contain several subcases; neither count measures coverage. No statement/branch coverage percentage or formal correctness proof is claimed.

PyTorch currently emits deprecation warnings for the PyTorch dynamic quantization API; QNNPACK can emit backend warnings. These remain visible. They do not establish a test failure, but a future API/backend migration needs renewed parity checks. Host-specific dependency warnings should be distinguished from assertion failures.

## Timeout configuration regression checks

`test_round_timeout.py` loads temporary YAML through the real configuration merger and checks the resulting Flower `ServerConfig` for simulation, deployed Cloud and deployed Fog servers. It checks fractional values, invalid/missing values, startup logging, saved metadata and independence from `socket_timeout`. Batch tests cover per-run overrides and execute the generated Apptainer TPM compatibility patch to verify that it routes sockets without replacing Flower's timeout constructor. These tests check configuration propagation, not real network expiry timing.

For the four live runtime/mode checks, attack cases, full-dataset scale tests and batch failure drills, follow the [preflight run plan](preflight.md).

## Optional check of your prepared dataset

`test_data_loader.py` predates this unittest suite. Its `test_consumer()` is a standalone function, so unittest discovery does **not** execute it or include it in the automated count.

After intentionally building artifacts for your current configuration, run:

```sh
python -B -m scripts.tests.test_data_loader
```

This consumes the actual configured training artifact and validates its manifest. It prints dimensions/dtypes and exits nonzero on failure. It does not train a model. Building artifacts is a separate, explicit operation:

```sh
python -B -m scripts.setup.build_artifacts
```

The builder can replace existing prepared artifacts and edge partitions; the normal automated suite does not invoke it against your data directories.

## Configuration used by the node implementation

Synthetic fixtures set their own small training budgets and datasets; passing this suite does not validate every value in the production YAML files or run the configured experiment. The defaults-contract test separately checks selected agreed defaults.

The rollback reference is fixed in the implementation: the previous accepted model. Only `security.rollback_threshold` is configurable; at `0.80`, a candidate at or below 80% of the reference accuracy rolls back. The architecture parity report distinguishes these rules from experiment selectors.

The existing `security.robustness_eval_attack` is the **single FGSM/PGD selector** for training augmentation and robustness evaluation. Training modifies model weights; robustness evaluation measures the model on perturbed inputs. Both operations share the selector.

| Key | Meaning and consumer |
| --- | --- |
| `security.robustness_eval_attack` | Both client adapters forward this to `EdgeTrainer`. Benign and label-flip training use it when augmentation is enabled; evaluation metadata preserves the same key for the offline robustness evaluator. |
| `security.benign_adv_ratio` | Shared fraction of each benign/label-flip batch receiving feature perturbations. The existing name is retained. Zero disables feature attacks in both paths; it does not disable label replacement. |
| `security.benign_eps` | Shared L-infinity perturbation budget in model-input coordinates. |
| `security.benign_alpha` | Shared PGD step size; unused by FGSM. The gradient-manipulation scale remains separate as `gradient_alpha`. |
| `security.benign_n_iter` | Shared PGD step count; unused by FGSM. |
| `training.edge_inference_int8` | Both client adapters forward this to `EdgeTrainer`. When enabled, it compares FP32 and dynamic INT8 CPU copies after fit; explicit client evaluation selects INT8 results. When disabled, automatic post-fit comparison is skipped and explicit evaluation uses FP32. Training remains floating point. |
| `training.edge_inference_samples` | Both clients construct their inference loader through `tier_edge/inference.py`: up to the configured number of clean validation rows, selected deterministically by `random_seed`. This is not the training sample count or Fog SHAP background size. |

Label flipping changes training labels; PGD/FGSM perturb input features. These are separate operations. A label-flipping node can train with or without feature augmentation, using the same training configuration as a benign node. There is no separate malicious augmentation budget and no forced PGD choice solely because labels were flipped. A zero augmentation ratio gives ordinary floating-point training on the prepared replacement labels. SHAP-aware poisoning uses a separate constrained optimizer without FGSM/PGD augmentation. The ordinary backdoor and gradient-manipulation paths retain their existing plain-training behavior.

`network.quantization_bits` selects 8-bit encoding, 16-bit conversion or 32-bit tensor transport through the shared payload/compression helpers; receivers reconstruct floating-point models. `training.clip_min` and `clip_max` support optional attack bounds. With null bounds, no global [0, 1] box is imposed on PCA coordinates; the perturbation budget still applies. `security.backdoor_poison_fraction` controls local poisoned rows, distinct from the fraction of backdoor nodes.

The Fog filter uses the population mean minus two standard deviations; there is no static SHAP threshold in YAML or in the Cloud/Fog constructor arguments. The SHAP-aware attacker uses an explicit raw-distance bound, `shap_aware_tau`, independently of the Fog’s population filter. Constrained runs require a positive bound; disabling `shap_aware_constrained` selects the matched unconstrained control. Feasibility is checked on the transmitted model and does not guarantee Fog acceptance.

`label_flip_ratio` and `p_flip` are separate, pre-existing controls. `shared/security/threat_profiler.py` uses `label_flip_ratio` to assign a rounded count of malicious **nodes**. Inside those nodes, `shared/security/adversarial_math.py` uses `p_flip` as the per-label probability of replacement, once per round and reused across local epochs. For example, 100 nodes and `label_flip_ratio: 0.2` assign 20 label-flip nodes; `p_flip: 0.2` selects roughly 20% of each such node's labels. Author-aligned replacement draws from all C classes, including the original, so the expected fraction actually changed is `p_flip * (C - 1) / C`. With 15 classes that is about 18.67%, not exactly 20%. Setting `label_flip_ratio: 0` assigns no ordinary label-flip nodes, regardless of `p_flip`.

## Design baseline and readiness

The documented core design can be held fixed while validating it: preprocessing, floating-point training, configurable update transport, separate CPU INT8 inference, paper-aligned SHAP filtering, authors' uniform fallback and macro-F1 convention, and the retained poisoning paths. The suite provides component regression evidence for that baseline, not a guarantee that deployment will reveal no defects. Bug fixes discovered during deployment require a new evidence run and a recorded revision; experimental settings should be selected and frozen before collecting thesis results.

The primary-architecture gaps listed below remain explicitly identified in the architecture parity report. They are distinct from the article's excluded future-work proposals. Full deploy startup, communication and target-device checks are still required before claiming a validated deployed implementation. This distinction avoids turning a design freeze into an unsupported claim of final correctness.

## Recorded verification

The [node-source evidence JSON](../scripts/tests/reports/node_source_tests.json) records a particular test run, including test counts, failures/errors/skips, dependency versions and file hashes. It describes that recorded snapshot; it does not automatically validate a later checkout. Compare its hashes with the files being evaluated, or rerun the documented command to generate current evidence. A documentation change to this guide also changes the suite's source fingerprint.

The suite covers shared attack selection, both client adapters, zero-augmentation label-flip training, constrained and matched unconstrained SHAP-aware poisoning, real mixed-batch training and a synthetic two-round flow. Fallback tests exercise zero and near-zero weights, eligibility, rewards and Cloud rollback. Macro-F1 tests distinguish absent classes from prediction-only and missed classes. These checks establish component behavior, not paper-scale accuracy or timing.

## What remains outside this evidence

Real TPM command execution and provisioning, TLS and deployed IPC/RPC, process crashes/restarts, concurrent GPU workers, CUDA/MPS numerical behavior, full-dataset training, five-seed statistical results, realistic attack effectiveness and target-device performance require separate validation. Missing token-envelope encryption, signed firmware/PCR-update manifests, live perception and protocol feature skew are documented gaps, not functionality supplied by this suite.
