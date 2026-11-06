# Metrics Inventory and Comparison with the ZTA-FL Article

## 1. Purpose and Scope

This document inventories the metrics and diagnostic records produced by the implementation and compares them with the measurements discussed in *Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems*.

The comparison works in both directions:

- Identify each article metric and whether the implementation calculates it directly, supports deriving it from recorded data, or has an unresolved coverage gap.
- Identify implementation measurements that provide additional visibility beyond the article, including quantized inference, authentication outcomes, quarantine, and aggregation diagnostics.

This reference describes metric definitions, output locations, comparison requirements, and instrumentation gaps. Metric availability is independent of numerical performance: a development run need not exercise every attack or reproduce the article's results to demonstrate collector coverage.

**Different experimental conditions do not automatically require different metric implementations.** Label-flipping accuracy, gradient-manipulation accuracy, clean accuracy during backdoor poisoning, and accuracy under collusion are applications of the same classification-accuracy metric. Their experiment configurations differ.

### 1.1 Reference and implementation map

Article references use [*Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems*, arXiv:2512.23809v1](https://arxiv.org/html/2512.23809v1), especially Sections V–IX, Tables II–VII, and Figures 3–6.

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Global evaluation | `src/tier_cloud/global_evaluator.py`; `src/shared/utils/metrics.py` | Evaluation loss, accuracy, macro-F1, macro One-vs-Rest ROC-AUC, and SHAP stability. |
| Global results and checkpoints | `src/tier_cloud/cloud_aggregator_sim.py`; `src/tier_cloud/cloud_aggregator_deploy.py` | Round histories, global model payload size, round timing, and checkpoint persistence. |
| Fog telemetry | `src/tier_fog/round_processing.py` | Authentication, update decisions, TrustDB snapshots, SHAP diagnostics, timing, and per-node measurements. |
| Admission and verification | `src/tier_fog/client_instructions.py`; `src/tier_fog/gatekeeper.py` | Separate admission and returned-update authentication exchanges. |
| SHAP aggregation | `src/tier_fog/shap_verifier.py` | Filtering statistics, aggregation weights, and uniform fallback. |
| Edge training and inference | `src/tier_edge/local_ids.py`; `src/tier_edge/inference.py` | Local optimization timing, update sizes, and paired FP32/INT8 inference measurements. |
| Checkpoint security evaluation | `scripts/ops/evaluate_security.py` | Writes checkpoint metrics and log summaries to `heavy_metrics.json`. |
| Evasion and backdoor metrics | `src/shared/security/adversarial_math.py`; `src/shared/security/backdoor_math.py` | Clean/adversarial accuracy and targeted backdoor ASR. |
| Security-log summaries | `src/shared/utils/security_metrics.py` | Authentication FAR/FRR, update acceptance/rejection, and quarantine counters. |
| SHAP-aware attack diagnostics | `src/shared/security/shap_aware.py` | Attack objective, optimization progress, raw attribution distance, and feasibility. |
| Result plotting | `scripts/ops/plot_metrics.py` | Per-run curves with offline metrics attached only to the evaluated checkpoint round. |

### 1.2 Coverage terminology

| Status | Meaning |
| --- | --- |
| **Implemented** | The implementation calculates or records the measurement. Population eligibility and enabled features determine whether a particular run contains a value. |
| **Derivable** | The specified inputs support the calculation, but the result requires post-processing or matched comparison runs. |
| **Partial** | Existing instrumentation covers only part of the required scope or lacks attribution needed for the stated comparison. |
| **Not exported** | The named output is absent from the structured metric artifacts, even if related internal state or logs exist. |
| **Not implemented** | The named calculation or experiment mechanism is absent from the current implementation. |

A metric can be implemented and legitimately undefined for a particular input population. `null` is not automatically a calculation failure.

## 2. Execution Phases and Artifact Boundaries

| Phase | What is measured | Frequency and destination |
| --- | --- | --- |
| **During each training round** | Global performance; SHAP scores and decisions; authentication and admission events; TrustDB state; timings; payload sizes; configured edge-inference measurements. | Recorded by round in the global and fog artifacts. Availability depends on participation and enabled features. |
| **On-demand checkpoint evaluation** | Adversarial evasion robustness and targeted backdoor ASR. | Once per evaluation invocation for a saved checkpoint, including an intermediate or final checkpoint. The entry point writes `heavy_metrics.json`. |
| **On-demand log aggregation** | Authentication FAR/FRR, update acceptance/rejection rates, quarantine counters, and observability counters. | Calculated from recorded events, not from model predictions. Currently invoked by `evaluate_security.py`. |
| **Cross-round or cross-experiment analysis** | Convergence summaries, seed statistics, baseline improvements, ablations, scalability comparisons, and aggregated efficiency results. | Derived from matched runs and round histories. The plotting script supports per-run curves; convergence criteria, cross-run statistics, and comparative efficiency summaries require separate analysis. |

Checkpoint evaluation does not require proof that training converged. It also does not need to be performed automatically after every round. Each experiment may use its own appropriate evaluation settings; an all-attacks-in-one-invocation sweep is not required.

**Current entry-point limitation:** `evaluate_security.py` accepts an experiment name, loads `global_model.pt`, and obtains `evaluated_round` from the last global-performance record. It has no explicit checkpoint-path or checkpoint-round argument. Evaluating an earlier checkpoint requires an execution arrangement that preserves the correct checkpoint identity and round; changing the weights alone would not correct the reported round.

Run artifacts are organized as follows, relative to `results/<experiment>/`:

| Artifact | Contents |
| --- | --- |
| `<experiment>.json` | Run metadata and `performance[]` global evaluations. |
| `global_model.pt` | Latest saved global model. |
| `global_model_round_<round>.pt` | Snapshots selected by the configured interval or explicit round list. |
| `trustdb/fog_<id>_state.json` | Authentication, admission, update decisions, node roles, and TrustDB snapshots under `rounds`. |
| `shap_scores/fog_<id>_state.json` | SHAP scores, filter statistics, and regional aggregation diagnostics under `rounds`. |
| `performance_metrics/fog_<id>_state.json` | Fog timers and per-node training, inference, attack, payload, and token measurements under `rounds`. |
| `heavy_metrics.json` | Checkpoint security evaluation, security-log summaries, and evaluation execution times. |

Security-log aggregation has a different time scope: `calculate_security_rates()` processes all rounds present in the matching log files. It does not automatically stop at the checkpoint's round.

## 3. Metrics and Records Collected During Training

### 3.1 Global predictive performance

Implementation: `GlobalEvaluator.evaluate()` and the shared metric functions; persisted by both cloud aggregators.

| Metric | Field or calculation | Article comparison | Coverage |
| --- | --- | --- | --- |
| Global classification accuracy | `performance[].global_accuracy` | Clean results in Table II; poisoning results in Tables III and VI; accuracy-based ablation and failure-mode results. | Implemented. The same collector supports different experiment conditions. |
| Global evaluation loss | `performance[].global_loss` | Additional training-progress diagnostic; no corresponding headline loss result is reported. | Implemented. This is evaluation loss, not automatically local training loss. |
| Global macro-F1 | `performance[].global_macro_f1` | Table II reports F1 without explicitly identifying the averaging convention. | Implemented; exact F1-definition parity with the article is unresolved. |
| Global macro One-vs-Rest ROC-AUC | `roc_auc_multiclass_details()`; `performance[].global_auc` | AUC is reported in Section IX; its multiclass averaging convention is not specified. | Implemented local convention: unweighted mean over observed test classes, with coverage and per-class AUC. |
| Accuracy, F1, and loss histories | Performance values indexed by `round` and `timestamp` | Accuracy histories support Figure 5; Figure 3 instead compares accuracy across poisoning configurations. F1 and loss histories add diagnostics. | Implemented; convergence summaries require post-processing. |

Accuracy is evaluated against the ground-truth labels of the evaluation set. Poisoning the training process does not require poisoning the test labels or changing the accuracy formula.

`macro_f1()` gives equal weight to classes present in either the ground-truth labels or predictions. If

$$
S = \{c : N_{\mathrm{true},c} + N_{\mathrm{pred},c} > 0\},
$$

then

$$
\operatorname{MacroF1} = \frac{1}{|S|}\sum_{c\in S} F1_c.
$$

Classes absent from both labels and predictions are omitted. Prediction-only classes remain in the mean and contribute zero F1, so their false positives count. The helper returns `0.0` for completely empty inputs; the global evaluator rejects an empty test split before calling it. This averaging convention must be retained when comparing F1 values.

Global loss is sample-weighted mean cross-entropy over the test split. Accuracy and macro-F1 are fractions, not percentages.

#### AUC definition and undefined values

`roc_auc_multiclass_details()` calculates **macro one-vs-rest ROC-AUC over observed test classes**. For each configured class with at least one positive and one negative test example, it calculates binary ROC-AUC from that class's probability column, then takes the unweighted mean of those class AUCs. Class eligibility depends only on true test labels, never on predictions. `roc_auc_multiclass()` exposes the same calculation as a scalar-or-`None` compatibility helper.

Softmax is computed over **all configured model outputs**. Neither the model head nor its output probabilities are resized or renormalized for the observed subset. The bundled Edge sample has ten observed classes (IDs 0–9) while the author-derived model retains fifteen outputs. Its AUC averages ten binary class AUCs; the five absent classes have undefined per-class AUC, not zero. When all fifteen classes occur in the test split, this procedure reduces to ordinary full-class macro OVR AUC.

The evaluator returns scalar `auc_defined` and JSON-string `auc_details` (to respect Flower's scalar metric transport), and includes `auc` only when defined. Both Cloud writers save `global_auc`, `global_auc_defined`, and a decoded `global_auc_details` object each round, including initialization round 0. Its fields are:

| Field within `global_auc_details` | Meaning |
| --- | --- |
| `method` | `macro_ovr_observed_classes`; explicitly identifies the averaging convention. |
| `status` | `defined`, `empty_test_split`, or `insufficient_test_classes`. |
| `configured_classes` | Number of model outputs, e.g. 15. |
| `evaluated_classes` | IDs with both positive and negative test examples, e.g. 0–9; its length gives coverage, e.g. 10 of 15. |
| `class_counts` | Counts indexed by configured class ID, including zeros; their sum is the test sample count. |
| `per_class_auc` | AUC indexed by configured class ID; unsupported classes are JSON `null`. |

Empty or single-class populations yield `None` from the helper; global evaluation rejects an empty test split before metric calculation. Invalid shapes, labels, probabilities, nonfinite results and calculation exceptions fail evaluation instead of silently excluding more classes. A callback that supplies no diagnostics produces null `global_auc_details`; no cause is inferred.

This observed-class averaging is a disclosed local evaluation convention, not a claim about the authors' unspecified AUC method or their reported 0.989 value. Keep the test split fixed across rounds and compare runs with matching class coverage and evaluation populations. Historical results used the prior full-class requirement and are not rewritten; their null values cannot be reconstructed from accuracy/F1 alone. Prepared manifests also retain split class counts, although results do not yet link to a specific manifest identity.

Both global evaluation and standalone checkpoint evaluation load the prepared test artifact through `get_dataset()`. `dataset_fraction` controls training partition sampling and also affects the fog validation sampling pool; it does not resample the global test split. Increasing that fraction does not by itself resolve missing test classes.

### 3.2 SHAP and aggregation diagnostics

Implementation: `compute_shap_stability()`, `shap_weighted_aggregate()`, and `process_region()`. Join `shap_scores` to `trustdb` by fog, round, and node identity.

| Measurement | Recorded fields | Article comparison | Coverage |
| --- | --- | --- | --- |
| Per-agent SHAP stability | `raw_shap_scores` | Stability definition in Section V-B; honest/malicious comparison in Section VII-C and Figure 6. | Implemented. These are scalar stability scores, not full feature-attribution vectors. |
| Filtering population sizes | `shap_filter.submitted_updates`, `valid_population_size` | Supports interpreting the article's filtering mechanism. | Implemented; additional denominator diagnostics. |
| Population mean and standard deviation | `shap_filter.mean`, `std` | Statistics used by the article's filtering rule. | Implemented. |
| Rejection threshold | `shap_filter.threshold` | Corresponds to the article's mean-minus-two-standard-deviations rule. | Implemented. |
| Flagged and rewarded identities | `saboteurs`, `rewarded_nodes` | Filtering and reputation operations in Sections V-A/V-B. | Implemented; decisions are not ground truth. |
| Unnormalized weight total | `shap_filter.raw_weight_total` | Relates to aggregation weighting; not a separately reported article result. | Implemented. |
| Uniform-weight fallback | `shap_filter.uniform_fallback` | Additional implementation diagnostic. | Implemented. |
| Regional aggregation/trust value | `total_regional_trust` | Additional aggregation diagnostic. | Implemented; aggregation weight passed to the cloud, not mean per-agent TrustDB reputation. |
| Optional SHAP-aware attack diagnostics | `node_shap_attack_metrics` | Potentially relevant to Section VIII-B. | Implemented in `performance_metrics`; populated for SHAP-aware training. Details below. |

Groupwise honest/malicious SHAP means and score trajectories are derivable by joining stability scores to node-role ground truth for the same identity and round. Invalid updates can retain an initialized score of `0.0`; use the valid-population count and rejection records when interpreting distributions. A zero entry is not always a successfully calculated stability score.

The implementation reduces GradientSHAP attributions to mean absolute importance per feature over matched validation examples. Stability is `1 - ||phi_model - phi_ref||₂ / (||phi_ref||₂ + 1e-6)`. The filter computes population standard deviation (`unbiased=False`) over valid scores and accepts scores at or above `mean - 2 * std`. A singleton valid population has zero standard deviation.

Ordinary weights are `max(0, stability) * validation_accuracy * sqrt(sample_count)` for updates that pass validation and filtering. Negative raw scores remain in the filtering population but receive zero ordinary weight. When the surviving weight total is below `1e-12`, survivors receive uniform weights and `total_regional_trust` becomes `1.0`; with no valid survivors, the regional weight is `0.0`. Otherwise, it is the raw weight sum. Passing the filter therefore does not necessarily mean receiving a positive ordinary aggregation weight.

`node_shap_attack_metrics` contains client-reported `shap_attack_*` fields: objective, solver, data-access assumption, seed, epochs, learning rate, clipping, backtracking budget, validation/explained sample counts, quantization bits, label-flip probability, step and candidate-evaluation counts, initial/final loss, final raw SHAP distance, and optimization status. Constrained attacks additionally report `shap_attack_tau` and `shap_attack_feasible`. These describe attack construction; fog acceptance is recorded independently in `update_outcomes`.

Full SHAP vectors need not be routinely exported to retain these scalar metrics. However, not exporting a vector is different from not computing the attributions required by the algorithm. Scalar stability scores alone also do not reconstruct the article's proposed cumulative feature-attribution drift analysis.

Negative stability values follow from the stability formula when attribution deviation exceeds the reference norm. Interpret `uniform_fallback` with `raw_weight_total` and the surviving population; neither negative values nor fallback alone establishes a calculation defect.

### 3.3 Authentication, admission, update decisions, and TrustDB

Implementation: admission instructions, `Gatekeeper`, `TrustDB`, and `process_region()`; persisted in `trustdb/fog_<id>_state.json`.

| Record | Fields | Meaning and article relationship |
| --- | --- | --- |
| Authentication and admission outcomes | `attestation_outcomes`, `admission_outcomes`; `authenticated`, `accepted`, `attempted`, `expected_valid`, `reason`, `tpm_id` | Records authentication validity separately from permission to participate. Supports Section VI-F security metrics. |
| Accepted/rejected identity lists | `attestation_accepted`, `attestation_rejected` | Supplementary diagnostics; security-rate calculations use the outcome dictionaries rather than these lists. |
| Attestation bypass state | `attestation_bypassed` | Distinguishes intentionally bypassed checks from measured authentication performance. |
| Update-verification decisions | `update_outcomes`; `role`, `accepted`, `stage` | Supports honest-update rejection and poisoned-update acceptance statistics. |
| Rejection explanations | `update_rejections` | Additional diagnostic context; some reasons combine SHAP and validation rejection. |
| Reputation and recovery state | `trust_db`; `score`, `is_quarantined`, `recovery_streak`, `last_attestation_passed`, `last_attestation_round`, `last_penalty_round` | Implements observability around the TrustDB policy in Section V-A. |
| Ground-truth roles | `node_ground_truths`; run-level `experiment_ground_truth` | Supplies evaluation labels; a poisoning role does not imply invalid TPM authentication. |

These records are implemented. They are raw inputs to the summaries in Section 5, not interchangeable security rates. Processing `status`, `error`, timestamps, and fog identity/tier metadata provide additional run-health and identification context rather than predictive metrics.

TrustDB records are snapshots taken in `process_region()`'s finalization step, after any authentication, penalties, rewards, or recovery transitions performed up to that point. They are end-of-processing state, including on error paths, rather than previous-round reputation values.

### 3.4 Timing and execution measurements

Implementation: cloud evaluation, edge training/client adapters, and fog round processing.

| Measurement | Fields | Article comparison | Coverage |
| --- | --- | --- | --- |
| Global round duration | `round_latency_total_sec` | Round duration and scalability in Section VI-F. | Implemented; instrumentation boundaries must remain consistent across experiments. |
| Reported local-work duration | `node_training_latencies`; `node_local_training_start_unix`, `node_local_training_end_unix` | Supports analysis of local training cost. | Implemented; not an isolated adversarial-training overhead measurement. |
| Admission duration | `latency_admission_ms` | Part of the authentication/admission workflow. | Implemented. |
| Update-attestation validation duration | `latency_update_attestation_ms`; `attestation_validation_start_unix`, `attestation_validation_end_unix` | Attestation timing in Section VI-F. | Implemented at fog/round scope. |
| Combined attestation duration | `latency_attestation_ms` | Attestation overhead. | Implemented; admission duration plus update-validation duration. |
| SHAP verification and aggregation duration | `latency_shap_computation_sec`; `shap_calculation_start_unix`, `shap_calculation_end_unix` | SHAP cost in Sections VI-A/VI-F. | Implemented. |
| Regional encoding interval | `aggregation_start_unix`, `aggregation_end_unix` | Additional phase-level timing. | Measures `encode_model()` after SHAP-weighted aggregation; duration is derivable by subtraction. |
| Fog processing interval | `round_start_unix`, `round_end_unix` | Additional fog-level timing. | Implemented; regional processing after local results return, not a full federated round. |

Timer boundaries are defined by the operations they enclose:

- `round_latency_total_sec` runs from cloud fit configuration to completion of centralized evaluation, before result/checkpoint persistence. An initial round-zero evaluation precedes fit configuration and is not a training-round duration.
- Local optimization timing includes local epochs, attack work performed inside that block, and final update encoding. It excludes the subsequent inference benchmark and attestation-token generation; GPU-lock waiting is outside the optimization timer. Client adapters provide a broader call-duration fallback when the trainer does not return an optimization timer, including attestation-only calls.
- `latency_admission_ms` accumulates per-client admission exchanges, including quote requests, response waits, and fog validation. It is not only signature-verification time.
- `latency_update_attestation_ms` surrounds `filter_node_updates()`, including ledger access, verification, outcome handling, and missing-response processing.
- `latency_shap_computation_sec` surrounds the entire `shap_weighted_aggregate()` call: attribution computation, validation inference, filtering, weight calculation, and weighted model aggregation.
- `aggregation_start_unix` and `aggregation_end_unix` surround regional model encoding after that call. They do not isolate weighted averaging.
- Fog `round_start_unix` and `round_end_unix` bound regional result processing and snapshot preparation, before persistence. They exclude earlier client training and admission.

Exclude attestation-only and other non-training attempts when summarizing actual training time. Keep seconds and milliseconds distinct, and do not add a combined timer to its own components.

No individual verification-duration series is exported. Dividing fog-wide duration by an attempt count gives normalized batch cost, not an individually measured verification latency. These boundaries also prevent treating the local optimization timer as isolated adversarial-training overhead.

### 3.5 Communication and model representation

Implementation: cloud evaluation, edge training/client adapters, and fog round processing.

| Measurement | Fields | Article comparison | Coverage |
| --- | --- | --- | --- |
| Global payload-size entry | `performance[].payload_size_mb` | Related to communication efficiency. | Implemented; not by itself a system-wide communication total. |
| Per-node model-update payload size | `node_payload_sizes` | Edge-to-fog update-volume accounting. | Implemented. |
| Model-update array bytes | `node_update_array_bytes` | Representation-size accounting. | Implemented for submitted updates. |
| Model-update tensor bytes | `node_update_tensor_bytes` | Another reported representation-size boundary. | Implemented; summed lengths of Flower-serialized tensor byte strings, including array serialization metadata. |
| Attestation token bytes | `token_sizes_bytes` | Related to the article's per-client attestation communication overhead. | Implemented; encoded length of the returned token JSON string, excluding admission exchanges and transport overhead. |
| FP32 and INT8 model-state sizes | `edge_fp32_state_bytes`, `edge_int8_state_bytes` | Quantized model size and reduction in Sections IV/VI-A/IX. | Implemented; model-state size and transport payload size are different measurements. |

The payload-size fields use array bytes divided by `2^20`. Their numerical convention is therefore MiB, even where a field is named `mb`. Use explicit byte units in comparisons.

Array bytes, tensor bytes, payload-size entries, and serialized model-state sizes are not independent traffic categories to be added together. Identify which representation is actually being counted at each communication boundary.

Edge-update totals are derivable by summing the appropriate per-node records across fogs. These records are populated after successful authentication, so they do not account for every received or attempted upload. The global `payload_size_mb` measures one global model representation, without multiplying by recipient count. Complete communication accounting remains partial: the exporters do not total all downlinks, fog-cloud transfers, admission exchanges, failed uploads, or transport headers. A comparison with the article's communication-per-round metric requires the same links, directions, and protocol boundary for both methods.

Fog metadata records `telemetry_mode` as `simulated` or `deployment`, and `fog_cloud_transport` as `in_process_wire_roundtrip` or `flower_grpc_via_tcp_ipc`. Simulated representation sizes and timers must retain that execution-mode qualification; they are not physical-network traffic measurements.

### 3.6 Quantized edge inference

Implementation: `evaluate_cpu_inference()`; fog output under `node_inference_metrics` in `performance_metrics`.

| Measurement | Fields | Article comparison | Coverage |
| --- | --- | --- | --- |
| FP32 versus INT8 accuracy | `edge_fp32_accuracy`, `edge_int8_accuracy` | Additional direct measurement of quantization's predictive effect. | Implemented. |
| FP32 versus INT8 loss | `edge_fp32_loss`, `edge_int8_loss` | Additional diagnostic. | Implemented. |
| FP32 versus INT8 inference latency | `edge_fp32_inference_ms_per_sample`, `edge_int8_inference_ms_per_sample` | Additional paired inference benchmark beyond the article's aggregate efficiency results. | Implemented. |
| Model-state byte sizes | `edge_fp32_state_bytes`, `edge_int8_state_bytes` | Quantization/model-size comparison. | Implemented. |
| Evaluation context | `edge_inference_samples`, `edge_inference_int8`, `edge_inference_cpu_threads` | Additional benchmark metadata. | Implemented. |

The benchmark compares CPU copies on identical clean held-out validation batches. Dynamic INT8 conversion targets `Linear` and `LSTM` modules. Inference timing excludes conversion and warmup, and divides accumulated forward-pass time by sample count. State bytes come from serializing each model's `state_dict` and include serialization metadata; they measure neither total RAM nor network traffic.

Model-size reduction, inference speedup, and accuracy difference are derivable from these paired measurements. Edge-local measurements must not be relabeled as global-model test accuracy.

Empty inference dictionaries and absent update-byte values for denied clients are not automatically missing instrumentation. Interpret them alongside admission and participation outcomes, rather than replacing absent measurements with successful zero-valued observations.

## 4. On-Demand Checkpoint Evaluations

Implementation: `scripts/ops/evaluate_security.py` and the adversarial/backdoor metric helpers. Output paths below are inside `heavy_metrics.json`.

### 4.1 Adversarial evasion robustness

**Output:** `metrics.adversarial_robustness`.

The evaluator calls `evaluate_robustness()` with the saved model, prepared test split, attack selection, perturbation budget, step size, iteration count, and clipping settings. The caller obtains these settings from `robustness_eval_attack`, `benign_eps`, `benign_alpha`, `benign_n_iter`, `clip_min`, and `clip_max`.

This provides the evaluation entry point for the article's adversarial-accuracy comparisons in Table IV and Figure 4. FGSM, PGD-7, and PGD-20 are attack configurations, not different accuracy formulas.

**Coverage: Implemented.** `evaluate_robustness()` supports `fgsm` and `pgd`; PGD-7 and PGD-20 use `pgd` with iteration counts 7 and 20.

| Nested field | Definition and unit |
| --- | --- |
| `clean_acc` | Correct clean predictions divided by evaluation sample count; fraction. |
| `adv_acc` | Correct predictions on adversarial inputs divided by the same count; fraction. |
| `acc_drop` | `clean_acc - adv_acc`; fraction difference. Multiply by 100 for percentage points. |
| `attack` | `fgsm` or `pgd`. |
| `eps` | Perturbation budget in model-input feature units. |

Every evaluation example is attacked, including examples misclassified before perturbation. The attack is untargeted, using ground-truth cross-entropy gradients and an L-infinity perturbation bound. PGD uses a random start and projects every step into the allowed perturbation interval. Optional feature bounds are expanded where needed to contain the original observation, so out-of-range held-out features still respect the perturbation budget. These perturbations operate on prepared model inputs, not raw network-traffic fields.

The output group records attack name and epsilon, but not step size, iteration count, clipping, sample count, or random-start state. Retain the run configuration and evaluation seed/state with the result for reproducible attack comparisons; the standalone entry point does not explicitly reseed PGD before evaluation.

**Empty-population caveat:** this helper divides by `max(total, 1)` and therefore returns zero accuracies and zero drop for empty inputs. Treat that case as an invalid evaluation, not a measured zero. Unlike the global evaluator and ASR helper, it does not reject the empty population or return an undefined value.

Robust accuracy measures the resulting model's behavior under perturbed inputs. Attributing an improvement specifically to adversarial training requires an appropriate comparison or ablation; a single robustness score does not isolate that contribution.

### 4.2 Targeted backdoor Attack Success Rate

**Output:** `metrics.backdoor_asr_percentage`.

**Coverage: Implemented by `compute_backdoor_asr()`.**

For target class `t`, eligibility is determined by the original labels:

$$
E = \{i : y_i \ne t\}.
$$

The configured trigger overwrites the specified feature columns for every eligible example. ASR is:

$$
\operatorname{ASR}_{\%} =
100\frac{\sum_{i\in E}\mathbf{1}[\arg\max f(\operatorname{trigger}(x_i)) = t]}{|E|}.
$$

The implementation excludes original target-class examples, evaluates all remaining examples in batches, returns `None` when none are eligible, and rejects nonfinite model outputs with an error. It does not restrict eligibility to examples that the untriggered model classified correctly.

The target is the attacker's chosen class, not necessarily a class representing malicious traffic. ASR measures targeted success, not arbitrary misclassification.

The training poisoner samples from all training rows, including target-class rows, and changes selected labels to the target. That training-sampling rule is distinct from the ASR denominator and is not a contradiction. The training poisoner implements a dirty-label backdoor. Clean-label backdoor generation is not implemented; applying the existing ASR evaluator to a clean-label experiment requires that separate training mechanism.

The evaluator also computes ASR for checkpoints trained without backdoor poisoning. Such a result is a trigger-susceptibility baseline, not evidence that the checkpoint underwent backdoor training. Clean accuracy under backdoor poisoning is obtained from the ordinary clean-evaluation metric for the corresponding model and evaluation population.

This metric corresponds to Section VI-E, Table VI, and the ASR concept in Table VII. The article does not fully specify its evaluation denominator, so matching the metric name does not prove identical evaluation eligibility.

### 4.3 Evaluation execution times

| Field | Meaning | Article relationship |
| --- | --- | --- |
| `execution_stats.robustness_calculation_time_sec` | Runtime of the robustness evaluator call. | Additional evaluation-cost diagnostic. |
| `execution_stats.asr_calculation_time_sec` | Runtime of the ASR evaluator call. | Additional evaluation-cost diagnostic. |
| `execution_stats.attestation_parsing_time_sec` | Runtime of security-log parsing and summarization. | Additional post-processing cost; not operational attestation latency. |

These output paths are written by `evaluate_security.py`. They do not measure on-device adversarial-training overhead or replace the per-round timers in Section 3.4.

## 5. Security Metrics Calculated from Round Logs

Implementation: `calculate_security_rates()`, invoked by `evaluate_security.py`; input files are `results/<experiment>/trustdb/*.json`. These summaries concern the events in the selected logs, not the predictive behavior of a checkpoint.

### 5.1 Attestation FAR and FRR

**Output group:** `metrics.attestation_gatekeeper`.

| Output | Exact calculation or meaning |
| --- | --- |
| `valid_attempts` | Included outcomes with Boolean `expected_valid = true` and Boolean `authenticated`. |
| `invalid_attempts` | Included outcomes with Boolean `expected_valid = false` and Boolean `authenticated`. |
| `false_rejections_count` | Included valid outcomes with `authenticated = false`. |
| `false_acceptances_count` | Included invalid outcomes with `authenticated = true`. |
| `FRR_percentage` | `100 * false_rejections_count / valid_attempts`. |
| `FAR_percentage` | `100 * false_acceptances_count / invalid_attempts`. |

**Coverage: Implemented.** These rates correspond to the authentication-rate concepts in Section VI-F. They are percentages, not unit-interval fractions.

The calculation uses `authenticated`, not admission `accepted`. A valid authenticated client denied participation by quarantine is therefore not counted as an authentication false rejection. Likewise, a poisoned model update does not make its sender an invalid authentication identity.

Both `admission_outcomes` and `attestation_outcomes` contribute events. Records from bypassed rounds are excluded from valid/invalid denominators. Outcomes explicitly marked `attempted = false` are also excluded. Missing `attempted` alone does not trigger that exclusion; records with non-Boolean validity or authentication fields are classified as unknown.

A zero denominator produces `None`. Missing security logs therefore do not establish zero FAR or FRR.

### 5.2 Additional attestation observability

The following counters are implemented in the same output group:

| Counter | Meaning |
| --- | --- |
| `unknown_attempts` | Included event records without usable Boolean validity/authentication labels. |
| `missing_responses` | Non-bypassed events explicitly marked as not attempted. |
| `reported_client_crashes` | Subset of those events whose reason is `client_training_crash`. |
| `bypassed_attempts` | Event records skipped because the round is marked `attestation_bypassed`. |
| `legacy_rounds_without_admission_records` | Processed round records lacking the `admission_outcomes` key. |

These provide visibility beyond the article's headline FAR/FRR results. They count processed records according to the code, not necessarily distinct clients or distinct physical challenges.

### 5.3 Model-update verification

**Output group:** `metrics.update_verification`.

| Output | Exact calculation or meaning |
| --- | --- |
| `honest_updates` | Update outcomes with role `benign` and Boolean `accepted`. |
| `honest_rejections` | Those honest outcomes with `accepted = false`. |
| `poisoned_updates` | Outcomes with recognized poisoning roles and Boolean `accepted`. |
| `poisoned_acceptances` | Those poisoned outcomes with `accepted = true`. |
| `unknown_updates` | Outcomes with unsupported roles or non-Boolean acceptance values. |
| `honest_rejection_percentage` | `100 * honest_rejections / honest_updates`. |
| `poisoned_acceptance_percentage` | `100 * poisoned_acceptances / poisoned_updates`. |

The recognized poisoning roles are `label_flip`, `gradient_manip`, `backdoor`, and `shap_aware`. Both percentages are `None` when their denominator is zero.

**Coverage: Implemented, but broader than an explicitly SHAP-only metric.** The calculator does not inspect `stage` or rejection reasons. It measures the decision population recorded in `update_outcomes`.

A SHAP-specific false-positive rate requires a population of benign updates actually evaluated by the relevant SHAP rule and rejection labels attributable to that rule. The producer records payload-validation failures at stage `payload`. Stage `shap` combines attribution/model validation and SHAP threshold decisions, with failures labeled `SHAP or validation rejection`. The exported honest-rejection rate should not automatically be presented as a pure SHAP-threshold FPR for every experiment.

This distinction matters for the extreme-non-IID filtering FPR in Section VIII-C/Table VII.

Poisoned-update rejection rate is derivable from the raw counts:

$$
100\frac{\text{poisoned\_updates}-\text{poisoned\_acceptances}}{\text{poisoned\_updates}}.
$$

It describes rejection among recorded recognized poisoned updates, not end-to-end rejection among all malicious clients. It is also different from classification accuracy under poisoning and from a theoretical detection-probability guarantee.

### 5.4 Quarantine and readmission

**Output group:** `metrics.quarantine`.

| Output | Exact meaning |
| --- | --- |
| `node_rounds` | Number of node-state records processed from `trust_db`. |
| `quarantined_node_rounds` | Number of those records with `is_quarantined = true`. |
| `readmissions` | Count of transitions from quarantined to non-quarantined for an identity within one processed file. |

**Coverage: Implemented.** These are additional quantitative diagnostics around the article's TrustDB policy.

A quarantine-exposure percentage is derivable as `100 * quarantined_node_rounds / node_rounds`. It is not a percentage of unique agents or an admission-denial rate. Readmission tracking resets between files and counts transitions between successive available snapshots. It does not inspect recovery-streak completion, and missing round snapshots can obscure the transition's exact round.

## 6. Derived Comparisons and Article Coverage

The measurements below generally reuse existing metrics. An unexecuted experiment is not a missing collector. Dedicated reporting automation remains separate from having the raw ingredients.

| Article measurement or comparison | Existing measurement to use | Status and remaining qualification |
| --- | --- | --- |
| Clean accuracy and F1 across datasets: Table II | Global accuracy and macro-F1. | Collector coverage exists; use the relevant dataset/experiment. Article F1 averaging remains unspecified. |
| Label-flipping and gradient-manipulation accuracy: Tables III/VI, Figure 3 | Global accuracy under each configured poisoning condition. | Covered by the existing accuracy collector; no special attack-accuracy field is required. |
| Clean accuracy retained during backdoor poisoning: Section VI-E | Clean global accuracy from the backdoor-trained model. | Covered; match checkpoint and evaluation population. |
| Evasion accuracy versus attack/budget: Table IV, Figure 4 | Checkpoint robustness evaluation. | Implemented: `clean_acc`, `adv_acc`, and `acc_drop` for configured FGSM/PGD. Retain attack budget and iteration settings. |
| Clean-accuracy retention: Section VI-D | Matching clean accuracy and an explicitly named reference accuracy. | Derivable when both are available; distinguish retained clean accuracy, a ratio, and a percentage-point change. The article does not fully define its retention calculation. |
| Ablation effects: Table V | Accuracy, robustness, ASR where relevant, and timing from matched configurations. | Reuses existing metric families; comparative analysis is separate. |
| Convergence rounds: Sections VI-F/VII-B, Figure 5 | Round-indexed global accuracy. | Derivable; no automatic convergence criterion or convergence-round exporter. Declare the threshold, tolerance, and persistence window used. |
| Honest versus malicious SHAP distributions: Section VII-C, Figure 6 | Stability scores joined to node-role labels. | Derivable; declare per-round versus pooled weighting. |
| Mean, standard deviation, paired significance tests: Section VI | Matched run-level results indexed by seed and condition. | Derivable after matched runs; no cross-run statistical exporter. Round-to-round variation is not seed-to-seed variation. |
| Communication reduction: Section VI-F/IX | Comparable communication totals for the method and baseline. | Partial: edge-update and returned-token sizes exist; the exporters do not total all system communication. |
| Model-size reduction: Section IV | Paired FP32/INT8 state sizes. | Derivable; do not substitute inference-state bytes for transmission bytes. |
| Computational overhead relative to a baseline: Section VI-F | Matched round durations. | Derivable; keep timing scope, hardware, and execution mode consistent. |
| Incremental adversarial-training cost: Section VI-F | Controlled training-time comparisons with/without adversarial training. | Derivable under matched conditions; no dedicated timer isolates adversarial-example generation or its incremental training cost. |
| Scalability accuracy and round duration: Sections IV/VI-F | Global accuracy, round duration, and client/fog counts. | Same collectors; larger deployments are separate experiments, not new metric types. |
| SHAP-aware, slow-poisoning, and collusion accuracy degradation: Section VIII-B/C | Accuracy trajectories and matched reference runs. | Accuracy collectors cover all conditions. SHAP-aware constrained/unconstrained training is implemented; slow-poisoning and coordinated-collusion scenarios require explicit attack orchestration, beyond selecting a malicious-client fraction. |
| Extreme-non-IID filtering FPR: Table VII | Honest-update decisions with sufficiently specific rejection attribution. | Partial: generic update rejection includes payload/validation failures; a SHAP-only rate requires separate threshold attribution. |
| Clean-label backdoor ASR: Table VII | Existing ASR evaluator applied to a clean-label-poisoned checkpoint. | ASR is implemented; clean-label backdoor generation is not implemented. The existing poisoner changes selected labels. |
| Replay and impersonation detection: Section VI-F | Invalid-authentication outcomes separated by attack type. | Derivable as invalid-attempt rejection for separately identified, single-attack runs. The current summary does not export separate attack-type rates for mixed populations. |
| Slow-poisoning detection delay and cumulative SHAP drift: Section VIII-B/C | Attack start, detection events, and suitable drift observations. | Not implemented as dedicated metrics. Scalar stability histories do not reconstruct full cumulative attribution drift. |
| SHAP sampling speedup and accuracy trade-off: Section VIII-D | Matched SHAP durations and accuracy measurements. | Timing and accuracy inputs exist. GradientSHAP sample counts are configurable; the article's KernelSHAP sampling optimization is not implemented by this GradientSHAP collector. |
| Selective adversarial-training cost/robustness trade-off: Section VIII-D | Matched timing and checkpoint robustness measurements. | Timing and robustness inputs exist. Training attacks a configured leading fraction of each batch; confidence-based sample selection is not implemented. |

For derived efficiency measurements, use explicit definitions:

$$
\text{Size reduction}_{\%}=100\left(1-\frac{B_{\mathrm{INT8}}}{B_{\mathrm{FP32}}}\right),
\qquad
\text{Inference speedup}=\frac{T_{\mathrm{FP32}}}{T_{\mathrm{INT8}}}.
$$

$$
\text{Communication reduction}_{\%}=100\left(1-\frac{B_{\mathrm{method}}}{B_{\mathrm{baseline}}}\right),
\qquad
\text{Time overhead}_{\%}=100\left(\frac{T_{\mathrm{method}}}{T_{\mathrm{baseline}}}-1\right).
$$

A difference between two percentage-valued accuracies is a percentage-point difference, not automatically a relative percentage improvement.

The article's theoretical security/detection bounds and estimates for future hardware deployments are not empirical metrics automatically reproduced by an observed rate or timer. Its privacy-leakage evaluation is explicitly left to future work; it should not be misclassified as an omitted reported benchmark.

## 7. Missing, Partial, and Potentially Misleading Outputs

| Item | Implementation behavior | Classification or action |
| --- | --- | --- |
| AUC availability explanation | New global results export observed-class macro AUC, its defined flag, method/status, evaluated IDs, actual test class counts and per-class AUC. Historical files lack these diagnostics. | Coverage is explicit; a direct manifest identity link remains absent. |
| Macro-F1 class population | Classes absent from both targets and predictions are omitted; prediction-only classes remain. | Preserve this definition when comparing against fixed-class or differently averaged F1. |
| SHAP-only false-positive rate | The update calculator pools stages; stage `shap` itself includes validation failures. | Generic honest-update rejection is implemented. A pure threshold FPR needs separate eligibility and rejection-cause fields. |
| Authentication-event identity | Admission and returned-update checks use separate challenges; the summary pools both dictionaries without deduplication. | Counts are stage events, not unique clients or independent trials. Separate stage summaries are not exported. |
| Very small security rates | `_percentage()` rounds percentages to four decimal places. | Positive rates below display resolution can appear as zero. Keep raw counts and adequate precision for rare-event claims. |
| Per-agent verification latency | Admission and returned-update timers cover fog-level workflows. | Partial; individual verification durations are not exported. |
| Complete system communication | Model arrays, serialized tensors, and returned tokens are measured, but all links, directions, failures, and protocol exchanges are not totaled. | Partial accounting scope; choose a common boundary before comparing totals. |
| Explicit model parameter count | The result schema has no parameter-count field. | Not exported; derivable from the model. Serialized state bytes are not a parameter count. |
| Individual normalized aggregation weights | Filtering statistics, weight total, and fallback are exported; normalized per-agent weights are not. | Additional diagnostic gap; acceptance alone does not imply positive weight. |
| Rollback decisions and counts | Cloud aggregation returns `aggregation_status = rolled_back` and logs a warning when candidate validation accuracy falls to or below the configured fraction of the previous accepted model's accuracy. | Rollback is implemented, but `performance[]` does not persist the decision or a rollback count. Repeated accuracy alone does not identify rollback. |
| Arbitrary-checkpoint provenance | The entry point loads `global_model.pt` and labels it with the latest logged round. | Snapshot files exist, but selecting a checkpoint path and round requires additional evaluation support or a consistent evaluation copy of the run. |
| Security-log cutoff | Every round in every matching TrustDB JSON file is processed. | The summary is not bounded by checkpoint round. Record the log range separately or restrict the input logs. |
| Evaluation-result preservation | `heavy_metrics.json` is opened in write mode. | Repeated evaluations replace the artifact; preserve distinct results when comparing attacks, budgets, or checkpoints. |
| Evaluation-population identity | Global and standalone evaluation load the same prepared test artifact for matching configuration. Dataset fraction does not select test examples at evaluation time. | Keep the dataset artifact and manifest identity with the checkpoint; the heavy result does not embed sample counts or an artifact identifier. |
| Robustness provenance and empty inputs | The result records attack and epsilon but omits other attack settings and population count; the helper returns zeros on an empty population. | Preserve full settings and random state, and require a nonempty evaluation population. |
| Cross-run statistical automation | Batch tooling runs and archives experiments; plotting produces per-run curves. | Mean/std, paired tests, convergence criteria, and comparative efficiency summaries require separate analysis. |
| SHAP weighting behavior | Raw negative scores enter filtering; their ordinary weights are clamped to zero. A near-zero surviving total triggers uniform fallback. | Implemented and recorded; assess score scale, valid population, and fallback together. |
| Slow-poisoning detection and cumulative attribution drift | Scalar stability histories are exported, but no cumulative feature-attribution drift or detection-delay calculation is exported. | Dedicated metrics are not implemented; scalar stability alone cannot reconstruct the required vector drift. |

Admission and update authentication are distinct exchanges even when their outcome fields have identical values. The pooled FAR/FRR counts both stages. Two stages for the same client and round can be correlated; their total must not be treated as a count of independent clients. Duplicate copies of the same event can leave rates unchanged while inflating counts.

The log calculator scans every JSON file under `trustdb` and does not deduplicate overlapping snapshots. Keep one authoritative history per fog there; copied histories would be counted again. The legacy-record counter measures absence of the `admission_outcomes` key, not independently established format age.

`plot_metrics.py` attaches ASR and robustness only to `evaluated_round`. A single offline evaluation supplies a checkpoint point, not a per-round robustness or ASR trajectory.

## 8. Reporting and Debugging Conventions

### 8.1 Units and undefined measurements

Global accuracy, macro-F1, ROC-AUC, edge accuracy, and robustness accuracies are fractions. ASR and security rates are percentages. Robustness `acc_drop` is a fraction difference; multiplying it by 100 gives percentage points. Loss is cross-entropy, timings use the units in their field names, and payload fields ending in `mb` use MiB.

An absent eligible population must be reported as undefined or invalid rather than as a successful zero rate. AUC, ASR, and security-rate helpers return `None` for their undefined populations; the robustness helper's empty-input zeros require explicit invalid-population handling. Distinguish undefined, disabled, not attempted, unknown, and measured zero wherever the available records permit it.

### 8.2 Experiment and evaluation identity

Retain the run configuration and identify the method, dataset, seed, split/preprocessing version, attack settings, enabled defenses, client/fog counts, and execution mode associated with each result. These are comparison metadata, not separate performance metrics.

For checkpoint evaluation, also preserve the checkpoint identity and round, evaluation settings, and evaluation population. Report the security-log round range independently of checkpoint identity. The evaluator exports `run_name`, `evaluated_round`, `timestamp`, and `human_readable_time`; it does not embed the checkpoint hash, evaluation sample counts, or security-log range. Keep those identifiers in the experiment record when comparing evaluations.

### 8.3 Selected diagnostics instead of unrestricted variable dumps

Routine telemetry consists of structured metric and decision records. Full local-variable dumps are not part of the metric schema.

Use explicitly enabled debugging output for selected state beyond the metric schema. Full attribution vectors, model tensors, raw samples, and sensitive attestation material need separate retention decisions.

## 9. Coverage Overview

The implementation records global predictive performance, SHAP statistics, authentication/admission decisions, TrustDB state, phase timing, update/token sizes, and paired FP32/INT8 inference measurements.

Checkpoint evaluation calculates FGSM/PGD robustness and targeted backdoor ASR. Log aggregation calculates authentication FAR/FRR, update acceptance/rejection summaries, and quarantine counters. Execution timers measure the cost of these offline calculations.

Article comparisons such as poisoning accuracy, ablations, scalability, convergence, and multi-seed results reuse these metric families under matched configurations. Additional inference, attack-construction, filtering, and security-observability diagnostics extend the inventory beyond the article's headline results.

Remaining gaps concern SHAP-specific rejection attribution, individual verification latency, complete communication accounting, structured rollback/weight outputs, evaluation provenance, cross-run reporting, and cumulative-drift/detection-delay metrics. Clean-label backdoor generation and confidence-selective adversarial training also require experiment mechanisms beyond the existing metric collectors.
