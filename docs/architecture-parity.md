# ZTA-FL implementation parity with the article

This is the implementation inventory. Each component records its implementation, evidence, status and limitations.

## 1. Specification and interpretation

The specification is *Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems*, arXiv:2512.23809v1, available as the [reference article](https://arxiv.org/abs/2512.23809v1). Section references below refer to that article.

This repository implements the article's architecture using Flower in a controlled environment. Published weaknesses are documented as limitations of the method.

****"Verbatim" below means an explicitly stated rule or equation is preserved. It does not mean Python source was copied from the PDF, nor that an entire component is exactly reproduced when only its central rule is. The author-code snapshot is available [here](https://github.com/ssam18/zta-federated-learning/).

### 1.1 Implementation policy

The implementation uses the article for architectural responsibilities and explicit rules, the supplied author code for unspecified operational details, and narrow correctness fixes where execution or measurement would otherwise be invalid. Section 2 lists the architectural and supporting parts by name. Section 3 describes each part, its implementation category, evidence, implementation status and limitations. Deferred work is explicitly distinguished from implemented behavior.

1. Use the article for the hierarchy, threat model, explicit equations and trust/rollback rules.
2. Use the supplied authors' code for unspecified operational details, provided that it does not contradict an explicit rule or produce an invalid measurement.
3. Retain narrow correctness fixes: valid payload handling, quarantine enforcement, evaluation without changing model state, and honest failure accounting. A claim of a bug must be supported by an executable counterexample or direct control-flow evidence.
4. Where the paper leaves a numerical domain undefined, prefer an existing author convention over inventing a detector. Record the exact departure and its consequences; do not label it verbatim.
5. Preserve known weaknesses of the method.
6. Missing capabilities of the article's primary architecture remain explicitly identified. A correctness rationale does not establish equal experimental results. The article's future-work proposals are outside this project's implementation scope, are retained here for completeness, and will not be implemented.

Architecture is assessed from implemented responsibilities, reachable code paths and configuration wiring. Configuration values select behavior for an individual run; their current values do not establish architectural coverage, absence or parity. Configurable capabilities are described independently of the settings currently selected.

The project is a Flower implementation of the published ZTA-FL architecture, informed by the supplied authors' PyTorch code, with explicitly documented implementation conventions and correctness fixes. It does not claim bit-for-bit source reproduction or reproduction of the published results.

The status vocabulary is deliberately strict:

| Status | Meaning |
| --- | --- |
| Direct rule | The named article rule is implemented as stated. Scope qualifications are explicit. |
| Integrated | The same architectural responsibility is carried out through Flower, shared code, or offline preparation rather than a separately named runtime module. |
| Partial / unverified | Part of the responsibility exists, but a stated detail is absent, an interpretation is required, or exact equivalence has not been established. |
| Not implemented | The named capability is absent. |
| Configurable | Source implements the capability, and configuration selects its use or parameters for a run. Current configuration values do not determine architectural coverage. |
| Outside implementation scope | An article future-work proposal retained for completeness. It will not be implemented and is not counted as a missing requirement of the primary architecture. |

A replacement algorithm is not an "indirect implementation." A transport encoding is not a quantized inference engine. Preparing CSV tensors is not live perception of network traffic. These distinctions apply throughout this document.

### 1.2 Implementation categories and evidence

Each implemented part has one primary implementation category below. That category describes its principal implemented behavior, not an unqualified claim about every line of code. Paper rules, author-supported details and local departures within the part are identified explicitly.

1. **Exact paper recreation.** The named architectural responsibility, rule or equation is implemented as explicitly stated in the article. This classification is limited to that named behavior; it does not establish whole-component, numerical or experimental equivalence.
2. **Authors' code.** The named behavior follows the supplied authors' implementation where the paper leaves operational details unspecified. This means source-supported behavior or tested component equivalence, not necessarily copied source or a verified upstream revision. Explicit differences remain disclosed.
3. **Implementation from scratch with an evidenced design rationale.** The repository supplies a local implementation, framework integration, convention, fallback or correctness fix not established as an exact paper rule or an equivalent author implementation. Its entry states the source constraint, the reason for the choice and the supporting evidence available. This category does not mean that third-party libraries were themselves written from scratch.

A missing capability has no implemented category: it remains **Not implemented**, with its requirement source stated. Naming a paper requirement is not the same as assigning category 1 to an absent implementation. An optional heuristic or helper whose origin or equivalence is not established is not promoted to an exact recreation. Experimental settings are distinguished from active behavior.

For category 3, a reasoned justification is not automatically empirical proof. Counterexamples, control-flow evidence and regression tests are distinguished from design rationales. A rationale alone does not establish numerical or experimental equivalence.

### 1.3 Evidence references

Evidence abbreviations: **Paper** means the linked article and named section; **AF** = [author Fog](https://github.com/ssam18/zta-federated-learning/blob/main/src/agentic/fog_agent.py); **AM** = [author model](https://github.com/ssam18/zta-federated-learning/blob/main/src/models/cnn_lstm.py); **AX** = [author attribution/metrics](https://github.com/ssam18/zta-federated-learning/blob/main/src/utils/metrics.py); **AT** = [author training](https://github.com/ssam18/zta-federated-learning/blob/main/src/security/adversarial.py); **AE** = [author Edge](https://github.com/ssam18/zta-federated-learning/blob/main/src/agentic/edge_agent.py); **AD** = [author loaders](https://github.com/ssam18/zta-federated-learning/blob/main/src/utils/data_loader.py); **AC** = [author configuration](https://github.com/ssam18/zta-federated-learning/blob/main/src/agentic/config.py); **AB** = [author backdoor](https://github.com/ssam18/zta-federated-learning/blob/main/src/security/backdoor.py); **AA** = [author attestation](https://github.com/ssam18/zta-federated-learning/blob/main/src/security/attestation.py). The links point to the authors’ public repository on `main`, which can change. The report’s author-code comparisons were made against a supplied snapshot that is not included in this checkout. The linked public files have not been verified as an exact match for that snapshot; these links provide access to the corresponding files, not proof of snapshot equivalence.

## 2. Architecture and supporting parts

1. Edge-Fog-Cloud hierarchy and Flower execution
2. Live network perception
3. Offline preprocessing and versioned data artifacts
4. Local data partitions and label/quantity skew
5. Protocol/layer feature skew
6. CNN-LSTM model, input interpretation and software framework
7. Model distribution, transport compression and payload validation
8. Dynamic INT8 Edge inference and telemetry
9. TLS/mTLS communication integration
10. Pre-round admission and TPM identity/integrity verification
11. Attestation exchanges across the Flower round lifecycle
12. Fog-public-key-encrypted token envelope and key management
13. Manufacturer-signed firmware and PCR update manifests
14. Trust scoring, rewards and quarantine
15. Five-valid-round rehabilitation and readmission
16. GradientSHAP attribution and feature-importance vector construction
17. Relative-L2 SHAP stability scoring
18. Cohort stability filtering
19. Parallel SHAP evaluation across agents
20. Stability-weighted aggregation and zero-weight fallback
21. Cloud validation and global-model rollback
22. Local Adam training and clean/adversarial batch composition
23. FGSM/PGD attack generation and evasion evaluation
24. Byzantine role allocation and attacker-budget validation
25. Label-flipping attacks and malicious-client training
26. Gradient-manipulation and parameter-delta scaling attacks
27. Backdoor poisoning and attack-success evaluation
28. SHAP-aware constrained poisoning and unconstrained control
29. Validation, global evaluation and metric conventions
30. Failure accounting, missing quotes and cryptographic FAR/FRR
31. Cumulative drift tracking
32. Heterogeneity-adaptive thresholds
33. Differential privacy
34. Alternative SHAP scaling schemes
35. Blockchain auditing
36. Hardware TPM/TrustZone validation

## 3. Purpose and implementation of each part

### 3.1. Edge-Fog-Cloud hierarchy and Flower execution

**Purpose.** Coordinate local Edge training, regional Fog authentication and aggregation, and Cloud aggregation and redistribution of the next global model.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The Flower process adapters, TCP bridge and simulation mapping are local integrations. The Edge-Fog-Cloud responsibilities themselves reproduce Paper IV; the process topology is not claimed to be specified by the paper.

**Implementation.** A deployed Fog is a Flower ServerApp for its Edges and a Flower ClientApp for the Cloud. A TCP bridge joins the two processes as one logical Fog. In simulation, the Cloud strategy invokes the same regional processor for virtual Fogs; Fog outputs cross the same encode/decode boundary as deployment. These are framework mappings and implementation placements, not substitute defenses or a new learning architecture.

**Evidence and design rationale.** Paper IV defines the tier responsibilities, while Flower is the thesis implementation framework. Sharing the regional processor preserves the same regional processing path across deployment and simulation. This integration does not establish that the physical process topology is prescribed by the article. Evidence: [Fog strategy](../src/tier_fog/fog_aggregator.py), [deployment Cloud](../src/tier_cloud/cloud_aggregator_deploy.py), [simulation Cloud](../src/tier_cloud/cloud_aggregator_sim.py), [Fog bridge](../src/tier_fog/fog_bridge_server.py), and [bridge client](../src/tier_edge/fog_bridge_client.py).

**Status and limits.** Implemented integration. Virtual Fogs do not reproduce independent machines, physical or network isolation, or deployed network timing. The supplied author snapshot does not establish the complete multi-Fog experiment orchestration.

### 3.2. Live network perception

**Purpose.** Extract network flows, statistics and time series for the intrusion-detection pipeline described in Paper IV.

**Implementation category.** **None - Not implemented.** This is an article capability, not a completed recreation or an equivalent offline integration.

**Implementation and evidence.** Runtime inputs are prepared tabular benchmark rows. The supplied author's `PerceptionModule` is also a pass-through that checks two-dimensional input shape; it does not supply the missing live extraction pipeline.

**Status and limits.** Preparing or loading CSV tensors is not live perception of network traffic. No live extraction mechanism is established by this report.

### 3.3. Offline preprocessing and versioned data artifacts

**Purpose.** Prepare and validate the train, validation, test and per-Edge artifacts consumed by the federated implementation.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The full offline/runtime pipeline and compatibility safeguards are a local implementation. The named MinMax/PCA/SMOTE responsibilities come from Paper VI-A, and conditional PCA follows AD. Exact equivalence of the complete author split/SMOTE runner is not established.

**Implementation.** The offline builder uses train-fitted MinMax, PCA only when input width exceeds 40, and training-only SMOTE. Narrower inputs are padded. Stratified 70/15/15 train/validation/test splits remain. Runtime loaders validate manifests and load the prepared tensors rather than fitting preprocessing during training.

**Evidence and design rationale.** Paper VI-A names MinMax/PCA/SMOTE; AD supports the conditional PCA choice. Train-only fitting is the documented preparation boundary. Version checks prevent evaluating incompatible checkpoints across incompatible feature pipelines: offline evaluation rejects checkpoints whose pipeline version is missing or different. Evidence: [offline builder](../scripts/setup/build_artifacts.py), [data loader](../src/shared/data/data_loader.py), [artifact manifests](../src/shared/data/artifact_manifest.py).

**Status and limits.** Implemented, but exact full-data author transformations/splits remain unverified because the complete author split/SMOTE runner is unavailable. This is centralized benchmark preparation, not live perception. Full datasets must be rebuilt separately. The attack-domain behavior is described under FGSM/PGD.

### 3.4. Local data partitions and label/quantity skew

**Purpose.** Supply each Edge with local training data and represent the existing label and quantity heterogeneity.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The full artifact partitioning and bounded power-distributed target sizes are local choices whose complete author equivalence is unverified. Local training follows the paper's architecture, and label-based partitioning has AD support.

**Implementation.** Clients consume prebuilt Edge partitions and train locally. Label subsets and bounded power-distributed target sizes are generated offline. Label and quantity skew are retained; protocol/layer feature skew is a separate missing part.

**Evidence and design rationale.** Paper VI-A requires label, quantity and feature heterogeneity; AD supplies label-based partitioning. The offline builder and runtime data loader implement the declared artifact-based execution path. Retaining supported label/quantity skew does not justify inventing an unsupported layer assignment. The report documents these implementation choices, not an independent proof that the full generated distribution matches the authors' experiment.

**Status and limits.** Local partitions and label/quantity skew are present. Cross-client sample reuse and achievable counts must be reported. This does not establish disjoint physical data collection; smoke-test partitions are not full-data equivalence.

### 3.5. Protocol/layer feature skew

**Purpose.** Represent the protocol- or IIoT-layer feature heterogeneity named in Paper VI-A, beyond label and quantity skew.

**Implementation category.** **None - Not implemented.** The existing partitioning is only partial coverage of the article's heterogeneity setup.

**Implementation and evidence.** Prepared 40-feature tensors have no partitioning rule for IIoT layers. AD supports label-based partitioning but does not resolve layer assignment. Work is deferred until original metadata and a defensible assignment are available.

**Reason for deferral and limits.** Assigning arbitrary PCA columns to protocols would invent a partitioning scheme. Deferral does not close this article gap; label and quantity skew are documented separately under local partitions.

### 3.6. CNN-LSTM model, input interpretation and software framework

**Purpose.** Provide the trainable intrusion-detection model used at the Edge and throughout federated aggregation/evaluation.

**Implementation category.** **2 - Authors' code.** The PyTorch CNN-LSTM class and tabular input interpretation have source-backed, tested component equivalence. This is not an exact recreation of the paper's TensorFlow environment or reported parameter count.

**Implementation.** The class matches AM's computational structure and seeded state, with matching train/eval logits in the tested tabular and sequence cases. AE explicitly requires two-dimensional tabular input; the author input path also uses tabular rows with a singleton convolution axis. Both default 40-feature, 15-class versions contain 414,415 parameters, whereas the article reports about 487K.

**Output classes and sample coverage.** The CNN-LSTM architecture, including the fifteen-output classification head used for Edge-IIoTset, is taken verbatim from the authors' public implementation: `n_classes=15` and `fc3 = nn.Linear(128, n_classes)`. Here “verbatim” refers to the layer architecture and output configuration, not byte-for-byte identity of the complete Python file. The public [experiment runner](https://github.com/ssam18/zta-federated-learning/blob/main/scripts/run_experiments.py) configures fifteen Edge classes, and the [agentic runner](https://github.com/ssam18/zta-federated-learning/blob/main/scripts/run_agentic_experiment.py) loads the bundled sample while explicitly setting `n_classes = 15`.

The bundled Edge CSV contains 10,000 rows and only ten observed classes (IDs 0–9); IDs 10–14 are absent. On inspection on 2026-11-05, the upstream sample and this repository's sample were byte-for-byte identical. Thus the sample experiment retains **15 outputs for 10 observed classes**, matching the authors' public setup rather than resizing the network. Paper VI-A describes the broader Edge dataset with fourteen attack types; normal traffic supplies the fifteenth class. Missing sample classes cannot be learned or evaluated as positive classes from this sample. AUC therefore uses an explicitly documented observed-test-class averaging convention (Section 3.29), without altering the model architecture or softmax probabilities. This does not establish reproduction of the paper's full-data results.

**Evidence and design rationale.** AM supplies the model comparison; AE supplies the input requirement. Rewriting the network or adding temporal windows would introduce a design not used by the provided input path, so the verified class and input interpretation are retained.

**Status and limits.** The component is implemented, but singleton-axis limitations and the parameter-count discrepancy remain. Both implementations use PyTorch, a shared discrepancy with the paper's TensorFlow 2.13 environment in VI-A. Captum GradientSHAP versus the author's handwritten integrated gradients is a separate algorithmic distinction, addressed under attribution.

### 3.7. Model distribution, transport compression and payload validation

**Purpose.** Exchange model updates and distribute models through Flower while preserving valid payloads across the transport boundary.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** Per-tensor encoding/reconstruction, payload validation and the scalar-buffer fix are repository integration choices. Model distribution is a paper responsibility, but the paper's reported 8-bit model does not establish this exact transport encoding.

**Implementation.** Flower parameters carry per-tensor encoded arrays reconstructed before PyTorch execution. Update transport supports 8-bit encoding, 16-bit conversion and 32-bit transfer, selected through `quantization_bits`. Fog outputs cross the same encoding/decoding boundary in deployment and simulation; Cloud rollback validation is performed after transport encoding/decoding. Separate CPU INT8 execution is documented under quantized inference, not treated as a consequence of transport encoding alone.

**Evidence and design rationale.** Valid payload handling is an explicit correctness requirement. 8-bit scalar buffers remain zero-dimensional NumPy arrays, allowing valid CNN-LSTM payloads to pass the existing validator. Regional integration is exercised at both 8 and 32 bits in the reported tests. Evidence: [compression](../src/shared/network/compression.py) and [regional integration tests](../scripts/tests/test_pipeline_integration.py).

**Status and limits.** Implemented integration. Transport encoding is not a quantized inference engine. Array bytes, serialized tensor payload bytes and serialized model-state size are distinct measurements; none alone establishes the article's 475KB model claim. Malformed payloads and missing updates are separately accounted for under failure handling.

### 3.8. Dynamic INT8 Edge inference and telemetry

**Purpose.** Run an 8-bit-capable Edge inference path and measure its behavior separately from floating-point training and compressed transport.

**Implementation category.** **2 - Authors' code.** The central inference mechanism is AM's post-training dynamic quantization. The isolated copies, real client evaluation, deterministic probes and telemetry are explicitly local integration/measurement additions, not a claim of complete author-code or paper-size equivalence.

**Implementation.** Dynamic INT8 Linear/LSTM CPU inference is integrated on a separate trained-model copy. Convolutions remain floating point; training, aggregation and SHAP remain floating point. Both simulation and deployment client adapters expose real evaluation. When `edge_inference_int8` is enabled, completed training rounds measure FP32/INT8 on a deterministic held-out validation probe and persist the metrics through Fog telemetry. `edge_inference_samples` controls the maximum probe size. Explicit client evaluation also uses the configured inference mode.

The measurements distinguish loss, accuracy, serialized state bytes, forward latency, array bytes and serialized tensor upload bytes. Local training timing excludes inference evaluation. The [recorded inference probe](../scripts/tests/reports/architecture_fidelity_probe.json) measured **694,044 serialized state bytes**, not the reported 475KB.

**Evidence and design rationale.** AM supports the quantization mechanism. Isolating the inference copy preserves the floating-point training/aggregation/attribution path while evaluating actual quantized execution. Client evaluation reports measured accuracy. Evidence: [deployment client](../src/entrypoints/deploy_client.py), [simulation client](../src/entrypoints/sim_client.py), [inference implementation](../src/tier_edge/inference.py), and [regional telemetry](../src/tier_fog/round_processing.py).

**Status and limits.** Implemented and tested, with an empirical gap remaining. No integer training or quantization-aware training (QAT) is claimed; the paper specifies quantization but does not establish integer-only training. The 475KB model size and target-hardware speedup remain unestablished. Per-Edge metrics are client-reported telemetry, not attestation-bound facts. The implemented deployment path still requires deployment validation. Hardware TPM/TrustZone validation is a separate article future-work item outside the implementation scope.

### 3.9. TLS/mTLS communication integration

**Purpose.** Supply the secure Edge-Fog communication responsibility in Paper IV through the deployed infrastructure.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** TLS/mTLS is supplied through repository-specific infrastructure/proxy integration, not the author's logging stub. It carries out the named paper responsibility without establishing complete transport-security equivalence.

**Implementation.** TLS/mTLS is provided outside `src/` by infrastructure/proxy integration. The existing working TLS and TPM integrations are retained. Authenticated poisoning participants remain permitted to receive models until the paper's trust/filter rules exclude them; the enrolled Byzantine threat model is not replaced by blanket exclusion.

**Evidence and design rationale.** Paper IV/V-A requires the communication and attestation responsibilities. AE's `SecureChannel` only logs a send, while AA is an HMAC attestation simulation. Replacing working integrations with these stubs would reduce deployment coverage. This is the source-supported rationale for keeping the integrations, not proof of a completed distributed security validation.

**Status and limits.** Integrated. Internal service and IPC routes rely on the controlled host/network boundary; transport integration does not establish complete end-to-end authentication. Real distributed TPM/TLS execution still requires deployment validation. Container hardening is outside scope. TLS is not the missing application-layer encrypted-token envelope.

### 3.10. Pre-round admission and TPM identity/integrity verification

**Purpose.** Verify an enrolled Edge's identity, freshness, reference PCR and reputation before admitting it to a round.

**Implementation category.** **1 - Exact paper recreation.** This classification covers fresh pre-round verification and the named identity/signature/freshness/PCR/reputation and enrollment/quarantine checks. The local TPM/gatekeeper integration is not copied from AA's HMAC simulation and does not establish hardware-security or whole-workflow equivalence.

**Implementation.** Every selected Edge receives a fresh admission challenge before receiving the round's parameters. Enrolled key/identity and PCR checks, signature verification, nonce/timestamp verification and quarantine gating exist. Validly enrolled poisoning participants remain eligible until the stated trust or filtering rules exclude them.

**Evidence and design rationale.** Paper IV/V-A requires these mechanisms and includes enrolled Byzantine participants. AA's HMAC simulation does not replace working TPM quote verification. AF's verified-quote mask bypasses its TrustDB quarantine status; this implementation enforces the specified quarantine gate. Evidence: [admission](../src/tier_fog/client_instructions.py), [TPM prover](../src/tier_edge/tpm_attestation.py), [verifier](../src/tier_fog/tpm_verifier.py), and [gatekeeper](../src/tier_fog/gatekeeper.py).

**Status and limits.** Implemented, subject to deployment validation. Software TPMs and trusted provisioning do not establish hardware security. Attestation does not bind the complete training computation or prove that updates are benign. Trust remains process-local. These checks do not implement the separate token encryption or manufacturer update workflow.

### 3.11. Attestation exchanges across the Flower round lifecycle

**Purpose.** Carry admission evidence and attested uploads through Flower's separate lifecycle operations without counting multiple exchanges as multiple recovery rounds.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** Admission and upload quotes are mapped to separate Flower lifecycle operations. The paper requires their responsibilities but does not prescribe an exact Flower exchange count.

**Implementation.** Flower `get_properties` performs admission. Admitted `fit` results carry a separate challenged quote. Rejected nodes currently return attestation-only `fit` results without parameters and also perform an additional quote. Admission and returned-update quotes are distinct physical exchanges.

**Evidence and design rationale.** The two exchanges serve the admission and upload responsibilities at their corresponding framework boundaries. Duplicate-round protection prevents them from accelerating rehabilitation: they are not two consecutive valid recovery rounds. Evidence: [admission instructions](../src/tier_fog/client_instructions.py), [TrustDB](../src/tier_fog/trust_db.py), and [recovery tests](../scripts/tests/test_round_admission.py).

**Status and limits.** Implemented integration, with exact exchange equivalence unclaimed. Neither exactly one nor exactly two physical exchanges is established as an explicit article requirement. Extra exchange cost and event-level FAR/FRR denominators differ from a one-quote interpretation. Missing/crashed/rejected outcomes and token processing are described under failure accounting.

### 3.12. Fog-public-key-encrypted token envelope and key management

**Purpose.** Provide the separate application-layer token encryption/decryption using the Fog public key named in Paper V-A.

**Implementation category.** **None - Not implemented.** This remains an explicit article gap.

**Implementation and evidence.** Neither the repository's described behavior nor the supplied modules implement the separate envelope. Concrete key provisioning and key-management workflows remain unspecified. The implementation decision is not to invent this subsystem in the current scope.

**Status and limits.** Neither a stub nor TLS alone closes the envelope requirement. A scoped thesis may report the omission, but cannot claim every security detail is implemented. The related firmware-update omission is documented separately under signed updates.

### 3.13. Manufacturer-signed firmware and PCR update manifests

**Purpose.** Supply the manufacturer-signed firmware/PCR update-manifest workflow named in Paper V-A.

**Implementation category.** **None - Not implemented.** This remains an explicit article gap, not an integrated equivalent.

**Implementation and evidence.** Initial ledger generation is trusted setup. The supplied modules do not implement the signed update subsystem, and concrete update workflows remain unspecified. The selected scope does not invent a firmware-update subsystem.

**Status and limits.** Trusted initial provisioning does not establish a manufacturer-authenticated update flow. Deferral cannot justify a claim that every security detail is implemented.

### 3.14. Trust scoring, rewards and quarantine

**Purpose.** Maintain participant reputation, reward accepted stable behavior and exclude participants whose trust falls below the article's threshold.

**Implementation category.** **1 - Exact paper recreation.** This classification covers the named trust constants, penalties, quarantine threshold and strict above-mean reward rule in Paper V-A. Missing-quote treatment is an additional local convention, not an exact paper requirement.

**Implementation.** Initial trust is 0.7. A reward adds 0.02, capped at 1. Trust is halved on attestation/filter failure, and participants are quarantined below 0.6. Rewards require an accepted stability score strictly above the cohort mean (`s > mean`), not equality and not a median comparison. Exact ties receive no reward. The same above-mean reward rule applies when eligible updates are aggregated using the uniform fallback.

**Evidence and design rationale.** Paper V-A and AF support the comparisons. Admission enforces the quarantine gate. Uniform fallback does not suppress rewards for accepted updates with above-mean stability. Evidence: [TrustDB](../src/tier_fog/trust_db.py), [regional processing](../src/tier_fog/round_processing.py), and [SHAP verifier](../src/tier_fog/shap_verifier.py).

**Status and limits.** Implemented, with missing-quote handling explicitly qualified. Missing quotes are recorded separately from verified failures and handled as an operational interpretation. Trust is process-local, and valid quotes do not guarantee honest training. Recovery is described under readmission. Small-cohort blindness and collusion sensitivity are not repaired by tuning the reward/filter rules.

### 3.15. Five-valid-round rehabilitation and readmission

**Purpose.** Allow quarantined participants to rejoin only after satisfying the article's consecutive-attestation recovery rule.

**Implementation category.** **1 - Exact paper recreation.** The central rule is five consecutive valid attestation rounds and reset to trust 0.65 before rejoining. The preflight placement and duplicate-round guard are local framework details that enforce that rule rather than create a second recovery policy.

**Implementation.** Readmission occurs on the fifth consecutive valid round's preflight quote. A node quarantined in round 1 that produces successful quotes in rounds 2-6 may train in round 6. An interruption breaks continuity. At most one recovery success is counted per round, regardless of the number of physical quotes.

**Evidence and design rationale.** Counting rounds rather than lifecycle messages preserves the stated consecutive-round rule in Flower. Evidence: [TrustDB](../src/tier_fog/trust_db.py), [admission](../src/tier_fog/client_instructions.py), and [recovery tests](../scripts/tests/test_round_admission.py).

**Status and limits.** Implemented, with regression coverage for fifth-valid-round readmission. This does not establish real distributed TPM/TLS validation or prove honest training by readmitted nodes.

### 3.16. GradientSHAP attribution and feature-importance vector construction

**Purpose.** Construct the model feature-importance vectors used by the stability comparison.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The complete vector-construction and state-safety procedure contains local conventions. Its GradientSHAP method follows the paper; the global reference and true-label targets have AF/AX support. Mean-absolute reduction and the choice of explained-sample count are not established as exact paper or author-code behavior.

**Implementation.** The repository uses Captum GradientSHAP. It targets true-label logits and averages absolute attributions over explained examples to construct a feature vector. `shap_explain_count` controls the number of explained samples, while `shap_val_samples` limits the Fog validation/background subset; available data can reduce either count. These are separate controls. The reference, reduction and sampling choices must be recorded for each experiment.

**Evidence and design rationale.** The paper names GradientSHAP, while AX uses handwritten zero-baseline integrated gradients. These are substantive algorithmic differences, so the explicit paper method takes precedence. The paper requires a feature-importance vector but does not fully specify how examples/classes are reduced. Retaining the existing explicit mean-absolute reduction avoids silently introducing another estimator, but the report supplies no proof that this reduction or a particular explained-sample count is author-equivalent.

Separately, AX's train-mode attribution updates BatchNorm state during measurement. Preserving model state avoids that side effect. Evidence: [attribution and stability](../src/shared/utils/metrics.py) and AF/AX.

**Status and limits.** Implemented with explicit conventions; full equivalence is unverified. The author computes mean per-example distances, whereas this implementation constructs the mean-absolute feature vector first. Sign and example-level information are lost; coverage depends on sampling. Changing the explained-sample count changes the experiment and is not a neutral fix. Attribution is approximate and expensive. The implementation does not reproduce the supplied SHAP outputs or RBF helper, and matching the named method or formula supplies no robustness guarantee. The denominator and score are described under stability.

### 3.17. Relative-L2 SHAP stability scoring

**Purpose.** Measure the difference between a participant's feature-importance vector and the reference vector.

**Implementation category.** **1 - Exact paper recreation.** The raw relative-L2 equation and actual reference-attribution norm reproduce the named paper rule. The numerical epsilon of 10^-6 is an AC-supported convention, not a value specified by the paper. Vector construction remains separately qualified.

**Implementation.** Once the vectors have been constructed, the implemented score is:

```text
s_i = 1 - ||phi_i - phi_ref||_2 / (||phi_ref||_2 + 10^-6)
```

The score is not clipped before filtering. The paper specifies epsilon without a numerical value; `10^-6` is a source-supported convention, not an additional constant transcribed from the article.

**Evidence and design rationale.** Paper V-B/VI-A supplies the equation. AF substitutes median client distance for the denominator, a substantive replacement that is not adopted. The implementation preserves the actual reference norm and raw scores instead. Evidence: [attribution and stability](../src/shared/utils/metrics.py), AF/AX and AC.

**Status and limits.** Direct rule for the named equation, with the epsilon convention disclosed. The author's attribution calculation and normalization differ from both the article and this implementation. Raw stability can be negative; its treatment during aggregation is a separate weighting convention. No robustness guarantee follows merely from equation fidelity.

### 3.18. Cohort stability filtering

**Purpose.** Reject participant updates whose stability falls below the article's cohort-based cutoff.

**Implementation category.** **1 - Exact paper recreation.** The strict mean-minus-two-standard-deviations comparison reproduces the paper rule. Population standard deviation is an AF-supported convention, not an extra specification transcribed from the paper.

**Implementation.** Raw finite scores enter the cohort mean and population standard deviation. The implemented decision rule is:

```text
reject i <=> s_i < mu_s - 2 * sigma_s
```

Equality passes. No MAD estimator, scale floor, positivity rejection or additional minimum-cohort rule is added. Accepted scores receive rewards only under the separate trust reward conditions.

**Evidence and design rationale.** Paper V-A/B and AF support the comparisons; AF uses population standard deviation. AC's small-scale preset changes `k` to 3 and rollback to 0.5, so that preset is not selected for the thesis baseline. Evidence: [SHAP verifier](../src/tier_fog/shap_verifier.py).

**Status and limits.** Implemented. Small-cohort blindness and collusion sensitivity remain. The theorem's honest-population variance and the observable cohort statistic are not asserted to be interchangeable. Thresholds are not tuned merely to make small samples look robust or improve reported results.

### 3.19. Parallel SHAP evaluation across agents

**Purpose.** Provide the parallel SHAP evaluation across ten agents described in Paper VI-A.

**Implementation category.** **None - Not implemented for the parallel capability.** The existing serialized execution is a local implementation choice (category 3), justified by RNG safety; it is not relabeled as ten-agent parallel evaluation.

**Implementation and evidence.** Within a Fog process, attribution is serialized to keep random-number-generator use safe. Independent deployed Fogs can still run concurrently. RNG safety motivates serialized execution; it does not establish equivalent parallel throughput.

**Status and limits.** Cross-Fog concurrency is not ten-agent parallel SHAP within one Fog. The missing parallel capability remains a gap.

### 3.20. Stability-weighted aggregation and zero-weight fallback

**Purpose.** Combine accepted updates according to stability, validation accuracy and data size, with defined behavior when no usable weight remains.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The full policy combines the paper's product for nonnegative scores, AF's nonnegative-weight convention and uniform fallback, and raw-score filtering. Forwarding the fallback aggregate with a positive regional weight is a local Flower integration convention. Neither the complete signed paper equation nor the complete author scoring pipeline is reproduced verbatim.

**Implementation.** The implemented unnormalized weight is:

```text
w_i = max(0, s_i) * acc_i * sqrt(n_i)
```

Scores are not clipped before the paper filter. AF also clamps weights nonnegative, but additionally clamps scores before filtering; that additional score clipping is not imported here. If total usable weight is below `10^-12`, eligible surviving updates are averaged uniformly, following AF. Invalid and rejected updates remain excluded. Above-mean rewards remain applicable. With no eligible updates, no regional aggregate is forwarded. A fallback aggregate is forwarded with regional weight 1, the sum of its normalized uniform weights, so Cloud processes it through normal aggregation and rollback. Diagnostics record the raw weight total and whether uniform fallback was used.

**Evidence and design rationale.** Paper V-B supplies the product on nonnegative scores. `AF._weighted_aggregate` supplies the nonnegative-weight convention and uniform fallback where the proportional-weight formula is undefined. The fallback follows the authors' operational choice where the paper supplies no explicit rule. Regional weight 1 allows that aggregate to participate in the existing Cloud path; this forwarding convention is not an explicit paper rule. Evidence: [SHAP verifier](../src/tier_fog/shap_verifier.py) and the regional processing implementation.

**Status and limits.** Implemented with the authors' uniform fallback and a local regional forwarding convention. Negative-score updates can pass the filter and receive above-mean rewards. They contribute zero weight during ordinary weighted aggregation but participate equally with other survivors during uniform fallback. No additional rejection threshold is introduced. The fallback remains subject to Cloud validation and rollback.

### 3.21. Cloud validation and global-model rollback

**Purpose.** Reject a sufficiently degraded candidate global model and retain the previous accepted/broadcast model.

**Implementation category.** **1 - Exact paper recreation.** The named rule is rollback at 80% or less for the global aggregate. The previous-accepted-model reference is a fixed local interpretation, and Cloud placement follows the implemented hierarchy; these qualifications remain explicit rather than implying complete author-code equivalence.

**Implementation.** With `rollback_threshold = 0.80`, the candidate is compared with the previous accepted/broadcast model after global aggregation and after transport encoding/decoding:

```text
rollback <=> candidate_accuracy <= 0.80 * previous_accepted_accuracy
```

Exactly 80% rolls back. The previous accepted model is a fixed implementation rule, not a YAML selector; only the rollback threshold is parameterized.

**Evidence and design rationale.** The article says 80% or lower and refers to the global model. AF instead uses strict `<` inside its Fog object. The article's boundary and the implemented hierarchy determine the choice here. Evidence: [global state](../src/tier_cloud/model_state.py).

**Status and limits.** Implemented. Relative rollback can permit cumulative decline and targeted damage. No best-ever reference or cumulative defense is added. Cloud validation supplies this decision; test reporting is handled separately under evaluation.

### 3.22. Local Adam training and clean/adversarial batch composition

**Purpose.** Train each Edge's local intrusion-detection model, including the implemented clean/adversarial augmentation path.

**Implementation category.** **2 - Authors' code.** The operative per-minibatch selection, gradient clipping and combined clean/adversarial pass follow AT. Adam and the training responsibility are named by the paper. State-safe attack generation is a local correctness difference, and exact dataset-level 70/30 membership is not claimed.

**Implementation.** Both trainers select the adversarial fraction per minibatch rather than maintaining permanent dataset-level adversarial membership. AT uses `int(batch_size * adv_ratio)` and `clip_grad_norm_` at 1.0. Approximately 30% adversarial examples per minibatch is the selected source-supported behavior where that ratio is enabled. Rounding makes the ratio approximate, and shuffled membership changes.

The clean and adversarial subsets are combined for a single training forward/loss pass. BatchNorm therefore observes the mixed batch.

**Evidence and design rationale.** Paper V-C states 70/30 subsets but does not specify lifetime membership across epochs. AT supplies the batching, clipping and combined-pass behavior. Focused checks verify mixed-batch statistics and real FGSM/PGD training, not improved accuracy. Evidence: [trainer](../src/tier_edge/local_ids.py), [adversarial routines](../src/shared/security/adversarial_math.py), and AT.

**Status and limits.** Implemented with shared configurable augmentation for benign and label-flip clients. Both client adapters forward `robustness_eval_attack` and the legacy-named `benign_adv_ratio`, `benign_eps`, `benign_alpha` and `benign_n_iter` settings to the trainer. The selector chooses FGSM or PGD; a zero augmentation ratio disables feature perturbation without disabling label replacement. There is no separate fixed malicious-client augmentation budget. A ratio of 0.3 selects the approximate per-minibatch 70/30 composition, subject to integer rounding. Experimental equivalence depends on the recorded run settings, not the configuration values currently selected. State-safe execution avoids accidental model changes, not every stochastic/numerical difference from the author path.

### 3.23. FGSM/PGD attack generation and evasion evaluation

**Purpose.** Generate adversarial examples for local augmentation and the FGSM/PGD evasion experiments named in the article.

**Implementation category.** **2 - Authors' code.** The attack-domain behavior follows AT for the paper's named FGSM/PGD families. Evaluation-mode/input-only gradients, model/gradient preservation and singleton support are local correctness differences, not claims of identical author execution.

**Implementation.** FGSM uses the sign step. PGD clamps its random initialization to the attack-batch range and then projects to the epsilon ball. Global attack bounds are optional, with nullable bounds handled across callers. Attack generation preserves modes, buffers and existing parameter gradients instead of mutating training state. A singleton adversarial subset is supported, unlike the source's train-mode attack path.

**Evidence and design rationale.** AT supplies the attack domain. State-preserving attack generation is a narrow correctness fix, separating attack construction from the intended combined training pass. Regression tests cover state safety and real FGSM/PGD training. Train-mode attribution can also alter BatchNorm state. Evidence: [adversarial routines](../src/shared/security/adversarial_math.py), [state-safety tests](../scripts/tests/test_architecture_fidelity.py), [training tests](../scripts/tests/test_node_training.py), and AT.

**Status and limits.** Paths exist, but paper-scale evasion results are not reproduced. VI-A's specified evaluations are FGSM, PGD-7 and PGD-20 at epsilon 0.05, 0.1, 0.15 and 0.2. Evaluation-mode attack generation differs from author execution and does not remove every stochastic/numerical difference. Raw-network realizability remains unverified. The C&W attack mentioned in Paper III is not implemented.

### 3.24. Byzantine role allocation and attacker-budget validation

**Purpose.** Assign attacker roles across the federation under the article's global Byzantine-fraction assumption.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** Seeded global role allocation and rounded-count validation are locally implemented under the paper's global beta < 0.5 constraint. Exact author placement is unknown; no per-Fog majority restriction is invented.

**Implementation.** The experiment seed shuffles role assignments across the complete federation. There is no guaranteed honest majority within each Fog. Per-role counts use Python `round(total_edges * ratio)` independently. Both the requested ratios and the realized rounded global Byzantine fraction must be below 0.5. Invalid rounded counts are rejected rather than silently reduced.

**Evidence and design rationale.** Paper III states a global budget, not a per-Fog majority guarantee. The [role allocator](../src/shared/security/threat_profiler.py) therefore shuffles global counts across all Edges.

**Status and limits.** Implemented. Individual Fogs can have malicious majorities and fail. Exact author placement remains unavailable without the experiment runners. Results must record the seed and realized composition.

### 3.25. Label-flipping attacks and malicious-client training

**Purpose.** Implement the label-poisoning attack family and its local training path for the article's experiments.

**Implementation category.** **2 - Authors' code.** This classification covers all-class replacement and once-per-round poisoned-label assignment following AE. The Adam training path supports optional feature augmentation through the shared local configuration policy. Shared attack selection and budgets, evaluation-mode gradients, singleton support and loader-batch behavior are explicit local differences; the complete malicious-training path is not claimed to reproduce AE/AC automatically.

**Implementation.** Replacement labels are sampled uniformly from all classes once per local round, before the epoch loop; poisoned labels are frozen across local epochs. The expected changed-label fraction is `p * (C - 1) / C`, because sampling can select the original class. The label-flip path uses Adam and receives the same configurable FGSM/PGD selector and augmentation budget as benign training. With zero augmentation it performs ordinary training on the prepared replacement labels; with positive augmentation it uses the selected feature attack. Label poisoning and feature augmentation are separate operations.

**Evidence and design rationale.** AE supports the frozen replacement labels and author malicious-training behavior. The repository's shared configuration policy is established by both client adapters and the trainer, rather than inferred from the settings currently selected. Evidence: [deployment client](../src/entrypoints/deploy_client.py), [simulation client](../src/entrypoints/sim_client.py), [trainer](../src/tier_edge/local_ids.py), [attack routines](../src/shared/security/adversarial_math.py), and [training-configuration tests](../scripts/tests/test_training_configuration.py). AE/AC provide a comparison reference, not fixed parameters imposed by this implementation.

**Status and limits.** Label sampling is author-aligned; training augmentation is configurable and does not automatically reproduce the author's PGD setup. Exact full experiments remain unverified without the complete runner/configuration evidence. Paper VI-A's cases use `beta = 0.1, 0.2, 0.3` and `p_flip = 0.2, 0.5, 1.0`. SHAP-aware label poisoning uses the separate constrained solver described under SHAP-aware poisoning. Its loss definition, attacker knowledge and numerical settings remain explicit implementation choices.

### 3.26. Gradient-manipulation and parameter-delta scaling attacks

**Purpose.** Realize the declared gradient-manipulation experiment by scaling the trained local parameter update.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The cumulative trainable-parameter-delta attack is locally specified. The paper names the family, but the exact author gradient-attack runner is absent.

**Implementation.** After the last local epoch, trainable parameters use:

```text
theta_attack = theta_ref + alpha * (theta_local - theta_ref)
```

Trained BatchNorm buffers remain valid rather than multiplying all state tensors. Paper VI-A's experiment magnitudes are -3, -1, 3 and 5, with alpha = 3 in the reported comparison.

**Evidence and design rationale.** Multiplying all state tensors can produce negative variance; limiting scaling to the cumulative trainable-parameter delta avoids that invalid-buffer consequence while retaining the declared manipulation experiment. Evidence: [attack routines](../src/shared/security/adversarial_math.py) and [state-safety tests](../scripts/tests/test_architecture_fidelity.py). This correctness rationale does not establish exact author-runner equivalence.

**Status and limits.** Implemented convention. Attack semantics are specified locally. Even alpha = 0 need not restore identical predictions because buffers differ. No exact author attack-strength claim follows.

### 3.27. Backdoor poisoning and attack-success evaluation

**Purpose.** Implant the documented trigger during local poisoning and evaluate attack success on eligible non-target examples.

**Implementation category.** **2 - Authors' code.** Trigger application and all-row sampling follow AB. The author trigger recipe and Paper VI-A's 10% local-poisoning scenario are supported through configurable parameters. Non-target-only ASR and inference/singleton safeguards are local measurement conventions. Complete scenario reproduction is not claimed.

**Implementation.** The implementation samples `floor(rate * N)` rows from the entire local partition and applies the selected trigger features, value and target class. The author recipe sets the last three features to 1.5 and uses target class 0; Paper VI-A specifies a 10% local poisoning rate. These scenario values do not define the limits of the configurable implementation. `backdoor_ratio` controls the fraction of attacking nodes separately from `backdoor_poison_fraction`, which controls poisoned rows within each attacking node. ASR is evaluated only on non-target examples.

**Evidence and design rationale.** AB supplies all-row sampling and the trigger recipe; Paper VI-A supplies the 10% scenario. Sampling the training partition is distinct from excluding already-target examples when measuring whether an attack changes a non-target prediction. Evidence: [backdoor routines](../src/shared/security/backdoor_math.py), [deployment client](../src/entrypoints/deploy_client.py), [simulation client](../src/entrypoints/sim_client.py), and AB.

**Status and limits.** Author sampling aligned. Already-target training rows can receive a trigger without changing their labels; tiny partitions can select zero rows. Architectural support is independent of the currently selected attacker fraction. The supported mechanism does not establish reproduction of the paper's `beta = 0.2` experiment or full scenario equivalence.

### 3.28. SHAP-aware constrained poisoning and unconstrained control

**Purpose.** Implement the SHAP-constrained poisoning scenario and unconstrained comparison reported in Paper VIII-B.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** The implementation follows the stated poisoning-objective and raw SHAP-distance constraint structure. The loss definitions, numerical solver, attacker knowledge and optimization settings are explicit local choices because the complete author attack runner is unavailable.

**Implementation.** Starting from the incoming global model, the attacker minimizes a poisoning loss subject to:

`||phi(candidate) - phi(reference)||_2 < tau`

The label-flip objective minimizes cross-entropy against labels replaced once per round. The alternative `gradient_manipulation` selection minimizes negative clean-label cross-entropy; it does not apply the separate parameter-delta scaling attack. This solver uses no FGSM/PGD augmentation.

Clipped gradient descent proposes a parameter update for each minibatch. Backtracking halves the proposed step until the encoded-and-decoded candidate improves that minibatch’s poisoning loss and satisfies the constraint. Otherwise, the previous feasible iterate is retained. BatchNorm buffers remain frozen. Local epochs, batch size, learning rate, gradient clipping and the backtracking limit define the optimization budget.

Attribution uses the owning Fog’s validation sample, clean labels and round-specific attribution seed, with the repository’s GradientSHAP and mean-absolute vector construction. Access to that sample and seed is an explicit attacker-knowledge assumption. The test split is not used by the attack.

`shap_aware_tau` supplies the positive raw-distance bound and must be explicitly set for constrained runs. `shap_aware_constrained: false` runs the same poisoning optimizer without the feasibility test. The final transmitted payload is checked again before return. Diagnostics record the constraint distance, objective, settings, accepted/rejected steps and initial/final loss. They distinguish no-progress outcomes from accepted optimization steps and are persisted as client-reported telemetry.

**Evidence and design rationale.** Paper VIII-B supplies the constrained objective and motivates the unconstrained comparison. Feasible backtracking implements that structure without a normalized stability stopping target or post-training blending loop. Transport-aware checks evaluate the model actually submitted to the Fog. The solver and measurement conventions are not claimed to be the authors’ exact implementation. Evidence: the [SHAP-aware solver](../src/shared/security/shap_aware.py), [shared Edge trainer](../src/tier_edge/local_ids.py), [regional telemetry](../src/tier_fog/round_processing.py) and [constrained-attack regression tests](../scripts/tests/test_shap_aware_attack.py).

**Status and limits.** Implemented for constrained and matched unconstrained experiments. Tests verify numerical feasibility, transport handling, objective improvement on controlled examples and integrated execution. They do not establish a global optimum, guaranteed poisoning effectiveness, Fog acceptance or reproduction of the reported accuracy degradation. Exact experimental equivalence remains unverified because the paper does not fully specify the attack loss, threshold, optimizer, budget or attacker data access. GradientSHAP remains approximate, and its vector-reduction convention is separately disclosed.

### 3.29. Validation, global evaluation and metric conventions

**Purpose.** Measure update quality, support rollback and report model/security performance without changing the model being measured.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** Evaluator placement, inference-mode measurement, singleton inclusion, null undefined results and real client telemetry are local integrations/conventions. Macro-F1 follows AX's class-inclusion convention. These mechanisms carry out the paper's evaluation responsibilities without claiming complete author-evaluator equivalence.

**Implementation.** Fog validation supplies SHAP and update accuracy. Cloud validation supplies rollback. The Cloud evaluator uses the test split for reported metrics; Fogs do not instantiate `GlobalEvaluator`. Real client evaluation reports the FP32/INT8 measures described under quantized inference.

Metrics are computed in inference mode, singleton chunks are included, and undefined results are represented as `null`. Following AX, macro-F1 excludes classes absent from both ground truth and predictions; prediction-only classes and ground-truth classes with no predictions remain included. ASR eligibility is limited to non-target examples as described under backdoor evaluation.

ROC-AUC uses the unweighted mean of binary one-vs-rest AUCs for classes with positive and negative test examples, keeping softmax over all configured outputs without renormalizing observed columns. A fifteen-output model evaluated on the ten-class Edge sample therefore reports ten-class observed macro AUC, plus the evaluated IDs, class counts and per-class values. Empty/single-class populations remain undefined; invalid inputs or calculation failures raise errors. This is a local metric convention: Paper IX reports AUC without defining its averaging, and the inspected public author metrics/runners do not implement AUC. See [AUC definition and output fields](metrics.md#auc-definition-and-undefined-values).

**Evidence and design rationale.** AB evaluates in train mode and skips singleton chunks, which can change the model or numerator. AX's train-mode attribution updates BatchNorm state during measurement. These findings support state-preserving evaluation and complete eligible-sample accounting. Macro-F1 follows AX by excluding classes absent from both truth and predictions. Evidence: [Cloud evaluator](../src/tier_cloud/global_evaluator.py), [metrics implementation](../src/shared/utils/metrics.py), [client inference](../src/tier_edge/inference.py), and AX/AB.

**Status and limits.** Implemented integration/conventions. Direct numerical comparisons require matching class support and metric semantics. Measurement capability and a passing CPU check are not evidence of reproduced accuracy, throughput or attack effectiveness. Per-Edge telemetry is client-reported, not attestation-bound. Cryptographic FAR/FRR exclusions are specified under failure accounting.

### 3.30. Failure accounting, missing quotes and cryptographic FAR/FRR

**Purpose.** Separate verified security outcomes from missing data, execution failures and malformed updates so that failures are not misreported as successful defenses.

**Implementation category.** **3 - Implementation from scratch with an evidenced design rationale.** Explicit event classification, valid-payload checks and cryptographic denominator handling implement parts of the honest-failure-accounting policy. Complete separation of operational verifier errors from authentication failures is not established.

**Implementation.** Regional statuses and outcome records distinguish empty results, reported training crashes, malformed payloads and rejected updates. Missing quotes, including crash reports without a token, are recorded as unattempted attestation events and excluded from cryptographic FAR/FRR denominators. If a crash report includes a token, that token follows normal verification; its outcome can contribute to the denominator when the expected validity is known and verification is not bypassed. A missing update is not evidence that poisoning was detected or neutralized.

**Evidence and design rationale.** Token presence determines whether available attestation evidence is evaluated; a client-provided crash label does not bypass verification. Evidence: [gatekeeper](../src/tier_fog/gatekeeper.py), [regional processing](../src/tier_fog/round_processing.py), and [security metrics](../src/shared/utils/security_metrics.py).

**Status and limits.** Implemented integration with an unresolved operational-error classification concern. The TPM verifier catches exceptions and returns `False`; the gatekeeper can consequently record an attempted authentication rejection. An operational failure, such as an unavailable verification executable, can therefore enter FRR accounting when the participant is labelled valid. This concern follows directly from control flow; its end-to-end operational effect requires a targeted failure-injection check.

Missing quotes do not prove innocent intent; legitimate failures can still exclude an update. Event-level FAR/FRR denominators and the cost of multiple physical quotes differ from a one-quote interpretation. Duplicate lifecycle exchanges cannot accelerate rehabilitation. Published FAR/FRR results are not reproduced merely by supporting their measurement.

### 3.31. Cumulative drift tracking

**Purpose.** Track cumulative drift, a capability identified among the article's extensions/future work in VIII-IX.

**Implementation category.** **None - Outside implementation scope.** This article future-work proposal is retained for completeness and will not be implemented. Its absence is not a missing requirement of the primary architecture.

**Implementation and limits.** The current rollback compares against the previous accepted model and permits cumulative decline; no cumulative defense or best-ever reference is added. The report provides no implemented drift-tracking algorithm or verification evidence. Its absence is retained as a scope boundary, not a claim that the core method already addresses cumulative drift.

### 3.32. Heterogeneity-adaptive thresholds

**Purpose.** Adapt thresholds to heterogeneity, as identified among the article's extensions/future work in VIII-IX.

**Implementation category.** **None - Outside implementation scope.** This article future-work proposal is retained for completeness and will not be implemented. Its absence is not a missing requirement of the primary architecture.

**Implementation and limits.** The current filter retains the fixed paper comparison and disclosed population-statistic convention. Thresholds are not tuned to conceal small-cohort weaknesses or improve reported results. The future capability remains outside the primary method.

### 3.33. Differential privacy

**Purpose.** Provide the differential-privacy capability named in the article's extensions/future work in VIII-IX.

**Implementation category.** **None - Outside implementation scope.** This article future-work proposal is retained for completeness and will not be implemented. Its absence is not a missing requirement of the primary architecture.

**Implementation and limits.** This proposed extension is not silently added to the primary method and is not an implemented security/privacy guarantee of this repository.

### 3.34. Alternative SHAP scaling schemes

**Purpose.** Provide the alternative SHAP scaling schemes named among the article's extensions/future work in VIII-IX.

**Implementation category.** **None - Outside implementation scope.** This article future-work proposal is retained for completeness and will not be implemented. Its absence is not a missing requirement of the primary architecture.

**Implementation and limits.** The primary implementation retains the explicitly described attribution, relative-L2 stability and filtering choices. Proposed alternative schemes are not silently added to the main method.

### 3.35. Blockchain auditing

**Purpose.** Provide the blockchain-auditing capability identified among the article's extensions/future work in VIII-IX.

**Implementation category.** **None - Outside implementation scope.** This article future-work proposal is retained for completeness and will not be implemented. Its absence is not a missing requirement of the primary architecture.

**Implementation and limits.** This remains a future-work/scope exclusion, not an existing property of the trust database, telemetry or attestation integration.

### 3.36. Hardware TPM/TrustZone validation

**Purpose.** Validate the relevant security mechanisms on hardware TPM/TrustZone targets, as identified in the article's extensions/future work in VIII-IX.

**Implementation category.** **None for hardware TPM/TrustZone validation - Outside implementation scope.** This article future-work item is retained for completeness and will not be implemented. The existing software TPM and trusted provisioning do not establish hardware-security evidence.

**Implementation and limits.** Hardware security is not established by software attestation or the CPU test suite. Real distributed TPM/TLS execution and target-hardware measurements remain outstanding; no hardware TPM/TrustZone validation is claimed.
