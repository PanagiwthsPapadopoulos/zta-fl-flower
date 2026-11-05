# ZTA-FL Experimental Run Specification — ZTA-FL Only

> **Scope:** This file specifies **ZTA-FL runs only**. Baseline methods (FedAvg, FedProx, Krum, Trimmed Mean, FLTrust, FLAME, RFA, Adv-FL) are excluded.
>
> **Coverage:** Every ZTA-FL training mentioned anywhere in the paper (tables, figures, and prose) is counted. Configurations that appear only as parameter grids in text (with no reported number) are included separately in §3.
>
> **Seeds:** 5 independent runs per configuration: `42, 123, 456, 789, 1011`.
> Mentioned in: **Page 5, VI.A, Reproducibility ¶1**.
>
> **Paragraph notation:** `¶n` = nth paragraph under the cited section heading, excluding bullets, tables, figures, and captions. If a paragraph spans pages, the page where the paragraph begins is cited.
>
> **Deduplication principle:** A run is only marked as a duplicate when the method, dataset, attack type, parameter values, and reported numeric value all match, and the paper itself identifies them as the same experiment. Every saving is evidenced in §6. No convenience merges.

---

## 1. Global Constants and Shared Setup

| Item            | Value                                                                                          | Mentioned in                                           |
| --------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Seeds           | 42, 123, 456, 789, 1011                                                                        | Page 5, VI.A, Reproducibility ¶1                       |
| Runs per config | 5                                                                                              | Page 4, VI, intro ¶1; Page 5, VI.A, Reproducibility ¶1 |
| Agents          | N = 100                                                                                        | Page 4, VI.A, ¶3                                       |
| Non-IID setup   | Label skew C=3; feature skew; quantity skew 500–5000                                           | Page 4, VI.A, ¶3                                       |
| Global rounds   | 100                                                                                            | Page 5, VI.A, Implementation ¶1                        |
| Local epochs    | 5                                                                                              | Page 5, VI.A, Implementation ¶1                        |
| Batch size      | 128                                                                                            | Page 5, VI.A, Implementation ¶1                        |
| Optimizer       | Adam, η = 0.001                                                                                | Page 5, VI.A, Implementation ¶1                        |
| Model           | CNN-LSTM, 487K params, 8-bit quantized to 475KB                                                | Page 5, VI.A, Implementation ¶1                        |
| ZTA-FL params   | Δt_max = 60s, τ_min = 0.6, 10 fog nodes                                                        | Page 5, VI.A, Implementation ¶1                        |
| SHAP config     | GradientSHAP; 100 background samples; 3.1s/round fog; 40 PCA features; shared validation set   | Page 5, VI.A, SHAP Configuration ¶1                    |
| TrustDB policy  | Init 0.7; +0.02 success; ×0.5 penalty; quarantine <0.6; 5 consecutive attestations; reset 0.65 | Page 3, V.A, TrustDB Policy bullets                    |

**Definitions:**
- **Configuration** — one (dataset, attack type, parameter) tuple. Each row in §2–§5 is one configuration.
- **Run** — one training execution with one seed. Each configuration has 5 runs (seeds 42, 123, 456, 789, 1011).
- **Executed runs** — total seeds across all executed configurations.

---

## 1.5 Execution Scope

This document distinguishes three classes of runs:

| Class | Sections | Count | Description |
|-------|----------|------:|-------------|
| **Executed** | §2 | 80 | Runs this reproduction will execute. |
| **Documented only** | §3, §5 | 215 | Runs the paper reports or parameterizes but this reproduction will not execute. Reasons stated in-line. |
| **Exploratory** | §4a, §4b | 50 | Failure-mode and efficiency runs referenced in discussion but not planned for execution. |

**Total ZTA-FL runs documented:** 345 across all classes. **Total runs to be executed:** 80.

All executed runs use the global constants in §1 and 5 seeds (42, 123, 456, 789, 1011).

---

# 2. ZTA-FL Runs — With Reported Numbers

Every ZTA-FL training that produces a number in the paper, organized by attack type. Each row is a **unique configuration**; sources are listed; duplicates are analyzed in §6.

## 2.1 Clean Data

| Configuration       | Sources                                  | Status   | Runs |
| ------------------- | ---------------------------------------- | -------- | ---: |
| Edge-IIoTset, Clean | Table II; Table V (+All Clean); Figure 5 | Executed |    5 |
| CIC-IDS2017, Clean  | Table II                                 | Executed |    5 |
| UNSW-NB15, Clean    | Table II                                 | Executed |    5 |

**Subtotal: 15 runs (all Executed).**

---

## 2.2 Label Flipping — Edge-IIoTset

| Configuration   | Sources                   | Status   | Runs |
| --------------- | ------------------------- | -------- | ---: |
| LabelFlip β=0.1 | Table III                 | Executed |    5 |
| LabelFlip β=0.2 | Table III                 | Executed |    5 |
| LabelFlip β=0.3 | Table III; Table VI (30%) | Executed |    5 |

**Subtotal: 15 runs (all Executed).**
**Note:** Table VI calculation contains an inconsistency. See §6.1.

---

## 2.3 Gradient Manipulation — Edge-IIoTset

| Configuration        | Sources                   | Status   | Runs |
| -------------------- | ------------------------- | -------- | ---: |
| GradManip β=0.1, α=3 | Table III                 | Executed |    5 |
| GradManip β=0.2, α=3 | Table III                 | Executed |    5 |
| GradManip β=0.3, α=3 | Table III; Table VI (30%) | Executed |    5 |

**Subtotal: 15 runs (all Executed).**
**Note:** Table VI calculation contains an inconsistency. See §6.1.
**Note:** Figure 3 is an ambiguous source. See §6.2 for the exclusion decision.

---

## 2.4 Backdoor Injection — Edge-IIoTset

| Configuration   | Sources                        | Status   | Runs |
| --------------- | ------------------------------ | -------- | ---: |
| Backdoor, β=0.2 | Page 5, VI.E ¶1; Table VI      | Executed |    5 |

**Subtotal: 5 runs (all Executed).**

---

## 2.5 Adversarial Evasion — Edge-IIoTset

| Configuration | Sources  | Status   | Runs |
| ------------- | -------- | -------- | ---: |
| FGSM ε=0.05   | Table IV | Executed |    5 |
| FGSM ε=0.10   | Table IV | Executed |    5 |
| PGD-7 ε=0.05  | Table IV | Executed |    5 |
| PGD-7 ε=0.10  | Table IV | Executed |    5 |
| PGD-20 ε=0.05 | Table IV | Executed |    5 |
| PGD-20 ε=0.10 | Table IV | Executed |    5 |

**Subtotal: 30 runs (all Executed).**
**Note:** Figure 4 is an ambiguous source. See §6.2 for the exclusion decision.

---

## §2 Summary

| Subsection      | Configurations | Executed runs |
| --------------- | -------------: | ------------: |
| 2.1 Clean       |              3 |            15 |
| 2.2 Label Flip  |              3 |            15 |
| 2.3 Gradient    |              3 |            15 |
| 2.4 Backdoor    |              1 |             5 |
| 2.5 Adversarial |              6 |            30 |
| **Total**       |         **16** |        **80** |

---

# 3. ZTA-FL Runs — Text Parameter Grids

These parameter values are stated in the paper in a prose-like manner, without excluding their appearance in any table or figure. They are included for completeness of the documented parameter space, but **not all** will be executed.

The rationale for this exclusion is methodological. The purpose of this reproduction is to compare the performance of our ZTA-FL implementation against the paper's reported ZTA-FL results, rather than to generate metrics in isolation. The paper states the parameter ranges for label flipping probability ($p_{flip}$), gradient manipulation magnitude ($\alpha$), backdoor poisoning rate, and adversarial perturbation budgets ($\epsilon$). However, the paper does not report any accuracy, F1, or attack success rate for the specific configurations listed in this section. For example, while the paper states $p_{flip} \in \{0.2, 0.5, 1.0\}$ and $\alpha \in \{-3, -1, 3, 5\}$, Tables III and VI only report results for a single value of each parameter.

Consequently, executing these runs would produce metrics that cannot be compared to any published baseline.

**Inclusion criteria for this reproduction:**
Only configurations for which the paper provides an explicit reported metric, whether in a table, a figure, or a prose statement, are included in the active run count. All such configurations are captured in §2 and §4. As such, most of the 170 runs listed in this section are therefore excluded from the final execution total. This ensures that every run executed in this reproduction serves a direct comparative purpose.

| Configuration                                       | Runs | Status          | Mentioned in                                      |
| --------------------------------------------------- | ---: | --------------- | ------------------------------------------------- |
| LabelFlip p_flip=0.1 (all β values (0.1, 0.2, 0.3)) |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| LabelFlip p_flip=0.2 (all β values (0.1, 0.2, 0.3)) |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1                 |
| LabelFlip p_flip=0.5 (all β values (0.1, 0.2, 0.3)) |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| GradManip α=-1 (all β values (0.1, 0.2, 0.3))       |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| GradManip α=-3 (all β values (0.1, 0.2, 0.3))       |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| GradManip α=3 (all β values (0.1, 0.2, 0.3))        |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| GradManip α=5 (all β values (0.1, 0.2, 0.3))        |   15 | Documented only | Page 4, VI.A, Attack scenarios ¶1; Page 2, III ¶2 |
| Backdoor β=0.2, rate=10%                            |    5 | Documented only | Page 4, VI.A, Attack scenarios ¶1                 |
| FGSM all ε (0.05, 0.1, 0.15, 0.2)                   |   20 | Documented only | Page 4, VI.A, Attack scenarios ¶1                 |
| PGD-7 all ε (0.05, 0.1, 0.15, 0.2)                  |   20 | Documented only | Page 4, VI.A, Attack scenarios ¶1                 |
| PGD-20 all ε (0.05, 0.1, 0.15, 0.2)                 |   20 | Documented only | Page 4, VI.A, Attack scenarios ¶1                 |

**Note:** There is a conflict with the `p_{flip}` parameter values. See §6.3.

**Subtotal: 170 runs (all Documented only).**

---

# 4. ZTA-FL Runs — Exploratory

Secondary runs referenced in discussion, limitations, or failure-mode analysis. None are planned for execution.

## 4a. Failure-Mode Runs

| Configuration                                     | Mentioned in                                            | Status      | Runs |
| ------------------------------------------------- | ------------------------------------------------------- | ----------- | ---: |
| ZTA-FL, SHAP-aware constrained poisoning          | Page 7, VIII.B, SHAP-Aware Attacks ¶1                   | Exploratory |    5 |
| ZTA-FL, Slow poisoning (<0.1 α/round, 50+ rounds) | Page 7, VIII.B, Adaptive Long-Term Poisoning Attacks ¶1 | Exploratory |    5 |
| ZTA-FL, Clean-label backdoor                      | Page 8, Table VII                                       | Exploratory |    5 |
| ZTA-FL, Collusion-based attack β=0.4              | Page 7, VIII.B, Collusion-Based Attacks ¶1              | Exploratory |    5 |
| ZTA-FL, Pathological non-IID (1 class per agent)  | Page 7, VIII.C, 1) Extreme Non-IID ¶1                   | Exploratory |    5 |

**Subtotal: 25 runs (all Exploratory).**

## 4b. Efficiency and Scalability Runs

| Configuration                                                 | Mentioned in                                                  | Status      | Runs |
| ------------------------------------------------------------- | ------------------------------------------------------------- | ----------- | ---: |
| ZTA-FL, Scalability stress test N=1,000 agents                | Page 6, VI.F, Scalability ¶1                                  | Exploratory |    5 |
| ZTA-FL, KernelSHAP background=20 samples                      | Page 8, VIII.D, Scalability and SHAP Optimizations ¶1, item 1 | Exploratory |    5 |
| ZTA-FL, Selective adversarial training (high-confidence only) | Page 8, VIII.D, Computation Cost ¶1                           | Exploratory |    5 |
| ZTA-FL, Feature Group SHAP                                    | Page 8, VIII.D, Scalability and SHAP Optimizations ¶1, item 2 | Exploratory |    5 |
| ZTA-FL, Asynchronous SHAP                                     | Page 8, VIII.D, Scalability and SHAP Optimizations ¶1, item 3 | Exploratory |    5 |

**Subtotal: 25 runs (all Exploratory).**

No duplicates exist between §4 and §2/§3. Every entry uses a distinct attack type, data distribution, scale, or optimization.

---

# 5. Documented but Not Executed — Component Ablation

ZTA-FL variants only. Baseline FL row is excluded (not ZTA). "+All" row is excluded (≡ main ZTA-FL runs, see §6).

| Configuration                              | Sources | Status          | Runs |
| ------------------------------------------ | ------- | --------------- | ---: |
| ZTA-FL, +Attestation, Clean                | Table V | Documented only |    5 |
| ZTA-FL, +Attestation, Poisoned             | Table V | Documented only |    5 |
| ZTA-FL, +Attestation, Adversarial          | Table V | Documented only |    5 |
| ZTA-FL, +SHAP Aggregation, Clean           | Table V | Documented only |    5 |
| ZTA-FL, +SHAP Aggregation, Poisoned        | Table V | Documented only |    5 |
| ZTA-FL, +SHAP Aggregation, Adversarial     | Table V | Documented only |    5 |
| ZTA-FL, +Adversarial Training, Clean       | Table V | Documented only |    5 |
| ZTA-FL, +Adversarial Training, Poisoned    | Table V | Documented only |    5 |
| ZTA-FL, +Adversarial Training, Adversarial | Table V | Documented only |    5 |

**Subtotal: 45 runs (all Documented only).**

> The component ablation study (Table V) was not reproduced. This decision was made for two reasons.
> First, the primary contribution of this work is the comparative evaluation of ZTA-FL against Byzantine-resilient and adversarial-robust federated learning baselines, which is covered by Tables II, III, IV, and VI. The ablation study addresses internal design decomposition rather than external performance.
> Second, the ablation variants are not fully specified in the original paper. For example, the precise attack parameters for the `Poisoned` and `Adversarial` columns are not stated. Reproducing them would require assumptions that could undermine the validity of the comparison. This omission is acknowledged as a limitation of the present reproduction.

---

# 6. Paper Inconsistencies

Diagnoses of internal contradictions in the source paper. These are reported as anomalies, not as evidence of flawed methodology.

## 6.1 Table VI

Table VI is titled *"Comparison with State-of-the-Art Byzantine Resilient FL Approaches on the Edge-IoT Dataset under 30% of Byzantine Attacker Rate."* The phrase **"under 30%"** is strict: it excludes β = 0.3, leaving only β ∈ {0.1, 0.2} for that table. However, a close comparison of `Table VI` against `Table III` and the `prose in Section VI.E` reveals several inconsistencies that are worth documenting. They are presented here as reporting anomalies, not as evidence of flawed methodology.

### (a) Label flipping at β = 0.3: same mean, different standard deviation

| Source             | ZTA-FL Label Flip Accuracy |
| ------------------ | -------------------------- |
| Table III, β = 0.3 | 93.2 ± 1.0                 |
| Table VI           | 93.2 ± 0.6                 |

The **mean is identical** (93.2), which strongly suggests the same underlying five-run experiment is being reported in both places. The standard deviation differs (1.0 vs. 0.6). This is a minor discrepancy that can arise from:
- rounding at different decimal places,
- reporting the standard deviation over a different subset of seeds,
- a simple transcription error.

Given the exact mean match, the probability that these are two independent run sets is very low. The most likely explanation is that the same five runs are being reported in both tables, with the standard deviation inadvertently altered or recomputed.

**Also note:** Table III reports this value at β = 0.3, which is exactly 30%, not *under* 30%. This contradicts the strict reading of Table VI's title. The title would be more accurate if it said *"at or under 30%"* or *"up to 30%."*

### (b) Gradient manipulation: a value not present in Table III

| Source             | ZTA-FL Gradient Manip Accuracy |
| ------------------ | ------------------------------ |
| Table III, β = 0.3 | 91.7 ± 1.2                     |
| Table VI           | 91.7 ± 0.7                     |

Once again, the mean is identical (91.7), with a difference in the standard deviation (1.2 vs. 0.7). The same comments and notes apply as above.

### (c) Backdoor: consistent with the prose

| Source               | ZTA-FL Backdoor ASR |
| -------------------- | ------------------- |
| Table VI             | 8.7 ± 1.5           |
| Section VI.E prose   | 8.7%                |

The prose states *"20% of the client devices are compromised with backdoors,"* which corresponds to β = 0.2. That is under 30%, so it fits the Table VI header. The mean matches exactly (8.7), confirming the same run set. No inconsistency here.

---

## 6.2 Figure-Caption Discrepancy on Figures 3 & 4

Figure 3 contains a single data plot illustrating the relationship between "Compromised Fraction ($\beta$)" and "Accuracy (%)" for three federated learning methods (ZTA-FL, FedAvg, Krum). However, the accompanying text reads, "*Fig. 3. Accuracy under poisoning: (a) label flipping, (b) gradient manipulation.*".

Similarly, Figure 4 contains a single data plot illustrating the relationship between "Perturbation budget ($\epsilon$)" and "Accuracy (%)" for three federated learning methods (ZTA-FL, FedAvg, Adversarial-FL). However, the accompanying text reads, "*Fig. 4. Accuracy vs. perturbation budget ϵ for FGSM and PGD-20 attacks.*".

The captions explicitly define a multi-panel figure structure containing two sub-figures, denoted as (a) and (b), which represent distinct experimental attack vectors. However, the graphical asset itself only displays a single plot in each figure. The visible plots lack an internal sub-figure label (such as an "(a)" or "(b)" typically placed within the chart area).

This discrepancy is a typesetting, compilation, or editorial error. Because the standalone plot is missing its corresponding sub-figure identifier, it is impossible to determine solely from the visual evidence whether the displayed data represents the "label flipping" scenario or the "gradient manipulation" scenario for Figure 3 and the "FGSM" or the "PGD-20" scenario for Figure 4.

Therefore Figures 3 & 4 are treated as an ambiguous source and are not used to expand the adversarial evasion grid. The active adversarial evasion runs are those explicitly tabulated in Table IV: FGSM at ε ∈ {0.05, 0.10} and PGD-20 at ε ∈ {0.05, 0.10}, plus PGD-7 at ε ∈ {0.05, 0.10}. The ε = 0.15 and 0.20 values mentioned in §VI.A are documented in §3 as text-only and are not executed.

---

## 6.3 Variable value Inconsistency on §III and §VI.A ¶3 for $p_{flip}$

There is a direct parameter mismatch between the theoretical model and the experimental evaluation:

*   **Section III (System Model):** Bounds the label flip probability to a maximum of 0.5, defining the range as $p_{flip} \in [0.1, 0.5]$.
*   **Section VI.A (Experiments):** Tests discrete values outside this established theoretical bound, specifically $p_{flip} \in \{0.2, 0.5, 1.0\}$.

---

## 6.4 The PGD-7 ε Grid Contradiction

Section VI.A states that the adversarial evasion experiments use perturbation bounds of ε ∈ {0.05, 0.1, 0.15, 0.2}. However, Table IV only reports tabulated results for ε = 0.05 and ε = 0.1. The experimental text defines a four-value testing grid, but the corresponding results table only provides data for two of those values.

---

## 6.5 The Gradient α Grid Contradiction

Section VI.A defines the gradient manipulation attack using a test grid of four scaling factors: α ∈ {-3, -1, 3, 5}. In contrast, the performance tables report only a single accuracy metric for this attack, which corresponds specifically to the α=3 scenario. The methodology promises a multi-parameter evaluation, but the tables only publish the tabulated results of one specific attack value.

---

## 6.6 The Dataset Count Contradiction

The paper's abstract explicitly states the experiments were conducted on **three** different IDS benchmarks: Edge-IIoTset, CIC-IDS2017, and UNSW-NB15. However, the fourth point in the Introduction's list of contributions claims the framework was evaluated on only **two** different data sets, omitting UNSW-NB15 entirely. This creates a direct textual contradiction regarding the primary scope of the study.

---

## 6.7 Label Flipping p_flip Reporting Gap

Section VI.A lists three separate probabilities for label flipping: $p_{flip} \in \{0.2, 0.5, 1.0\}$. However, none of the performance tables explicitly state which of those three $p_{flip}$ values is actually being reported in the data. To resolve this, the decision is to assume the tabulated runs represent one of the values, and manually add the other two values as separate entries (under §3) to complete the data set.

---

# 7. Reproduction Decisions

Decisions made in response to the inconsistencies in §6, and decisions about scope and structure independent of any inconsistency.

## 7.1 Deduplication
Table VI values match Table III values (label flip β=0.3: 93.2; gradient manipulation β=0.3: 91.7) and Table VI's backdoor ASR matches §VI.E prose (8.7%). These are treated as the same underlying five-run experiments and are not counted twice.

## 7.2 Figure 4 exclusion
See §6.2 and the decision note therein.

## 7.3 Ablation exclusion
See §5.

## 7.4 p_flip reporting gap
§VI.A lists p_flip ∈ {0.2, 0.5, 1.0} but no table states which value is active. Decision: the active value is treated as unspecified; all three values are documented in §3 as text-only.

## 7.5 Dataset count
Three datasets are used: Edge-IIoTset, CIC-IDS2017, UNSW-NB15. The Introduction's "two different data sets" is treated as an error.

---

# 8. Summary of Runs to Be Executed

This reproduction will execute **80 ZTA-FL runs** total. These are exactly the executed configurations listed in §2, and they all use the global constants in §1 and the five seeds `42, 123, 456, 789, 1011` (5 runs per configuration).

| Executed group                       | Source subsection | Configurations |   Runs |
| ------------------------------------ | ----------------- | -------------: | -----: |
| Clean data                           | §2.1              |              3 |     15 |
| Label flipping — Edge-IIoTset        | §2.2              |              3 |     15 |
| Gradient manipulation — Edge-IIoTset | §2.3              |              3 |     15 |
| Backdoor injection — Edge-IIoTset    | §2.4              |              1 |      5 |
| Adversarial evasion — Edge-IIoTset   | §2.5              |              6 |     30 |
| **Total**                            |                   |         **16** | **80** |

**Executed configurations by group:**

- **Clean data (§2.1):** Edge-IIoTset Clean; CIC-IDS2017 Clean; UNSW-NB15 Clean.
- **Label flipping — Edge-IIoTset (§2.2):** LabelFlip β=0.1; LabelFlip β=0.2; LabelFlip β=0.3.
- **Gradient manipulation — Edge-IIoTset (§2.3):** GradManip β=0.1, α=3; GradManip β=0.2, α=3; GradManip β=0.3, α=3.
- **Backdoor injection — Edge-IIoTset (§2.4):** Backdoor β=0.2.
- **Adversarial evasion — Edge-IIoTset (§2.5):** FGSM ε=0.05; FGSM ε=0.10; PGD-7 ε=0.05; PGD-7 ε=0.10; PGD-20 ε=0.05; PGD-20 ε=0.10.

All other documented runs in §3, §4a, §4b, and §5 are **not** part of the executed total.