# Dataset preparation: behavior and design choices

[download_datasets.py](../scripts/ops/download_datasets.py) downloads the benchmark CSVs and calls [preprocess_datasets.py](../scripts/ops/preprocess_datasets.py) to produce numeric CSV inputs for the existing artifact builder. This document describes the implemented choices and their rationale.

## Boundary between cleaning and training preparation

The cleaner combines the selected source files within each benchmark, removes identical records, selects numeric features and validates labels. It replaces unusable UNSW numeric values with zero and excludes affected rows for Edge and CIC. Each benchmark produces its own CSV; benchmarks are never merged with each other or with the shipped snippets.

[build_artifacts.py](../scripts/setup/build_artifacts.py) remains responsible for seeded, stratified train/validation/test splitting, MinMax scaling, PCA, optional training-only SMOTE, tensor creation and Edge partitions. Keeping these operations there preserves the existing model-input pipeline and avoids a second implementation of training preprocessing.

The cleaned CSVs contain 42 Edge features, 77 CIC features or 42 UNSW features. The builder reduces these to the existing 40-feature model input. Compatibility means a numeric CSV with a recognized target column, followed by the existing tensor format; it does not mean reproducing the snippet's exact feature names or values.

With `simulate_global_leakage` disabled, the builder fits scaling and PCA on training data. Enabling that experimental option fits them on the full input instead. Split fractions and the random seed remain builder settings: for example, `val_split: 0.15` and `test_split: 0.15` produce a 70/15/15 split. The downloader neither chooses nor rewrites these settings.

## Source selection

| Benchmark | Sources used for cleaning | Rationale |
| --- | --- | --- |
| Edge-IIoTset | Both `DNN-EdgeIIoT-dataset.csv` and `ML-EdgeIIoT-dataset.csv` from the configured Ferrag Kaggle dataset | Both share the expected 63-column schema. Combining them preserves distinct available records; deduplication removes identical overlap. |
| CIC-IDS2017 | All eight daily MachineLearningCVE CSVs listed in the cleaner | Includes the source files for all days and attack categories. The downloader uses its configured Hugging Face mirror and attempts a Kaggle fallback if that download fails. |
| UNSW-NB15 | `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` from the configured Kaggle mirror | Uses one consistent full-data schema and lets the existing builder create the experiment's three splits. |

The UNSW prepared training/testing files and supporting metadata are also downloaded. The two prepared splits remain references and contribute no rows to the cleaned CSV. Their feature schema differs from the four full files, and appending them is not treated as a way to obtain additional independent observations. The downloader can correct their swapped filenames when their row counts match the reversed expected sizes; this does not affect which data enters training.

The four full UNSW files are headerless. Their first rows are observations, not headers. The cleaner applies the fixed 49-column source schema from [dataset_schema.py](../src/shared/data/dataset_schema.py).

Required cleaning sources must all exist and be nonempty. Edge and CIC headers must match the expected names and order after surrounding header whitespace is removed. Unexpected schemas, malformed row widths and unknown labels stop preparation for that benchmark instead of silently changing the input population or class meaning.

## Numeric feature policy

Numeric measurements are retained without categorical encoding. UNSW uses the constant-zero replacement policy described below; Edge and CIC use row exclusion. No category vocabulary or imputation statistic is fitted by the cleaner. Feature selection affects the representation and does not by itself establish equivalence to a preprocessing recipe that uses categorical features.

For Edge, the following text, address and categorical columns are excluded:

| Group | Excluded columns |
| --- | --- |
| Time and addresses | `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.dst.proto_ipv4`, `arp.src.proto_ipv4` |
| HTTP | `http.file_data`, `http.request.uri.query`, `http.request.method`, `http.referer`, `http.request.full_uri`, `http.request.version` |
| TCP and DNS | `tcp.options`, `tcp.payload`, `dns.qry.name` |
| MQTT | `mqtt.conack.flags`, `mqtt.msg_decoded_as`, `mqtt.msg`, `mqtt.protoname`, `mqtt.topic` |

Edge's binary `Attack_label` is also excluded from the features. `Attack_type` supplies the multiclass target and is written as `label`.

For UNSW, `srcip`, `dstip`, `proto`, `state` and `service` are excluded. Numeric ports and timestamps remain features. `attack_cat` supplies the target; binary `Label` is used for label consistency checks and is excluded from the output features.

CIC retains its numeric measurements and uses `Label` as the target. Its repeated `Fwd Header Length` column is collapsed only when both copies agree for every record. Surrounding whitespace and numerically equivalent spellings such as `1` and `1.0` are accepted for this comparison. A disagreement stops preparation rather than choosing one copy arbitrarily.

The exact feature order and source schemas are defined in [dataset_schema.py](../src/shared/data/dataset_schema.py). Each successful cleaning report records the emitted feature list, excluded source columns and collapsed duplicate columns.

## Labels and compatibility

| Input | Classes | Handling |
| --- | --- | --- |
| Full Edge | 15 | Uses the full dataset's class names, including `DDoS_TCP`, `DDoS_UDP`, `DDoS_HTTP`, `DDoS_ICMP` and `MITM`. |
| Shipped Edge snippet | 15 | Retains its separate legacy mapping, including `Scanning` and `MITM_ARP`. Those labels are not aliases for full-data DDoS classes. |
| CIC | 15 | Includes `FTP-Patator`, `SSH-Patator`, `DoS slowloris`, `DoS Slowhttptest` and `DoS GoldenEye` in addition to the ten previously supported classes. |
| UNSW | 10 | Uses explicit attack categories and `Normal`; `Backdoors` is normalized to `Backdoor`. |

[data_loader.py](../src/shared/data/data_loader.py) selects the full Edge mapping when the feature-column set matches the complete 42-feature cleaned schema. Other numeric Edge inputs retain the legacy snippet mapping. This preserves snippet compatibility without merging the two taxonomies. Class IDs must be interpreted using the mapping for the corresponding input.

For UNSW, an empty `attack_cat` becomes `Normal` only when binary `Label` is `0`. Binary labels must be `0` or `1`, and recognized categories must agree with them. Missing attack categories on attack rows and contradictory labels fail validation.

The cleaner normalizes explicit dash variants in CIC's three Web Attack labels: Unicode en dash, ASCII hyphen, Windows-1252 dash and the Unicode replacement character. It strips surrounding target whitespace but does not infer arbitrary label aliases. Unknown classes never become benign by default. The canonical mappings are defined in the shared schema module and copied into the cleaning report.

## Identical records and invalid measurements

Deduplication compares the complete original sequence of parsed CSV field strings across all selected files of a benchmark. The key includes discarded text, addresses and labels. It does not normalize field whitespace, numeric spelling or label aliases. CSV quoting and line-ending syntax are not part of that key because comparison happens after CSV parsing.

Consequently, records that differ only in an excluded address remain distinct even if their emitted numeric rows are identical. Likewise, source values `1` and `1.0` are distinct for record deduplication. This avoids broadening an identical-record rule into feature-based or approximate deduplication.

Rows are streamed, and a temporary SQLite database stores complete deduplication keys on disk. This avoids holding all original records in RAM and avoids relying solely on a hash for equality. The temporary database and staged output require additional disk space while cleaning runs.

Among unique records, numeric handling is dataset-specific:

| Input value in a retained feature | Edge and CIC | UNSW |
| --- | --- | --- |
| Empty or whitespace-only | Exclude row | Replace cell with `0` |
| Failed numeric conversion, including text and hexadecimal port strings | Exclude row | Replace cell with `0` |
| NaN or positive/negative infinity | Exclude row | Replace cell with `0` |
| Finite value that overflows float32 | Exclude row | Replace cell with `0` |
| Valid finite float32 measurement | Retain | Retain |

UNSW applies this policy across all 42 retained features, including `ct_flw_http_mthd`, `is_ftp_login` and `ct_ftp_cmd`. These columns remain present even when many entries require replacement. Numeric failures alone do not remove UNSW records. Near the float32 limit, values that round to the finite maximum are retained at that rounded value; they are not classified as overflow. Other valid numeric spellings are preserved, with float32 conversion performed by the existing loader.

The basis is the public ZTA-FL [data loader](https://github.com/ssam18/zta-federated-learning/blob/main/src/utils/data_loader.py), inspected on 2026-10-25: `_safe_float` (lines 149–154) supplies zero on conversion failure, and `load_unsw_nb15` (lines 340–352) applies float32 conversion followed by zero replacement for NaN/infinity. The cleaner matches this numeric policy while retaining explicit schema and label validation. Unknown labels and malformed rows remain fatal errors.

The [paper's dataset methodology](https://arxiv.org/html/2512.23809v1) specifies scaling, PCA, balancing and splitting but no missing-value rule. This alignment therefore concerns the released implementation; it does not prove which cleaning produced the published metrics. Zero replacement is a reproducibility choice, not evidence that each missing measurement actually was zero. It uses no fitted statistics or held-out observations.

Invalid values in discarded text fields do not trigger numeric replacement or rejection. Deduplication remains before replacement: two different original records that become equal after zero filling are both retained. There is no second deduplication pass over the transformed features.

For a successful run, the counts reconcile as:

```text
input_rows = duplicate_rows + invalid_numeric_rows + kept_rows
```

Duplicate counting precedes numeric handling. If an affected record occurs twice, its first occurrence is excluded for Edge/CIC or retained with replacements for UNSW; its second occurrence counts only as a duplicate. Per-field and per-reason totals count cell occurrences, so they can exceed the number of affected rows.

`preprocessing_report.json` records these totals, per-source counts, invalid fields/reasons, excluded-row class counts, retained class counts, class mappings, output paths and completion time. For UNSW it also records the numeric policy's source and review date, `zero_filled_rows`, `zero_filled_values`, and replacement counts by field, reason and class, both overall and per source. Reasons are `missing`, `invalid`, `nonfinite` and `float32_overflow`. A row with several replacements counts once in the affected-row and class totals, but once per replaced cell in the value, field and reason totals. `zero_filled_rows` is a subset of `kept_rows`; it is not an additional term in the reconciliation equation. `invalid_numeric_rows` counts exclusions only and is zero for a successful UNSW run under this policy.

A retained class count of zero remains visible. Reports describe completed cleaning runs; failed attempts leave the previously published CSV/report in place. Both exclusions and replacements can affect experiments, so the full-run counts should accompany dataset descriptions. Replacement counters are zero for Edge and CIC.

## Storage and publication

The default output root is the repository's `data/` directory, independently of the shell's working directory:

```text
data/<dataset>/
  raw/                              shipped snippet, kept in place
  full_dataset/
    <existing_basename>.csv          cleaned numeric input
    preprocessing_report.json       cleaning decisions and counts
    downloads/                      downloaded source files
  artifacts/                        tensors and manifests made by the builder
```

| Dataset directory | Cleaned basename | Target column |
| --- | --- | --- |
| `edge_iiotset` | `network_traffic_samples.csv` | `label` |
| `cic_ids2017` | `network_flows.csv` | `Label` |
| `unsw_nb15` | `intrusion_records.csv` | `attack_cat` |

Keeping each cleaned CSV one directory below its dataset root preserves the builder's existing resolution of the sibling `artifacts/` directory. A root-level `datasets/` directory is not an input or output location used by the default command.

After cleaning succeeds, the script replaces the previous cleaned CSV and its report. The shipped snippet stays in `raw/`; no archive directories or permanent copies of previous cleaned outputs are created. Previous outputs are held temporarily inside the cleaning workspace so publication exceptions can restore them. Those temporary copies are removed when successful publication completes. This is exception recovery, not a guarantee of a multi-file transaction surviving abrupt process or machine termination. Downloaded sources remain available for reruns. Each benchmark publishes independently; a later benchmark's failure does not undo an earlier successful one.

`data/*/full_dataset/` is ignored by Git because of its size. The existing artifact ignore rule also remains in effect. The script does not edit experiment configuration or create or delete training artifacts.

## Running and checking the result

Download mode requires Python 3.9 or later, `requests` and `kagglehub`. Cleaning alone uses the Python standard library and the repository's schema module. Artifact building requires the project's separate ML dependencies.

Run from the repository root:

```sh
python3 -m pip install requests kagglehub
python3 scripts/ops/download_datasets.py

# Select benchmarks or a different data root:
python3 scripts/ops/download_datasets.py --only edge unsw
python3 scripts/ops/download_datasets.py --out /path/to/data

# Reuse completed sources already under full_dataset/downloads/:
python3 scripts/ops/download_datasets.py --only unsw --preprocess-only
```

Existing nonempty download files are reused by default. `--no-skip-existing` requests replacement downloads. HTTP transfers use resumable `.part` files and validate resume offsets and available length information. Kaggle file moves are staged so an interrupted copy is not exposed under the completed filename. Reusing an existing source by size alone is not a checksum verification.

Changing the cleaning code does not rewrite an existing cleaned CSV automatically. Rerun preprocessing for UNSW to apply zero replacement to all four downloaded full-data files. The final summary reports retained rows, removed duplicates, excluded rows, replaced values and affected retained rows. Once the updated CSV is selected, the builder detects its changed content and rebuilds stale artifacts as described below.

`--verify` adds advisory source row-count checks; a printed mismatch is not itself a fatal error or proof of source corruption. It does not validate model metrics. `--smoke` downloads a reduced selection for connectivity checks and never publishes cleaned inputs; Edge downloads can still be large. `--no-ml` is supported only with `--smoke`, because full Edge preparation requires both CSVs. `--with-pdf` additionally downloads the Edge documentation PDF. Smoke mode and preprocessing-only mode are mutually exclusive.

For each successful benchmark, inspect `preprocessing_report.json` and set the experiment's `dataset_path` to the path printed by the command. For example:

```yaml
dataset: edge_iiotset
dataset_path: data/edge_iiotset/full_dataset/network_traffic_samples.csv
```

Fetching the full data alone does not change the selected input. `dataset_path` determines which CSV the builder uses. Leaving it pointed at `raw/` continues to use the snippet, even if full data has been downloaded or old artifacts have been deleted.

## Automatic artifact rebuilding

The builder differentiates inputs by CSV content through `source_sha256` in the artifact manifest, not by a filename convention or a separate snippet/full flag. Loader code, label-schema code, class count, preprocessing options, seed and split fractions also participate in validation. Identical CSV bytes with identical preprocessing can reuse base tensors even if the source path changes.

Snippet and full inputs for the same benchmark share `data/<dataset>/artifacts/`. Once `dataset_path` selects a different input, the builder detects the provenance mismatch, rebuilds the train/validation/test tensors for the selected configuration and regenerates its Edge partitions. Existing artifacts do not need to be deleted first. Other cached preprocessing variants are validated when selected; the downloader does not indiscriminately purge them.

All four standard launchers invoke the builder before starting the federation and abort if compilation fails:

- [Docker deployment](../scripts/deployment/deploy_code_docker.sh)
- [Apptainer deployment](../scripts/deployment/deploy_code_apptainer.sh)
- [Docker simulation](../scripts/simulation/deploy_code_sim_docker.sh)
- [Apptainer simulation](../scripts/simulation/deploy_code_sim_apptainer.sh)

Missing compiled artifacts or manifests also trigger rebuilding on the next launcher run. Deleting them is therefore a possible way to force regeneration, but is unnecessary when switching inputs because provenance already handles that change. Node processes themselves only validate and consume artifacts; when bypassing these launchers, run `scripts/setup/build_artifacts.py` before starting nodes.

The builder creates its own train/validation/test splits from the cleaned input; it does not consume UNSW's supplied split membership. The cleaner's streaming memory behavior does not apply to the builder, which loads the numeric dataset into memory.

The downloader exits nonzero when any requested benchmark fails. A successful cleaned CSV establishes input-format readiness, not successful full-data training or reproduction of paper metrics. Rare retained classes may still be insufficient for the builder's stratified splitting or SMOTE requirements; the cleaner does not invent observations to satisfy those requirements.

Focused regression tests are in [test_dataset_preparation.py](../scripts/tests/test_dataset_preparation.py):

```sh
python -B scripts/tests/run_tests.py --pattern test_dataset_preparation.py
```

They exercise small full-schema fixtures, exact deduplication, invalid-row counts, UNSW zero replacement and its audit counts, class mappings, source selection, publication recovery, mocked download interruptions and cleaned Edge/UNSW inputs passing through the real artifact builder. UNSW tests compare numeric results with the released conversion sequence, preserve distinct records that coincide after replacement, and reject malformed rows and invalid labels. They also verify switching from snippet artifacts to full-data artifacts in the same directory, reuse of unchanged base tensors and regeneration after compiled artifacts are removed. They do not download or process the full benchmarks. The broader suite and evidence format are documented in [the test guide](testing.md).
