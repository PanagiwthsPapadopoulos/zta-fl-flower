# CIC-IDS2017 Dataset

## Overview

CIC-IDS2017 provides labelled network-flow data generated at the Canadian Institute for Cybersecurity. Traffic was captured from July 3 to July 7, 2017. Monday contains benign traffic; attacks were introduced on subsequent days. CICFlowMeter extracts flow measurements, and B-Profile supplies benign traffic profiles. The [official dataset description](https://www.unb.ca/cic/datasets/ids-2017.html) documents the network, attack schedule and collection procedure.

## Traffic categories

The repository's full-data mapping includes benign traffic, FTP/SSH brute force, DoS variants, Heartbleed, web attacks, infiltration, botnet traffic, port scanning and DDoS. The complete class names are defined in [dataset_schema.py](../../src/shared/data/dataset_schema.py); source-day and attack-tool details are available in the [official schedule](https://www.unb.ca/cic/datasets/ids-2017.html).

## Repository data layout

All paths below are relative to the repository root.

| Path | Purpose |
| --- | --- |
| `data/cic_ids2017/raw/network_flows.csv` | Shipped small snippet for quick checks |
| `data/cic_ids2017/full_dataset/downloads/` | Downloaded source files |
| `data/cic_ids2017/full_dataset/network_flows.csv` | Cleaned full data: 77 numeric features and a target column |
| `data/cic_ids2017/full_dataset/preprocessing_report.json` | Counts, exclusions, labels and feature order |
| `data/cic_ids2017/artifacts/` | Training/validation/test tensors, preprocessing state and Edge partitions |

Download and clean from the repository root:

```sh
python3 scripts/ops/download_datasets.py --only cic
```

Select `data/cic_ids2017/full_dataset/network_flows.csv` as `dataset_path` in `config/training.yaml`. The launcher's artifact builder handles experiment splitting, scaling, PCA and configured training resampling. It rebuilds incompatible cached artifacts automatically. The full download does not overwrite or merge with the shipped snippet. See [dataset preparation](../dataset-preparation.md) for the complete policy and preprocessing-only commands.

## Download (Full Dataset)

The dataset is distributed by the Canadian Institute for Cybersecurity
(University of New Brunswick):

- **Landing page**: [CIC-IDS2017 (UNB)](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Direct file index**: [CIC FTP archive](http://cicresearch.ca/CICDataset/CIC-IDS-2017/)

The downloader uses its configured Hugging Face mirror for all eight daily MachineLearningCVE CSVs, with a Kaggle fallback. It removes identical original records and rows with invalid retained numeric values. The repeated header-length feature is collapsed only when the two source columns agree. See [dataset preparation](../dataset-preparation.md) for the exact schema and supported 15-class mapping.

## Feature Set

Each row represents a network flow. The cleaner retains 77 numeric measurements after validating and collapsing the duplicate header-length column. These include:
- Basic flow statistics (duration, packet counts, byte counts)
- Packet length statistics (min, max, mean, std)
- Inter-arrival time statistics for forward and backward directions
- TCP flag counts (FIN, SYN, RST, PSH, ACK, URG)
- Active and idle time statistics
- Subflow statistics and window sizes

## Citation

> I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion
> Detection Dataset and Intrusion Traffic Characterization," in *Proc. ICISSP*, 2018, pp. 108–116.
