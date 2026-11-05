# UNSW-NB15 Dataset

## Overview

UNSW-NB15 was generated in the UNSW Canberra Cyber Range using IXIA PerfectStorm, with packet capture through tcpdump and feature extraction using Argus, Bro-IDS and additional algorithms. The official release includes four full CSV files containing 2,540,044 records, plus prepared training/testing subsets. See the [official dataset description](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for collection details and source files.

The four headerless files use the 49-column schema declared in [dataset_schema.py](../../src/shared/data/dataset_schema.py), including target fields. The repository retains 42 numeric input features and the multiclass target; it does not treat all source columns as model features.

## Attack Categories

| Category | Description |
|----------|-------------|
| Normal | Legitimate network activity from the cyber range testbed |
| Generic | Generic attack patterns not tied to a specific exploit |
| Exploits | Known CVE-based exploitation attempts |
| Fuzzers | Malformed or randomised packet injection |
| DoS | Denial-of-service flooding and resource exhaustion |
| Reconnaissance | Network scanning, probing, and enumeration |
| Backdoor | Persistent access tools and reverse shell traffic |
| Analysis | File and network analysis using attack tools |
| Shellcode | Shellcode injection payload traffic |
| Worms | Automated worm propagation and scanning |

## Repository data layout

All paths below are relative to the repository root.

| Path | Purpose |
| --- | --- |
| `data/unsw_nb15/raw/intrusion_records.csv` | Shipped small snippet for quick checks |
| `data/unsw_nb15/full_dataset/downloads/` | Downloaded source files |
| `data/unsw_nb15/full_dataset/intrusion_records.csv` | Cleaned full data: 42 numeric features and a target column |
| `data/unsw_nb15/full_dataset/preprocessing_report.json` | Counts, exclusions, labels and feature order |
| `data/unsw_nb15/artifacts/` | Training/validation/test tensors, preprocessing state and Edge partitions |

Download and clean from the repository root:

```sh
python3 scripts/ops/download_datasets.py --only unsw
```

Select `data/unsw_nb15/full_dataset/intrusion_records.csv` as `dataset_path` in `config/training.yaml`. The launcher's artifact builder handles experiment splitting, scaling, PCA and configured training resampling. It rebuilds incompatible cached artifacts automatically. The full download does not overwrite or merge with the shipped snippet. See [dataset preparation](../dataset-preparation.md) for the complete policy and preprocessing-only commands.

## Download (Full Dataset)

The dataset is distributed by the University of New South Wales Canberra
Cyber group:

- **Landing page**: [UNSW-NB15 (UNSW Canberra)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Official file archive (replaces retired CloudStor)**: [UNSW SharePoint folder](https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag) — pcap, BRO, Argus, and CSV variants
- **IEEE DataPort (subscription required)**: [UNSW-NB15 train/test CSVs](https://ieee-dataport.org/documents/unswnb15-dataset)
- **Direct CSV (UQ NetFlow variant, no login)**: [NF-UNSW-NB15-v2 (43-feature CSV)](https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848)

The downloader cleans the four original headerless `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` files from its configured mirror. Prepared training/testing files are downloaded as references and do not add rows. The artifact builder creates the experiment's train/validation/test splits. Invalid numeric cells are replaced with zero using the authors' conversion policy; labels and row widths are validated strictly. See [dataset preparation](../dataset-preparation.md). The NetFlow variant linked above has a different schema and is not an interchangeable input to this cleaner.

## Feature Set

The source schema includes network-layer statistics, connection attributes and label fields. Input measurements include:
- Flow identifiers (source/destination IP, port, protocol, state)
- Volume metrics (bytes, packets, load, window size, jitter)
- Connection timing (duration, inter-packet gaps, RTT, SYN/ACK timing)
- Service and state classification flags
- Time-window aggregation counters (e.g., `ct_srv_src`, `ct_dst_ltm`)

## Citation

> N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection
> Systems," in *Proc. MilCIS*, 2015, pp. 1–6.
