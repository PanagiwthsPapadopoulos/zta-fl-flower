# Edge-IIoTset Dataset

## Overview

Edge-IIoTset was developed for centralized and federated intrusion-detection research using a purpose-built IoT/IIoT testbed. The publication describes seven layers, more than ten device types, and fourteen attacks grouped into DoS/DDoS, information gathering, man-in-the-middle, injection and malware threats. See the [dataset publication](https://doi.org/10.1109/ACCESS.2022.3165809) for the testbed equipment and collection methodology.

## Shipped snippet label mapping (15 classes)

The table below describes the legacy snippet labels. The cleaned full dataset uses a separate 15-class mapping, including `DDoS_TCP`, `DDoS_UDP`, `DDoS_HTTP`, `DDoS_ICMP` and `MITM`. These names are not interchangeable with the snippet taxonomy. Both mappings are documented in [dataset preparation](../dataset-preparation.md) and implemented in the [loader](../../src/shared/data/data_loader.py) and [full-data schema](../../src/shared/data/dataset_schema.py).

| Class | Description |
|-------|-------------|
| Normal | Legitimate IIoT device communication |
| DoS_TCP | TCP-based denial-of-service flooding |
| DoS_UDP | UDP flood targeting IIoT endpoints |
| Scanning | Network reconnaissance and host discovery |
| MITM_Attack | Legacy man-in-the-middle label |
| Fingerprinting | Device fingerprinting via crafted probes |
| Password | Password attack label |
| Port_Scanning | SYN scan and version detection sweeps |
| Ransomware | Ransomware label |
| Backdoor | Backdoor label |
| Vulnerability_Scanner | Automated vulnerability scanner probes |
| Upload | Upload attack label |
| SQL_Injection | SQL injection label |
| XSS | Cross-site scripting label |
| MITM_ARP | ARP spoofing and cache poisoning |

## Repository data layout

All paths below are relative to the repository root.

| Path | Purpose |
| --- | --- |
| `data/edge_iiotset/raw/network_traffic_samples.csv` | Shipped small snippet for quick checks |
| `data/edge_iiotset/full_dataset/downloads/` | Downloaded source files |
| `data/edge_iiotset/full_dataset/network_traffic_samples.csv` | Cleaned full data: 42 numeric features and a target column |
| `data/edge_iiotset/full_dataset/preprocessing_report.json` | Counts, exclusions, labels and feature order |
| `data/edge_iiotset/artifacts/` | Training/validation/test tensors, preprocessing state and Edge partitions |

Download and clean from the repository root:

```sh
python3 scripts/ops/download_datasets.py --only edge
```

Select `data/edge_iiotset/full_dataset/network_traffic_samples.csv` as `dataset_path` in `config/training.yaml`. The launcher's artifact builder handles experiment splitting, scaling, PCA and configured training resampling. It rebuilds incompatible cached artifacts automatically. The full download does not overwrite or merge with the shipped snippet. See [dataset preparation](../dataset-preparation.md) for the complete policy and preprocessing-only commands.

## Download (Full Dataset)

The full Edge-IIoTset corpus is hosted on IEEE DataPort:

- **Landing page**: [IEEE DataPort — Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
- **Mirror (Kaggle, no login)**: [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
- **Companion paper (DOI 10.1109/ACCESS.2022.3165809)**: [IEEE Access](https://ieeexplore.ieee.org/document/9751703/)

The downloader uses the configured Kaggle source and combines its DNN and ML CSV files, removing identical original records. See [dataset preparation](../dataset-preparation.md) for numeric exclusions and invalid-row handling.

## Feature Extraction

The publication describes feature extraction using Zeek and TShark from multiple data sources. The repository consumes the selected CSV exports rather than reconstructing raw captures. Its cleaner retains 42 numeric features; the builder then produces the model input described in [dataset preparation](../dataset-preparation.md). The repository's numeric subset should not be confused with the complete feature set reported in the [publication](https://doi.org/10.1109/ACCESS.2022.3165809).

## Citation

If you use this data in your work, please cite the original Edge-IIoTset publication:

> M. A. Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT
> and IIoT Applications for Centralized and Federated Learning," *IEEE Access*, vol. 10,
> pp. 40281–40306, 2022.
