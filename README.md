# Zero-Trust Agentic Federated Learning (Flower Implementation)

A Flower implementation of the [ZTA-FL framework](https://arxiv.org/abs/2512.23809) for Industrial IoT intrusion detection. It combines an Edge–Fog–Cloud training hierarchy, software TPM attestation, SHAP-based filtering and aggregation, and adversarial training.

Read the [architecture parity report](docs/architecture-parity.md) for the relationship to the paper and authors' implementation, disclosed deviations and implementation limits.

## Guides

| Guide | Purpose |
| --- | --- |
| [Architecture parity](docs/architecture-parity.md) | Component behavior, evidence and deviations from the reference work |
| [Deployment](docs/deployment.md) | Run the distributed three-tier network with Docker or Apptainer |
| [Simulation](docs/simulation.md) | Run the Ray-based simulation and external software TPMs |
| [Overnight batches](docs/batch-runs.md) | Run sequential experiments on one persistent simulation environment |
| [Dataset preparation](docs/dataset-preparation.md) | Download, clean and select full datasets; understand preprocessing choices |
| [Testing](docs/testing.md) | Run component tests and interpret their evidence and limits |
| [Preflight runs](docs/preflight.md) | Exact live test settings and acceptance criteria before long experiments |
| [Monitoring and results](docs/monitoring-and-results.md) | Follow live progress, inspect logs and produce plots/security metrics |

## Architecture and execution modes

Edges train local intrusion-detection models. Fogs verify attestation, evaluate SHAP stability and aggregate eligible updates. The Cloud combines Fog updates, validates candidate models and reports global metrics.

| | Deployment | Simulation |
| --- | --- | --- |
| Runtime | Flower SuperLink, SuperNode and SuperExec services | Flower simulation with Ray actors |
| Cloud/Fog execution | Separate services with inter-tier communication | Central engine with in-process Fog aggregation |
| Edge execution | Separate client services | Ephemeral Ray actors |
| Software TPMs | One emulator per Edge | External emulators used by the Edge actors |
| Platforms | Docker Compose and Apptainer | Docker Compose and Apptainer |

The project launchers mount local source, scripts and configuration into the existing runtime images. Code and documentation changes do not require rebuilding or publishing images. Stop active runs before changing source, then boot and dispatch again. Dependency or system-tool changes may still require updated images. The runtime-specific guides explain the mounts and startup steps.

## Repository layout

```text
zta-fl-flower/
├── config/              # Network, training and security YAML
├── data/                # Shipped snippets, downloaded full datasets and training artifacts
├── docker/              # Image definitions and generated Compose configuration
├── docs/                # Operational guides, parity report and dataset references
├── images/              # Local Apptainer runtime images
├── logs/                # Live node and system logs
├── results/             # Metrics, checkpoints and batch archives
├── runtime/             # TPM state, certificates, locks and run state
├── scripts/
│   ├── deployment/      # Distributed network boot and dispatch
│   ├── simulation/      # Simulation boot and dispatch
│   ├── setup/           # Dataset artifacts, topology and provisioning
│   ├── ops/             # Downloads, batches, monitoring, plotting and evaluation
│   └── tests/           # Automated tests and recorded evidence
├── src/                 # Shared logic and Edge, Fog and Cloud implementations
└── pyproject.toml       # Project dependencies and Flower application definition
```

## Prerequisites and configuration

Install Docker Compose or Apptainer, Python and OpenSSL before following the relevant run guide. The project pins Flower to `1.30.0`; the [deployment guide](docs/deployment.md) includes host setup and CLI checks. The [test guide](docs/testing.md) covers the additional ML dependencies needed for source tests.

Configuration lives in `config/`: `network.yaml` defines topology and transport, `training.yaml` selects data and learning settings, and `security.yaml` selects defenses and attack scenarios. These settings describe an experiment; they do not establish which architectural capabilities exist. Keep topology fixed within an overnight batch.

## Datasets

| Dataset | Reference |
| --- | --- |
| Edge-IIoTset | [Data layout, label compatibility and citation](docs/datasets/edge-iiotset.md) |
| CIC-IDS2017 | [Data layout, flow features and citation](docs/datasets/cic-ids2017.md) |
| UNSW-NB15 | [Data layout, full-data source selection and citation](docs/datasets/unsw-nb15.md) |

The shipped snippets remain under `data/<dataset>/raw/`. The downloader writes separately under `data/<dataset>/full_dataset/`; training artifacts remain under `data/<dataset>/artifacts/`. Follow [dataset preparation](docs/dataset-preparation.md) to select a full dataset and rebuild incompatible artifacts automatically.

## Acknowledgments and References

This repository is built using the open-source **Flower** federated learning framework:
* [Flower (flwr) GitHub Repository](https://github.com/adap/flower)
* [Flower Official Documentation](https://flower.dev/)

The architecture, logic, and experimental design implemented here are derived from the original **ZTA-FL** paper. If this architecture and methodology helps your research, please refer to and cite the original authors:

```bibtex
@article{singh2025ztafl,
  title   = {Zero-Trust Agentic Federated Learning for Secure {IIoT} Defense Systems},
  author  = {Singh, Samaresh Kumar and Roy, Joyjit and So, Martin},
  journal = {arXiv preprint arXiv:2512.23809},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.23809},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
