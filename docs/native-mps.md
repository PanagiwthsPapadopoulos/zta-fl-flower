# Native Flower training on Apple Silicon

Run from the repository with its native ARM64 Python environment:

```bash
.venv/bin/python scripts/simulation/run_native_mps.py --smoke-test
```

The smoke test copies configuration, selects one Fog and two Edges, and runs two
rounds with PGD training. It retains real TPM attestation, SHAP filtering, CPU
INT8 inference, global evaluation and checkpoint saving. It is an integration
check, not evidence of model accuracy or security effectiveness at scale.

To use the current experiment configuration instead:

```bash
.venv/bin/python scripts/simulation/run_native_mps.py
```

Requirements: native macOS with working PyTorch MPS, the project's Python
dependencies (including Flower 1.30.0 and Ray), running Docker, and the existing
`panagiotispapadopoulos/zta-edge-node:latest` image. No macOS TPM tools are needed.
The selected configuration must enable the cloud, fog and edge GPU tiers.

The launcher creates a unique directory under `runtime/native-mps/`. Configuration,
derived dataset artifacts, TPM state, logs, Flower state and results stay there;
raw data is referenced by symlink. The source configuration and original dataset
artifacts are not overwritten. The supplied `--config-dir` selects another config
directory; relative dataset paths are resolved against the repository root.

Flower's SuperLink, simulation server and Ray clients run natively. Only software
TPMs and their attestation/verification commands run in Docker. The bridge calls
the existing TPM implementations through `docker exec`, with JSON over stdin and
stdout. TPM ports are not published, and insecure attestation bypass is disabled.
Container names and the SuperLink endpoint are unique to the run. Normal exit,
errors and Ctrl+C stop that run's processes and containers; state and logs remain.

The model definition, layers, weights and calculations are unchanged. Opt-in
hooks place the existing adaptive-pooling operation on CPU and return its output
to MPS, retaining autograd. This avoids the installed MPS backend's restriction
on pooling from length 1 to length 4. Hooks are installed at model creation in
every Flower worker and survive the CPU inference copies. Automatic MPS fallback
is disabled so other unsupported operations fail visibly.

One Ray client worker is used by default. `--workers 2` permits two concurrent
clients, sharing the Mac's memory and GPU. Up to one worker per CPU core is supported;
CPU reservations adjust so Ray never advertises more cores than the host has.
Ray GPU reservations remain zero;
PyTorch selects MPS inside each worker. The existing Docker/NVIDIA and Apptainer
launchers do not set the native flags and retain their original behavior.

The printed run directory contains `flower.log`, `superlink.log`, `tpm.log`,
`prepare.log`, and `native_status.json`. Per-node logs are under `logs/` and
metrics/checkpoints under `results/<run-name>/`. `COMPLETED` means Flower returned
and the final configured round was saved; inspect regional metrics for rejected
or failed clients, particularly in adversarial experiments.

Regression tests:

```bash
.venv/bin/python scripts/tests/run_tests.py --pattern test_native_mps.py
.venv/bin/python scripts/tests/run_tests.py
```

The MPS-specific test skips when MPS is unavailable. The live `--smoke-test` runs
the actual Flower CLI and TPM containers rather than mocks.

TPM transport failures include bounded, redacted container stderr. Cleanup attempts
process shutdown, TPM log collection and container removal independently.
`native_status.json` records `cleanup_errors`; a completed training run with cleanup
errors exits nonzero so an unattended queue cannot silently continue.
