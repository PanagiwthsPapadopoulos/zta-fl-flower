# Native Flower training on Apple Silicon

Install all dependencies from the repository root in a native ARM64 Python 3.11 environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` installs the project dependencies from `pyproject.toml` and the
host tools; no separate Captum, TorchMetrics or Flower install is needed. These
are dependency ranges, not a full version lock. Preserve resolved versions when
matching another machine's experiment environment.

Run from the repository with that environment:

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

## Sequential MPS batches

Use the same experiment-folder format as the Docker/Apptainer queue. Each entry
starts from a frozen copy of the repository's three YAML configs, then applies
that experiment's overrides. Source files and archived experiment configs stay
unchanged. Run from the repository root:

```bash
# Inspect all resolved configurations without starting training or Docker.
.venv/bin/python scripts/ops/batch_runs_mps.py \
  --queue experiments_archived/paper_80 --workers 2 --dry-run

# Run the queue sequentially with native MPS and TPM-only Docker.
.venv/bin/python scripts/ops/batch_runs_mps.py \
  --queue experiments_archived/paper_80 --workers 2
```

One worker is the default. More workers share unified memory and the same GPU;
maximum concurrency can increase swap and make training slower. Choose the
worker count for the host; it does not change the model, batch size, epochs,
seeds, aggregation or security policies. This command runs every configuration
in the selected queue, including entries previously run outside this batch.
Use a separate queue folder containing only the desired configurations to
exclude earlier experiments or divide independent experiments among machines.

Each experiment starts a fresh native launcher, Flower/Ray processes and TPM
containers. This deliberately avoids retaining worker/GPU state between runs.
Different topologies are supported because each experiment provisions its own
TPMs. Raw data is symlinked, but derived artifacts are prepared separately for
each run. Allow disk space for retained artifacts and checkpoints.

The printed batch archive is `results/native_batch_archives/<session>/`. Its
`batch.json` stores the frozen configurations, settings, statuses, completed
rounds and paths to the native runs. `attempts/<attempt>/` contains the config
snapshot, status and `launcher.log`. Full native logs and results remain under
`runtime/native-mps/<attempt>/`.

`SUCCESS` requires a zero launcher exit, confirmed cleanup, completed lifecycle,
the requested final saved round and a nonempty checkpoint. Persisted missing
responses or client training crashes give `DEGRADED`; security exclusions alone
do not. The queue stops on any non-success result. It does not change or retry
training automatically.

```bash
# Resume pending entries, preserving successes and recorded settings.
.venv/bin/python scripts/ops/batch_runs_mps.py \
  --resume results/native_batch_archives/<session>

# Explicitly retry unsuccessful entries from round one, retaining attempt history.
.venv/bin/python scripts/ops/batch_runs_mps.py \
  --resume results/native_batch_archives/<session> --retry-failed
```

Resume uses the saved settings/configuration and the current source code and
Python environment. Keep the same revision and dependencies for reproducibility;
CLI worker/timeout/image/smoke settings apply when creating a new batch.
A session lock prevents two supervisors from executing the same archive.
Ctrl+C or `--run-timeout` (default 172800 seconds, including preparation) requests
native cleanup. `--shutdown-timeout` defaults to 180 seconds. If cleanup cannot
be confirmed, the queue refuses to proceed or retry: inspect the native status,
processes and generated Compose file before recovery. Hard-killed supervisors
are not automatically recovered when their launcher's cleanup is unknown.

For an explicit integration smoke test, select a small queue and add
`--smoke-test`. Each entry then uses the existing two-round, two-client smoke
profile; the archive records this setting. Do not use smoke results as full
experiment results. `--run-dir` on the single-run launcher is used internally to
assign unique output paths and refuses an existing directory.

## Empty TPM image / early startup failure

Use an image name directly with `--tpm-image`, or omit the option to use the
default image. Passing `--tpm-image "$TPM_IMAGE"` from a new shell where that
variable is unset passes an empty string. Both native launchers reject empty
image arguments before creating a run.

Older batches may report unconfirmed cleanup when `docker image inspect ''`
failed before the run directory was created. Inspect `launcher.log` to establish
the cause; missing status alone does not establish that resources were never
started. If the first experiment failed at that image check, no TPMs or training
were started: retain the failed archive and start a new batch with the correct
image. Resuming uses the frozen image setting, so it would reuse the empty value.
