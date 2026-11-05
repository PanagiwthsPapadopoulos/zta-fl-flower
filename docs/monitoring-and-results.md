# Monitoring and results

Run the commands below from the repository root. Both deployment and simulation use the same monitoring and plotting scripts. Install the host dependencies with `python3 -m pip install -r requirements.txt`, or use the existing runtime images as shown below.

## Live pipeline display

```sh
# Append-only snapshots when progress changes:
python3 scripts/ops/verify_pipeline.py -w

# Full-screen display in an interactive terminal:
python3 scripts/ops/verify_pipeline.py --live
```

The display follows phase markers in the node logs. It helps locate stalled progress; a completed display alone does not validate attack effectiveness, metric fidelity or every security property. Inspect the node logs and saved results as well. For component checks before a live run, see [testing](testing.md).

To use the image's Python dependencies with the current local monitor:

```sh
# Docker: separate read-only monitoring container, with no network boot required.
docker run --rm -it --entrypoint /python/venv/bin/python3 \
  --workdir /app --env PYTHONPATH=/app \
  --mount "type=bind,source=$PWD,target=/app,readonly" \
  panagiotispapadopoulos/zta-cloud-node:latest \
  scripts/ops/verify_pipeline.py -w

# Apptainer:
apptainer exec --bind "$PWD:/app:ro" --pwd /app --env PYTHONPATH=/app \
  images/cloud_node.sif /python/venv/bin/python3 scripts/ops/verify_pipeline.py -w
```

Both commands use the local script, regardless of the version bundled in the image. `--live` needs an interactive terminal; use `-w` for terminals that do not handle the full-screen display well.

## Logs

```sh
tail -F logs/nodes/cloud/cloud_server.jsonl
tail -F logs/nodes/fog_1/fog_1_server.jsonl
tail -F logs/nodes/edge_1_1/edge_1_1.jsonl
tail -F logs/system/run_cloud.log
```

Deployment also writes per-fog submission streams such as `logs/system/run_fog1.log`. Simulation dispatches one central run. System boot logs are under `logs/system/`.

Standalone launchers clear previous logs during startup or dispatch, so copy needed evidence before restarting. The [batch runner](batch-runs.md) archives each attempt's logs and runtime state automatically after stopping it.

## Saved outputs

Each training run writes to `results/<experiment_name>/`:

| Output | Contents |
| --- | --- |
| `<experiment_name>.json` | Run metadata and per-round metrics, including accuracy, macro-F1, loss, latency and payload size |
| `global_model.pt`, `global_model_round_<r>.pt` | Final/latest and periodic model checkpoints |
| `trustdb/fog_<i>_state.json` | Attestation outcomes, roles, trust scores and quarantine state |
| `shap_scores/fog_<i>_state.json` | Per-node SHAP stability scores and filtering/reward outcomes |
| `performance_metrics/fog_<i>_state.json` | Recorded training, attestation, SHAP and aggregation timings |

Managed simulations also record lifecycle status, resolved runtime settings and saved-round progress. See [batch evidence](batch-runs.md) for completion criteria, crash/deadline outcomes and archive contents. A partial result directory can remain after a failed run; its existence does not establish completion.

## Plots and security evaluation

```sh
# Plot a particular run:
python3 scripts/ops/plot_metrics.py results/<experiment_name>

# Plot the newest directory under results/:
python3 scripts/ops/plot_metrics.py

# Select panels:
python3 scripts/ops/plot_metrics.py results/<experiment_name> --panels loss asr pgd

# Compute additional security metrics for a saved run:
python3 scripts/ops/evaluate_security.py '<experiment_name>'
```

Available panels are `accuracy_f1`, `loss`, `asr`, `pgd` and `fgsm`. Plots are saved inside the selected run directory. Supply an explicit run path when using batch archives, since automatic selection chooses the newest directory under `results/`, which may be `batch_archives/`.

Security evaluation uses the saved model and dataset artifacts and writes `heavy_metrics.json`. Undefined measurements remain explicit; missing evidence is not a zero error rate. Consult the [architecture parity report](architecture-parity.md) for metric definitions and limitations.
