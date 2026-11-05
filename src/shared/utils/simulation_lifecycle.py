"""Explicit completion evidence for supervised simulation batches."""
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import traceback


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def track_simulation(config, flower_config):
    """Normal return and exceptions are distinct; a killed process leaves STARTED."""
    attempt = config.get("batch_attempt")
    if not attempt:
        yield
        return
    name = str(config["run_name"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or attempt != name:
        raise ValueError("Batch attempt and safe run_name must match")
    folder = Path("results") / name
    versions = {}
    for package in ("flwr", "torch", "numpy", "captum", "scikit-learn", "imbalanced-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    env_keys = ("TIER", "ZTA_TIER", "ZTA_GPU_AVAILABLE", "NUM_FOGS", "N_SIM",
                "FLWR_HOME", "RAY_TMPDIR", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS",
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV", "RAY_INCLUDE_DASHBOARD",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "TZ")
    atomic_json(folder / "resolved_runtime.json", {
        "attempt": attempt, "resolved_config": config, "flower_run_config": dict(flower_config),
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "environment": {key: os.environ[key] for key in env_keys if key in os.environ},
    })
    status = {"attempt": attempt, "status": "STARTED", "started_at": stamp()}
    atomic_json(folder / "batch_lifecycle.json", status)
    try:
        yield
    except BaseException:
        status.update(status="FAILED", finished_at=stamp(), traceback=traceback.format_exc())
        atomic_json(folder / "batch_lifecycle.json", status)
        raise
    else:
        status.update(status="COMPLETED", finished_at=stamp())
        atomic_json(folder / "batch_lifecycle.json", status)


def record_saved_round(metadata, round_number):
    """Called only after metrics and model writes have both finished."""
    attempt = metadata.get("batch_attempt")
    if attempt:
        atomic_json(Path("results") / attempt / "batch_progress.json", {
            "attempt": attempt, "completed_round": round_number, "saved_at": stamp(),
        })
