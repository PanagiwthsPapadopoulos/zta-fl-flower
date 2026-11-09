"""Simulation-only configuration; deployment GPU locks are deliberately unrelated."""
import math
import os
from pathlib import Path
import tomllib


def simulation_config(flower_config):
    from src.shared.utils.config_loader import get_merged_config
    config = get_merged_config(flower_config)
    # Queueing virtual clients must not inherit the deployment freshness budget.
    window = float(config.get("simulation_tpm_freshness_window", 120))
    if not math.isfinite(window) or window < 120:
        raise ValueError("simulation_tpm_freshness_window must be at least 120 seconds")
    config["tpm_freshness_window"] = max(float(config["tpm_freshness_window"]), window)
    return config


def cpu_budget():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        available = min(available, int(os.environ["SLURM_CPUS_PER_TASK"]))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            available = min(available, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    # Leave a CPU for Flower supervision, the coordinator, and TPM services.
    return max(1, available - 1)


def federation_config(manifest, clients, edge_device, available_cpus):
    if clients <= 0 or available_cpus <= 0:
        raise ValueError("Simulation requires positive client and CPU counts")
    local = manifest.get("tool", {}).get("flwr", {}).get("federations", {}).get("local", {})
    opts = local.get("options", local)
    backend = opts.get("backend", {})
    cres = backend.get("client-resources", opts.get("client-resources", {}))
    iargs = backend.get("init-args", opts.get("init-args", {}))
    cpus = float(cres.get("num-cpus", min(2, available_cpus)))
    total_cpus = float(iargs.get("num-cpus", available_cpus))
    gpus = float(cres.get("num-gpus", 1.0 if edge_device == "cuda" else 0.0))
    if not all(math.isfinite(v) for v in (cpus, total_cpus, gpus)):
        raise ValueError("Simulation resources must be finite")
    if not 0 < cpus <= total_cpus <= available_cpus:
        raise ValueError("Simulation CPU requests exceed the available worker budget")
    if edge_device == "cuda" and not 0 < gpus <= 1:
        raise ValueError("CUDA simulation clients must reserve a positive GPU fraction (at most 1)")
    if edge_device != "cuda" and gpus != 0:
        raise ValueError("CPU simulation clients cannot reserve GPUs")
    config = {"num-supernodes": clients, "client-resources-num-cpus": cpus,
              "client-resources-num-gpus": gpus, "init-args-num-cpus": total_cpus}
    if "num-gpus" in iargs:
        total_gpus = float(iargs["num-gpus"])
        if not math.isfinite(total_gpus) or total_gpus < gpus or total_gpus < 0:
            raise ValueError("Invalid simulation GPU capacity")
        config["init-args-num-gpus"] = total_gpus
    if "verbose" in opts:
        config["verbose"] = str(opts["verbose"]).lower()
    return " ".join(f"{key}={value}" for key, value in config.items())


if __name__ == "__main__":
    from src.shared.utils.hardware import get_device
    manifest = tomllib.loads(Path("/app/pyproject.toml").read_text())
    print(federation_config(manifest, int(os.environ["N_SIM"]), get_device(tier="edge"), cpu_budget()))
