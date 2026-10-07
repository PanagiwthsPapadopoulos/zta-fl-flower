import logging
import os
import torch
from functools import lru_cache

logger = logging.getLogger("zta.hardware")


@lru_cache(maxsize=1)
def _gpu_enabled_tiers() -> tuple:
    """Reads `gpu_enabled_tiers` from YAML once per process lifetime.

    Caching is safe: every configuration change goes through deploy_code,
    which ships a fresh bundle and restarts the ClientApp/ServerApp
    processes, so the next process picks up the new value.
    """
    try:
        from src.shared.utils.config_loader import load_yaml_configs
        tiers = load_yaml_configs().get("gpu_enabled_tiers", []) or []
        return tuple(str(t).lower() for t in tiers)
    except Exception:
        return ()


def _current_tier() -> str:
    """Reads the tier identity injected by the boot script via ZTA_TIER."""
    return os.getenv("ZTA_TIER", "unknown").lower()


def _host_gpu_available() -> bool:
    """Reads the host-capability flag injected by the boot script.

    Defaults to True when unset so that ad-hoc invocations (tests,
    notebooks, direct python -m ...) behave as if the host had a GPU
    and only the tier whitelist mattered.
    """
    return os.getenv("ZTA_GPU_AVAILABLE", "true").lower() == "true"


def get_device() -> str:
    """Returns the torch device string for the current process.

    Policy, evaluated in order:
      1. Tier not in `gpu_enabled_tiers`            -> "cpu"
      2. Host flagged as GPU-less by the boot script -> "cpu" (+ debug msg)
      3. CUDA probe returns True                    -> "cuda"
      4. MPS probe returns True                     -> "mps"
      5. Otherwise                                  -> "cpu"
    """
    tier = _current_tier()
    enabled_tiers = _gpu_enabled_tiers()

    if tier not in enabled_tiers:
        return "cpu"

    if not _host_gpu_available():
        logger.debug(
            "[hardware] tier=%s is whitelisted in gpu_enabled_tiers, "
            "but this machine has no GPU support (ZTA_GPU_AVAILABLE=false). "
            "Falling back to CPU.",
            tier,
        )
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dynamic_thread_limit(estimated_task_vram_gb: float = 1.0, fallback_limit: int = 1) -> int:
    """Calculates safe concurrent GPU/CPU workers to prevent memory crashes."""
    device = get_device()
    try:
        if device == "cuda":
            free_mem, _ = torch.cuda.mem_get_info()
            return max(1, int((free_mem / (1024 ** 3) - 1.0) // estimated_task_vram_gb))
        elif device == "mps":
            return 2
        else:
            return max(1, os.cpu_count() // 2)
    except Exception:
        return fallback_limit


def get_dataloader_kwargs() -> dict:
    """Centralizes optimal DataLoader memory and worker configurations."""
    return {
        "num_workers": 0,
        "pin_memory": get_device() != "cpu",
    }
