import os
import torch

def get_device() -> str:
    """Cross-platform hardware detection (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dynamic_thread_limit(estimated_task_vram_gb: float = 1.0, fallback_limit: int = 1) -> int:
    """Calculates safe concurrent GPU/CPU workers to prevent memory crashes."""
    try:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            return max(1, int((free_mem / (1024 ** 3) - 1.0) // estimated_task_vram_gb))
        elif torch.backends.mps.is_available():
            return 2  # MPS shares system memory, keep conservative
        else:
            return max(1, os.cpu_count() // 2)
    except Exception:
        return fallback_limit

def get_dataloader_kwargs() -> dict:
    """Centralizes optimal DataLoader memory and worker configurations."""
    return {
        "num_workers": 0, 
        "pin_memory": get_device() != "cpu"
    }