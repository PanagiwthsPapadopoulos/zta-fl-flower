"""Stable experiment seeds, independent of worker scheduling and Python hash salt."""
import hashlib
import random

import numpy as np
import torch


def derive_seed(master_seed: int, *parts: object) -> int:
    value = ":".join(str(p) for p in (master_seed, *parts))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def seed_training(master_seed: int, client_index: int, round_num: int) -> int:
    seed = derive_seed(master_seed, "training", client_index, round_num)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Stable algorithm selection; cross-device bitwise equality is not promised.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed
