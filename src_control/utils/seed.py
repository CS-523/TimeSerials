"""Random seed helper for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + GPU) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism — costs a bit of speed, gains reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False