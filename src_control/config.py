"""Global configuration for the time_serials modern control-theory pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Config:
    ROOT: str = "/kefu-nas/ybkong/time_serials-master"
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- I/O paths ----
    DATA_SUBDIRS: Tuple[int, ...] = (1, 2, 3, 4, 5)
    DATA_PROCESSED_DIR: str = "data/processed"
    CHECKPOINT_DIR: str = "checkpoints"
    RESULTS_DIR: str = "results"
    FIGURES_DIR: str = "results/figures"
    METRICS_DIR: str = "results/metrics"
    PREDICTIONS_DIR: str = "results/predictions"

    # ---- Preprocess ----
    SEQ_LEN: int = 320   # > p95 of T across the dataset; truncate + pad to this
    Z_THRESHOLD: float = 5.0
    TRAIN_RATIO: float = 0.8
    X_COLS: Tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
    Y_COLS: Tuple[str, ...] = ("y1", "y2", "y3", "y4")

    # ---- Model ----
    N_STATE: int = 16         # linear SS state dim
    HIDDEN: int = 128
    MAMBA_STATE: int = 64
    MAMBA_STRUCT: int = 32

    # ---- Training ----
    LR: float = 1e-3
    LR_MIN: float = 1e-5
    BS: int = 16
    EPOCHS: int = 200
    TEACHER_FORCING_DECAY: int = 50
    ROLLOUT_HORIZON: int = 32
    X_FORECAST_CONTEXT: int = 32   # context length for the x1–x8 forecaster
    PATIENCE: int = 30
    GRAD_CLIP: float = 1.0
    VAL_RATIO: float = 0.15

    # ---- Plotting ----
    FIGSIZE: Tuple[int, int] = (8, 5)
    DPI: int = 150


def get_config() -> Config:
    """Return a fresh Config instance (callers should treat it as immutable)."""
    return Config()


def resolve_paths(cfg: Config) -> Config:
    """Resolve all relative paths against ROOT and create directories if missing."""
    for attr in (
        "DATA_PROCESSED_DIR",
        "CHECKPOINT_DIR",
        "FIGURES_DIR",
        "METRICS_DIR",
        "PREDICTIONS_DIR",
    ):
        path = getattr(cfg, attr)
        if not os.path.isabs(path):
            setattr(cfg, attr, os.path.join(cfg.ROOT, path))
        os.makedirs(getattr(cfg, attr), exist_ok=True)
    return cfg