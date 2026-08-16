"""Global configuration for the time_serials modern control-theory pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

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
    DECISION_COLS: Tuple[str, ...] = ("x3", "x4", "x6", "x8")  # for MPC
    FIXED_INPUT_COLS: Tuple[str, ...] = ("x1", "x2", "x5")  # held fixed in MPC

    # ---- Variable ranges observed in data (used as MPC bounds & sanity checks) ----
    VAR_RANGES: dict = field(default_factory=lambda: {
        "x1": (6.0, 8.5),
        "x2": (370.0, 445.0),
        "x3": (0.0, 110.0),
        "x4": (26.0, 36.0),
        "x5": (0.0, 110.0),
        "x6": (5_000.0, 50_000.0),
        "x7": (0.0, 150_000.0),  # monotonic increasing cumulative
        "x8": (0.0, 1500.0),
        "y1": (0.0, 100.0),
        "y2": (0.0, 500.0),
        "y3": (0.0, 12_000.0),
        "y4": (0.0, 10_000.0),
    })

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

    # ---- Optimization ----
    OPT_HORIZON: int = 16
    OPT_WEIGHTS: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0),
    ])
    OPT_N_STARTS: int = 5
    OPT_LR: float = 0.05
    OPT_MAX_ITER: int = 200

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