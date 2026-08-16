"""Visualization helpers — single-source palette, headless-friendly.

Each ``plot_*`` function saves a PNG at 150 dpi.  Color palette follows a
qualitative scheme inspired by the dataviz skill defaults — `#4C72B0`
(blue), `#DD8452` (orange), `#55A467` (green), `#C44E52` (red).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ("#4C72B0", "#DD8452", "#55A467", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C")
DPI = 150


def _save(fig, out_path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Prediction overlay
# --------------------------------------------------------------------------- #
def plot_prediction_overlay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    sample_idx: int,
    out_path: str | Path,
    title: Optional[str] = None,
) -> None:
    """4-panel line plot of y1..y4 (true vs predicted) for one sample.

    y_true / y_pred / mask : (N, T, 4)
    """
    yt = y_true[sample_idx]
    yp = y_pred[sample_idx]
    mk = mask[sample_idx]

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    names = ("y1", "y2", "y3", "y4")
    T = yt.shape[0]
    for j, name in enumerate(names):
        ax = axes[j]
        ax.plot(np.arange(T), yt[:, j], color=PALETTE[0], lw=1.2, label="true")
        ax.plot(np.arange(T), yp[:, j], color=PALETTE[1], lw=1.2, ls="--",
                 label="predicted")
        # Highlight observed points
        obs_idx = np.where(mk[:, j])[0]
        if len(obs_idx) > 0:
            ax.scatter(obs_idx, yt[obs_idx, j], s=18, color=PALETTE[2],
                        zorder=5, label="observed")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Timestep")
    if title:
        fig.suptitle(title)
    _save(fig, out_path)


def plot_prediction_overlay_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    sample_indices: Iterable[int],
    out_path: str | Path,
) -> None:
    """Plot several samples (one column each) with 4 rows (y1..y4)."""
    sample_indices = list(sample_indices)
    n = len(sample_indices)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 10), sharex=True)
    if n == 1:
        axes = axes.reshape(4, 1)
    names = ("y1", "y2", "y3", "y4")
    for col, idx in enumerate(sample_indices):
        yt = y_true[idx]; yp = y_pred[idx]; mk = mask[idx]
        for j, name in enumerate(names):
            ax = axes[j, col]
            ax.plot(yt[:, j], color=PALETTE[0], lw=1.0, label="true")
            ax.plot(yp[:, j], color=PALETTE[1], lw=1.0, ls="--", label="pred")
            ax.set_title(f"sample {idx} — {name}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0, col].legend(fontsize=7)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Anomaly / preprocessing summary
# --------------------------------------------------------------------------- #
def plot_anomaly_summary(
    sample_x: np.ndarray,
    anomaly_mask: np.ndarray,
    out_path: str | Path,
    sample_id: str = "",
) -> None:
    """Plot all 8 x columns with anomalous points highlighted."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    names = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
    for j in range(8):
        ax = axes[j // 2, j % 2]
        ax.plot(sample_x[:, j], color=PALETTE[0], lw=1.0, label="x")
        ax.scatter(np.where(anomaly_mask[:, j])[0],
                    sample_x[anomaly_mask[:, j], j],
                    color=PALETTE[3], s=24, marker="x", label="anomaly")
        ax.set_title(names[j], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Anomaly summary — {sample_id}")
    _save(fig, out_path)