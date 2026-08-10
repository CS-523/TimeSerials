"""Visualization helpers — single-source palette, headless-friendly.

Each ``plot_*`` function saves a PNG at 150 dpi.  Color palette follows a
qualitative scheme inspired by the dataviz skill defaults — `#4C72B0`
(blue), `#DD8452` (orange), `#55A467` (green), `#C44E52` (red).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

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
# MPC / Pareto
# --------------------------------------------------------------------------- #
def plot_pareto(
    points: list[tuple[float, float]],
    baseline: Optional[Tuple[float, float]],
    out_path: str | Path,
    title: str = "Pareto frontier (y4_sum vs Y)",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if points:
        arr = np.asarray(points)
        ax.scatter(arr[:, 0], arr[:, 1], c=PALETTE[0], s=40, alpha=0.7,
                    label="optimized")
    if baseline is not None:
        ax.scatter([baseline[0]], [baseline[1]], c=PALETTE[3], s=160,
                    marker="*", zorder=10, label="baseline (last-input policy)")
    ax.set_xlabel("Σ y4 over horizon")
    ax.set_ylabel("Predicted final Y")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_optimized_trajectory(
    u_opt: np.ndarray,
    u_base: np.ndarray,
    y_opt: np.ndarray,
    y_base: np.ndarray,
    sample_idx: int,
    out_path: str | Path,
    decision_names: Tuple[str, ...] = ("x3", "x4", "x6", "x8"),
) -> None:
    """Two-panel: decision vars and y4 trajectories for opt vs baseline."""
    H = u_opt.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for j, name in enumerate(decision_names):
        axes[0].plot(np.arange(H), u_opt[:, j], color=PALETTE[j],
                      lw=1.5, label=f"{name} (opt)")
        axes[0].plot(np.arange(H), u_base[:, j], color=PALETTE[j], lw=1.5,
                      ls="--", label=f"{name} (base)")
    axes[0].set_ylabel("decision variables")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(H), y_opt[:, 3], color=PALETTE[0], lw=1.5,
                  label="y4 (opt)")
    axes[1].plot(np.arange(H), y_base[:, 3], color=PALETTE[0], lw=1.5,
                  ls="--", label="y4 (base)")
    axes[1].set_ylabel("y4")
    axes[1].set_xlabel("horizon step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Optimized vs baseline trajectories (sample {sample_idx})")
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