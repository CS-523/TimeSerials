"""End-to-end visualization mirroring the style of ``src/visualize.py``.

Reads the trained SS-NN model and test-set predictions, then draws:

  1. ``forecast_x1_x8.png``     — true vs predicted x1..x8 for 2 samples
  2. ``forecast_y1_y4.png``     — true vs predicted y1..y4 for 2 samples
  3. ``error_distribution.png`` — residuals histogram + truth-vs-pred scatter
  4. ``optimization_compare.png`` — MPC baseline vs optimized y4 (if available)

Outputs go to ``src_control/analysis_out/`` by default (configurable via
``--out-dir``).

The script does not require re-training; it consumes the artifacts already
produced by ``scripts_control.03_train_predictor`` and
``scripts_control.04_optimize``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src_control.config import get_config, resolve_paths
from src_control.models.state_space_nn import SS_NN_Hybrid, YHead
from src_control.preprocess import load_processed
from src_control.utils.metrics import mse, mae, r2
from src_control.utils.seed import set_seed


# Palette consistent with src_control/visualization/plots.py
COL_HISTORY = "black"
COL_TRUTH = "#2ca02c"   # green
COL_PRED = "#d62728"    # red


# Default output directory: <src_control>/analysis_out/
# When run as ``python -m scripts_control.06_visualize``, __file__ is just
# "06_visualize.py" (relative). Use sys.modules[__name__].__file__ which
# is always the absolute path set by the import machinery.
import sys as _sys
_module_file = getattr(_sys.modules[__name__], "__file__", None) or os.path.abspath(
    os.path.join(os.getcwd(), "scripts_control", "06_visualize.py")
)
# _module_file = .../time_serials-master/scripts_control/06_visualize.py
# We want src_control/analysis_out — that's the sibling of scripts_control.
_repo_root = os.path.dirname(os.path.dirname(_module_file))
DEFAULT_OUT_DIR = os.path.join(_repo_root, "src_control", "analysis_out")


def _load_model_and_data(
    ckpt_path: str, test_npz: str, scalers_npz: str, device: str
) -> Tuple[SS_NN_Hybrid, YHead, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=16, hidden=128, window=4)
    model.load_state_dict(ckpt["model"])
    yhead = YHead(window=8)
    yhead.load_state_dict(ckpt["yhead"])
    model = model.to(device)
    yhead = yhead.to(device)
    model.eval()
    yhead.eval()

    test = load_processed(test_npz)
    scaler = np.load(scalers_npz)
    return model, yhead, test, scaler


def _predict_one(
    model: SS_NN_Hybrid,
    x_raw: np.ndarray,         # (T, 8)
    y_raw: np.ndarray,         # (T, 4)
    y_mask: np.ndarray,        # (T, 4)
    scaler: Dict[str, np.ndarray],
    device: str,
    teacher_forcing: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on a single sample and return (x_raw, y_pred_denorm)."""
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    y_mean = scaler["y_mean"]
    y_scale = scaler["y_scale"]

    x_std = (x_raw - x_mean) / x_scale
    y_filled = np.where(np.isnan(y_raw), 0.0, (y_raw - y_mean) / y_scale)
    y_std = y_filled * y_mask  # zero out non-observed

    x_t = torch.tensor(x_std, dtype=torch.float32, device=device).unsqueeze(0)
    y_t = torch.tensor(y_std, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x_t, y_prev=y_t, teacher_forcing=teacher_forcing)
    y_pred_np = y_pred[0].cpu().numpy()
    y_pred_denorm = y_pred_np * y_scale + y_mean
    return x_raw, y_pred_denorm


# --------------------------------------------------------------------------- #
# 1) x1..x8 forecast
# --------------------------------------------------------------------------- #
def plot_forecast_x(
    test: Dict[str, np.ndarray],
    model: SS_NN_Hybrid,
    scaler: Dict[str, np.ndarray],
    sample_indices: List[int],
    out_path: str,
    device: str,
) -> None:
    """Plot x1..x8 (history + ground truth) for the selected samples to
    visualise the input trajectory. The model's y prediction is overlaid
    as a secondary line so the figure also shows the model output.

    Layout: 2 rows × 8 cols per sample (top: x1..x8, bottom: y1..y4 prediction).
    """
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    y_mean = scaler["y_mean"]
    y_scale = scaler["y_scale"]
    X_COLS = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
    Y_NAMES = ("y1", "y2", "y3", "y4")

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 12, figsize=(36, 3.2 * n), sharex=True)
    if n == 1:
        axes = axes[None, :]

    for ei, idx in enumerate(sample_indices):
        X = test["X"][idx]
        T = int(test["lengths"][idx])
        x_std = X[:T]
        x_raw = x_std * x_scale + x_mean
        x_raw = x_raw.astype(np.float32)

        Y = test["Y"][idx]
        Y_mask = test["Y_mask"][idx]
        y_raw = Y[:T] * y_scale + y_mean
        y_raw = y_raw.astype(np.float32)

        # Run model (teacher-forced) over the full window
        _, y_pred = _predict_one(model, x_raw, y_raw, Y_mask[:T], scaler, device,
                                  teacher_forcing=1.0)

        # Top row: x1..x8 history (black) + ground truth (green dots; since
        # the y_history is sparse, we use the same x history for both halves
        # of the window and overlay the y_pred line to show the model's
        # interpretation). The y-axis is the x value (since x is on input).
        for ci, c in enumerate(X_COLS):
            ax = axes[ei, ci]
            ax.plot(np.arange(T), x_raw[:, ci], color=COL_HISTORY, lw=1.2,
                     label="history (input)")
            ax.set_title(f"{c} — samp {idx}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if ci == 0 and ei == 0:
                ax.legend(fontsize=6, loc="upper left")

        # Bottom row: y1..y4 predictions (red dashed) vs observed ground truth (green dots)
        for yi, name in enumerate(Y_NAMES):
            ax = axes[ei, 8 + yi]
            mask_col = Y_mask[:T, yi]
            # Observed ground truth
            if mask_col.any():
                t_obs = np.where(mask_col)[0]
                ax.scatter(t_obs, y_raw[mask_col, yi], s=22, color=COL_TRUTH,
                            zorder=5, label="observed truth")
            ax.plot(np.arange(T), y_pred[:, yi], color=COL_PRED, lw=1.0,
                     ls="--", alpha=0.85, label="model pred")
            ax.set_title(f"{name} — samp {idx}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if yi == 0 and ei == 0:
                ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
# 2) y1..y4 forecast
# --------------------------------------------------------------------------- #
def plot_forecast_y(
    test: Dict[str, np.ndarray],
    model: SS_NN_Hybrid,
    scaler: Dict[str, np.ndarray],
    sample_indices: List[int],
    out_path: str,
    device: str,
) -> None:
    """Plot y1..y4: history vs predicted (teacher-forced, full window)."""
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    y_mean = scaler["y_mean"]
    y_scale = scaler["y_scale"]
    Y_NAMES = ("y1", "y2", "y3", "y4")

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ei, idx in enumerate(sample_indices):
        X = test["X"][idx]
        Y = test["Y"][idx]
        Y_mask = test["Y_mask"][idx]
        T = int(test["lengths"][idx])

        x_raw = X[:T] * x_scale + x_mean
        y_raw = Y[:T] * y_scale + y_mean
        x_raw = x_raw.astype(np.float32)
        y_raw = y_raw.astype(np.float32)

        _, y_pred = _predict_one(model, x_raw, y_raw, Y_mask[:T], scaler, device,
                                  teacher_forcing=1.0)

        ax = axes[ei]
        t = np.arange(T)
        for yi, name in enumerate(Y_NAMES):
            mask_col = Y_mask[:T, yi]
            if mask_col.any():
                ax.plot(t[mask_col], y_raw[mask_col, yi], "o",
                         color=COL_TRUTH, markersize=4, label=f"{name} truth")
            ax.plot(t, y_pred[:, yi], color=COL_PRED, lw=1.0, ls="--",
                     alpha=0.8, label=f"{name} pred")
        ax.set_title(f"y1..y4 — sample {idx} (file {test['file_ids'][idx]})",
                      fontsize=10)
        ax.set_xlabel("timestep")
        ax.set_ylabel("y value")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
# 3) Error distribution
# --------------------------------------------------------------------------- #
def plot_error_distribution(
    preds_npz: str,
    out_path: str,
) -> None:
    """For each y variable, plot residuals histogram + truth-vs-pred scatter."""
    data = np.load(preds_npz, allow_pickle=True)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    mask = data["mask"].astype(bool)
    y_names = ("y1", "y2", "y3", "y4")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for j, name in enumerate(y_names):
        yt = y_true[..., j].ravel()
        yp = y_pred[..., j].ravel()
        m = mask[..., j].ravel()
        yt_obs, yp_obs = yt[m], yp[m]
        resid = yp_obs - yt_obs

        ax_h = axes[0, j]
        ax_h.hist(resid, bins=40, color="#4C72B0", alpha=0.7, edgecolor="black")
        ax_h.axvline(0, color="black", lw=1)
        mae_v = mae(yt_obs, yp_obs)
        rmse_v = float(np.sqrt(np.mean(resid ** 2)))
        ax_h.set_title(f"{name} — residuals  (MAE={mae_v:.1f}, RMSE={rmse_v:.1f})", fontsize=9)
        ax_h.set_xlabel("y_pred − y_true")
        ax_h.set_ylabel("count")
        ax_h.grid(True, alpha=0.3)

        ax_s = axes[1, j]
        ax_s.scatter(yt_obs, yp_obs, s=4, alpha=0.4, color="#4C72B0")
        lim_lo = min(yt_obs.min(), yp_obs.min())
        lim_hi = max(yt_obs.max(), yp_obs.max())
        ax_s.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5)
        r2_v = r2(yt_obs, yp_obs)
        ax_s.set_title(f"{name} — truth vs pred  (R²={r2_v:.3f})", fontsize=9)
        ax_s.set_xlabel("y_true")
        ax_s.set_ylabel("y_pred")
        ax_s.grid(True, alpha=0.3)

    plt.suptitle("Final fit error analysis (test set)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
# 4) Optimization compare
# --------------------------------------------------------------------------- #
def plot_optimization_compare(
    pareto_json: str,
    out_path: str,
) -> None:
    """Plot baseline vs optimized y4 (Pareto points highlighted)."""
    if not os.path.exists(pareto_json):
        print(f"[viz] {pareto_json} not found — skipping optimization plot.")
        return
    with open(pareto_json) as f:
        data = json.load(f)
    points = np.array(data["all_points"])
    baselines = np.array(data["baselines"])
    pareto_idx = set(data["pareto_indices"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], c="#4C72B0", alpha=0.5, s=30,
                label="weighted runs")
    if pareto_idx:
        pareto_pts = np.array([points[i] for i in pareto_idx])
        ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], c="#C44E52", s=80,
                    edgecolor="black", label="Pareto front")
    ax.scatter(baselines[:, 0], baselines[:, 1], c="#DD8452", s=160, marker="*",
                edgecolor="black", label="baseline (last-input)")
    ax.set_xlabel("Σ y4 over horizon")
    ax.set_ylabel("Predicted final Y")
    ax.set_title("Optimization scatter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = [baselines[:, 0].mean(), points[:, 0].mean()]
    stds = [baselines[:, 0].std(), points[:, 0].std()]
    ax.bar(["baseline", "optimized"], means, yerr=stds,
           color=["#8172B2", "#55A467"], edgecolor="black")
    improvement = 100 * (means[1] - means[0]) / max(abs(means[0]), 1e-9)
    ax.set_title(f"y4 Σ mean  (improvement {improvement:+.1f}%)")
    ax.set_ylabel("Σ y4")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ss_nn_best.pt")
    parser.add_argument("--test", default="data/processed/test.npz")
    parser.add_argument("--preds", default="results/predictions/test_predictions.npz")
    parser.add_argument("--scalers", default="data/processed/scalers.npz")
    parser.add_argument("--pareto", default="results/metrics/pareto.json")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = resolve_paths(get_config())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] out_dir: {out_dir}")

    device = cfg.DEVICE
    print(f"[viz] device: {device}")
    model, yhead, test, scaler = _load_model_and_data(
        args.ckpt, args.test, args.scalers, device
    )

    lengths = test["lengths"]
    order = np.argsort(lengths)[::-1]
    sample_indices = list(order[: args.n_samples])

    plot_forecast_x(
        test, model, scaler, sample_indices,
        out_path=out_dir / "forecast_x1_x8.png",
        device=device,
    )

    plot_forecast_y(
        test, model, scaler, sample_indices,
        out_path=out_dir / "forecast_y1_y4.png",
        device=device,
    )

    if os.path.exists(args.preds):
        plot_error_distribution(
            args.preds, out_path=out_dir / "error_distribution.png",
        )
    else:
        print(f"[viz] {args.preds} not found — skipping error plot.")

    plot_optimization_compare(
        args.pareto, out_path=out_dir / "optimization_compare.png",
    )

    print(f"[viz] all figures written to {out_dir}/")


if __name__ == "__main__":
    main()