"""End-to-end visualization mirroring the style of ``src/visualize.py``.

Usage::

    python -m scripts_control.06_visualize \
        --ckpt checkpoints/ss_nn_best.pt \
        --x-ckpt checkpoints/x_recon_best.pt \
        --test data/processed/test.npz \
        --preds results/predictions/test_predictions.npz \
        --scalers data/processed/scalers.npz \
        --pareto results/metrics/pareto.json \
        --out-dir src_control/analysis_out \
        --n-samples 2

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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
COL_PRED = "#d62728"    # red

# Per-channel styles for y1..y4 (distinct color + marker per channel).
Y_COLORS = ("#4C72B0", "#DD8452", "#55A467", "#C44E52")  # blue, orange, green, red
Y_MARKERS = ("o", "s", "^", "D")


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


def _warn_missing(artifact: str, path: str, consequence: str) -> None:
    """Emit a prominent, non-silent warning for a missing optional artifact.

    Optional artifacts (y model, x model, preds, pareto JSON) are skipped so
    the script can still draw whatever *is* available, but the skip must never
    be silent — it goes to stderr with an explicit WARNING banner.
    """
    print(
        f"[viz] WARNING: {artifact} not found: {path}\n"
        f"[viz]          → {consequence}",
        file=sys.stderr,
    )


def _load_model_and_data(
    ckpt_path: str, test_npz: str, scalers_npz: str, device: str, skipped: List[str]
) -> Tuple[Optional[SS_NN_Hybrid], Optional[YHead], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    test = load_processed(test_npz)
    scaler = np.load(scalers_npz)

    if not os.path.exists(ckpt_path):
        _warn_missing("y model checkpoint", ckpt_path,
                      "y panels + forecast_y1_y4.png will be skipped")
        skipped.append(f"y model checkpoint: {ckpt_path}")
        return None, None, test, scaler

    ckpt = torch.load(ckpt_path, map_location=device)
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=16, hidden=128, window=4)
    model.load_state_dict(ckpt["model"])
    yhead = YHead(window=8)
    yhead.load_state_dict(ckpt["yhead"])
    model = model.to(device)
    yhead = yhead.to(device)
    model.eval()
    yhead.eval()
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


def _load_x_model(ckpt_path: str, device: str, skipped: List[str]) -> Optional[SS_NN_Hybrid]:
    """Load the independent x-reconstruction model, or ``None`` if absent.

    ``checkpoints/x_recon_best.pt`` stores a raw ``state_dict`` (from
    ``scripts_control.08_train_x_model``), unlike the y checkpoint which wraps
    ``{"model": ..., "yhead": ...}``.
    """
    if not os.path.exists(ckpt_path):
        _warn_missing("x-reconstruction model checkpoint", ckpt_path,
                      "x̂ overlay (red dashed) will be skipped")
        skipped.append(f"x model checkpoint: {ckpt_path}")
        return None
    x_model = SS_NN_Hybrid(dim_u=8, dim_y=8, n_state=16, hidden=128, window=4)
    x_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    x_model = x_model.to(device)
    x_model.eval()
    return x_model


def _predict_x(
    x_model: SS_NN_Hybrid,
    x_raw: np.ndarray,          # (T, 8) raw x values
    scaler: Dict[str, np.ndarray],
    device: str,
) -> np.ndarray:
    """Run the x-reconstruction model on x and return denormalized x̂ (T, 8)."""
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    x_std = (x_raw - x_mean) / x_scale
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        x_pred = x_model(x_t, y_prev=None, teacher_forcing=0.0)
    x_pred_np = x_pred[0].cpu().numpy()
    return x_pred_np * x_scale + x_mean


# --------------------------------------------------------------------------- #
# 1) x1..x8 forecast
# --------------------------------------------------------------------------- #
def plot_forecast_x(
    test: Dict[str, np.ndarray],
    scaler: Dict[str, np.ndarray],
    sample_indices: List[int],
    out_path: str,
    device: str,
    x_model: Optional[SS_NN_Hybrid] = None,
) -> None:
    """Plot x1..x8 (history + reconstructed x̂) for the selected samples.

    y predictions live in ``plot_forecast_y`` / ``forecast_y1_y4.png``, so this
    figure is x-only: 8 columns per sample.
    """
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    X_COLS = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")

    n = len(sample_indices)
    n_cols = 8
    fig, axes = plt.subplots(n, n_cols, figsize=(3.0 * n_cols, 3.2 * n), sharex=True)
    if n == 1:
        axes = axes[None, :]

    for ei, idx in enumerate(sample_indices):
        X = test["X"][idx]
        T = int(test["lengths"][idx])
        x_raw = (X[:T] * x_scale + x_mean).astype(np.float32)

        # Optional x-reconstruction overlay (independent x denoising model)
        x_pred = _predict_x(x_model, x_raw, scaler, device) if x_model is not None else None

        # x1..x8 history (black) + reconstructed x̂ (red dashed).
        for ci, c in enumerate(X_COLS):
            ax = axes[ei, ci]
            ax.plot(np.arange(T), x_raw[:, ci], color=COL_HISTORY, lw=1.2,
                     label="history (input)")
            if x_pred is not None:
                ax.plot(np.arange(T), x_pred[:, ci], color=COL_PRED, lw=1.0,
                        ls="--", alpha=0.85, label="x̂ pred")
            ax.set_title(f"{c} — samp {idx}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if ci == 0 and ei == 0:
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
    """Plot y1..y4, one subplot per channel (independently auto-scaled).

    Each channel gets its own y-axis so the small y1/y2 channels are not
    squashed by the much larger y3/y4 range.
    """
    x_mean = scaler["x_mean"]
    x_scale = scaler["x_scale"]
    y_mean = scaler["y_mean"]
    y_scale = scaler["y_scale"]
    Y_NAMES = ("y1", "y2", "y3", "y4")

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n), sharex=True)
    if n == 1:
        axes = axes[None, :]

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

        t = np.arange(T)
        for yi, name in enumerate(Y_NAMES):
            ax = axes[ei, yi]
            color = Y_COLORS[yi]
            mask_col = Y_mask[:T, yi]
            if mask_col.any():
                ax.plot(t[mask_col], y_raw[mask_col, yi],
                        marker=Y_MARKERS[yi], linestyle="none", color=color,
                        markersize=5, label="observed truth")
            ax.plot(t, y_pred[:, yi], color=color, lw=1.0, ls="--",
                     alpha=0.85, label="model pred")
            ax.set_title(f"{name} — sample {idx}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if yi == 0:
                ax.set_ylabel("y value", fontsize=8)
            if ei == n - 1:
                ax.set_xlabel("timestep", fontsize=8)
            if yi == 0 and ei == 0:
                ax.legend(fontsize=7, loc="upper left")

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
    skipped: List[str],
) -> None:
    """Plot baseline vs optimized y4 (Pareto points highlighted)."""
    if not os.path.exists(pareto_json):
        _warn_missing("Pareto JSON", pareto_json,
                      "optimization_compare.png will be skipped")
        skipped.append(f"Pareto JSON: {pareto_json}")
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
    parser.add_argument("--ckpt", default=None,
                        help="y model checkpoint (default: <out-root>/checkpoints/ss_nn_best.pt, "
                             "or checkpoints/ss_nn_best.pt when --out-root is unset).")
    parser.add_argument("--x-ckpt", default=None,
                        help="Optional x-reconstruction model for the x1..x8 overlay "
                             "(default: <out-root>/checkpoints/x_recon_best.pt).")
    parser.add_argument("--test", default="data/processed/test.npz")
    parser.add_argument("--preds", default=None,
                        help="Test predictions npz (default: <out-root>/results/predictions/test_predictions.npz).")
    parser.add_argument("--scalers", default="data/processed/scalers.npz")
    parser.add_argument("--pareto", default=None,
                        help="Pareto JSON (default: <out-root>/results/metrics/pareto.json).")
    parser.add_argument("--out-root", default=None,
                        help="Root for model/prediction artifacts. When set, --ckpt/--x-ckpt/"
                             "--preds/--pareto default to <out-root>/checkpoints/... and "
                             "<out-root>/results/... (e.g. --out-root scripts_control).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve artifact paths: an explicit --ckpt/--x-ckpt/--preds/--pareto wins;
    # otherwise they default under --out-root (mirrors 08_train_x_model's output).
    out_root = Path(args.out_root) if args.out_root else None

    def _artifact(rel: str, val: Optional[str]) -> str:
        if val is not None:
            return val
        return str(out_root / rel) if out_root else rel

    ckpt = _artifact("checkpoints/ss_nn_best.pt", args.ckpt)
    x_ckpt = _artifact("checkpoints/x_recon_best.pt", args.x_ckpt)
    preds = _artifact("results/predictions/test_predictions.npz", args.preds)
    pareto = _artifact("results/metrics/pareto.json", args.pareto)

    set_seed(args.seed)
    cfg = resolve_paths(get_config())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] out_dir: {out_dir}")

    device = cfg.DEVICE
    print(f"[viz] device: {device}")
    skipped: List[str] = []
    model, yhead, test, scaler = _load_model_and_data(
        ckpt, args.test, args.scalers, device, skipped
    )
    x_model = _load_x_model(x_ckpt, device, skipped)

    if model is None and x_model is None:
        print("[viz] neither y model nor x model found — nothing to draw. "
              "Train 08_train_x_model (x) or 03_train_predictor (y) first.")
        return

    lengths = test["lengths"]
    order = np.argsort(lengths)[::-1]
    sample_indices = list(order[: args.n_samples])

    plot_forecast_x(
        test, scaler, sample_indices,
        out_path=out_dir / "forecast_x1_x8.png",
        device=device,
        x_model=x_model,
    )

    if model is not None:
        plot_forecast_y(
            test, model, scaler, sample_indices,
            out_path=out_dir / "forecast_y1_y4.png",
            device=device,
        )

    if os.path.exists(preds):
        plot_error_distribution(
            preds, out_path=out_dir / "error_distribution.png",
        )
    else:
        _warn_missing("test predictions npz", preds,
                      "error_distribution.png will be skipped")
        skipped.append(f"test predictions npz: {preds}")

    plot_optimization_compare(
        pareto, out_path=out_dir / "optimization_compare.png", skipped=skipped,
    )

    if skipped:
        print("\n" + "=" * 62, file=sys.stderr)
        print("[viz] WARNING — the following optional artifacts were missing "
              "and skipped:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
        print("=" * 62, file=sys.stderr)

    print(f"[viz] all figures written to {out_dir}/")


if __name__ == "__main__":
    main()