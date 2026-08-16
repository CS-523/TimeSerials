"""Optimize future x3/x4/x6/x8 to maximize y4 (README task 4).

Composes two frozen ``SS_NN_Hybrid`` models:

* ``checkpoints/x_forecast_best.pt`` (dim_y=8) — default "no-intervention"
  future x1–x8 trajectory from a past context.
* ``checkpoints/ss_nn_best.pt`` (dim_y=4) — reward: x1–x8 sequence → y1–y4,
  y4 = output[..., 3].

Only the controllable inputs x3/x4/x6/x8 are optimized (bounded); the
non-controllable x1/x2/x5/x7 follow the forecast.

Usage::

    python -m scripts_control.04_optimize \
        --data data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --context 32 --horizon 48 \
        --n-samples 1 --plot-max 1

    # batch evaluate the whole test set (distribution of y4 gains)
    python -m scripts_control.04_optimize --all-test

Outputs (under ``results/optimization/``):

* ``optimized_vs_baseline_{i}.png`` — default vs optimized trajectory (raw space)
* ``optimize_metrics.json`` — per-sample baseline/optimized y4 + summary stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src_control.models.state_space_nn import SS_NN_Hybrid
from src_control.preprocess import load_processed
from src_control.optimization import (
    CONTROL_BOUNDS_RAW,
    CONTROL_NAMES,
    DECISION_IDX,
    maximize_y4,
)

ROOT = Path(__file__).resolve().parent.parent


def _resolve(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def load_models(x_ckpt: str, y_ckpt: str, device: torch.device):
    """Load the x-forecaster (dim_y=8) and y-predictor (dim_y=4) checkpoints."""
    x_model = SS_NN_Hybrid(dim_u=8, dim_y=8).to(device)
    x_model.load_state_dict(
        torch.load(x_ckpt, map_location=device, weights_only=False)
    )
    y_model = SS_NN_Hybrid(dim_u=8, dim_y=4).to(device)
    y_model.load_state_dict(
        torch.load(y_ckpt, map_location=device, weights_only=False)["model"]
    )
    return x_model, y_model


def plot_sample(
    out_path,
    title,
    x_default_raw,
    x_opt_raw,
    x_truth_raw,
    y4_default_raw,
    y4_opt_raw,
    y4_truth=None,
    y4_mask=None,
    bounds=None,
):
    """Plot default vs optimized control trajectories (raw space) + y4."""
    if bounds is None:
        bounds = CONTROL_BOUNDS_RAW
    H = x_default_raw.shape[0]
    t = np.arange(H)
    fig, axes = plt.subplots(
        1, len(DECISION_IDX) + 1, figsize=(4 * (len(DECISION_IDX) + 1), 3.2)
    )

    for k, (name, j) in enumerate(zip(CONTROL_NAMES, DECISION_IDX)):
        ax = axes[k]
        ax.plot(t, x_default_raw[:, j], label="default", color="#42a5f5", lw=1.5)
        ax.plot(t, x_opt_raw[:, j], label="optimized", color="#ef5350", lw=1.5)
        ax.plot(t, x_truth_raw[:, j], label="truth", color="#66bb6a", lw=1.0, ls="--", alpha=0.8)
        lo, hi = bounds[k]
        ax.axhline(lo, color="gray", lw=0.7, ls=":")
        ax.axhline(hi, color="gray", lw=0.7, ls=":")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)

    ax = axes[-1]
    if y4_mask is not None and y4_mask.any():
        obs = y4_mask.astype(bool)
        ax.plot(t[obs], y4_default_raw[obs], marker="o", ls="none",
                color="#42a5f5", label="y4 default (pred @obs)")
        ax.plot(t[obs], y4_opt_raw[obs], marker="o", ls="none",
                color="#ef5350", label="y4 optimized (pred @obs)")
        if y4_truth is not None:
            ax.scatter(t[obs], y4_truth[obs], color="green", s=22,
                       marker="x", label="y4 truth")
    else:
        ax.text(0.5, 0.5, "no y4 obs in window", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
    ax.set_title("y4 (raw, @observed steps)", fontsize=10)
    ax.legend(fontsize=7)

    axes[0].set_xlabel("future step")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def train_distribution_bounds(train_path, x_mean, x_scale, quantile=0.01):
    """统计训练集 x3/x4/x6/x8 的原始空间分布界（去 padding）。

    quantile > 0: 用 [quantile, 1-quantile] 分位数；quantile == 0: 用 min/max。
    返回 ``(4, 2)`` 数组，顺序与 ``DECISION_IDX`` 一致。
    """
    train = load_processed(train_path)
    X = train["X"]
    lengths = train["lengths"].astype(int)
    rows = np.concatenate([X[i, :l] for i, l in enumerate(lengths)], axis=0)
    raw = rows * np.asarray(x_scale) + np.asarray(x_mean)  # (M, 8) 原始空间
    ctrl = raw[:, list(DECISION_IDX)]  # (M, 4)
    if quantile > 0:
        lo = np.percentile(ctrl, quantile * 100.0, axis=0)
        hi = np.percentile(ctrl, (1.0 - quantile) * 100.0, axis=0)
    else:
        lo = ctrl.min(axis=0)
        hi = ctrl.max(axis=0)
    return np.stack([lo, hi], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/test.npz")
    ap.add_argument("--scalers", default="data/processed/scalers.npz")
    ap.add_argument("--x-ckpt", default="checkpoints/x_forecast_best.pt")
    ap.add_argument("--y-ckpt", default="checkpoints/ss_nn_best.pt")
    ap.add_argument("--out-dir", default="results/optimization")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--effort-penalty", type=float, default=0.0)
    ap.add_argument("--train-data", default="data/processed/train.npz",
                    help="training split used to derive in-distribution bounds")
    ap.add_argument("--bounds-mode", choices=("physical", "train"), default="train",
                    help="search bounds: 'physical' = permissive VAR_RANGES; "
                         "'train' = training-distribution range")
    ap.add_argument("--bound-quantile", type=float, default=0.01,
                    help="train mode: quantile q -> [q, 1-q] bounds; 0 = min/max")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="number of samples to optimize (ignored with --all-test)")
    ap.add_argument("--all-test", action="store_true",
                    help="optimize every valid test sample (distribution view)")
    ap.add_argument("--plot-max", type=int, default=5,
                    help="max per-sample PNGs to save (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device = {device}")

    data = load_processed(_resolve(args.data))
    scalers = np.load(_resolve(args.scalers))
    x_mean = scalers["x_mean"].astype(np.float64)
    x_scale = scalers["x_scale"].astype(np.float64)
    y_mean = scalers["y_mean"].astype(np.float64)
    y_scale = scalers["y_scale"].astype(np.float64)

    X = data["X"]            # (N, 320, 8) standardized
    Y = data["Y"]            # (N, 320, 4)
    Y_mask = data["Y_mask"].astype(bool)
    lengths = data["lengths"].astype(int)
    file_ids = data["file_ids"]

    x_model, y_model = load_models(_resolve(args.x_ckpt), _resolve(args.y_ckpt), device)

    if args.bounds_mode == "train":
        control_bounds_raw = train_distribution_bounds(
            _resolve(args.train_data), x_mean, x_scale, args.bound_quantile
        )
        print("control bounds (train distribution):")
        for name, (lo, hi) in zip(CONTROL_NAMES, control_bounds_raw):
            print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
    else:
        control_bounds_raw = CONTROL_BOUNDS_RAW

    C, H = args.context, args.horizon
    valid = np.where(lengths >= C + H)[0]
    idxs = valid.tolist() if args.all_test else valid[: args.n_samples].tolist()
    if not idxs:
        raise SystemExit(
            f"No sample is long enough for context={C}+horizon={H}; "
            f"lower --context/--horizon."
        )
    print(f"optimizing {len(idxs)} sample(s): context={C}, horizon={H}, "
          f"n_starts={args.n_starts}, max_iter={args.max_iter}, "
          f"effort_penalty={args.effort_penalty}")

    out_dir = Path(_resolve(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rank, i in enumerate(idxs):
        L = int(lengths[i])
        s = L - H  # forecast the tail, matching 08_train_x_model's convention
        ctx = torch.tensor(X[i, s - C : s], dtype=torch.float32, device=device)

        res = maximize_y4(
            ctx, x_model, y_model, x_mean, x_scale, H,
            y4_phase_mask=Y_mask[i, s : s + H, 3],
            control_bounds_raw=control_bounds_raw,
            n_starts=args.n_starts, max_iter=args.max_iter, lr=args.lr,
            effort_penalty=args.effort_penalty, seed=args.seed + i, device=device,
        )

        x_default_raw = res["x_default_norm"] * x_scale + x_mean
        x_opt_raw = res["x_opt_norm"] * x_scale + x_mean
        x_truth_raw = X[i, s : s + H] * x_scale + x_mean
        y4_default_raw = res["y4_default_norm"] * y_scale[3] + y_mean[3]
        y4_opt_raw = res["y4_opt_norm"] * y_scale[3] + y_mean[3]

        base = float(res["reward_default_norm"]) * y_scale[3] + y_mean[3]
        opt = float(res["reward_opt_norm"]) * y_scale[3] + y_mean[3]
        gain = opt - base
        gain_pct = gain / abs(base) * 100.0 if base != 0 else 0.0

        # physical-bounds check (tiny tolerance for float denormalization noise)
        ctrl_opt_raw = x_opt_raw[:, DECISION_IDX]
        lo_b = CONTROL_BOUNDS_RAW[:, 0]
        hi_b = CONTROL_BOUNDS_RAW[:, 1]
        viol = max(0.0, float((lo_b - ctrl_opt_raw).max()), float((ctrl_opt_raw - hi_b).max()))

        results.append({
            "file_id": str(file_ids[i]),
            "length": L,
            "baseline_y4_raw": round(base, 3),
            "opt_y4_raw": round(opt, 3),
            "gain_raw": round(gain, 3),
            "gain_pct": round(gain_pct, 3),
            "in_bounds": bool(viol <= 1e-2),
            "max_bound_violation": round(viol, 5),
        })
        print(f"[{rank + 1}/{len(idxs)}] {results[-1]['file_id']}: "
              f"y4 {base:.1f} -> {opt:.1f}  (+{gain:.1f}, {gain_pct:+.2f}%)")

        if rank < args.plot_max:
            y4_truth = Y[i, s : s + H, 3] * y_scale[3] + y_mean[3]
            m = Y_mask[i, s : s + H, 3]
            plot_sample(
                out_dir / f"optimized_vs_baseline_{i}.png",
                title=str(file_ids[i]),
                x_default_raw=x_default_raw,
                x_opt_raw=x_opt_raw,
                x_truth_raw=x_truth_raw,
                y4_default_raw=y4_default_raw,
                y4_opt_raw=y4_opt_raw,
                y4_truth=y4_truth if m.any() else None,
                y4_mask=m,
                bounds=control_bounds_raw,
            )

    gains = np.array([r["gain_raw"] for r in results], dtype=np.float64)
    summary = {
        "config": {
            "context": C, "horizon": H, "n_starts": args.n_starts,
            "max_iter": args.max_iter, "lr": args.lr,
            "effort_penalty": args.effort_penalty,
        },
        "n_samples": len(results),
        "gain_raw_mean": float(gains.mean()),
        "gain_raw_median": float(np.median(gains)),
        "gain_pct_mean": float(np.mean([r["gain_pct"] for r in results])),
        "n_improved": int((gains > 0).sum()),
        "n_degraded": int((gains < 0).sum()),
    }

    payload = {"summary": summary, "samples": results}
    with open(out_dir / "optimize_metrics.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved → {(out_dir / 'optimize_metrics.json').resolve()}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
