"""Run Pareto MPC on a subset of the test set, save results.

Usage::

    python -m scripts_control.04_optimize \
        --ckpt checkpoints/ss_nn_best.pt \
        --data  data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --n-samples 10 --out results/metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src_control.config import get_config, resolve_paths
from src_control.models.state_space_nn import SS_NN_Hybrid, YHead
from src_control.optimization.mpc_optimizer import (
    OptConfig,
    optimize_pareto,
    pareto_front,
)
from src_control.preprocess import load_processed
from src_control.utils.seed import set_seed
from src_control.visualization.plots import plot_optimized_trajectory, plot_pareto


def _build_x_history(test: dict, idx: int, scaler_x) -> np.ndarray:
    """Return the last observed (raw, denormalized) inputs for sample idx.

    Returns shape (T_hist, 8).
    """
    X_std = test["X"][idx]
    lengths = test["lengths"][idx]
    T = int(lengths)
    x_std = X_std[:T]                      # (T, 8)
    # Reverse standardization
    x_mean = scaler_x["x_mean"]
    x_scale = scaler_x["x_scale"]
    x_raw = x_std * x_scale + x_mean
    return x_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ss_nn_best.pt")
    parser.add_argument("--data", default="data/processed/test.npz")
    parser.add_argument("--scalers", default="data/processed/scalers.npz")
    parser.add_argument("--out-metrics", default="results/metrics")
    parser.add_argument("--out-figures", default="results/figures")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = resolve_paths(get_config())

    out_metrics = Path(args.out_metrics)
    out_figures = Path(args.out_figures)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=16, hidden=128, window=4)
    model.load_state_dict(ckpt["model"])
    yhead = YHead(window=8)
    yhead.load_state_dict(ckpt["yhead"])

    # Load data
    test = load_processed(args.data)
    scaler = np.load(args.scalers)

    # Compute standardized means for x1, x2, x5 (fixed inputs)
    # mean of standardized inputs = (raw_mean - x_mean) / x_scale
    # Equivalently: use the standardized data directly
    X_test = test["X"]
    # Mean over all observed (non-zero-padded) timesteps
    lengths = test["lengths"]
    # Compute raw means for x1/x2/x5 from test set (only as a fallback)
    raw_means = np.zeros(8)
    raw_stds = np.ones(8)
    for j in range(8):
        col_obs = []
        for i in range(X_test.shape[0]):
            T = int(lengths[i])
            x_raw = X_test[i, :T, j] * scaler["x_scale"][j] + scaler["x_mean"][j]
            col_obs.append(x_raw)
        col_obs = np.concatenate(col_obs, axis=0)
        raw_means[j] = col_obs.mean()
        raw_stds[j] = col_obs.std()

    # For optimization, we want **standardized** fixed values:
    # raw_mean → standardized = (raw_mean - x_mean) / x_scale
    fixed_x1_x2_x5_std = tuple(
        (raw_means[i] - scaler["x_mean"][i]) / scaler["x_scale"][i]
        for i in (0, 1, 4)
    )

    opt_cfg = OptConfig(
        horizon=args.horizon,
        n_starts=args.n_starts,
        max_iter=30,  # capped for runtime
        device="cpu",
    )

    n_samples = min(args.n_samples, X_test.shape[0])
    sample_indices = list(range(n_samples))

    all_points = []
    all_baselines = []
    per_sample_records = []

    for sample_idx in sample_indices:
        x_history = _build_x_history(test, sample_idx, scaler)
        print(f"sample {sample_idx}: x_history shape={x_history.shape}, "
              f"last y4 region mean={x_history[-5:, 7].mean():.1f}")
        res = optimize_pareto(
            model=model,
            yhead=yhead,
            x_history=x_history,
            cfg=opt_cfg,
            fixed_x1_x2_x5=fixed_x1_x2_x5_std,
        )
        all_points.extend(res["points"])
        all_baselines.append((res["baseline"]["y4_sum"], res["baseline"]["Y_pred"]))
        per_sample_records.append({
            "sample_idx": sample_idx,
            "file_id": str(test["file_ids"][sample_idx]),
            "trajectories": [
                {
                    "weights": list(t["weights"]),
                    "u_dec": t["u_dec"].tolist(),
                    "y4_sum": t["y4_sum"],
                    "Y_pred": t["Y_pred"],
                    "y_full": t["y_full"].tolist(),
                }
                for t in res["trajectories"]
            ],
            "baseline": {
                "u_dec": res["baseline"]["u_dec"].tolist(),
                "y_full": res["baseline"]["y_full"].tolist(),
                "y4_sum": res["baseline"]["y4_sum"],
                "Y_pred": res["baseline"]["Y_pred"],
            },
        })

        # Plot the first weight (y4-only) vs baseline
        opt_traj = res["trajectories"][0]
        plot_optimized_trajectory(
            u_opt=opt_traj["u_dec"],
            u_base=res["baseline"]["u_dec"],
            y_opt=opt_traj["y_full"],
            y_base=res["baseline"]["y_full"],
            sample_idx=sample_idx,
            out_path=out_figures / f"optimized_vs_baseline_{sample_idx}.png",
        )

    # Aggregate Pareto plot
    pareto_idx = pareto_front(all_points)
    plot_pareto(
        points=all_points,
        baseline=(
            np.mean([b[0] for b in all_baselines]),
            np.mean([b[1] for b in all_baselines]),
        ),
        out_path=out_figures / "pareto_frontier.png",
        title=f"Pareto frontier — {n_samples} test samples ({len(pareto_idx)} non-dominated)",
    )

    # Save JSON
    summary = {
        "n_samples": n_samples,
        "horizon": args.horizon,
        "n_starts": args.n_starts,
        "weights": opt_cfg.weights,
        "bounds": opt_cfg.bounds,
        "fixed_x1_x2_x5_std": fixed_x1_x2_x5_std,
        "all_points": all_points,
        "baselines": all_baselines,
        "pareto_indices": pareto_idx,
        "per_sample": per_sample_records,
    }
    (out_metrics / "pareto.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {out_metrics / 'pareto.json'}")
    print(f"Saved {(out_figures / 'pareto_frontier.png')}")


if __name__ == "__main__":
    main()