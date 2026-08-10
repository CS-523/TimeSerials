"""End-to-end smoke test for the modern control theory pipeline.

Runs the four main scripts with reduced parameters so the full pipeline
finishes in a few minutes. Useful as a CI check and as a quick "does
everything work?" confirmation after code changes.

Usage::

    python -m scripts_control.05_smoke_test
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    # Use a temp directory for all artifacts so we don't pollute the repo.
    with tempfile.TemporaryDirectory(prefix="smoke_test_") as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data" / "processed"
        ckpt_dir = tmp_path / "checkpoints"
        metrics_dir = tmp_path / "results" / "metrics"
        figures_dir = tmp_path / "results" / "figures"
        preds_dir = tmp_path / "results" / "predictions"

        for d in (data_dir, ckpt_dir, metrics_dir, figures_dir, preds_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. Preprocess
        print("\n[1/4] preprocess…")
        subprocess.run(
            [sys.executable, "-m", "src_control.preprocess",
             "--out", str(data_dir), "--seq-len", "128"],
            check=True,
        )

        # 2. Feature analysis
        print("\n[2/4] feature analysis…")
        subprocess.run(
            [sys.executable, "-m", "src_control.analysis.correlation",
             "--data", str(data_dir / "train.npz"),
             "--out", str(figures_dir)],
            check=True,
        )

        # 3. Train predictor (only 5 epochs to keep runtime short)
        print("\n[3/4] train predictor (5 epochs)…")
        subprocess.run(
            [sys.executable, "-m", "scripts_control.03_train_predictor",
             "--data", str(data_dir / "train.npz"),
             "--test", str(data_dir / "test.npz"),
             "--scalers", str(data_dir / "scalers.npz"),
             "--epochs", "5", "--bs", "16", "--tf-decay", "2",
             "--patience", "5",
             "--out-dir", str(ckpt_dir),
             "--metrics-dir", str(metrics_dir),
             "--predictions-dir", str(preds_dir)],
            check=True,
        )

        # 4. MPC optimization (1 sample, 1 start, short horizon)
        print("\n[4/4] MPC optimization (1 sample, horizon=8)…")
        subprocess.run(
            [sys.executable, "-m", "scripts_control.04_optimize",
             "--ckpt", str(ckpt_dir / "ss_nn_best.pt"),
             "--data", str(data_dir / "test.npz"),
             "--scalers", str(data_dir / "scalers.npz"),
             "--n-samples", "1", "--horizon", "8", "--n-starts", "1",
             "--out-metrics", str(metrics_dir),
             "--out-figures", str(figures_dir)],
            check=True,
        )

        # 5. Verify expected outputs exist
        print("\n[verify] checking outputs…")
        expected = [
            data_dir / "aligned_dataset.npz",
            data_dir / "train.npz",
            data_dir / "test.npz",
            data_dir / "scalers.npz",
            ckpt_dir / "ss_nn_best.pt",
            ckpt_dir / "ss_nn_last.pt",
            metrics_dir / "training_log.json",
            metrics_dir / "test_metrics.json",
            metrics_dir / "pareto.json",
            preds_dir / "test_predictions.npz",
            figures_dir / "correlation_heatmap.png",
            figures_dir / "mi_heatmap.png",
            figures_dir / "pca_scree.png",
            figures_dir / "lag_x_to_y4.png",
            figures_dir / "granger_xy.png",
            figures_dir / "analysis_report.json",
            figures_dir / "pareto_frontier.png",
        ]
        missing = [p for p in expected if not p.exists()]
        if missing:
            print(f"FAILED: missing files:\n  " + "\n  ".join(str(p) for p in missing))
            sys.exit(1)
        print(f"All {len(expected)} expected files present.")
        print("SMOKE TEST PASSED.")


if __name__ == "__main__":
    main()