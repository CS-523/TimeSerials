"""
可视化（x 预测专用版）
=====================
本版本只可视化 x1-x8 的预测（不带 y）。

依赖：model_out/forecaster_best.pt
输出（analysis_out/）：
  - forecast_x1_x8.png：x1-x8 真实 vs 预测
  - error_per_dim.png：分维度 RMSE 柱状图
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS, split_experiments, Scaler
from model_forecaster import PathIntegratorForecaster


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = PathIntegratorForecaster(
        dim_x=8, dim_state=args["dim_state"], hidden=args["hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    return model, x_scaler


def main():
    base = "/kefu-nas/ybkong/time_serials-master"
    out = os.path.join(base, "src/analysis_out")
    model_dir = os.path.join(base, "src/model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_scaler = load_model(os.path.join(model_dir, "forecaster_best.pt"), device)

    exps = load_all(base)
    _, _, test_exps = split_experiments(exps)
    test_exps = sorted(test_exps, key=lambda e: len(e), reverse=True)[:2]
    print(f"[viz] 可视化实验: {[os.path.basename(e.file) for e in test_exps]}")

    # 1) 真实 vs 预测的 x1-x8（多步 rollout，2 条样例）
    fig, axes = plt.subplots(len(test_exps), 8, figsize=(24, 4 * len(test_exps)))
    if len(test_exps) == 1:
        axes = axes[None, :]
    for ei, exp in enumerate(test_exps):
        in_len, T_out = 24, 16
        if len(exp) < in_len + T_out:
            in_len = min(20, len(exp) // 2)
            T_out = min(12, len(exp) - in_len)
        x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_dict = model(x_in, T_out=T_out)
        x_future = exp.df[X_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
        pred_x = out_dict["pred_x"][0].cpu().numpy() * x_scaler.std + x_scaler.mean
        for ci, c in enumerate(X_COLS):
            ax = axes[ei, ci]
            t_in = np.arange(in_len)
            t_out = np.arange(in_len, in_len + T_out)
            ax.plot(t_in, x_raw[:, ci], "k-", label="history", linewidth=1.5)
            ax.plot(t_out, x_future[:, ci], "g-", label="truth", linewidth=1.5, marker="o", markersize=3)
            ax.plot(t_out, pred_x[:, ci], "r--", label="pred", linewidth=1.5, marker="x", markersize=3)
            ax.set_title(f"{c} ({os.path.basename(exp.file)})", fontsize=9)
            if ci == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "forecast_x1_x8.png"), dpi=120)
    plt.close()
    print(f"[viz] 写出 forecast_x1_x8.png")

    # 2) 分维度 RMSE（柱状图 + 数值）
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        per_dim = np.array(metrics.get("rmse_x_per_dim", [0] * 8))
        per_dim_orig = np.array(metrics.get("rmse_x_per_dim_orig", [0] * 8))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(X_COLS, per_dim, color="steelblue")
        axes[0].set_title(f"分维度 RMSE(x) 标准化空间  (均值={metrics.get('rmse_x', 0):.4f})")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[1].bar(X_COLS, per_dim_orig, color="darkorange")
        axes[1].set_title(f"分维度 RMSE(x) 原始空间  (均值={metrics.get('rmse_x_orig_mean', 0):.2f})")
        axes[1].set_ylabel("RMSE")
        axes[1].grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "error_per_dim.png"), dpi=120)
        plt.close()
        print(f"[viz] 写出 error_per_dim.png")

    print(f"[viz] 完成。所有图片已写入 {out}/")


if __name__ == "__main__":
    main()