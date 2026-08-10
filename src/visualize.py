"""
可视化与综合报告生成
===================
读取训练好的模型在测试集上的预测结果，画：
  1. 真实 vs 预测的 x1-x8（多步 rollout，可视化 1~2 条样例）
  2. 真实 vs 预测的 y4
  3. 优化前后 y4 / Y 对比
  4. 误差分布（直方图）

依赖：model_out/forecaster_best.pt, model_out/test_predictions.npz, model_out/optimization_results.json
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
from data_loader import load_all, X_COLS, Y_INT_COLS, split_experiments, Scaler, YScaler
from model_forecaster import PathIntegratorForecaster


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = PathIntegratorForecaster(
        dim_x=8, dim_y=4, dim_state=args["dim_state"], hidden=args["hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    y_scaler = YScaler(means=ckpt["y_scaler"]["mean"], stds=ckpt["y_scaler"]["std"])
    return model, x_scaler, y_scaler


def main():
    base = "/kefu-nas/ybkong/time_serials-master"
    out = os.path.join(base, "src/analysis_out")
    model_dir = os.path.join(base, "src/model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_scaler, y_scaler = load_model(os.path.join(model_dir, "forecaster_best.pt"), device)

    exps = load_all(base)
    _, _, test_exps = split_experiments(exps)
    test_exps = sorted(test_exps, key=lambda e: len(e), reverse=True)[:2]   # 选 2 条最长的做可视化
    print(f"[viz] 可视化实验: {[os.path.basename(e.file) for e in test_exps]}")

    fig, axes = plt.subplots(len(test_exps), 8, figsize=(24, 4 * len(test_exps)))
    if len(test_exps) == 1:
        axes = axes[None, :]
    for ei, exp in enumerate(test_exps):
        # 用前 in_len=24 步做输入，预测接下来 T_out=12 步
        in_len, T_out = 24, 16
        if len(exp) < in_len + T_out:
            in_len = min(20, len(exp) // 2)
            T_out = min(12, len(exp) - in_len)
        x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        y_raw = exp.df[Y_INT_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
        y_in_mask = (~np.isnan(y_raw)).astype(np.float32)
        y_in_filled = np.nan_to_num(y_raw, nan=0.0)
        y_in_norm = (y_in_filled - y_scaler.means) / y_scaler.stds
        y_in = torch.from_numpy(y_in_norm * y_in_mask).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_dict = model(x_in, y_in, T_out=T_out)
        # 真实未来段
        x_future = exp.df[X_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
        y_future = exp.df[Y_INT_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
        # 反标准化预测
        pred_x = out_dict["pred_x"][0].cpu().numpy() * x_scaler.std + x_scaler.mean
        pred_y = out_dict["pred_y"][0].cpu().numpy() * y_scaler.stds + y_scaler.means
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

    # y4 预测图
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ei, exp in enumerate(test_exps):
        in_len, T_out = 24, 32
        if len(exp) < in_len + T_out:
            in_len = min(20, len(exp) // 2)
            T_out = min(12, len(exp) - in_len)
        x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        y_raw = exp.df[Y_INT_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
        y_in_mask = (~np.isnan(y_raw)).astype(np.float32)
        y_in_filled = np.nan_to_num(y_raw, nan=0.0)
        y_in_norm = (y_in_filled - y_scaler.means) / y_scaler.stds
        y_in = torch.from_numpy(y_in_norm * y_in_mask).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_dict = model(x_in, y_in, T_out=T_out)
        y_future = exp.df[Y_INT_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
        pred_y = out_dict["pred_y"][0].cpu().numpy() * y_scaler.stds + y_scaler.means
        ax = axes[ei]
        t_out = np.arange(in_len, in_len + T_out)
        for yi, name in enumerate(["y1", "y2", "y3", "y4"]):
            ax.plot(t_out, y_future[:, yi], "-", label=f"{name} truth", alpha=0.5)
            ax.plot(t_out, pred_y[:, yi], "--", label=f"{name} pred", alpha=0.7)
        ax.set_title(f"y1-y4: {os.path.basename(exp.file)}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "forecast_y1_y4.png"), dpi=120)
    plt.close()
    print(f"[viz] 写出 forecast_y1_y4.png")

    # 优化前后对比（如果文件存在）
    opt_path = os.path.join(model_dir, "optimization_results.json")
    if os.path.exists(opt_path):
        with open(opt_path) as f:
            opt = json.load(f)
        rows = opt["rows"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        y4_base = np.array([r["y4_base"] for r in rows])
        y4_opt = np.array([r["y4_opt"] for r in rows])
        ax = axes[0]
        ax.scatter(y4_base, y4_opt, alpha=0.7)
        lim = [min(y4_base.min(), y4_opt.min()) - 100, max(y4_base.max(), y4_opt.max()) + 100]
        ax.plot(lim, lim, "k--", alpha=0.5, label="y=x")
        ax.set_xlabel("y4 baseline (原 rollout)")
        ax.set_ylabel("y4 optimized (控制优化后)")
        ax.set_title(f"基线 vs 优化 y4 ({opt['method']})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.bar(["baseline", "optimized"], [y4_base.mean(), y4_opt.mean()],
               yerr=[y4_base.std(), y4_opt.std()], color=["gray", "orange"])
        ax.set_title(f"y4 均值对比: 提升 {opt['summary']['improvement_pct']:.1f}%")
        ax.set_ylabel("y4")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "optimization_compare.png"), dpi=120)
        plt.close()
        print(f"[viz] 写出 optimization_compare.png")

    # 训练曲线（如果没有，从日志里复原）— 简化：跳过
    print(f"[viz] 完成。所有图片已写入 {out}/")


if __name__ == "__main__":
    main()