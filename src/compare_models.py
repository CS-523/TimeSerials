"""
对比分析：PathIntegrator vs LSTM
=================================
同时加载两个 ckpt，在同一组测试样本上做对比，画：
  1. PathInt vs LSTM 真实 vs 预测对比图（2 条样例 × 8 列）
  2. 分维度 RMSE 对比柱状图
  3. 失败样本分析：哪些样本 x7 预测最差

依赖：
  - src/model_out/forecaster_pathint.pt  (用 --save-name pathint 训练产出)
  - src/model_out/forecaster_lstm.pt     (用 --save-name lstm    训练产出)
  - src/model_out/test_metrics_pathint.json
  - src/model_out/test_metrics_lstm.json
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
from model_lstm import LSTMForecaster


def load_ckpt(ckpt_path, model_cls, model_kwargs, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    return model, x_scaler


@torch.no_grad()
def predict_seq(model, x_scaler, exp, in_len, T_out, device):
    """对一条实验：用前 in_len 步做输入，rollout T_out 步；返回预测 x 与真实 x（原始空间）。"""
    x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
    x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
    out = model(x_in, T_out=T_out)
    pred_x = out["pred_x"][0].cpu().numpy() * x_scaler.std + x_scaler.mean
    truth_x = exp.df[X_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
    return x_raw, truth_x, pred_x


def main():
    base = "/kefu-nas/ybkong/time_serials-master"
    out = os.path.join(base, "src/analysis_out")
    model_dir = os.path.join(base, "src/model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载两个模型
    pathint_ckpt = os.path.join(model_dir, "forecaster_pathint.pt")
    lstm_ckpt = os.path.join(model_dir, "forecaster_lstm.pt")

    if os.path.exists(pathint_ckpt):
        pi_model, x_scaler_pi = load_ckpt(
            pathint_ckpt, PathIntegratorForecaster,
            dict(dim_x=8, dim_state=128, hidden=128), device,
        )
    else:
        pi_model = x_scaler_pi = None
        print("[compare] 未找到 forecaster_pathint.pt；只用当前 forecaster_best.pt")

    if os.path.exists(lstm_ckpt):
        lstm_model, x_scaler_l = load_ckpt(
            lstm_ckpt, LSTMForecaster,
            dict(dim_x=8, hidden=128, num_layers=2, dropout=0.1), device,
        )
    else:
        lstm_model = x_scaler_l = None

    # 兼容：如果两个模型不存在则退化为"对当前 best 做多视角可视化"
    if pi_model is None and lstm_model is None:
        # 用当前 ckpt 作为 pi_model
        ckpt_path = os.path.join(model_dir, "forecaster_best.pt")
        pi_model, x_scaler_pi = load_ckpt(
            ckpt_path, PathIntegratorForecaster,
            dict(dim_x=8, dim_state=128, hidden=128), device,
        )
        x_scaler = x_scaler_pi
    else:
        x_scaler = x_scaler_pi or x_scaler_l

    exps = load_all(base)
    _, _, test_exps = split_experiments(exps)
    test_exps = sorted(test_exps, key=lambda e: len(e), reverse=True)[:3]
    print(f"[compare] 可视化实验: {[os.path.basename(e.file) for e in test_exps]}")

    # 2) 画对比图（每条实验单独一张）
    in_len, T_out = 24, 16
    for exp in test_exps:
        truth = exp.df[X_COLS].iloc[:in_len + T_out].to_numpy(dtype=np.float32)
        t_full = np.arange(len(truth))
        t_in = np.arange(in_len)
        t_out = np.arange(in_len, in_len + T_out)

        # 生成预测
        x_raw_in, truth_x, pred_pi = (None, None, None)
        if pi_model is not None:
            x_raw_in, truth_x, pred_pi = predict_seq(pi_model, x_scaler_pi, exp, in_len, T_out, device)
        pred_lstm = None
        if lstm_model is not None:
            _, truth_x_l, pred_lstm = predict_seq(lstm_model, x_scaler_l, exp, in_len, T_out, device)
            if truth_x is None:
                truth_x = truth_x_l

        fig, axes = plt.subplots(2, 4, figsize=(24, 7))
        fig.suptitle(f"{os.path.basename(exp.file)} — in={in_len}, out={T_out}", fontsize=12)
        for ci, c in enumerate(X_COLS):
            ax = axes[ci // 4, ci % 4]
            ax.plot(t_in, truth[:in_len, ci], "k-", label="history", linewidth=1.2)
            ax.plot(t_out, truth_x[:, ci], "g-", label="truth", linewidth=1.5, marker="o", markersize=3)
            if pred_pi is not None:
                ax.plot(t_out, pred_pi[:, ci], "r--", label="PathInt", linewidth=1.5, marker="x", markersize=3)
            if pred_lstm is not None:
                ax.plot(t_out, pred_lstm[:, ci], "b:", label="LSTM", linewidth=1.5, marker="s", markersize=2, alpha=0.7)
            ax.set_title(c, fontsize=10)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(out, f"compare_{os.path.basename(exp.file).replace('.csv','')}.png")
        plt.savefig(out_path, dpi=110)
        plt.close()
        print(f"[compare] 写出 {out_path}")

    # 3) 分维度 RMSE 对比（如果两个 metrics 文件都存在）
    metrics_files = {
        "PathInt": os.path.join(model_dir, "test_metrics_pathint.json"),
        "LSTM": os.path.join(model_dir, "test_metrics_lstm.json"),
    }
    avail = {k: v for k, v in metrics_files.items() if os.path.exists(v)}
    if len(avail) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        data = {k: np.array(json.load(open(v))["rmse_x_per_dim"]) for k, v in avail.items()}
        x = np.arange(8)
        width = 0.35
        for i, (name, vals) in enumerate(data.items()):
            axes[0].bar(x + (i - 0.5) * width, vals, width, label=name)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(X_COLS)
        axes[0].set_ylabel("RMSE (标准化空间)")
        axes[0].set_title("分维度 RMSE(x) — PathInt vs LSTM")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # 相对差异 %（PathInt - LSTM）/ LSTM
        if "PathInt" in data and "LSTM" in data:
            diff_pct = (data["PathInt"] - data["LSTM"]) / data["LSTM"] * 100
            colors = ["green" if d < 0 else "red" for d in diff_pct]
            axes[1].bar(X_COLS, diff_pct, color=colors)
            axes[1].axhline(0, color="k", linewidth=0.8)
            axes[1].set_ylabel("相对差 (PathInt - LSTM) / LSTM × 100%")
            axes[1].set_title("PathInt 相对 LSTM 的 RMSE 变化（绿=胜，红=负）")
            axes[1].grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out_path = os.path.join(out, "rmse_compare_bars.png")
        plt.savefig(out_path, dpi=110)
        plt.close()
        print(f"[compare] 写出 {out_path}")

    print(f"[compare] 完成")


if __name__ == "__main__":
    main()