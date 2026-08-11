"""
预测可视化对比：四模型 × 每组一条样本
=======================================

在同一个 test 集上，为每组选一条代表性样本，画出：
  - 黑色实线：历史输入
  - 绿色实线 ○：真实未来
  - 4 条模型预测线：
      LSTM shared      (蓝色虚线)
      LSTM group_head  (蓝色实线)
      PathInt shared   (红色虚线)
      PathInt group_head (红色实线)

依赖：model_out/forecaster_{lstm,pathint}_{shared,group_head}.pt
输出：analysis_out/compare_predictions.png
"""

from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, sample_windows, Scaler, X_COLS
from model_lstm import LSTMForecaster
from model_forecaster import PathIntegratorForecaster
from model_multigroup import (
    LSTMForecasterFiLM,
    PathIntegratorForecasterFiLM,
)
from train_multigroup import split_experiments_groupwise, WindowXGDataset


def build_model(backbone: str, mode: str, hidden=128, dim_state=128):
    if backbone == "lstm":
        if mode == "shared":
            return LSTMForecaster(dim_x=8, hidden=hidden, num_layers=2, dropout=0.1)
        else:
            return LSTMForecasterFiLM(n_groups=5, dim_x=8, hidden=hidden,
                                       num_layers=2, dropout=0.1)
    else:
        if mode == "shared":
            return PathIntegratorForecaster(dim_x=8, dim_state=dim_state, hidden=hidden)
        else:
            return PathIntegratorForecasterFiLM(n_groups=5, dim_x=8,
                                                  dim_state=dim_state, hidden=hidden)


def load_ckpt(ckpt_path, backbone, mode, device, hidden=128, dim_state=128):
    model = build_model(backbone, mode, hidden=hidden, dim_state=dim_state).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict(model, x_in, backbone, mode, group_id, T_out, device):
    """对单条样本做预测，返回 pred_x (1, T_out, 8) numpy 原始空间。"""
    x_t = torch.from_numpy(x_in).float().unsqueeze(0).to(device)  # (1, L, 8)
    with torch.no_grad():
        if mode == "shared":
            out = model(x_t, T_out=T_out)
        else:
            g = torch.tensor([group_id], device=device)
            out = model(x_t, g, T_out=T_out)
    return out["pred_x"][0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/remote-home/sunxiaoting/ybkong/timserials/time-serials-mac")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim-state", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--in-len", type=int, default=24, help="输入历史步数")
    ap.add_argument("--T-out", type=int, default=16, help="预测未来步数")
    ap.add_argument("--n-samples-per-group", type=int, default=2,
                    help="每组选几条样本")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, "src", "analysis_out")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model_dir = os.path.join(args.base_dir, "src", "model_out")

    # ──── 1. 加载数据，统一 groupwise split ────
    exps = load_all(args.base_dir)
    train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=args.seed)
    print(f"实验: train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")

    x_scaler = Scaler.fit(train_exps)

    # ──── 2. 加载 4 个模型 ────
    model_specs = [
        ("LSTM shared",       "lstm",    "shared",     "forecaster_lstm_shared.pt"),
        ("LSTM group_head",   "lstm",    "group_head", "forecaster_lstm_group_head.pt"),
        ("PathInt shared",    "pathint", "shared",     "forecaster_pathint_shared.pt"),
        ("PathInt group_head","pathint", "group_head", "forecaster_pathint_group_head.pt"),
    ]

    models = {}
    for label, backbone, mode, ckpt_name in model_specs:
        ckpt_path = os.path.join(model_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 找不到 {ckpt_path}")
            continue
        m = load_ckpt(ckpt_path, backbone, mode, device,
                      hidden=args.hidden, dim_state=args.dim_state)
        models[label] = {"model": m, "backbone": backbone, "mode": mode}
        print(f"[加载] {label}")

    if len(models) < 4:
        print(f"[警告] 只找到 {len(models)} 个模型，继续...")

    # ──── 3. 每组选代表性样本 ────
    in_len, T_out = args.in_len, args.T_out
    # 从 test set 里每组挑几个足够长的实验
    samples_by_group = {g: [] for g in range(5)}
    for exp in test_exps:
        g = int(exp.group) - 1
        if len(exp) >= in_len + T_out:
            if len(samples_by_group[g]) < args.n_samples_per_group:
                samples_by_group[g].append(exp)

    # 检查是否每组都有样本
    for g in range(5):
        if not samples_by_group[g]:
            print(f"[警告] group {g+1} 无足够长样本，跳过")

    n_cols = args.n_samples_per_group
    n_rows = 5  # 5 groups

    # ──── 4. 画图：每行一个 group × 每列一条样本 ────
    # 只画一个代表性维度（如 x1），避免过于拥挤。如果想画全部 8 维，可加 --all-dims
    # 这里默认画 x1(=0) 和 x5(=4) 两个维度，分别做两张图
    dims_to_plot = [(5, "x6"), (6, "x7"), (7, "x8")]  # 选高 CV 维度（真正有预测难度）

    # 颜色方案
    colors = {
        "LSTM shared":       "#64b5f6",  # 浅蓝 虚线
        "LSTM group_head":   "#1565c0",  # 深蓝 实线
        "PathInt shared":    "#ef9a9a",  # 浅红 虚线
        "PathInt group_head":"#c62828",  # 深红 实线
    }
    linestyles = {
        "LSTM shared":       "--",
        "LSTM group_head":   "-",
        "PathInt shared":    "--",
        "PathInt group_head":"-",
    }
    model_order = ["LSTM shared", "LSTM group_head", "PathInt shared", "PathInt group_head"]

    for dim_idx, dim_name in dims_to_plot:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 3.2 * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"Prediction Comparison — {dim_name}  |  "
                     f"blue=LSTM  red=PathInt  dashed=shared  solid=group_head",
                     fontsize=13, fontweight="bold", y=1.01)

        for gi in range(5):
            exps_g = samples_by_group[gi]
            for si in range(n_cols):
                ax = axes[gi, si]
                if si >= len(exps_g):
                    ax.set_visible(False)
                    continue

                exp = exps_g[si]
                x_raw = exp.df[X_COLS].iloc[:in_len + T_out].to_numpy(dtype=np.float32)
                x_norm = x_scaler.transform(x_raw)
                x_in = x_norm[:in_len]           # 模型输入（标准化）
                x_future_raw = x_raw[in_len:in_len + T_out]  # 真实未来（原始空间）
                group_id = int(exp.group) - 1

                # 画历史
                t_in = np.arange(in_len)
                t_out = np.arange(in_len, in_len + T_out)
                ax.plot(t_in, x_raw[:in_len, dim_idx], "k-", linewidth=1.5, label="History")
                # 画真实未来
                ax.plot(t_out, x_future_raw[:, dim_idx], "k-", linewidth=2.0,
                        marker="o", markersize=3, label="Truth")

                # 画各模型预测
                for label in model_order:
                    if label not in models:
                        continue
                    info = models[label]
                    pred_norm = predict(info["model"], x_in, info["backbone"],
                                        info["mode"], group_id, T_out, device)
                    pred_raw = pred_norm * x_scaler.std + x_scaler.mean

                    ax.plot(t_out, pred_raw[:, dim_idx],
                            color=colors[label], linestyle=linestyles[label],
                            linewidth=1.3, marker="x", markersize=2.5, label=label)

                ax.set_title(f"Group {gi+1} — {os.path.basename(exp.file)}", fontsize=9)
                ax.grid(True, alpha=0.3)
                if gi == 0 and si == 0:
                    ax.legend(fontsize=6.5, loc="best")

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"compare_predictions_{dim_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[写出] {out_path}")

    print("[完成] 所有对比图已生成。")


if __name__ == "__main__":
    main()
