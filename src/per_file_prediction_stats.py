"""
按文件夹分组的预测偏差统计
=============================
对测试集里每条实验做预测，统计每个文件的预测偏差（真实值 vs 预测值的 RMSE），
再按所属文件夹（1/2/3/4/5）分组，画 boxplot + scatter。

支持两种模型 ckpt：
  - PathIntegratorForecaster (model_forecaster.py)
  - LSTMForecaster (model_lstm.py)
按文件名前缀自动选择。

用法：
    python src/per_file_prediction_stats.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS, split_experiments, Scaler
from model_forecaster import PathIntegratorForecaster
from model_lstm import LSTMForecaster


def _state_keys(state):
    return list(state.keys())


def load_model_any(ckpt_path:str, device):
    """按 state_dict 形状自动选 PathInt 或 LSTM。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    state = ckpt["model"]
    is_lstm = any(k.startswith("lstm.") for k in state.keys())
    if is_lstm:
        model = LSTMForecaster(dim_x=8, hidden=args["hidden"], num_layers=2, dropout=0.1).to(device)
    else:
        model = PathIntegratorForecaster(
            dim_x=8, dim_state=args["dim_state"], hidden=args["hidden"]
        ).to(device)
    model.load_state_dict(state)
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    model.eval()
    return model, x_scaler, "lstm" if is_lstm else "pathint"


@torch.no_grad()
def predict_one(model, x_scaler, exp, in_len=24, T_out=16, device="cpu"):
    """对一条实验做 in_len 步 → T_out 步预测；返回标准化空间的预测 + 真值。"""
    if len(exp) < in_len + T_out:
        return None, None
    x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
    x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
    out = model(x_in, T_out=T_out)
    pred = out["pred_x"][0].cpu().numpy()                  # 标准化空间
    truth = exp.df[X_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
    truth_norm = (truth - x_scaler.mean) / x_scaler.std
    return pred, truth_norm


def per_dim_rmse(pred, truth):
    """pred/truth: (T, 8)。返回 (8,) 各维度 RMSE（标准化空间）。"""
    return np.sqrt(((pred - truth) ** 2).mean(axis=0))


def main():
    base = "/kefu-nas/ybkong/time_serials-master"
    out = os.path.join(base, "src/analysis_out")
    model_dir = os.path.join(base, "src/model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_scaler, model_kind = load_model_any(
        os.path.join(model_dir, "forecaster_best.pt"), device,
    )
    print(f"[stats] 使用模型: {model_kind}")

    exps = load_all(base)
    _, _, test_exps = split_experiments(exps, seed=42)
    print(f"[stats] test experiments: {len(test_exps)}")

    # 每文件总体 RMSE（标准化空间，8 维平均）
    records = []
    for exp in test_exps:
        pred, truth = predict_one(model, x_scaler, exp, in_len=24, T_out=16, device=device)
        if pred is None:
            continue
        rmse_per_dim = per_dim_rmse(pred, truth)
        overall_rmse = float(rmse_per_dim.mean())
        records.append({
            "file": os.path.basename(exp.file),
            "group": exp.group,
            "rmse_overall": overall_rmse,
            "rmse_per_dim": rmse_per_dim.tolist(),
        })

    if not records:
        print("[stats] no records")
        return

    # 排序：按 group 升序，再按 rmse
    records.sort(key=lambda r: (r["group"], r["rmse_overall"]))

    # ============ 打印表格 ============
    print("\n=== 每个文件的预测偏差（标准化空间整体 RMSE）===")
    print(f"{'file':<22} {'group':<6} {'rmse':>8}")
    for r in records:
        print(f"{r['file']:<22} {r['group']:<6} {r['rmse_overall']:>8.4f}")

    # ============ 按文件夹统计 ============
    groups = ["1", "2", "3", "4", "5"]
    group_rmses = {g: [] for g in groups}
    for r in records:
        group_rmses[r["group"]].append(r["rmse_overall"])

    print("\n=== 按文件夹均值 / 中位数 ===")
    for g in groups:
        vals = group_rmses[g]
        if vals:
            print(f"  dir {g}: n={len(vals):>3}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}")

    # ============ 图 1：按组别的 boxplot ============
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [group_rmses[g] if group_rmses[g] else [np.nan] for g in groups]
    bp = ax.boxplot(data, labels=[f"dir {g}" for g in groups], patch_artist=True, showmeans=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(plt.cm.tab10(int(g) - 1))
        patch.set_alpha(0.6)
    ax.set_ylabel("整体 RMSE（标准化空间）")
    ax.set_title("测试集每个文件的预测偏差 — 按文件夹分组 boxplot")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out1 = os.path.join(out, "per_file_rmse_boxplot.png")
    plt.savefig(out1, dpi=120)
    plt.close()
    print(f"\n[stats] 写出 {out1}")

    # ============ 图 2：散点（横轴=文件 index，按组上色） ============
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10
    x_positions = list(range(len(records)))
    for i, r in enumerate(records):
        ax.scatter(i, r["rmse_overall"], color=cmap(int(r["group"]) - 1), s=40, alpha=0.85)
    # 在底部画每个 group 的范围条
    group_ranges = {}
    for i, r in enumerate(records):
        group_ranges.setdefault(r["group"], []).append(i)
    ymin, ymax = ax.get_ylim()
    for g, idxs in group_ranges.items():
        if not idxs:
            continue
        ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5, alpha=0.08, color=cmap(int(g) - 1))
        ax.text((min(idxs) + max(idxs)) / 2, ymin, f"dir {g}",
                ha="center", va="bottom", fontsize=9, color=cmap(int(g) - 1))
    ax.set_xticks([])
    ax.set_xlabel("测试文件（按 group 排序）")
    ax.set_ylabel("整体 RMSE（标准化空间）")
    ax.set_title("每个文件的预测偏差 — 散点 + 颜色标注所属文件夹")
    ax.grid(True, alpha=0.3, axis="y")
    # 图例
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap(int(g) - 1), markersize=8, label=f"dir {g}")
               for g in groups]
    ax.legend(handles=handles, loc="upper right", title="文件夹")
    plt.tight_layout()
    out2 = os.path.join(out, "per_file_rmse_scatter.png")
    plt.savefig(out2, dpi=120)
    plt.close()
    print(f"[stats] 写出 {out2}")

    # ============ 图 3：分维度的 RMSE 箱线图（按文件夹） ============
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    for ci in range(8):
        ax = axes[ci // 4, ci % 4]
        dim_vals = {g: [] for g in groups}
        for r in records:
            dim_vals[r["group"]].append(r["rmse_per_dim"][ci])
        data = [dim_vals[g] for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_color(cmap(int(g) - 1))
        ax.set_title(f"{X_COLS[ci]}", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ci % 4 == 0:
            ax.set_ylabel("RMSE")
    fig.suptitle("每个维度的预测 RMSE — 按文件夹分组 boxplot", fontsize=12)
    plt.tight_layout()
    out3 = os.path.join(out, "per_dim_rmse_boxplot.png")
    plt.savefig(out3, dpi=120)
    plt.close()
    print(f"[stats] 写出 {out3}")


if __name__ == "__main__":
    main()


@torch.no_grad()
def predict_one(model, x_scaler, exp, in_len=24, T_out=16, device="cpu"):
    """对一条实验做 in_len 步 → T_out 步预测；返回标准化空间的预测 + 真值。"""
    if len(exp) < in_len + T_out:
        return None, None
    x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
    x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().unsqueeze(0).to(device)
    out = model(x_in, T_out=T_out)
    pred = out["pred_x"][0].cpu().numpy()                  # 标准化空间
    truth = exp.df[X_COLS].iloc[in_len:in_len + T_out].to_numpy(dtype=np.float32)
    truth_norm = (truth - x_scaler.mean) / x_scaler.std
    return pred, truth_norm


def per_dim_rmse(pred, truth):
    """pred/truth: (T, 8)。返回 (8,) 各维度 RMSE（标准化空间）。"""
    return np.sqrt(((pred - truth) ** 2).mean(axis=0))


def main():
    base = "/kefu-nas/ybkong/time_serials-master"
    out = os.path.join(base, "src/analysis_out")
    model_dir = os.path.join(base, "src/model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_scaler, model_kind = load_model_any(
        os.path.join(model_dir, "forecaster_best.pt"), device,
    )
    print(f"[stats] 使用模型: {model_kind}")

    exps = load_all(base)
    _, _, test_exps = split_experiments(exps, seed=42)
    print(f"[stats] test experiments: {len(test_exps)}")

    # 每文件总体 RMSE（标准化空间，8 维平均）
    records = []
    for exp in test_exps:
        pred, truth = predict_one(model, x_scaler, exp, in_len=24, T_out=16, device=device)
        if pred is None:
            continue
        rmse_per_dim = per_dim_rmse(pred, truth)
        overall_rmse = float(rmse_per_dim.mean())
        records.append({
            "file": os.path.basename(exp.file),
            "group": exp.group,
            "rmse_overall": overall_rmse,
            "rmse_per_dim": rmse_per_dim.tolist(),
        })

    if not records:
        print("[stats] no records")
        return

    # 排序：按 group 升序，再按 rmse
    records.sort(key=lambda r: (r["group"], r["rmse_overall"]))

    # ============ 打印表格 ============
    print("\n=== 每个文件的预测偏差（标准化空间整体 RMSE）===")
    print(f"{'file':<22} {'group':<6} {'rmse':>8}")
    for r in records:
        print(f"{r['file']:<22} {r['group']:<6} {r['rmse_overall']:>8.4f}")

    # ============ 按文件夹统计 ============
    groups = ["1", "2", "3", "4", "5"]
    group_rmses = {g: [] for g in groups}
    for r in records:
        group_rmses[r["group"]].append(r["rmse_overall"])

    print("\n=== 按文件夹均值 / 中位数 ===")
    for g in groups:
        vals = group_rmses[g]
        if vals:
            print(f"  dir {g}: n={len(vals):>3}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}")

    # ============ 图 1：按组别的 boxplot ============
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [group_rmses[g] if group_rmses[g] else [np.nan] for g in groups]
    bp = ax.boxplot(data, labels=[f"dir {g}" for g in groups], patch_artist=True, showmeans=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(plt.cm.tab10(int(g) - 1))
        patch.set_alpha(0.6)
    ax.set_ylabel("整体 RMSE（标准化空间）")
    ax.set_title("测试集每个文件的预测偏差 — 按文件夹分组 boxplot")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out1 = os.path.join(out, "per_file_rmse_boxplot.png")
    plt.savefig(out1, dpi=120)
    plt.close()
    print(f"\n[stats] 写出 {out1}")

    # ============ 图 2：散点（横轴=文件 index，按组上色） ============
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10
    x_positions = list(range(len(records)))
    for i, r in enumerate(records):
        ax.scatter(i, r["rmse_overall"], color=cmap(int(r["group"]) - 1), s=40, alpha=0.85)
    # 在底部画每个 group 的范围条
    group_ranges = {}
    for i, r in enumerate(records):
        group_ranges.setdefault(r["group"], []).append(i)
    for g, idxs in group_ranges.items():
        if not idxs:
            continue
        ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5, alpha=0.08, color=cmap(int(g) - 1))
        ax.text((min(idxs) + max(idxs)) / 2, ax.get_ylim()[0], f"dir {g}",
                ha="center", va="bottom", fontsize=9, color=cmap(int(g) - 1))
    ax.set_xticks([])
    ax.set_xlabel("测试文件（按 group 排序）")
    ax.set_ylabel("整体 RMSE（标准化空间）")
    ax.set_title("每个文件的预测偏差 — 散点 + 颜色标注所属文件夹")
    ax.grid(True, alpha=0.3, axis="y")
    # 图例
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap(int(g) - 1), markersize=8, label=f"dir {g}")
               for g in groups]
    ax.legend(handles=handles, loc="upper right", title="文件夹")
    plt.tight_layout()
    out2 = os.path.join(out, "per_file_rmse_scatter.png")
    plt.savefig(out2, dpi=120)
    plt.close()
    print(f"[stats] 写出 {out2}")

    # ============ 图 3：分维度的 RMSE 箱线图（按文件夹） ============
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    for ci in range(8):
        ax = axes[ci // 4, ci % 4]
        dim_vals = {g: [] for g in groups}
        for r in records:
            dim_vals[r["group"]].append(r["rmse_per_dim"][ci])
        data = [dim_vals[g] for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(cmap(int(g) - 1))
            patch.set_alpha(0.6)
        ax.set_title(f"{X_COLS[ci]}", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ci % 4 == 0:
            ax.set_ylabel("RMSE")
    fig.suptitle("每个维度的预测 RMSE — 按文件夹分组 boxplot", fontsize=12)
    plt.tight_layout()
    out3 = os.path.join(out, "per_dim_rmse_boxplot.png")
    plt.savefig(out3, dpi=120)
    plt.close()
    print(f"[stats] 写出 {out3}")


if __name__ == "__main__":
    main()