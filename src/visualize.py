"""
可视化（x 预测专用版，多 backbone × 多 mode）
=============================================
只可视化 x1-x8 的预测（不带 y）。

与 train_multigroup.py / compare_multigroup.py 对齐：按 --backbone + --mode 选择
模型类、checkpoint、forward 签名（group_head/independent 需传 group_id），
并现场重算分维度 RMSE（不再读可能与文件名/key 对不上的 JSON）。

依赖：model_out/ 下的 checkpoint（如 forecaster_lstm_group_head.pt）
输出（analysis_out/）：
  - forecast_x1_x8_{backbone}_{mode}.png ：x1-x8 真实 vs 预测（多步 rollout，2 条样例）
  - error_per_dim_{backbone}_{mode}.png  ：分维度 RMSE 柱状图（标准化 / 原始空间）

用法示例（默认：输入窗口锚在开头，预测紧接输入之后）：
  python src/visualize.py --base-dir /path/to/repo --backbone lstm --mode group_head
  python src/visualize.py --backbone pathint --mode independent --ckpt src/model_out/xxx.pt

尾部锚定（外推序列最后 H 步，对齐 scripts_control/06_visualize 的语义）：
  python src/visualize.py --base-dir <repo> --backbone lstm --mode group_head \
      --tail-anchor --in-len 32 --t-out 32

参数说明：
  --in-len C      输入窗口长度（历史/上下文步数），默认 24
  --t-out H       预测视野（外推步数），默认 16
  --tail-anchor   尾部锚定：s = T − H，历史画 [0, s)，预测画 [s, s+H)，
                  输入窗口取 s 前最后 in_len 步（默认关闭）
  --pred-start    预测区间起点（非 tail-anchor 时可用；tail-anchor 下被忽略）
  --split S       用哪个 split 画预测曲线 + 算 RMSE：train/val/test，默认 test

六路模型命令（--backbone × --mode → 6 个模型类，均可追加 --tail-anchor 做尾部锚定）：
  lstm    shared        → LSTMForecaster
  lstm    group_head    → LSTMForecasterFiLM
  lstm    independent   → LSTMForecaster5Models
  pathint shared        → PathIntegratorForecaster
  pathint group_head    → PathIntegratorForecasterFiLM
  pathint independent   → PathIntegratorForecaster5Models
  例：python src/visualize.py --backbone lstm --mode group_head \
          --tail-anchor --in-len 32 --t-out 32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS, sample_windows, Scaler
from model_lstm import LSTMForecaster
from model_forecaster import PathIntegratorForecaster
from model_multigroup import (
    LSTMForecasterFiLM, LSTMForecaster5Models,
    PathIntegratorForecasterFiLM, PathIntegratorForecaster5Models,
)
from train_multigroup import (
    split_experiments_groupwise, WindowXGDataset, pad_collate_xg,
    evaluate as evaluate_multigroup,
)


# ===================== 模型构建（6 路分发，对齐 compare_multigroup） =====================
def build_model(backbone: str, mode: str, hidden: int = 128,
                dim_state: int = 128, num_layers: int = 2, dropout: float = 0.1):
    """根据 backbone + mode 构建模型。默认超参与训练脚本一致。"""
    if backbone == "lstm":
        if mode == "shared":
            return LSTMForecaster(dim_x=8, hidden=hidden, num_layers=num_layers,
                                  dropout=dropout)
        if mode == "group_head":
            return LSTMForecasterFiLM(n_groups=5, dim_x=8, hidden=hidden,
                                      num_layers=num_layers, dropout=dropout)
        if mode == "independent":
            return LSTMForecaster5Models(n_groups=5, dim_x=8, hidden=hidden,
                                         num_layers=num_layers, dropout=dropout)
        raise ValueError(f"unknown mode for backbone 'lstm': {mode!r}")
    if backbone == "pathint":
        if mode == "shared":
            return PathIntegratorForecaster(dim_x=8, dim_state=dim_state, hidden=hidden)
        if mode == "group_head":
            return PathIntegratorForecasterFiLM(n_groups=5, dim_x=8,
                                                dim_state=dim_state, hidden=hidden)
        if mode == "independent":
            return PathIntegratorForecaster5Models(n_groups=5, dim_x=8,
                                                   dim_state=dim_state, hidden=hidden)
        raise ValueError(f"unknown mode for backbone 'pathint': {mode!r}")
    raise ValueError(f"unknown backbone: {backbone!r}")


def resolve_ckpt_path(model_dir: str, backbone: str, mode: str, ckpt_override: str | None) -> str:
    """精确匹配唯一的约定文件名；找不到就报错，不做静默 fallback。"""
    if ckpt_override:
        return ckpt_override
    p = os.path.join(model_dir, f"forecaster_{backbone}_{mode}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"找不到 checkpoint: {p}\n"
            f"请先运行 train_multigroup.py / compare_multigroup.py 生成，或用 --ckpt 显式指定路径。"
        )
    return p


def load_model(ckpt_path: str, backbone: str, mode: str, device):
    """加载 checkpoint，重建模型与 scaler（scaler 缺失时返回 None，由调用方 refit）。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args") or {}
    model = build_model(
        backbone, mode,
        hidden=args.get("hidden", 128),
        dim_state=args.get("dim_state", 128),
        num_layers=args.get("num_layers", 2),
        dropout=args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    xs = ckpt.get("x_scaler")
    x_scaler = None
    if xs is not None:
        x_scaler = Scaler(mean=np.asarray(xs["mean"]), std=np.asarray(xs["std"]))
    return model, x_scaler


def build_eval_loader(exps, x_scaler: Scaler, seed: int):
    """构造与训练一致的 groupwise loader（带 group_id）。

    ``seed`` 是已含 split 偏移的采样种子：train=seed+0 / val=seed+1 / test=seed+2。
    """
    samples = sample_windows(exps, rng_seed=seed)
    m = {i: int(e.group) - 1 for i, e in enumerate(exps)}
    for s in samples:
        s.group = m[s.exp_id]
    ds = WindowXGDataset(samples, x_scaler)
    return DataLoader(ds, batch_size=32, shuffle=False,
                      collate_fn=pad_collate_xg, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/kefu-nas/ybkong/time_serials-master")
    ap.add_argument("--backbone", choices=["lstm", "pathint"], default="lstm")
    ap.add_argument("--mode", choices=["shared", "group_head", "independent"],
                    default="group_head")
    ap.add_argument("--ckpt", default=None, help="显式指定 checkpoint 路径（覆盖自动推断）")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="用哪个 split 的实验画预测曲线 + 算 RMSE（默认 test）")
    ap.add_argument("--in-len", type=int, default=24,
                    help="输入窗口长度 in_len（历史/上下文步数），默认 24")
    ap.add_argument("--t-out", type=int, default=16,
                    help="预测视野 H（外推步数），默认 16；tail-anchor 下即 s = T − H 的 H")
    ap.add_argument("--pred-start", type=int, default=None,
                    help="预测区间起点（步）；默认=输入长度（紧接输入之后）。输入始终从 0 开始")
    ap.add_argument("--tail-anchor", action="store_true",
                    help="尾部锚定：s = T − H，历史画 [0, s)，预测画 [s, s+H)，"
                         "输入窗口取 s 前最后 in_len 步（对齐 scripts_control 外推语义）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = args.base_dir
    out = os.path.join(base, "src", "analysis_out")
    model_dir = os.path.join(base, "src", "model_out")
    os.makedirs(out, exist_ok=True)
    device = torch.device(args.device)

    ckpt_path = resolve_ckpt_path(model_dir, args.backbone, args.mode, args.ckpt)
    print(f"[viz] backbone={args.backbone} mode={args.mode}")
    print(f"[viz] checkpoint: {ckpt_path}")
    model, x_scaler = load_model(ckpt_path, args.backbone, args.mode, device)

    exps = load_all(base)
    train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=args.seed)
    if x_scaler is None:
        # compare_multigroup 的 ckpt 不存 scaler → 用 groupwise train 集 refit（与训练一致）
        x_scaler = Scaler.fit(train_exps)
        print("[viz] ckpt 无 scaler，已用 groupwise train 集 refit")

    split_exps = {"train": train_exps, "val": val_exps, "test": test_exps}[args.split]

    # 1) 真实 vs 预测的 x1-x8（每 group 各取最长的一条实验）
    viz_exps = []
    for g in ["1", "2", "3", "4", "5"]:
        gs = [e for e in split_exps if e.group == g]
        if gs:
            viz_exps.append(max(gs, key=lambda e: len(e)))
    print(f"[viz] 可视化实验（{args.split}）: "
          f"{[f'G{e.group} {os.path.basename(e.file)}' for e in viz_exps]}")

    fig, axes = plt.subplots(len(viz_exps), 8, figsize=(24, 4 * len(viz_exps)))
    if len(viz_exps) == 1:
        axes = axes[None, :]
    for ei, exp in enumerate(viz_exps):
        in_len, T_out = args.in_len, args.t_out
        if len(exp) < in_len + T_out:
            in_len = min(20, len(exp) // 2)
            T_out = min(12, len(exp) - in_len)

        if args.tail_anchor:
            # 尾部锚定：s = T − H，历史画 [0, s)，预测画 [s, s+H)，
            # 输入窗口取 s 前最后 in_len 步（对齐 scripts_control 外推语义）。
            s = len(exp) - T_out
            input_start = s - in_len
            pred_start = s
            T_roll = T_out  # 只从输入末端 rollout H 步
            x_hist = exp.df[X_COLS].iloc[:s].to_numpy(dtype=np.float32)
            x_in_raw = exp.df[X_COLS].iloc[input_start:s].to_numpy(dtype=np.float32)
        else:
            # 输入始终从 0 开始；预测区间起点可设置（不能早于输入结束，不能超出实验末尾）
            input_start = 0
            pred_start = in_len if args.pred_start is None else max(int(args.pred_start), in_len)
            pred_start = min(pred_start, max(in_len, len(exp) - T_out))
            T_roll = pred_start - in_len + T_out  # 从输入末端自回归 rollout 到预测区间末端
            x_hist = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
            x_in_raw = x_hist

        x_in = torch.from_numpy(x_scaler.transform(x_in_raw)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if args.mode == "shared":
                out_dict = model(x_in, T_out=T_roll)
            else:
                gid = torch.tensor([int(exp.group) - 1], device=device)
                out_dict = model(x_in, gid, T_out=T_roll)
        pred_full = out_dict["pred_x"][0].cpu().numpy() * x_scaler.std + x_scaler.mean
        pred_x = pred_full if args.tail_anchor else pred_full[pred_start - in_len:]
        x_future = exp.df[X_COLS].iloc[pred_start:pred_start + T_out].to_numpy(dtype=np.float32)
        print(f"[viz] G{exp.group} {os.path.basename(exp.file)}: "
              f"input {input_start}-{pred_start - 1}, pred {pred_start}-{pred_start + T_out - 1}")
        for ci, c in enumerate(X_COLS):
            ax = axes[ei, ci]
            t_hist = np.arange(x_hist.shape[0])
            t_out = np.arange(pred_start, pred_start + T_out)
            ax.plot(t_hist, x_hist[:, ci], "k-", label="history", linewidth=1.5)
            ax.plot(t_out, x_future[:, ci], "g-", label="truth", linewidth=2.5,
                    marker="o", markersize=4)
            ax.plot(t_out, pred_x[:, ci], "r--", label="pred", linewidth=1.0,
                    marker="x", markersize=3, alpha=0.7)
            ax.set_title(f"{c} ({os.path.basename(exp.file)})", fontsize=9)
            if ci == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    split_tag = "" if args.split == "test" else f"_{args.split}"
    tag = f"{args.backbone}_{args.mode}{split_tag}"
    plt.savefig(os.path.join(out, f"forecast_x1_x8_{tag}.png"), dpi=120)
    plt.close()
    print(f"[viz] 写出 forecast_x1_x8_{tag}.png")

    # 2) 分维度 RMSE（现场重算，柱状图 + 数值）
    seed_off = {"train": 0, "val": 1, "test": 2}[args.split]
    eval_loader = build_eval_loader(split_exps, x_scaler, args.seed + seed_off)
    metrics = evaluate_multigroup(model, eval_loader, device, args.mode, x_scaler,
                                  return_by_group=True)
    per_dim_norm = np.array(metrics["rmse_x_per_dim_norm"])
    per_dim_orig = np.array(metrics["rmse_x_per_dim_orig"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(X_COLS, per_dim_norm, color="steelblue")
    axes[0].set_title(f"Per-dim RMSE(x), normalized space [{args.split}]  "
                      f"(mean={metrics['rmse_x_norm']:.4f})")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(X_COLS, per_dim_orig, color="darkorange")
    axes[1].set_title(f"Per-dim RMSE(x), original space [{args.split}]  "
                      f"(mean={metrics['rmse_x_orig']:.2f})")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"error_per_dim_{tag}.png"), dpi=120)
    plt.close()
    print(f"[viz] 写出 error_per_dim_{tag}.png")

    print(f"[viz] 完成。所有图片已写入 {out}/")


if __name__ == "__main__":
    main()
