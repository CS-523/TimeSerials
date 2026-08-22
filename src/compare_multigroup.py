"""
全面对比：LSTM vs PathInt 骨干 × 三种组策略
===============================================

在同一份 groupwise test 集上对比 6 个模型：

  LSTM  backbone:  shared / group_head / independent
  PathInt backbone: shared / group_head / independent

输出：
  src/model_out/compare_metrics.json   — 汇总指标
  src/model_out/compare_bars_lstm.png   — LSTM 柱状图
  src/model_out/compare_bars_pathint.png — PathInt 柱状图
  src/model_out/compare_bars_all.png    — 合并柱状图

用法：
  cd src

  # 训练（推荐：LSTM 和 PathInt 使用各自最优学习率）
  python compare_multigroup.py \\
    --base-dir /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac \\
    --epochs 60 \\
    --device cuda \\
    --lr-lstm 2e-3 \\
    --lr-pathint 5e-4

  # 预测模式（加载已有 checkpoint）
  python compare_multigroup.py \\
    --base-dir /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac \\
    --device cuda --skip-train \\
    --predict --predict-model "LSTM group_head" \\
    --predict-cycle-start 0 --predict-input-len 24 --predict-output-len 16

  # 预测模式（迭代长程预测）
  python compare_multigroup.py \\
    --base-dir /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac \\
    --device cuda --skip-train \\
    --predict --predict-model "PathInt group_head" \\
    --predict-cycle-start 0 --predict-input-len 24 --predict-output-len 60 \\
    --predict-chunk-len 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, sample_windows, Scaler, YScaler, X_COLS
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


# ────────────────────────── 工具 ──────────────────────────
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(backbone: str, mode: str, **kwargs):
    """根据 backbone + mode 构建模型。"""
    if backbone == "lstm":
        if mode == "shared":
            return LSTMForecaster(dim_x=8, hidden=kwargs.get("hidden", 128),
                                  num_layers=2, dropout=0.1)
        elif mode == "group_head":
            return LSTMForecasterFiLM(n_groups=5, dim_x=8,
                                      hidden=kwargs.get("hidden", 128),
                                      num_layers=2, dropout=0.1)
        else:  # independent
            return LSTMForecaster5Models(n_groups=5, dim_x=8,
                                         hidden=kwargs.get("hidden", 128),
                                         num_layers=2, dropout=0.1)
    else:  # pathint
        ds = kwargs.get("dim_state", 128)
        h = kwargs.get("hidden", 128)
        if mode == "shared":
            return PathIntegratorForecaster(dim_x=8, dim_state=ds, hidden=h)
        elif mode == "group_head":
            return PathIntegratorForecasterFiLM(n_groups=5, dim_x=8,
                                                dim_state=ds, hidden=h)
        else:  # independent
            return PathIntegratorForecaster5Models(n_groups=5, dim_x=8,
                                                   dim_state=ds, hidden=h)


# ────────────────────────── 训练 / 评估 ──────────────────────────
def train_one_epoch(model, loader, opt, device, mode: str, use_tf: bool = False):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        out_lens = batch["out_lens"].to(device)
        T_out = int(out_lens.max().item())
        if mode == "shared":
            out = model(x_in, T_out=T_out, x_out=x_out if use_tf else None)
        else:
            group_ids = batch["group_ids"].to(device)
            out = model(x_in, group_ids, T_out=T_out,
                        x_out=x_out if use_tf else None)
        x_loss = 0.0
        cnt = 0
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            L_eff = min(L, T_out)
            if L_eff > 0:
                x_loss += F.mse_loss(out["pred_x"][i, :L_eff], x_out[i, :L_eff])
                cnt += 1
        loss = x_loss / max(cnt, 1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


def train_model(model, train_loader, val_loader, device, args,
                mode: str, ckpt_path: str, x_scaler, y_scaler, lr: float,
                log_dir: str | None = None):
    """两阶段训练：Teacher Forcing (破冰) → 纯自回归 (抗压纠偏)。"""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}  LR={lr}")

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    if writer:
        print(f"  TensorBoard: {log_dir}")

    tf_epochs = getattr(args, "tf_epochs", 0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        use_tf = (tf_epochs > 0) and (ep < tf_epochs)
        if ep == 0:
            print(f"  阶段1: Teacher Forcing  (epoch 1-{tf_epochs})" if use_tf
                  else "  阶段1: Teacher Forcing (跳过)")
        if ep == tf_epochs:
            print(f"  >>> 切换至阶段2: 纯自回归 rollout <<<")

        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, mode, use_tf=use_tf)
        sched.step()
        val_m = evaluate_multigroup(model, val_loader, device, mode, x_scaler,
                                    return_by_group=False)
        elapsed = time.time() - t0
        tag = "[TF]" if use_tf else "[AR]"
        print(f"  {tag} Epoch {ep+1:02d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"val_rmse={val_m['rmse_x_norm']:.4f} | {elapsed:.1f}s")
        if writer is not None:
            writer.add_scalar("train/loss", tr_loss, ep)
            writer.add_scalar("val/rmse_norm", val_m["rmse_x_norm"], ep)
            writer.add_scalar("val/rmse_orig", val_m["rmse_x_orig"], ep)
            writer.add_scalar("lr", sched.get_last_lr()[0], ep)
        if val_m["rmse_x_norm"] < best_val:
            best_val = val_m["rmse_x_norm"]
            torch.save({
                "model": model.state_dict(),
                "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std},
                "args": vars(args),
            }, ckpt_path)

    if writer:
        writer.close()
    return n_params


# ────────────────────────── 画图 ──────────────────────────
def plot_comparison(results: dict, model_order: list, model_labels: dict,
                    colors: list, title: str, out_path: str):
    """画 5 组 × N 模型的柱状图。"""
    fig, axes = plt.subplots(1, 5, figsize=(min(4.5 * len(model_order), 22), 5.5),
                             sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for gi, g_str in enumerate(["1", "2", "3", "4", "5"]):
        ax = axes[gi]
        values = []
        for m_name in model_order:
            if m_name not in results:
                values.append(0)
                continue
            bg = results[m_name]["by_group"]
            # handle both int keys (0-4) and str keys ("1"-"5")
            g_int = int(g_str) - 1
            g_data = bg.get(g_int) if g_int in bg else bg.get(g_str, {})
            v = g_data.get("rmse_x_orig", 0) if isinstance(g_data, dict) else 0
            values.append(v if v else 0)

        x_pos = np.arange(len(model_order))
        bars = ax.bar(x_pos, values, color=colors[:len(model_order)],
                      edgecolor="white", linewidth=0.5)
        ax.set_title(f"Group {g_str}", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=7)
        ax.set_ylabel("RMSE (original space)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图表保存至: {out_path}")


def _save_metrics_json(results: dict, out_path: str):
    """把 results（含 by_group）序列化成 JSON。"""
    serializable = {}
    for name, data in results.items():
        serializable[name] = {
            "backbone": data["backbone"],
            "mode": data["mode"],
            "n_params": data["n_params"],
            "rmse_x_norm": data["rmse_x_norm"],
            "rmse_x_orig": data["rmse_x_orig"],
            "by_group": {},
        }
        for g_key, g_val in data["by_group"].items():
            g_key_str = str(g_key + 1) if isinstance(g_key, int) else g_key
            serializable[name]["by_group"][g_key_str] = {
                k: v for k, v in g_val.items()
                if k in ("n", "rmse_x_norm", "rmse_x_orig")
            }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"指标保存至: {out_path}")


def plot_tf_vs_ar(results_ar: dict, results_tf: dict, model_order: list,
                  model_labels: dict, out_path: str):
    """AR（纯自回归）vs TF（teacher forcing 一步外推）整体 RMSE 柱状对比。"""
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(model_order)), 5))
    x_pos = np.arange(len(model_order))
    ar_vals = [results_ar[m]["rmse_x_orig"] if m in results_ar else 0
               for m in model_order]
    tf_vals = [results_tf[m]["rmse_x_orig"] if m in results_tf else 0
               for m in model_order]
    w = 0.38
    ax.bar(x_pos - w / 2, ar_vals, w, label="Autoregressive (AR)", color="#42a5f5")
    ax.bar(x_pos + w / 2, tf_vals, w, label="Teacher Forcing (TF)", color="#66bb6a")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels.get(m, m) for m in model_order], fontsize=8)
    ax.set_ylabel("RMSE (original space)")
    ax.set_title("Overall RMSE: Autoregressive vs Teacher Forcing (test set)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars in ax.containers:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图表保存至: {out_path}")


# ────────────────────────── 主流程 ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="D:/Code/timeserials_claude/time-serials-mac")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="统一学习率（覆盖 --lr-lstm / --lr-pathint）")
    ap.add_argument("--lr-lstm", type=float, default=2e-3)
    ap.add_argument("--lr-pathint", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim-state", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练，仅用已有 ckpt 评估和画图")
    ap.add_argument("--tf-epochs", type=int, default=10,
                    help="前 N 个 epoch 使用 Teacher Forcing（喂真实值）；之后纯自回归")
    ap.add_argument("--modes", default="shared,group_head",
                    help="要运行的组策略，逗号分隔。可选: shared,group_head,independent "
                         "（默认: shared,group_head，不含 independent）")
    ap.add_argument("--log-dir", default=None,
                    help="TensorBoard 日志目录，默认 {out_dir}/tensorboard")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, "src", "model_out")
    if args.log_dir is None:
        args.log_dir = os.path.join(args.out_dir, "tensorboard")
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # ──── 1. 数据（统一 groupwise split）────
    exps = load_all(args.base_dir)
    train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=args.seed)
    print(f"实验: train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")
    for g in ["1", "2", "3", "4", "5"]:
        n_tr = sum(1 for e in train_exps if e.group == g)
        n_va = sum(1 for e in val_exps if e.group == g)
        n_te = sum(1 for e in test_exps if e.group == g)
        print(f"  group {g}: train={n_tr}, val={n_va}, test={n_te}")

    x_scaler = Scaler.fit(train_exps)
    print(f"x_scaler std mean = {x_scaler.std.mean():.4f}")
    # 与任务 2 一致：在全量（train+val+test）实验上 fit y_scaler
    # y 是稀疏中间目标，train-only 样本可能太少；任务 2 的 fit_scalers 也用全量
    all_exps_for_y = train_exps + val_exps + test_exps
    y_scaler = YScaler.fit(all_exps_for_y)
    print(f"y_scaler means={y_scaler.means.round(3).tolist()}, "
          f"stds={y_scaler.stds.round(3).tolist()}")
    # 落盘 y_scaler（与 x_scaler 落盘风格一致）
    np.savez(os.path.join(args.out_dir, "scalers_y.npz"),
             y_mean=y_scaler.means, y_scale=y_scaler.stds)
    print(f"y_scaler 落盘: {os.path.join(args.out_dir, 'scalers_y.npz')}")

    train_samples = sample_windows(train_exps, rng_seed=args.seed)
    val_samples = sample_windows(val_exps, rng_seed=args.seed + 1)
    test_samples = sample_windows(test_exps, rng_seed=args.seed + 2)

    def attach(samples, subset):
        m = {i: int(e.group) - 1 for i, e in enumerate(subset)}
        for s in samples:
            s.group = m[s.exp_id]
    attach(train_samples, train_exps)
    attach(val_samples, val_exps)
    attach(test_samples, test_exps)
    print(f"样本: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    train_ds = WindowXGDataset(train_samples, x_scaler, y_scaler)
    val_ds = WindowXGDataset(val_samples, x_scaler, y_scaler)
    test_ds = WindowXGDataset(test_samples, x_scaler, y_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate_xg, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pad_collate_xg, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=pad_collate_xg, num_workers=0)

    # ──── 2. 训练 / 加载 6 个模型 ────
    results = {}
    results_train = {}
    results_tf = {}

    all_model_configs = [
        # (display_name, backbone, mode, ckpt_filename)
        ("LSTM shared",      "lstm",    "shared",      "forecaster_lstm_shared.pt"),
        ("LSTM group_head",  "lstm",    "group_head",  "forecaster_lstm_group_head.pt"),
        ("LSTM independent", "lstm",    "independent", "forecaster_lstm_independent.pt"),
        ("PathInt shared",      "pathint", "shared",      "forecaster_pathint_shared.pt"),
        ("PathInt group_head",  "pathint", "group_head",  "forecaster_pathint_group_head.pt"),
        ("PathInt independent", "pathint", "independent", "forecaster_pathint_independent.pt"),
    ]

    # 按 --modes 参数过滤
    selected_modes = set(args.modes.split(","))
    model_configs = [mc for mc in all_model_configs if mc[2] in selected_modes]
    if not model_configs:
        print(f"[错误] --modes 过滤后无模型可选（--modes={args.modes}）")
        sys.exit(1)
    print(f"模式: {args.modes} → {len(model_configs)} 个模型: "
          f"{[m[0] for m in model_configs]}")

    for display_name, backbone, mode, ckpt_name in model_configs:
        ckpt_path = os.path.join(args.out_dir, ckpt_name)
        print(f"\n{'=' * 50}")
        print(f"[{display_name}]  backbone={backbone}, mode={mode}")

        if args.skip_train and os.path.exists(ckpt_path):
            print(f"  跳过训练，加载已有 ckpt: {ckpt_path}")
        elif not args.skip_train:
            model = build_model(backbone, mode, hidden=args.hidden,
                                dim_state=args.dim_state).to(device)
            model_log_dir = os.path.join(args.log_dir, f"{backbone}_{mode}")
            # per-backbone LR
            if args.lr is not None:
                lr = args.lr
            elif backbone == "lstm":
                lr = args.lr_lstm
            else:
                lr = args.lr_pathint
            n_params = train_model(model, train_loader, val_loader, device, args,
                                   mode, ckpt_path, x_scaler, y_scaler, lr,
                                   log_dir=model_log_dir)

        # 加载最佳 ckpt 并评估
        model = build_model(backbone, mode, hidden=args.hidden,
                            dim_state=args.dim_state).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate_multigroup(model, test_loader, device, mode, x_scaler,
                                           return_by_group=True)

        n_params = sum(p.numel() for p in model.parameters())
        results[display_name] = {
            "backbone": backbone,
            "mode": mode,
            "n_params": n_params,
            "rmse_x_norm": test_metrics["rmse_x_norm"],
            "rmse_x_orig": test_metrics["rmse_x_orig"],
            "by_group": test_metrics["by_group"],
        }
        print(f"  整体 RMSE(x) 原始空间 = {test_metrics['rmse_x_orig']:.4f}")
        for g in range(5):
            bg = test_metrics["by_group"].get(g, {})
            if bg.get("rmse_x_orig") is not None:
                print(f"    group {g+1}: RMSE={bg['rmse_x_orig']:.4f}")

        # 训练集评测（过拟合检查）与 teacher-forcing 评测（一步外推）
        train_metrics = evaluate_multigroup(model, train_loader, device, mode, x_scaler,
                                            return_by_group=True)
        tf_metrics = evaluate_multigroup(model, test_loader, device, mode, x_scaler,
                                         return_by_group=True, use_tf=True)
        results_train[display_name] = {
            "backbone": backbone, "mode": mode, "n_params": n_params,
            "rmse_x_norm": train_metrics["rmse_x_norm"],
            "rmse_x_orig": train_metrics["rmse_x_orig"],
            "by_group": train_metrics["by_group"],
        }
        results_tf[display_name] = {
            "backbone": backbone, "mode": mode, "n_params": n_params,
            "rmse_x_norm": tf_metrics["rmse_x_norm"],
            "rmse_x_orig": tf_metrics["rmse_x_orig"],
            "by_group": tf_metrics["by_group"],
        }
        print(f"  训练集 RMSE(x) 原始空间 = {train_metrics['rmse_x_orig']:.4f} | "
              f"TF 一步外推 RMSE(x) = {tf_metrics['rmse_x_orig']:.4f}")

    # ──── 3. 保存 JSON ────
    _save_metrics_json(results, os.path.join(args.out_dir, "compare_metrics.json"))
    _save_metrics_json(results_train, os.path.join(args.out_dir, "compare_metrics_train.json"))
    _save_metrics_json(results_tf, os.path.join(args.out_dir, "compare_metrics_tf.json"))

    # ──── 4. 画图 ────
    model_labels = {
        "LSTM shared":      "LSTM\nshared",
        "LSTM group_head":  "LSTM\ngroup_head",
        "LSTM independent": "LSTM\nindependent",
        "PathInt shared":      "PathInt\nshared",
        "PathInt group_head":  "PathInt\ngroup_head",
        "PathInt independent": "PathInt\nindependent",
    }
    MODE_COLORS = {"shared": "#78909c", "group_head": "#42a5f5", "independent": "#ef5350"}

    # 分 backbone 画
    lstm_order = [n for n, d in results.items() if d["backbone"] == "lstm"]
    pi_order = [n for n, d in results.items() if d["backbone"] == "pathint"]
    if lstm_order:
        plot_comparison(results, lstm_order, model_labels,
                        [MODE_COLORS.get(results[n]["mode"], "#999") for n in lstm_order],
                        "LSTM Backbone — Per-Group RMSE (original space)",
                        os.path.join(args.out_dir, "compare_bars_lstm.png"))
    if pi_order:
        plot_comparison(results, pi_order, model_labels,
                        ["#78909c" if results[n]["mode"] == "shared" else "#66bb6a" for n in pi_order],
                        "PathInt Backbone — Per-Group RMSE (original space)",
                        os.path.join(args.out_dir, "compare_bars_pathint.png"))

    # 合并图：所有选中模型
    all_order = list(results.keys())
    all_colors = [
        {"LSTM shared": "#78909c", "LSTM group_head": "#42a5f5", "LSTM independent": "#ef5350",
         "PathInt shared": "#78909c", "PathInt group_head": "#66bb6a", "PathInt independent": "#ff9800"}.get(n, "#999")
        for n in all_order
    ]
    if len(all_order) > 1:
        plot_comparison(results, all_order, model_labels, all_colors,
                        "All Models — Per-Group RMSE (original space)",
                        os.path.join(args.out_dir, "compare_bars_all.png"))

    # 训练集合并图（过拟合检查）
    train_order = list(results_train.keys())
    if len(train_order) > 1:
        train_colors = [
            {"LSTM shared": "#78909c", "LSTM group_head": "#42a5f5", "LSTM independent": "#ef5350",
             "PathInt shared": "#78909c", "PathInt group_head": "#66bb6a", "PathInt independent": "#ff9800"}.get(n, "#999")
            for n in train_order
        ]
        plot_comparison(results_train, train_order, model_labels, train_colors,
                        "All Models — Per-Group RMSE, TRAIN (original space)",
                        os.path.join(args.out_dir, "compare_bars_all_train.png"))

    # TF vs AR 整体 RMSE 对比
    tf_order = [n for n in all_order if n in results_tf]
    if tf_order:
        plot_tf_vs_ar(results, results_tf, tf_order, model_labels,
                      os.path.join(args.out_dir, "compare_bars_tf_vs_ar.png"))

    # ──── 5. 汇总表格 ────
    print(f"\n{'=' * 85}")
    header = f"{'Model':<22} {'Backbone':<8} {'Mode':<12} {'Params':>10} {'RMSE':>10}"
    for g in range(5):
        header += f" {'G'+str(g+1):>8}"
    print(header)
    print("-" * 85)

    all_order = lstm_order + pi_order
    for m_name in all_order:
        d = results.get(m_name)
        if not d:
            continue
        b = d.get("backbone", "?")
        m = d.get("mode", "?")
        print(f"{m_name:<22} {b:<8} {m:<12} {d['n_params']:>10,} {d['rmse_x_orig']:>10.1f}",
              end="")
        for g_str in ["1", "2", "3", "4", "5"]:
            g_int = int(g_str) - 1
            bg = d["by_group"].get(g_int, d["by_group"].get(g_str, {}))
            v = bg.get("rmse_x_orig") if isinstance(bg, dict) else None
            if v is not None:
                print(f" {v:>8.1f}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()
    print(f"{'=' * 85}")

    # ──── 6. 对比分析 ────
    print("\n=== 自回归 AR vs Teacher Forcing TF（整体 RMSE，原始空间）===")
    print(f"{'Model':<22} {'AR':>10} {'TF':>10} {'漂移(AR-TF)':>12} {'漂移占比':>10}")
    for m_name in all_order:
        ar = results.get(m_name, {}).get("rmse_x_orig")
        tf = results_tf.get(m_name, {}).get("rmse_x_orig")
        if ar is None or tf is None:
            continue
        drift = ar - tf
        ratio = drift / ar * 100 if ar > 0 else 0.0
        print(f"{m_name:<22} {ar:>10.1f} {tf:>10.1f} {drift:>12.1f} {ratio:>9.1f}%")

    print("\n=== group_head vs shared 提升幅度 ===")
    for backbone in ["LSTM", "PathInt"]:
        gh_name = f"{backbone} group_head"
        sh_name = f"{backbone} shared"
        if gh_name in results and sh_name in results:
            gh_rmse = results[gh_name]["rmse_x_orig"]
            sh_rmse = results[sh_name]["rmse_x_orig"]
            improvement = (sh_rmse - gh_rmse) / sh_rmse * 100
            print(f"  {backbone}: shared={sh_rmse:.1f} -> group_head={gh_rmse:.1f}  "
                  f"提升 {improvement:+.1f}%")

    print("\n=== group_head vs independent（数据效率）===")
    for backbone in ["LSTM", "PathInt"]:
        gh_name = f"{backbone} group_head"
        ind_name = f"{backbone} independent"
        if gh_name in results and ind_name in results:
            gh_rmse = results[gh_name]["rmse_x_orig"]
            ind_rmse = results[ind_name]["rmse_x_orig"]
            gh_params = results[gh_name]["n_params"]
            ind_params = results[ind_name]["n_params"]
            diff = (gh_rmse - ind_rmse) / ind_rmse * 100
            print(f"  {backbone}: group_head={gh_rmse:.1f} ({gh_params:,}p) vs "
                  f"independent={ind_rmse:.1f} ({ind_params:,}p)  "
                  f"差异 {diff:+.1f}%")

    print("\n=== LSTM vs PathInt 同模式对比 ===")
    for mode in sorted(selected_modes):
        lstm_name = f"LSTM {mode}"
        pi_name = f"PathInt {mode}"
        if lstm_name in results and pi_name in results:
            lstm_rmse = results[lstm_name]["rmse_x_orig"]
            pi_rmse = results[pi_name]["rmse_x_orig"]
            diff = (pi_rmse - lstm_rmse) / lstm_rmse * 100
            winner = "PathInt 优" if diff < 0 else "LSTM 优"
            print(f"  {mode}: LSTM={lstm_rmse:.1f}  PathInt={pi_rmse:.1f}  "
                  f"({diff:+.1f}%, {winner})")


if __name__ == "__main__":
    main()
