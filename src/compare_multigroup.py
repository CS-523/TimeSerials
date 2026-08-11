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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, sample_windows, Scaler, X_COLS
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
def train_one_epoch(model, loader, opt, device, mode: str):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        out_lens = batch["out_lens"].to(device)
        T_out = int(out_lens.max().item())
        if mode == "shared":
            out = model(x_in, T_out=T_out)
        else:
            group_ids = batch["group_ids"].to(device)
            out = model(x_in, group_ids, T_out=T_out)
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
                mode: str, ckpt_path: str, x_scaler):
    """统一的训练循环（支持 shared / group_head / independent）。"""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, mode)
        sched.step()
        val_m = evaluate_multigroup(model, val_loader, device, mode, x_scaler,
                                    return_by_group=False)
        elapsed = time.time() - t0
        print(f"  Epoch {ep+1:02d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"val_rmse={val_m['rmse_x_norm']:.4f} | {elapsed:.1f}s")
        if val_m["rmse_x_norm"] < best_val:
            best_val = val_m["rmse_x_norm"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
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
        ax.set_ylabel("RMSE (原始空间)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=6)

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
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dim-state", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练，仅用已有 ckpt 评估和画图")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, "src", "model_out")
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

    train_ds = WindowXGDataset(train_samples, x_scaler)
    val_ds = WindowXGDataset(val_samples, x_scaler)
    test_ds = WindowXGDataset(test_samples, x_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate_xg, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pad_collate_xg, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=pad_collate_xg, num_workers=0)

    # ──── 2. 训练 / 加载 6 个模型 ────
    results = {}

    model_configs = [
        # (display_name, backbone, mode, ckpt_filename)
        ("LSTM shared",      "lstm",    "shared",      "forecaster_lstm_shared.pt"),
        ("LSTM group_head",  "lstm",    "group_head",  "forecaster_lstm_group_head.pt"),
        ("LSTM independent", "lstm",    "independent", "forecaster_lstm_independent.pt"),
        ("PathInt shared",      "pathint", "shared",      "forecaster_pathint_shared.pt"),
        ("PathInt group_head",  "pathint", "group_head",  "forecaster_pathint_group_head.pt"),
        ("PathInt independent", "pathint", "independent", "forecaster_pathint_independent.pt"),
    ]

    for display_name, backbone, mode, ckpt_name in model_configs:
        ckpt_path = os.path.join(args.out_dir, ckpt_name)
        print(f"\n{'=' * 50}")
        print(f"[{display_name}]  backbone={backbone}, mode={mode}")

        if args.skip_train and os.path.exists(ckpt_path):
            print(f"  跳过训练，加载已有 ckpt: {ckpt_path}")
        elif not args.skip_train:
            model = build_model(backbone, mode, hidden=args.hidden,
                                dim_state=args.dim_state).to(device)
            n_params = train_model(model, train_loader, val_loader, device, args,
                                   mode, ckpt_path, x_scaler)

        # 加载最佳 ckpt 并评估
        model = build_model(backbone, mode, hidden=args.hidden,
                            dim_state=args.dim_state).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
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

    # ──── 3. 保存 JSON ────
    out_json = os.path.join(args.out_dir, "compare_metrics.json")
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
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n指标保存至: {out_json}")

    # ──── 4. 画图 ────
    model_labels = {
        "LSTM shared":      "LSTM\nshared",
        "LSTM group_head":  "LSTM\ngroup_head",
        "LSTM independent": "LSTM\nindependent",
        "PathInt shared":      "PathInt\nshared",
        "PathInt group_head":  "PathInt\ngroup_head",
        "PathInt independent": "PathInt\nindependent",
    }

    # 分 backbone 画两张
    lstm_order = ["LSTM shared", "LSTM group_head", "LSTM independent"]
    lstm_colors = ["#78909c", "#42a5f5", "#ef5350"]  # grey, blue, red
    plot_comparison(results, lstm_order, model_labels, lstm_colors,
                    "LSTM Backbone — Per-Group RMSE (original space)",
                    os.path.join(args.out_dir, "compare_bars_lstm.png"))

    pi_order = ["PathInt shared", "PathInt group_head", "PathInt independent"]
    pi_colors = ["#78909c", "#66bb6a", "#ef5350"]  # grey, green, red
    plot_comparison(results, pi_order, model_labels, pi_colors,
                    "PathInt Backbone — Per-Group RMSE (original space)",
                    os.path.join(args.out_dir, "compare_bars_pathint.png"))

    # 合并图：只画 group_head 和 independent，LSTM vs PathInt
    best_order = ["LSTM group_head", "PathInt group_head",
                  "LSTM independent", "PathInt independent"]
    best_colors = ["#42a5f5", "#66bb6a", "#ef5350", "#ff9800"]
    plot_comparison(results, best_order, model_labels, best_colors,
                    "Head-to-Head: group_head vs independent (LSTM vs PathInt)",
                    os.path.join(args.out_dir, "compare_bars_all.png"))

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
    for mode in ["shared", "group_head", "independent"]:
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
