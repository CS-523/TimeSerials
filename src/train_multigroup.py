"""
多组预测模型训练入口
=========================

不修改 src/train_forecaster.py 与 src/model_lstm.py 任何代码。
新增 3 个训练模式：
  --mode shared       : 全共享 LSTMForecaster（5 组一起训，baseline）
  --mode group_head   : LSTMForecasterFiLM（共享骨干 + 组适配头，推荐）
  --mode independent  : LSTMForecaster5Models（5 个独立 LSTM，精度上界）

数据切分（关键设计）：
  现有 split_experiments 是按"实验 id"随机切 train/val/test。
  在 per_group 模式下，某个 group 的所有实验可能全分到 train/val/test 之一，
  会导致该 group 在某个集合里"没样本"，无法逐 group 评估。

  所以这里采用 group-wise split：
  1) 对每个 group g，把该 group 的所有实验随机切 train/val/test (0.7/0.1/0.2)；
  2) 再把"所有 group 的 train"合在一起组成"总 train"等。
  这样能保证每个 group 在三集合里都有样本。

评估口径：mode × group × dim 的 RMSE，全部在同一个 test 集合上。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    Experiment, WindowSample, X_COLS,
    load_all, sample_windows, split_experiments, Scaler,
)
from model_lstm import LSTMForecaster
from model_multigroup import LSTMForecasterFiLM, LSTMForecaster5Models


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== 1. group-wise split =====================
def split_experiments_groupwise(
    exps: List[Experiment], ratios=(0.7, 0.1, 0.2), seed: int = 42,
) -> Tuple[List[Experiment], List[Experiment], List[Experiment]]:
    """对每个 group 单独做随机切分 train/val/test (0.7/0.1/0.2)。"""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for g in ["1", "2", "3", "4", "5"]:
        gs = [e for e in exps if e.group == g]
        if not gs:
            continue
        idx = np.arange(len(gs))
        rng.shuffle(idx)
        n_train = int(ratios[0] * len(gs))
        n_val = int(ratios[1] * len(gs))
        # 至少各 1 个（防止某 group 实验太少）
        n_train = max(n_train, 1) if len(gs) >= 3 else max(n_train, 1)
        n_val = max(n_val, 0)
        n_test = max(len(gs) - n_train - n_val, 0)
        train += [gs[i] for i in idx[:n_train]]
        val   += [gs[i] for i in idx[n_train:n_train + n_val]]
        test  += [gs[i] for i in idx[n_train + n_val:]]
    return train, val, test


# ===================== 2. 数据集（带 group_id） =====================
class WindowXGDataset(Dataset):
    """带 group_id 的窗口数据集（pooled / film 模式使用）。"""
    def __init__(self, samples, x_scaler: Scaler):
        self.samples = samples
        self.xs = x_scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_in = self.xs.transform(s.x_in)
        x_out = self.xs.transform(s.x_out)
        return {
            "x_in": torch.from_numpy(x_in).float(),
            "x_out": torch.from_numpy(x_out).float(),
            "in_len": s.in_len,
            "out_len": s.out_len,
            "group_id": int(s.group),  # attach_group 已转为 0..4
        }


def pad_collate_xg(batch):
    """变长样本 pad，带 group_id。"""
    max_in = max(b["in_len"] for b in batch)
    max_out = max(b["out_len"] for b in batch)
    B = len(batch)
    x_in = torch.zeros(B, max_in, 8)
    x_out = torch.zeros(B, max_out, 8)
    in_lens = torch.zeros(B, dtype=torch.long)
    out_lens = torch.zeros(B, dtype=torch.long)
    group_ids = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        x_in[i, :b["in_len"]] = b["x_in"]
        x_out[i, :b["out_len"]] = b["x_out"]
        in_lens[i] = b["in_len"]
        out_lens[i] = b["out_len"]
        group_ids[i] = b["group_id"]
    return dict(x_in=x_in, x_out=x_out, in_lens=in_lens, out_lens=out_lens,
                group_ids=group_ids)


# ===================== 3. 训练 / 评估 =====================
def _slice_pred(out_pred: torch.Tensor, out_lens: torch.Tensor) -> torch.Tensor:
    """对每个样本只取 out_len 范围，pad 部分剔除。返回 list。"""
    T_out = out_pred.size(1)
    out = []
    for i in range(out_pred.size(0)):
        L = int(out_lens[i].item())
        L_eff = min(L, T_out)
        out.append(out_pred[i, :L_eff])
    return out


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
        # 计算每个样本的 MSE（变长对齐）
        x_loss = 0.0
        cnt = 0
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            L_eff = min(L, T_out)
            if L_eff > 0:
                x_loss = x_loss + F.mse_loss(out["pred_x"][i, :L_eff], x_out[i, :L_eff])
                cnt += 1
        x_loss = x_loss / max(cnt, 1)
        loss = x_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, mode: str,
             x_scaler: Scaler, return_by_group: bool = True):
    """
    整体 RMSE + per-group RMSE。
    """
    model.eval()
    # 全局累计
    sse_x = 0.0
    cnt_x = 0
    per_dim_sse = np.zeros(8, dtype=np.float64)
    per_dim_cnt = np.zeros(8, dtype=np.float64)
    # per-group 累计
    by_group_sse = {g: 0.0 for g in range(5)}
    by_group_cnt = {g: 0 for g in range(5)}
    by_group_dim_sse = {g: np.zeros(8, dtype=np.float64) for g in range(5)}
    by_group_dim_cnt = {g: np.zeros(8, dtype=np.float64) for g in range(5)}

    for batch in loader:
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        out_lens = batch["out_lens"].to(device)
        group_ids = batch["group_ids"]
        T_out = int(out_lens.max().item())
        if mode == "shared":
            out = model(x_in, T_out=T_out)
        else:
            out = model(x_in, group_ids.to(device), T_out=T_out)

        pred = out["pred_x"]
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            L_eff = min(L, T_out)
            if L_eff > 0:
                g = int(group_ids[i].item())
                diff = pred[i, :L_eff] - x_out[i, :L_eff]
                d2 = (diff ** 2)
                sse_x += float(d2.sum().item())
                cnt_x += L_eff * 8
                for c in range(8):
                    per_dim_sse[c] += float(d2[:, c].sum().item())
                    per_dim_cnt[c] += L_eff
                by_group_sse[g] += float(d2.sum().item())
                by_group_cnt[g] += L_eff * 8
                for c in range(8):
                    by_group_dim_sse[g][c] += float(d2[:, c].sum().item())
                    by_group_dim_cnt[g][c] += L_eff

    rmse_x = math.sqrt(sse_x / max(cnt_x, 1))
    per_dim_rmse = np.sqrt(per_dim_sse / np.maximum(per_dim_cnt, 1))
    out = {
        "rmse_x_norm": rmse_x,
        "rmse_x_orig": rmse_x * float(x_scaler.std.mean()),
        "rmse_x_per_dim_norm": per_dim_rmse.tolist(),
        "rmse_x_per_dim_orig": (per_dim_rmse * x_scaler.std).tolist(),
        "n": int(cnt_x / 8),
    }
    if return_by_group:
        by_group = {}
        for g in range(5):
            n_g = by_group_cnt[g]
            if n_g == 0:
                by_group[g] = {"n": 0, "rmse_x_norm": None, "rmse_x_per_dim_norm": None,
                                "rmse_x_per_dim_orig": None}
                continue
            rmse_g = math.sqrt(by_group_sse[g] / (n_g * 8))
            d_rmse = np.sqrt(by_group_dim_sse[g] / np.maximum(by_group_dim_cnt[g], 1))
            by_group[g] = {
                "n": int(n_g),
                "rmse_x_norm": rmse_g,
                "rmse_x_orig": rmse_g * float(x_scaler.std.mean()),
                "rmse_x_per_dim_norm": d_rmse.tolist(),
                "rmse_x_per_dim_orig": (d_rmse * x_scaler.std).tolist(),
            }
        out["by_group"] = by_group
    return out


# ===================== 4. 模型构建 =====================
def build_model(mode: str, dim_x: int, hidden: int, num_layers: int,
                dropout: float, bidirectional: bool, n_groups: int = 5):
    if mode == "shared":
        return LSTMForecaster(dim_x=dim_x, hidden=hidden, num_layers=num_layers,
                                dropout=dropout, bidirectional=bidirectional)
    if mode == "group_head":
        return LSTMForecasterFiLM(n_groups=n_groups, dim_x=dim_x, hidden=hidden,
                                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    if mode == "independent":
        return LSTMForecaster5Models(n_groups=n_groups, dim_x=dim_x, hidden=hidden,
                                      num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    raise ValueError(f"unknown mode: {mode}")


# ===================== 5. 主函数 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/kefu-nas/ybkong/time_serials-master")
    ap.add_argument("--out-dir", default="/kefu-nas/ybkong/time_serials-master/src/model_out")
    ap.add_argument("--mode", choices=["shared", "group_head", "independent"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # 数据
    exps = load_all(args.base_dir)
    train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=args.seed)
    print(f"[{args.mode}] experiments: train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")
    for g in ["1", "2", "3", "4", "5"]:
        n_tr = sum(1 for e in train_exps if e.group == g)
        n_va = sum(1 for e in val_exps if e.group == g)
        n_te = sum(1 for e in test_exps if e.group == g)
        print(f"  group {g}: train={n_tr}, val={n_va}, test={n_te}")

    x_scaler = Scaler.fit(train_exps)
    np.savez(os.path.join(args.out_dir, f"scalers_{args.mode}.npz"),
             x_mean=x_scaler.mean, x_std=x_scaler.std)
    print(f"[{args.mode}] x_scaler mean={x_scaler.mean}, std={x_scaler.std}")

    # 滑窗样本
    # 把 Experiment.group 映射到 exp_id（在 sample_windows 时记录的），
    # 因为 WindowSample 不带 group，我们从原始 exps 取。
    exp_group = {i: e.group for i, e in enumerate(exps)}

    train_samples = sample_windows(train_exps, rng_seed=args.seed)
    val_samples   = sample_windows(val_exps, rng_seed=args.seed + 1)
    test_samples  = sample_windows(test_exps, rng_seed=args.seed + 2)
    # 关键修正：sample_windows 的 exp_id 是**整个 exps 列表**的索引，
    # 但 train_samples/val_samples/test_samples 是基于 train_exps/val_exps/test_exps 的子集。
    # 这里 sample_windows 是单独传入 train_exps/val_exps/test_exps 的，所以 exp_id 必然是 0..len(train_exps)-1。
    # 但安全起见我们仍然按"传入 exps 列表的索引"理解。
    def attach_group(samples, exps_subset):
        # 重建 exp_id → group 映射：s.exp_id 是 exps_subset 内的索引
        m = {i: int(e.group) - 1 for i, e in enumerate(exps_subset)}
        for s in samples:
            s.group = m[s.exp_id]  # 给对象挂一个属性

    attach_group(train_samples, train_exps)
    attach_group(val_samples, val_exps)
    attach_group(test_samples, test_exps)
    print(f"[{args.mode}] samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    train_ds = WindowXGDataset(train_samples, x_scaler)
    val_ds   = WindowXGDataset(val_samples, x_scaler)
    test_ds  = WindowXGDataset(test_samples, x_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                collate_fn=pad_collate_xg, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=pad_collate_xg, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=pad_collate_xg, num_workers=0)

    # 模型
    model = build_model(args.mode, dim_x=8, hidden=args.hidden,
                         num_layers=args.num_layers, dropout=args.dropout,
                         bidirectional=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.mode}] 模型参数量: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, args.mode)
        sched.step()
        val_metrics = evaluate(model, val_loader, device, args.mode, x_scaler,
                                 return_by_group=False)
        elapsed = time.time() - t0
        print(f"Epoch {ep+1:02d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"val_rmse_x(norm)={val_metrics['rmse_x_norm']:.4f} | {elapsed:.1f}s")
        if val_metrics["rmse_x_norm"] < best_val:
            best_val = val_metrics["rmse_x_norm"]
            torch.save({
                "model": model.state_dict(),
                "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std},
                "args": vars(args),
            }, os.path.join(args.out_dir, f"forecaster_{args.mode}_best.pt"))
            print(f"  -> saved best (rmse_x_norm={best_val:.4f})")

    # 测试
    ckpt = torch.load(os.path.join(args.out_dir, f"forecaster_{args.mode}_best.pt"),
                       map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, args.mode, x_scaler,
                              return_by_group=True)

    print(f"\n[{args.mode}] === TEST ===")
    print(f"整体 RMSE(x) 标准化空间 = {test_metrics['rmse_x_norm']:.4f}")
    print(f"整体 RMSE(x) 原始空间均值 = {test_metrics['rmse_x_orig']:.4f}")
    print("分维度 RMSE(x)（标准化空间 / 原始空间）：")
    for i, name in enumerate(X_COLS):
        print(f"   {name}: norm={test_metrics['rmse_x_per_dim_norm'][i]:.4f}  "
              f"orig={test_metrics['rmse_x_per_dim_orig'][i]:.2f}")
    print("\n逐 group RMSE(x)（原始空间）：")
    for g in range(5):
        b = test_metrics["by_group"][g]
        if b["n"] == 0:
            print(f"  group {g+1}: n=0 (test 集合无样本)")
            continue
        per_dim_str = " ".join(f"{X_COLS[i]}={b['rmse_x_per_dim_orig'][i]:.2f}"
                                for i in range(8))
        print(f"  group {g+1}: n={b['n']:4d}, rmse_x(orig)={b['rmse_x_orig']:.4f}  | {per_dim_str}")

    # 持久化（仅保留浮点字段）
    out = {
        "mode": args.mode,
        "n_params": n_params,
        "rmse_x_norm": test_metrics["rmse_x_norm"],
        "rmse_x_orig": test_metrics["rmse_x_orig"],
        "rmse_x_per_dim_norm": test_metrics["rmse_x_per_dim_norm"],
        "rmse_x_per_dim_orig": test_metrics["rmse_x_per_dim_orig"],
        "by_group": {str(g+1): {k: v for k, v in test_metrics["by_group"][g].items()
                                  if k in ("n", "rmse_x_norm", "rmse_x_orig")}
                     for g in range(5)},
    }
    out_path = os.path.join(args.out_dir, f"test_metrics_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.mode}] saved → {out_path}")


if __name__ == "__main__":
    main()
