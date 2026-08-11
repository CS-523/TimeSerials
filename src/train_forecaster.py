"""
训练流程：过程预测模型（仅预测 x1-x8）
=====================================
本版本只训练 x 的预测，不涉及 y1-y4 / Y 标签。

支持：
  - 任意起点、任意 in_len、任意 out_len 的训练样本
  - 自适应 batch 内 pad collate（变长样本 → batch 内最大长度）
  - 借鉴 pretrain_path_integrator.py 的 cosine LR schedule + grad clip
  - 三种训练模式：shared（全共享）、group_head（FiLM 适配头）、independent（逐组独立）
  - 支持 PathInt 和 LSTM 两种 backbone 与三种模式的任意组合

用法：
    # 默认：PathInt 全共享训练（向后兼容）
    python src/train_forecaster.py --epochs 30

    # PathInt + FiLM 适配头
    python src/train_forecaster.py --model pathint --mode group_head --epochs 30

    # PathInt + 逐组独立模型
    python src/train_forecaster.py --model pathint --mode independent --epochs 30

    # LSTM + FiLM 适配头
    python src/train_forecaster.py --model lstm --mode group_head --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_all, sample_windows, split_experiments,
    Scaler,
)
from model_forecaster import PathIntegratorForecaster
from model_lstm import LSTMForecaster
from model_multigroup import (
    LSTMForecasterFiLM, LSTMForecaster5Models,
    PathIntegratorForecasterFiLM, PathIntegratorForecaster5Models,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== 数据集（仅 x）=====================
class WindowXDataset(Dataset):
    """只含 x 的窗口数据集；y 标签完全不参与训练。"""
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
        }


def pad_collate_x(batch):
    """把变长 x 样本 pad 到 batch 内最大长度。"""
    max_in = max(b["in_len"] for b in batch)
    max_out = max(b["out_len"] for b in batch)
    B = len(batch)
    x_in = torch.zeros(B, max_in, 8)
    x_out = torch.zeros(B, max_out, 8)
    in_lens = torch.zeros(B, dtype=torch.long)
    out_lens = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        x_in[i, :b["in_len"]] = b["x_in"]
        x_out[i, :b["out_len"]] = b["x_out"]
        in_lens[i] = b["in_len"]
        out_lens[i] = b["out_len"]
    return dict(x_in=x_in, x_out=x_out, in_lens=in_lens, out_lens=out_lens)


# ===================== 多组工具（mode = group_head / independent 时使用）=====================
def split_experiments_groupwise(exps, ratios=(0.7, 0.1, 0.2), seed: int = 42):
    """对每个 group 单独做随机切分 train/val/test，保证每组在各集合都有样本。"""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for g in ["1", "2", "3", "4", "5"]:
        gs = [e for e in exps if e.group == g]
        if not gs:
            continue
        idx = np.arange(len(gs))
        rng.shuffle(idx)
        n_train = max(int(ratios[0] * len(gs)), 1)
        n_val = max(int(ratios[1] * len(gs)), 0)
        train += [gs[i] for i in idx[:n_train]]
        val += [gs[i] for i in idx[n_train:n_train + n_val]]
        test += [gs[i] for i in idx[n_train + n_val:]]
    return train, val, test


class WindowXGDataset(Dataset):
    """带 group_id 的窗口数据集。"""
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
            "group_id": int(s.group),
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


def attach_group(samples, exps_subset):
    """给 WindowSample 列表按 exp_id 挂载 group 属性（映射为 0..4）。"""
    m = {i: int(e.group) - 1 for i, e in enumerate(exps_subset)}
    for s in samples:
        s.group = m[s.exp_id]


# ===================== 训练 / 评估 =====================
def train_one_epoch(model, loader, opt, device, mode: str = "shared"):
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

        # 每个样本只算自己 out_len 范围内的 MSE（变长对齐）
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
def evaluate(model, loader, device, mode: str = "shared",
             x_scaler=None, return_per_dim: bool = False,
             return_by_group: bool = False):
    model.eval()
    sse_x = 0.0
    cnt_x = 0
    per_dim_sse = np.zeros(8, dtype=np.float64)
    per_dim_cnt = np.zeros(8, dtype=np.float64)
    x_preds, x_trues = [], []
    # 逐组累计
    by_group_sse = {g: 0.0 for g in range(5)}
    by_group_cnt = {g: 0 for g in range(5)}
    for batch in loader:
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        out_lens = batch["out_lens"].to(device)
        T_out = int(out_lens.max().item())
        if mode == "shared":
            pred = model(x_in, T_out=T_out)["pred_x"]
        else:
            group_ids = batch["group_ids"]
            pred = model(x_in, group_ids.to(device), T_out=T_out)["pred_x"]
        # 每个样本只算自己 out_len 范围内的 MSE
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            L_eff = min(L, T_out)
            if L_eff > 0:
                diff = pred[i, :L_eff] - x_out[i, :L_eff]
                d2 = (diff ** 2)
                sse_x += float(d2.sum().item())
                cnt_x += L_eff * 8
                for c in range(8):
                    per_dim_sse[c] += float(d2[:, c].sum().item())
                    per_dim_cnt[c] += L_eff
                if return_by_group:
                    g = int(batch["group_ids"][i].item()) if mode != "shared" else 0
                    by_group_sse[g] += float(d2.sum().item())
                    by_group_cnt[g] += L_eff * 8
        x_preds.append(pred.cpu().numpy())
        x_trues.append(x_out.cpu().numpy())
    rmse_x = math.sqrt(sse_x / max(cnt_x, 1))
    out = {"rmse_x": rmse_x, "x_preds": x_preds, "x_trues": x_trues}
    if return_per_dim:
        per_dim_rmse = np.sqrt(per_dim_sse / np.maximum(per_dim_cnt, 1))
        out["rmse_x_per_dim"] = per_dim_rmse.tolist()
    if return_by_group and mode != "shared":
        by_group = {}
        for g in range(5):
            n_g = by_group_cnt[g]
            if n_g == 0:
                by_group[g] = {"n": 0, "rmse_x": None, "rmse_x_per_dim": None}
                continue
            rmse_g = math.sqrt(by_group_sse[g] / (n_g * 8))
            by_group[g] = {"n": int(n_g), "rmse_x": rmse_g}
        out["by_group"] = by_group
    return out


# ===================== 主函数 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out-dir", default="/kefu-nas/ybkong/time_serials-master/src/model_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim-state", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", choices=["pathint", "lstm"], default="pathint",
                    help="选择模型：pathint（PathIntegratorForecaster）或 lstm（LSTMForecaster）")
    ap.add_argument("--mode", choices=["shared", "group_head", "independent"], default="shared",
                    help="训练模式：shared=全共享(默认); group_head=共享骨干+逐组FiLM头; independent=逐组独立模型")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # 数据
    exps = load_all(args.base_dir)
    if len(exps) == 0:
        print(f"[ERROR] base-dir '{args.base_dir}' 下未找到任何数据，请检查路径是否正确")
        print(f"        示例: python src/train_forecaster.py --base-dir \"D:\\Code\\timeserials_claude\\time-serials-mac\" --epochs 30")
        sys.exit(1)
    if args.mode == "shared":
        train_exps, val_exps, test_exps = split_experiments(exps, seed=args.seed)
    else:
        train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=args.seed)
    print(f"[train] experiments: train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")
    if args.mode != "shared":
        for g in ["1", "2", "3", "4", "5"]:
            n_tr = sum(1 for e in train_exps if e.group == g)
            n_va = sum(1 for e in val_exps if e.group == g)
            n_te = sum(1 for e in test_exps if e.group == g)
            print(f"  group {g}: train={n_tr}, val={n_va}, test={n_te}")
    x_scaler = Scaler.fit(train_exps)
    scaler_suffix = f"_{args.mode}" if args.mode != "shared" else ""
    np.savez(os.path.join(args.out_dir, f"scalers{scaler_suffix}.npz"),
             x_mean=x_scaler.mean, x_std=x_scaler.std)
    print(f"[train] x_scaler.mean={x_scaler.mean}, std={x_scaler.std}")

    # 滑窗样本
    train_samples = sample_windows(train_exps, rng_seed=args.seed)
    val_samples = sample_windows(val_exps, rng_seed=args.seed + 1)
    test_samples = sample_windows(test_exps, rng_seed=args.seed + 2)
    print(f"[train] samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    if args.mode == "shared":
        train_ds = WindowXDataset(train_samples, x_scaler)
        val_ds = WindowXDataset(val_samples, x_scaler)
        test_ds = WindowXDataset(test_samples, x_scaler)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=pad_collate_x, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=pad_collate_x, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=pad_collate_x, num_workers=0)
    else:
        attach_group(train_samples, train_exps)
        attach_group(val_samples, val_exps)
        attach_group(test_samples, test_exps)
        train_ds = WindowXGDataset(train_samples, x_scaler)
        val_ds = WindowXGDataset(val_samples, x_scaler)
        test_ds = WindowXGDataset(test_samples, x_scaler)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=pad_collate_xg, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=pad_collate_xg, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=pad_collate_xg, num_workers=0)

    # 模型
    if args.model == "lstm":
        if args.mode == "shared":
            model = LSTMForecaster(dim_x=8, hidden=args.hidden, num_layers=2, dropout=0.1).to(device)
        elif args.mode == "group_head":
            model = LSTMForecasterFiLM(n_groups=5, dim_x=8, hidden=args.hidden,
                                       num_layers=2, dropout=0.1).to(device)
        else:  # independent
            model = LSTMForecaster5Models(n_groups=5, dim_x=8, hidden=args.hidden,
                                           num_layers=2, dropout=0.1).to(device)
    else:  # pathint
        if args.mode == "shared":
            model = PathIntegratorForecaster(dim_x=8, dim_state=args.dim_state, hidden=args.hidden).to(device)
        elif args.mode == "group_head":
            model = PathIntegratorForecasterFiLM(n_groups=5, dim_x=8,
                                                  dim_state=args.dim_state, hidden=args.hidden).to(device)
        else:  # independent
            model = PathIntegratorForecaster5Models(n_groups=5, dim_x=8,
                                                     dim_state=args.dim_state, hidden=args.hidden).to(device)
    print(f"[train] 模型: {args.model}, 模式: {args.mode}, 参数量: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, args.mode)
        sched.step()
        val_metrics = evaluate(model, val_loader, device, mode=args.mode,
                               return_per_dim=True)
        elapsed = time.time() - t0
        per_dim_str = " ".join(f"x{i+1}={val_metrics['rmse_x_per_dim'][i]:.3f}"
                                for i in range(8))
        print(f"Epoch {ep+1:02d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"val_rmse_x(norm)={val_metrics['rmse_x']:.4f} | {elapsed:.1f}s")
        print(f"          per-dim: {per_dim_str}")
        if val_metrics["rmse_x"] < best_val:
            best_val = val_metrics["rmse_x"]
            if args.mode == "shared":
                ckpt_name = "forecaster_best.pt"
                ckpt_model = f"forecaster_{args.model}.pt"
            else:
                ckpt_name = f"forecaster_{args.model}_{args.mode}_best.pt"
                ckpt_model = ckpt_name
            torch.save({
                "model": model.state_dict(),
                "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std},
                "args": vars(args),
            }, os.path.join(args.out_dir, ckpt_name))
            if ckpt_model != ckpt_name:
                torch.save({
                    "model": model.state_dict(),
                    "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std},
                    "args": vars(args),
                }, os.path.join(args.out_dir, ckpt_model))
            print(f"  -> saved best (rmse_x={best_val:.4f})")

    # 测试
    if args.mode == "shared":
        ckpt_path = os.path.join(args.out_dir, "forecaster_best.pt")
    else:
        ckpt_path = os.path.join(args.out_dir, f"forecaster_{args.model}_{args.mode}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, mode=args.mode,
                            x_scaler=x_scaler, return_per_dim=True,
                            return_by_group=(args.mode != "shared"))

    # 报告：分维度 RMSE（标准化空间 → 原始空间）
    per_dim_norm = np.array(test_metrics["rmse_x_per_dim"])
    per_dim_orig = per_dim_norm * x_scaler.std
    test_metrics["rmse_x_per_dim_orig"] = per_dim_orig.tolist()
    test_metrics["rmse_x_orig_mean"] = float(per_dim_orig.mean())
    print(f"\n[test] 整体 RMSE(x) 标准化空间 = {test_metrics['rmse_x']:.4f}")
    print(f"[test] 整体 RMSE(x) 原始空间均值 = {test_metrics['rmse_x_orig_mean']:.4f}")
    print("[test] 分维度 RMSE(x)（标准化空间 / 原始空间）：")
    for i, name in enumerate(["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]):
        print(f"   {name}: norm={per_dim_norm[i]:.4f}  orig={per_dim_orig[i]:.2f}")

    # 逐组报告
    if args.mode != "shared" and "by_group" in test_metrics:
        print("\n逐 group RMSE(x)（原始空间）：")
        for g in range(5):
            b = test_metrics["by_group"][g]
            if b["n"] == 0:
                print(f"  group {g+1}: n=0 (test 集合无样本)")
            else:
                print(f"  group {g+1}: n={b['n']:4d}, rmse_x(orig)={b['rmse_x']:.4f}")

    # 持久化测试结果
    import json
    metrics_file = f"test_metrics{scaler_suffix}.json"
    metrics_to_save = {
        "model": args.model,
        "mode": args.mode,
        "rmse_x": float(test_metrics["rmse_x"]),
        "rmse_x_orig_mean": float(test_metrics["rmse_x_orig_mean"]),
        "rmse_x_per_dim": [float(v) for v in test_metrics["rmse_x_per_dim"]],
        "rmse_x_per_dim_orig": [float(v) for v in test_metrics["rmse_x_per_dim_orig"]],
    }
    if args.mode != "shared" and "by_group" in test_metrics:
        metrics_to_save["by_group"] = {
            str(g + 1): test_metrics["by_group"][g]
            for g in range(5)
        }
    with open(os.path.join(args.out_dir, metrics_file), "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    # 保存预测样例
    np.savez(os.path.join(args.out_dir, f"test_predictions{scaler_suffix}.npz"),
             x_preds=np.concatenate(test_metrics["x_preds"]),
             x_trues=np.concatenate(test_metrics["x_trues"]))


if __name__ == "__main__":
    main()