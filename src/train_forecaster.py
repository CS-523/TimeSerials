"""
训练流程：过程预测模型
=====================
支持：
  - 任意起点、任意 in_len、任意 out_len 的训练样本
  - 借鉴 pretrain_path_integrator.py 的 TBPTT（截断反向传播）：
    对于过长的输入序列，按 window_steps 切窗口做前向+反传+状态 detach
  - 主任务：x 预测 MSE
  - 辅助任务：y4 预测 MSE（掩码掉 NaN）
  - 终值任务：Y 预测 MSE（每个实验一个 Y）

用法：
    python src/train_forecaster.py --epochs 30
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
    Scaler, YScaler,
)
from model_forecaster import PathIntegratorForecaster, masked_mse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== 数据集 =====================
class WindowDatasetV2(Dataset):
    """V2：把标准化器存为成员，collate 时直接做归一化。"""
    def __init__(self, samples, x_scaler, y_scaler):
        self.samples = samples
        self.xs = x_scaler
        self.ys = y_scaler
        self.y_mean = torch.from_numpy(y_scaler.means).float()
        self.y_std = torch.from_numpy(y_scaler.stds).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_in = self.xs.transform(s.x_in)
        y_in_raw = s.y_in.copy()
        y_in_mask = (~np.isnan(y_in_raw)).astype(np.float32)
        y_in_filled = np.nan_to_num(y_in_raw, nan=0.0)
        y_in_norm = (y_in_filled - self.ys.means) / self.ys.stds
        y_in = y_in_norm * y_in_mask
        x_out = self.xs.transform(s.x_out)
        y_out_raw = s.y_out.copy()
        y_out_mask = (~np.isnan(y_out_raw)).astype(np.float32)
        y_out_filled = np.nan_to_num(y_out_raw, nan=0.0)
        y_out_norm = (y_out_filled - self.ys.means) / self.ys.stds
        return {
            "x_in": torch.from_numpy(x_in).float(),
            "y_in": torch.from_numpy(y_in).float(),
            "y_in_mask": torch.from_numpy(y_in_mask).float(),
            "x_out": torch.from_numpy(x_out).float(),
            "y_out": torch.from_numpy(y_out_norm).float(),
            "y_out_mask": torch.from_numpy(y_out_mask).float(),
            "Y": torch.tensor(s.Y if s.Y is not None else float("nan")),
            "Y_mask": torch.tensor(1.0 if s.Y is not None else 0.0),
            "in_len": s.in_len,
            "out_len": s.out_len,
        }


def pad_collate_v2(batch):
    max_in = max(b["in_len"] for b in batch)
    max_out = max(b["out_len"] for b in batch)
    B = len(batch)
    x_in = torch.zeros(B, max_in, 8)
    y_in = torch.zeros(B, max_in, 4)
    y_in_mask = torch.zeros(B, max_in, 4)
    x_out = torch.zeros(B, max_out, 8)
    y_out = torch.zeros(B, max_out, 4)
    y_out_mask = torch.zeros(B, max_out, 4)
    Y = torch.zeros(B, 1)
    Y_mask = torch.zeros(B, 1)
    in_lens = torch.zeros(B, dtype=torch.long)
    out_lens = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        x_in[i, :b["in_len"]] = b["x_in"]
        y_in[i, :b["in_len"]] = b["y_in"]
        y_in_mask[i, :b["in_len"]] = b["y_in_mask"]
        x_out[i, :b["out_len"]] = b["x_out"]
        y_out[i, :b["out_len"]] = b["y_out"]
        y_out_mask[i, :b["out_len"]] = b["y_out_mask"]
        Y[i, 0] = b["Y"]
        Y_mask[i, 0] = b["Y_mask"]
        in_lens[i] = b["in_len"]
        out_lens[i] = b["out_len"]
    return dict(x_in=x_in, y_in=y_in, y_in_mask=y_in_mask, x_out=x_out, y_out=y_out,
                y_out_mask=y_out_mask, Y=Y, Y_mask=Y_mask, in_lens=in_lens, out_lens=out_lens)


# ===================== 训练 / 评估 =====================
def train_one_epoch(model, loader, opt, device, max_T_out=None, w_x=1.0, w_y=1.0, w_Y=0.5, y4_boost=3.0):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_in = batch["x_in"].to(device)
        y_in = batch["y_in"].to(device)
        x_out = batch["x_out"].to(device)
        y_out = batch["y_out"].to(device)
        y_out_mask = batch["y_out_mask"].to(device)
        Y = batch["Y"].to(device)
        Y_mask = batch["Y_mask"].to(device)
        out_lens = batch["out_lens"].to(device)
        # T_out 取 batch 内最大
        T_out = int(out_lens.max().item()) if max_T_out is None else min(int(out_lens.max().item()), max_T_out)
        out = model(x_in, y_in, T_out=T_out)
        # 各样本只取自己的 out_len 范围
        # x loss: 标准化空间 MSE
        x_loss = 0.0
        cnt = 0
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            if L > 0:
                # 超出 T_out 的部分用 0 padding 算 0 loss
                L_eff = min(L, T_out)
                x_loss = x_loss + F.mse_loss(out["pred_x"][i, :L_eff], x_out[i, :L_eff])
                cnt += 1
        x_loss = x_loss / max(cnt, 1)
        # y4 loss：y4 是必选且稀疏，权重加大
        y_loss = masked_mse(out["pred_y"], y_out, y_out_mask)
        # y4 单独再加权一次（y_out_mask[:, :, 3] 控制）
        y4_loss = masked_mse(out["pred_y"][:, :, 3:4], y_out[:, :, 3:4], y_out_mask[:, :, 3:4])
        # Y loss
        Y_pred = out["pred_Y"]
        Y_loss = masked_mse(Y_pred, Y, Y_mask)
        loss = w_x * x_loss + w_y * y_loss + y4_boost * y4_loss + w_Y * Y_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, max_T_out=None):
    model.eval()
    sse_x = 0.0
    cnt_x = 0
    sse_y4 = 0.0
    cnt_y4 = 0
    sse_Y = 0.0
    cnt_Y = 0
    y_preds, y_trues, x_preds, x_trues = [], [], [], []
    for batch in loader:
        x_in = batch["x_in"].to(device)
        y_in = batch["y_in"].to(device)
        x_out = batch["x_out"].to(device)
        y_out = batch["y_out"].to(device)
        y_out_mask = batch["y_out_mask"].to(device)
        Y = batch["Y"].to(device)
        Y_mask = batch["Y_mask"].to(device)
        out_lens = batch["out_lens"].to(device)
        T_out = int(out_lens.max().item()) if max_T_out is None else min(int(out_lens.max().item()), max_T_out)
        out = model(x_in, y_in, T_out=T_out)
        for i in range(x_out.size(0)):
            L = int(out_lens[i].item())
            L_eff = min(L, T_out)
            if L_eff > 0:
                sse_x += float(((out["pred_x"][i, :L_eff] - x_out[i, :L_eff]) ** 2).sum().item())
                cnt_x += L_eff * 8
                # y4（索引 3）
                for t in range(L_eff):
                    if y_out_mask[i, t, 3] > 0.5:
                        sse_y4 += float((out["pred_y"][i, t, 3] - y_out[i, t, 3]) ** 2)
                        cnt_y4 += 1
        for i in range(Y.size(0)):
            if Y_mask[i] > 0.5:
                sse_Y += float((out["pred_Y"][i, 0] - Y[i, 0]) ** 2)
                cnt_Y += 1
        y_preds.append(out["pred_y"][:, :, 3].cpu().numpy())
        y_trues.append(y_out[:, :, 3].cpu().numpy())
        x_preds.append(out["pred_x"].cpu().numpy())
        x_trues.append(x_out.cpu().numpy())
    rmse_x = math.sqrt(sse_x / max(cnt_x, 1))
    rmse_y4 = math.sqrt(sse_y4 / max(cnt_y4, 1))
    rmse_Y = math.sqrt(sse_Y / max(cnt_Y, 1))
    return {"rmse_x": rmse_x, "rmse_y4": rmse_y4, "rmse_Y": rmse_Y,
            "y_preds": y_preds, "y_trues": y_trues,
            "x_preds": x_preds, "x_trues": x_trues}


# ===================== 主函数 =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/kefu-nas/ybkong/time_serials-master")
    ap.add_argument("--out-dir", default="/kefu-nas/ybkong/time_serials-master/src/model_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim-state", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-per-exp", type=int, default=8)
    ap.add_argument("--y4-boost", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # 数据
    exps = load_all(args.base_dir)
    train_exps, val_exps, test_exps = split_experiments(exps, seed=args.seed)
    print(f"[train] experiments: train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")
    x_scaler = Scaler.fit(train_exps)
    y_scaler = YScaler.fit(train_exps)
    np.savez(os.path.join(args.out_dir, "scalers.npz"),
             x_mean=x_scaler.mean, x_std=x_scaler.std,
             y_mean=y_scaler.means, y_std=y_scaler.stds)
    print(f"[train] x_scaler.mean={x_scaler.mean}, std={x_scaler.std}")
    print(f"[train] y_scaler.mean={y_scaler.means}, std={y_scaler.stds}")

    # 滑窗样本
    train_samples = sample_windows(train_exps, rng_seed=args.seed)
    val_samples = sample_windows(val_exps, rng_seed=args.seed + 1)
    test_samples = sample_windows(test_exps, rng_seed=args.seed + 2)
    print(f"[train] samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    train_ds = WindowDatasetV2(train_samples, x_scaler, y_scaler)
    val_ds = WindowDatasetV2(val_samples, x_scaler, y_scaler)
    test_ds = WindowDatasetV2(test_samples, x_scaler, y_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate_v2, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pad_collate_v2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=pad_collate_v2, num_workers=0)

    # 模型
    model = PathIntegratorForecaster(dim_x=8, dim_y=4, dim_state=args.dim_state, hidden=args.hidden).to(device)
    print(f"[train] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device, y4_boost=args.y4_boost)
        sched.step()
        val_metrics = evaluate(model, val_loader, device)
        # 计算"原始空间"的 RMSE
        elapsed = time.time() - t0
        print(f"Epoch {ep+1:02d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"val_rmse_x(norm)={val_metrics['rmse_x']:.4f} | val_rmse_y4(norm)={val_metrics['rmse_y4']:.4f} | "
              f"val_rmse_Y(norm)={val_metrics['rmse_Y']:.4f} | {elapsed:.1f}s")
        # 保存最优
        if val_metrics["rmse_x"] + val_metrics["rmse_y4"] < best_val:
            best_val = val_metrics["rmse_x"] + val_metrics["rmse_y4"]
            torch.save({
                "model": model.state_dict(),
                "x_scaler": {"mean": x_scaler.mean, "std": x_scaler.std},
                "y_scaler": {"mean": y_scaler.means, "std": y_scaler.stds},
                "args": vars(args),
            }, os.path.join(args.out_dir, "forecaster_best.pt"))
            print(f"  -> saved best (combined={best_val:.4f})")

    # 测试
    ckpt = torch.load(os.path.join(args.out_dir, "forecaster_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    # 反标准化空间下的 RMSE
    x_std = x_scaler.std
    y_std = y_scaler.stds  # (4,)
    test_metrics["rmse_x_orig"] = test_metrics["rmse_x"] * float(np.sqrt((x_std ** 2).mean()))
    test_metrics["rmse_y4_orig"] = test_metrics["rmse_y4"] * float(y_std[3])
    # Y 自身没有标准化器，给出原始尺度预测
    Y_all = np.array([e.Y for e in exps if e.Y is not None], dtype=np.float32)
    Y_std_val = float(Y_all.std() + 1e-6)
    test_metrics["rmse_Y_orig"] = test_metrics["rmse_Y"] * Y_std_val
    print(f"\n[test] rmse_x (orig) = {test_metrics['rmse_x_orig']:.4f}")
    print(f"[test] rmse_y4 (orig) = {test_metrics['rmse_y4_orig']:.4f}")
    print(f"[test] rmse_Y (orig)  = {test_metrics['rmse_Y_orig']:.4f}")
    # 持久化测试结果
    import json
    metrics_to_save = {k: float(v) for k, v in test_metrics.items()
                       if k in ("rmse_x", "rmse_y4", "rmse_Y", "rmse_x_orig", "rmse_y4_orig", "rmse_Y_orig")}
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    # 保存预测样例
    np.savez(os.path.join(args.out_dir, "test_predictions.npz"),
             y_preds=np.concatenate(test_metrics["y_preds"]),
             y_trues=np.concatenate(test_metrics["y_trues"]),
             x_preds=np.concatenate(test_metrics["x_preds"]),
             x_trues=np.concatenate(test_metrics["x_trues"]))


if __name__ == "__main__":
    main()