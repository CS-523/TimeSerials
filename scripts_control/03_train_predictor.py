"""Train the hybrid SS-NN predictor on the processed dataset.

Usage::

    python -m scripts_control.03_train_predictor \
        --data data/processed/train.npz \
        --test  data/processed/test.npz  \
        --epochs 200 --bs 16 --lr 1e-3 --horizon 32

Outputs (under ``checkpoints/`` and ``results/metrics/``):

* ``checkpoints/ss_nn_best.pt``  — best val-loss weights
* ``checkpoints/ss_nn_last.pt``  — final-epoch weights
* ``results/metrics/training_log.json``  — per-epoch losses
* ``results/metrics/test_metrics.json``   — per-variable MSE/MAE/R² on the test set
* ``results/predictions/test_predictions.npz``  — saved predictions
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src_control.config import get_config, resolve_paths
from src_control.models.state_space import n4sid
from src_control.models.state_space_nn import SS_NN_Hybrid, YHead, init_hybrid_from_n4sid
from src_control.preprocess import load_processed
from src_control.utils.metrics import per_variable_metrics
from src_control.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# N4SID init from training data
# --------------------------------------------------------------------------- #
def fit_n4sid_on_train(ds: Dict[str, np.ndarray], order: int = 16, n_lags: int = 8):
    """Fit a linear SS using the training set, average across samples."""
    X = ds["X"]  # (N, T, 8)
    Y = ds["Y"]  # (N, T, 4)
    Y_mask = ds["Y_mask"]
    N = X.shape[0]
    As, Bs, Cs, Ds = [], [], [], []
    for i in range(N):
        u = X[i]
        # Use only observed y entries; fill missing with NaN then linear-interpolate.
        y = Y[i].copy()
        for j in range(4):
            valid = Y_mask[i, :, j]
            col = y[:, j].copy()
            if not valid.all() and valid.any():
                idx = np.arange(len(col))
                col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
                y[:, j] = col
        try:
            m = n4sid(u, y, order=order, n_lags=n_lags)
            As.append(m.A); Bs.append(m.B); Cs.append(m.C); Ds.append(m.D)
        except Exception:
            continue
    if not As:
        raise RuntimeError("N4SID failed on all training samples")
    return (
        np.mean(np.stack(As), axis=0),
        np.mean(np.stack(Bs), axis=0),
        np.mean(np.stack(Cs), axis=0),
        np.mean(np.stack(Ds), axis=0),
    )


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def masked_mse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE on observed y entries (mask True)."""
    diff = (y_pred - y_true) ** 2
    diff = diff * mask.float()
    n = mask.float().sum().clamp_min(1.0)
    return diff.sum() / n


def make_loaders(
    train: Dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
    bs: int,
) -> Tuple[DataLoader, DataLoader]:
    N = train["X"].shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = int(round(N * val_ratio))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.tensor(train["X"][tr_idx], dtype=torch.float32)
    Y_tr = torch.tensor(train["Y"][tr_idx], dtype=torch.float32)
    M_tr = torch.tensor(train["Y_mask"][tr_idx], dtype=torch.float32)

    X_va = torch.tensor(train["X"][val_idx], dtype=torch.float32)
    Y_va = torch.tensor(train["Y"][val_idx], dtype=torch.float32)
    M_va = torch.tensor(train["Y_mask"][val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr, M_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, Y_va, M_va), batch_size=bs, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.npz")
    parser.add_argument("--test", default="data/processed/test.npz")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--scalers", default="data/processed/scalers.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--n-state", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--tf-decay", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = resolve_paths(get_config())
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    metrics_dir = Path(args.metrics_dir)
    pred_dir = Path(args.predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    train = load_processed(args.data)
    test = load_processed(args.test)
    print(f"  train X={train['X'].shape}, test X={test['X'].shape}")

    # Fit N4SID on training data for SS init
    print("Fitting N4SID baseline on training data…")
    A, B, C, D = fit_n4sid_on_train(train, order=args.n_state, n_lags=8)
    print(f"  N4SID recovered: A shape={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")

    # Build model
    device = cfg.DEVICE
    model = SS_NN_Hybrid(
        dim_u=8, dim_y=4, n_state=args.n_state, hidden=args.hidden, window=4
    )
    init_hybrid_from_n4sid(model, type("M", (), {"A": A, "B": B, "C": C, "D": D})())
    yhead = YHead(window=8)
    model = model.to(device)
    yhead = yhead.to(device)

    # Loaders
    train_loader, val_loader = make_loaders(
        train, val_ratio=args.val_ratio, seed=args.seed, bs=args.bs
    )

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(yhead.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)

    history = []
    best_val = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Teacher-forcing schedule: 1 → 0 over tf_decay epochs
        tf = max(0.0, 1.0 - epoch / max(1, args.tf_decay))
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for X, Y, M in train_loader:
            X = X.to(device); Y = Y.to(device); M = M.to(device)
            optim.zero_grad()
            # 50% teacher-forced, 50% pure AR — robust to both regimes
            use_tf = (torch.rand(1).item() < 0.5)
            if use_tf:
                y_pred = model(X, y_prev=Y, teacher_forcing=tf)
            else:
                # Pure AR: y_prev=None. Residual MLP sees [u, y_lin, y_lin].
                y_pred = model(X, y_prev=None, teacher_forcing=0.0)
            loss_pred = masked_mse(y_pred, Y, M)
            # Y head: regress last 8 timesteps of y → final Y
            y_tail = Y[:, -8:, :]
            with torch.no_grad():
                yp_tail = y_pred.detach()[:, -8:, :]
            y_pred_for_head = torch.cat([Y[:, :-8, :], yp_tail], dim=1)[:, -8:, :]
            y_final_true = (Y * M).sum(dim=(1, 2)) / M.sum(dim=(1, 2)).clamp_min(1.0)
            y_pred_final = yhead(y_pred_for_head)
            loss_y = (y_pred_final - y_final_true) ** 2
            loss_y = loss_y.mean()

            loss = loss_pred + 0.1 * loss_y
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batch += 1
        ep_loss /= max(1, n_batch)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for X, Y, M in val_loader:
                X = X.to(device); Y = Y.to(device); M = M.to(device)
                y_pred = model(X, y_prev=Y, teacher_forcing=1.0)
                val_loss += float(masked_mse(y_pred, Y, M).item())
                n_v += 1
        val_loss /= max(1, n_v)

        history.append({"epoch": epoch, "train": ep_loss, "val": val_loss, "tf": tf})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            torch.save({"model": model.state_dict(), "yhead": yhead.state_dict()},
                       out_dir / "ss_nn_best.pt")
            print(f"  saved → {(out_dir / 'ss_nn_best.pt').resolve()}")
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{args.epochs}  train={ep_loss:.4f}  val={val_loss:.4f}  "
                  f"tf={tf:.2f}  best_val={best_val:.4f}  ({elapsed:.0f}s)")
        if no_improve >= args.patience:
            print(f"  early stop at epoch {epoch} (no improve for {args.patience} epochs)")
            break

    # Save last
    torch.save({"model": model.state_dict(), "yhead": yhead.state_dict()},
               out_dir / "ss_nn_last.pt")
    print(f"  saved → {(out_dir / 'ss_nn_last.pt').resolve()}")

    # Save history
    (metrics_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    print(f"  saved → {(metrics_dir / 'training_log.json').resolve()}")

    # ----------------------------------------------------------------------- #
    # Test evaluation
    # ----------------------------------------------------------------------- #
    print("Evaluating on test set…")
    ckpt = torch.load(out_dir / "ss_nn_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    yhead.load_state_dict(ckpt["yhead"])
    model.eval()

    Xt = torch.tensor(test["X"], dtype=torch.float32).to(device)
    Yt = torch.tensor(test["Y"], dtype=torch.float32).to(device)
    Mt = torch.tensor(test["Y_mask"], dtype=torch.float32).to(device)

    test_metrics = {}
    all_preds = []
    all_truths = []
    with torch.no_grad():
        bs_eval = args.bs
        for i in range(0, Xt.shape[0], bs_eval):
            xb = Xt[i : i + bs_eval]
            yb = Yt[i : i + bs_eval]
            y_pred = model(xb, y_prev=yb, teacher_forcing=1.0)
            all_preds.append(y_pred.cpu().numpy())
            all_truths.append(yb.cpu().numpy())
    y_pred_all = np.concatenate(all_preds, axis=0)
    y_true_all = np.concatenate(all_truths, axis=0)
    mask_all = Mt.cpu().numpy().astype(bool)

    # Compute metrics in original y space (denormalize via scaler_y)
    scaler_path = Path(args.scalers)
    if not scaler_path.exists():
        scaler_path = Path(cfg.ROOT) / "data" / "processed" / "scalers.npz"
    scaler = np.load(scaler_path)
    y_mean = scaler["y_mean"]
    y_scale = scaler["y_scale"]
    y_pred_denorm = y_pred_all * y_scale + y_mean
    y_true_denorm = y_true_all * y_scale + y_mean

    per_var = per_variable_metrics(y_true_denorm, y_pred_denorm,
                                names=["y1", "y2", "y3", "y4"],
                                mask=mask_all)
    test_metrics = {
        "per_variable": per_var,
        "overall_mse": float(np.mean([
            per_var[k]["mse"] for k in per_var
        ])),
    }
    (metrics_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"  saved → {(metrics_dir / 'test_metrics.json').resolve()}")

    # Save predictions
    np.savez(
        pred_dir / "test_predictions.npz",
        y_true=y_true_denorm,
        y_pred=y_pred_denorm,
        mask=mask_all,
        u=test["X"],
        file_ids=test["file_ids"],
    )
    print(f"  saved → {(pred_dir / 'test_predictions.npz').resolve()}")
    print(f"Test metrics: {json.dumps(per_var, indent=2)}")
    print("Done.")


if __name__ == "__main__":
    main()