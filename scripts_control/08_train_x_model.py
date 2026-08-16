"""Train an independent x1–x8 multi-step-ahead forecaster.

The model is ``SS_NN_Hybrid(dim_u=8, dim_y=8)`` with ``init_x_recon`` (identity
D feedthrough), reinterpreted as a **lag-1 predictor**: given a context of past
x, predict the next H future values of x1–x8. Training uses a teacher-forcing
schedule (mirroring ``scripts_control.03_train_predictor``) with a 50/50
teacher-forced / autoregressive split per batch; evaluation is a **true
autoregressive rollout** — the model feeds its own previous prediction back at
each step (``forecast(..., x_future_gt=None, teacher_forcing=0.0)``).

Usage::

    python -m scripts_control.08_train_x_model \
        --data    data/processed/train.npz \
        --test    data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --epochs 200 --bs 16 --lr 1e-3 \
        --context 32 --horizon 32 --tf-decay 50 --patience 30

Outputs (under ``checkpoints/``, ``results/metrics/``, ``results/predictions/``):

* ``checkpoints/x_forecast_best.pt``  — best val-AR weights (raw state_dict)
* ``checkpoints/x_forecast_last.pt``  — final-epoch weights
* ``results/metrics/x_forecast_training_log.json`` — per-epoch losses
* ``results/metrics/x_forecast_metrics.json``       — per-x-variable MSE/MAE/R²
* ``results/predictions/test_x_forecast.npz``  — x_true / x_pred / mask / context
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src_control.config import get_config, resolve_paths
from src_control.models.state_space_nn import SS_NN_Hybrid, init_x_recon
from src_control.preprocess import load_processed
from src_control.utils.metrics import per_variable_metrics
from src_control.utils.seed import set_seed


X_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")


# --------------------------------------------------------------------------- #
# Forecast pair construction & loss helpers
# --------------------------------------------------------------------------- #
def build_forecast_tensors(
    X: np.ndarray,
    lengths: np.ndarray,
    C: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (context, target) pairs anchored at each sample's tail.

    ``X`` is ``(N, T, 8)`` standardized; ``lengths`` is ``(N,)``. For sample i
    the forecast starts at ``s_i = L_i − H``; context is ``X[i, s_i−C : s_i]``
    and target is ``X[i, s_i : s_i+H]``. Both are fully valid (no padding), and
    the forecast lands exactly at the end of the sample. Samples shorter than
    ``C + H`` are dropped.

    Returns ``context`` ``(M, C, 8)``, ``target`` ``(M, H, 8)``, ``mask``
    ``(M, H)`` (all-True by construction), and ``keep`` ``(N,)`` bool mask
    marking which original samples were retained (to align ``file_ids``).
    """
    s = lengths - H                       # (N,) forecast start per sample
    keep = s - C >= 0                     # need a full context window
    s = s[keep]
    Xk = X[keep]
    M = int(s.shape[0])
    c_idx = s[:, None] - C + np.arange(C)[None, :]   # (M, C)
    t_idx = s[:, None] + np.arange(H)[None, :]       # (M, H)
    context = Xk[np.arange(M)[:, None], c_idx]       # (M, C, 8)
    target = Xk[np.arange(M)[:, None], t_idx]        # (M, H, 8)
    mask = np.ones((M, H), dtype=bool)
    return context, target, mask, keep


def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over positions where ``mask`` is True (padding excluded)."""
    diff = (pred - true) ** 2
    diff = diff * mask.float()
    n = mask.float().sum().clamp_min(1.0)
    return diff.sum() / n


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def make_loaders(
    context: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    val_ratio: float,
    seed: int,
    bs: int,
) -> tuple[DataLoader, DataLoader]:
    M = context.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(M)
    n_val = int(round(M * val_ratio))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    ctx_tr = torch.tensor(context[tr_idx], dtype=torch.float32)
    tgt_tr = torch.tensor(target[tr_idx], dtype=torch.float32)
    msk_tr = torch.tensor(mask[tr_idx], dtype=torch.float32)
    ctx_va = torch.tensor(context[val_idx], dtype=torch.float32)
    tgt_va = torch.tensor(target[val_idx], dtype=torch.float32)
    msk_va = torch.tensor(mask[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(ctx_tr, tgt_tr, msk_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(ctx_va, tgt_va, msk_va), batch_size=bs, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.npz")
    parser.add_argument("--test", default="data/processed/test.npz")
    parser.add_argument("--scalers", default="data/processed/scalers.npz")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument(
        "--out-root", default=None,
        help="Output root: if set, checkpoints/metrics/predictions go under "
             "<out-root>/checkpoints, <out-root>/results/metrics, "
             "<out-root>/results/predictions (e.g. --out-root scripts_control). "
             "Overrides the individual --out-dir/--metrics-dir/--predictions-dir.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-state", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--tf-decay", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = resolve_paths(get_config())
    set_seed(args.seed)
    device = cfg.DEVICE

    if args.out_root:
        out_root = Path(args.out_root)
        out_dir = out_root / "checkpoints"
        metrics_dir = out_root / "results" / "metrics"
        pred_dir = out_root / "results" / "predictions"
    else:
        out_dir = Path(args.out_dir)
        metrics_dir = Path(args.metrics_dir)
        pred_dir = Path(args.predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    train = load_processed(args.data)
    test = load_processed(args.test)
    print(f"  train X={train['X'].shape}, test X={test['X'].shape}, device={device}")

    # Build x-forecaster (dim_u=8, dim_y=8) with persistence (identity-D) warm start.
    model = SS_NN_Hybrid(
        dim_u=8, dim_y=8, n_state=args.n_state, hidden=args.hidden, window=4
    )
    init_x_recon(model)
    model = model.to(device)

    context, target, mask, _ = build_forecast_tensors(
        train["X"], train["lengths"], args.context, args.horizon
    )
    n_dropped = train["X"].shape[0] - context.shape[0]
    if context.shape[0] == 0:
        raise RuntimeError("No sample is long enough for context+horizon; lower --context/--horizon.")
    print(f"  forecast pairs: {context.shape[0]} (dropped {n_dropped} short samples)")

    train_loader, val_loader = make_loaders(
        context, target, mask, val_ratio=args.val_ratio, seed=args.seed, bs=args.bs
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=1e-5
    )

    history = []
    best_val_ar = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Teacher-forcing schedule: 1 → 0 over tf_decay epochs
        tf = max(0.0, 1.0 - epoch / max(1, args.tf_decay))
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for ctx, tgt, msk in train_loader:
            ctx = ctx.to(device)
            tgt = tgt.to(device)
            msk = msk.to(device)
            msk3 = msk.unsqueeze(-1).expand(-1, -1, 8)   # (B, H, 8)

            optim.zero_grad()
            # 50% teacher-forced, 50% pure AR — robust to both regimes
            use_tf = (torch.rand(1).item() < 0.5)
            if use_tf:
                x_hat = model.forecast(ctx, args.horizon, x_future_gt=tgt, teacher_forcing=tf)
            else:
                x_hat = model.forecast(ctx, args.horizon, x_future_gt=None, teacher_forcing=0.0)
            loss = masked_mse(x_hat, tgt, msk3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batch += 1
        ep_loss /= max(1, n_batch)
        scheduler.step()

        # Validation — teacher-forced (val) and true-AR (val_ar)
        model.eval()
        val_loss = 0.0
        val_ar_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for ctx, tgt, msk in val_loader:
                ctx = ctx.to(device)
                tgt = tgt.to(device)
                msk = msk.to(device)
                msk3 = msk.unsqueeze(-1).expand(-1, -1, 8)
                val_loss += float(masked_mse(
                    model.forecast(ctx, args.horizon, x_future_gt=tgt, teacher_forcing=1.0),
                    tgt, msk3,
                ).item())
                val_ar_loss += float(masked_mse(
                    model.forecast(ctx, args.horizon, x_future_gt=None, teacher_forcing=0.0),
                    tgt, msk3,
                ).item())
                n_v += 1
        val_loss /= max(1, n_v)
        val_ar_loss /= max(1, n_v)

        history.append({"epoch": epoch, "train": ep_loss, "val": val_loss,
                        "val_ar": val_ar_loss, "tf": tf})
        # Model selection & early stop on true-AR (the deployment mode).
        if val_ar_loss < best_val_ar - 1e-6:
            best_val_ar = val_ar_loss
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "x_forecast_best.pt")
            print(f"  saved → {(out_dir / 'x_forecast_best.pt').resolve()}")
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{args.epochs}  train={ep_loss:.4f}  "
                  f"val={val_loss:.4f}  val_ar={val_ar_loss:.4f}  tf={tf:.2f}  "
                  f"best_val_ar={best_val_ar:.4f}  ({elapsed:.0f}s)")
        if no_improve >= args.patience:
            print(f"  early stop at epoch {epoch} (no improve for {args.patience} epochs)")
            break

    torch.save(model.state_dict(), out_dir / "x_forecast_last.pt")
    print(f"  saved → {(out_dir / 'x_forecast_last.pt').resolve()}")
    (metrics_dir / "x_forecast_training_log.json").write_text(json.dumps(history, indent=2))
    print(f"  saved → {(metrics_dir / 'x_forecast_training_log.json').resolve()}")

    # ----------------------------------------------------------------------- #
    # Test evaluation (true autoregressive rollout)
    # ----------------------------------------------------------------------- #
    print("Evaluating on test set…")
    model.load_state_dict(torch.load(out_dir / "x_forecast_best.pt", map_location=device))
    model.eval()

    ctx_t, tgt_t, msk_t, keep_t = build_forecast_tensors(
        test["X"], test["lengths"], args.context, args.horizon
    )
    ctx_t_t = torch.tensor(ctx_t, dtype=torch.float32).to(device)
    with torch.no_grad():
        x_pred = model.forecast(ctx_t_t, args.horizon, x_future_gt=None, teacher_forcing=0.0)
    x_pred_all = x_pred.cpu().numpy()

    # Denormalize into original x space
    scaler_path = Path(args.scalers)
    if not scaler_path.exists():
        scaler_path = Path(cfg.ROOT) / "data" / "processed" / "scalers.npz"
    x_mean = np.zeros(8, dtype=np.float32)
    x_scale = np.ones(8, dtype=np.float32)
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        x_mean = scaler["x_mean"]
        x_scale = scaler["x_scale"]
    x_pred_denorm = x_pred_all * x_scale + x_mean
    x_true_denorm = tgt_t * x_scale + x_mean

    mask_3d = np.broadcast_to(msk_t[..., None], (msk_t.shape[0], args.horizon, 8))

    per_var = per_variable_metrics(
        x_true_denorm, x_pred_denorm, names=list(X_NAMES), mask=mask_3d
    )
    (metrics_dir / "x_forecast_metrics.json").write_text(
        json.dumps({"per_variable": per_var}, indent=2)
    )
    print(f"  saved → {(metrics_dir / 'x_forecast_metrics.json').resolve()}")

    start_idx = test["lengths"][keep_t] - args.horizon
    np.savez(
        pred_dir / "test_x_forecast.npz",
        x_true=x_true_denorm,
        x_pred=x_pred_denorm,
        mask=msk_t,
        context=ctx_t * x_scale + x_mean,
        start_idx=start_idx,
        file_ids=test["file_ids"][keep_t],
    )
    print(f"  saved → {(pred_dir / 'test_x_forecast.npz').resolve()}")
    print(f"Test metrics: {json.dumps(per_var, indent=2)}")
    print("Done.")


if __name__ == "__main__":
    main()
