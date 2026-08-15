"""Train an independent x1–x8 denoising/reconstruction model.

Mirrors ``scripts_control.03_train_predictor``, but the prediction target is
**x** instead of **y**. The model is ``SS_NN_Hybrid(dim_u=8, dim_y=8)`` —
input = corrupted x (masked + noisy), output = clean x̂. The existing
y-predictor / MPC pipeline is left untouched.

The corruption at training time is what makes the task non-trivial: with a
clean input the model would just learn the identity. Masking entries and
adding noise teaches it to impute missing values and smooth anomalies.

Usage::

    python -m scripts_control.08_train_x_model \
        --data    data/processed/train.npz \
        --test    data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --epochs 200 --bs 16 --lr 1e-3 --patience 30 

Outputs (under ``checkpoints/``, ``results/metrics/``, ``results/predictions/``):

* ``checkpoints/x_recon_best.pt``  — best val-loss weights
* ``checkpoints/x_recon_last.pt``  — final-epoch weights
* ``results/metrics/x_recon_training_log.json`` — per-epoch losses
* ``results/metrics/x_recon_metrics.json``       — per-x-variable MSE/MAE/R²
* ``results/predictions/test_x_predictions.npz``  — x_true / x_pred / valid mask
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
# Corruption & loss helpers
# --------------------------------------------------------------------------- #
def corrupt_x(
    x: torch.Tensor,
    mask_prob: float,
    noise_std: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Zero out a random fraction of entries and add Gaussian noise.

    ``x`` is ``(B, T, 8)`` on its device; ``gen`` must live on the same device.
    """
    mask = torch.rand(x.shape, generator=gen, device=x.device) < mask_prob
    noise = torch.randn(x.shape, generator=gen, device=x.device) * noise_std
    xc = torch.where(mask, torch.zeros_like(x), x.clone())
    return xc + noise


def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over positions where ``mask`` is True (padding excluded)."""
    diff = (pred - true) ** 2
    diff = diff * mask.float()
    n = mask.float().sum().clamp_min(1.0)
    return diff.sum() / n


def padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T) bool mask: True for ``t < length[b]`` (non-padded positions)."""
    idx = torch.arange(T, device=lengths.device)[None, :]
    return idx < lengths[:, None]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def make_loaders(
    train: dict,
    val_ratio: float,
    seed: int,
    bs: int,
) -> tuple[DataLoader, DataLoader]:
    N = train["X"].shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = int(round(N * val_ratio))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.tensor(train["X"][tr_idx], dtype=torch.float32)
    L_tr = torch.tensor(train["lengths"][tr_idx], dtype=torch.long)
    X_va = torch.tensor(train["X"][val_idx], dtype=torch.float32)
    L_va = torch.tensor(train["lengths"][val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, L_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, L_va), batch_size=bs, shuffle=False)
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
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.1)
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
    T = train["X"].shape[1]
    print(f"  train X={train['X'].shape}, test X={test['X'].shape}, device={device}")

    # Build x-reconstruction model (dim_u=8, dim_y=8) with identity-D warm start.
    model = SS_NN_Hybrid(
        dim_u=8, dim_y=8, n_state=args.n_state, hidden=args.hidden, window=4
    )
    init_x_recon(model)
    model = model.to(device)

    train_loader, val_loader = make_loaders(
        train, val_ratio=args.val_ratio, seed=args.seed, bs=args.bs
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=1e-5
    )

    history = []
    best_val = float("inf")
    no_improve = 0
    t0 = time.time()
    train_gen = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for X, L in train_loader:
            X = X.to(device)
            L = L.to(device)
            p_mask = padding_mask(L, T).unsqueeze(-1).expand(-1, -1, 8)

            optim.zero_grad()
            x_corrupt = corrupt_x(X, args.mask_prob, args.noise_std, train_gen)
            x_pred = model(x_corrupt, y_prev=None, teacher_forcing=0.0)
            loss = masked_mse(x_pred, X, p_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batch += 1
        ep_loss /= max(1, n_batch)
        scheduler.step()

        # Validation — deterministic corruption (fresh fixed seed each epoch)
        model.eval()
        val_loss = 0.0
        n_v = 0
        val_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        with torch.no_grad():
            for X, L in val_loader:
                X = X.to(device)
                L = L.to(device)
                p_mask = padding_mask(L, T).unsqueeze(-1).expand(-1, -1, 8)
                x_corrupt = corrupt_x(X, args.mask_prob, args.noise_std, val_gen)
                x_pred = model(x_corrupt, y_prev=None, teacher_forcing=0.0)
                val_loss += float(masked_mse(x_pred, X, p_mask).item())
                n_v += 1
        val_loss /= max(1, n_v)

        history.append({"epoch": epoch, "train": ep_loss, "val": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "x_recon_best.pt")
            print(f"  saved → {(out_dir / 'x_recon_best.pt').resolve()}")
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{args.epochs}  train={ep_loss:.4f}  "
                  f"val={val_loss:.4f}  best_val={best_val:.4f}  ({elapsed:.0f}s)")
        if no_improve >= args.patience:
            print(f"  early stop at epoch {epoch} (no improve for {args.patience} epochs)")
            break

    torch.save(model.state_dict(), out_dir / "x_recon_last.pt")
    print(f"  saved → {(out_dir / 'x_recon_last.pt').resolve()}")
    (metrics_dir / "x_recon_training_log.json").write_text(json.dumps(history, indent=2))
    print(f"  saved → {(metrics_dir / 'x_recon_training_log.json').resolve()}")

    # ----------------------------------------------------------------------- #
    # Test evaluation (deterministic corruption)
    # ----------------------------------------------------------------------- #
    print("Evaluating on test set…")
    model.load_state_dict(torch.load(out_dir / "x_recon_best.pt", map_location=device))
    model.eval()

    Xt = torch.tensor(test["X"], dtype=torch.float32).to(device)
    Lt = torch.tensor(test["lengths"], dtype=torch.long).to(device)
    test_gen = torch.Generator(device=device).manual_seed(args.seed + 2)

    all_preds = []
    bs_eval = args.bs
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs_eval):
            xb = Xt[i : i + bs_eval]
            lb = Lt[i : i + bs_eval]
            x_corrupt = corrupt_x(xb, args.mask_prob, args.noise_std, test_gen)
            x_pred = model(x_corrupt, y_prev=None, teacher_forcing=0.0)
            all_preds.append(x_pred.cpu().numpy())
    x_pred_all = np.concatenate(all_preds, axis=0)
    x_true_all = Xt.cpu().numpy()

    # Denormalize into original x space (optional scaler)
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
    x_true_denorm = x_true_all * x_scale + x_mean

    # Valid mask (N, T, 8): non-padded positions, broadcast across channels.
    lengths_np = Lt.cpu().numpy()
    valid_2d = (np.arange(T)[None, :] < lengths_np[:, None]).astype(bool)
    valid_3d = np.broadcast_to(valid_2d[..., None], (valid_2d.shape[0], T, 8)).copy()

    per_var = per_variable_metrics(
        x_true_denorm, x_pred_denorm, names=list(X_NAMES), mask=valid_3d
    )
    (metrics_dir / "x_recon_metrics.json").write_text(
        json.dumps({"per_variable": per_var}, indent=2)
    )
    print(f"  saved → {(metrics_dir / 'x_recon_metrics.json').resolve()}")

    np.savez(
        pred_dir / "test_x_predictions.npz",
        x_true=x_true_denorm,
        x_pred=x_pred_denorm,
        mask=valid_3d,
        file_ids=test["file_ids"],
    )
    print(f"  saved → {(pred_dir / 'test_x_predictions.npz').resolve()}")
    print(f"Test metrics: {json.dumps(per_var, indent=2)}")
    print("Done.")


if __name__ == "__main__":
    main()
