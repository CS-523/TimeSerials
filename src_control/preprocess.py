"""Data preprocessing: anomaly detection, missing-value handling, scaling, splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src_control.data_loader import Sample, X_COLS, Y_COLS


@dataclass
class FittedScalers:
    """Container for fitted scalers."""
    scaler_x: StandardScaler
    scaler_y: StandardScaler

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            x_mean=self.scaler_x.mean_,
            x_scale=self.scaler_x.scale_,
            y_mean=self.scaler_y.mean_,
            y_scale=self.scaler_y.scale_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FittedScalers":
        data = np.load(path)
        sx = StandardScaler()
        sx.mean_ = data["x_mean"]
        sx.scale_ = data["x_scale"]
        sx.n_features_in_ = len(data["x_mean"])
        sy = StandardScaler()
        sy.mean_ = data["y_mean"]
        sy.scale_ = data["y_scale"]
        sy.n_features_in_ = len(data["y_mean"])
        return cls(scaler_x=sx, scaler_y=sy)


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
def detect_anomalies_x(
    x: np.ndarray, z_threshold: float = 5.0
) -> np.ndarray:
    """Per-column z-score; mark ``True`` where ``|z| > z_threshold``.

    Returns a boolean array of shape ``(T, 8)`` — ``True`` = anomalous.
    """
    out = np.zeros_like(x, dtype=bool)
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        mu = col[valid].mean()
        sd = col[valid].std(ddof=1) if valid.sum() > 1 else 0.0
        if sd <= 0.0:
            continue
        z = np.abs((col - mu) / sd)
        out[:, j] = z > z_threshold
    return out


def detect_anomalies_y(
    y: np.ndarray, y_mask: np.ndarray, z_threshold: float = 5.0
) -> np.ndarray:
    """Anomaly mask for ``y`` considering only observed entries.

    Returns a boolean array of shape ``(T, 4)``.
    """
    out = np.zeros_like(y, dtype=bool)
    for j in range(y.shape[1]):
        col = y[:, j]
        valid = y_mask[:, j] & ~np.isnan(col)
        if not valid.any():
            continue
        mu = col[valid].mean()
        sd = col[valid].std(ddof=1) if valid.sum() > 1 else 0.0
        if sd <= 0.0:
            continue
        z = np.abs((col - mu) / sd)
        out[:, j] = valid & (z > z_threshold)
    return out


def anomaly_report(samples: List[Sample], z_threshold: float = 5.0) -> Dict[str, int]:
    """Total anomaly counts per column across the dataset."""
    counts = {c: 0 for c in X_COLS + Y_COLS}
    for s in samples:
        ax = detect_anomalies_x(s.x, z_threshold)
        ay = detect_anomalies_y(s.y, s.y_present_mask, z_threshold)
        for j, c in enumerate(X_COLS):
            counts[c] += int(ax[:, j].sum())
        for j, c in enumerate(Y_COLS):
            counts[c] += int(ay[:, j].sum())
    return counts


# --------------------------------------------------------------------------- #
# Missing-value handling for y (y is sparse by design)
# --------------------------------------------------------------------------- #
def fill_missing_y(y: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    """Forward-fill each y column, then back-fill leading NaN with first observed.

    **Trailing unobserved positions remain NaN** (no extrapolation). The
    "observed" positions are those marked True in ``y_mask``; cells with
    ``y_mask=False`` are *always* kept NaN in the output.
    """
    y_out = np.full_like(y, np.nan)
    for j in range(y.shape[1]):
        col = y[:, j].copy()
        msk = y_mask[:, j]

        # Forward fill on observed positions only
        last = np.nan
        for i in range(y.shape[0]):
            if msk[i] and not np.isnan(col[i]):
                last = col[i]
            elif msk[i] and not np.isnan(last):
                col[i] = last

        # Back-fill leading observed-NaN with first observed
        valid_idx = np.where(msk & ~np.isnan(col))[0]
        if len(valid_idx) > 0:
            first = valid_idx[0]
            first_val = col[first]
            for k in range(first):
                if msk[k]:
                    col[k] = first_val

        y_out[:, j] = col
    return y_out


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def fit_scalers(samples: List[Sample]) -> FittedScalers:
    """Fit StandardScaler on x (all rows) and y (per-column, observed rows only)."""
    sx = StandardScaler()
    sy = StandardScaler()

    X = np.concatenate([s.x for s in samples if s.T > 0], axis=0)
    sx.fit(X)

    # Per-column y fitting: build a 2-D matrix (N_obs_rows, 4) by aligning
    # observations row-by-row across columns where each column has its own mask.
    # Easiest robust approach: build per-column arrays, fit per column manually.
    means = np.zeros(4)
    stds = np.ones(4)
    for j in range(4):
        col_obs = []
        for s in samples:
            valid = s.y_present_mask[:, j] & ~np.isnan(s.y[:, j])
            if valid.any():
                col_obs.append(s.y[valid, j])
        if col_obs:
            v = np.concatenate(col_obs, axis=0)
            means[j] = v.mean()
            stds[j] = v.std() if v.std() > 0 else 1.0
    # Manually create a fitted scaler_y (avoid sklearn partial-fit complications)
    sy.mean_ = means
    sy.scale_ = stds
    sy.n_features_in_ = 4
    sy.var_ = stds ** 2
    sy.n_samples_seen_ = 1
    return FittedScalers(scaler_x=sx, scaler_y=sy)


def apply_scalers(
    x: np.ndarray,
    y: np.ndarray,
    scalers: FittedScalers,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted scalers; missing y cells become ``fill_value``."""
    x_std = scalers.scaler_x.transform(x)
    y_std = np.where(np.isnan(y), fill_value, scalers.scaler_y.transform(y))
    return x_std.astype(np.float32), y_std.astype(np.float32)


# --------------------------------------------------------------------------- #
# Padding / dataset construction
# --------------------------------------------------------------------------- #
def pad_or_truncate(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, cycle: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad (post) or truncate to ``seq_len``. Returns padded arrays and original length."""
    T = x.shape[0]
    if T >= seq_len:
        return x[:seq_len], y[:seq_len], mask[:seq_len], cycle[:seq_len], seq_len

    pad = seq_len - T
    x_pad = np.concatenate([x, np.zeros((pad, x.shape[1]), dtype=x.dtype)], axis=0)
    y_pad = np.concatenate([y, np.zeros((pad, y.shape[1]), dtype=y.dtype)], axis=0)
    mask_pad = np.concatenate([mask, np.zeros((pad, mask.shape[1]), dtype=bool)], axis=0)
    cycle_pad = np.concatenate([cycle, np.zeros((pad,), dtype=cycle.dtype)], axis=0)
    return x_pad, y_pad, mask_pad, cycle_pad, T


def build_dataset(
    samples: List[Sample],
    scalers: FittedScalers,
    seq_len: int = 64,
) -> Dict[str, np.ndarray]:
    """Build padded, scaled tensors from a list of :class:`Sample`."""
    N = len(samples)
    X = np.zeros((N, seq_len, 8), dtype=np.float32)
    Y = np.zeros((N, seq_len, 4), dtype=np.float32)
    Y_mask = np.zeros((N, seq_len, 4), dtype=bool)
    Cycle = np.zeros((N, seq_len), dtype=np.int64)
    Lengths = np.zeros((N,), dtype=np.int64)
    Y_final = np.zeros((N,), dtype=np.float32)
    FileIds: List[str] = []

    for i, s in enumerate(samples):
        if s.T == 0:
            FileIds.append(s.file_id)
            Y_final[i] = s.Y if np.isfinite(s.Y) else 0.0
            continue
        # Anomaly removal: set anomalous x entries to NaN, then interpolate linearly.
        x_clean = s.x.copy()
        ax = detect_anomalies_x(x_clean)
        x_clean[ax] = np.nan
        # Forward/back fill x
        for j in range(8):
            col = x_clean[:, j]
            valid = ~np.isnan(col)
            if not valid.any():
                col[:] = 0.0
            else:
                idx = np.arange(len(col))
                col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
            x_clean[:, j] = col

        # Fill missing y
        y_filled = fill_missing_y(s.y, s.y_present_mask)

        # Scale
        x_std, y_std = apply_scalers(x_clean, y_filled, scalers)

        # Pad/truncate
        xp, yp, mp, cp, T = pad_or_truncate(
            x_std, y_std, s.y_present_mask, s.cycle, seq_len
        )
        X[i] = xp
        Y[i] = yp
        Y_mask[i] = mp
        Cycle[i] = cp
        Lengths[i] = T
        Y_final[i] = s.Y if np.isfinite(s.Y) else 0.0
        FileIds.append(s.file_id)

    return {
        "X": X,
        "Y": Y,
        "Y_mask": Y_mask,
        "cycle": Cycle,
        "lengths": Lengths,
        "Y_final": Y_final,
        "file_ids": np.asarray(FileIds),
    }


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #
def split_dataset(
    ds: Dict[str, np.ndarray], ratio: float = 0.8, seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Random sample-level split; reproducible with ``seed``."""
    N = ds["X"].shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = int(round(N * ratio))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train = {k: ds[k][train_idx] for k in ds if k != "file_ids"}
    test = {k: ds[k][test_idx] for k in ds if k != "file_ids"}
    train["file_ids"] = ds["file_ids"][train_idx]
    test["file_ids"] = ds["file_ids"][test_idx]
    return train, test


def save_processed(ds: Dict[str, np.ndarray], path: str | Path) -> None:
    saveable = {k: v for k, v in ds.items() if k != "file_ids"}
    saveable["file_ids"] = np.asarray(ds["file_ids"], dtype=object)
    np.savez(path, **saveable)


def load_processed(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    import argparse
    from src_control.data_loader import parse_all

    parser = argparse.ArgumentParser(description="Build processed dataset.")
    parser.add_argument("--root", default="/kefu-nas/ybkong/time_serials-master")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, _ = parse_all(args.root)
    print(f"Loaded {len(samples)} samples")

    scalers = fit_scalers(samples)
    scalers.save(out_dir / "scalers.npz")
    print("Scalers fitted")

    ds = build_dataset(samples, scalers, seq_len=args.seq_len)
    print(f"Built dataset: X={ds['X'].shape}, Y={ds['Y'].shape}, lengths mean={ds['lengths'].mean():.1f}")

    train_ds, test_ds = split_dataset(ds, ratio=args.ratio, seed=args.seed)
    print(f"Train: {train_ds['X'].shape[0]} samples, Test: {test_ds['X'].shape[0]} samples")

    save_processed(ds, out_dir / "aligned_dataset.npz")
    save_processed(train_ds, out_dir / "train.npz")
    save_processed(test_ds, out_dir / "test.npz")
    print(f"Saved to {out_dir}")