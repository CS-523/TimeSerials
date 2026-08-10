"""Correlation, mutual-information, PCA, lag and Granger-causality analysis.

Operates on a processed dataset dict (see ``preprocess.build_dataset``).
Outputs a set of PNG figures and a JSON summary under ``out_dir``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

X_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
Y_NAMES = ("y1", "y2", "y3", "y4")
ALL_NAMES = X_NAMES + Y_NAMES


def _flatten_with_mask(
    ds: Dict[str, np.ndarray], max_rows: int = 200_000
) -> pd.DataFrame:
    """Flatten (N, T, :) tensors into a single DataFrame using observed-y mask.

    For ``y`` columns, only observed positions are kept (NaN elsewhere).
    """
    X = ds["X"]   # (N, T, 8)
    Y = ds["Y"]   # (N, T, 4)
    mask = ds["Y_mask"]  # (N, T, 4)
    N, T = X.shape[:2]

    x_flat = X.reshape(-1, 8)
    y_flat = Y.reshape(-1, 4)
    m_flat = mask.reshape(-1, 4)

    # Replace unobserved y with NaN so correlations handle them gracefully.
    y_with_nan = np.where(m_flat, y_flat, np.nan)

    # Truncate if too many rows (downsample uniformly for speed)
    total = x_flat.shape[0]
    if total > max_rows:
        idx = np.random.RandomState(0).choice(total, size=max_rows, replace=False)
        x_flat = x_flat[idx]
        y_with_nan = y_with_nan[idx]

    cols = {name: x_flat[:, j] for j, name in enumerate(X_NAMES)}
    for j, name in enumerate(Y_NAMES):
        cols[name] = y_with_nan[:, j]
    return pd.DataFrame(cols)


def correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson and Spearman correlation matrices across all 12 variables."""
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    return pearson, spearman


def mutual_information_matrix(df: pd.DataFrame, n_bins: int = 16) -> np.ndarray:
    """Pairwise MI for the 12 variables.

    For each ordered pair (i, j), discretize j into quantile bins and call
    sklearn's ``mutual_info_regression``. The returned MI is symmetrized by
    averaging with the transpose.
    """
    arr = df.values
    n_vars = arr.shape[1]
    mi = np.zeros((n_vars, n_vars))
    bin_edges = np.quantile(arr[~np.isnan(arr).any(axis=1)], np.linspace(0, 1, n_bins + 1), axis=0)
    # Replace NaN rows with column medians (MI cannot handle NaN directly).
    arr_clean = arr.copy()
    for j in range(n_vars):
        col = arr_clean[:, j]
        valid = ~np.isnan(col)
        if not valid.all():
            med = np.nanmedian(col)
            col[~valid] = med
            arr_clean[:, j] = col

    for i in range(n_vars):
        X = arr_clean[:, [j for j in range(n_vars) if j != i]]
        y_target = arr_clean[:, i]
        # Discretize each X column
        Xb = np.zeros_like(X)
        for k in range(X.shape[1]):
            edges = np.unique(np.quantile(X[:, k], np.linspace(0, 1, n_bins + 1)))
            Xb[:, k] = np.digitize(X[:, k], edges[1:-1])
        yb = np.digitize(y_target, np.unique(np.quantile(y_target, np.linspace(0, 1, n_bins + 1)))[1:-1])
        scores = mutual_info_regression(Xb, yb, discrete_features=True, random_state=0)
        for k, j in enumerate([jj for jj in range(n_vars) if jj != i]):
            mi[i, j] = scores[k]
    # Symmetrize
    mi_sym = (mi + mi.T) / 2.0
    np.fill_diagonal(mi_sym, 0.0)
    return mi_sym


def pca_importance(df: pd.DataFrame, n_components: int = 4) -> Dict[str, np.ndarray]:
    """PCA on x variables only; return explained variance + loadings."""
    X = df[list(X_NAMES)].values
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    return {
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "loadings": pca.components_.T * np.sqrt(pca.explained_variance_),
    }


def granger_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """Linear Granger test: at each lag k, fit AR models with/without lagged y.

    Returns ratio ``SSR_reduced / SSR_full`` per lag k=1..max_lag. Lower ratio
    ⇒ y helps predict x.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    T = x.shape[0]
    ratios = np.full(max_lag, np.nan)
    for k in range(1, max_lag + 1):
        if T <= 2 * k + 1:
            continue
        Y_full = x[k:]
        # Build design matrix: [1, x_{t-1}, ..., x_{t-k}, y_{t-1}, ..., y_{t-k}]
        cols = [np.ones_like(Y_full)]
        for lag in range(1, k + 1):
            cols.append(x[k - lag : T - lag])
        for lag in range(1, k + 1):
            cols.append(y[k - lag : T - lag])
        X_full = np.column_stack(cols)

        # Reduced: drop y lags
        cols_r = [np.ones_like(Y_full)]
        for lag in range(1, k + 1):
            cols_r.append(x[k - lag : T - lag])
        X_red = np.column_stack(cols_r)

        # Solve via least squares
        try:
            beta_full, *_ = np.linalg.lstsq(X_full, Y_full, rcond=None)
            beta_red, *_ = np.linalg.lstsq(X_red, Y_full, rcond=None)
        except np.linalg.LinAlgError:
            continue
        res_full = Y_full - X_full @ beta_full
        res_red = Y_full - X_red @ beta_red
        ss_full = float(np.sum(res_full ** 2))
        ss_red = float(np.sum(res_red ** 2))
        if ss_full <= 1e-12:
            ratios[k - 1] = 1.0
        else:
            ratios[k - 1] = ss_red / ss_full
    return ratios


def lag_cross_correlation(df: pd.DataFrame, max_lag: int = 5) -> np.ndarray:
    """Cross-correlation between each pair of x variables and each y.

    Returns array of shape ``(8, 4, 2*max_lag+1)`` where index ``max_lag+k`` is
    the correlation of x[:,j].shift(k) with y[:,i].
    """
    out = np.zeros((8, 4, 2 * max_lag + 1))
    for j in range(8):
        xcol = df[X_NAMES[j]].values
        for i in range(4):
            ycol = df[Y_NAMES[i]].values
            for k, lag in enumerate(range(-max_lag, max_lag + 1)):
                if lag < 0:
                    a = xcol[:lag]
                    b = ycol[-lag:]
                elif lag > 0:
                    a = xcol[lag:]
                    b = ycol[:-lag]
                else:
                    a = xcol
                    b = ycol
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() < 2:
                    continue
                aa = a[mask] - a[mask].mean()
                bb = b[mask] - b[mask].mean()
                denom = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
                out[j, i, k] = float((aa * bb).sum() / denom) if denom > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _save(fig, path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation: x1..x8 vs y1..y4")
    _save(fig, out_path)


def plot_mi_heatmap(mi: np.ndarray, names: List[str], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mi, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Mutual information (quantile-binned)")
    _save(fig, out_path)


def plot_pca_scree(pca_info: Dict[str, np.ndarray], out_path: str | Path) -> None:
    evr = pca_info["explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, len(evr) + 1), evr, color="#4C72B0", label="Per component")
    cum = np.cumsum(evr)
    ax.plot(range(1, len(evr) + 1), cum, "o-", color="#C44E52", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(range(1, len(evr) + 1))
    ax.legend()
    ax.set_title("PCA on x1..x8")
    _save(fig, out_path)


def plot_lag_x_to_y4(lag_corr: np.ndarray, out_path: str | Path) -> None:
    """Plot x_j vs y4 cross-correlation vs lag."""
    fig, ax = plt.subplots(figsize=(8, 5))
    max_lag = (lag_corr.shape[2] - 1) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    for j in range(8):
        ax.plot(lags, lag_corr[j, 3], marker="o", label=X_NAMES[j])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag (x lag relative to y4)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Cross-correlation: x_j vs y4")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_path)


def plot_granger_xy(granger_xy: np.ndarray, granger_yx: np.ndarray, out_path: str | Path) -> None:
    """Plot top Granger ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    max_lag = granger_xy.shape[1]
    lags = np.arange(1, max_lag + 1)
    for i in range(4):
        axes[0].plot(lags, granger_xy[i], marker="o", label=f"x→{Y_NAMES[i]}")
    axes[0].axhline(1.0, color="black", lw=0.5, ls="--")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("SSR_red / SSR_full")
    axes[0].set_title("Granger causality: each x_j → y")
    axes[0].legend(fontsize=7)

    for j in range(8):
        axes[1].plot(lags, granger_yx[j], marker="o", label=f"{X_NAMES[j]}←y4")
    axes[1].axhline(1.0, color="black", lw=0.5, ls="--")
    axes[1].set_xlabel("Lag")
    axes[1].set_title("Granger causality: y4 → each x_j")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def run_analysis(ds: Dict[str, np.ndarray], out_dir: str | Path) -> Dict[str, str]:
    """Compute all analyses and write outputs. Returns dict of generated files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _flatten_with_mask(ds)
    print(f"[analysis] flattened shape: {df.shape}")

    outputs: Dict[str, str] = {}

    # 1) Correlation heatmap (Pearson)
    pearson, _spearman = correlation_matrix(df)
    p1 = out_dir / "correlation_heatmap.png"
    plot_correlation_heatmap(pearson, p1)
    outputs["correlation_heatmap"] = str(p1)

    # 2) Mutual information
    mi = mutual_information_matrix(df, n_bins=16)
    p2 = out_dir / "mi_heatmap.png"
    plot_mi_heatmap(mi, list(ALL_NAMES), p2)
    outputs["mi_heatmap"] = str(p2)

    # 3) PCA on x
    pca_info = pca_importance(df[list(X_NAMES)], n_components=4)
    p3 = out_dir / "pca_scree.png"
    plot_pca_scree(pca_info, p3)
    outputs["pca_scree"] = str(p3)

    # 4) Lag cross-correlation
    lag_corr = lag_cross_correlation(df, max_lag=5)
    p4 = out_dir / "lag_x_to_y4.png"
    plot_lag_x_to_y4(lag_corr, p4)
    outputs["lag_x_to_y4"] = str(p4)

    # 5) Granger causality (x_j → y_i and y4 → x_j)
    granger_xy = np.zeros((4, 5))   # (n_y, max_lag)
    granger_yx = np.zeros((8, 5))   # (n_x, max_lag)
    for j in range(8):
        xcol = df[X_NAMES[j]].values
        ratios = granger_lag(xcol, df["y4"].values, max_lag=5)
        granger_yx[j] = ratios
        for i in range(4):
            ratios2 = granger_lag(xcol, df[Y_NAMES[i]].values, max_lag=5)
            granger_xy[i] = ratios2
    p5 = out_dir / "granger_xy.png"
    plot_granger_xy(granger_xy, granger_yx, p5)
    outputs["granger_xy"] = str(p5)

    # 6) JSON report
    report = {
        "pearson_y4_vs_x": {
            f"x{j+1}": float(pearson.iloc[j, 11]) for j in range(8)
        },
        "pca_explained_variance_ratio": pca_info["explained_variance_ratio"].tolist(),
        "granger_y4_to_x": {
            X_NAMES[j]: granger_yx[j].tolist() for j in range(8)
        },
        "granger_x_to_y4": {
            f"x{j+1}": granger_yx[j].tolist() for j in range(8)
        },
    }
    rp = out_dir / "analysis_report.json"
    rp.write_text(json.dumps(report, indent=2))
    outputs["analysis_report"] = str(rp)

    return outputs


if __name__ == "__main__":
    import argparse
    from src_control.preprocess import load_processed

    parser = argparse.ArgumentParser(description="Run feature analysis.")
    parser.add_argument("--data", default="data/processed/train.npz")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    ds = load_processed(args.data)
    print(f"Loaded dataset: X={ds['X'].shape}")
    files = run_analysis(ds, args.out)
    print(f"Generated {len(files)} files in {args.out}")
    for k, v in files.items():
        print(f"  {k}: {v}")