"""Regression / forecasting metrics."""
from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error. NaN-safe (skip pairs where either side is NaN)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if not mask.any():
        return float("nan")
    return float(np.mean((yt[mask] - yp[mask]) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = ~(np.isnan(yt) | np.isnan(yp)) & (np.abs(yt) > eps)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if not mask.any():
        return float("nan")
    yt, yp = yt[mask], yp[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def per_variable_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: list[str], mask: np.ndarray | None = None
) -> dict:
    """Return {name: {mse, mae, mape, r2}} for each output variable.

    If ``mask`` is provided, only positions where ``mask[..., j]`` is True
    are used per variable (skips missing-y entries that were filled to 0
    during preprocessing).
    """
    assert y_true.shape == y_pred.shape, "shape mismatch"
    assert y_true.shape[-1] == len(names)
    out = {}
    for j, name in enumerate(names):
        yt = y_true[..., j].ravel()
        yp = y_pred[..., j].ravel()
        if mask is not None:
            m = mask[..., j].ravel().astype(bool)
            yt = yt[m]
            yp = yp[m]
        out[name] = {
            "mse": mse(yt, yp),
            "mae": mae(yt, yp),
            "mape": mape(yt, yp),
            "r2": r2(yt, yp),
        }
    return out