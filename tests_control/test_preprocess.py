"""Tests for ``src_control.preprocess``."""
from __future__ import annotations

import numpy as np
import pytest

from src_control.data_loader import parse_all
from src_control.preprocess import (
    anomaly_report,
    build_dataset,
    detect_anomalies_x,
    detect_anomalies_y,
    fill_missing_y,
    fit_scalers,
    split_dataset,
)


ROOT = "/kefu-nas/ybkong/time_serials-master"


@pytest.fixture(scope="module")
def samples():
    s, _ = parse_all(ROOT)
    return s


def test_anomaly_detection_basic():
    # Use data with very obvious outlier so z-score test is unambiguous.
    x = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4], [50000.0, -50000.0]])
    ax = detect_anomalies_x(x, z_threshold=2.0)
    assert ax.shape == x.shape
    assert ax[-1].all(), f"last row should be anomalous: {ax[-1]}"
    assert not ax[:-1].any(), f"first rows should be normal: {ax[:-1]}"


def test_fill_missing_y_forward_fill():
    y = np.array([[np.nan, np.nan], [np.nan, 4.0], [3.0, np.nan], [np.nan, np.nan]])
    msk = np.array([[True, True], [True, True], [True, True], [True, True]])
    out = fill_missing_y(y, msk)
    # col 0: forward fill from row 2 (3.0); back-fill rows 0,1 with 3.0
    assert out[0, 0] == 3.0
    assert out[1, 0] == 3.0
    assert out[3, 0] == 3.0  # forward fill from row 2
    # col 1: back-fill rows 0 with 4.0
    assert out[0, 1] == 4.0
    # forward fill row 2 with 4.0
    assert out[2, 1] == 4.0


def test_fill_missing_y_keeps_unobserved_nan():
    """Cells with mask=False stay NaN even if neighbors are observed."""
    y = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]])
    msk = np.array([[True, True], [False, True], [False, False]])
    out = fill_missing_y(y, msk)
    # Row 1 col 0 unobserved → stays NaN
    assert np.isnan(out[1, 0])
    # Row 2 unobserved everywhere → NaN
    assert np.isnan(out[2, 0])
    assert np.isnan(out[2, 1])


def test_anomaly_report_runs(samples):
    rep = anomaly_report(samples[:10], z_threshold=5.0)
    assert set(rep.keys()) == {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
                                "y1", "y2", "y3", "y4"}
    for v in rep.values():
        assert isinstance(v, int) and v >= 0


def test_build_dataset_shape(samples):
    scalers = fit_scalers(samples)
    ds = build_dataset(samples, scalers, seq_len=64)
    assert ds["X"].shape == (len(samples), 64, 8)
    assert ds["Y"].shape == (len(samples), 64, 4)
    assert ds["Y_mask"].shape == (len(samples), 64, 4)
    assert ds["lengths"].shape == (len(samples),)
    assert ds["Y_final"].shape == (len(samples),)
    # All X entries should be finite (anomalies interpolated)
    assert np.isfinite(ds["X"]).all()
    assert ds["lengths"].min() > 0


def test_split_dataset_random(samples):
    scalers = fit_scalers(samples)
    ds = build_dataset(samples, scalers, seq_len=64)
    train, test = split_dataset(ds, ratio=0.8, seed=42)
    assert train["X"].shape[0] + test["X"].shape[0] == ds["X"].shape[0]
    n_train_expected = int(round(ds["X"].shape[0] * 0.8))
    assert train["X"].shape[0] == n_train_expected
    # Disjoint sets
    train_ids = set(train["file_ids"].tolist())
    test_ids = set(test["file_ids"].tolist())
    assert train_ids.isdisjoint(test_ids)