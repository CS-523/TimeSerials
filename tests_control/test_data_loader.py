"""Tests for ``src_control.data_loader``."""
from __future__ import annotations

import numpy as np
import pytest

from src_control.data_loader import (
    discover_csvs,
    parse_all,
    parse_sample,
    Sample,
    X_COLS,
    Y_COLS,
)


ROOT = "/kefu-nas/ybkong/time_serials-master"


def test_discover_csvs():
    paths = discover_csvs(ROOT)
    assert len(paths) == 171
    # Sorted: should begin with subdir 1 and end with subdir 5
    assert str(paths[0]).startswith(f"{ROOT}/1/")
    assert str(paths[-1]).startswith(f"{ROOT}/5/")
    assert paths == sorted(paths)


@pytest.mark.parametrize("subdir,fid", [(1, "230276"), (1, "230296"), (5, "240349")])
def test_parse_sample_basic(subdir, fid):
    path = f"{ROOT}/{subdir}/{fid}.csv"
    s = parse_sample(path)
    assert isinstance(s, Sample)
    assert s.subdir == subdir
    assert s.file_id == fid

    # Shape sanity
    assert s.x.ndim == 2 and s.x.shape[1] == 8
    assert s.y.shape == (s.x.shape[0], 4)
    assert s.y_present_mask.shape == (s.x.shape[0], 4)
    assert s.cycle.shape == (s.x.shape[0],)
    assert s.T > 0
    assert np.isfinite(s.Y)

    # Cycles should be non-decreasing (within file) and integer
    if s.T >= 2:
        diffs = np.diff(s.cycle)
        assert (diffs >= 0).all(), f"cycle not monotonic in {fid}: {s.cycle[:20]}"

    # Each x-row carries at most one y observation (mask <= 1)
    # Actually each row may have multiple y values measured together, so mask
    # is bool and any row may have 0..4 trues.


def test_y_alignment_to_x():
    """y values present in mask should align with non-NaN y entries."""
    s = parse_sample(f"{ROOT}/1/230276.csv")
    mask_present = s.y_present_mask.sum(axis=0)
    nan_count = np.isnan(s.y).sum(axis=0)
    # NaN count and False count should match per column
    assert (mask_present + nan_count == s.T).all()


def test_all_y_columns_observed_at_least_somewhere():
    samples, _ = parse_all(ROOT)
    for j, name in enumerate(Y_COLS):
        total_obs = sum(s.y_present_mask[:, j].sum() for s in samples)
        assert total_obs > 0, f"{name} observed nowhere"


def test_no_sample_misses_final_y():
    samples, _ = parse_all(ROOT)
    for s in samples:
        assert np.isfinite(s.Y), f"{s.file_id} missing Y"