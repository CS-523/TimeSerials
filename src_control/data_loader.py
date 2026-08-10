"""CSV discovery and canonical parsing for the time_serials dataset.

Each CSV holds a single experiment with the schema::

    datime, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, 周期, Y

Two row patterns repeat per ``周期``:

* **x-row** — ``datime`` ends with ``:00:00.000`` or ``:30:00.000``; all 8 x
  columns filled, y columns blank, ``周期`` matches the carrying cycle index.
* **boundary row** — ``datime`` ends with ``:45:12.530``; x columns blank,
  y columns may be populated; ``周期`` has been incremented (so it tags the
  *next* cycle).

The final row has ``datime == 'NaT'``, all fields blank except the ``Y`` value.

Per the README, y1–y4 are aligned to the most recent x-row (i.e., the
measurement is "carried by" the prior 30-min x-row as its context).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# NOTE: row classification is driven by **content**, not by ``datime`` shape:
#   * x-row     : at least one x column is non-NaN.
#   * boundary  : all x columns are NaN (the row's job is to carry the next
#     cycle index and possibly y measurements).
# This is robust to the wide variety of ``datime`` formats across files.
_BOUNDARY_TIME_RE = None  # retained for backwards-compat import; unused.

X_COLS: Tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
Y_COLS: Tuple[str, ...] = ("y1", "y2", "y3", "y4")


@dataclass
class Sample:
    """Canonical parsed representation of a single experiment."""
    file_id: str
    subdir: int
    x: np.ndarray            # (T, 8) dense x rows, NaN for missing
    y: np.ndarray            # (T, 4) y aligned to x rows (carried), NaN where unmeasured
    y_present_mask: np.ndarray  # (T, 4) bool — observed at this x-row
    cycle: np.ndarray        # (T,) cycle index per x-row
    Y: float                 # final outcome
    datime_first: pd.Timestamp
    datime_last: pd.Timestamp

    @property
    def T(self) -> int:
        return int(self.x.shape[0])

    def summary(self) -> dict:
        return {
            "file_id": self.file_id,
            "subdir": self.subdir,
            "T": self.T,
            "n_cycles": int(np.nanmax(self.cycle) + 1) if self.T else 0,
            "Y": float(self.Y),
            "datime_first": str(self.datime_first),
            "datime_last": str(self.datime_last),
            "y_observed": {
                col: int(self.y_present_mask[:, j].sum())
                for j, col in enumerate(Y_COLS)
            },
        }


def discover_csvs(root: str | Path) -> List[Path]:
    """Return sorted list of all CSVs under ``root/{1,2,3,4,5}/*.csv``."""
    root = Path(root)
    paths: List[Path] = []
    for sub in (1, 2, 3, 4, 5):
        sub_dir = root / str(sub)
        if not sub_dir.is_dir():
            continue
        paths.extend(sorted(sub_dir.glob("*.csv")))
    return paths


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to float, preserving NaN for blanks."""
    return pd.to_numeric(series, errors="coerce").astype(np.float64)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a single CSV with proper typing.

    * ``datime`` parsed with ``errors='coerce'`` so the final ``NaT`` row survives.
    * All x/y/周期/Y columns coerced to float.
    * Blank cells become NaN.
    """
    df = pd.read_csv(path, na_values=["", "NaN", "nan", None])
    df.columns = [c.strip() for c in df.columns]

    if "datime" in df.columns:
        df["datime"] = pd.to_datetime(df["datime"], errors="coerce")

    for col in X_COLS + Y_COLS + ("周期", "Y"):
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    return df


def _row_is_x_row(row: pd.Series) -> bool:
    """An x-row has at least one x column filled."""
    return any(pd.notna(row.get(c)) for c in X_COLS)


def _row_is_boundary(row: pd.Series) -> bool:
    """A boundary row has all x columns blank (and may carry y + the next cycle)."""
    return all(pd.isna(row.get(c)) for c in X_COLS) and not pd.isna(row.get("datime"))


def parse_sample(path: str | Path) -> Sample:
    """Parse one CSV into a :class:`Sample`.

    Algorithm:
        1. Iterate rows in order. Classify each row as **x-row** (any x column
           filled) or **boundary row** (all x blank, datime present). The
           classification uses **content** rather than ``datime`` shape, since
           the boundary timestamp varies across files (e.g. ``08:45:12.530``,
           ``19:11:00``, ``23:50:00``, ...).
        2. Track the *carrying* ``周期`` index (updated whenever a boundary
           row is encountered — its 周期 belongs to the upcoming x-rows).
        3. For x-rows: push x values; if a pending y (from the most recent
           boundary row) exists, attach it to this x-row as the carrying
           measurement, then clear pending.
        4. For boundary rows: hold y values until the next x-row arrives; if
           no further x-row appears, attach to the last x-row.
        5. Final NaT row contributes the Y value.
    """
    path = Path(path)
    file_id = path.stem
    subdir = int(path.parent.name)

    df = load_csv(path)

    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    mask_list: List[List[bool]] = []
    cycle_list: List[int] = []

    pending_y: List[float] = [np.nan] * 4
    pending_cycle: int = 0
    has_pending = False

    datime_first = pd.NaT
    datime_last = pd.NaT

    Y_final = float("nan")

    for _, row in df.iterrows():
        ts = row.get("datime")

        if pd.isna(ts):
            # Final Y row (NaT, everything else blank).
            y_val = row.get("Y")
            if pd.notna(y_val):
                Y_final = float(y_val)
            continue

        # Track first / last real timestamps
        if pd.isna(datime_first):
            datime_first = ts
        datime_last = ts

        if _row_is_x_row(row):
            x_vals = [row.get(c, np.nan) for c in X_COLS]
            x_list.append([float(v) if pd.notna(v) else np.nan for v in x_vals])

            if has_pending:
                y_list.append(list(pending_y))
                mask_list.append([bool(pd.notna(v)) for v in pending_y])
                cycle_list.append(int(pending_cycle))
                has_pending = False
            else:
                y_list.append([np.nan] * 4)
                mask_list.append([False] * 4)
                cycle_list.append(int(pending_cycle))

        elif _row_is_boundary(row):
            new_y = [row.get(c, np.nan) for c in Y_COLS]
            new_cycle = row.get("周期")
            new_cycle = int(new_cycle) if pd.notna(new_cycle) else pending_cycle + 1
            pending_cycle = new_cycle
            pending_y = [float(v) if pd.notna(v) else np.nan for v in new_y]
            has_pending = True

        else:
            # Unrecognized row — silently skip
            continue

    # If pending y remained at EOF (boundary row without following x-row),
    # attach it to the last x-row.
    if has_pending and x_list:
        y_list[-1] = list(pending_y)
        mask_list[-1] = [bool(pd.notna(v)) for v in pending_y]
        cycle_list[-1] = int(pending_cycle)

    x_arr = np.asarray(x_list, dtype=np.float64) if x_list else np.zeros((0, 8))
    y_arr = np.asarray(y_list, dtype=np.float64) if y_list else np.zeros((0, 4))
    mask_arr = np.asarray(mask_list, dtype=bool) if mask_list else np.zeros((0, 4), dtype=bool)
    cycle_arr = np.asarray(cycle_list, dtype=np.int64) if cycle_list else np.zeros((0,), dtype=np.int64)

    return Sample(
        file_id=file_id,
        subdir=subdir,
        x=x_arr,
        y=y_arr,
        y_present_mask=mask_arr,
        cycle=cycle_arr,
        Y=Y_final,
        datime_first=datime_first,
        datime_last=datime_last,
    )


def parse_all(root: str | Path) -> Tuple[List[Sample], List[Path]]:
    """Parse every CSV under root/{1..5}. Returns (samples, paths)."""
    paths = discover_csvs(root)
    samples = [parse_sample(p) for p in paths]
    return samples, paths


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sanity-check CSV parsing.")
    parser.add_argument("--root", default="/kefu-nas/ybkong/time_serials-master")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    samples, paths = parse_all(args.root)
    print(f"Discovered {len(paths)} CSVs")
    for s in samples[: args.limit]:
        print(json.dumps(s.summary(), indent=2, default=str))
        print("---")