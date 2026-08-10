"""
Plot the time evolution of normalized x1..x8 (left) and y1..y4 (right).

Layout
------
- 5 rows × 2 columns: one row per group dir 1..5
  - left column  : x1..x8 (solid, dense)
  - right column : y1..y4 + Y (solid lines + small markers at each
                 valid y1..y4 measurement, since y rows are sparse;
                 Y is a horizontal segment at the file's final value)
- x-axis : time (hours, file-aligned at t=0)
- y-axis : z-score normalized value (mean/std computed over ALL exps, i.e.
           the same scaler used by train_forecaster.py)

Colors: categorical palette from the dataviz skill (validated CVD ΔE=9.1
and normal-vision ΔE=19.6 against the surface #fcfcfb).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_all, split_experiments, Scaler, YScaler, X_COLS, Y_INT_COLS,
)

BASE = "/kefu-nas/ybkong/time_serials-master"
OUT_DIR = os.path.join(BASE, "src/analysis_out")
GROUPS = ["1", "2", "3", "4", "5"]

# ---- categorical palette (dataviz skill reference instance, light mode) ----
PALETTE_LIGHT = [
    "#2a78d6",  # 1  blue
    "#eb6834",  # 2  orange
    "#1baf7a",  # 3  aqua
    "#eda100",  # 4  yellow
    "#e87ba4",  # 5  magenta
    "#008300",  # 6  green
    "#4a3aa7",  # 7  violet
    "#e34948",  # 8  red
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# y colors: reuse palette in a different order so the two panels don't
# share a hue at the same index.
COLORS_Y = [
    PALETTE_LIGHT[3],  # y1 yellow
    PALETTE_LIGHT[5],  # y2 green
    PALETTE_LIGHT[6],  # y3 violet
    PALETTE_LIGHT[1],  # y4 orange
    "#0b0b0b",          # Y   ink (final target, draw as horizontal segment)
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load + scaler ----
    exps = load_all(BASE)
    train_exps, _, _ = split_experiments(exps, seed=42)
    # fit on the TRAIN split (matches what train_forecaster.py does)
    x_scaler = Scaler.fit(train_exps)
    y_scaler = YScaler.fit(train_exps)

    # ---- per-experiment time series, file-aligned at t=0 ----
    # Each entry: {
    #   "name": str,
    #   "t":    (T,) hours,
    #   "x":    (T, 8) normalized,
    #   "y":    (T, 4) normalized (NaN where missing),
    #   "Y":    float (final value, raw scale),
    #   "Y_n":  float (normalized Y for plotting in right panel),
    # }
    series_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for e in exps:
        df = e.df.copy()
        if len(df) < 2:
            continue
        # time axis: 30-min timestep → t = idx * 0.5h
        t = np.arange(len(df)) * 0.5
        # x normalized
        x_raw = df[X_COLS].to_numpy(dtype=np.float32)
        x_norm = x_scaler.transform(x_raw)
        # y normalized (NaN preserved)
        y_raw = df[Y_INT_COLS].to_numpy(dtype=np.float32)
        y_norm = y_scaler.transform(y_raw)
        # Y (final target) — single value per file. Normalize for plotting
        Y_raw = e.Y if e.Y is not None else np.nan
        Y_norm = (Y_raw - y_scaler.means[:4]) / y_scaler.stds[:4]   # (4,)
        # use Y_norm[3] (y4 row) as the representative final target value;
        # if you want a different column, swap the index below
        Y_norm_repr = Y_norm[3] if np.isfinite(Y_norm[3]) else np.nan

        series_by_group[e.group].append({
            "name": os.path.splitext(os.path.basename(e.file))[0],
            "t": t,
            "x": x_norm,
            "y": y_norm,
            "Y_norm": Y_norm_repr,
        })

    # ---- figure: 5 rows × 2 cols (x | y) ----
    fig, axes = plt.subplots(
        5, 2, figsize=(15, 22),
        facecolor=SURFACE, sharex=False,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    fig.patch.set_facecolor(SURFACE)

    for gi, g in enumerate(GROUPS):
        ax_x, ax_y = axes[gi, 0], axes[gi, 1]
        ser_list = series_by_group[g]

        for ax in (ax_x, ax_y):
            ax.set_facecolor(SURFACE)

        if not ser_list:
            ax_x.set_title(f"dir {g}: (no data)", color=INK_PRIMARY)
            ax_y.set_title(f"dir {g}: (no data)", color=INK_PRIMARY)
            continue

        # panel-level y-limits derived from the data actually plotted
        t_max = max(s["t"][-1] for s in ser_list)

        # ---- LEFT: x1..x8 ----
        ax_x.set_xlim(0, max(t_max * 1.02, 1.0))
        ax_x.set_ylim(-5, 5)
        ax_x.axhline(0, color=INK_MUTED, lw=0.6, ls="--", alpha=0.6)
        ax_x.set_ylabel("Normalized value (z-score)", color=INK_PRIMARY, fontsize=9)
        ax_x.set_title(
            f"Group {g} — {len(ser_list)} files   x1..x8",
            color=INK_PRIMARY, fontsize=10, loc="left",
        )
        ax_x.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
        ax_x.tick_params(colors=INK_SECONDARY, labelsize=8)
        for s in ax_x.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)
        # x curves (dense)
        for s in ser_list:
            for ci in range(8):
                ax_x.plot(s["t"], s["x"][:, ci],
                           color=PALETTE_LIGHT[ci], lw=0.6, alpha=0.35)

        # legend on the LEFT panel (x)
        handles_x = [
            plt.Line2D([0], [0], color=PALETTE_LIGHT[i], lw=2, label=f"x{i+1}")
            for i in range(8)
        ]
        ax_x.legend(handles=handles_x, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), fontsize=6,
                    ncol=2, frameon=False, labelcolor=INK_PRIMARY)

        # ---- RIGHT: y1..y4 + Y ----
        ax_y.set_xlim(0, max(t_max * 1.02, 1.0))
        ax_y.set_ylim(-2, 2)
        ax_y.axhline(0, color=INK_MUTED, lw=0.6, ls="--", alpha=0.6)
        ax_y.set_ylabel("Normalized value (z-score)", color=INK_PRIMARY, fontsize=9)
        ax_y.set_title(
            f"Group {g} — y1..y4 + Y",
            color=INK_PRIMARY, fontsize=10, loc="left",
        )
        ax_y.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
        ax_y.tick_params(colors=INK_SECONDARY, labelsize=8)
        for s in ax_y.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)
        # y curves: each y1..y4 line connects all of its own valid points
        # across NaNs (so sparse measurements appear as one polyline per
        # series) and marks each valid point with a circle. NaN gaps are
        # bridged by linear interpolation so the line stays continuous,
        # but only between the first and last valid sample of each series.
        for s in ser_list:
            t = s["t"]
            for ci in range(4):
                y_col = s["y"][:, ci]
                mask = np.isfinite(y_col)
                if not mask.any():
                    continue
                xs = t[mask]
                ys = y_col[mask]
                # interpolate across NaNs between first and last valid point
                first, last = np.argmax(mask), len(mask) - 1 - np.argmax(mask[::-1])
                interp = np.interp(t[first:last + 1], xs, ys)
                # build the (t, y) array for plotting: real values at
                # valid samples, interpolated values in between
                plot_t = t[first:last + 1]
                plot_y = interp.copy()
                # restore true values at measured points so markers are
                # exact, and so the line passes through real data
                plot_y[mask[first:last + 1]] = y_col[first:last + 1][
                    mask[first:last + 1]
                ]
                # draw the continuous line (with interpolated fill between
                # first/last valid samples) — NO markers on this call, so
                # only real measurements get dots in the next call below
                ax_y.plot(plot_t, plot_y, color=COLORS_Y[ci], lw=1.0,
                          alpha=0.9, ls="-")
                # mark ONLY the originally valid points with circles
                ax_y.plot(xs, ys, color=COLORS_Y[ci], ls="none",
                          marker="o", markersize=3.0,
                          markerfacecolor=COLORS_Y[ci],
                          markeredgecolor=COLORS_Y[ci])
            # Y: horizontal segment at the file's final value, spanning t in
            # [t_end*0.9, t_end]; only plot if finite
            Yn = s["Y_norm"]
            if np.isfinite(Yn):
                ax_y.hlines(
                    y=Yn,
                    xmin=s["t"][-1] * 0.92, xmax=s["t"][-1],
                    color=COLORS_Y[4], lw=2.0, alpha=0.9,
                )

        # legend on the RIGHT panel (y)
        handles_y = [
            plt.Line2D([0], [0], color=COLORS_Y[i], lw=2, ls="-",
                       label=Y_INT_COLS[i])
            for i in range(4)
        ] + [
            plt.Line2D([0], [0], color=COLORS_Y[4], lw=2,
                       label="Y (final)"),
        ]
        ax_y.legend(handles=handles_y, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), fontsize=6,
                    ncol=2, frameon=False, labelcolor=INK_PRIMARY)

        ax_x.set_xlabel("Time (hours, file-aligned at t=0)", color=INK_PRIMARY, fontsize=9)
        ax_y.set_xlabel("Time (hours, file-aligned at t=0)", color=INK_PRIMARY, fontsize=9)

    fig.suptitle(
        "Time evolution of normalized x1..x8 (left) and y1..y4 + Y (right) — "
        "z-score (train split mean/std)",
        color=INK_PRIMARY, fontsize=12, y=0.997,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.996])
    out_path = os.path.join(OUT_DIR, "evolution_xy.png")
    plt.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
