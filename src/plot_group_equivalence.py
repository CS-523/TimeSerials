"""
============================================================
把跨 group 一致性分析的结果固化成图片
============================================================

读 `src/analysis_out/group_equivalence_summary.json`，画 5 张图到同目录：

  group_marginal_w1.png        L1 边缘: W₁/σ 两两对比 (8 x 10 热图)
  group_marginal_ks.png         L1 边缘: -log10(KS p) (8 x 10 热图)
  group_acf_overlay.png         L2 自相关: 5 组 x1..x8 的 ACF 叠加 (2×4 子图)
  group_mi_delta.png            L3 互信息: 5 组两两 MI 矩阵差异 (8×8 热图)
  group_dag_overlay.png         L4 Granger 因果: 5 张 8×8 邻接矩阵 + 一张对称差汇总

依赖：matplotlib + numpy
============================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 配 CJK 字体，否则中文标题/轴标签会变方框
for _font in ("WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC"):
    if any(_font in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font]
        break
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────────────────────
BASE = Path("/kefu-nas/ybkong/time_serials-master")
SRC = BASE / "src"
OUT = SRC / "analysis_out"
SUMMARY = OUT / "group_equivalence_summary.json"
sys.path.insert(0, str(SRC))

X_LABELS = [f"x{i}" for i in range(1, 9)]
X_LABELS_SHORT = [f"x{i}" for i in range(1, 9)]


def _load_summary() -> Dict:
    with open(SUMMARY) as f:
        return json.load(f)


def _parse_pair(pair_str: str) -> tuple:
    a, b = pair_str.split("|")
    return a, b


def _to_float(x) -> float:
    """Robust float coercion (raise on None / NaN string)."""
    return float(x)


def _grid_keys(d: Dict[str, float]) -> List[str]:
    """Return keys sorted: outer index → inner index → pair-key string."""
    keys = list(d.keys())
    return keys


# ═════════════════════════════════════════════════════════════
# 1. L1 边缘：W₁ / σ 热图
# ═════════════════════════════════════════════════════════════
def plot_L1_w1(summary: Dict) -> None:
    groups = summary["groups"]
    w1 = summary["L1_edge"]["w1"]          # { "0": { "1|2": v, ... }, "1": ..., ... }
    n_x = len(X_LABELS)
    pairs = []
    for i in range(n_x):
        for k, v in w1[str(i)].items():
            pairs.append((i, k, v))
    pairs_sorted = sorted(pairs, key=lambda t: (t[0], t[1]))

    # Build 8 x (5 choose 2) matrix
    pair_labels = sorted(set(k for i, k, _ in pairs_sorted),
                          key=lambda p: (int(p.split("|")[0]), int(p.split("|")[1])))
    mat = np.full((n_x, len(pair_labels)), np.nan)
    for i, k, v in pairs_sorted:
        j = pair_labels.index(k)
        mat[i, j] = _to_float(v)

    fig, ax = plt.subplots(figsize=(1.0 * len(pair_labels) + 2, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.2)
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_x))
    ax.set_yticklabels(X_LABELS)
    ax.set_xlabel("group pair")
    ax.set_title("L1 W₁/σ_pooled (z-score 空间)\n绿=组间几乎无差异，红=差异大  ·  阈值 0.05")
    for i in range(n_x):
        for j in range(len(pair_labels)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 0.1 else "white")
    plt.colorbar(im, ax=ax, shrink=0.75, label="W₁ / σ")
    plt.tight_layout()
    out = OUT / "group_marginal_w1.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


# ═════════════════════════════════════════════════════════════
# 2. L1 边缘：KS p-value 热图（取 -log10）
# ═════════════════════════════════════════════════════════════
def plot_L1_ks(summary: Dict, alpha_bonf: float = 0.05 / 80) -> None:
    groups = summary["groups"]
    ks = summary["L1_edge"]["ks_p"]
    n_x = len(X_LABELS)
    pairs = []
    for i in range(n_x):
        for k, v in ks[str(i)].items():
            pairs.append((i, k, _to_float(v)))
    pairs_sorted = sorted(pairs, key=lambda t: (t[0], t[1]))
    pair_labels = sorted(set(k for _, k, _ in pairs_sorted),
                          key=lambda p: (int(p.split("|")[0]), int(p.split("|")[1])))
    mat = np.full((n_x, len(pair_labels)), np.nan)
    for i, k, v in pairs_sorted:
        j = pair_labels.index(k)
        mat[i, j] = -np.log10(max(v, 1e-300))

    fig, ax = plt.subplots(figsize=(1.0 * len(pair_labels) + 2, 5))
    im = ax.imshow(mat, aspect="auto", cmap="Reds",
                   vmin=0, vmax=max(-np.log10(alpha_bonf), 1.0))
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_x))
    ax.set_yticklabels(X_LABELS)
    ax.set_xlabel("group pair")
    ax.set_title(f"L1 KS 检验 −log10(p)（z-score 空间）\n越红 = 拒绝'同分布'越强  ·  Bonferroni α = {alpha_bonf:.2e}")
    for i in range(n_x):
        for j in range(len(pair_labels)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.75, label="−log10(p)")
    plt.tight_layout()
    out = OUT / "group_marginal_ks.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


# ═════════════════════════════════════════════════════════════
# 3. L2 自相关：5 组 x1..x8 的 ACF 叠加
# ═════════════════════════════════════════════════════════════
def plot_L2_acf(summary: Dict) -> None:
    """We need raw x data again — load from data folder."""
    from data_loader import load_all, X_COLS
    assert X_COLS == [f"x{i}" for i in range(1, 9)]
    exps = load_all(str(BASE))
    by_group: Dict[str, List[np.ndarray]] = {}
    for e in exps:
        arr = e.df[X_COLS].to_numpy(dtype=np.float32)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) > 0:
            by_group.setdefault(e.group, []).append(arr)
    by_group = {g: np.concatenate(vs, axis=0) for g, vs in by_group.items()}

    groups = summary["groups"]
    nlags = 30
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharex=True)
    for i, xname in enumerate(X_LABELS):
        ax = axes[i // 4, i % 4]
        for g in groups:
            acf = _acf_1d(by_group[g][:, i], nlags=nlags)
            ax.plot(range(1, nlags + 1), acf[1:],
                     marker="o" if i == 0 else None, linewidth=1.4,
                     label=f"group {g}")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1.96 / np.sqrt(len(by_group[groups[0]])), color="gray",
                   lw=0.5, linestyle="--")
        ax.axhline(-1.96 / np.sqrt(len(by_group[groups[0]])), color="gray",
                   lw=0.5, linestyle="--")
        acf_l2_for_x = summary["L2_autocorr"]["acf_l2"][str(i)]
        max_l2_for_x = max(acf_l2_for_x.values()) if acf_l2_for_x else 0.0
        ax.set_title(f"{xname}  (max ACF L₂ = {max_l2_for_x:.3f})")
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorr")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("L2 自相关结构 — 5 组叠加  ·  阈值 ACF L₂ < 0.05 → 节奏应重合", y=1.02)
    plt.tight_layout()
    out = OUT / "group_acf_overlay.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def _acf_1d(x: np.ndarray, nlags: int = 30) -> np.ndarray:
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    out = np.array([np.dot(x[: len(x) - k], x[k:]) / var for k in range(nlags + 1)])
    return out


# ═════════════════════════════════════════════════════════════
# 4. L3 互信息：pair-wise mean |ΔMI| 热图
# ═════════════════════════════════════════════════════════════
def plot_L3_mi_delta(summary: Dict) -> None:
    mi = summary["L3_mutual_info"]["mi_diff"]   # { "i|j": mean |ΔMI| }
    n_x = len(X_LABELS)
    mat = np.zeros((n_x, n_x))
    pair_lookup: Dict[tuple, float] = {}
    for k, v in mi.items():
        i, j = k.split("|")
        pair_lookup[(int(i), int(j))] = _to_float(v)
        pair_lookup[(int(j), int(i))] = _to_float(v)

    for i in range(n_x):
        for j in range(n_x):
            if i == j:
                mat[i, j] = 0
            elif (i, j) in pair_lookup:
                mat[i, j] = pair_lookup[(i, j)]
            else:
                mat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=0.5)
    ax.set_xticks(range(n_x))
    ax.set_xticklabels(X_LABELS)
    ax.set_yticks(range(n_x))
    ax.set_yticklabels(X_LABELS)
    ax.set_title("L3 互信息 |ΔMI|  (5 组两两 MI 矩阵差异均值)\n"
                  "白 = 同组，色越深 = 5 组对 (x_i, x_j) 耦合差异越大  ·  阈值 0.01 比特")
    for i in range(n_x):
        for j in range(n_x):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.25 else "black")
    plt.colorbar(im, ax=ax, shrink=0.85, label="|ΔMI|  (bit)")
    plt.tight_layout()
    out = OUT / "group_mi_delta.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


# ═════════════════════════════════════════════════════════════
# 5. L4 Granger 因果：5 张 8×8 邻接矩阵 + 一张对称差
# ═════════════════════════════════════════════════════════════
def plot_L4_dag(summary: Dict) -> None:
    """Re-derive DAGs from raw data (summary doesn't store the matrices)."""
    from data_loader import load_all, X_COLS
    from statsmodels.tsa.api import VAR

    exps = load_all(str(BASE))
    by_group: Dict[str, np.ndarray] = {}
    for e in exps:
        arr = e.df[X_COLS].to_numpy(dtype=np.float32)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) > 0:
            by_group.setdefault(e.group, []).append(arr)
    by_group = {g: np.concatenate(vs, axis=0) for g, vs in by_group.items()}

    groups = summary["groups"]
    max_lag = 3
    p_threshold = 0.05
    dags: Dict[str, np.ndarray] = {}
    for g, X in by_group.items():
        try:
            res = VAR(X).fit(maxlags=max_lag)
            params = np.asarray(res.params)
            pvals   = np.asarray(res.pvalues)
            exog    = list(res.model.exog_names)
            n_x = X.shape[1]
            mat = np.zeros((n_x, n_x), dtype=int)
            for lag in range(1, max_lag + 1):
                for j in range(n_x):
                    name = f"L{lag}.y{j+1}"
                    if name not in exog:
                        continue
                    row = exog.index(name)
                    for i in range(n_x):
                        if pvals[row, i] < p_threshold and abs(params[row, i]) > 1e-6:
                            mat[i, j] = 1
        except Exception:
            mat = np.zeros((X.shape[1], X.shape[1]), dtype=int)
        dags[g] = mat

    # 5 张 DAG + 1 张对称差汇总
    n_x = len(X_LABELS)
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 4),
                              sharey=True)
    sym = np.zeros((n_x, n_x), dtype=int)
    for ax, g in zip(axes, groups):
        m = dags[g]
        sym |= m
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(X_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_x))
        ax.set_yticklabels(X_LABELS, fontsize=8)
        ax.set_title(f"group {g}  (edges = {int(m.sum())})")
    # 对称差 = 出现 > 1 次的边
    fig.suptitle("L4 Granger 因果图  (5 张邻接矩阵)  ·  threshold p < 0.05, max_lag = 3",
                 y=1.02)
    plt.tight_layout()
    out1 = OUT / "group_dag_overlay.png"
    plt.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"[plot] {out1}")

    # 对称差热图
    # "出现在 ≥ 1 张 DAG 上的边" — 用颜色标"出现频次"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sym, cmap="viridis", vmin=0, vmax=len(groups))
    ax.set_xticks(range(n_x))
    ax.set_xticklabels(X_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_x))
    ax.set_yticklabels(X_LABELS, fontsize=9)
    ax.set_title(f"L4 Granger 因果边出现频次（5 张图叠加）\n"
                  f"颜色 = 该边在 5 个 group 的因果图中出现多少次  ·  "
                  f"max = {int(sym.max())}/{len(groups)}")
    for i in range(n_x):
        for j in range(n_x):
            v = int(sym[i, j])
            if v > 0:
                ax.text(j, i, f"{v}", ha="center", va="center",
                        fontsize=10, color="white" if v < len(groups) / 2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.85, label="出现次数 (0~5)")
    plt.tight_layout()
    out2 = OUT / "group_dag_overlay_count.png"
    plt.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"[plot] {out2}")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
def main() -> None:
    if not SUMMARY.exists():
        print(f"[plot] 缺 {SUMMARY}；先跑 `python analyze_group_equivalence.py`")
        return
    summary = _load_summary()
    print(f"[plot] 读了 {summary['groups']} 5 组数据")
    plot_L1_w1(summary)
    plot_L1_ks(summary)
    plot_L2_acf(summary)
    plot_L3_mi_delta(summary)
    plot_L4_dag(summary)
    print(f"[plot] 全部 6 张图已写入 {OUT}/")


if __name__ == "__main__":
    main()
