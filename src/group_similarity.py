"""
group_similarity.py
==================
量化"5 个 group 的演化规律有多相似"，以证明可以共用一个模型。

核心思路：
  1. 把每条实验（长度不一）通过线性重采样 + z-score 归一化成 100 点轨迹
  2. 在每个 (变量, 归一化时刻) 上做"组内 vs 组间"对比：
       - Fréchet / DTW 距离   → 衡量形状相似度
       - Wasserstein / KS 检验 → 衡量分布相似度
       - 变异系数 (CV) 对比    → 衡量波动幅度
       - 线性回归斜率对比     → 衡量单调趋势
  3. 输出：
       - 每对的相似度矩阵（csv）
       - 每对的 Fréchet & Wasserstein 距离图
       - 汇总结论：5 组能否共用模型

用法：
  python group_similarity.py
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS, Y_INT_COLS  # noqa: E402

warnings.filterwarnings("ignore")

BASE = "/kefu-nas/ybkong/time_serials-master"
OUT_DIR = os.path.join(BASE, "src/analysis_out")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = ["1", "2", "3", "4", "5"]
N_INTERP = 100  # 归一化长度


# ----------------------------- 1. 数据准备 -----------------------------
def resample_to_grid(series: np.ndarray, n_points: int = N_INTERP) -> np.ndarray | None:
    """
    把任意长度的 1-D 序列**重采样**到 n_points 个等距点（不做标准化）。
    用于"绝对轨迹"对齐；返回原始量级，NaN 序列返回 None。
    """
    s = np.asarray(series, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 3:
        return None
    if np.std(s) < 1e-9:
        return None
    x_old = np.linspace(0.0, 1.0, len(s))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, s)


def resample_and_normalize(series: np.ndarray, n_points: int = N_INTERP,
                            global_mean: float | None = None,
                            global_std: float | None = None) -> np.ndarray | None:
    """
    把任意长度的 1-D 序列重采样到 n_points，并做 z-score 标准化。
    两种用法：
      1) 都不传 global_* → 用**该序列自己**的 mean/std 归一化（形状相似度）
      2) 传 global_mean/global_std → 用跨组共享的 mean/std 归一化（绝对轨迹相似度）
    返回归一化后的轨迹；序列太短或全常数则返回 None。
    """
    interp = resample_to_grid(series, n_points)
    if interp is None:
        return None
    if global_mean is None:
        mu, sd = interp.mean(), interp.std()
    else:
        mu, sd = global_mean, global_std
    if sd < 1e-9:
        return None
    return (interp - mu) / sd


def compute_global_scalers(exps, var_list):
    """
    对每个变量，跨所有实验+所有 group 计算一次 mean/std。
    用这个做 z-score，所有轨迹就处于同一坐标系下。
    返回 {var: (mean, std)}。
    """
    out = {}
    for v in var_list:
        pool = []
        for e in exps:
            if v not in e.df.columns:
                continue
            s = e.df[v].to_numpy(dtype=np.float64)
            s = s[~np.isnan(s)]
            if len(s):
                pool.append(s)
        if not pool:
            out[v] = (0.0, 1.0)
            continue
        all_vals = np.concatenate(pool)
        out[v] = (float(all_vals.mean()), float(all_vals.std() + 1e-9))
    return out


def build_trajectories(exps, var_list, normalize=True,
                        global_scalers: dict | None = None):
    """
    对每个实验、每个变量，重采样并(可选地)归一化成 (n_exps, n_points) 数组。
    - normalize=True  且 global_scalers=None : 每条序列各自 z-score（形状版）
    - normalize=True  且 global_scalers 给定  : 用跨组共享 mean/std 做 z-score（绝对版）
    - normalize=False : 仅重采样不标准化（用于反复算 scaler 的中间步骤）
    """
    by_group = {g: {v: [] for v in var_list} for g in GROUPS}
    for e in exps:
        for v in var_list:
            if v not in e.df.columns:
                continue
            raw = e.df[v].to_numpy(dtype=np.float64)
            if not normalize:
                tr = resample_to_grid(raw)
            elif global_scalers is None:
                tr = resample_and_normalize(raw)
            else:
                gm, gs = global_scalers[v]
                tr = resample_and_normalize(raw, global_mean=gm, global_std=gs)
            if tr is not None:
                by_group[e.group][v].append(tr)
    # 转 numpy 数组
    for g in GROUPS:
        for v in var_list:
            by_group[g][v] = np.asarray(by_group[g][v])  # shape: (n_exp, N_INTERP)
    return by_group


# ----------------------------- 2. 形状相似度 -----------------------------
def frechet_distance(P, Q):
    """
    离散 Fréchet 距离。P, Q: shape (n,)、(p,)。
    用 scipy.spatial.distance.cdist 一次算全部成对距离，再走 O(np) 递推。
    """
    from scipy.spatial.distance import cdist
    n, p = P.shape[0], Q.shape[0]
    # 全部成对距离 matrix (n, p)，避免在递推里反复调 np.linalg.norm
    # cdist 强制要求 2-D，所以把单变量曲线压成 (n, 1)
    D = cdist(P.reshape(-1, 1), Q.reshape(-1, 1), metric="euclidean")
    ca = np.full((n, p), -1.0)
    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = D[i, j]
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        return ca[i, j]
    return c(n - 1, p - 1)


def _all_within_group_frechet(arr):
    """一个 group 内的所有 (i<j) 配对 Fréchet 距离，返回均值；序列不足 2 条返回 NaN。"""
    if len(arr) < 2:
        return np.nan
    # 向量化配对：把 (n, T) 拉成 (n_pairs, T)
    n = len(arr)
    idx_i, idx_j = np.triu_indices(n, k=1)
    if len(idx_i) == 0:
        return np.nan
    # cdist 一次算所有配对的成对距离矩阵
    from scipy.spatial.distance import cdist
    D = cdist(arr[idx_i], arr[idx_j])
    # 然后对每对做 Fréchet 递推
    return float(np.mean([frechet_distance(arr[i], arr[j]) for i, j in zip(idx_i, idx_j)]))


def _cross_group_frechet(a, b, max_pairs=2000):
    """
    跨组 Fréchet：组 A 的 n_a 条 vs 组 B 的 n_b 条，最多 max_pairs 对随机采样后取均值。
    max_pairs 默认 2000 在保统计意义的前提下大幅降低耗时。
    """
    if len(a) == 0 or len(b) == 0:
        return np.nan
    from scipy.spatial.distance import cdist
    n_pairs = min(len(a) * len(b), max_pairs)
    if n_pairs < len(a) * len(b):
        rng = np.random.default_rng(42)
        ai = rng.integers(0, len(a), size=n_pairs)
        bj = rng.integers(0, len(b), size=n_pairs)
    else:
        ai, bj = np.indices((len(a), len(b)))
        ai, bj = ai.ravel(), bj.ravel()
    # 这次要算的轨迹数量可能很大，所以**预先**算好每条轨迹与对方所有轨迹的
    # 配对距离矩阵，再分批走 Fréchet
    vals = []
    for i, j in zip(ai, bj):
        vals.append(frechet_distance(a[i], b[j]))
    return float(np.mean(vals))


def pairwise_frechet(traj_dict, var, max_cross_pairs=2000):
    """
    对每个变量，计算 group 内/间的平均 Fréchet 距离。
    返回: (group_pair_dist, within_dist_per_group, between_avg)
        - group_pair_dist[(gi, gj)] : 组 i vs 组 j 的平均 Fréchet
        - within_dist_per_group[g] : 组内实验两两的 Fréchet 均值
        - between_avg              : 所有组间均值的总平均
    """
    within = {}
    for g in GROUPS:
        arr = traj_dict[g][var]
        within[g] = _all_within_group_frechet(arr)

    pair = {}
    for i, gi in enumerate(GROUPS):
        for j, gj in enumerate(GROUPS):
            if j <= i:
                continue
            pair[(gi, gj)] = _cross_group_frechet(
                traj_dict[gi][var], traj_dict[gj][var], max_pairs=max_cross_pairs
            )

    # 把 (gi, gj) 与 (gj, gi) 统一成同一对
    pair_sym = {}
    for (gi, gj), d in pair.items():
        key = tuple(sorted((gi, gj)))
        pair_sym[key] = d

    valid = [v for v in pair_sym.values() if np.isfinite(v)]
    between_avg = float(np.mean(valid)) if valid else np.nan

    return pair_sym, within, between_avg

    # 把 (gi, gj) 与 (gj, gi) 统一成同一对
    pair_sym = {}
    for (gi, gj), d in pair.items():
        key = tuple(sorted((gi, gj)))
        pair_sym[key] = d

    valid = [v for v in pair_sym.values() if np.isfinite(v)]
    between_avg = float(np.mean(valid)) if valid else np.nan

    return pair_sym, within, between_avg


# ----------------------------- 3. 分布相似度 -----------------------------
def wasserstein_1d(a, b):
    """1-D Wasserstein 距离 (经验 CDF)。"""
    a = np.sort(a); b = np.sort(b)
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        # 简单实现：合并排序后逐点计算 |CDF_a - CDF_b|
        all_pts = np.concatenate([a, b])
        all_pts.sort()
        cdf_a = np.searchsorted(a, all_pts, side="right") / len(a)
        cdf_b = np.searchsorted(b, all_pts, side="right") / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_pvalue(a, b):
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).pvalue)
    except Exception:
        return np.nan


def pairwise_wasserstein(traj_dict, var, timepoints=None):
    """
    对每个 (变量, 归一化时刻), 在两个 group 的取值分布上做 Wasserstein + KS。
    返回 dict[(gi, gj)] = {wasserstein_mean, ks_pvalue_median}
    """
    if timepoints is None:
        timepoints = np.arange(0, N_INTERP, 5)  # 0,5,10,...,95

    out = {tuple(sorted((gi, gj))): {"ws": [], "ks": []}
           for gi in GROUPS for gj in GROUPS if gi < gj}

    for t in timepoints:
        for i, gi in enumerate(GROUPS):
            for j, gj in enumerate(GROUPS):
                if j <= i:
                    continue
                key = tuple(sorted((gi, gj)))
                a = traj_dict[gi][var][:, t]
                b = traj_dict[gj][var][:, t]
                if len(a) < 2 or len(b) < 2:
                    continue
                out[key]["ws"].append(wasserstein_1d(a, b))
                out[key]["ks"].append(ks_pvalue(a, b))

    # 汇总
    summary = {}
    for k, v in out.items():
        ws = np.asarray(v["ws"])
        ks = np.asarray(v["ks"])
        ws = ws[np.isfinite(ws)]
        ks = ks[np.isfinite(ks)]
        summary[k] = {
            "ws_mean": float(np.mean(ws)) if len(ws) else np.nan,
            "ks_p_median": float(np.median(ks)) if len(ks) else np.nan,
            "ks_p_min": float(np.min(ks)) if len(ks) else np.nan,
        }
    return summary


# ----------------------------- 4. 波动幅度 / 斜率对比 -----------------------------
def cv_of_group(traj_dict, var):
    """变异系数：组内每条轨迹 std / |mean| 的均值。"""
    out = {}
    for g in GROUPS:
        arr = traj_dict[g][var]
        if len(arr) == 0:
            out[g] = np.nan; continue
        cvs = []
        for tr in arr:
            mu = np.mean(tr)
            sd = np.std(tr)
            if abs(mu) > 1e-6:
                cvs.append(abs(sd / mu))
        out[g] = float(np.mean(cvs)) if cvs else np.nan
    return out


def slope_of_group(traj_dict, var):
    """每个时间段的平均斜率 (终点-起点)/N，组内再取平均。"""
    out = {}
    for g in GROUPS:
        arr = traj_dict[g][var]
        if len(arr) == 0:
            out[g] = np.nan; continue
        sl = []
        for tr in arr:
            sl.append((tr[-1] - tr[0]) / (len(tr) - 1))
        out[g] = float(np.mean(sl)) if sl else np.nan
    return out


# ----------------------------- 5. 主流程 -----------------------------
def main():
    print("加载数据 ...")
    exps = load_all(BASE)
    print(f"  实验总数 = {len(exps)}")
    for g in GROUPS:
        n = sum(1 for e in exps if e.group == g)
        print(f"  group {g}: {n} 条")

    # 用所有变量
    var_list = X_COLS + Y_INT_COLS

    # === 跨组跨实验的全局 scaler（用于"绝对轨迹"比较）===
    print("\n构建归一化轨迹（两套 z-score）...")
    global_scalers = compute_global_scalers(exps, var_list)

    # 形状版：每条序列各自 z-score（消除量级差异，看形状）
    # 绝对版：用跨组共享 mean/std 做 z-score（看绝对轨迹是否落在同一坐标系）
    traj_shape = build_trajectories(exps, var_list, normalize=True,
                                    global_scalers=None)
    traj_abs   = build_trajectories(exps, var_list, normalize=True,
                                    global_scalers=global_scalers)

    # === 对每套（shape / absolute）算指标 ===
    summaries = {}
    for tag, traj in [("shape", traj_shape), ("absolute", traj_abs)]:
        print(f"\n[1/3-{tag}] 计算指标 ...")
        frechet_table = []
        ws_table = []
        cv_table = []
        slope_table = []

        for var in var_list:
            pair_sym, within, between = pairwise_frechet(traj, var)
            ws_summary = pairwise_wasserstein(traj, var)

            for (gi, gj), d in pair_sym.items():
                wg_i = within.get(gi, np.nan)
                wg_j = within.get(gj, np.nan)
                ratio = d / np.nanmean([wg_i, wg_j]) if (
                    np.isfinite(d) and np.isfinite(np.nanmean([wg_i, wg_j])) and np.nanmean([wg_i, wg_j]) > 0
                ) else np.nan
                frechet_table.append({
                    "variable": var, "group_i": gi, "group_j": gj,
                    "frechet_between": d,
                    "frechet_within_gi": wg_i,
                    "frechet_within_gj": wg_j,
                    "between_over_within": ratio,
                })

            for (gi, gj), d in ws_summary.items():
                ws_table.append({
                    "variable": var, "group_i": gi, "group_j": gj,
                    "ws_mean": d["ws_mean"],
                    "ks_p_median": d["ks_p_median"],
                    "ks_p_min": d["ks_p_min"],
                })

            cvs = cv_of_group(traj, var)
            sls = slope_of_group(traj, var)
            for g in GROUPS:
                cv_table.append({"variable": var, "group": g, "cv": cvs.get(g, np.nan)})
                slope_table.append({"variable": var, "group": g, "mean_slope": sls.get(g, np.nan)})

            bw_vals = [v for v in within.values() if np.isfinite(v)]
            msg = f"  {tag} | {var}: between/within"
            if np.isfinite(between) and bw_vals:
                msg += f" = {between / np.mean(bw_vals):.2f}"
            else:
                msg += " = skipped (insufficient data)"
            print(msg)

        pd.DataFrame(frechet_table).to_csv(os.path.join(OUT_DIR, f"group_frechet_{tag}.csv"), index=False)
        pd.DataFrame(ws_table).to_csv(os.path.join(OUT_DIR, f"group_wasserstein_ks_{tag}.csv"), index=False)
        pd.DataFrame(cv_table).to_csv(os.path.join(OUT_DIR, f"group_cv_{tag}.csv"), index=False)
        pd.DataFrame(slope_table).to_csv(os.path.join(OUT_DIR, f"group_slope_{tag}.csv"), index=False)

        df_f = pd.DataFrame(frechet_table)
        df_w = pd.DataFrame(ws_table)
        ratio_median = df_f["between_over_within"].median()
        p_median = df_w["ks_p_median"].median()
        summaries[tag] = {
            "ratio_median": ratio_median, "p_median": p_median,
            "df_f": df_f, "df_w": df_w,
        }

    print(f"\nCSV 写入 {OUT_DIR}")

    # === 汇总结论（用 shape 版做主结论，absolute 版做辅证）===
    print("\n[2/3] 汇总结论 ...")
    main_s = summaries["shape"]
    abs_s  = summaries["absolute"]
    print(f"  shape 版:   Fréchet 组间/组内比值中位数 = {main_s['ratio_median']:.3f}  |  KS p 中位数 = {main_s['p_median']:.3f}")
    print(f"  absolute版: Fréchet 组间/组内比值中位数 = {abs_s['ratio_median']:.3f}  |  KS p 中位数 = {abs_s['p_median']:.3f}")

    # === 可视化 ===
    print("\n[3/3] 绘图 ...")

    # 图1：Fréchet 比值分布（两套并排）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tag in zip(axes, ["shape", "absolute"]):
        s = summaries[tag]
        df = s["df_f"]["between_over_within"].dropna()
        ax.hist(df, bins=20,
                color="#2a78d6" if tag == "shape" else "#eb6834",
                alpha=0.85)
        ax.axvline(1.0, color="red", ls="--", lw=1.5, label="between = within")
        ax.axvline(s["ratio_median"], color="black", ls="-", lw=1.5,
                   label=f"median = {s['ratio_median']:.2f}")
        ax.set_title(f"Fréchet distance ratio [{tag}]\n"
                     "(组间 / 组内比值分布)")
        ax.set_xlabel("between / within")
        ax.set_ylabel("频次")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "group_similarity_summary.png"), dpi=110)
    plt.close(fig)

    # 图2：每个 group 每个变量的归一化平均曲线（用 absolute 版更直观）
    n_vars = len(var_list)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    for i, var in enumerate(var_list):
        ax_ = axes[i]
        for g in GROUPS:
            arr = traj_abs[g][var]
            if len(arr) == 0:
                continue
            mean_tr = arr.mean(0)
            std_tr = arr.std(0)
            x = np.linspace(0, 1, len(mean_tr))
            ax_.plot(x, mean_tr, lw=1.6, label=f"group {g} (n={len(arr)})")
            ax_.fill_between(x, mean_tr - std_tr, mean_tr + std_tr, alpha=0.15)
        ax_.set_title(f"{var}  [absolute z-score]", fontsize=9)
        ax_.set_xlabel("归一化时间" if i // n_cols == n_rows - 1 else "")
        ax_.grid(True, alpha=0.3)
        if i == 0:
            ax_.legend(fontsize=7)
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)
    fig2.suptitle("各 group 归一化轨迹（绝对版 z-score，均值 ± 标准差）— 越重合越说明可以共用模型", y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "group_similarity_curves.png"), dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"图写入 {OUT_DIR}")

    # === 决策建议 ===
    print("\n=== 结论 ===")
    rm, pm = main_s["ratio_median"], main_s["p_median"]
    rm_a, pm_a = abs_s["ratio_median"], abs_s["p_median"]

    if rm < 1 and pm > 0.05:
        print(f"  ✅ 形状版中位数 ratio={rm:.2f} (<1) 且 KS p 中位数={pm:.2f} (>0.05)")
        print("  → 5 个 group 在大多数变量上的'形状'和'分布'难以区分，")
        print("    完全可以共用一个模型。")
    elif rm < 1.5 and pm > 0.01:
        print(f"  ⚠️ 形状版中位数 ratio={rm:.2f} (略>1) 且 KS p 中位数={pm:.2f}")
        print("  → 5 组在多数变量上相似，但少数变量存在差异。建议：")
        print("    - 先用单模型 baseline，对表现差的变量再考虑分组微调。")
        if rm_a > 1.5 or pm_a < 0.01:
            print(f"  ⚠️ 注意：绝对版 ratio={rm_a:.2f} / KS p={pm_a:.2f} 提示**绝对轨迹**偏离较大，")
            print("    - 这通常说明各组整体水平/方差不同，z-score 时记得**用全训练集 fit** 再 transform，避免泄露。")
    else:
        print(f"  ❌ 形状版中位数 ratio={rm:.2f} (>1.5) 或 KS p 中位数={pm:.2f}")
        print("  → 5 组之间差异显著，不建议直接共用一个模型。")


if __name__ == "__main__":
    main()