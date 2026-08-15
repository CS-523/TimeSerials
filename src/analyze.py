"""
参数关联性分析
================
对所有实验的 x1-x8 / y1-y4 / Y 做：
1. 描述性统计（min/max/mean/std/分位数）
2. 相关性矩阵（Pearson）
3. 单变量回归：哪些 x 与 Y 最相关
4. y1-y4 之间的关联，以及 y4 与 x 的关联
5. 输出：
   - summary_stats.csv
   - correlation_matrix.png
   - feature_vs_Y_importance.csv
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
from data_loader import load_all, X_COLS, Y_INT_COLS


def _plot_corr_heatmap(corr_df: pd.DataFrame, title: str, save_path: str):
    """绘制相关性矩阵热图并保存。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_yticklabels(corr_df.columns)
    ax.set_title(title)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _plot_xy_corr_heatmap(corr_df: pd.DataFrame, title: str, save_path: str):
    """绘制 x1-x8 vs y1-y4 交叉相关热图（非方阵，8 行 x 4 列）并保存。"""
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _plot_top10_Y_bar(corr_df: pd.DataFrame, title: str, save_path: str):
    """绘制与 Y 的 Pearson 相关系数 top-10 柱状图。"""
    y_corr = corr_df["Y"].drop("Y").sort_values(key=abs, ascending=True)
    top10 = y_corr.tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d64545" if v < 0 else "#3b7dd8" for v in top10.values]
    bars = ax.barh(range(len(top10)), top10.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title(title)

    # 数值标签
    for i, (v, bar) in enumerate(zip(top10.values, bars)):
        x_pos = v + 0.01 if v >= 0 else v - 0.01
        ha = "left" if v >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _print_top10_table(corr_df: pd.DataFrame, label: str):
    """打印与 Y 相关系数 top-10 的 Markdown 表格。"""
    y_corr = corr_df["Y"].drop("Y").sort_values(key=abs, ascending=False)
    top10 = y_corr.head(10)
    print(f"\n### {label} — Top-10 特征 vs Y (Pearson r)")
    print("| 排名 | 特征 | Pearson r | 方向 |")
    print("|------|------|-----------|------|")
    for i, (feat, val) in enumerate(top10.items(), 1):
        direction = "正相关 ↑" if val > 0 else "负相关 ↓"
        bar = "█" * min(int(abs(val) * 20), 10)
        print(f"| {i} | {feat} | {val:+.4f} | {direction} {bar} |")
    print()


def main(base_dir: str = None,
         out_dir: str = None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if out_dir is None:
        out_dir = os.path.join(base_dir, "src", "analysis_out")
    os.makedirs(out_dir, exist_ok=True)
    exps = load_all(base_dir)
    print(f"[analyze] 实验数 = {len(exps)}")
    if len(exps) == 0:
        print(f"[analyze] 错误：在 {base_dir}/{{1..5}}/ 下未找到任何 CSV 文件。")
        print(f"[analyze] 请确认数据目录存在，或通过 main(base_dir='...') 指定正确路径。")
        return

    # === 1. 收集所有 x / y 数据 ===
    all_x = pd.concat([e.df[X_COLS] for e in exps], axis=0, ignore_index=True)
    all_y = pd.concat([e.df[Y_INT_COLS] for e in exps], axis=0, ignore_index=True)
    # Y 是实验级单值
    Y_arr = np.array([e.Y for e in exps if e.Y is not None], dtype=np.float32)
    # 实验级 x 特征：用"末态"和"均值"
    feat_last = pd.DataFrame([e.df[X_COLS].iloc[-1].values for e in exps if e.Y is not None],
                              columns=[f"{c}__last" for c in X_COLS])
    feat_mean = pd.DataFrame([e.df[X_COLS].mean().values for e in exps if e.Y is not None],
                              columns=[f"{c}__mean" for c in X_COLS])
    feat_delta = pd.DataFrame([(e.df[X_COLS].iloc[-1] - e.df[X_COLS].iloc[0]).values
                                for e in exps if e.Y is not None],
                               columns=[f"{c}__delta" for c in X_COLS])
    # 实验级 y 聚合特征
    y_mean_cols = [f"{c}__mean" for c in Y_INT_COLS]
    y_last_cols = [f"{c}__last" for c in Y_INT_COLS]
    feat_y_mean = pd.DataFrame([
        e.df[Y_INT_COLS].mean().values for e in exps if e.Y is not None
    ], columns=y_mean_cols)
    feat_y_last = pd.DataFrame([
        e.df[Y_INT_COLS].dropna(how="all").iloc[-1].values
        if len(e.df[Y_INT_COLS].dropna(how="all")) > 0
        else [np.nan] * len(Y_INT_COLS)
        for e in exps if e.Y is not None
    ], columns=y_last_cols)

    # 构建实验级特征大表（x 聚合 + y 聚合 + Y），用于 Y 关联分析
    feat_all = pd.concat([feat_last, feat_mean, feat_delta, feat_y_mean, feat_y_last], axis=1)
    feat_all["Y"] = Y_arr

    # === 1b. 各数据项有效个数统计（横表）===
    total_rows = len(all_x)
    print(f"\n{'='*70}")
    print(f"数据项有效个数统计  |  总行数(时序): {total_rows}  |  总实验: {len(exps)}  (Y 有效: {len(Y_arr)})")
    print(f"{'='*70}")
    cols = X_COLS + Y_INT_COLS + ["Y"]
    print(f"{'数据项':<6}", end="")
    for c in cols:
        print(f"  {c:>6}", end="")
    print(f"\n{'有效':<6}", end="")
    for c in X_COLS:
        print(f"  {int(all_x[c].notna().sum()):>6}", end="")
    for c in Y_INT_COLS:
        print(f"  {int(all_y[c].notna().sum()):>6}", end="")
    print(f"  {len(Y_arr):>6}")
    print(f"{'完备率':<6}", end="")
    for c in X_COLS:
        print(f"  {all_x[c].notna().sum()/total_rows*100:>5.1f}%", end="")
    for c in Y_INT_COLS:
        print(f"  {all_y[c].notna().sum()/total_rows*100:>5.1f}%", end="")
    print(f"  {len(Y_arr)/len(exps)*100:>5.1f}%")
    print()

    # === 2. 描述性统计 ===
    desc = all_x.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc["missing_pct"] = (all_x.isna().mean() * 100).round(2)
    desc.to_csv(os.path.join(out_dir, "summary_stats.csv"))
    print(f"[analyze] 描述统计已写入 summary_stats.csv")

    # === 3. 相关性矩阵（全局）===
    corr_df = pd.concat([all_x, all_y], axis=1).corr()
    corr_df.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))
    _plot_corr_heatmap(corr_df, "Correlation: x1-x8 & y1-y4 (all groups)",
                       os.path.join(out_dir, "correlation_matrix.png"))

    # === 3a. x1-x8 vs y1-y4 交叉相关（8x4 子矩阵）===
    xy_corr = corr_df.loc[X_COLS, Y_INT_COLS]
    xy_corr.to_csv(os.path.join(out_dir, "x_y_correlation.csv"))
    _plot_xy_corr_heatmap(xy_corr, "x1-x8 vs y1-y4 (Pearson, all groups)",
                          os.path.join(out_dir, "x_y_correlation.png"))
    print(f"[analyze] x-y 交叉相关热图已写入 x_y_correlation.png")

    # === 3b. 按 group 分别绘制相关性矩阵 ===
    for g in sorted(set(e.group for e in exps)):
        g_exps = [e for e in exps if e.group == g]
        g_x = pd.concat([e.df[X_COLS] for e in g_exps], axis=0, ignore_index=True)
        g_y = pd.concat([e.df[Y_INT_COLS] for e in g_exps], axis=0, ignore_index=True)
        g_corr = pd.concat([g_x, g_y], axis=1).corr()
        g_corr.to_csv(os.path.join(out_dir, f"correlation_matrix_group_{g}.csv"))
        _plot_corr_heatmap(g_corr, f"Correlation: x1-x8 & y1-y4 (group {g})",
                           os.path.join(out_dir, f"correlation_matrix_group_{g}.png"))
        print(f"[analyze] group {g} 相关性热图已写入 correlation_matrix_group_{g}.png")

    # === 3c. Y 与所有特征的 Pearson 相关性 top-10 柱状图（全局 + 分 group）===
    corr_all = feat_all.corr()
    corr_all.to_csv(os.path.join(out_dir, "Y_correlation_matrix.csv"))
    _plot_top10_Y_bar(corr_all, "Top-10 features vs Y (Pearson, all groups)",
                      os.path.join(out_dir, "Y_top10_bar.png"))
    print(f"[analyze] Y top-10 柱状图已写入 Y_top10_bar.png")

    for g in sorted(set(e.group for e in exps)):
        g_exps = [e for e in exps if e.group == g and e.Y is not None]
        if len(g_exps) < 3:
            print(f"[analyze] group {g} 有效实验数不足({len(g_exps)})，跳过 Y top-10")
            continue
        g_y_mean = pd.DataFrame([e.df[Y_INT_COLS].mean().values for e in g_exps], columns=y_mean_cols)
        g_y_last = pd.DataFrame([
            e.df[Y_INT_COLS].dropna(how="all").iloc[-1].values
            if len(e.df[Y_INT_COLS].dropna(how="all")) > 0
            else [np.nan] * len(Y_INT_COLS)
            for e in g_exps
        ], columns=y_last_cols)
        g_last = pd.DataFrame([e.df[X_COLS].iloc[-1].values for e in g_exps],
                              columns=[f"{c}__last" for c in X_COLS])
        g_mean = pd.DataFrame([e.df[X_COLS].mean().values for e in g_exps],
                              columns=[f"{c}__mean" for c in X_COLS])
        g_delta = pd.DataFrame([(e.df[X_COLS].iloc[-1] - e.df[X_COLS].iloc[0]).values for e in g_exps],
                               columns=[f"{c}__delta" for c in X_COLS])
        g_feat = pd.concat([g_last, g_mean, g_delta, g_y_mean, g_y_last], axis=1)
        g_feat["Y"] = [e.Y for e in g_exps]
        g_corr_all = g_feat.corr()
        g_corr_all.to_csv(os.path.join(out_dir, f"Y_correlation_matrix_group_{g}.csv"))
        _plot_top10_Y_bar(g_corr_all, f"Top-10 features vs Y (Pearson, group {g})",
                          os.path.join(out_dir, f"Y_top10_bar_group_{g}.png"))
        print(f"[analyze] group {g} Y top-10 柱状图已写入 Y_top10_bar_group_{g}.png")

    # === 3d. 输出 top-10 表格（方便 LLM 阅读）===
    _print_top10_table(corr_all, "全局")

    for g in sorted(set(e.group for e in exps)):
        g_csv = os.path.join(out_dir, f"Y_correlation_matrix_group_{g}.csv")
        if os.path.exists(g_csv):
            g_corr = pd.read_csv(g_csv, index_col=0)
            _print_top10_table(g_corr, f"group {g}")

    # === 4. x / y 特征 vs Y 的关联（Spearman）===
    feat_last["Y"] = Y_arr
    feat_mean["Y"] = Y_arr
    feat_delta["Y"] = Y_arr
    feat_y_mean["Y"] = Y_arr
    feat_y_last["Y"] = Y_arr
    spec_x = feat_last.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_mean = feat_mean.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_delta = feat_delta.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_y_mean = feat_y_mean.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_y_last = feat_y_last.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)

    # 合并所有 Spearman 结果
    spec_all = pd.concat([spec_x, spec_mean, spec_delta, spec_y_mean, spec_y_last], axis=1)
    spec_all.columns = ["x_last", "x_mean", "x_delta", "y_mean", "y_last"]
    spec_all.to_csv(os.path.join(out_dir, "feature_vs_Y_importance.csv"))
    print(f"[analyze] 特征 vs Y 重要性已写入 feature_vs_Y_importance.csv")
    print("\n=== 末态 x vs Y 的 Spearman 相关 ===")
    print(spec_x)
    print("\n=== 均值 x vs Y 的 Spearman 相关 ===")
    print(spec_mean)
    print("\n=== 增量 x vs Y 的 Spearman 相关 ===")
    print(spec_delta)
    print("\n=== y 均值 vs Y 的 Spearman 相关 ===")
    print(spec_y_mean)
    print("\n=== y 末态 vs Y 的 Spearman 相关 ===")
    print(spec_y_last)

    # === 5. y1-y4 自身关联 + y 与 x 关联（y4 是必选输出）===
    y_corr = all_y.corr()
    y_corr.to_csv(os.path.join(out_dir, "y_correlation.csv"))
    print(f"\n[y1-y4 相关性]")
    print(y_corr)

    # === 6. 实验总长 + 末态 vs Y 的散点图 ===
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, c in enumerate(X_COLS):
        ax = axes[i // 4, i % 4]
        ax.scatter(feat_last[f"{c}__last"], Y_arr, alpha=0.5, s=12)
        ax.set_xlabel(f"{c} (last)")
        ax.set_ylabel("Y")
        ax.set_title(f"{c} vs Y (r={spec_x.loc[f'{c}__last']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "x_last_vs_Y.png"), dpi=120)
    plt.close()
    print(f"[analyze] 末态 x vs Y 散点图已写入 x_last_vs_Y.png")

    return {
        "n_exps": len(exps),
        "summary": desc,
        "corr": corr_df,
        "x_last_vs_Y": spec_x.to_dict(),
    }


if __name__ == "__main__":
    main()