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


def main(base_dir: str = "/kefu-nas/ybkong/time_serials-master",
         out_dir: str = "/kefu-nas/ybkong/time_serials-master/src/analysis_out"):
    os.makedirs(out_dir, exist_ok=True)
    exps = load_all(base_dir)
    print(f"[analyze] 实验数 = {len(exps)}")

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
    feat_last["Y"] = Y_arr

    # === 2. 描述性统计 ===
    desc = all_x.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc["missing_pct"] = (all_x.isna().mean() * 100).round(2)
    desc.to_csv(os.path.join(out_dir, "summary_stats.csv"))
    print(f"[analyze] 描述统计已写入 summary_stats.csv")

    # === 3. 相关性矩阵 ===
    corr_df = pd.concat([all_x, all_y], axis=1).corr()
    corr_df.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_yticklabels(corr_df.columns)
    ax.set_title("Correlation: x1-x8 & y1-y4")
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=120)
    plt.close()
    print(f"[analyze] 相关性热图已写入 correlation_matrix.png")

    # === 4. x 特征 vs Y 的关联（Spearman 更稳）===
    feat_last["Y"] = Y_arr
    feat_mean["Y"] = Y_arr
    feat_delta["Y"] = Y_arr
    spec_x = feat_last.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_mean = feat_mean.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)
    spec_delta = feat_delta.corr(method="spearman")["Y"].drop("Y").sort_values(key=abs, ascending=False)

    pd.DataFrame({
        "last_spearman": spec_x,
        "mean_spearman": spec_mean,
        "delta_spearman": spec_delta,
    }).to_csv(os.path.join(out_dir, "feature_vs_Y_importance.csv"))
    print(f"[analyze] 特征 vs Y 重要性已写入 feature_vs_Y_importance.csv")
    print("\n=== 末态特征 vs Y 的 Spearman 相关 ===")
    print(spec_x)
    print("\n=== 均值特征 vs Y 的 Spearman 相关 ===")
    print(spec_mean)
    print("\n=== 增量特征 vs Y 的 Spearman 相关 ===")
    print(spec_delta)

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