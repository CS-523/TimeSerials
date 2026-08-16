"""
Polynomial regression: 用 y1~y4（+ 可选 x1~x8 衍生特征）拟合实验终值 Y
==========================================================================
原理：StandardScaler → PolynomialFeatures(degree 2/3) → Ridge，把中间目标 y1~y4
      非线性地回归到最终结果 Y。Y 是每个实验的终值标量（非时序），模型拟合的是
      "实验早/中期信号 → 实验终值" 的映射，而非预测未来时序。

三种训练配置（互斥）：
  --mode last      : 全局模型。每个实验取 y1~y4 的最后一个有效值 → 4 个特征。
  --mode window    : 全局模型。取 y1~y4 的最后 N 个观测 → N×4 个特征（--window，默认 4）。
  --per-group      : 每组一个模型。为每个 y/x 列算 last/mean/delta 衍生特征，再按
                     GROUP_FEATURES 给每组挑 top-8 特征，各组独立训练 Ridge。
                     （此开关优先级高于 --mode，会自动改用 engineered 特征）

常用参数：
  --degree {2,3}   多项式阶数（默认 2）
  --alpha FLOAT    岭回归正则强度；默认不指定，由 RidgeCV 自动选择
  --drop-y4        只保留 y1~y3（y3/y4 高度共线）
  --seed INT       数据划分随机种子（默认 42）
  --base-dir       数据根目录（含 1/..5/ 子目录，默认 .）
  --out-dir        输出目录（默认 ./src/model_out）

划分方式：
  全局模型  : split_experiments 按实验 7:1:2 → train/val/test，train+val 训练、test 评估。
  每组模型  : 每组内 80/20 划分训练/测试，最终指标按组汇总。

Usage:
    python src/train_y_poly.py --degree 2 --mode last
    python src/train_y_poly.py --degree 2 --mode window --window 4
    python src/train_y_poly.py --degree 3 --mode window --window 8 --drop-y4
    python src/train_y_poly.py --degree 2 --per-group

输出（写入 --out-dir，tag = deg{degree}_{mode}）：
  y_poly_{tag}.pkl              训练好的 Pipeline / 分组 Pipeline dict
  y_poly_{tag}_metrics.json     RMSE/MAE/R² + 每组细分
  y_poly_{tag}_predictions.csv  逐样本 Y_true / Y_pred
  y_poly_{tag}.png              预测散点 + 残差图
"""
from __future__ import annotations

import argparse, os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, split_experiments, Y_INT_COLS, X_COLS

# ---- Per-group feature selection (based on Pearson r top-8 within each group) ----
GROUP_FEATURES = {
    1: ["y4__last", "y3__mean", "x2__mean", "y3__last", "x7__mean"],
    2: ["y4__last", "y3__last", "x2__mean", "y3__mean", "y4__mean"],
    3: ["x8__delta", "x8__last", "x6__last", "x6__delta", "x4__delta"],
    4: ["y1__last", "x2__delta", "x2__last", "x6__delta", "x5__mean"],
    5: ["y4__last", "x2__mean", "y1__mean", "x6__mean", "y2__mean"],
}


def _last_valid(series):
    """Return last non-NaN value in a series, or NaN if all NaN."""
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def _first_valid(series):
    """Return first non-NaN value, or NaN."""
    s = series.dropna()
    return float(s.iloc[0]) if len(s) > 0 else np.nan


def extract_engineered_features(exps):
    """Compute all engineered features: last, mean, delta for every y and x column.

    Returns (X, Y_true, groups, files, feature_names) where X has columns like
    'y4__last', 'x2__mean', 'x7__delta', etc.
    """
    all_cols = list(Y_INT_COLS) + list(X_COLS)  # y1..y4, x1..x8
    rows = []
    for e in exps:
        row = {"Y": e.Y, "group": int(e.group), "file": e.file}
        for c in all_cols:
            s = e.df[c]
            row[f"{c}__last"] = _last_valid(s)
            row[f"{c}__mean"] = s.mean()
            row[f"{c}__delta"] = _last_valid(s) - _first_valid(s)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    # Build feature list: last, mean, delta for each col
    feat_names = [f"{c}__{agg}" for c in all_cols for agg in ["last", "mean", "delta"]]
    # Drop columns with too many NaNs (>10%)
    valid_feats = [f for f in feat_names if tbl[f].isna().mean() < 0.1]
    tbl = tbl.dropna(subset=valid_feats).copy()
    X = tbl[valid_feats].to_numpy(dtype=np.float64)
    y = tbl["Y"].to_numpy(dtype=np.float64)
    groups = tbl["group"].to_numpy(dtype=int)
    files = tbl["file"].to_numpy()
    return X, y, groups, files, valid_feats


def extract_features_last(exps, y_cols=Y_INT_COLS):
    """(mode=last) Last valid y values -> len(y_cols) features per experiment."""
    rows = []
    for e in exps:
        row = {"Y": e.Y, "group": int(e.group), "file": e.file}
        for c in y_cols:
            row[c] = _last_valid(e.df[c])
        rows.append(row)
    tbl = pd.DataFrame(rows).dropna(subset=y_cols).copy()
    X = tbl[list(y_cols)].to_numpy(dtype=np.float64)
    y = tbl["Y"].to_numpy(dtype=np.float64)
    groups = tbl["group"].to_numpy(dtype=int)
    files = tbl["file"].to_numpy()
    feat_names = list(y_cols)
    return X, y, groups, files, feat_names


def extract_features_window(exps, window=4, y_cols=Y_INT_COLS):
    """(mode=window) Last `window` y observations -> window*len(y_cols) features."""
    p = len(y_cols)
    rows = []
    for e in exps:
        df = e.df[list(y_cols)].copy()
        obs_rows = df.dropna(how="all", subset=y_cols)
        if len(obs_rows) == 0:
            continue
        tail = obs_rows.tail(window).copy()
        tail = tail.ffill().fillna(0.0)
        if len(tail) < window:
            pad = pd.DataFrame(np.zeros((window - len(tail), p)), columns=y_cols)
            tail = pd.concat([pad, tail], ignore_index=True)
        feat = tail.to_numpy(dtype=np.float64).flatten()
        rows.append({"feat": feat, "Y": e.Y, "group": int(e.group), "file": e.file})
    if not rows:
        raise RuntimeError("No valid experiments with y observations found")
    X = np.stack([r["feat"] for r in rows], axis=0)
    y = np.array([r["Y"] for r in rows], dtype=np.float64)
    groups = np.array([r["group"] for r in rows], dtype=int)
    files = np.array([r["file"] for r in rows])
    feat_names = [f"{col}_t{i}" for i in range(window) for col in y_cols]
    return X, y, groups, files, feat_names


def main():
    ap = argparse.ArgumentParser(description="Polynomial regression: y1-y4 -> Y")
    ap.add_argument("--base-dir", default=".", help="Data root with 1/..5/ subdirs")
    ap.add_argument("--out-dir", default="./src/model_out", help="Output directory")
    ap.add_argument("--degree", type=int, default=2, choices=[2, 3],
                    help="Polynomial degree (2 or 3)")
    ap.add_argument("--mode", choices=["last", "window"], default="last",
                    help="Feature mode: last value or sliding window")
    ap.add_argument("--window", type=int, default=4,
                    help="Number of y observations in window (mode=window)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Ridge alpha (default: RidgeCV auto-select)")
    ap.add_argument("--drop-y4", action="store_true",
                    help="Drop y4 (keep y1-y3 only; y3/y4 are highly collinear)")
    ap.add_argument("--per-group", action="store_true",
                    help="Train separate model for each group (1-5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Determine which y columns to use
    y_cols = ["y1", "y2", "y3"] if args.drop_y4 else list(Y_INT_COLS)
    y_tag = "y123" if args.drop_y4 else "y1234"

    # 1. Load & extract features
    exps = load_all(args.base_dir)
    print(f"Loaded experiments: {len(exps)}")

    if args.mode == "last":
        X, Y_true, groups, files, raw_names = extract_features_last(exps, y_cols=y_cols)
        mode_tag = "last"
    elif args.mode == "window":
        X, Y_true, groups, files, raw_names = extract_features_window(exps, args.window, y_cols=y_cols)
        mode_tag = f"win{args.window}"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # If per-group: always use engineered features so we can select per group
    if args.per_group:
        X_all, Y_true, groups, files, all_feat_names = extract_engineered_features(exps)
        mode_tag = f"engineered_pergroup"
    else:
        if args.mode == "last":
            X_all, Y_true, groups, files, all_feat_names = extract_features_last(exps, y_cols=y_cols)
            mode_tag = f"{y_tag}_last"
        else:
            X_all, Y_true, groups, files, all_feat_names = extract_features_window(exps, args.window, y_cols=y_cols)
            mode_tag = f"{y_tag}_win{args.window}"

    print(f"Valid samples: {len(Y_true)}")
    print(f"Available features: {len(all_feat_names)}")
    print(f"Y range: [{Y_true.min():.2f}, {Y_true.max():.2f}]")
    print(f"Groups: {sorted(set(groups))}")

    # 2. Build Ridge model builder
    def make_pipe():
        poly = PolynomialFeatures(degree=args.degree, include_bias=True)
        scaler = StandardScaler()
        if args.alpha is not None:
            ridge = Ridge(alpha=args.alpha)
        else:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        return Pipeline([("scaler", scaler), ("poly", poly), ("ridge", ridge)])

    if not args.per_group:
        # ===== Global model (original behavior) =====
        train_exps, val_exps, test_exps = split_experiments(exps, seed=args.seed)

        def _get_indices(exp_list):
            idxs = []
            for e in exp_list:
                for i, f in enumerate(files):
                    if f == e.file:
                        idxs.append(i)
                        break
            return np.array(idxs, dtype=int)

        train_idx = _get_indices(train_exps)
        val_idx = _get_indices(val_exps)
        test_idx = _get_indices(test_exps)
        trainval_idx = np.concatenate([train_idx, val_idx])
        print(f"Train+val: {len(trainval_idx)}  Test: {len(test_idx)}")

        X_train, Y_train = X_all[trainval_idx], Y_true[trainval_idx]
        X_test, Y_test = X_all[test_idx], Y_true[test_idx]
        groups_test = groups[test_idx]
        files_test = files[test_idx]

        pipe = make_pipe()
        pipe.fit(X_train, Y_train)
        ridge = pipe.named_steps["ridge"]
        poly = pipe.named_steps["poly"]

        Y_pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(Y_test, Y_pred)))
        mae = float(mean_absolute_error(Y_test, Y_pred))
        r2 = float(r2_score(Y_test, Y_pred))
        alpha_used = float(ridge.alpha_) if args.alpha is None else args.alpha

        print(f"\n===== Global Model (deg={args.degree} {mode_tag}) =====")
        print(f"Ridge alpha: {alpha_used:.2f}  Poly features: {poly.n_output_features_}")
        print(f"RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.4f}")

        # Per-group breakdown
        print("\n--- Per-group ---")
        per_group = {}
        for g in sorted(set(groups_test)):
            mask = groups_test == g
            if mask.sum() == 0: continue
            g_rmse = float(np.sqrt(mean_squared_error(Y_test[mask], Y_pred[mask])))
            g_mae = float(mean_absolute_error(Y_test[mask], Y_pred[mask]))
            g_r2 = float(r2_score(Y_test[mask], Y_pred[mask]))
            per_group[f"group_{g}"] = {"rmse": g_rmse, "mae": g_mae, "r2": g_r2, "n": int(mask.sum())}
            print(f"  G{g}: RMSE={g_rmse:.1f}  MAE={g_mae:.1f}  R2={g_r2:.3f}  n={mask.sum()}")

        # Save
        tag = f"deg{args.degree}_{mode_tag}"
        joblib.dump(pipe, os.path.join(args.out_dir, f"y_poly_{tag}.pkl"))
        metrics = {
            "model": "global", "degree": args.degree,
            "alpha": alpha_used, "n_raw_features": X.shape[1],
            "n_poly_features": poly.n_output_features_,
            "n_train": len(trainval_idx), "n_test": len(test_idx),
            "rmse": rmse, "mae": mae, "r2": r2, "per_group": per_group,
        }
        with open(os.path.join(args.out_dir, f"y_poly_{tag}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame({"file": files_test, "group": groups_test,
                      "Y_true": Y_test, "Y_pred": Y_pred})\
          .to_csv(os.path.join(args.out_dir, f"y_poly_{tag}_predictions.csv"), index=False)

        # Coefficients
        coef = ridge.coef_ if args.alpha is not None else ridge.coef_
        poly_names = poly.get_feature_names_out(raw_names)
        coef_df = pd.DataFrame({"feature": poly_names, "coef": coef})\
                    .sort_values("coef", key=abs, ascending=False)
        print(f"\nTop 8 coefficients:")
        print(coef_df.head(8).to_string(index=False))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(Y_test, Y_pred, alpha=0.6, c=groups_test, cmap="Set2",
                        edgecolors="grey", linewidths=0.3)
        lo, hi = min(Y_test.min(), Y_pred.min()), max(Y_test.max(), Y_pred.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="perfect")
        axes[0].set_xlabel("Y true"); axes[0].set_ylabel("Y pred")
        axes[0].set_title(f"Global deg={args.degree}\nRMSE={rmse:.1f} MAE={mae:.1f} R2={r2:.3f}")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        residuals = Y_test - Y_pred
        axes[1].scatter(Y_pred, residuals, alpha=0.6, c=groups_test, cmap="Set2",
                        edgecolors="grey", linewidths=0.3)
        axes[1].axhline(0, color="k", linestyle="--", alpha=0.4)
        axes[1].set_xlabel("Y pred"); axes[1].set_ylabel("Residual")
        axes[1].set_title(f"Global deg={args.degree}: Residuals")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"y_poly_{tag}.png")
        plt.savefig(fig_path, dpi=120); plt.close()
        print(f"Saved: {fig_path}")

    else:
        # ===== Per-group models with group-specific features =====
        rng = np.random.RandomState(args.seed)
        all_pipes = {}
        all_preds, all_trues, all_groups_test, all_files_test = [], [], [], []
        per_group_metrics = {}

        print(f"\n===== Per-Group Models (deg={args.degree}) =====")

        for g in sorted(set(groups)):
            # Select group-specific features (top-8 from Pearson correlation)
            g_feats = GROUP_FEATURES.get(g, all_feat_names[:8])
            # Keep only features that exist in the dataframe
            g_feats = [f for f in g_feats if f in all_feat_names]
            g_col_idx = [list(all_feat_names).index(f) for f in g_feats]

            g_mask = groups == g
            g_X = X_all[g_mask][:, g_col_idx]
            g_Y = Y_true[g_mask]
            g_files = files[g_mask]
            n_total = len(g_Y)

            # Within-group 80/20 split
            idx = np.arange(n_total)
            rng.shuffle(idx)
            n_train = int(n_total * 0.8)
            train_idx_g, test_idx_g = idx[:n_train], idx[n_train:]

            pipe = make_pipe()
            pipe.fit(g_X[train_idx_g], g_Y[train_idx_g])
            ridge = pipe.named_steps["ridge"]
            poly = pipe.named_steps["poly"]
            alpha_used = float(ridge.alpha_) if args.alpha is None else args.alpha

            Y_pred_g = pipe.predict(g_X[test_idx_g])
            g_rmse = float(np.sqrt(mean_squared_error(g_Y[test_idx_g], Y_pred_g)))
            g_mae = float(mean_absolute_error(g_Y[test_idx_g], Y_pred_g))
            g_r2 = float(r2_score(g_Y[test_idx_g], Y_pred_g))

            all_pipes[g] = pipe
            all_preds.append(Y_pred_g)
            all_trues.append(g_Y[test_idx_g])
            all_groups_test.append(np.full(len(test_idx_g), g))
            all_files_test.append(g_files[test_idx_g])

            # Top coefficients
            coef = ridge.coef_ if args.alpha is not None else ridge.coef_
            poly_names = poly.get_feature_names_out(g_feats)
            top3 = pd.DataFrame({"feature": poly_names, "coef": coef})\
                     .sort_values("coef", key=abs, ascending=False).head(3)

            per_group_metrics[f"group_{g}"] = {
                "features": g_feats, "n_train": n_train, "n_test": len(test_idx_g),
                "alpha": alpha_used, "n_poly": poly.n_output_features_,
                "rmse": g_rmse, "mae": g_mae, "r2": g_r2,
                "top3": top3[["feature", "coef"]].to_dict("records"),
            }
            print(f"  G{g}: features={g_feats}  n_train={n_train}")
            print(f"       alpha={alpha_used:.1f}  poly={poly.n_output_features_}  "
                  f"RMSE={g_rmse:.1f}  MAE={g_mae:.1f}  R2={g_r2:.4f}")
            print(f"       top3: {top3['feature'].tolist()}")

        # Overall metrics (pooled across groups)
        Y_pred_all = np.concatenate(all_preds)
        Y_true_all = np.concatenate(all_trues)
        groups_all = np.concatenate(all_groups_test)
        files_all = np.concatenate(all_files_test)
        overall_rmse = float(np.sqrt(mean_squared_error(Y_true_all, Y_pred_all)))
        overall_mae = float(mean_absolute_error(Y_true_all, Y_pred_all))
        overall_r2 = float(r2_score(Y_true_all, Y_pred_all))

        print(f"\n  Overall (per-group, tailored features): RMSE={overall_rmse:.1f}  "
              f"MAE={overall_mae:.1f}  R2={overall_r2:.4f}")

        # Save
        tag = f"deg{args.degree}_{mode_tag}"
        joblib.dump({"pipes": all_pipes, "group_features": GROUP_FEATURES},
                    os.path.join(args.out_dir, f"y_poly_{tag}.pkl"))
        metrics = {
            "model": "per_group_tailored", "degree": args.degree,
            "rmse": overall_rmse, "mae": overall_mae, "r2": overall_r2,
            "per_group": per_group_metrics,
        }
        with open(os.path.join(args.out_dir, f"y_poly_{tag}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        pd.DataFrame({"file": files_all, "group": groups_all,
                      "Y_true": Y_true_all, "Y_pred": Y_pred_all})\
          .to_csv(os.path.join(args.out_dir, f"y_poly_{tag}_predictions.csv"), index=False)

        # Plot: one color per group
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for g in sorted(set(groups)):
            g_mask = groups_all == g
            axes[0].scatter(Y_true_all[g_mask], Y_pred_all[g_mask], alpha=0.6,
                            label=f"G{g}", edgecolors="grey", linewidths=0.3)
        lo, hi = min(Y_true_all.min(), Y_pred_all.min()), max(Y_true_all.max(), Y_pred_all.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.4)
        axes[0].set_xlabel("Y true"); axes[0].set_ylabel("Y pred")
        axes[0].set_title(f"Per-group deg={args.degree}\nRMSE={overall_rmse:.1f} MAE={overall_mae:.1f} R2={overall_r2:.3f}")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        residuals = Y_true_all - Y_pred_all
        for g in sorted(set(groups)):
            g_mask = groups_all == g
            axes[1].scatter(Y_pred_all[g_mask], residuals[g_mask], alpha=0.6,
                            label=f"G{g}", edgecolors="grey", linewidths=0.3)
        axes[1].axhline(0, color="k", linestyle="--", alpha=0.4)
        axes[1].set_xlabel("Y pred"); axes[1].set_ylabel("Residual")
        axes[1].set_title(f"Per-group deg={args.degree}: Residuals")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"y_poly_{tag}.png")
        plt.savefig(fig_path, dpi=120); plt.close()
        print(f"Saved: {fig_path}")

    return metrics


if __name__ == "__main__":
    main()
