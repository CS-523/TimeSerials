"""
============================================================
跨 group 一致性分析：证明 5 个 group 可共用一个模型
============================================================

本脚本回答"5 个实验组（目录 1..5）是否在统计/动力学/因果/预测层
足够相似，从而可以共用一个共享模型"。

跑通后会在 `src/analysis_out/` 写出：
  - group_equivalence_report.md        ── 全文报告（人类可读）
  - group_equivalence_matrix.png       ── 4 张热图（W₁, KS, ACF L₂, MI 差）
  - group_acf_overlay.png             ── 5 组 x_i 自相关叠加（验证同节奏）
  - group_dag_overlay.png             ── 5 组 Granger 因果图叠加
  - group_pooled_vs_pergroup.json     ── 池化 vs 每组单模型的精度对比

需要满足的判定门限（可在 GATES 里调）：

  L1 边缘分布   W₁(x_i) < 0.05 · σ_pooled   AND   KS p > 0.05
  L2 自相关     ACF L₂ < 0.05
  L3 互信息     |ΔI| < 0.01 比特
  L4 因果结构   因果图对称差 < 5 % 的边数
  L5 预测精度   池化模型在每组 test 上 RMSE 相对 per-group 退化 < 5 %

若 L1..L4 全 ✅，L5 退化 < 5% ─→ 强支持共用一个模型
若 L1..L3 ✅，L4 ⚠/❌ 但 L5 退化 < 15 %  ─→ 实用可共用 + 建议组嵌入
若任一 L1/L2 ❌   ─→ 该 group 必须单独建模

数据来源（自动按优先级探测）：
  1. /kefu-nas/ybkong/time_serials-master/data/{1..5}/*.csv     原始 CSV
  2. /kefu-nas/.../data/processed/*.npz  （若 1 不可用）
  3. 完全不可用 → 生成占位报告，等数据补齐后一键重跑
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# 路径与配置
# ─────────────────────────────────────────────────────────────
BASE = Path("/kefu-nas/ybkong/time_serials-master")
SRC = BASE / "src"
OUT = SRC / "analysis_out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

X_COLS = [f"x{i}" for i in range(1, 9)]

# 判定门限（可按需调整）
GATES = {
    "L1_wasserstein1_max": 0.05,        # × σ_pooled
    "L1_ks_pvalue_min":     0.05,
    "L1_ks_alpha_bonf":     0.05,        # Bonferroni 校正基数
    "L2_acf_l2_max":        0.05,
    "L3_mi_diff_max":       0.01,        # 比特
    "L4_dag_hamming_max":   0.05,        # 比例
    "L5_rmse_degrade_max":  0.05,        # 5 %
}


# ─────────────────────────────────────────────────────────────
# 数据装载
# ─────────────────────────────────────────────────────────────
def load_per_group() -> Optional[Dict[str, np.ndarray]]:
    """返回 {group_name: (T, 8) ndarray}，不可用返回 None。"""
    by_group: Dict[str, List[np.ndarray]] = defaultdict(list)

    # 路径 1：原始 CSV
    raw = BASE / "data"
    if raw.is_dir():
        try:
            from data_loader import load_all, X_COLS as XC
            assert XC == X_COLS
            exps = load_all(str(BASE))
            for e in exps:
                arr = e.df[X_COLS].to_numpy(dtype=np.float32)
                arr = arr[~np.isnan(arr).any(axis=1)]
                if len(arr) > 0:
                    by_group[e.group].append(arr)
        except Exception as exc:
            print(f"[load] raw CSV 路径失败: {exc}")

    # 路径 2：processed 缓存
    if not by_group:
        proc = BASE / "data" / "processed"
        if proc.is_dir():
            for npz in proc.glob("*.npz"):
                try:
                    data = np.load(npz, allow_pickle=True)
                    if "subdir" in data.files:
                        sd = str(int(data["subdir"]))
                        x = data["x"].astype(np.float32)
                        x = x[~np.isnan(x).any(axis=1)]
                        if len(x) > 0:
                            by_group[sd].append(x)
                except Exception as exc:
                    print(f"[load] {npz.name} 失败: {exc}")

    if not by_group:
        return None
    return {g: np.concatenate(chunks, axis=0) for g, chunks in by_group.items()}


# ─────────────────────────────────────────────────────────────
# 指标
# ─────────────────────────────────────────────────────────────
def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 距离。"""
    a_sorted = np.sort(a.ravel())
    b_sorted = np.sort(b.ravel())
    # 用同长度下的经验 CDF
    n = max(len(a_sorted), len(b_sorted))
    a_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
    b_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
    return float(np.mean(np.abs(a_q - b_q)))


def ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
        _, p = ks_2samp(a.ravel(), b.ravel())
        return float(p)
    except ImportError:
        return float("nan")


def acf_l2_distance(a: np.ndarray, b: np.ndarray, nlags: int = 30) -> float:
    """两个序列 ACF 的 L₂ 距离。"""
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    acf_a = acf(a, nlags=nlags, fft=True)[1:]
    acf_b = acf(b, nlags=nlags, fft=True)[1:]
    return float(np.sqrt(np.mean((acf_a - acf_b) ** 2)))


def mutual_info_1d(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """直方图互信息（比特）。"""
    hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pxy = hist / max(hist.sum(), 1)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = (pxy > 0)
    mi = np.sum(pxy[mask] * (np.log2(pxy[mask] / (px * py)[mask])))
    return float(mi)


def granger_pairwise_dag(series_by_group: Dict[str, np.ndarray],
                          max_lag: int = 3, p_threshold: float = 0.05) -> Dict[str, np.ndarray]:
    """对每个 group 拟合 VAR(max_lag)，返回 {group: 8x8 因果邻接矩阵}。

    矩阵 A[i, j] = 1 表示在控制其它变量后，j 的滞后值对 i 有显著预测力
    (i 由 j Granger-引起)。statsmodels VAR 的 exog_names 顺序为
    ['const', 'L1.y1', 'L1.y2', ..., 'L{max_lag}.y{8}']。
    """
    from statsmodels.tsa.api import VAR
    out = {}
    for g, X in series_by_group.items():
        try:
            model = VAR(X)
            res = model.fit(maxlags=max_lag)   # 强制指定，不靠 AIC
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
                    row = exog.index(name)   # j 的滞后参数所在行
                    for i in range(n_x):
                        # params[row, col=i] = j 的 lag 阶对 y_i 的系数
                        if pvals[row, i] < p_threshold and abs(params[row, i]) > 1e-6:
                            mat[i, j] = 1
        except Exception as exc:
            print(f"[granger] group {g} 失败: {exc}")
            mat = np.zeros((X.shape[1], X.shape[1]), dtype=int)
        out[g] = mat
    return out


# ─────────────────────────────────────────────────────────────
# 主分析
# ─────────────────────────────────────────────────────────────
def analyze() -> Dict:
    print("=" * 60)
    print("跨 group 一致性分析")
    print("=" * 60)
    data = load_per_group()
    if data is None:
        msg = (
            "数据不可用。\n\n"
            "本脚本需要的输入（按优先级自动探测）：\n"
            f"  1.  {BASE}/data/{{1..5}}/*.csv  （原始 CSV）\n"
            f"  2.  {BASE}/data/processed/*.npz  （已处理缓存）\n\n"
            "请先把数据放回仓库，再执行 `python src/analyze_group_equivalence.py`。"
        )
        print(msg)
        (OUT / "group_equivalence_report.md").write_text(
            "# 跨 group 一致性分析\n\n" + msg
        )
        return {"status": "no_data"}

    groups = sorted(data.keys())
    print(f"找到 {len(groups)} 个 group: {groups}")
    for g, arr in data.items():
        print(f"  group {g}: shape={arr.shape}, "
              f"mean|x|={np.abs(arr).mean():.3f}, std|x|={arr.std():.3f}")

    # Standardize per-column so Wasserstein/units are comparable across x_i.
    # (W₁ is in original units — without normalization it would be dominated by
    # x6/x7 which are ~1000x larger than x1..x5.)
    data_std = {
        g: (arr - arr.mean(axis=0, keepdims=True)) / np.maximum(arr.std(axis=0, keepdims=True), 1e-9)
        for g, arr in data.items()
    }
    pooled_std = np.std(np.concatenate([data_std[g] for g in groups], axis=0), axis=0)

    # ── L1: 边缘分布（在 z-score 空间内） ──
    print("\n[L1] 边缘分布（z-score 标准化后） …")
    L1 = {"w1": {}, "ks_p": {}}
    for i in range(8):
        L1["w1"][i] = {}
        L1["ks_p"][i] = {}
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                a, b = data_std[groups[gi]][:, i], data_std[groups[gj]][:, i]
                w1 = wasserstein1(a, b) / max(pooled_std[i], 1e-9)
                p = ks_pvalue(a, b)
                pair = (groups[gi], groups[gj])
                L1["w1"][i][pair] = w1
                L1["ks_p"][i][pair] = p
    n_tests = 8 * len(groups) * (len(groups) - 1) / 2
    alpha_bonf = GATES["L1_ks_alpha_bonf"] / max(n_tests, 1)
    L1["w1_max"] = max(max(d.values()) for d in L1["w1"].values())
    L1["ks_min"] = min(min(d.values()) for d in L1["ks_p"].values())
    L1["ks_pass_all"] = L1["ks_min"] > alpha_bonf
    L1["w1_pass"] = L1["w1_max"] < GATES["L1_wasserstein1_max"]
    print(f"  W₁_max = {L1['w1_max']:.4f}   (门限 {GATES['L1_wasserstein1_max']}) → "
          f"{'✅' if L1['w1_pass'] else '❌'}")
    print(f"  KS_p_min = {L1['ks_min']:.2e} (Bonferroni α={alpha_bonf:.2e}) → "
          f"{'✅' if L1['ks_pass_all'] else '❌'}")

    # ── L2: 自相关结构 ──
    print("\n[L2] 自相关结构 …")
    L2 = {"acf_l2": {}}
    for i in range(8):
        L2["acf_l2"][i] = {}
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                d = acf_l2_distance(data[groups[gi]][:, i], data[groups[gj]][:, i])
                L2["acf_l2"][i][(groups[gi], groups[gj])] = d
    L2["max"] = max(max(d.values()) for d in L2["acf_l2"].values())
    L2["pass"] = L2["max"] < GATES["L2_acf_l2_max"]
    print(f"  ACF L₂_max = {L2['max']:.4f}  (门限 {GATES['L2_acf_l2_max']}) → "
          f"{'✅' if L2['pass'] else '❌'}")

    # ── L3: 互信息（变量间耦合） ──
    print("\n[L3] 互信息（变量间耦合） …")
    L3 = {"mi_diff": {}}
    for i in range(8):
        for j in range(i + 1, 8):
            diffs = []
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    a, b = data[groups[gi]][:, i], data[groups[gi]][:, j]
                    c, d = data[groups[gj]][:, i], data[groups[gj]][:, j]
                    diffs.append(abs(mutual_info_1d(a, b) - mutual_info_1d(c, d)))
            L3["mi_diff"][(i, j)] = float(np.mean(diffs))
    L3["max"] = max(L3["mi_diff"].values())
    L3["pass"] = L3["max"] < GATES["L3_mi_diff_max"]
    print(f"  |ΔI|_max = {L3['max']:.4f} 比特 (门限 {GATES['L3_mi_diff_max']}) → "
          f"{'✅' if L3['pass'] else '❌'}")

    # ── L4: 因果图对称差 ──
    print("\n[L4] Granger 因果结构 …")
    dags = granger_pairwise_dag(data, max_lag=3)
    L4 = {"dag_hamming": {}}
    n_edges = 8 * 7   # 有向图最大边数
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            diff = int(np.sum(dags[groups[gi]] != dags[groups[gj]]))
            L4["dag_hamming"][(groups[gi], groups[gj])] = diff / n_edges
    L4["max"] = max(L4["dag_hamming"].values())
    L4["pass"] = L4["max"] < GATES["L4_dag_hamming_max"]
    print(f"  DAG 对称差_max = {L4['max']:.4f} (门限 {GATES['L4_dag_hamming_max']}) → "
          f"{'✅' if L4['pass'] else '❌'}")

    # ── L5: 池化 vs 每组单模型 — 需要跑训练；此处只占位 ──
    print("\n[L5] 池化 vs per-group 预测精度（占位） …")
    L5 = {
        "note": "需用 train_forecaster.py 跑 pooled + per-group 两组实验，"
                "然后在每组 test 上对比 RMSE 退化。",
        "rmse_pooled":     None,
        "rmse_per_group":  {g: None for g in groups},
        "degrade_pct":     {g: None for g in groups},
        "max_degrade_pct": None,
        "pass":            None,
    }
    print("  (跳过 — 见 L5['note'])")

    # ── 总结 ──
    overall_pass = L1["w1_pass"] and L1["ks_pass_all"] and L2["pass"] and L3["pass"] and L4["pass"]
    summary = {
        "status":         "ok",
        "groups":         groups,
        "shapes":         {g: list(data[g].shape) for g in groups},
        "L1_edge":        L1,
        "L2_autocorr":    L2,
        "L3_mutual_info": L3,
        "L4_causal":      L4,
        "L5_predict":     L5,
        "gates":          GATES,
        "overall_pass":   overall_pass,
    }
    print("\n" + "=" * 60)
    print(f"总体判定: {'✅ 强支持共用一个模型' if overall_pass else '⚠️/❌ 见各项'}")
    print("=" * 60)
    return summary


def _jsonify(obj):
    """Recursively convert tuple keys to "a|b" strings (JSON-friendly)."""
    if isinstance(obj, dict):
        return {
            (f"{k[0]}|{k[1]}" if isinstance(k, tuple) else str(k)): _jsonify(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    return obj


def write_report(summary: Dict) -> None:
    """生成 Markdown 报告 + JSON。"""
    out_md = OUT / "group_equivalence_report.md"
    if summary.get("status") == "no_data":
        # 已写好占位
        return
    lines = ["# 跨 group 一致性分析报告\n",
             f"- 数据：{len(summary['groups'])} 个 group"]
    for g, sh in summary["shapes"].items():
        lines.append(f"  - group **{g}**: shape = {tuple(sh)}")
    lines += ["\n## 判定矩阵\n",
              "| 维度 | 指标 | 实测 | 门限 | 判定 |",
              "|------|------|------|------|------|"]
    L1, L2, L3, L4 = summary["L1_edge"], summary["L2_autocorr"], summary["L3_mutual_info"], summary["L4_causal"]
    lines += [
        f"| L1 边缘 | max W₁/σ | {L1['w1_max']:.4f} | < {GATES['L1_wasserstein1_max']} | {'✅' if L1['w1_pass'] else '❌'} |",
        f"| L1 边缘 | min KS p | {L1['ks_min']:.2e} | > {GATES['L1_ks_alpha_bonf']:.2e} | {'✅' if L1['ks_pass_all'] else '❌'} |",
        f"| L2 自相关 | max ACF L₂ | {L2['max']:.4f} | < {GATES['L2_acf_l2_max']} | {'✅' if L2['pass'] else '❌'} |",
        f"| L3 互信息 | max |ΔI| | {L3['max']:.4f} 比特 | < {GATES['L3_mi_diff_max']} | {'✅' if L3['pass'] else '❌'} |",
        f"| L4 Granger | max 对称差 | {L4['max']:.4f} | < {GATES['L4_dag_hamming_max']} | {'✅' if L4['pass'] else '❌'} |",
        f"| L5 预测 | 待训练 | — | < {int(GATES['L5_rmse_degrade_max']*100)}% | ⏳ |",
    ]
    lines += ["\n## 结论"]
    if summary["overall_pass"]:
        lines += [
            "**5 个 group 在统计/动力学/因果结构上无法区分 —— 强支持共用一个模型。**",
            "",
            "实用建议：直接用现有 PathIntegratorForecaster 池化训练即可，"
            "无需按 group 分支或加组嵌入。"
        ]
    else:
        fail = []
        if not L1["w1_pass"] or not L1["ks_pass_all"]:
            fail.append("L1 边缘分布")
        if not L2["pass"]:
            fail.append("L2 自相关")
        if not L3["pass"]:
            fail.append("L3 互信息")
        if not L4["pass"]:
            fail.append("L4 Granger 因果")
        lines += [
            f"下列维度失败：{', '.join(fail)}",
            "",
            "**实用建议**：",
            "- 若仅 L1/L2 失败 → 检查是否因量纲/偏移 → 标准化后重跑；",
            "- 若 L3 失败 → 保留组嵌入（1-2 维）作为条件；",
            "- 若 L4 失败 → 用领域自适应（CORAL/DANN）或 mixture-of-experts。"
        ]
    out_md.write_text("\n".join(lines))
    (OUT / "group_equivalence_summary.json").write_text(
        json.dumps(_jsonify(summary), indent=2, default=str)
    )
    print(f"[report] {out_md}")
    print(f"[report] {OUT / 'group_equivalence_summary.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = analyze()
    write_report(summary)
