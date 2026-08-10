"""
综合报告 / PPT 资料生成（x 预测版）
====================================
根据分析、训练结果生成 Markdown 报告。
本版本不包含 y / 优化部分（因为本轮模型只预测 x）。

用法：
    python src/make_report.py
"""
from __future__ import annotations

import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS


BASE = "/kefu-nas/ybkong/time_serials-master"
MODEL_OUT = os.path.join(BASE, "src/model_out")
ANALYSIS_OUT = os.path.join(BASE, "src/analysis_out")
REPORT_PATH = os.path.join(BASE, "src/REPORT.md")


def _read_json(path: str, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    test_metrics = _read_json(os.path.join(MODEL_OUT, "test_metrics.json"), {})
    import pandas as pd
    imp_path = os.path.join(ANALYSIS_OUT, "feature_vs_Y_importance.csv")
    imp_df = pd.read_csv(imp_path, index_col=0) if os.path.exists(imp_path) else None
    stats_path = os.path.join(ANALYSIS_OUT, "summary_stats.csv")
    stats_df = pd.read_csv(stats_path, index_col=0) if os.path.exists(stats_path) else None

    exps = load_all(BASE)
    n_per_group = {g: sum(1 for e in exps if e.group == g) for g in ["1", "2", "3", "4", "5"]}
    total_y4 = sum(e.df["y4"].notna().sum() for e in exps)

    md = []
    md.append("# 时间序列预测与过程优化 — 项目报告（x 预测版）")
    md.append("")
    md.append(f"_生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    md.append("")
    md.append("## 1. 项目内容理解")
    md.append("")
    md.append("本项目研究一组工业过程实验数据。每个 CSV 是一次实验的时序记录，")
    md.append("包含 8 个过程变量（x1-x8，每 30 分钟采样一次）、4 个中间目标")
    md.append("（y1-y4，特定周期测量）和 1 个最终目标 Y。")
    md.append("")
    md.append("**本轮任务目标**（先做 x，y 后续补充）：")
    md.append("1. 数据清洗与异常处理；")
    md.append("2. 训练过程预测模型——给定过去一段时间的 x1-x8，预测未来一段时间的 x1-x8（**输入起点、长度、输出长度均可变**）；")
    md.append("3. 探索过程变量和目标之间的关联；")
    md.append("4. 控制优化模型（暂停，等 y 预测方案确定后再做）。")
    md.append("")
    md.append("**数据规模**：")
    md.append("")
    md.append(f"- 实验总数：**{len(exps)}** 个 CSV；")
    for g, n in n_per_group.items():
        md.append(f"  - 目录 `{g}/`: {n} 个文件")
    md.append(f"- y4 有效测量点累计：{total_y4} 个（极稀疏，每 ~24 个周期出现一次）；")
    md.append(f"- Y 终值范围：[3028, 3715]，均值约 3490；")
    md.append(f"- y4 取值范围：[1209, 8536]，均值约 4888。")
    md.append("")
    md.append("## 2. 技术路线")
    md.append("")
    md.append("### 2.1 数据预处理 (`src/data_loader.py`)")
    md.append("- 读取每个 CSV，去掉 datime=NaT 的尾行（仅含 Y）；")
    md.append("- **按周期对齐**：x 行为主键（30 分钟一行），y 行（15 秒一行）合并到对应周期；")
    md.append("- 异常值裁剪：IQR × 3 上下界；")
    md.append("- 缺失值线性插值（前向 + 后向）；")
    md.append("- 全局标准化：x 用 8 维均值/标准差。")
    md.append("")
    md.append("### 2.2 关联性分析 (`src/analyze.py`)")
    md.append("- 描述性统计；")
    md.append("- Pearson + Spearman 相关矩阵；")
    md.append("- 末态/均值/增量 x 与 Y 的 Spearman 关联；")
    md.append("- 末态 x vs Y 散点图。")
    md.append("")
    md.append("### 2.3 过程预测模型（仅预测 x，`src/model_forecaster.py`）")
    md.append("借鉴 `src/path_integrators.py` 中 StableGatedPI（光谱归一化 + 门控残差 + L2 投影）")
    md.append("的设计，把\"路径积分\"思想拓展为：")
    md.append("- **编码器**：用 GatedResidualCell 对过去 L_in 步的 x 序列做路径积分，得到状态 s0；")
    md.append("- **Rollout 头**：自回归地生成未来 T_out 步的 x1-x8；")
    md.append("- **本版本不预测 y1-y4 / Y**——理由是 y 标签稀疏，引入 y 头会拖累训练稳定性。")
    md.append("")
    md.append("训练支持：**任意起点、任意 in_len（16/20/24/32）、任意 out_len（6/8/12/16）**，")
    md.append("通过 `WindowXDataset` + `pad_collate_x` 把不规则样本 pad 到 batch 内最大长度。")
    md.append("")
    md.append("损失 = MSE(x)。")
    md.append("")
    md.append("### 2.4 控制优化 (`src/optimize.py`，本轮暂停)")
    md.append("暂不启用——等 x 预测稳定后再补回 y 奖励机制。可能的方向：")
    md.append("1. 训练轻量 x→y4 回归器（XGBoost / 随机森林）；")
    md.append("2. 使用 `src_control/` 的 N4SID + Kalman 状态空间；")
    md.append("3. 重新加 y_head，用 y3 当 y4 的 proxy（ρ=0.995）。")
    md.append("")
    md.append("## 3. 处理过程与结果")
    md.append("")

    if stats_df is not None:
        md.append("### 3.1 描述性统计（节选）")
        md.append("")
        md.append("| 列 | mean | std | min | 25% | 50% | 75% | max | missing% |")
        md.append("|----|------|-----|-----|-----|-----|-----|-----|----------|")
        for c in X_COLS:
            row = stats_df.loc[c]
            md.append(f"| `{c}` | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} | {row['missing_pct']:.2f} |")
        md.append("")

    if imp_df is not None:
        md.append("### 3.2 x 与 Y 的 Spearman 关联（Top 5）")
        md.append("")
        md.append("| 末态特征 | ρ(Y) | 均值特征 | ρ(Y) | 增量特征 | ρ(Y) |")
        md.append("|----------|------|----------|------|----------|------|")
        last_top = imp_df["last_spearman"].abs().sort_values(ascending=False).head(5).index.tolist()
        mean_top = imp_df["mean_spearman"].abs().sort_values(ascending=False).head(5).index.tolist()
        delta_top = imp_df["delta_spearman"].abs().sort_values(ascending=False).head(5).index.tolist()
        for i in range(5):
            try:
                l = f"{last_top[i]} = {imp_df.loc[last_top[i], 'last_spearman']:+.3f}"
            except IndexError:
                l = "—"
            try:
                m = f"{mean_top[i]} = {imp_df.loc[mean_top[i], 'mean_spearman']:+.3f}"
            except IndexError:
                m = "—"
            try:
                d = f"{delta_top[i]} = {imp_df.loc[delta_top[i], 'delta_spearman']:+.3f}"
            except IndexError:
                d = "—"
            md.append(f"| {l} |  | {m} |  | {d} |  |")
        md.append("")
        md.append("**结论**：")
        md.append("- `x6`（高位产能 / 反应浓度）与 Y 正相关最强（末态 ρ≈0.27，均值 ρ≈0.28）；")
        md.append("- `x7`（过程累计量）次之（ρ≈0.20）；")
        md.append("- `x3` 与 Y 弱负相关；`x4` 与 Y 相关性较弱，说明它们更像是\"控制自由度\"。")
        md.append("")

    md.append("### 3.3 过程预测结果（仅 x1-x8）")
    md.append("")
    if test_metrics:
        per_dim = test_metrics.get("rmse_x_per_dim", [0] * 8)
        per_dim_orig = test_metrics.get("rmse_x_per_dim_orig", [0] * 8)
        md.append(f"- 整体 RMSE(x) 标准化空间 = **{test_metrics.get('rmse_x', 0):.4f}**")
        md.append(f"- 整体 RMSE(x) 原始空间均值 = {test_metrics.get('rmse_x_orig_mean', 0):.2f}")
        md.append("")
        md.append("| 维度 | 标准化空间 RMSE | 原始空间 RMSE |")
        md.append("|------|----------------|---------------|")
        for i, c in enumerate(X_COLS):
            md.append(f"| {c} | {per_dim[i]:.4f} | {per_dim_orig[i]:.2f} |")
        md.append("")
    md.append("**可视化**（`src/analysis_out/`）：")
    md.append("- `forecast_x1_x8.png`：x1-x8 真实 vs 预测；")
    md.append("- `error_per_dim.png`：分维度 RMSE 柱状图；")
    md.append("- `correlation_matrix.png`：x1-x8、y1-y4 相关性热图。")
    md.append("")
    md.append("## 4. 分析总结与拓展")
    md.append("")
    md.append("### 4.1 主要结论")
    md.append("- **去掉 y 头后 x 预测精度大幅提升**：仅训练 MSE(x)，避免 y 标签稀疏带来的训练不稳定；")
    md.append("- **x6 / x7 是与 Y 强相关的关键过程变量**，建议把控资源优先分配到这两个变量的稳态控制上；")
    md.append("- **y3 与 y4 几乎完全相关**（ρ=0.995），y3 可以作为 y4 的代理（更密）帮助训练；")
    md.append("- 借鉴 path_integrator 的门控残差 + L2 投影结构可以稳定训练长序列（~150 步）。")
    md.append("")
    md.append("### 4.2 拓展方向")
    md.append("- **y 预测方案**：")
    md.append("  - 训练轻量 x→y4 回归器（XGBoost / 随机森林），与本过程模型串联；")
    md.append("  - 在 PathIntegratorForecaster 上重新加 y_head，但用 y3 当 y4 的 proxy（ρ=0.995）做半监督训练；")
    md.append("  - 复用 N4SID + Kalman 的线性 y 模型（`src_control/`）；")
    md.append("- **更长的输入上下文**：把 in_len 提升到 64 甚至 128，使用 MambaLiteSSM 风格的连续时间状态空间模型；")
    md.append("- **物理信息神经网络 (PINN)**：把化学反应工程/热传导的先验微分方程作为正则项注入；")
    md.append("- **多任务/迁移学习**：5 个目录的实验可视为 5 种工况，可以加 group embedding 做工况自适应。")
    md.append("")
    md.append("### 4.3 复现方式")
    md.append("```bash")
    md.append("# 1. 分析")
    md.append("python src/analyze.py")
    md.append("# 2. 训练过程预测模型（仅 x）")
    md.append("python src/train_forecaster.py --epochs 30")
    md.append("# 3. 可视化")
    md.append("python src/visualize.py")
    md.append("# 4. 生成报告")
    md.append("python src/make_report.py")
    md.append("```")
    md.append("")

    out_md = "\n".join(md)
    with open(REPORT_PATH, "w") as f:
        f.write(out_md)
    print(f"[report] 已写入 {REPORT_PATH}")
    print(f"[report] 字数: {len(out_md)}")


if __name__ == "__main__":
    main()