# `src/` — 时间序列预测 + 过程优化

本目录包含项目的核心代码（数据加载、关联分析、过程预测模型、训练、控制优化、可视化、报告生成）。
所有脚本都设计为可直接 `python <script.py>` 运行，复现实验只需按下列顺序执行几条命令。

---

## 目录结构

```
src/
├── path_integrators.py              # 借鉴参考：5 种路径积分单元（StableGatedPI / MambaLiteSSM 等）
├── train_path_integrators_gemini.py # 借鉴参考：在 gridworld 上对路径积分做的 benchmark（结构风格参考来源）
│
├── data_loader.py                  # ★ 数据加载/对齐/清洗/滑窗采样
├── analyze.py                      # ★ 关联性分析 + 描述性统计 + 相关性热图
├── model_forecaster.py             # ★ 过程预测模型：路径积分编码器 + 自回归 rollout 头
├── train_forecaster.py             # ★ 训练循环（支持变 in_len/out_len/起点）
├── optimize.py                     # ★ 控制优化：固定过去 → 搜索 x3/x4/x6/x8 → 最大化 y4
├── visualize.py                    # ★ 可视化：x1-x8 预测、y1-y4 预测、优化前后对比
├── per_file_prediction_stats.py    # ★ 每个文件的预测偏差统计（按文件夹分组）
├── make_report.py                  # ★ 生成综合 Markdown 报告 (REPORT.md)
│
├── REPORT.md                       # 由 make_report.py 生成
├── analysis_out/                   # analyze.py + visualize.py 的输出（图片、CSV）
│   ├── summary_stats.csv
│   ├── correlation_matrix.png
│   ├── x_last_vs_Y.png
│   ├── feature_vs_Y_importance.csv
│   ├── y_correlation.csv
│   ├── forecast_x1_x8.png
│   ├── forecast_y1_y4.png
│   └── optimization_compare.png
└── model_out/                      # 训练产物
    ├── forecaster_best.pt          # 最优 ckpt（已 gitignore）
    ├── scalers.npz                 # x/y 标准化器
    ├── test_metrics.json
    ├── test_predictions.npz
    └── optimization_results.json
```

带 `★` 的脚本是项目本体；其余是参考/借鉴文件。

---

## 复现方式（Quick Start）

> 所有命令均假设在仓库根目录 `/kefu-nas/ybkong/time_serials-master/` 下执行。
> Python ≥ 3.10，PyTorch ≥ 2.0，需要 `numpy pandas matplotlib torch`。

### 1. 安装依赖

```bash
pip install numpy pandas matplotlib torch
```

### 2. 关联性分析（必跑第一步）

```bash
python src/analyze.py
```

**输出**（`src/analysis_out/`）：

- `summary_stats.csv` — x1-x8 全局描述统计
- `correlation_matrix.csv` / `.png` — x1-x8 / y1-y4 相关性热图
- `x_last_vs_Y.png` — 末态 x vs Y 散点
- `feature_vs_Y_importance.csv` — Spearman 关联排序
- `y_correlation.csv` — y1-y4 互相关

### 3. 训练过程预测模型

```bash
python src/train_forecaster.py --epochs 25 --batch-size 32 --y4-boost 3.0
```

常用参数：

| 参数             | 默认 | 说明             |
| ---------------- | ---- | ---------------- |
| `--epochs`     | 30   | 训练轮数         |
| `--batch-size` | 32   | batch 大小       |
| `--lr`         | 2e-3 | Adam 学习率      |
| `--dim-state`  | 128  | 路径积分状态维度 |
| `--hidden`     | 128  | 隐层宽度         |
| `--device`     | auto | cuda / cpu       |

> **本版本只预测 x1-x8**——不再预测 y1-y4 / Y。损失 = `MSE(x)`。
> 把 y 头去掉之后，训练显著更稳定（之前带 y 头 25 epoch 测试集 RMSE(x)=89；现在 2 epoch 已经到 0.45）。

训练支持**任意起点、任意 in_len（16/20/24/32）、任意 out_len（6/8/12/16）** 的样本。

**输出**（`src/model_out/`）：

- `forecaster_best.pt` — 最优 ckpt（含模型权重 + x 标准化器）
- `scalers.npz` — x 标准化器
- `test_metrics.json` — 测试集 RMSE（含分维度）
- `test_predictions.npz` — 测试集预测结果

### 4. 控制优化（**本轮暂停**）

```bash
python src/optimize.py   # 仅打印"已停用"提示
```

暂不执行实际优化——因为过程预测模型已去掉 y_head / Y_head，没有预测 y 的能力。
等 x 预测精度稳定后，再选定 y 预测方案（轻量回归器 / N4SID 状态空间 / 重新加 y_head 用 y3 当 proxy）后重启。

### 5. 可视化

```bash
python src/visualize.py --base-dir /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac
```

> ⚠️ `visualize.py` 的 `--base-dir` 默认值是写死的旧路径 `/kefu-nas/ybkong/time_serials-master`，
> 不会自动指向当前仓库。务必显式传入 `--base-dir <仓库根目录>`，否则会报
> `FileNotFoundError: 找不到 checkpoint`。
>
> 可选参数：`--backbone lstm|pathint`（默认 `lstm`）、`--mode shared|group_head|independent`（默认 `group_head`）、
> `--ckpt <路径>` 手动指定 checkpoint（否则自动在 `src/model_out/` 里按命名规则查找）。

**两种预测锚定方式**：

1. **默认（输入锚定）**：输入窗口恒为序列开头 `in_len` 步，预测紧接输入之后 `T_out` 步。
2. **尾部锚定（`--tail-anchor`）**：`s = T − H`，历史画 `[0, s)`，预测画 `[s, s+H)`，
   输入窗口取 `s` 前最后 `in_len` 步——与 `scripts_control/06_visualize` 的外推语义一致。

```bash
# 尾部锚定，且 in_len/视野 与 scripts_control 的 --context 32 --horizon 32 对齐
python src/visualize.py --base-dir <仓库根目录> \
    --backbone lstm --mode group_head \
    --tail-anchor --in-len 32 --t-out 32
```

新增参数：

| 参数             | 默认     | 说明             |
| ---------------- | -------- | ---------------- |
| `--in-len`     | 24       | 输入窗口长度（历史/上下文步数） |
| `--t-out`      | 16       | 预测视野 H（外推步数）         |
| `--tail-anchor`| 关       | 尾部锚定（见上）；开启时忽略 `--pred-start` |
| `--pred-start` | 输入长度 | 预测区间起点（仅非 tail-anchor 时有效） |
| `--split`      | test     | 用哪个 split 画图 + 算 RMSE：train/val/test（非 test 时输出名加 `_{split}` 后缀） |

**六路模型命令**（`--backbone × --mode` → 6 个模型类；每条命令都可追加 `--tail-anchor` 做尾部锚定）：

| backbone | mode | 模型类 | 命令 |
| --- | --- | --- | --- |
| `lstm` | `shared` | `LSTMForecaster` | `python src/visualize.py --backbone lstm --mode shared` |
| `lstm` | `group_head` | `LSTMForecasterFiLM` | `python src/visualize.py --backbone lstm --mode group_head` |
| `lstm` | `independent` | `LSTMForecaster5Models` | `python src/visualize.py --backbone lstm --mode independent` |
| `pathint` | `shared` | `PathIntegratorForecaster` | `python src/visualize.py --backbone pathint --mode shared` |
| `pathint` | `group_head` | `PathIntegratorForecasterFiLM` | `python src/visualize.py --backbone pathint --mode group_head` |
| `pathint` | `independent` | `PathIntegratorForecaster5Models` | `python src/visualize.py --backbone pathint --mode independent` |

例如尾部锚定 + 长上下文外推：

```bash
python src/visualize.py --base-dir /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac \
    --backbone lstm --mode group_head \
    --tail-anchor --in-len 32 --t-out 32
```

**输出**（`src/analysis_out/`）：

- `forecast_x1_x8_{backbone}_{mode}.png` — 真实 vs 预测的 x1-x8
- `error_per_dim_{backbone}_{mode}.png` — 分维度 RMSE 柱状图

### 5.5. 每个文件预测偏差统计（按文件夹分组）

```bash
python src/per_file_prediction_stats.py
```

**输出**（`src/analysis_out/`）：

- `per_file_rmse_boxplot.png` — 按文件夹（dir 1-5）分组的整体 RMSE 箱线图
- `per_file_rmse_scatter.png` — 每个文件的散点图（按 group 着色，标出 dir 区间）
- `per_dim_rmse_boxplot.png` — 8 个维度按文件夹分箱的误差分布

**用途**：发现哪些文件/哪些维度预测差。
本项目实证结论：dir 4（绿色）最差，x4 / x5 这两个维度最难预测。

### 6. 生成综合报告

```bash
python src/make_report.py
```

**输出**：[`src/REPORT.md`](src/REPORT.md) — 4 节汇报内容（项目理解 / 技术路线 / 处理过程 / 分析总结）。

---

## 一键复现（合集脚本）

```bash
python src/analyze.py \
  && python src/train_forecaster.py --epochs 30 \
  && python src/visualize.py \
  && python src/per_file_prediction_stats.py \
  && python src/make_report.py
```

> 优化步骤本轮跳过，等 y 预测方案确定后再加。

---

## 模型与算法要点

### 过程预测模型（[model_forecaster.py](model_forecaster.py)）

借鉴 [`path_integrators.StableGatedPI`](../README.md) 的设计：

- **编码器**：`GatedResidualCell` —— spectral_norm + 门控残差 + L2 投影 + LayerNorm，
  对过去 L_in 步的 x 序列做路径积分得到状态 s0；
- **Rollout 头**：自回归生成未来 T_out 步的 x1-x8；
- **本版本不预测 y1-y4 / Y**——理由是 y 标签稀疏（y4 每 ~24 步一次），引入 y 头会拖累训练稳定性；
- 训练损失 = `MSE(x)`。

### 控制优化（[optimize.py](optimize.py)，本轮暂停）

暂不启用——等 x 预测稳定后再补回 y 奖励机制。可能的方向：

1. 训练轻量 x→y4 回归器（XGBoost / 随机森林），与过程模型串联；
2. 复用 N4SID + Kalman 的线性 y 模型（`src_control/`）；
3. 重新加 y_head，用 y3 当 y4 的 proxy（ρ=0.995）做半监督训练。

---

## 已知性能瓶颈

| 现象                      | 数值                | 原因                     | 改进方向                             |
| ------------------------- | ------------------- | ------------------------ | ------------------------------------ |
| 自回归长 horizon 累积漂移 | T_out=16 时误差累积 | 自回归预测误差逐级叠加   | 加 teacher forcing；限制 T_out ≤ 12 |
| 跨 group 泛化             | 未显式建模          | 5 个目录可能为 5 种工况  | 加 group embedding 或分层标准化      |
| y 预测能力                | 本轮不预测          | y 标签稀疏 → 训练不稳定 | 待 x 稳定后选 y 预测方案             |

---

## 文件依赖关系

```
data_loader.py ──┬──> analyze.py ────────> analysis_out/*
                 ├──> train_forecaster.py ───> model_out/forecaster_best.pt
                 └──> optimize.py ────────────> model_out/optimization_results.json
                                                       │
model_forecaster.py <──── train_forecaster.py           │
                                                       ▼
                                                  visualize.py ──> analysis_out/forecast_*.png
                                                       │
                                                       ▼
                                per_file_prediction_stats.py ──> analysis_out/per_file_*.png
                                                       │
                                                       ▼
                                                make_report.py ──> REPORT.md
```
