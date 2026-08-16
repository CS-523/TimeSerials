# src_control — 现代控制理论管道（多元时序预测 + 约束优化）

本目录包含 README 任务的全部代码实现：**数据分析与清洗 → 特征分析 → 状态空间预测模型 → MPC Pareto 优化**。采用现代控制理论方法（线性状态空间 N4SID + Kalman + 混合 NN 残差 + YHead + L-BFGS Pareto MPC），与项目现有 `src/` 目录**完全独立、无冲突**。

---

## 目录结构

```
src_control/
├── config.py                  全局配置（路径、超参、种子、设备、变量范围）
├── data_loader.py             CSV 发现与标准解析（按内容而非时间格式分类行）
├── preprocess.py              异常检测、缺失值填充、标准化、80/20 划分
├── import_legacy.py           注入 ../src 路径以只读复用 path_integrators
├── analysis/
│   └── correlation.py         Pearson / MI / PCA / Granger / 滞后互相关
├── models/
│   ├── state_space.py         N4SID 子空间辨识 + Kalman 滤波（纯 numpy）
│   └── state_space_nn.py      混合 SS + NN 残差预测模型（SS_NN_Hybrid / YHead）
├── optimization/
│   └── mpc_optimizer.py       多起点 L-BFGS Pareto MPC
├── visualization/
│   └── plots.py               统一可视化（预测叠加、Pareto、轨迹）
└── utils/
    ├── seed.py                随机种子
    └── metrics.py             MSE / MAE / MAPE / R²

scripts_control/               CLI 驱动脚本（同样不与现有 scripts/ 冲突）
├── 03_train_predictor.py      训练 y1–y4 预测模型（线性 SS + NN 残差 + YHead）
├── 08_train_x_model.py        训练独立的 x1–x8 多步外推预测模型
├── 04_optimize.py             运行 Pareto MPC（决策变量 x3/x4/x6/x8）
├── 06_visualize.py            综合可视化（y + 可选的 x̂ 外推）
├── 07_ppt_figures.py          生成 8 张中文标注的 PPT 图
└── 05_smoke_test.py           端到端冒烟测试（临时目录，4 步）

tests_control/                 pytest 单元测试（24 项）

ppt/outline.md                 PPT 大纲与模块映射
```

---

## 运行流程

完整复现按以下顺序执行（步骤 3.5 为可选）。

### 步骤 1 — 数据预处理

将 171 个 CSV 解析为对齐、清洗、标准化后的 npz 张量：

```bash
python -m src_control.preprocess --out data/processed --seq-len 320
```

> `--seq-len` 的 CLI 默认值是 **64**；`config.py` 推荐 **320**（> 数据集中序列长度的 p95，截断 + 填充到该长度）。完整复现请显式传 `--seq-len 320`。另有 `--root`（CSV 所在目录，默认取 `config.ROOT`）、`--ratio`（训练比例，默认 0.8）。

输出：
- `data/processed/aligned_dataset.npz` —— 全部 171 样本
- `data/processed/train.npz`         —— 训练集（137 样本）
- `data/processed/test.npz`          —— 测试集（34 样本）
- `data/processed/scalers.npz`       —— 标准化参数（x/y 的 mean、scale）

### 步骤 2 — 特征分析

计算相关矩阵、互信息、PCA、Granger 因果和滞后互相关，输出 5 张 PNG + JSON 报告：

```bash
python -m src_control.analysis.correlation \
    --data data/processed/train.npz \
    --out results/figures
```

输出：
- `results/figures/correlation_heatmap.png`
- `results/figures/mi_heatmap.png`
- `results/figures/pca_scree.png`
- `results/figures/lag_x_to_y4.png`
- `results/figures/granger_xy.png`
- `results/figures/analysis_report.json`

### 步骤 3 — 训练预测 y1-y4 模型

用 N4SID 初始化线性 SS，再用 AdamW + 余弦退火训练混合 SS-NN 模型（**纯 x → y 前馈**，mask-aware MSE，无 teacher forcing）：

```bash
python -m scripts_control.03_train_predictor \
    --data data/processed/train.npz \
    --test data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 --patience 30
```

模型结构：`SS_NN_Hybrid(dim_u=8, dim_y=4, n_state=16, hidden=128, window=4)` —— x1–x8 是外生输入，`LinearSSTorch` 线性 SS 基线（N4SID 初始化）+ `ResidualMLP` 残差，输出 y1–y4；外加 `YHead(window=8)` 从 y 末 8 步回归最终 `Y`（总损失 = y 的 mask-aware MSE + 0.1×Y 回归 MSE）。纯 x → y 前馈：只用 x1–x8 预测 y1–y4，无 teacher forcing、无 y 自回归反馈；checkpoint 为 `{"model":…, "yhead":…}` 字典。

关键训练参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--epochs` | 200 | 训练轮数 |
| `--bs` | 16 | 批大小 |
| `--lr` | 1e-3 | 初始学习率（AdamW, weight_decay=1e-4） |
| `--horizon` | 32 | （未使用，保留仅为兼容） |
| `--n-state` | 16 | 线性 SS 状态维度 |
| `--hidden` | 128 | 残差 MLP 隐藏层 |
| `--tf-decay` | 50 | （已废弃，保留仅为兼容） |
| `--patience` | 30 | 早停耐心 |
| `--val-ratio` | 0.15 | 验证集比例 |

输出：
- `checkpoints/ss_nn_best.pt`     —— 最佳验证集权重（`{"model","yhead"}` 字典）
- `checkpoints/ss_nn_last.pt`     —— 末 epoch 权重
- `results/metrics/training_log.json`
- `results/metrics/test_metrics.json`  —— y1–y4 各变量 MSE / MAE / R²
- `results/predictions/test_predictions.npz`

快速冒烟训练（用于 CI，2–3 分钟）：

```bash
python -m scripts_control.03_train_predictor \
    --epochs 5 --bs 16 --patience 5
```

### 步骤 3.5 — 训练 x1–x8 多步外推预测模型（可选）

默认模型只预测 y1–y4（x1–x8 是外生输入）。若要模型**也预测 x1–x8 的未来值**（多步外推），另训一个结构相同、输出维度换成 8 的独立模型（`SS_NN_Hybrid(dim_u=8, dim_y=8)`）：给定过去 C 步 x，预测未来 H 步 x。训练用 **teacher forcing**，推理用**真正的反馈自回归 rollout**（把模型自己上一时刻的预测喂回下一步）。**不影响 y 模型与 MPC。**

```bash
# 加 --out-root 可把模型输出一键放进 scripts_control/（默认在项目根 checkpoints/、results/）
python -m scripts_control.08_train_x_model \
    --data  data/processed/train.npz \
    --test  data/processed/test.npz  \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 \
    --context 32 --horizon 32 --tf-decay 50 --patience 30 \
    --out-root scripts_control
```

- `--context C` / `--horizon H`：上下文长度 / 外推步数——给定过去 C 步 x，预测未来 H 步（要求样本长度 ≥ C+H）
- `--tf-decay`：teacher-forcing 衰减 epoch 数（1 → 0）；训练每 batch 50/50 混合 TF 与真自回归
- `--out-root <dir>`：输出根目录，设置后模型输出统一放到 `<dir>/checkpoints`、`<dir>/results/metrics`、`<dir>/results/predictions`

输出（默认 `checkpoints/`、`results/`；`--out-root scripts_control` 时为 `scripts_control/checkpoints/`、`scripts_control/results/`）：
- `checkpoints/x_forecast_best.pt`     —— 最佳验证集权重（存的是**裸 state_dict**）
- `checkpoints/x_forecast_last.pt`     —— 末 epoch 权重
- `results/metrics/x_forecast_training_log.json`
- `results/metrics/x_forecast_metrics.json`  —— 各 x 变量 MSE / MAE / R²
- `results/predictions/test_x_forecast.npz`

### 步骤 4 — MPC Pareto 优化

加载最佳模型，对测试集多个样本优化决策变量 `(x3, x4, x6, x8)`（`x1/x2/x5` 固定，`x7` 为单调累积量不参与优化），扫描 5 组权重生成 Pareto 前沿：

```bash
python -m scripts_control.04_optimize \
    --ckpt checkpoints/ss_nn_best.pt \
    --data data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --n-samples 10 --horizon 16 --n-starts 3
```

> 决策变量范围由 `config.VAR_RANGES` 给出（如 x3 0–110、x4 26–36、x6 5k–50k、x8 0–1500）。5 组 Pareto 权重见 `config.OPT_WEIGHTS`。

输出：
- `results/metrics/pareto.json`                          —— Pareto 点 + 基线
- `results/figures/pareto_frontier.png`                   —— 前沿散点图
- `results/figures/optimized_vs_baseline_<idx>.png`       —— 单样本轨迹对比

### 步骤 5 — 端到端冒烟测试

在临时目录跑完整流水线（**预处理 → 特征分析 → 5-epoch 训练 → 1 样本优化**），验证所有关键输出文件存在：

```bash
python -m scripts_control.05_smoke_test
```

期望 ~5 分钟内完成，无错误退出。

### 步骤 6 — 综合可视化（拟合误差 + 优化对比）

加载最佳模型与测试集预测，画 4 张 PNG 到 `src_control/analysis_out/`。**y 模型和 x 模型都是可选的**——哪个 checkpoint 存在就画哪个：

```bash
# 模型产物在 scripts_control/ 下时（08 用了 --out-root scripts_control），这里同样加 --out-root
python -m scripts_control.06_visualize \
    --test data/processed/test.npz \
    --scalers data/processed/scalers.npz 
	#\
    #--out-root scripts_control
```

- `--out-root <dir>`：模型/产物查找根目录（与 08 的 `--out-root` 对应），`--ckpt`/`--x-ckpt`/`--preds`/`--pareto` 默认取 `<dir>/checkpoints/...`、`<dir>/results/...`；不传则用项目根 `checkpoints/`、`results/`

输出（默认到 `src_control/analysis_out/`，可用 `--out-dir` 覆盖）：
- `forecast_x1_x8.png`     —— 2 个测试样本的 x1..x8 历史 + 未来外推预测（若 x 模型存在则叠加红色虚线，浅灰点线为真实未来）；仅画 x，y 预测见下面独立图
- `forecast_y1_y4.png`     —— y1..y4 历史 + 真实 vs 预测（仅 y 模型存在时）
- `error_distribution.png` —— 残差直方图 + 真实-预测散点（含 MAE / RMSE / R²）
- `optimization_compare.png` —— MPC 优化前后 y4 Σ 对比，Pareto 前沿高亮

### 步骤 7 — PPT 图（中文标注）

生成 8 张中文标注的 PPT 图到 `src_control/analysis_out/ppt/`（`--figs` 可指定图号子集，如 `1,2,5`）：

```bash
python -m scripts_control.07_ppt_figures \
    --preds results/predictions/test_predictions.npz \
    --figs all
```

输出：`01_model_architecture.png`、`02_data_pipeline.png`、`03_n4sid_concept.png`、`04_training_strategy.png`、`05_prediction_overlay.png`、`06_error_analysis.png`、`07_mpc_optimization.png`、`08_summary_dashboard.png`

---

## 单元测试

24 项 pytest 覆盖数据加载、预处理、状态空间辨识、Kalman、混合模型、MPC 优化器：

```bash
pytest tests_control/
```

---

## 与现有 `src/` 的关系

`src_control/` 通过 `src_control/import_legacy.py` 把 `../src` 加入 `sys.path`，**只读复用** `src/path_integrators.py` 等模块。**没有任何对现有 `src/` 及其下文件的修改、重命名或覆盖**。两套代码可独立运行。

---

## 关键参数速查

| 组别 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| 预处理 | `SEQ_LEN` | 320 | 序列填充 / 截断长度（CLI `--seq-len` 默认 64，需显式传 320） |
|  | `TRAIN_RATIO` | 0.8 | 训练集比例 |
|  | `Z_THRESHOLD` | 5.0 | 异常 z-score 阈值 |
| 模型 | `N_STATE` | 16 | 线性 SS 状态维度 |
|  | `HIDDEN` | 128 | 残差 MLP 隐藏层 |
| 训练 | `LR` | 1e-3 | 初始学习率 |
|  | `BS` | 16 | 批大小 |
|  | `EPOCHS` | 200 | 训练 epoch |
|  | `TEACHER_FORCING_DECAY` | 50 | （已废弃） |
|  | `PATIENCE` | 30 | 早停耐心 |
| 优化 | `OPT_HORIZON` | 16 | MPC 预测视野 |
|  | `OPT_N_STARTS` | 5 | 多起点数量（CLI `--n-starts` 默认 3） |
|  | `OPT_MAX_ITER` | 200 | L-BFGS 最大迭代 |
|  | `DECISION_COLS` | x3, x4, x6, x8 | MPC 决策变量 |
|  | `FIXED_INPUT_COLS` | x1, x2, x5 | MPC 中固定不变的输入 |
|  | `VAR_RANGES` | 见 config | 各变量观测范围（MPC 边界） |

修改后请同步更新 `config.py`。
