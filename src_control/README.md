# src_control — 现代控制理论管道（多元时序预测 + 约束优化）

本目录包含 README 任务的全部代码实现：**数据分析与清洗 → 特征分析 → 状态空间预测模型 → MPC Pareto 优化**。采用现代控制理论方法（线性状态空间 N4SID + Kalman + 混合 NN 残差 + L-BFGS Pareto MPC），与项目现有 `src/` 目录**完全独立、无冲突**。

---

## 目录结构

```
src_control/
├── config.py                  全局配置（路径、超参、种子、设备）
├── data_loader.py             CSV 发现与标准解析（按内容而非时间格式分类行）
├── preprocess.py              异常检测、缺失值填充、标准化、80/20 划分
├── import_legacy.py           注入 ../src 路径以只读复用 path_integrators
├── analysis/
│   └── correlation.py         Pearson / MI / PCA / Granger / 滞后互相关
├── models/
│   ├── state_space.py         N4SID 子空间辨识 + Kalman 滤波（纯 numpy）
│   └── state_space_nn.py      混合 SS + NN 残差预测模型
├── optimization/
│   └── mpc_optimizer.py       多起点 L-BFGS Pareto MPC
├── visualization/
│   └── plots.py               统一可视化（预测叠加、Pareto、轨迹）
└── utils/
    ├── seed.py                随机种子
    └── metrics.py             MSE / MAE / MAPE / R²

scripts_control/               CLI 驱动脚本（同样不与现有 scripts/ 冲突）
├── 03_train_predictor.py      训练预测模型
├── 04_optimize.py             运行 Pareto MPC
└── 05_smoke_test.py           端到端冒烟测试

tests_control/                 pytest 单元测试（24 项）

ppt/outline.md                 PPT 大纲与模块映射
```

---

## 运行流程

完整复现按以下顺序执行五个步骤。

### 步骤 1 — 数据预处理

将 171 个 CSV 解析为对齐、清洗、标准化后的 npz 张量：

```bash
python -m src_control.preprocess --out data/processed --seq-len 320
```

输出：
- `data/processed/aligned_dataset.npz` —— 全部 171 样本
- `data/processed/train.npz`         —— 训练集（137 样本）
- `data/processed/test.npz`          —— 测试集（34 样本）
- `data/processed/scalers.npz`       —— 标准化参数

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

### 步骤 3 — 训练预测模型

用 N4SID 初始化线性 SS，再用 AdamW + 余弦 LR 训练混合 SS-NN 模型（mask-aware MSE + 半 teacher forcing）：

```bash
python -m scripts_control.03_train_predictor \
    --data  data/processed/train.npz \
    --test  data/processed/test.npz  \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 \
    --tf-decay 50 --patience 30
```

输出：
- `checkpoints/ss_nn_best.pt`     —— 最佳验证集权重
- `checkpoints/ss_nn_last.pt`     —— 末 epoch 权重
- `results/metrics/training_log.json`
- `results/metrics/test_metrics.json`  —— 各变量 R²、MSE、MAE
- `results/predictions/test_predictions.npz`

快速冒烟训练（用于 CI，2–3 分钟）：

```bash
python -m scripts_control.03_train_predictor \
    --epochs 5 --bs 16 --tf-decay 2 --patience 5
```

### 步骤 4 — MPC Pareto 优化

加载最佳模型，对测试集多个样本进行 `(x3, x4, x6, x8)` 优化，扫描 5 组权重生成 Pareto 前沿：

```bash
python -m scripts_control.04_optimize \
    --ckpt checkpoints/ss_nn_best.pt \
    --data data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --n-samples 10 --horizon 16 --n-starts 3
```

输出：
- `results/metrics/pareto.json`                          —— Pareto 点 + 基线
- `results/figures/pareto_frontier.png`                   —— 前沿散点图
- `results/figures/optimized_vs_baseline_<idx>.png`       —— 单样本轨迹对比

### 步骤 5 — 端到端冒烟测试

在临时目录跑完整流水线（5 epoch 训练 + 1 样本优化），验证所有输出文件存在：

```bash
python -m scripts_control.05_smoke_test
```

期望 ~5 分钟内完成，无错误退出。

### 步骤 6 — 综合可视化（拟合误差 + 优化对比）

加载最佳模型与测试集预测，画 4 张 PNG 到 `src_control/analysis_out/`：

```bash
python -m scripts_control.06_visualize \
    --ckpt checkpoints/ss_nn_best.pt \
    --test data/processed/test.npz \
    --preds results/predictions/test_predictions.npz \
    --scalers data/processed/scalers.npz \
    --pareto results/metrics/pareto.json
```

输出（默认到 `src_control/analysis_out/`，可用 `--out-dir` 覆盖）：
- `forecast_x1_x8.png`     —— 2 个测试样本的 x1..x8 历史 + 真实 vs 预测
- `forecast_y1_y4.png`     —— y1..y4 历史 + 真实 vs 预测
- `error_distribution.png` —— 残差直方图 + 真实-预测散点（含 MAE / RMSE / R²）
- `optimization_compare.png` —— MPC 优化前后 y4 Σ 对比，Pareto 前沿高亮

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
| 预处理 | `SEQ_LEN` | 320 | 序列填充 / 截断长度 |
|  | `TRAIN_RATIO` | 0.8 | 训练集比例 |
|  | `Z_THRESHOLD` | 5.0 | 异常 z-score 阈值 |
| 模型 | `N_STATE` | 16 | 线性 SS 状态维度 |
|  | `HIDDEN` | 128 | 残差 MLP 隐藏层 |
| 训练 | `LR` | 1e-3 | 初始学习率 |
|  | `BS` | 16 | 批大小 |
|  | `EPOCHS` | 200 | 训练 epoch |
|  | `TEACHER_FORCING_DECAY` | 50 | teacher forcing 衰减 epoch |
|  | `PATIENCE` | 30 | 早停耐心 |
| 优化 | `OPT_HORIZON` | 16 | MPC 预测视野 |
|  | `OPT_N_STARTS` | 5 | 多起点数量 |
|  | `OPT_MAX_ITER` | 200 | L-BFGS 最大迭代 |

修改后请同步更新 `config.py`。
