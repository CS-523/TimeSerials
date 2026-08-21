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
├── train_y_poly.py                 # ★ 预测实验终值 Y（多项式回归 y1~y4 → Y）
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

### 3.5. 预测实验终值 Y（多项式回归 `train_y_poly.py`）

`train_y_poly.py` 不预测未来时序，而是用中间目标 y1~y4（+ 可选 x1~x8 衍生特征）做多项式回归，
拟合每个实验的**最终结果 Y**（标量）。管线：`StandardScaler → PolynomialFeatures(degree 2/3) → Ridge`。

```bash
python src/train_y_poly.py --degree 2 --mode last
python src/train_y_poly.py --degree 2 --mode window --window 4
python src/train_y_poly.py --degree 2 --per-group
```

三种模式区别（**特征定义见下节**）：

| 模式 | 特征构成 | 训练方式 |
| --- | --- | --- |
| `--mode last` | y1~y4 各自的**最后一个有效观测值** → 4 特征 | **1 个全局模型**（train+val 训练 / test 评估） |
| `--mode window` | y1~y4 最后 N 个**有观测的行**，按行展平 → N×4 特征 | **1 个全局模型** |
| `--per-group` | 每组 36 候选特征里挑 top-8（Pearson \|r\|） | **5 个独立模型**（每组一个，每组内 80/20 划分） |

> ⚠️ **`--per-group` 不是"一个全局模型 + 分组评估"，而是「为每组各训练一个独立的 Ridge 模型」**。最终指标按组汇总（pooled RMSE/MAE/R²）。5 个组 → 5 个 `pipe` → 5 套独立的 `StandardScaler/PolynomialFeatures/Ridge` 参数。

常用参数：`--degree {2,3}`（多项式阶数）、`--alpha`（默认 RidgeCV 自动选）、
`--drop-y4`（去掉与 y3 共线的 y4）、`--seed`、`--base-dir`、`--out-dir`。

#### 特征精确定义

**前置说明**：y1~y4 在数据中**稀疏**——并不是每行都有观测，可能连续几行都是 NaN，偶发一行 4 个 y 都有值。"有效" = 该行对应 y 列为非 NaN。

1. **`_last_valid(series)`**（`train_y_poly.py:66-69`）：
   ```python
   def _last_valid(series):
       s = series.dropna()           # 先丢掉 NaN
       return float(s.iloc[-1]) if len(s) > 0 else np.nan
   ```
   → **"最后一个有效值" = 该列 y 时间序列里最后一个非 NaN 的值**（不是时间上的"末尾行"，而是观测上的"末尾有效观测"）。例：y1 列在 [0, 2, 5, 8, 9, 10, ..., 47, 49] 行有观测，则 y1 的"最后一个有效值"是第 49 行的值。
   - 每个 y1~y4 各自取自己的"最后一个有效值"，可能来自**不同的时间步**。
   - 全列 NaN 时返回 NaN，该实验被丢弃。

2. **`--mode last` 的特征**：`(y1_last, y2_last, y3_last, y4_last)`，4 维。

3. **`--mode window --window N` 的特征**（`extract_features_window`，`train_y_poly.py:124-147`）：
   - 第 1 步：从每个实验的 y1~y4 时序里挑出**至少有一个 y 被观测到的行**（`dropna(how="all", subset=y_cols)`）。
   - 第 2 步：取这些行的**最后 N 行**（`obs_rows.tail(N)`）。这是 N 个**观测时刻**，不是时间上等距的 N 步。
   - 第 3 步：每行内 NaN → 先 `ffill`（沿列方向取最近一次观测）；仍 NaN 的填 0。`tail` 现在是 N×4 矩阵。
   - 第 4 步：若该实验的观测行数 < N，前端补 0 行（`pad`）至 N 行。
   - 第 5 步：N×4 矩阵 `flatten()` → 长度 4N 的一维特征向量。
   - 顺序：从最早到最近（补的 0 行在最前，最后 N 行观测按时间顺序排），然后按 `y1,y2,y3,y4` 列内展平。
   - **默认 `--window 4` → 16 维**；`--window 8` → 32 维。

4. **`--per-group` 的特征**（`extract_engineered_features`，`train_y_poly.py:78-104`）：
   - 对 **12 列**（y1~y4 + x1~x8）各算 3 个衍生量 = 36 个候选：
     - `__last`：`_last_valid(series)`，见上。
     - `__mean`：全列均值（NaN 忽略）。
     - `__delta`：`_last_valid − _first_valid`（首末有效值之差，反映累计变化量）。
   - 丢掉 NaN 占比 >10% 的列（实际几乎不丢）。
   - 对**每个组**用组内 Pearson \|r\| 与 Y 的相关性排序，**取前 8 个**作为该组的特征（`GROUP_FEATURES` 字典硬编码，见下表）。
   - **每组用自己的 8 维特征 + 自己的 Ridge 模型**。

#### 各模式一键对比（`run_y_poly_compare.sh`）

```bash
bash run_y_poly_compare.sh   # 在仓库根目录执行
```

脚本循环跑 8 个组合（last/window/per-group × degree 2/3 + drop-y4 变体），
输出到 `src/model_out_compare/`，并在末尾打印对比表（组合可在脚本顶部 `RUNS` 数组里增删）。

| degree | mode | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| 2 | per-group | **67.1** | **53.9** | 0.298 |
| 3 | window n=8 | 84.1 | 71.8 | **0.491** |
| 2 | last | 101.5 | 76.0 | 0.257 |
| 2 | last (drop-y4) | 108.2 | 81.1 | 0.157 |
| 2 | window n=4 | 113.2 | 82.6 | 0.076 |
| 2 | window n=8 | 115.1 | 77.8 | 0.046 |
| 3 | last | 116.1 | 82.8 | 0.028 |
| 3 | per-group | 249.2 | 92.2 | -8.677 |

**结论**：

- `per-group`(deg2) RMSE 最低（67.1）——「每组专属特征 + 5 个独立模型」有效；
- `window n=8 + deg3` 的 R² 最高(0.491)，是**全局模型**里最均衡的选择——长窗口 + 三次项捕捉到更复杂的"y 末态 → Y 终值"关系；
- `--drop-y4` 反而更差（R² 0.257→0.157）——y4 信息有价值，不建议去掉；
- window deg2 比 last 更差：特征维度从 4 涨到 16/32，但 RidgeCV 自动选的 α 过大（100~1000）导致欠拟合；只有升到 deg3 才真正发挥窗口优势；
- `per-group deg3` 严重过拟合（G2 RMSE 达 524）——分组样本少（约 20~34 个/组）却塞进 165 个多项式特征，**每组都不建议用三次项**。

#### 模型结构（`train_y_poly.py`）

**注意：本模型是经典统计学习回归（不是深度神经网络）**——`sklearn.Pipeline` 三段式：

```
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ StandardScaler   │ →  │ PolynomialFeatures   │ →  │ Ridge / RidgeCV      │
│ (x − μ) / σ      │    │ (degree 2/3,  含 bias)│    │ 线性 + L2 正则       │
└──────────────────┘    └──────────────────────┘    └──────────────────────┘
        ↑                          ↑                          ↑
   4 / 16 / 32 / 8 维        多项式展开                    输出 1 维 Y
```

**3 段详解**：

| 阶段 | 类 | 作用 | 输出维度（4 特征示例） |
| --- | --- | --- | --- |
| 1. 标准化 | `StandardScaler` | `x_std = (x − μ) / σ`；`μ/σ` 仅在 train+val 上 fit | 4 维 |
| 2. 多项式展开 | `PolynomialFeatures(degree=d, include_bias=True)` | 生成 `1, x, x², x₁x₂, x³, x₁²x₂ …`；**含 1 阶偏置** | d=2 → 15 维；d=3 → 35 维 |
| 3. 岭回归 | `Ridge(alpha)` 或 `RidgeCV(alphas=[0.01, 0.1, 1, 10, 50, 100, 500, 1000])` | 线性回归 + L2 正则；RidgeCV 自动按 LOO-MSE 选 α | 1 维（Y） |

**输入维度按模式变化**：

| 模式 | 原始特征数 | deg=2 展开后 | deg=3 展开后 |
| --- | --- | --- | --- |
| `--mode last`（y1~y4 末值） | 4 | 15 | 35 |
| `--mode window --window 4` | 16 | 153 | 969 |
| `--mode window --window 8` | 32 | 561 | 6545 |
| `--per-group`（每组 top-8） | 8 | 45 | 165 |

**Per-group = 5 个独立模型**（重点）：

```
all 171 exps
   │
   ├─ group 1 (~30 exps) ──[80/20 split]──▶ pipe_g1  (自己的 StandardScaler + Poly(8→45) + Ridge(45→1))
   ├─ group 2 (~34 exps) ──[80/20 split]──▶ pipe_g2  (独立的一套参数)
   ├─ group 3 (~28 exps) ──[80/20 split]──▶ pipe_g3
   ├─ group 4 (~40 exps) ──[80/20 split]──▶ pipe_g4
   └─ group 5 (~39 exps) ──[80/20 split]──▶ pipe_g5
   │
   └─▶ 把 5 组的 test 预测 concat ──▶ 整体 pooled RMSE/MAE/R²
```

每个 `pipe_g` 有自己的 `mean/std/poly_features/ridge.coef_`；`joblib.dump` 出来是 `{"pipes": {1:pipe, 2:pipe, 3:pipe, 4:pipe, 5:pipe}, "group_features": GROUP_FEATURES}`。预测新数据时需知道它属于哪个组，再调对应 `pipe`。

**Per-group 特征表**（`GROUP_FEATURES` 字典，代码 `train_y_poly.py:57-63`）：

| 组 | 选取特征（每组 8 个，此处列前 5） | top 特征含义 |
| --- | --- | --- |
| 1 | y4__last, y3__mean, x2__mean, y3__last, x7__mean | **y 末值主导**（y3/y4 相关性最强） |
| 2 | y4__last, y3__last, x2__mean, y3__mean, y4__mean | **y 末值主导**（同 G1，但 y4__mean 进了 top） |
| 3 | x8__delta, x8__last, x6__last, x6__delta, x4__delta | **x 衍生主导**（无 y 入 top；x8 累计变化最重要） |
| 4 | y1__last, x2__delta, x2__last, x6__delta, x5__mean | **y1 + x2 主导**（与 G1/G2/G5 不同，y4 没进 top） |
| 5 | y4__last, x2__mean, y1__mean, x6__mean, y2__mean | y4 + x2/x6 均值（多元） |

每组从 `last/mean/delta` 三个衍生（每列 3 个）共 36 个候选里按组内 Pearson |r| 排序选前 8 个；**不同组可能选完全不同的特征子集**——这正是 per-group 模式的核心价值。

**模型参数量**（线性模型，参数即多项式系数 + 截距）：

- `--mode last, deg=2`：15 系数 + 1 截距 = 16 个；
- `--per-group, deg=2`：每组 45+1 = 46 个（5 组共 230 个）；
- `--mode window 8, deg=3`：**6546 个**（参数最多，对应表中 R²=0.491 也是窗口模式的最优解）。

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

> 完整原理（动作来源、三重稳定化公式、与 StableGatedPI 的差异）见
> [path_integrator_principle.md](path_integrator_principle.md)。

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
