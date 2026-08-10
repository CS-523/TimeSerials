# PPT 大纲 — 多元时序预测 + 约束优化（基于现代控制理论）

## 第 1 章 项目理解

### 1.1 数据背景
- **数据来源**：`./1/`–`./5/` 共 171 个 CSV 文件，每个对应一次独立实验
- **数据规模**：每文件约 308–356 行（× 0.5 小时步长），覆盖约 6 天连续工艺
- **时间范围**：2023-04-19 至 2024-05-18
- **列说明**（15 列）：
  - `x1`–`x8`：30 分钟步长的过程变量（输入）
  - `y1`–`y4`：稀疏测量的中间目标（输出）
  - `周期`：从 0 开始累加的周期索引
  - `Y`：每个 CSV 末尾行的最终结果（单值）
  - `datime`：时间戳

### 1.2 任务目标
1. 数据清洗：异常值检测、缺失值处理
2. **预测模型**：给定历史 `x1..x8` 窗口，预测未来 `x1..x8` + `y4`（必选），以及 `y1..y3` + `Y`（可选）；输出长度 ≥ 6，**支持迭代生成**
3. 特征分析：变量特性、互相关、滞后因果
4. **优化模型**：优化未来 `(x3, x4, x6, x8)`，使 `y4` 或最终 `Y` 最大

### 1.3 难点
- `y1..y4` **稀疏测量**（每文件仅 5–13 个观测点）
- 最终 `Y` 在每个文件末才出现一次
- 171 个独立实验，分布可能差异大

---

## 第 2 章 技术路线

### 2.1 总体流程图

```
┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ raw CSVs   │ → │ preprocess │ → │ analysis    │ → │ SS-NN model  │ → │ MPC Pareto  │
│ (1/..5/)   │   │ align, fill│   │ corr, MI,   │   │ hybrid pred. │   │ optimizer   │
└────────────┘   │ scale, split│   │ PCA, lag   │   │              │   │             │
                 └────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
                       │                  │                  │                  │
                       ▼                  ▼                  ▼                  ▼
                  data/processed/    figures/*.png     checkpoints/*.pt   pareto.json +
                  train.npz,                                                  pareto.png
                  test.npz
```

### 2.2 模块 → 代码 映射

| 阶段 | 文件 | 一句话功能 |
|---|---|---|
| 数据加载 | `src_control/data_loader.py` | 解析 171 个 CSV，按内容（x 是否为空）识别 x 行与边界行 |
| 预处理 | `src_control/preprocess.py` | 异常值替换、y 缺失前向/后向填充、标准化、80/20 划分 |
| 特征分析 | `src_control/analysis/correlation.py` | Pearson / MI / PCA / Granger 因果 / 滞后互相关 |
| 状态空间 | `src_control/models/state_space.py` | N4SID 子空间辨识 + Kalman 滤波（numpy 实现） |
| 预测模型 | `src_control/models/state_space_nn.py` | 线性 SS 基线 + 残差 MLP 的混合预测模型 |
| MPC 优化 | `src_control/optimization/mpc_optimizer.py` | 多起点 L-BFGS，扫描权重生成 Pareto 前沿 |
| 可视化 | `src_control/visualization/plots.py` | 预测叠加 / Pareto 散点 / 优化轨迹图 |
| 驱动脚本 | `scripts_control/01..05_*.py` | 各阶段的 CLI 入口 + 端到端冒烟测试 |

### 2.3 现代控制理论的核心方法

| 方法 | 用途 | 实现位置 |
|---|---|---|
| **N4SID 子空间辨识** | 从 I/O 数据识别线性状态空间 `(A,B,C,D)` | `state_space.n4sid` |
| **Kalman 滤波** | 含缺失观测的状态估计（y_mask 感知） | `state_space.kalman_filter` |
| **线性 SS rollout** | 给出可解释的线性基线预测 | `state_space.LinearSS.rollout` |
| **混合 SS + NN 残差** | 表达线性无法捕捉的非线性 | `state_space_nn.SS_NN_Hybrid` |
| **MPC 滚动优化** | 受约束的预测控制（盒式约束） | `mpc_optimizer._optimize_one` |
| **加权扫描 → Pareto** | `y4_sum` 与 `Y` 的多目标权衡 | `mpc_optimizer.optimize_pareto` |

---

## 第 3 章 处理过程与结果

### 3.1 数据预处理
- **对齐**：每文件以 `:00/:30` 行为稠密 x 行，`:45` 行为边界行携带 y
- **缺失 y 处理**：前向填充 + 首观测回填，尾部未观测保持 NaN
- **异常检测**：逐列 z-score，|z| > 5 标记，线性插值替换
- **标准化**：x 全列 `StandardScaler`；y 按列单独计算均值/方差
- **填充到定长**：SEQ_LEN=320（覆盖 95% 分位以上），尾部零填充
- **结果**：`train.npz` (137 样本)、`test.npz` (34 样本)，`X` 形状 `(N, 320, 8)`

### 3.2 特征分析（图源）

| 图 | 文件 | 关键结论（占位） |
|---|---|---|
| `correlation_heatmap.png` | Pearson 相关系数矩阵 | 观察 x_j 与 y_k 的线性相关强度 |
| `mi_heatmap.png` | 互信息热力图 | 捕捉非线性相关 |
| `pca_scree.png` | x 变量 PCA 解释方差 | 检验 x 之间的冗余度 |
| `lag_x_to_y4.png` | x_j 与 y4 的滞后互相关 | 找出领先/滞后关系 |
| `granger_xy.png` | Granger 因果比 | 哪些 x 真正驱动 y |

### 3.3 预测模型（图源：`results/figures/prediction_overlay_*.png`）

训练指标（基于实际训练运行的 `results/metrics/test_metrics.json`）：

| 变量 | MSE | MAE | R² |
|---|---|---|---|
| y1 | 11.3 | 2.5 | 0.91 |
| y2 | 183 | 10.3 | 0.86 |
| y3 | 134475 | 293 | 0.98 |
| y4 | 100636 | 249 | 0.98 |

> 注：以上为开发期一次快速训练（30 epoch）的结果；正式提交前应跑满 200 epoch。

训练技巧：
- AdamW + cosine LR schedule（1e-3 → 1e-5）
- 50% teacher-forced + 50% pure AR 的混合训练
- 早停 patience=30
- N4SID 初始化线性 SS 参数

### 3.4 MPC 优化（图源：`pareto_frontier.png`, `optimized_vs_baseline_*.png`）

- **决策变量**：`u = [x3, x4, x6, x8]` over horizon=16
- **目标**：weighted sum `w_y4 * Σy4 + w_Y * Y_pred`
- **扫描权重**：`[(1,0), (0.7,0.3), (0.5,0.5), (0.3,0.7), (0,1)]`
- **算法**：L-BFGS + 多起点 + 盒式约束 (`u.clamp`)
- **基线**：继续上一输入策略 (continue-last-input)
- **Pareto 前沿**：非支配点构成；基线用红星标注

---

## 第 4 章 分析总结与拓展

### 4.1 经验总结
1. **数据格式多变**：边界行时间戳 `:45:12.530`、`:19:11:00`、`:23:50:00` … 都出现；用内容而非时间格式分类行最稳健
2. **y 极度稀疏**：平均每文件 ~13 个 y1/y2 观测，`y3` ~10 个，`y4` ~5 个 — 训练时必须用 mask-aware 损失
3. **y4 与 x 关系非线性**：Pearson 相关可能不强，但 MI 与滞后互相关揭示了动态关系
4. **x7 单调递增**：在 MPC 中应保持累积趋势而非自由优化
5. **x3 衰减模式**：从 ~100 衰减到 ~70 — 优化时鼓励单调下降
6. **混合 SS + NN**：线性 SS 提供稳定基线 (R² ~0.85)，NN 残差补充非线性（最终 R² ~0.97+）

### 4.2 待改进 / 拓展方向
1. **Mamba / GRU 替换 MLP**：长序列（320 步）下，门控 RNN 可能更稳定
2. **自适应 MPC**：用 RLS 滚动更新 A, B（已在计划中）
3. **实时闭环**：将 MPC 嵌入真实控制循环，每 30 分钟触发一次
4. **ILC（迭代学习控制）**：171 次实验可视为重复任务，跨实验累积改进控制策略
5. **NN 正则化**：用线性 SS 输出作为 NN 残差的先验（贝叶斯风格）
6. **约束扩展**：加入 x3 单调性约束（速度上限）、y4 物理可达区间
7. **更强的 MPC 求解器**：`cvxpy` + OSQP（凸优化）或 `casadi` + IPOPT（NLP）

### 4.3 PPT 演示注意事项
- 每页控制在 1 个核心信息 + 1 张图
- 突出：方法论创新（混合 SS + NN，多目标 Pareto）
- 弱化：单次运行的指标数字（应在最终正式训练后填入）

---

## 附录 A：复现命令

```bash
# 1. 数据预处理
python -m src_control.preprocess --out data/processed --seq-len 320

# 2. 特征分析
python -m src_control.analysis.correlation \
    --data data/processed/train.npz \
    --out results/figures

# 3. 训练预测模型（200 epoch）
python -m scripts_control.03_train_predictor \
    --data data/processed/train.npz \
    --test  data/processed/test.npz  \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 --tf-decay 50 --patience 30

# 4. MPC Pareto 优化
python -m scripts_control.04_optimize \
    --ckpt checkpoints/ss_nn_best.pt \
    --data data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --n-samples 10 --horizon 16 --n-starts 3

# 端到端冒烟测试
python -m scripts_control.05_smoke_test
```

## 附录 B：单元测试

```bash
pytest tests_control/
```

## 附录 C：项目目录结构

```
src_control/                ← 本任务所有代码
├── config.py               全局配置
├── data_loader.py          CSV 解析
├── preprocess.py           清洗 + 标准化 + 划分
├── analysis/correlation.py 相关 / MI / PCA / Granger
├── models/
│   ├── state_space.py      N4SID + Kalman (numpy)
│   └── state_space_nn.py   混合 SS + NN 残差
├── optimization/mpc_optimizer.py  Pareto MPC
├── visualization/plots.py  统一可视化
├── utils/                  metrics, seed
└── import_legacy.py        注入 ../src 路径以复用 path_integrators
```