# scripts_control — 命令行驱动脚本

本目录是 `src_control/`（现代控制理论管道）的 CLI 入口。所有脚本都用 `python -m scripts_control.<name>` 运行，与项目根目录的 `scripts/` 无冲突。

## 脚本一览

| 脚本 | 作用 | 产物 |
|---|---|---|
| `03_train_predictor.py` | 训练 y1–y4 预测模型（线性 SS + NN 残差） | `checkpoints/ss_nn_best.pt` 等 |
| `08_train_x_model.py` | 训练独立的 x1–x8 去噪/重构模型 | `checkpoints/x_recon_best.pt` 等 |
| `04_optimize.py` | MPC Pareto 优化（决策变量 x3/x4/x6/x8） | `results/metrics/pareto.json` 等 |
| `06_visualize.py` | 综合可视化（y 预测 / x̂ 叠加，两者都可选） | `src_control/analysis_out/*.png` |
| `07_ppt_figures.py` | 生成 8 张中文标注的 PPT 图 | `src_control/analysis_out/ppt/*.png` |
| `05_smoke_test.py` | 端到端冒烟测试（临时目录，几分钟） | 无（临时目录自动清理） |

---

## 前置：数据预处理

先把 171 个 CSV 解析成对齐、清洗、标准化后的 npz。`--root` 指向包含 `1/ 2/ 3/ 4/ 5/` 子目录的地方：

```bash
python -m src_control.preprocess \
    --root /remote-home/sunxiaoting/ybkong/timserials/time-serials-mac \
    --out data/processed --seq-len 320
```

> 如果 `config.py` 里的 `ROOT`（默认 `/kefu-nas/ybkong/time_serials-master`）在那台机器上就是数据所在位置，可直接省略 `--root`。

产物：`data/processed/{aligned_dataset,train,test,scalers}.npz`

---

## 1. 训练 y 预测模型（x → y）

```bash
python -m scripts_control.03_train_predictor \
    --data  data/processed/train.npz \
    --test  data/processed/test.npz  \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 \
    --tf-decay 50 --patience 30
```

- 主要参数：`--n-state 16`（SS 状态维）、`--hidden 128`（残差 MLP）、`--horizon 32`
- 产物：`checkpoints/ss_nn_best.pt` / `ss_nn_last.pt`、`results/metrics/training_log.json`、`results/metrics/test_metrics.json`、`results/predictions/test_predictions.npz`

快速冒烟（CI）：

```bash
python -m scripts_control.03_train_predictor \
    --epochs 5 --bs 16 --tf-decay 2 --patience 5
```

## 2. 训练 x1–x8 去噪/重构模型（可选、独立）

默认模型只预测 y。若要**也让 x1–x8 有预测值**（去噪/缺失填补），另训一个输出维度换成 8 的独立模型，**不影响 y 模型与 MPC**：

```bash
# 加 --out-root 可把模型输出一键放进 scripts_control/（见下方说明）
python -m scripts_control.08_train_x_model \
    --data  data/processed/train.npz \
    --test  data/processed/test.npz  \
    --scalers data/processed/scalers.npz \
    --epochs 200 --bs 16 --lr 1e-3 \
    --mask-prob 0.15 --noise-std 0.1 --patience 30 
	# \
    #--out-root scripts_control
```

- `--mask-prob` / `--noise-std`：训练时对 x 的随机 mask 比例 / 高斯噪声强度（去噪任务的污染强度）
- `--out-root <dir>`：输出根目录，设置后模型输出统一放到 `<dir>/checkpoints`、`<dir>/results/metrics`、`<dir>/results/predictions`（默认 `checkpoints/`、`results/`，与 03 一致）
- 产物：`checkpoints/x_recon_best.pt`（裸 state_dict）、`results/metrics/x_recon_metrics.json`、`results/predictions/test_x_predictions.npz`（用 `--out-root scripts_control` 时为 `scripts_control/checkpoints/...` 等）

训练完成后即可加载做推理，详见下方「加载 x 模型做去噪/填补」。

## 3. MPC Pareto 优化

加载 y 模型，对测试集样本优化 `(x3, x4, x6, x8)` 以最大化 `Σy4` 与预测 `Y`：

```bash
python -m scripts_control.04_optimize \
    --ckpt checkpoints/ss_nn_best.pt \
    --data data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --n-samples 10 --horizon 16 --n-starts 3
```

- 产物：`results/metrics/pareto.json`、`results/figures/pareto_frontier.png`、`results/figures/optimized_vs_baseline_<idx>.png`

## 4. 综合可视化

画预测叠加图、误差分布、优化对比。**y 模型和 x 模型都是可选的**——哪个 checkpoint 存在就画哪个：

```bash
# 模型产物在 scripts_control/ 下时（08 用了 --out-root scripts_control），这里同样加 --out-root
python -m scripts_control.06_visualize \
    --test data/processed/test.npz \
    --scalers data/processed/scalers.npz \
    --out-root scripts_control
```

- `--out-root <dir>`：模型/产物查找根目录，`--ckpt`/`--x-ckpt`/`--preds`/`--pareto` 默认取 `<dir>/checkpoints/...`、`<dir>/results/...`；与 08 训练脚本的 `--out-root` 对应，两侧用同一个目录即可
- `--ckpt`（y 模型）默认 `<out-root>/checkpoints/ss_nn_best.pt`（不传 `--out-root` 时是 `checkpoints/ss_nn_best.pt`），缺失时跳过 `forecast_y1_y4.png`
- `--x-ckpt`（x 模型）默认 `<out-root>/checkpoints/x_recon_best.pt`，缺失时跳过 x̂ 叠加
- 两个都不存在则直接退出并提示；缺哪个都会在 stderr 打印 `WARNING` 并在末尾汇总
- 产物（默认到 `src_control/analysis_out/`）：
  - `forecast_x1_x8.png` — x1..x8 历史 + x̂ 重构预测（红色虚线）；仅画 x，y 预测在下面独立图里
  - `forecast_y1_y4.png`（仅 y 模型存在时）、`error_distribution.png`、`optimization_compare.png`

## 5. PPT 图

生成 8 张中文标注的 PPT 图（`--figs` 可指定图号子集，如 `1,2,5`）：

```bash
python -m scripts_control.07_ppt_figures \
    --preds results/predictions/test_predictions.npz \
    --figs all
```

产物：`src_control/analysis_out/ppt/*.png`

## 6. 端到端冒烟测试

在临时目录跑完整流水线（预处理 + 训练 + 优化），验证不报错：

```bash
python -m scripts_control.05_smoke_test
```

---

## 典型工作流顺序

```
preprocess ──► 03 训练 y 模型 ──► 04 优化 ──► 06 可视化
preprocess ──► 08 训练 x 模型 ─────────────► 06 可视化（只画 x 也行）
```

两条路径相互独立：只预测 x 就只跑 08（完全不需要 03/04）。

---

## 加载 x 模型做去噪/缺失填补（最小示例）

```python
import numpy as np, torch
from src_control.models.state_space_nn import SS_NN_Hybrid

model = SS_NN_Hybrid(dim_u=8, dim_y=8, n_state=16, hidden=128, window=4)
model.load_state_dict(torch.load("checkpoints/x_recon_best.pt", map_location="cpu"))
model.eval()

scaler = np.load("data/processed/scalers.npz")
x_mean, x_scale = scaler["x_mean"], scaler["x_scale"]

x_raw = ...                     # (T, 8) 原始单位的 x
x_std = np.nan_to_num((x_raw - x_mean) / x_scale, nan=0.0)  # NaN(缺失) 当 0 喂入

with torch.no_grad():
    x_hat = model(torch.tensor(x_std, dtype=torch.float32).unsqueeze(0),
                  y_prev=None, teacher_forcing=0.0)
x_hat = x_hat[0].cpu().numpy() * x_scale + x_mean   # 反标准化 → 去噪/填补后的 x1–x8
```
