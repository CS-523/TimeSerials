#!/usr/bin/env bash
# =============================================================================
# run_all.sh — 4 项任务最优模型：训练 + 预测 总流程脚本
# =============================================================================
#
# 依据《实验日志.md》选定每个模块的最优模型，按依赖顺序一键复现全流程。
#
#   # 任务（模块）         最优模型                      依据（实验日志）
#   1  多组 x1–x8 预测对比   PathInt shared      RMSE 1372.5；PathInt 优于 LSTM 12.4%
#   2  预测 y1–y4            SS_NN_Hybrid (03)   y4 R²=0.987；纯 x→y 前馈（YHead 已删）
#   3  预测实验终值 Y        per-group deg2      RMSE 67.1 最低
#   4  优化 x3/x4/x6/x8     04_optimize          train 1%–99% 界 + 采样相位对齐，收敛外推
#
# ── 实验方法 ────────────────────────────────────────────────────────────────
# 任务1（x1–x8 多组预测对比）：5 组实验 × 8 过程变量；split_experiments_groupwise
#   按实验 ID 分组切分(seed=42) + Scaler 8 维标准化。LSTM(hidden=128×2,dropout=0.1,
#   lr=2e-3) vs PathInt(dim_state=128,hidden=128,lr=5e-4)，组策略 shared/group_head
#   (FiLM 头 γ,β∈R^8)。epochs=60，前 10 epoch Teacher Forcing 后纯自回归 rollout，
#   loss=MSE(x)。结论：PathInt shared 最优（RMSE 1372.5）。
#
# 任务2（y1–y4 预测，SS_NN_Hybrid）：线性状态空间基线 LinearSSTorch（N4SID 初始化
#   A∈R16×16,B∈R16×8,C∈R4×16,D∈R4×8）+ 残差 MLP(window=4, 64→128→128→4, GELU)。
#   前向纯 x→y：x_{t+1}=Ax_t+Bu_t, y_lin=Cx_t+Du_t → 拼 [u,y_lin] → 残差修正
#   y_pred=y_lin+y_res；无 teacher forcing、无 y 自回归反馈。AdamW(lr=1e-3,wd=1e-4)
#   + cosine 退火 + early stop(patience=30)，损失 masked_mse（仅观测位）。YHead 已删。
#
# 任务3（Y 终值预测，train_y_poly per-group deg2）：StandardScaler → PolynomialFeatures(2)
#   → Ridge（RidgeCV 选 α）；每组从 36 个 last/mean/delta 候选(y1–y4+x1–x8)按组内
#   Pearson|r| 挑 top-8，每组独立 Ridge，组内 80/20 划分。结论 RMSE 67.1 最低。
#
# 任务4（优化 x3/x4/x6/x8 → max y4）：冻结 x_forecast(dim_y=8) 给默认轨迹、ss_nn(dim_y=4)
#   当奖励；仅可控量 x3/x4/x6/x8(DECISION_IDX=2,3,5,7) 是搜索变量，投影梯度上升
#   (Adam+多起点+clamp 投影)，loss=-mean(y4)+λ‖x_ctrl−x_default‖²；采样相位对齐
#   (H=48 覆盖 24h 采样，reward 只在观测相位取均值) + 分布内约束(train 1%–99% 分位数界)。
#
# 用法（在仓库根目录执行）：
#   bash run_all.sh                      # 全量：预处理 + 训练 + 预测/评估 + 优化
#   bash run_all.sh --skip-train         # 复用已有 checkpoint，只跑预测/评估/优化
#   bash run_all.sh --device cpu         # 指定计算设备（默认 cuda，无 GPU 自动 cpu）
#
# 前置依赖：python -m src_control.preprocess 需先产出 data/processed/*.npz。
# =============================================================================

set -euo pipefail

# ---- 参数解析 ---------------------------------------------------------------
SKIP_TRAIN=0
DEVICE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-train) SKIP_TRAIN=1 ;;
        --device) DEVICE="$2"; shift ;;
        -h|--help) awk 'NR>1 && /^#/ { if ($0 ~ /^# ----/) exit; sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
        *) echo "未知参数: $1（可用 --skip-train / --device / --help）" >&2; exit 2 ;;
    esac
    shift
done

# ---- 路径 / 设备 -------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -z "$DEVICE" ]; then
    DEVICE="cuda"
    python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1 || DEVICE="cpu"
fi

PY="python"   # 若环境用 python3，改成 python3

step() { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }

# ---- 阶段 0：数据预处理（前置） ----------------------------------------------
step "阶段 0 · 数据预处理（对齐/清洗/标准化 → data/processed/*.npz）"
$PY -m src_control.preprocess --root "$ROOT" --out data/processed --seq-len 320

# ---- 阶段 1：任务 1 · 多组预测对比 → 最优 PathInt shared ----------------------
step "任务 1 · 多组 x1–x8 预测对比（最优 = PathInt shared）"
if [ "$SKIP_TRAIN" -eq 0 ]; then
    step "  1a · 训练对比（LSTM/PathInt × shared/group_head，共 4 模型）"
    $PY src/compare_multigroup.py --base-dir "$ROOT" --epochs 60 --device "$DEVICE" \
        --lr-lstm 2e-3 --lr-pathint 5e-4 --modes shared,group_head
fi
step "  1b · 预测/评估（复用 ckpt → compare_metrics.json + compare_bars_*.png）"
$PY src/compare_multigroup.py --base-dir "$ROOT" --device "$DEVICE" \
    --skip-train --modes shared,group_head

# ---- 阶段 2：任务 2 · 预测 y1–y4 → SS_NN_Hybrid ------------------------------
step "任务 2 · 预测 y1–y4（最优 = SS_NN_Hybrid / 03_train_predictor）"
if [ "$SKIP_TRAIN" -eq 0 ]; then
    step "  2a · 训练 y1–y4 预测器"
    $PY -m scripts_control.03_train_predictor \
        --data data/processed/train.npz --test data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --epochs 200 --bs 16 --lr 1e-3 --patience 30
fi
step "  2b · 预测可视化（forecast_y1_y4.png / error_distribution.png）"
$PY -m scripts_control.06_visualize \
    --test data/processed/test.npz --scalers data/processed/scalers.npz

# ---- 阶段 3：任务 3 · 预测终值 Y → per-group deg2 -----------------------------
step "任务 3 · 预测实验终值 Y（最优 = per-group deg2）"
$PY src/train_y_poly.py --base-dir "$ROOT" --degree 2 --per-group

# ---- 阶段 4：任务 4 · 优化 x3/x4/x6/x8 使 y4 最大 -----------------------------
step "任务 4 · 优化未来 x3/x4/x6/x8 使 y4 最大（04_optimize）"
if [ "$SKIP_TRAIN" -eq 0 ]; then
    step "  4a · 前置：训练 x1–x8 外推模型（x_forecast_best.pt，优化器硬依赖）"
    $PY -m scripts_control.08_train_x_model \
        --data data/processed/train.npz --test data/processed/test.npz \
        --scalers data/processed/scalers.npz \
        --epochs 200 --bs 16 --lr 1e-3 --context 32 --horizon 32 --tf-decay 50 --patience 30
fi
step "  4b · 优化（train 1%–99% 界 + 采样相位对齐，全 test 集）"
$PY -m scripts_control.04_optimize \
    --data data/processed/test.npz --scalers data/processed/scalers.npz \
    --all-test --horizon 48 --bounds-mode train --bound-quantile 0.01

printf '\n\033[1;32m全部完成。\033[0m 关键产物：\n'
printf '  - 任务1: src/model_out/compare_metrics.json（PathInt shared RMSE 应最低）\n'
printf '  - 任务2: checkpoints/ss_nn_best.pt + results/metrics/test_metrics.json\n'
printf '  - 任务3: src/model_out/y_poly_deg2_engineered_pergroup_{metrics.json,predictions.csv}\n'
printf '  - 任务4: results/optimization/optimize_metrics.json\n'


