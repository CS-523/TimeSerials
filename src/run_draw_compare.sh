#!/usr/bin/env bash
# =============================================================================
# run_draw_compare.sh — 一次性重绘多组对比图（加载已有 checkpoint，跳过训练）
#
# 作用：
#   加载 src/model_out/ 下已有的 4 个 checkpoint（LSTM / PathInt × shared / group_head），
#   在 groupwise test 集上重新评估，并一次生成全部对比图与指标 JSON。
#
# 产物（src/model_out/）：
#   compare_bars_lstm.png          LSTM 两种模式的 per-group RMSE 柱状图
#   compare_bars_pathint.png       PathInt 两种模式的 per-group RMSE 柱状图
#   compare_bars_all.png           所有模型合并柱状图
#   compare_bars_all_train.png     训练集合并柱状图（过拟合检查）
#   compare_bars_tf_vs_ar.png      自回归 vs Teacher Forcing 整体 RMSE 对比
#   compare_metrics.json           test 集指标（含 by_group）
#   compare_metrics_train.json     训练集指标
#   compare_metrics_tf.json        TF 一步外推指标
#
# 用法：
#   bash run_draw_compare.sh                          # 默认：设备自动检测（有 CUDA 用 cuda）
#   DEVICE=cpu bash run_draw_compare.sh               # 显式指定设备
#   MODES=shared,group_head bash run_draw_compare.sh  # 指定参与对比的模式
#
# 环境变量：
#   DEVICE  计算设备，默认自动检测（有 CUDA 用 cuda，否则 cpu）
#   MODES   组策略列表（逗号分隔），可选 shared / group_head，默认 shared,group_head
#           （independent 模式已移除，不要传）
#
# 依赖：需先有对应 checkpoint 文件（forecaster_{lstm,pathint}_{shared,group_head}.pt），
#       否则会报「找不到 checkpoint」。可用 compare_multigroup.py 训练生成。
# =============================================================================
set -euo pipefail

# 中文输出 locale 兜底：当前 locale 非 UTF-8 时自动切到 C.UTF-8，避免中文乱码
if [ -z "${LANG:-}" ] || ! locale charmap 2>/dev/null | grep -qi 'UTF-8'; then
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
fi

# 项目根目录：本脚本在 src/ 下，根目录是上一级
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$BASE_DIR/src"

# 设备自动检测：未显式指定时，有 CUDA 用 cuda，否则 cpu
if [ -z "${DEVICE:-}" ]; then
    if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi

MODES=${MODES:-shared,group_head}

cd "$SRC_DIR"
echo "==== 重绘对比图（skip-train）：modes={$MODES} device=$DEVICE ===="
python compare_multigroup.py \
    --base-dir "$BASE_DIR" \
    --skip-train \
    --device "$DEVICE" \
    --modes "$MODES"

echo "==== done ===="
ls -1 "$SRC_DIR/model_out/compare_bars_"*.png 2>/dev/null || true
