#!/usr/bin/env bash
# 一次性跑 3 个模式，便于在同 test 集合上对比 RMSE
set -e
cd /root/workspace/kefu-nas/ybkong/time_serials-master/src

EPOCHS=${EPOCHS:-20}
DEVICE=${DEVICE:-cpu}

echo "==== shared (全共享 baseline) ===="
python train_multigroup.py --mode shared --epochs $EPOCHS --device $DEVICE

echo "==== group_head (共享骨干+组适配头) ===="
python train_multigroup.py --mode group_head --epochs $EPOCHS --device $DEVICE

echo "==== independent (逐组独立,精度上界) ===="
python train_multigroup.py --mode independent --epochs $EPOCHS --device $DEVICE

echo "==== done ===="
ls -1 ../src/model_out/test_metrics_*.json 2>/dev/null || true
