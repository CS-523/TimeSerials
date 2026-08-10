#!/usr/bin/env bash
# 一次性跑 3 个模式，便于在同 test 集合上对比 RMSE
set -e
cd /root/workspace/kefu-nas/ybkong/time_serials-master/src

EPOCHS=${EPOCHS:-20}
DEVICE=${DEVICE:-cpu}

echo "==== pooled ===="
python train_multigroup.py --mode pooled --epochs $EPOCHS --device $DEVICE

echo "==== film ===="
python train_multigroup.py --mode film --epochs $EPOCHS --device $DEVICE

echo "==== per_group ===="
python train_multigroup.py --mode per_group --epochs $EPOCHS --device $DEVICE

echo "==== done ===="
ls -1 ../src/model_out/test_metrics_*.json 2>/dev/null || true
