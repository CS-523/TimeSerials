"""
优化模型：未来 x3/x4/x6/x8 使 y4 或 Y 最大
==========================================
**本轮暂时停用**——因为过程预测模型去掉了 y_head/Y_head，没有预测 y 的能力。

本轮目标：先把 x 预测精度做到极致（详见 train_forecaster.py）。
后续策略：等 y 预测方案确定后再启用本脚本。可能的选项：
  1. 训练一个轻量 x→y4 回归器（XGBoost / 随机森林），用其预测 y4 做奖励
  2. 用 N4SID 状态空间 + Kalman（src_control/）的线性 y 模型
  3. 在 PathIntegratorForecaster 上重新加 y_head，但用 y3 当 y4 的 proxy（ρ=0.995）

启动方式（待 y 方案确定后）：
    python src/optimize.py --method es --n-exps 20 --T-out 8

当前状态：脚本只保留 CLI 入口和模型加载代码作为占位，不执行实际优化。
"""
from __future__ import annotations

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import Scaler
from model_forecaster import PathIntegratorForecaster


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = PathIntegratorForecaster(
        dim_x=8, dim_state=args["dim_state"], hidden=args["hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    return model, x_scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kefu-nas/ybkong/time_serials-master/src/model_out/forecaster_best.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_scaler = load_model(args.ckpt, device)
    print(f"[optimize] 模型已加载（dim_state={model.dim_state}），但本轮优化器已停用。")
    print("[optimize] 请等 y 预测方案确定后重新启用本脚本。")


if __name__ == "__main__":
    main()