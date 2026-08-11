"""
多组 / 多头预测模型
====================

在已有 LSTMForecaster 之上做的"组条件"扩展，**不修改 model_lstm.py**。
- LSTMForecasterFiLM：共享 LSTM 编码器 + 5 个 group-conditional head
  - 每个 head 是 (γ_g, β_g) ∈ R^8 的 FiLM 仿射参数
  - forward: 共享编码 → 每步把 group-specific γ 乘到隐藏、再加 β → 解码 Δx
- LSTMForecaster5Models：5 个独立 LSTMForecaster（per-group 切分 + 训）

两种结构均输出 {pred_x: (B, T_out, 8)}，与原 LSTMForecaster 接口完全一致，
可以直接复用 src/train_forecaster.py 里已有的 train_one_epoch / evaluate 逻辑。

设计动机见 src/group_equivalence_report.md：4 个独立维度（L1/L2/L3/L4）
全部失败 → 不建议纯池化模型。这里提供"组条件 / per-group"两种替代结构作为
end-to-end 实证。


三种模式对比（shared=全共享  group_head=共享骨干+组适配头  independent=逐组独立）：

  # Windows PowerShell（单行，在 src/ 目录下执行）
  python train_multigroup.py --base-dir "D:/Code/timeserials_claude/time-serials-mac" --mode shared --epochs 30
  python train_multigroup.py --base-dir "D:/Code/timeserials_claude/time-serials-mac" --mode group_head --epochs 30
  python train_multigroup.py --base-dir "D:/Code/timeserials_claude/time-serials-mac" --mode independent --epochs 30

  # Git Bash（支持 \ 换行）
  EPOCHS=30 DEVICE=cpu bash run_multigroup_compare.sh

产物：src/model_out/ 下生成 forecaster_{mode}_best.pt 和 test_metrics_{mode}.json
对比 L5 结论：看 test_metrics_*.json 中 by_group 字段，group_head 是否接近 independent（上界）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.backends.cudnn as _cudnn

# 兼容 PyTorch + cuDNN LSTM 在某些环境下报 CUDNN_STATUS_NOT_INITIALIZED 的问题
_cudnn.enabled = False

from model_lstm import LSTMForecaster  # 复用已有编码器
from model_forecaster import PathIntegratorForecaster  # 复用 PathInt 编码器


# ===================== A. 共享骨干 + group-conditional FiLM 头 =====================
class LSTMForecasterFiLM(nn.Module):
    """
    共享 LSTM 编码器；每个 group 一个 FiLM 头 (γ_g, β_g) ∈ R^8，共 5×8×2=80 个参数。

    训练时：根据 sample.group_id 选对应 FiLM 头
    推理时：先做"组分类"（用 group_id），再选头

    这样既共享共性，又允许每组有自己的 (μ 偏移 + σ 缩放)。
    """
    def __init__(self, n_groups: int = 5, dim_x: int = 8,
                 hidden: int = 128, num_layers: int = 2, dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.n_groups = n_groups
        self.dim_x = dim_x
        # 共享骨干
        self.backbone = LSTMForecaster(
            dim_x=dim_x, hidden=hidden,
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
        )
        # enc_dim = hidden * (2 if bidirectional else 1)
        self.enc_dim = self.backbone.num_directions * self.backbone.hidden
        # 每组一个 (γ_g, β_g) —— 用 Embedding 学习
        # γ 初始化为 1（不缩放），β 初始化为 0（不偏移），与原解码器行为一致
        self.gamma = nn.Embedding(n_groups, dim_x)
        self.beta = nn.Embedding(n_groups, dim_x)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def decode_with_head(self, h_last: torch.Tensor, x_last: torch.Tensor,
                         gamma: torch.Tensor, beta: torch.Tensor, T_out: int) -> torch.Tensor:
        """沿用 LSTMForecaster 的循环解码风格，但在每步融合 (γ, β)。"""
        preds = []
        s = x_last
        h = h_last
        head = self.backbone.head  # 共享 head 矩阵
        # head 输入 = [γ * s + β, h_last]  → 注意：我们要让每组都有自己的偏移/缩放
        # 但 head 本身是共享的：把 γ,β 应用到 s 上
        for _ in range(T_out):
            s_affined = gamma * s + beta
            inp = torch.cat([s_affined, h], dim=-1)
            dx = head(inp)              # (B, 8) Δx
            x_next = s + dx
            preds.append(x_next)
            s = x_next
        return torch.stack(preds, dim=1)

    def forward(self, past_x: torch.Tensor, group_id: torch.Tensor,
                T_out: int) -> dict:
        """
        past_x:   (B, L, 8)  标准化后
        group_id: (B,)         整数 [0, n_groups-1]
        T_out:    int
        """
        h_last, _ = self.backbone.encode(past_x)            # (B, enc_dim)
        x_last = past_x[:, -1, :]                            # (B, 8)
        gamma = self.gamma(group_id)                          # (B, 8)
        beta = self.beta(group_id)                            # (B, 8)
        pred_x = self.decode_with_head(h_last, x_last, gamma, beta, T_out)
        return {"pred_x": pred_x}


# ===================== B. 5 个独立 LSTM（per-group 5 模型） =====================
class LSTMForecaster5Models(nn.Module):
    """
    5 个独立 LSTMForecaster，每个 group 一个。
    forward 时按 group_id 选对应模型。
    """
    def __init__(self, n_groups: int = 5, dim_x: int = 8,
                 hidden: int = 128, num_layers: int = 2, dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.n_groups = n_groups
        self.dim_x = dim_x
        self.models = nn.ModuleList([
            LSTMForecaster(
                dim_x=dim_x, hidden=hidden,
                num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
            )
            for _ in range(n_groups)
        ])

    def forward(self, past_x: torch.Tensor, group_id: torch.Tensor,
                T_out: int) -> dict:
        """
        past_x:   (B, L, 8)
        group_id: (B,)
        T_out:    int

        返回时把 B 个样本按 group 排序，再选各自 model 推理，再按原顺序拼回。
        """
        B = past_x.size(0)
        # 1) 按 group 排序，记录原位置
        order = torch.argsort(group_id)
        inv_order = torch.argsort(order)
        sorted_x = past_x[order]
        sorted_g = group_id[order]

        # 2) 逐 group 调用对应模型
        out_chunks = []
        for g in sorted_g.unique():
            mask = (sorted_g == g)
            xs = sorted_x[mask]
            out = self.models[int(g.item())](xs, T_out=T_out)["pred_x"]
            out_chunks.append(out)

        # 3) 拼接
        sorted_pred = torch.cat(out_chunks, dim=0)
        # 4) 还原原顺序
        pred_x = sorted_pred[inv_order]
        return {"pred_x": pred_x}


# ===================== C. 共享 PathInt 骨干 + group-conditional FiLM 头 =====================
class PathIntegratorForecasterFiLM(nn.Module):
    """
    共享 PathIntegratorForecaster 编码器；每个 group 一个 FiLM 头 (γ_g, β_g) ∈ R^8。

    与 LSTMForecasterFiLM 设计一致：FiLM 作用于 decode 每步的 x_last，
    PathInt 状态 s 正常演化不受 FiLM 影响。
    """
    def __init__(self, n_groups: int = 5, dim_x: int = 8,
                 dim_state: int = 128, hidden: int = 128):
        super().__init__()
        self.n_groups = n_groups
        self.dim_x = dim_x
        # 共享 PathInt 骨干
        self.backbone = PathIntegratorForecaster(
            dim_x=dim_x, dim_state=dim_state, hidden=hidden,
        )
        # 每组一个 (γ_g, β_g)
        self.gamma = nn.Embedding(n_groups, dim_x)
        self.beta = nn.Embedding(n_groups, dim_x)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, past_x: torch.Tensor, group_id: torch.Tensor,
                T_out: int) -> dict:
        """
        past_x:   (B, L, 8)  标准化后
        group_id: (B,)         整数 [0, n_groups-1]
        T_out:    int
        """
        s = self.backbone.encode(past_x)                   # (B, dim_state)
        x_last = past_x[:, -1, :]                           # (B, 8)
        gamma = self.gamma(group_id)                         # (B, 8)
        beta = self.beta(group_id)                           # (B, 8)

        pred_x = []
        for _ in range(T_out):
            # FiLM 仿射作用于 x_last，再与 PathInt 状态 s 拼接
            x_affined = gamma * x_last + beta
            inp = torch.cat([s, x_affined], dim=-1)
            dx = self.backbone.x_head(inp)
            x_next = x_last + dx
            pred_x.append(x_next)
            x_last = x_next
            # PathInt 状态自我演化（不经过 FiLM）
            a = self.backbone.x_proj(x_next)
            s = self.backbone.gated_cell(s, a)
        pred_x = torch.stack(pred_x, dim=1)                # (B, T_out, 8)
        return {"pred_x": pred_x}


# ===================== D. 5 个独立 PathInt（per-group 5 模型） =====================
class PathIntegratorForecaster5Models(nn.Module):
    """
    5 个独立 PathIntegratorForecaster，每个 group 一个。
    forward 时按 group_id 选对应模型。
    """
    def __init__(self, n_groups: int = 5, dim_x: int = 8,
                 dim_state: int = 128, hidden: int = 128):
        super().__init__()
        self.n_groups = n_groups
        self.dim_x = dim_x
        self.models = nn.ModuleList([
            PathIntegratorForecaster(
                dim_x=dim_x, dim_state=dim_state, hidden=hidden,
            )
            for _ in range(n_groups)
        ])

    def forward(self, past_x: torch.Tensor, group_id: torch.Tensor,
                T_out: int) -> dict:
        """
        past_x:   (B, L, 8)
        group_id: (B,)
        T_out:    int
        """
        B = past_x.size(0)
        # 按 group 排序，记录原位置
        order = torch.argsort(group_id)
        inv_order = torch.argsort(order)
        sorted_x = past_x[order]
        sorted_g = group_id[order]

        # 逐 group 调用对应模型
        out_chunks = []
        for g in sorted_g.unique():
            mask = (sorted_g == g)
            xs = sorted_x[mask]
            out = self.models[int(g.item())](xs, T_out=T_out)["pred_x"]
            out_chunks.append(out)

        # 拼接并还原原顺序
        sorted_pred = torch.cat(out_chunks, dim=0)
        pred_x = sorted_pred[inv_order]
        return {"pred_x": pred_x}


# ===================== 快速自检 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("LSTMForecasterFiLM（共享骨干 + 组适配头）")
    film = LSTMForecasterFiLM(n_groups=5, dim_x=8, hidden=128, num_layers=2)
    n_film = sum(p.numel() for p in film.parameters())
    print(f"  参数量: {n_film:,}")
    print(f"  其中 FiLM 头参数: {5 * 8 * 2}（gamma + beta）")
    print(f"  共享骨干参数: {n_film - 5 * 8 * 2:,}")

    B, L_in, T_out = 4, 20, 8
    x = torch.randn(B, L_in, 8)
    g = torch.randint(0, 5, (B,))
    out = film(x, g, T_out=T_out)
    print(f"  输入: {x.shape}  输出 pred_x: {out['pred_x'].shape}")
    print(f"  [OK] FiLM forward 正常")

    print()
    print("LSTMForecaster5Models（per-group 独立 LSTM，精度上界）")
    per5 = LSTMForecaster5Models(n_groups=5, dim_x=8, hidden=128, num_layers=2)
    n_per5 = sum(p.numel() for p in per5.parameters())
    print(f"  参数量: {n_per5:,}（~ {n_per5 / n_film:.1f}× FiLM 的参数量）")
    out5 = per5(x, g, T_out=T_out)
    print(f"  输入: {x.shape}  输出 pred_x: {out5['pred_x'].shape}")
    print(f"  [OK] 5Models forward 正常")

    print()
    loss = torch.nn.functional.mse_loss(out["pred_x"], torch.randn(B, T_out, 8))
    loss.backward()
    print(f"  [OK] LSTM FiLM backward 正常, loss = {loss.item():.4f}")

    print()
    print("=" * 50)
    print("PathIntegratorForecasterFiLM（共享 PathInt 骨干 + 组适配头）")
    pi_film = PathIntegratorForecasterFiLM(n_groups=5, dim_x=8, dim_state=128, hidden=128)
    n_pi_film = sum(p.numel() for p in pi_film.parameters())
    print(f"  参数量: {n_pi_film:,}")
    print(f"  其中 FiLM 头参数: {5 * 8 * 2}（gamma + beta）")
    print(f"  共享 PathInt 骨干参数: {n_pi_film - 5 * 8 * 2:,}")

    out_pi = pi_film(x, g, T_out=T_out)
    print(f"  输入: {x.shape}  输出 pred_x: {out_pi['pred_x'].shape}")
    print(f"  [OK] PathInt FiLM forward 正常")

    print()
    print("PathIntegratorForecaster5Models（per-group 独立 PathInt）")
    pi_per5 = PathIntegratorForecaster5Models(n_groups=5, dim_x=8, dim_state=128, hidden=128)
    n_pi_per5 = sum(p.numel() for p in pi_per5.parameters())
    print(f"  参数量: {n_pi_per5:,}（~ {n_pi_per5 / n_pi_film:.1f}× PathInt FiLM 的参数量）")
    out_pi5 = pi_per5(x, g, T_out=T_out)
    print(f"  输入: {x.shape}  输出 pred_x: {out_pi5['pred_x'].shape}")
    print(f"  [OK] PathInt 5Models forward 正常")

    print()
    loss_pi = torch.nn.functional.mse_loss(out_pi["pred_x"], torch.randn(B, T_out, 8))
    loss_pi.backward()
    print(f"  [OK] PathInt FiLM backward 正常, loss = {loss_pi.item():.4f}")
    print("=" * 50)
