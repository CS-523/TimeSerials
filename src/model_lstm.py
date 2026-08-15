"""
LSTM 时序预测模型
=================
与 PathIntegratorForecaster 并存的 baseline，用于对比。

设计要点：
  - 输入 x (B, L_in, 8) → 单向 LSTM 编码成 hidden state
  - 取末步 hidden → MLP 解码成未来 T_out 步的 x1-x8（**非自回归**，teacher forcing 风格）
  - 这样能避免自回归 rollout 的累积漂移问题

用法：
    python src/train_forecaster.py --model lstm --epochs 30
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 兼容 PyTorch + cuDNN LSTM 在某些环境下报 CUDNN_STATUS_NOT_INITIALIZED 的问题
import torch.backends.cudnn as _cudnn
_cudnn.enabled = False


class LSTMForecaster(nn.Module):
    """
    编码器-解码器风格的 LSTM：
      编码器：单层/多层 LSTM，输入 x1-x8
      解码器：MLP 将末态 hidden + cell 映射到 (T_out * 8) 形状的输出

    输入：past_x (B, L_in, 8)
    输出：pred_x  (B, T_out, 8)
    """

    def __init__(self, dim_x: int = 8, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.dim_x = dim_x
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 输入投影（可选：让 x 进入一个隐藏层再加 LSTM）
        self.input_proj = nn.Linear(dim_x, hidden)

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 解码器 MLP：hidden + cell → T_out * 8
        enc_dim = hidden * self.num_directions
        # head 接受 (x_last(8) + h_last(enc_dim))
        self.head = nn.Linear(8 + enc_dim, 8)

    def encode(self, past_x: torch.Tensor):
        """
        past_x: (B, L, 8) → (h_T, c_T)
        """
        x = self.input_proj(past_x)
        # LSTM 全序列扫描
        out, (h_T, c_T) = self.lstm(x)
        # h_T / c_T shape: (num_layers * num_directions, B, hidden)
        # 取最后一层的 hidden + cell
        if self.num_directions == 2:
            h_last = torch.cat([h_T[-2], h_T[-1]], dim=-1)   # (B, hidden*2)
            c_last = torch.cat([c_T[-2], c_T[-1]], dim=-1)
        else:
            h_last = h_T[-1]                                # (B, hidden)
            c_last = c_T[-1]
        return h_last, c_last

    def forward(self, past_x: torch.Tensor, T_out: int,
                x_out: torch.Tensor | None = None) -> dict:
        """
        past_x: (B, L_in, 8)
        T_out:  预测步数
        x_out:  (B, T_out, 8) 真实未来值（标准化空间），用于 teacher forcing；
                None 时使用纯自回归。
        """
        h_last, c_last = self.encode(past_x)              # (B, enc_dim), (B, hidden)
        B = past_x.size(0)
        x_last = past_x[:, -1, :]                          # (B, 8)
        preds = []
        s = x_last
        h = h_last
        for t in range(T_out):
            inp = torch.cat([s, h], dim=-1)                # (B, 8 + enc_dim)
            dx = self.head(inp)                            # (B, 8) — 预测 Δx
            x_next = s + dx
            preds.append(x_next)
            # teacher forcing: 喂真实值作为下一步输入；否则用自己的预测
            s = x_out[:, t] if x_out is not None else x_next
        pred_x = torch.stack(preds, dim=1)                # (B, T_out, 8)
        return {"pred_x": pred_x}


if __name__ == "__main__":
    m = LSTMForecaster(dim_x=8, hidden=128, num_layers=2)
    B, L_in, T_out = 4, 20, 8
    px = torch.randn(B, L_in, 8)
    out = m(px, T_out=T_out)
    print("pred_x:", out["pred_x"].shape)
    print(f"#params: {sum(p.numel() for p in m.parameters()):,}")
    target = torch.randn(B, T_out, 8)
    import torch.nn.functional as F
    loss = F.mse_loss(out["pred_x"], target)
    loss.backward()
    print(f"loss = {loss.item():.4f}; backward ok")