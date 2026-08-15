"""
过程预测模型：Path-Integrator-based Sequence Forecaster（仅预测 x1-x8）
=====================================================================
设计思路（借鉴 path_integrators.py 中的 StableGatedPI / ResidualPI）：
  1. 用一个"输入编码器"把过去 L_in 步的 x 序列编码成一个 state vector s0；
     编码器内部使用类 StableGatedPI 的"门控残差 + L2 投影"路径积分结构，
     解决长程依赖和梯度爆炸。
  2. 之后用"自回归 rollout"产生未来 T_out 步的 x1-x8：
     每步把当前 state 过一个 transition 模块得到 Δx_t，
     再把 (state, Δx) 送进一个"读出网络"得到下一步 x。
  3. **本版本只预测 x1-x8，不预测 y1-y4 / Y**——理由是 y 标签稀疏（y4 每 ~24 步
     才出现一次），引入 y 头会拖累训练稳定性。先把 x 预测做精，后续再单独考虑 y。

训练损失：
  - 过程预测主任务：MSE on x（输出窗口的 8 列）

参考：
  - path_integrators.StableGatedPI：spectral_norm + 门控残差 + L2 投影
  - path_integrators.MambaLiteSSM：连续时间离散化思想（dt, A_log, B, C）
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# ===================== 编码器单元 =====================
class GatedResidualCell(nn.Module):
    """
    单步 transition —— 借鉴 StableGatedPI 的设计。
    公式：
        s_cand = M(a_t) @ s
        g_t    = sigmoid(W_g [a_t, s])
        s'     = (1 - g) * s + g * s_cand
        s'     = L2_normalize(s')
    其中 M(a_t) 用 spectral_norm 限制谱半径，避免长序列爆炸。
    """
    def __init__(self, dim_in: int, dim_state: int, hidden: int = 128):
        super().__init__()
        self.dim_state = dim_state
        # transition M：a_t -> ΔM（dim_state x dim_state）
        self.M_net = nn.Sequential(
            spectral_norm(nn.Linear(dim_in, hidden)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden, dim_state * dim_state)),
        )
        # 更新门 g_t
        self.gate = nn.Sequential(
            nn.Linear(dim_in + dim_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_state),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # s: (B, D_state); a: (B, D_in)
        delta_M = self.M_net(a).view(-1, self.dim_state, self.dim_state)
        s_cand = torch.bmm(delta_M, s.unsqueeze(-1)).squeeze(-1)
        g = self.gate(torch.cat([a, s], dim=-1))
        s = (1.0 - g) * s + g * s_cand
        s = self.norm(s)
        # L2 unit-sphere 投影
        s = F.normalize(s, p=2, dim=-1)
        return s


class ContinuousTimeCell(nn.Module):
    """
    借鉴 MambaLiteSSM：连续时间离散化 h_t = exp(-A*dt) * h_{t-1} + B*u_t
    这里用于把"过程输入"转成"状态增量"，给 readout 用。
    """
    def __init__(self, dim_in: int, dim_state: int, hidden: int = 64):
        super().__init__()
        self.dim_state = dim_state
        self.proj = nn.Linear(dim_in, hidden)
        self.dt_proj = nn.Linear(hidden, dim_state)
        self.B_proj = nn.Linear(hidden, dim_state)
        A_init = torch.arange(1, dim_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(a))
        dt = F.softplus(self.dt_proj(x))
        B = self.B_proj(x)
        A = -torch.exp(self.A_log)
        h = torch.exp(dt * A) * h + dt * B
        return h


# ===================== 主模型 =====================
class PathIntegratorForecaster(nn.Module):
    """
    输入：past_x (B, L_in, 8)
    输出：pred_x (B, T_out, 8)   —— 仅预测过程变量 x1-x8，不预测 y1-y4 / Y
    """
    def __init__(self, dim_x: int = 8, dim_state: int = 128, hidden: int = 128):
        super().__init__()
        self.dim_x = dim_x
        self.dim_state = dim_state

        # 输入编码：x → 隐藏向量作为路径积分的 action a_t
        self.x_proj = nn.Linear(dim_x, hidden)
        self.action_dim = hidden
        # 路径积分单元（借鉴 StableGatedPI：spectral_norm + 门控残差 + L2 投影）
        self.gated_cell = GatedResidualCell(self.action_dim, dim_state, hidden=hidden)
        # 连续时间单元（借鉴 MambaLiteSSM）
        self.cont_cell = ContinuousTimeCell(self.action_dim, dim_state, hidden=hidden)
        # 初始状态
        self.s0 = nn.Parameter(torch.randn(dim_state) * 0.02)

        # rollout head：state + 上一步 x → 下一步 x
        self.x_head = nn.Sequential(
            nn.Linear(dim_state + dim_x, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_x),
        )

    # ---------- 输入编码（过去 L_in 步的 path integrator）----------
    def encode(self, past_x: torch.Tensor) -> torch.Tensor:
        """
        past_x: (B, L, 8)
        返回 s0: (B, dim_state)
        """
        B, L, _ = past_x.shape
        a = self.x_proj(past_x)                            # (B, L, hidden)
        # 循环扫描
        s = F.normalize(self.s0, dim=-1).expand(B, -1)
        h = torch.zeros(B, self.dim_state, device=past_x.device)
        last_s = s
        for t in range(L):
            s = self.gated_cell(s, a[:, t])
            h = self.cont_cell(h, a[:, t])
            last_s = s
        s0 = last_s + 0.1 * h
        return s0

    # ---------- 自回归 rollout ----------
    @torch.no_grad()
    def rollout(self, past_x: torch.Tensor, T_out: int) -> torch.Tensor:
        """推理时 rollout T_out 步；返回 pred_x (B, T_out, 8)。"""
        self.eval()
        return self._rollout(past_x, T_out)

    def _rollout(self, past_x: torch.Tensor, T_out: int,
                 x_out: torch.Tensor | None = None) -> torch.Tensor:
        s = self.encode(past_x)                            # (B, D)
        x_last = past_x[:, -1, :]                          # (B, 8)
        pred_x = []
        for t in range(T_out):
            inp = torch.cat([s, x_last], dim=-1)
            dx = self.x_head(inp)
            x_next = x_last + dx                           # 残差式预测
            pred_x.append(x_next)
            # teacher forcing: 用真实值更新 x_last 和状态 s
            if x_out is not None:
                x_last = x_out[:, t]
                a = self.x_proj(x_out[:, t])
            else:
                x_last = x_next
                a = self.x_proj(x_next)
            s = self.gated_cell(s, a)
        pred_x = torch.stack(pred_x, dim=1)                # (B, T, 8)
        return pred_x

    def forward(self, past_x: torch.Tensor, T_out: int,
                x_out: torch.Tensor | None = None) -> dict:
        """
        past_x: (B, L_in, 8)
        T_out:  预测步数
        x_out:  (B, T_out, 8) 真实未来值（标准化空间），用于 teacher forcing；
                None 时使用纯自回归。
        """
        pred_x = self._rollout(past_x, T_out, x_out=x_out)
        return {"pred_x": pred_x}


# ===================== 训练/评估工具 =====================
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """mask 为 1/0 浮点 (B, T, ...)；缺省全部 1。"""
    if mask is None:
        return F.mse_loss(pred, target)
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


if __name__ == "__main__":
    m = PathIntegratorForecaster(dim_x=8, dim_y=4, dim_state=64, hidden=64)
    B, L_in, T_out = 4, 20, 8
    px = torch.randn(B, L_in, 8)
    py = torch.randn(B, L_in, 4)
    out = m(px, py, T_out=T_out)
    print("pred_x:", out["pred_x"].shape, "pred_y:", out["pred_y"].shape, "pred_Y:", out["pred_Y"].shape)
    print(f"#params: {sum(p.numel() for p in m.parameters()):,}")
    # 跑一次反向
    target_x = torch.randn(B, T_out, 8)
    target_y = torch.randn(B, T_out, 4)
    loss = F.mse_loss(out["pred_x"], target_x) + F.mse_loss(out["pred_y"], target_y)
    loss.backward()
    print(f"loss = {loss.item():.4f};  backward ok")