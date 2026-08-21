# PathIntegrator 实现原理（时序预测版）

> 本文解释 `src/model_forecaster.py` 里 `PathIntegratorForecaster` 的**当前实现**。
> 它借鉴了 `src/path_integrators.py` 中的 `StableGatedPI`（谱归一化 + 门控残差 + L2 投影）
> 与 `MambaLiteSSM`（连续时间离散化），但针对"时序预测没有显式动作"这一事实做了适配。
>
> 结论先行：**代码已经实现了"三重稳定化"推荐方案，本轮不改代码**，本文只把原理讲清楚。

---

## 1. 两段式总览

`PathIntegratorForecaster`（[model_forecaster.py:95](model_forecaster.py#L95)）是一个
"编码 → 自回归 rollout" 的两段结构：

```
past_x (B, L_in, 8)  ── encode ──▶  s0 (B, D)  ── rollout ──▶  pred_x (B, T_out, 8)
```

- **encode**：把过去 `L_in` 步的过程变量 `x1-x8` "积分"成一个状态向量 `s0`（路径积分编码器）；
- **rollout**：从 `s0` 出发自回归地预测未来 `T_out` 步的 `x1-x8`（残差式，`x_next = x_last + Δx`）。

本版本**只预测 x1-x8，不预测 y1-y4 / Y**（y 标签稀疏，引入 y 头会拖累训练稳定性）。

---

## 2. 动作的来源：没有显式动作怎么办

原始 PathIntegrator（[path_integrators.py](path_integrators.py)）假设输入是一串**显式动作**
`actions`（如 `dim_action=4` 的控制量）。但本时序任务里唯一的观测是过程变量 `x1-x8`，**没有动作**。

当前做法：**把观测本身投影成"伪动作"**。在 [model_forecaster.py:106](model_forecaster.py#L106)
定义了一个线性投影层：

```python
self.x_proj = nn.Linear(dim_x, hidden)   # 8 → 128
self.action_dim = hidden
```

于是每一步的"动作"是观测的线性投影：

```
a_t = x_proj(x_t) = W_proj · x_t ∈ R^128        (x_t ∈ R^8)
```

这三处都统一用 `x_proj` 造动作：

| 位置 | 代码 | 动作来源 |
| --- | --- | --- |
| 编码器扫历史 | [model_forecaster.py:129](model_forecaster.py#L129) | `a = self.x_proj(past_x)`，把整段历史投影成动作序列 |
| rollout·teacher forcing | [model_forecaster.py:160-161](model_forecaster.py#L160-L161) | `a = self.x_proj(x_out[:, t])`，投影真实未来值 |
| rollout·纯自回归 | [model_forecaster.py:163-164](model_forecaster.py#L163-L164) | `a = self.x_proj(x_next)`，投影自己预测的值 |

**概念对应**：

| | 原始 PathIntegrator | 时序预测版（本文对象） |
| --- | --- | --- |
| 动作 `a_t` | 外部控制输入 | **观测的投影** `W_proj·x_t`，无外部输入 |
| 被积的"路径" | 动作序列编码的运动轨迹 | 过程变量 `x` 自身的轨迹 |
| 状态 `s_t` | 对动作历史的运行总结 | 对 `x` 历史（过去 `L_in` 步）的运行总结 |
| 输出 | 状态序列 | 状态 `s` 送入 `x_head` 预测 `Δx` |

一句话：**把观测 `x` 当作自生的伪动作，让 PI 去积分 `x` 的轨迹得到状态 `s`，再用 `s` 做自回归预测。**
这样在没有任何显式动作的情况下，把"路径积分"这套稳定化机制完整移植到了时序预测上。

---

## 3. 核心单元 GatedResidualCell（三重稳定化）

`GatedResidualCell`（[model_forecaster.py:31](model_forecaster.py#L31)）是路径积分的单步转移，
对应参考实现里的 `StableGatedPI`。它的 `forward(s, a)`（[model_forecaster.py:59-68](model_forecaster.py#L59-L68)）：

```python
delta_M = self.M_net(a).view(-1, D, D)      # (B, D, D)  转移矩阵
s_cand  = torch.bmm(delta_M, s.unsqueeze(-1)).squeeze(-1)  # 候选状态
g       = self.gate(torch.cat([a, s], dim=-1))             # 更新门
s       = (1.0 - g) * s + g * s_cand                       # 门控凸组合
s       = self.norm(s)                                     # LayerNorm
s       = F.normalize(s, p=2, dim=-1)                      # L2 球面投影
```

写成逐项公式（`D = dim_state`，`a_t` 为第 2 节的伪动作）：

```
ΔM_t    = reshape( W_2 · GELU( W_1 · a_t ) )          ∈ R^{D×D}    (1) 转移矩阵
s_cand  = ΔM_t · s_{t-1}                                        (2) 候选状态
g_t     = σ( W'_2 · GELU( W'_1 · [a_t ; s_{t-1}] ) )  ∈ (0,1)^D   (3) 更新门
s'_t    = (1 - g_t) ⊙ s_{t-1} + g_t ⊙ s_cand                    (4) GRU 式凸组合
s''_t   = LayerNorm(s'_t)                                         (5) 逐维归一化
s_t     = s''_t / ‖s''_t‖₂                                       (6) L2 球面投影
```

其中：

- `W_1, W_2` 都套了 `spectral_norm`（[model_forecaster.py:45-49](model_forecaster.py#L45-L49)）；
- 初始状态 `s_0` 来自可学习参数 `self.s0`，在 `encode` 里先做单位化
  `s = F.normalize(self.s0, dim=-1)`（[model_forecaster.py:131](model_forecaster.py#L131)）。

### 三重稳定化各起什么作用

| 手段 | 公式位置 | 作用 |
| --- | --- | --- |
| **谱归一化** | (1) 的 `W_1, W_2` | 限制转移矩阵生成器的增益，抑制长序列累积放大 |
| **门控凸组合** | (4) | 门 `g_t` 在 `(0,1)^D`，`(1-g)⊙s_{t-1}` 起跨步跳连（skip connection），缓解梯度消失 |
| **L2 球面投影** | (6) | 每步强制 `‖s_t‖₂ = 1`，杜绝尺度漂移 |

其中 **L2 投影是最硬的约束**：它让状态模长恒为 1，网络被迫只能靠调整向量的**方向**（空间夹角）
来区分不同状态，从根本上杜绝了"尺度漂移带来的数值坍缩或爆炸"。这正是推荐方案里强调
"强制模长为 1 → 丧失在长度上做文章的自由度"的设计意图。

---

## 4. 辅助支路 ContinuousTimeCell（Mamba-Lite）

`encode` 里除了 `GatedResidualCell`，还并行跑一条连续时间支路 `ContinuousTimeCell`
（[model_forecaster.py:71](model_forecaster.py#L71)，借鉴 `MambaLiteSSM`）：

```python
x  = F.gelu(self.proj(a))
dt = F.softplus(self.dt_proj(x))
B  = self.B_proj(x)
A  = -torch.exp(self.A_log)
h  = torch.exp(dt * A) * h + dt * B
```

即离散化形式：

```
h_t = exp(dt · A) · h_{t-1} + dt · B
```

其中 `dt` 由 softplus 保证为正，`A = -exp(A_log)` 保证为负（`exp(dt·A) < 1`，指数遗忘）。

两条支路在编码末尾合并（[model_forecaster.py:138](model_forecaster.py#L138)）：

```
s0 = last_s + 0.1 * h
```

`0.1` 是固定的缩放系数，让连续时间支路只作为**小幅补充记忆**叠加在门控路径积分结果上。

---

## 5. encode 与 rollout 完整数据流

### encode（[model_forecaster.py:123-139](model_forecaster.py#L123-L139)）

```
输入 past_x (B, L_in, 8)
a = x_proj(past_x)                        # (B, L_in, hidden)
s = normalize(s0).expand(B, -1)           # 初始状态单位化
h = zeros(B, D)
for t in 1..L_in:
    s = GatedResidualCell(s, a[:, t])     # 门控路径积分
    h = ContinuousTimeCell(h, a[:, t])    # 连续时间支路
s0 = s + 0.1 * h                          # 合并两条支路
```

### rollout（[model_forecaster.py:148-167](model_forecaster.py#L148-L167)）

```
s = encode(past_x)
x_last = past_x[:, -1, :]
for t in 1..T_out:
    dx      = x_head([s ; x_last])        # 状态 + 上一步 x → Δx
    x_next  = x_last + dx                 # 残差式预测
    记录 x_next
    if teacher forcing:
        x_last = x_out[:, t]              # 用真实未来值
    else:
        x_last = x_next                   # 用自己预测的值
    a = x_proj(x_last)
    s = GatedResidualCell(s, a)           # 状态随新动作继续演化
```

`x_head` 是一个两层 MLP（[model_forecaster.py:116-120](model_forecaster.py#L116-L120)）：
`Linear(dim_state + dim_x → hidden) → GELU → Linear(hidden → dim_x)`。

### 状态 → 输出的映射：x_head 残差

路径积分得到状态 `s` 之后，**不是**直接把 `s` 解码成 x，而是把 `s` 当作"上下文"，与上一步的 x
拼起来，让 `x_head` 预测**增量** `Δx`，再残差相加：

```
inp    = [ s_t ; x_last ]         ∈ R^{D+8}        # 拼接状态(D=128) 与上一步 x(8)
Δx_t   = x_head(inp)               ∈ R^8           # 2 层 MLP 预测增量
x_next = x_last + Δx_t                              # 残差：只预测变化量
```

两个要点：

1. **残差式输出**：网络只需预测 `Δx`（相对上一步的偏移），而不是绝对值——这大幅降低拟合难度，
   与 rollout 里"用上一步 x 当锚点"的设计一致。

2. **状态是"递推"而非"累加"**：`s_t` 是每一步被 `GatedResidualCell` 递推更新后的**单个隐状态**，
   不是把所有中间状态沿时间求和。
   - `encode` 扫完 `L_in` 步只保留**最后一步**的状态（`last_s + 0.1*h`），中间状态全部丢弃；
   - `rollout` 里每一步用**当前**状态 `s_t` 拼上 `x_last` 做一次读出，随后状态再随新动作继续递推。

   "积分"体现在这种**逐步递推**里——就像 ODE 积分器或 RNN，最终那个状态本身就是整条路径的
   压缩表示，而不是把每步的 `s` 显式求和。

---

## 6. 与参考 StableGatedPI 的 3 处差异（均已确认保留）

当前 `GatedResidualCell` 相对参考实现 `StableGatedPI`（[path_integrators.py:70](path_integrators.py#L70)）
有 3 处**刻意保留**的差异：

| # | 参考方案 | 当前实现 | 差异 |
| --- | --- | --- | --- |
| 1 | 更新门仅依赖动作 `g_t = σ(...(a_t))` | 门依赖 `[a_t ; s_{t-1}]` | 门能感知当前状态，表达能力更强 |
| 2 | 仅 L2 球面投影 | `LayerNorm` 后再 L2 | 多一层逐维可学习归一化 |
| 3 | 谱归一化套在生成层（同） | 同 | 无差异（见第 7 节 caveat） |

即：当前代码 = 推荐方案 + 「门感知状态」+「LayerNorm」两个增强项，谱约束严格程度与参考实现同源。

---

## 7. 关键 caveat：谱归一化到底约束了什么

代码注释写"spectral_norm 限制谱半径，避免长序列爆炸"，这里**略有夸大**，需要澄清：

`spectral_norm` 套在 `nn.Linear(hidden, D×D)` 上，约束的是这条**线性映射的算子范数**
`σ_max(W_2) = 1`，也就是 `R^hidden → R^{D²}` 这个空间的增益；它**并不等于** reshape 出来的
`D×D` 矩阵 `M_t` 自身的谱范数 `‖M_t‖₂`。

因此：

- `‖M_t‖₂ ≤ 1` 只是**近似成立**（由生成器增益间接限制）；
- 真正硬性防止状态爆炸 / 漂移的，是每步末尾的 **L2 球面投影**——`‖s_t‖₂ = 1` 恒成立，
  状态模长被死死钉在单位球面上，无论 `M_t` 的谱范数如何都不会发散。

若日后需要更严格的 `‖M_t‖₂ ≤ 1`，可对 reshape 出的 `D×D` 矩阵做真正的矩阵级谱归一化
（幂迭代 / SNGAN 风格，或 Frobenius 缩放兜底），但当前方案依赖 L2 投影已足够稳定。

---

## 8. 多组扩展（简述）

`src/model_multigroup.py` 里的两个 PathInt 变体**复用同一个 `GatedResidualCell`**，仅头 / 分组方式不同：

- `PathIntegratorForecasterFiLM`（[model_multigroup.py:165](model_multigroup.py#L165)）：
  共享 PathInt 编码器，每个 group 一个 FiLM 头 `(γ_g, β_g) ∈ R^8`，作用于 decode 每步的 `x_last`；
  状态 `s` 的演化（`gated_cell`）不受 FiLM 影响。
- `PathIntegratorForecaster5Models`（[model_multigroup.py:221](model_multigroup.py#L221)）：
  5 个独立 `PathIntegratorForecaster`，按 `group_id` 选对应模型。

两者在 rollout 中更新状态用的都是同一段逻辑（`a = x_proj(x)` → `s = gated_cell(s, a)`），
与单模型版完全一致。
