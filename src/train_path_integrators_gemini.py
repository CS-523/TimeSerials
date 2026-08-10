"""
Pretraining Script & Benchmark Suite for Path Integrator Architectures.
Evaluates model stability across trajectory lengths T = [32, 64, 128, 256].

Norm Drift 越接近 0，说明模型的路径积分在数值尺度上越稳定；越大，说明状态向量的
"长度"在轨迹内部剧烈波动/爆炸，是坍缩/爆炸问题的早期信号。
"""

import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from path_integrators import (
    ResidualPathIntegration,
    RecurrentPositionEncoder,
    StableGatedPI,
    ComplexOrthoPI,
    MambaLiteSSM
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE = 11
DIM_ACTION = 4       # Up, Down, Left, Right
DIM_STRUCTURE = 32
TEMPERATURE = 0.15

# 对齐 pretrain_path_integrator.py 的 DIM_ACTION=16（train_gridworld_unified.py 里的定义）。
# train_model(..., use_action_embedding=True) 时，原始 4 维 one-hot 动作会先过一个可训练的
# nn.Embedding(4, ACTION_EMBED_DIM) 再喂给 path integrator（而不是直接用 4 维 one-hot）。
# pretrain_path_integrator.py 的消融实验（见其顶部文档"--stable-gated-onehot"一节）已经发现：
# 在 --fresh-boundary clamp（能正常收敛）环境下，可训练 embedding 比固定 one-hot 收敛明显更快、
# 上限更高；这里加上同样的开关方便直接在本脚本的 benchmark 环境里复现/验证这一结论。
ACTION_EMBED_DIM = 16

ACTIONS = [
    (0, 1),   # Up
    (0, -1),  # Down
    (-1, 0),  # Left
    (1, 0)    # Right
]


def set_seed(seed):
    """固定 random/torch/torch.cuda 的随机种子，让同参数的多次运行可复现。
    seed=None（默认）时不做任何设置，保持原有的完全随机行为。"""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_grid_trajectories(batch_size, seq_len, grid_size=GRID_SIZE):
    action_seqs = torch.zeros(batch_size, seq_len, 4)
    pos_seqs = torch.zeros(batch_size, seq_len, 2, dtype=torch.long)

    for b in range(batch_size):
        curr_x = random.randint(0, grid_size - 1)
        curr_y = random.randint(0, grid_size - 1)

        for t in range(seq_len):
            act_idx = random.randint(0, 3)
            dx, dy = ACTIONS[act_idx]
            curr_x = max(0, min(grid_size - 1, curr_x + dx))
            curr_y = max(0, min(grid_size - 1, curr_y + dy))

            action_seqs[b, t, act_idx] = 1.0
            pos_seqs[b, t, 0] = curr_x
            pos_seqs[b, t, 1] = curr_y

    return action_seqs.to(DEVICE), pos_seqs.to(DEVICE)


def position_separability_loss(structurals, positions, temperature=TEMPERATURE):
    B, T, D = structurals.shape
    flat_structs = F.normalize(structurals.reshape(B * T, D), p=2, dim=-1)
    flat_pos = positions.reshape(B * T, 2)

    sim_matrix = torch.matmul(flat_structs, flat_structs.T) / temperature

    pos_x_eq = flat_pos[:, 0].unsqueeze(0) == flat_pos[:, 0].unsqueeze(1)
    pos_y_eq = flat_pos[:, 1].unsqueeze(0) == flat_pos[:, 1].unsqueeze(1)
    pos_mask = (pos_x_eq & pos_y_eq).float()

    diag_mask = torch.eye(B * T, device=structurals.device)
    pos_mask = pos_mask * (1.0 - diag_mask)

    exp_sim = torch.exp(sim_matrix) * (1.0 - diag_mask)
    sum_exp_sim = exp_sim.sum(dim=-1, keepdim=True) + 1e-8

    pos_sim = torch.exp(sim_matrix) * pos_mask
    pos_loss = -torch.log((pos_sim.sum(dim=-1) + 1e-8) / sum_exp_sim)

    valid_mask = pos_mask.sum(dim=-1) > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=structurals.device, requires_grad=True)
    return pos_loss[valid_mask].mean()


def embed_actions(action_seqs, action_embedding):
    """action_seqs 是 (B, T, 4) 的 one-hot 动作向量；若传入 action_embedding（nn.Embedding(4, D)），
    先转成离散动作 id 再过 embedding，得到 (B, T, D) 连续向量；否则原样返回 one-hot 向量。"""
    if action_embedding is None:
        return action_seqs
    action_ids = action_seqs.argmax(dim=-1)
    return action_embedding(action_ids)


def _evaluate_metrics_chunk(structurals, pos_seqs):
    """对单个 chunk（B_chunk 条轨迹）计算 nn_acc/gap/norm_drift，避免一次性 materialize 全量 N×N 矩阵。"""
    B_chunk, T, D = structurals.shape
    flat_structs = F.normalize(structurals.reshape(B_chunk * T, D), p=2, dim=-1)
    flat_pos = pos_seqs.reshape(B_chunk * T, 2)

    sim_matrix = torch.matmul(flat_structs, flat_structs.T)

    pos_x_eq = flat_pos[:, 0].unsqueeze(0) == flat_pos[:, 0].unsqueeze(1)
    pos_y_eq = flat_pos[:, 1].unsqueeze(0) == flat_pos[:, 1].unsqueeze(1)
    same_pos_mask = (pos_x_eq & pos_y_eq)

    diag_mask = torch.eye(B_chunk * T, dtype=torch.bool, device=structurals.device)
    same_pos_mask = same_pos_mask & (~diag_mask)
    diff_pos_mask = (~same_pos_mask) & (~diag_mask)

    pos_sims = sim_matrix[same_pos_mask]
    neg_sims = sim_matrix[diff_pos_mask]

    avg_pos_sim = pos_sims.mean().item() if pos_sims.numel() > 0 else 0.0
    avg_neg_sim = neg_sims.mean().item() if neg_sims.numel() > 0 else 0.0
    gap = avg_pos_sim - avg_neg_sim

    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked[diag_mask] = -1e9
    nearest_indices = torch.argmax(sim_matrix_masked, dim=-1)

    nearest_pos = flat_pos[nearest_indices]
    correct = (nearest_pos[:, 0] == flat_pos[:, 0]) & (nearest_pos[:, 1] == flat_pos[:, 1])
    nn_acc = correct.float().mean().item()

    norms = torch.norm(structurals, p=2, dim=-1)
    norm_drift = norms.std(dim=-1).mean().item()

    return nn_acc, gap, norm_drift


def evaluate_metrics(model, seq_len=256, eval_batches=16, batch_size=16, action_embedding=None):
    """按 eval_batches 循环，每次只生成/前向/计算 batch_size 条轨迹，
    避免一次性把 eval_batches*batch_size 条轨迹拼成一个 N×N（N = eval_batches*batch_size*seq_len）
    相似度矩阵——序列长（seq_len 大）时该矩阵会平方级增长导致显存 OOM（详见诊断记录）。
    分块计算后再对 eval_batches 个 chunk 的指标取平均，近似原来的全量指标。
    """
    model.eval()
    with torch.no_grad():
        nn_accs, gaps, norm_drifts = [], [], []
        for _ in range(eval_batches):
            action_seqs, pos_seqs = generate_grid_trajectories(batch_size, seq_len)
            structurals = model(embed_actions(action_seqs, action_embedding))
            nn_acc, gap, norm_drift = _evaluate_metrics_chunk(structurals, pos_seqs)
            nn_accs.append(nn_acc)
            gaps.append(gap)
            norm_drifts.append(norm_drift)

    return (
        sum(nn_accs) / len(nn_accs),
        sum(gaps) / len(gaps),
        sum(norm_drifts) / len(norm_drifts),
    )


def train_model(engine_name, seq_len=256, steps=300, batch_size=32, lr=1e-3, use_anchor_loss=True,
                 use_action_embedding=False, tbptt=False, window_steps=None, seed=None):
    """
    Complete PyTorch training loop for Path Integrators:
    Forward Pass -> InfoNCE + Anchor Loss -> Backpropagation BPTT -> Optimizer Step

    use_action_embedding: 若为 True，跳过直接喂 4 维 one-hot 动作向量，改为先过一个随机初始化、
        可训练的 nn.Embedding(4, ACTION_EMBED_DIM) 再喂给 path integrator（对齐
        pretrain_path_integrator.py 默认的 --pi-type stable_gated 做法，dim_action 相应变为
        ACTION_EMBED_DIM=16）。用于在本 benchmark 环境里验证可训练 embedding 相对固定 one-hot
        的提升幅度。

    tbptt / window_steps: 截断反向传播（Truncated BPTT），对齐 pretrain_path_integrator.py::
        run_pretraining_tbptt 的做法。开启后把长度为 seq_len 的轨迹沿时间切成多个连续的长度为
        window_steps 的窗口，依次对每个窗口做前向 + 计算 sep_loss + 反传 + 更新，并把窗口末状态
        detach 后作为下一个窗口的 prev_structural 传入下一次前向——反传深度严格限制在
        window_steps 以内（避免长序列坍缩），但整条 seq_len 长轨迹的动作-位置数据都会被用于
        训练，而不是只训练"轨迹开头这一段"。tbptt=True 时必须同时指定 window_steps。

    seed: 若指定，训练开始前（模型/embedding 初始化之前）固定随机种子，让同参数的多次运行可复现；
        None（默认）保持原有的完全随机行为。
    """
    if tbptt and window_steps is None:
        raise ValueError("tbptt=True 时必须同时指定 window_steps")

    set_seed(seed)

    dim_action = ACTION_EMBED_DIM if use_action_embedding else DIM_ACTION
    action_embedding = nn.Embedding(4, ACTION_EMBED_DIM).to(DEVICE) if use_action_embedding else None

    if engine_name == "ResidualPI":
        model = ResidualPathIntegration(dim_action, DIM_STRUCTURE)
    elif engine_name == "RecurrentEncoder":
        model = RecurrentPositionEncoder(dim_action, DIM_STRUCTURE)
    elif engine_name == "StableGatedPI":
        model = StableGatedPI(dim_action, DIM_STRUCTURE)
    elif engine_name == "ComplexOrthoPI":
        model = ComplexOrthoPI(dim_action, DIM_STRUCTURE)
    elif engine_name == "MambaLiteSSM":
        model = MambaLiteSSM(dim_action, DIM_STRUCTURE)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    model = model.to(DEVICE)

    anchor_probe = nn.Linear(DIM_STRUCTURE, 2).to(DEVICE)
    params = list(model.parameters()) + list(anchor_probe.parameters())
    if action_embedding is not None:
        params += list(action_embedding.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(steps):
        model.train()
        action_seqs, pos_seqs = generate_grid_trajectories(batch_size, seq_len)

        if tbptt:
            prev_structural = None
            for window_start in range(0, seq_len, window_steps):
                window_end = min(window_start + window_steps, seq_len)
                window_actions = embed_actions(action_seqs[:, window_start:window_end], action_embedding)
                window_pos = pos_seqs[:, window_start:window_end]

                structs, last_state = model(window_actions, prev_structural=prev_structural, return_state=True)
                sep_loss = position_separability_loss(structs, window_pos)

                if use_anchor_loss:
                    pred_pos = anchor_probe(structs)
                    target_pos = window_pos.float() / GRID_SIZE
                    anchor_loss = F.mse_loss(pred_pos, target_pos)
                    total_loss = sep_loss + 0.02 * anchor_loss
                else:
                    total_loss = sep_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                prev_structural = last_state.detach()
        else:
            structs = model(embed_actions(action_seqs, action_embedding))

            sep_loss = position_separability_loss(structs, pos_seqs)

            if use_anchor_loss:
                pred_pos = anchor_probe(structs)
                target_pos = pos_seqs.float() / GRID_SIZE
                anchor_loss = F.mse_loss(pred_pos, target_pos)
                total_loss = sep_loss + 0.02 * anchor_loss
            else:
                total_loss = sep_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    nn_acc, gap, norm_drift = evaluate_metrics(model, seq_len=seq_len, action_embedding=action_embedding)
    return nn_acc, gap, norm_drift


def run_benchmark_sweep(use_action_embedding=False, tbptt=False, window_steps=None, seed=None):
    lengths = [32, 64, 128, 256]
    # engines = ["ResidualPI", "RecurrentEncoder", "StableGatedPI", "ComplexOrthoPI", "MambaLiteSSM"]
    engines = ["StableGatedPI"]

    print("==========================================================================")
    print("      PATH INTEGRATOR LONG-SEQUENCE BENCHMARK (T = 32 -> 256)            ")
    print(f"      Device: {DEVICE}")
    print(f"      use_action_embedding: {use_action_embedding}")
    print(f"      tbptt: {tbptt}, window_steps: {window_steps}, seed: {seed}")
    print("==========================================================================")

    for engine in engines:
        print(f"\n--- Testing & Training Engine: {engine} ---")
        print(f"{'Seq Length (T)':<15} | {'nn_acc':<12} | {'gap':<12} | {'Norm Drift':<12}")
        print("-" * 60)
        for T in lengths:
            nn_acc, gap, norm_drift = train_model(
                engine, seq_len=T, steps=10000, use_action_embedding=use_action_embedding,
                tbptt=tbptt, window_steps=window_steps, seed=seed,
            )
            print(f"{T:<15} | {nn_acc:<12.4f} | {gap:<12.4f} | {norm_drift:<12.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-action-embedding', action='store_true',
                         help='默认关闭：喂固定 4 维 one-hot 动作向量给 path integrator（本脚本原有行为，'
                              '对应 pretrain_path_integrator.py 里 --stable-gated-onehot 开启时的做法）。'
                              '开启后，动作先过一个随机初始化、可训练的 nn.Embedding(4, ACTION_EMBED_DIM=16)'
                              '（对应 pretrain_path_integrator.py 默认的 --pi-type stable_gated 做法，'
                              '不开 --stable-gated-onehot 时的行为）。pretrain_path_integrator.py 的消融'
                              '实验已发现：在能正常收敛的环境下，可训练 embedding 比固定 one-hot 收敛明显'
                              '更快、上限更高，这里加上同样的开关方便在本 benchmark 环境里直接复现/验证。')
    parser.add_argument('--tbptt', action='store_true',
                         help='开启截断反向传播（Truncated BPTT），对齐 pretrain_path_integrator.py::'
                              '--tbptt 的做法。需要同时指定 --window-steps。')
    parser.add_argument('--window-steps', type=int, default=None,
                         help='--tbptt 模式下每个窗口的长度（反传深度上限）。')
    parser.add_argument('--seed', type=int, default=None,
                         help='固定随机种子，让同参数的多次运行可复现；不传则保持完全随机（默认）。')
    args = parser.parse_args()
    run_benchmark_sweep(
        use_action_embedding=args.use_action_embedding,
        tbptt=args.tbptt, window_steps=args.window_steps, seed=args.seed,
    )

    
    
