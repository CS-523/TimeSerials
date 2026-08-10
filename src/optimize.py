"""
优化模型：在固定初始状态（输入一段时间 x）下，寻找未来 T 步的
x3 / x4 / x6 / x8 控制序列，使预测的 y4（或 Y）最大。

两套方法：
  1. 梯度法：复用训练好的 PathIntegratorForecaster
       - 把 x3/x4/x6/x8 在未来 T_out 步设为可学习参数（其它 x 列由模型自回归 rollout 得到）
       - 反向传播最大化 y4（y4 头 + 末端 Y 头加权和）
  2. 进化策略（ES）/CEM：在控制参数空间采样，最大化预测 y4

评估：在测试集上对每个实验输入前 L_in 步，输出"最优控制序列 + 优化得到的 y4 / Y"。

使用：
    python src/optimize.py --method gradient
    python src/optimize.py --method es
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, X_COLS, Y_INT_COLS, split_experiments, Scaler, YScaler
from model_forecaster import PathIntegratorForecaster


# x3 / x4 / x6 / x8 在 X_COLS 里的索引
CTRL_IDX = [X_COLS.index(c) for c in ("x3", "x4", "x6", "x8")]


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = PathIntegratorForecaster(
        dim_x=8, dim_y=4, dim_state=args["dim_state"], hidden=args["hidden"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    x_scaler = Scaler(mean=ckpt["x_scaler"]["mean"], std=ckpt["x_scaler"]["std"])
    y_scaler = YScaler(means=ckpt["y_scaler"]["mean"], stds=ckpt["y_scaler"]["std"])
    return model, x_scaler, y_scaler


# ====================== 1. 梯度法 ======================
def gradient_optimize(model, x_scaler, y_scaler, x_in: torch.Tensor, y_in: torch.Tensor,
                      T_out: int, n_iters: int = 80, lr: float = 0.05,
                      x_init: np.ndarray | None = None,
                      x_bounds_norm: tuple = (-2.5, 2.5),
                      lambda_y4: float = 1.0, lambda_Y: float = 0.3,
                      w_smooth: float = 0.01,
                      device: torch.device = torch.device("cpu")) -> dict:
    """
    x_in:  (1, L_in, 8) — 标准化空间
    y_in:  (1, L_in, 4)
    x_init: (T_out, 8) — 初始猜测（标准化空间），缺省用 x_in 最后一步重复
    返回：最优控制序列、最优 y4 / Y
    """
    model.eval()
    # 初始化控制（保持 leaf 状态以便 optimizer 优化）
    if x_init is None:
        x_init = x_in[0, -1:, :].repeat(T_out, 1).detach().cpu().numpy()
    ctrl = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=True)
    # 二次确认是 leaf
    assert ctrl.is_leaf
    # 不 unsqueeze：optimizer 只对 leaf 友好；后续 forward 时再 unsqueeze

    # 把 x_in / y_in 复制出 batch 大小为 1
    x_in_t = x_in.to(device)
    y_in_t = y_in.to(device)

    opt = torch.optim.Adam([ctrl], lr=lr)

    best_y4 = -1e9
    best_ctrl = None

    # 让所有"非控制列"由模型自回归生成：
    # 我们用"原 x_head"生成完整 x，再把控制列替换为 ctrl
    # 但更直接：x_head 输出 Δx，我们让 (x_last + Δx) 中的控制列 = ctrl
    # 实现：用 ctrl 替换预测 x 的控制列
    with torch.enable_grad():
        for it in range(n_iters):
            # 临时让模型输出 Δx（残差），然后把控制列替换为 ctrl
            # 复用 forward 但 x_head 输出的是"绝对" x（我已经写成 x_last + dx）
            # 所以我们做：model forward → pred_x (1, T_out, 8) → 控制列强制 = ctrl
            out = model(x_in_t, y_in_t, T_out=T_out)
            pred_x = out["pred_x"]
            # 控制列替换
            x_full = pred_x.clone()
            x_full[:, :, CTRL_IDX] = ctrl.unsqueeze(0)[:, :, CTRL_IDX]
            # 平滑正则：相邻步控制变化量
            d_ctrl = ctrl.unsqueeze(0)[:, 1:, :] - ctrl.unsqueeze(0)[:, :-1, :]
            smooth_loss = (d_ctrl ** 2).mean()

            # 重新跑一次 readout（因为控制列改了）
            # 简化：让 pred_y 也基于修改后的 x 重算
            # 重新 encode + rollout
            s = model.encode(x_in_t, y_in_t)
            x_last = x_in_t[:, -1, :]
            ys = []
            for t in range(T_out):
                inp = torch.cat([s, x_last], dim=-1)
                y_t = model.y_head(inp)
                ys.append(y_t)
                x_next = x_full[:, t, :]
                x_last = x_next
                a = model.x_proj(x_next) + model.y_proj(y_t)
                s = model.gated_cell(s, a)
            ys = torch.stack(ys, dim=1)
            # 末端 Y
            Y_pred = model.Y_head(torch.cat([s, x_last], dim=-1))[0, 0]
            # 目标：最大化末步 y4 + 0.3 * Y - 0.01 * 平滑
            y4 = ys[0, -1, 3]                   # 标准化空间
            # 把 y4 / Y 还原到原始空间只是显示用，不影响优化
            loss = -y4 - lambda_Y * Y_pred + w_smooth * smooth_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            # 限幅到合理标准化范围
            with torch.no_grad():
                ctrl.clamp_(x_bounds_norm[0], x_bounds_norm[1])

            if (-y4).item() > best_y4:
                best_y4 = (-y4).item()
                best_ctrl = ctrl.detach().clone()

    # 用 best_ctrl 重新 rollout 一次得到 y4 / Y
    with torch.no_grad():
        s = model.encode(x_in_t, y_in_t)
        x_last = x_in_t[:, -1, :]
        x_full = best_ctrl.unsqueeze(0)  # (1, T_out, 8)
        # 非控制列用模型自回归得到
        out0 = model(x_in_t, y_in_t, T_out=T_out)
        non_ctrl_idx = [i for i in range(8) if i not in CTRL_IDX]
        x_full[:, :, non_ctrl_idx] = out0["pred_x"][:, :, non_ctrl_idx]
        # rollout
        s = model.encode(x_in_t, y_in_t)
        x_last = x_in_t[:, -1, :]
        ys = []
        for t in range(T_out):
            inp = torch.cat([s, x_last], dim=-1)
            y_t = model.y_head(inp)
            ys.append(y_t)
            x_last = x_full[:, t, :]
            a = model.x_proj(x_last) + model.y_proj(y_t)
            s = model.gated_cell(s, a)
        ys = torch.stack(ys, dim=1)
        Y_pred = model.Y_head(torch.cat([s, x_last], dim=-1))[0, 0]

    # 反标准化
    y4_pred = float(ys[0, -1, 3].cpu()) * float(y_scaler.stds[3]) + float(y_scaler.means[3])
    Y_pred_orig = float(Y_pred.cpu()) * float(np.array([e.Y for e in [type('o', (), {'Y': 1})()]]).std())  # 占位，下面用真实换算

    ctrl_orig = best_ctrl[0].cpu().numpy() * x_scaler.std + x_scaler.mean

    return {
        "ctrl_x": ctrl_orig,                     # (T_out, 8) 原始空间
        "ctrl_x_norm": best_ctrl[0].cpu().numpy(),  # (T_out, 8) 标准化空间
        "y4_pred_norm": float(ys[0, -1, 3].cpu()),
        "Y_pred_norm": float(Y_pred.cpu()),
        "y4_pred_orig": y4_pred,
        "Y_pred_norm_full": float(Y_pred.cpu()),
    }


# ====================== 2. 进化策略 ======================
def es_optimize(model, x_scaler, y_scaler, x_in: torch.Tensor, y_in: torch.Tensor,
                T_out: int, n_iters: int = 50, pop_size: int = 32,
                sigma: float = 0.5, elite_frac: float = 0.25,
                x_init: np.ndarray | None = None,
                x_bounds_norm: tuple = (-2.5, 2.5),
                device: torch.device = torch.device("cpu")) -> dict:
    """
    进化策略：直接在标准化空间的 (T_out, 4) 控制变量（仅 x3/x4/x6/x8）上做 CEM 搜索。
    """
    model.eval()
    n_ctrl = len(CTRL_IDX)
    if x_init is None:
        x_init = x_in[0, -1, CTRL_IDX].detach().cpu().numpy()
        x_init = np.tile(x_init, (T_out, 1))
    mu = torch.tensor(x_init, dtype=torch.float32, device=device)
    sd = torch.ones_like(mu) * sigma

    def _fitness(ctrl_pop: torch.Tensor) -> torch.Tensor:
        """
        ctrl_pop: (P, T_out, 4) — 仅 x3/x4/x6/x8 标准化值
        """
        P = ctrl_pop.size(0)
        # 构造完整 x_pop (P, T_out, 8)
        x_pop = torch.zeros(P, T_out, 8, device=device)
        # 静态列用 x_in 最后一步重复
        last_x = x_in[:, -1, :].expand(P, -1, -1)  # (P, 1, 8)
        # 先 rollout 一次 base 序列（非控制列）
        with torch.no_grad():
            x_in_rep = x_in.expand(P, -1, -1)
            y_in_rep = y_in.expand(P, -1, -1)
            out = model(x_in_rep, y_in_rep, T_out=T_out)
            base_x = out["pred_x"]
        # 填控制列
        x_pop = base_x.clone()
        for k, idx in enumerate(CTRL_IDX):
            x_pop[:, :, idx] = ctrl_pop[:, :, k]
        # 重新做 y readout（用修改后的 x 序列）
        with torch.no_grad():
            s = model.encode(x_in_rep, y_in_rep)
            x_last = x_in_rep[:, -1, :]
            ys = []
            for t in range(T_out):
                inp = torch.cat([s, x_last], dim=-1)
                y_t = model.y_head(inp)
                ys.append(y_t)
                x_last = x_pop[:, t, :]
                a = model.x_proj(x_last) + model.y_proj(y_t)
                s = model.gated_cell(s, a)
            ys = torch.stack(ys, dim=1)
            Y_pred = model.Y_head(torch.cat([s, x_last], dim=-1))[:, 0]
        # fitness = 末端 y4
        return ys[:, -1, 3] + 0.3 * Y_pred

    best_y = -1e9
    best_ctrl = None
    n_elite = max(2, int(pop_size * elite_frac))
    for it in range(n_iters):
        # 采样
        eps = torch.randn(pop_size, T_out, n_ctrl, device=device)
        ctrl_pop = mu.unsqueeze(0) + sd.unsqueeze(0) * eps
        ctrl_pop = ctrl_pop.clamp(x_bounds_norm[0], x_bounds_norm[1])
        fitness = _fitness(ctrl_pop)
        # 取 elite
        topk = torch.topk(fitness, n_elite).indices
        elite = ctrl_pop[topk]
        # 更新
        mu = elite.mean(dim=0)
        sd = (elite - mu.unsqueeze(0)).std(dim=0).clamp(min=0.05)
        if fitness.max().item() > best_y:
            best_y = fitness.max().item()
            best_ctrl = ctrl_pop[fitness.argmax()].detach().clone()
    if best_ctrl is None:
        best_ctrl = mu
    return {
        "ctrl_norm": best_ctrl.cpu().numpy(),   # (T_out, 4)
        "best_score_norm": best_y,
    }


# ====================== 评估：在测试集上对每条实验做优化 ======================
@torch.no_grad()
def _rollout_with_ctrl(model, x_scaler, y_scaler, x_in_norm, y_in_norm,
                        T_out, ctrl_norm_full):
    """ctrl_norm_full: (T_out, 8) 标准化空间全 8 列；返回 y4 / Y 原始空间预测。"""
    model.eval()
    x_in_t = x_in_norm.unsqueeze(0)
    y_in_t = y_in_norm.unsqueeze(0)
    s = model.encode(x_in_t, y_in_t)
    x_last = x_in_t[:, -1, :]
    ys = []
    for t in range(T_out):
        inp = torch.cat([s, x_last], dim=-1)
        y_t = model.y_head(inp)
        ys.append(y_t)
        x_last = ctrl_norm_full[t].unsqueeze(0).to(x_in_t.device)
        a = model.x_proj(x_last) + model.y_proj(y_t)
        s = model.gated_cell(s, a)
    ys = torch.stack(ys, dim=1)
    Y_pred = model.Y_head(torch.cat([s, x_last], dim=-1))[0, 0]
    y4_pred_norm = float(ys[0, -1, 3].cpu())
    Y_pred_norm = float(Y_pred.cpu())
    y4_pred_orig = y4_pred_norm * float(y_scaler.stds[3]) + float(y_scaler.means[3])
    return y4_pred_norm, Y_pred_norm, y4_pred_orig


def evaluate_optimization(model, x_scaler, y_scaler, test_exps, device,
                          in_len: int = 24, T_out: int = 12,
                          method: str = "es"):
    """对每条测试实验做优化，比较'自然 rollout' vs '优化控制'下的 y4 / Y。"""
    rows = []
    for ei, exp in enumerate(test_exps):
        if len(exp) < in_len + T_out:
            continue
        # 用前 in_len 步作为输入
        x_raw = exp.df[X_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        y_raw = exp.df[Y_INT_COLS].iloc[:in_len].to_numpy(dtype=np.float32)
        # 标准化
        x_in = torch.from_numpy(x_scaler.transform(x_raw)).float().to(device)
        y_in_mask = (~np.isnan(y_raw)).astype(np.float32)
        y_in_filled = np.nan_to_num(y_raw, nan=0.0)
        y_in_norm = (y_in_filled - y_scaler.means) / y_scaler.stds
        y_in = torch.from_numpy(y_in_norm * y_in_mask).float().to(device)

        # baseline：自然 rollout
        with torch.no_grad():
            out = model(x_in.unsqueeze(0), y_in.unsqueeze(0), T_out=T_out)
        y4_base = float(out["pred_y"][0, -1, 3].cpu()) * float(y_scaler.stds[3]) + float(y_scaler.means[3])
        Y_base = float(out["pred_Y"][0, 0].cpu())
        # 真值：实验末段
        # 取真实 y4 出现位置（最后一段）
        y4_true = exp.df["y4"].dropna()
        y4_true_last = float(y4_true.iloc[-1]) if len(y4_true) > 0 else float("nan")
        # 实验 Y
        Y_true = exp.Y

        # 优化
        if method == "gradient":
            opt_res = gradient_optimize(model, x_scaler, y_scaler,
                                        x_in.unsqueeze(0), y_in.unsqueeze(0),
                                        T_out=T_out, device=device)
            ctrl_norm_full = opt_res["ctrl_x_norm"]  # (T_out, 8)
            y4_opt = opt_res["y4_pred_orig"]
            Y_opt_norm = opt_res["Y_pred_norm_full"]
        else:  # ES
            opt_res = es_optimize(model, x_scaler, y_scaler,
                                  x_in.unsqueeze(0), y_in.unsqueeze(0),
                                  T_out=T_out, device=device)
            # 把 ctrl_norm 嵌入完整 8 列
            x_in_unsq = x_in.unsqueeze(0)
            with torch.no_grad():
                base = model(x_in_unsq, y_in.unsqueeze(0), T_out=T_out)
            full = base["pred_x"][0].cpu().numpy()
            for k, idx in enumerate(CTRL_IDX):
                full[:, idx] = opt_res["ctrl_norm"][:, k]
            ctrl_norm_full = full
            y4_opt_n, Y_opt_n, y4_opt = _rollout_with_ctrl(
                model, x_scaler, y_scaler, x_in, y_in, T_out,
                torch.from_numpy(ctrl_norm_full).float())
            Y_opt_norm = Y_opt_n

        rows.append({
            "exp": os.path.basename(exp.file),
            "group": exp.group,
            "y4_base": y4_base,
            "y4_opt": y4_opt,
            "y4_true_last": y4_true_last,
            "Y_base_norm": Y_base,
            "Y_opt_norm": Y_opt_norm,
            "Y_true": Y_true,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kefu-nas/ybkong/time_serials-master/src/model_out/forecaster_best.pt")
    ap.add_argument("--out", default="/kefu-nas/ybkong/time_serials-master/src/model_out/optimization_results.json")
    ap.add_argument("--method", default="es", choices=["gradient", "es"])
    ap.add_argument("--n-exps", type=int, default=20, help="评估多少条实验（避免太慢）")
    ap.add_argument("--in-len", type=int, default=24)
    ap.add_argument("--T-out", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, x_scaler, y_scaler = load_model(args.ckpt, device)
    print(f"[optimize] loaded ckpt: {args.ckpt}")

    exps = load_all("/kefu-nas/ybkong/time_serials-master")
    _, _, test_exps = split_experiments(exps)
    test_exps = test_exps[:args.n_exps]
    print(f"[optimize] evaluating on {len(test_exps)} test exps, method={args.method}")

    rows = evaluate_optimization(model, x_scaler, y_scaler, test_exps, device,
                                 in_len=args.in_len, T_out=args.T_out, method=args.method)
    # 汇总
    import numpy as np
    y4_b = np.array([r["y4_base"] for r in rows if r["y4_base"] == r["y4_base"]])
    y4_o = np.array([r["y4_opt"] for r in rows if r["y4_opt"] == r["y4_opt"]])
    print(f"\n[optimize] 基线 y4 均值 = {y4_b.mean():.2f}")
    print(f"[optimize] 优化 y4 均值 = {y4_o.mean():.2f}")
    print(f"[optimize] 提升       = {y4_o.mean() - y4_b.mean():.2f} ({(y4_o.mean()/y4_b.mean()-1)*100:.1f}%)")

    with open(args.out, "w") as f:
        json.dump({
            "method": args.method,
            "n_exps": len(rows),
            "rows": rows,
            "summary": {
                "y4_base_mean": float(y4_b.mean()) if len(y4_b) else None,
                "y4_opt_mean": float(y4_o.mean()) if len(y4_o) else None,
                "improvement_abs": float((y4_o - y4_b).mean()) if len(y4_b) and len(y4_o) else None,
                "improvement_pct": float((y4_o.mean() / y4_b.mean() - 1) * 100) if len(y4_b) and len(y4_o) else None,
            }
        }, f, indent=2)
    print(f"[optimize] 结果已写入 {args.out}")


if __name__ == "__main__":
    main()