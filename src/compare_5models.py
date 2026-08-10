"""5 模型快速对比（跳过训练，直接评估 + 画图）"""
import json, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all, sample_windows, Scaler, X_COLS
from model_lstm import LSTMForecaster
from model_forecaster import PathIntegratorForecaster
from model_multigroup import (
    LSTMForecasterFiLM, LSTMForecaster5Models,
    PathIntegratorForecasterFiLM,
)
from train_multigroup import (
    split_experiments_groupwise, WindowXGDataset, pad_collate_xg,
    evaluate as evaluate_mg,
)

BASE = "D:/Code/timeserials_claude/time-serials-mac"
OUT = os.path.join(BASE, "src", "model_out")
SEED = 42
DEVICE = torch.device("cpu")

def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)

def build_model(backbone, mode):
    if backbone == "lstm":
        if mode == "shared":
            return LSTMForecaster(dim_x=8, hidden=128, num_layers=2, dropout=0.1)
        elif mode == "group_head":
            return LSTMForecasterFiLM(n_groups=5, dim_x=8, hidden=128, num_layers=2, dropout=0.1)
        else:
            return LSTMForecaster5Models(n_groups=5, dim_x=8, hidden=128, num_layers=2, dropout=0.1)
    else:  # pathint
        if mode == "shared":
            return PathIntegratorForecaster(dim_x=8, dim_state=128, hidden=128)
        elif mode == "group_head":
            return PathIntegratorForecasterFiLM(n_groups=5, dim_x=8, dim_state=128, hidden=128)

set_seed(SEED)

# data
exps = load_all(BASE)
train_exps, val_exps, test_exps = split_experiments_groupwise(exps, seed=SEED)
x_scaler = Scaler.fit(train_exps)
test_samples = sample_windows(test_exps, rng_seed=SEED + 2)

def attach(samples, subset):
    m = {i: int(e.group) - 1 for i, e in enumerate(subset)}
    for s in samples:
        s.group = m[s.exp_id]
attach(test_samples, test_exps)

test_ds = WindowXGDataset(test_samples, x_scaler)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                         collate_fn=pad_collate_xg, num_workers=0)

# models to evaluate
models = [
    ("LSTM shared",      "lstm",    "shared",      "forecaster_lstm_shared.pt"),
    ("LSTM group_head",  "lstm",    "group_head",  "forecaster_lstm_group_head.pt"),
    ("LSTM independent", "lstm",    "independent", "forecaster_lstm_independent.pt"),
    ("PathInt shared",      "pathint", "shared",      "forecaster_pathint_shared.pt"),
    ("PathInt group_head",  "pathint", "group_head",  "forecaster_pathint_group_head.pt"),
]

results = {}
for display_name, backbone, mode, ckpt_name in models:
    ckpt_path = os.path.join(OUT, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} 不存在")
        continue
    model = build_model(backbone, mode).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate_mg(model, test_loader, DEVICE, mode, x_scaler, return_by_group=True)
    n_params = sum(p.numel() for p in model.parameters())
    results[display_name] = {
        "backbone": backbone, "mode": mode, "n_params": n_params,
        "rmse_x_norm": metrics["rmse_x_norm"], "rmse_x_orig": metrics["rmse_x_orig"],
        "by_group": metrics["by_group"],
    }
    print(f"[{display_name}] 整体 RMSE(orig) = {metrics['rmse_x_orig']:.1f}  ({n_params:,} params)")
    for g in range(5):
        bg = metrics["by_group"].get(g, {})
        if bg.get("rmse_x_orig"):
            print(f"   G{g+1}: {bg['rmse_x_orig']:.1f}", end="")
    print()

# save JSON
serializable = {}
for name, d in results.items():
    serializable[name] = {
        "backbone": d["backbone"], "mode": d["mode"], "n_params": d["n_params"],
        "rmse_x_norm": d["rmse_x_norm"], "rmse_x_orig": d["rmse_x_orig"],
        "by_group": {},
    }
    for gk, gv in d["by_group"].items():
        gks = str(gk + 1) if isinstance(gk, int) else gk
        serializable[name]["by_group"][gks] = {
            k: v for k, v in gv.items() if k in ("n", "rmse_x_norm", "rmse_x_orig")
        }
with open(os.path.join(OUT, "compare_metrics.json"), "w") as f:
    json.dump(serializable, f, indent=2)

# ========== charts ==========
def plot_bars(results, model_order, model_labels, colors, title, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for gi, g_str in enumerate(["1", "2", "3", "4", "5"]):
        ax = axes[gi]
        values = []
        for m_name in model_order:
            bg = results[m_name]["by_group"]
            g_int = int(g_str) - 1
            g_data = bg.get(g_int, bg.get(g_str, {}))
            v = g_data.get("rmse_x_orig", 0) if isinstance(g_data, dict) else 0
            values.append(v if v else 0)
        x_pos = np.arange(len(model_order))
        bars = ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Group {g_str}", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=6.5)
        ax.set_ylabel("RMSE (original)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图表: {out_path}")

model_labels = {
    "LSTM shared": "LSTM shared", "LSTM group_head": "LSTM group_head",
    "LSTM independent": "LSTM independent",
    "PathInt shared": "PathInt shared", "PathInt group_head": "PathInt group_head",
}

# Chart 1: LSTM 三种模式
plot_bars(results,
          ["LSTM shared", "LSTM group_head", "LSTM independent"],
          model_labels,
          ["#78909c", "#42a5f5", "#ef5350"],
          "LSTM Backbone — Per-Group RMSE (10 epochs)",
          os.path.join(OUT, "compare_bars_lstm.png"))

# Chart 2: PathInt 两种模式
plot_bars(results,
          ["PathInt shared", "PathInt group_head"],
          model_labels,
          ["#78909c", "#66bb6a"],
          "PathInt Backbone — Per-Group RMSE (10 epochs)",
          os.path.join(OUT, "compare_bars_pathint.png"))

# Chart 3: group_head head-to-head (LSTM vs PathInt)
plot_bars(results,
          ["LSTM shared", "LSTM group_head", "PathInt shared", "PathInt group_head"],
          model_labels,
          ["#90a4ae", "#42a5f5", "#a5d6a7", "#66bb6a"],
          "Shared vs Group Head — LSTM vs PathInt (10 epochs)",
          os.path.join(OUT, "compare_bars_all.png"))

# ========== summary table ==========
print(f"\n{'='*80}")
print(f"{'Model':<22} {'Params':>10} {'RMSE(orig)':>12}  {'G1':>8} {'G2':>8} {'G3':>8} {'G4':>8} {'G5':>8}")
print("-"*80)
order = ["LSTM shared", "LSTM group_head", "LSTM independent",
         "PathInt shared", "PathInt group_head"]
for m_name in order:
    d = results.get(m_name)
    if not d: continue
    print(f"{m_name:<22} {d['n_params']:>10,} {d['rmse_x_orig']:>12.1f}  ", end="")
    for g_str in ["1","2","3","4","5"]:
        g_int = int(g_str)-1
        bg = d["by_group"].get(g_int, d["by_group"].get(g_str, {}))
        v = bg.get("rmse_x_orig") if isinstance(bg, dict) else None
        print(f"{v:>8.1f}" if v else f"{'N/A':>8}", end=" ")
    print()
print(f"{'='*80}")

# analysis
print("\n=== group_head 相对 shared 的提升 ===")
for bb in ["LSTM", "PathInt"]:
    gh = results.get(f"{bb} group_head")
    sh = results.get(f"{bb} shared")
    if gh and sh:
        imp = (sh["rmse_x_orig"] - gh["rmse_x_orig"]) / sh["rmse_x_orig"] * 100
        print(f"  {bb}: {sh['rmse_x_orig']:.1f} -> {gh['rmse_x_orig']:.1f}  ({imp:+.1f}%)")

print("\n=== LSTM vs PathInt 同模式对比 ===")
for mode in ["shared", "group_head"]:
    l = results.get(f"LSTM {mode}")
    p = results.get(f"PathInt {mode}")
    if l and p:
        diff = (p["rmse_x_orig"] - l["rmse_x_orig"]) / l["rmse_x_orig"] * 100
        print(f"  {mode}: LSTM={l['rmse_x_orig']:.1f}  PathInt={p['rmse_x_orig']:.1f}  ({diff:+.1f}%)")

print("\nDone.")
