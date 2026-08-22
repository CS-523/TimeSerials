"""
数据加载与预处理模块
====================
负责：
1. 读取 5 个目录的所有 CSV 文件
2. 合并/对齐 x1-x8（整点行）和 y1-y4（"15秒"行）
3. 清洗异常值、缺失值
4. 输出统一格式的 (time, x1..x8, y1..y4, Y, 周期) 时序表
5. 提供按"周期"对齐的滑动窗口（输入长度可变、起点可变、输出长度可变）

数据观察：
- 每个 CSV 对应一个实验 (Y 单值)
- "整点"行 (如 09:00:00, 09:30:00)：包含 x1-x8，缺失 x 行可能为偶发
- "15秒"行 (如 09:45:12.530)：通常 x 全空，y1-y4 数据出现于此
- y1-y4 缺失 = 该周期未测量
- 周期索引从 0 开始，单调递增
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------- 列定义 ----------
X_COLS = [f"x{i}" for i in range(1, 9)]                # 8 个过程变量
Y_INT_COLS = ["y1", "y2", "y3", "y4"]                  # 中间目标
META_COLS = ["datime", "周期", "Y"]
ALL_COLS = X_COLS + Y_INT_COLS + ["Y"]                 # 用于读 csv + 对齐


# ===================== 1. 单文件读取 =====================
def _read_one_csv(path: str) -> pd.DataFrame:
    """读一个 csv，去掉完全空白的尾行（datime=NaT）。"""
    df = pd.read_csv(path, dtype=str)            # 先全读为字符串，避免 NA 自动转
    # 丢掉 datime='NaT' 这种仅含 Y 的尾行
    if len(df) > 0:
        last = df.iloc[-1]
        if pd.isna(last.get("datime", None)) or str(last.get("datime", "")).strip() in ("", "NaT", "nan"):
            df = df.iloc[:-1].copy()
    # 数值化
    for c in X_COLS + Y_INT_COLS + ["周期", "Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _extract_Y(path: str) -> Optional[float]:
    """从 csv 末尾或任何一行的 Y 列抽出实验终值。"""
    df = pd.read_csv(path, dtype=str)
    y_vals = pd.to_numeric(df["Y"], errors="coerce").dropna()
    if len(y_vals) == 0:
        return None
    return float(y_vals.iloc[0])


# ===================== 2. 周期对齐：x 行作为主键 =====================
def _align_to_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    按"周期"对齐：对每个周期保留 x 行；如果该周期没 x 行但有 y，则 y 来自'15秒行'。
    由于 y 行携带的 datime 是 09:45:12.530 这种，y 应归属到"周期"对应的同一周期号。
    每个周期最多保留 1 行（取非空字段更多的那行）。
    """
    # 缺失 datime 当作未知，但周期一定有
    df = df.copy()
    # 把 y1-y4 数值迁移到对应的 x 行（按 周期 配对）
    # 先建立一个字典：周期 -> {y1,y2,y3,y4}
    y_rows = df[df[X_COLS].isna().all(axis=1) & df[Y_INT_COLS].notna().any(axis=1)]
    y_map = y_rows.groupby("周期")[["y1", "y2", "y3", "y4"]].first()

    # 取 x 行（任意 x 不空）
    x_rows = df[df[X_COLS].notna().any(axis=1)].copy()
    # 合并 y
    x_rows = x_rows.merge(y_map, left_on="周期", right_index=True, how="left",
                          suffixes=("", "_dup"))
    # 去重列
    for c in Y_INT_COLS:
        dup = f"{c}_dup"
        if dup in x_rows.columns:
            x_rows[c] = x_rows[c].fillna(x_rows[dup])
            x_rows.drop(columns=[dup], inplace=True)

    # 一些文件可能完全没有 x 行；为安全起见，补充 y-only 行
    if len(x_rows) == 0 and len(y_rows) > 0:
        x_rows = y_rows.reset_index()

    # 排序 + 重置索引
    x_rows = x_rows.sort_values("周期").reset_index(drop=True)
    return x_rows


# ===================== 3. 异常清洗（z-score 标记 + 线性插值） =====================
# 旧版用 IQR×3 clip，仅"压低幅度"，形状仍是尖刺；LSTM 仍要拟合 0.20→0.35→0.19
# 这种 75% 跳变，N4SID 的 Hankel 矩阵被"接近边界"行主导。
# 新版用 |z|>5 标记为 NaN + 线性插值，把尖刺"完全替换"为邻居均值，符合
# "工业过程连续平滑"物理假设。N4SID 才能正确识别低秩动态。
Z_THRESHOLD = 5.0  # 与任务 2 一致（src_control/preprocess.py:48）


def detect_anomalies_x(df: pd.DataFrame, z_threshold: float = Z_THRESHOLD,
                       max_iter: int = 3) -> pd.DataFrame:
    """对每列 x 用 |z|>z_threshold 标记异常（z 由该列有效值 mean/std 算），返回 (T,8) bool。

    使用 **迭代 trim**（最多 max_iter 轮）：每一轮重新计算 mean/std
    时排除已标记的异常位，避免极端值（如 9999）主导 mean 让自身
    |z| < threshold 而漏检。任务 2 的单遍实现对极短序列会漏检，
    此处兼容之。
    """
    out = pd.DataFrame(False, index=df.index, columns=X_COLS)
    for c in X_COLS:
        col = df[c]
        valid = col.notna().to_numpy()
        if valid.sum() < 2:
            continue
        v = col.to_numpy(dtype=np.float64).copy()
        idx_valid = np.where(valid)[0]
        outlier = np.zeros_like(valid)
        for _ in range(max_iter):
            mu = v[idx_valid].mean()
            sd = v[idx_valid].std() + 1e-6
            z = np.zeros_like(v)
            z[idx_valid] = (v[idx_valid] - mu) / sd
            new_outlier = (np.abs(z) > z_threshold) & valid
            if new_outlier.sum() == outlier.sum() and np.array_equal(new_outlier, outlier):
                break
            outlier = new_outlier
            idx_valid = np.where(valid & ~outlier)[0]
            if len(idx_valid) < 2:
                break
        out[c] = outlier
    return out


def _clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗策略（与任务 2 保持一致）：
    - x 异常值：|z|>5 标记为 NaN，再用 np.interp 在 valid 索引间线性内插
    - y 缺失：保留为 NaN（训练时用掩码）
    """
    df = df.copy()
    # 1. 标记 x 异常：用 numpy 修改列（避免 .loc[..., col] / .to_numpy() 在某些
    #    pandas 版本下触发 "assignment destination is read-only"）。
    mask = detect_anomalies_x(df)
    for c in X_COLS:
        col = df[c].to_numpy(dtype=np.float64).copy()
        col[mask[c].to_numpy(dtype=bool)] = np.nan
        df[c] = col
    # 2. 线性插值（按周期顺序，valid 索引间）
    for c in X_COLS:
        col = df[c].to_numpy(dtype=np.float64).copy()
        valid = ~np.isnan(col)
        if not valid.any():
            df[c] = 0.0
            continue
        if valid.all():
            continue
        idx = np.arange(len(col))
        col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
        df[c] = col
    return df


# ===================== 4. 多文件加载 =====================
@dataclass
class Experiment:
    file: str
    group: str                      # 目录名 "1".."5"
    df: pd.DataFrame                # 已对齐清洗后的时序表
    Y: Optional[float] = None       # 终值目标

    def __len__(self) -> int:
        return len(self.df)

    @property
    def length(self) -> int:
        return len(self.df)


def load_all(base_dir: str = ".") -> List[Experiment]:
    """加载 5 个目录下的所有 csv，组成 Experiment 列表。"""
    exps: List[Experiment] = []
    for g in ["1", "2", "3", "4", "5"]:
        files = sorted(glob.glob(os.path.join(base_dir, g, "*.csv")))
        for f in files:
            try:
                raw = _read_one_csv(f)
                aligned = _align_to_period(raw)
                cleaned = _clean_anomalies(aligned)
                # Y：从原 csv 的 Y 列抽取
                Y = _extract_Y(f)
                exps.append(Experiment(file=f, group=g, df=cleaned, Y=Y))
            except Exception as e:
                print(f"[load_all] 跳过 {f}: {e}")
    return exps


# ===================== 5. 滑窗采样（支持变起点 / 变长度） =====================
@dataclass
class WindowSample:
    exp_id: int                      # Experiment 索引
    start: int                       # 输入起点（周期下标）
    in_len: int                      # 输入窗口长度
    out_len: int                     # 输出窗口长度
    x_in: np.ndarray                 # (in_len, 8)
    y_in: np.ndarray                 # (in_len, 4)  缺失位 NaN
    x_out: np.ndarray                # (out_len, 8)
    y_out: np.ndarray                # (out_len, 4)
    Y: Optional[float]               # 实验终值


def sample_windows(
    exps: List[Experiment],
    in_lens: Tuple[int, ...] = (16, 20, 24, 32),
    out_lens: Tuple[int, ...] = (6, 8, 12, 16),
    min_total: int = 24,             # 实验总长至少这么多周期才参与采样
    rng_seed: int = 42,
) -> List[WindowSample]:
    """
    从所有实验中随机采样 (起点, in_len, out_len) 三元组，构造训练样本。
    起点可变、输入长度可变、输出长度可变。
    """
    rng = np.random.default_rng(rng_seed)
    samples: List[WindowSample] = []
    for ei, exp in enumerate(exps):
        n = len(exp)
        if n < min_total:
            continue
        # 每条实验采 ~10 个窗口
        n_samples = max(1, n // 30)
        for _ in range(n_samples):
            in_len = int(rng.choice(in_lens))
            out_len = int(rng.choice(out_lens))
            max_start = n - (in_len + out_len)
            if max_start < 0:
                continue
            start = int(rng.integers(0, max_start + 1))
            x_in = exp.df[X_COLS].iloc[start:start + in_len].to_numpy(dtype=np.float32)
            y_in = exp.df[Y_INT_COLS].iloc[start:start + in_len].to_numpy(dtype=np.float32)
            x_out = exp.df[X_COLS].iloc[start + in_len:start + in_len + out_len].to_numpy(dtype=np.float32)
            y_out = exp.df[Y_INT_COLS].iloc[start + in_len:start + in_len + out_len].to_numpy(dtype=np.float32)
            samples.append(WindowSample(
                exp_id=ei, start=start, in_len=in_len, out_len=out_len,
                x_in=x_in, y_in=y_in, x_out=x_out, y_out=y_out, Y=exp.Y,
            ))
    return samples


# ===================== 6. 全局标准化 =====================
@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, exps: List[Experiment]) -> "Scaler":
        arrs = []
        for e in exps:
            arrs.append(e.df[X_COLS].to_numpy())
        if not arrs:
            raise ValueError("Cannot fit Scaler: no experiments provided (train_exps is empty)")
        all_x = np.concatenate(arrs, axis=0)
        return cls(mean=all_x.mean(0).astype(np.float32), std=(all_x.std(0) + 1e-6).astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.std + self.mean


@dataclass
class YScaler:
    """y1-y4 量级差异大，每个目标单独标准化。"""
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def fit(cls, exps: List[Experiment]) -> "YScaler":
        arrs = []
        for e in exps:
            arrs.append(e.df[Y_INT_COLS].to_numpy())
        if not arrs:
            raise ValueError("Cannot fit YScaler: no experiments provided (train_exps is empty)")
        all_y = np.concatenate(arrs, axis=0)
        return cls(means=np.nanmean(all_y, axis=0).astype(np.float32),
                   stds=(np.nanstd(all_y, axis=0) + 1e-6).astype(np.float32))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.means) / self.stds

    def inverse(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.stds + self.means


# ===================== 7. 一站式：训练 / 验证 / 测试 切分 =====================
def split_experiments(exps: List[Experiment], ratios=(0.7, 0.1, 0.2), seed: int = 42) -> Tuple[List[Experiment], List[Experiment], List[Experiment]]:
    """按实验 id 随机切分 train/val/test，避免数据泄露。"""
    rng = np.random.default_rng(seed)
    n = len(exps)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = [exps[i] for i in idx[:n_train]]
    val = [exps[i] for i in idx[n_train:n_train + n_val]]
    test = [exps[i] for i in idx[n_train + n_val:]]
    return train, val, test


if __name__ == "__main__":
    base = "/kefu-nas/ybkong/time_serials-master"
    exps = load_all(base)
    print(f"实验总数: {len(exps)}")
    for e in exps[:3]:
        print(f"\n{e.file} (group={e.group}, len={e.length}, Y={e.Y})")
        print(e.df.head(3))
    sc = Scaler.fit(exps)
    ysc = YScaler.fit(exps)
    print(f"\nX scaler mean={sc.mean}\nX scaler std ={sc.std}")
    print(f"Y scaler mean={ysc.means}\nY scaler std ={ysc.stds}")
    samples = sample_windows(exps, in_lens=(20,), out_lens=(8,), rng_seed=42)
    print(f"\n样本数: {len(samples)}")
    print("首个样本:")
    s = samples[0]
    print(f"  in_len={s.in_len}, out_len={s.out_len}, start={s.start}")
    print(f"  x_in.shape={s.x_in.shape}, y_out.shape={s.y_out.shape}")
    print(f"  x_in[:2]={s.x_in[:2]}")
    print(f"  y_out={s.y_out}")