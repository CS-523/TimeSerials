#!/usr/bin/env bash
# =============================================================================
# 运行 src/train_y_poly.py 的各模式，并汇总对比 RMSE / MAE / R²
#
# 用法:
#   bash run_y_poly_compare.sh
#
# 输出:
#   ./src/model_out_compare/   各模式产物 (pkl / metrics.json / predictions.csv / png)
#   终端末尾打印 Markdown 对比表（按 RMSE 升序，越好越靠前）+ 每组 R² 细分表
#
# 说明:
#   - 每次运行前会清空 ./src/model_out_compare，只保留本次结果，方便对比
#     （不影响 src/model_out 里的历史产物）
#   - 想增减对比项，编辑下方 RUNS 数组即可，格式: "标签   --参数..."
#   - 可从任意目录调用，脚本会自动 cd 到自身所在目录
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="${BASE_DIR:-.}"                       # 数据根目录（含 1/..5/ 子目录）
OUT_DIR="${OUT_DIR:-./src/model_out_compare}"   # 本次对比的独立输出目录（运行前清空）
PYTHON="${PYTHON:-python}"
SCRIPT="src/train_y_poly.py"

# ---- 对比组合（标签仅用于日志；汇总表按输出文件名自动解析）----
RUNS=(
  "last_deg2       --degree 2 --mode last"
  "last_deg3       --degree 3 --mode last"
  "last_deg2_noY4  --degree 2 --mode last --drop-y4"
  "window4_deg2    --degree 2 --mode window --window 4"
  "window8_deg2    --degree 2 --mode window --window 8"
  "window8_deg3    --degree 3 --mode window --window 8"
  "pergroup_deg2   --degree 2 --per-group"
  "pergroup_deg3   --degree 3 --per-group"
)

echo "===== 清空输出目录: $OUT_DIR ====="
rm -rf "$OUT_DIR"

for entry in "${RUNS[@]}"; do
  label="${entry%%  *}"
  params="${entry#*  }"
  echo ""
  echo "########## [$label]  $PYTHON $SCRIPT $params ##########"
  "$PYTHON" "$SCRIPT" --base-dir "$BASE_DIR" --out-dir "$OUT_DIR" $params
done

# ---- 汇总对比表 ----
echo ""
echo "==================== 效果对比 ===================="
"$PYTHON" - "$OUT_DIR" <<'PY'
import json, glob, os, re, sys

out_dir = sys.argv[1]

def parse(name):
    """y_poly_deg2_y1234_last_metrics.json -> (degree:int, mode:str)"""
    tag = name[len("y_poly_"):-len("_metrics.json")]
    m = re.match(r"deg(\d+)_(.+)", tag)
    degree = int(m.group(1))
    rest = m.group(2)
    drop_y4 = rest.startswith("y123_")          # y1234_last vs y123_last
    if rest == "engineered_pergroup":
        mode = "per-group"
    elif "win" in rest:
        mode = "window n=" + re.search(r"win(\d+)", rest).group(1)
    elif "last" in rest:
        mode = "last"
    else:
        mode = rest
    if drop_y4:
        mode += " (drop-y4)"
    return degree, mode

rows = []
files = sorted(glob.glob(os.path.join(out_dir, "y_poly_*_metrics.json")))
for f in files:
    with open(f) as fh:
        m = json.load(fh)
    degree, mode = parse(os.path.basename(f))
    rows.append((degree, mode, m["rmse"], m["mae"], m["r2"], m))

rows.sort(key=lambda r: r[2])                   # 按 RMSE 升序

print(f"{'degree':<7}{'mode':<20}{'RMSE':>8}{'MAE':>8}{'R2':>9}")
print("-" * 52)
for degree, mode, rmse, mae, r2, _ in rows:
    print(f"{degree:<7}{mode:<20}{rmse:>8.1f}{mae:>8.1f}{r2:>9.4f}")

# 每组 R² 细分
per_groups = {}
for _, mode, _, _, _, m in rows:
    pg = m.get("per_group")
    if pg:
        per_groups[mode] = pg

if per_groups:
    groups = sorted({k for pg in per_groups.values() for k in pg},
                    key=lambda s: int(s.split("_")[1]))
    print("\n每组 R²（行=模式，列=组）:")
    header = f"{'mode':<24}" + "".join(f"{'G' + g.split('_')[1]:>8}" for g in groups)
    print(header)
    print("-" * (24 + 8 * len(groups)))
    for mode, pg in per_groups.items():
        cells = "".join(f"{pg[g]['r2']:>8.3f}" if g in pg and 'r2' in pg[g] else f"{'':>8}"
                        for g in groups)
        print(f"{mode:<24}{cells}")
PY
