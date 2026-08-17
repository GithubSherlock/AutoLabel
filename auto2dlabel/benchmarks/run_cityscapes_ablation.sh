#!/usr/bin/env bash
# ============================================================
# run_cityscapes_ablation.sh — cityscapes 分割消融矩阵一键运行
#
# 目标：二分定位 mask mAP 根因（检测步 vs 分割步）。
# 自动探测 GPU/CPU 选矩阵，串行执行并打印对照摘要。
#
# 用法（任意工作目录可运行，脚本自动切回仓库根）:
#   bash run_cityscapes_ablation.sh                # 自动选矩阵（50 图）
#   bash run_cityscapes_ablation.sh --max-images 20
#
# 矩阵解读（配合 docs/ 与 test-v0.3.md「cityscapes 消融」节）:
#   det-only       隔离检测器（bbox mAP，GT mask 派生 bbox）
#   box-prompted   隔离分割器（GT bbox 直接 prompt，检测零误差）
#   full           组合效应（det-only + box-prompted 对照）
# ============================================================
set -euo pipefail

# ── CWD 自定位：脚本在 auto2dlabel/benchmarks/ 下，仓库根为其上两级 ──
cd "$(dirname "$0")/../.."

MAX_IMAGES=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-images) [[ -z "${2:-}" ]] && { echo "❌ --max-images 缺少参数"; exit 1; }
                      MAX_IMAGES="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

BENCH="python auto2dlabel/benchmarks/cityscapes_benchmark.py"
ARGS="--max-images ${MAX_IMAGES}"

if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "== GPU 模式：全精度消融矩阵（yolo26x 检测侧 + sam3 强分割器） =="
    RUNS=(
        "det-only yolo26x conf0.3  | $BENCH --det-only --model yolo26x.pt $ARGS --conf 0.3"
        "det-only yolo26x conf0.5  | $BENCH --det-only --model yolo26x.pt $ARGS --conf 0.5"
        "box-prompted FastSAM-s     | $BENCH --box-prompted --seg-model FastSAM-s.pt $ARGS"
        "box-prompted sam2_l        | $BENCH --box-prompted --seg-model sam2_l.pt $ARGS"
        "box-prompted sam3          | $BENCH --box-prompted --seg-model sam3.pt $ARGS"
        "full sam3 (det yolo26x)    | $BENCH --seg-model sam3.pt --model yolo26x.pt $ARGS"
    )
else
    echo "== CPU 模式：轻量消融矩阵（yolo11n 覆盖检测侧） =="
    RUNS=(
        "det-only yolo11n conf0.3  | $BENCH --det-only --model yolo11n.pt $ARGS --conf 0.3"
        "det-only yolo11n conf0.5  | $BENCH --det-only --model yolo11n.pt $ARGS --conf 0.5"
        "box-prompted FastSAM-s     | $BENCH --box-prompted --seg-model FastSAM-s.pt $ARGS"
        "box-prompted sam2_l        | $BENCH --box-prompted --seg-model sam2_l.pt $ARGS"
        "full FastSAM prompt-conf 0 | $BENCH --seg-model FastSAM-s.pt --model yolo11n.pt --prompt-conf 0.0 $ARGS"
    )
fi

# 提取最近一次 cityscapes 结果的 mAP（JSON 为唯一权威）
latest_mAP() {
    local js
    js="$(ls -t benchmarks_outputs/cityscapes_*.json 2>/dev/null | head -1)"
    [[ -z "$js" ]] && { echo "?"; return; }
    python3 -c "import json,sys; s=json.load(open('$js')).get('summary',{}); print(s.get('mAP@0.5', s.get('mAP', '?')))" 2>/dev/null || echo "?"
}

LABELS=(); CMDS=()
while IFS= read -r entry; do
    LABELS+=("${entry%%|*}")
    CMDS+=("${entry#*|}")
done <<< "$(printf '%s\n' "${RUNS[@]}")"

SUMMARY=""
for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    cmd="${CMDS[$i]}"
    echo ""
    echo "== [$((i+1))/${#LABELS[@]}] ${label} =="
    echo "  \$ ${cmd}"
    out="$($cmd 2>&1)" && rc=0 || rc=$?
    if [[ $rc -ne 0 ]]; then
        SUMMARY+="❌ ${label}: 退出码 $rc\n"
        echo "$out" | tail -20
        continue
    fi
    map_val="$(latest_mAP)"
    SUMMARY+="✅ ${label}: mAP@0.5 = ${map_val}\n"
    echo "$out" | grep -E "完成！|mAP@" | tail -3
done

echo ""
echo "===== 消融对照（详见 benchmarks_outputs/cityscapes_*.json） ====="
printf "$SUMMARY"
