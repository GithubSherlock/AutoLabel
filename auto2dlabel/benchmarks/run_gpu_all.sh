#!/usr/bin/env bash
# ============================================================
# run_gpu_all.sh — GPU 服务器一键全跑总入口
#
# 按序串行执行（单步失败不阻断后续，最后汇总退出码）:
#   1. 环境自检（nvidia-smi / 数据集目录 / 权重）
#   2. 全量 11 数据集（run_benchmarks.sh all，默认 yolo26x.pt）
#   3. OBB 458 图全量（yolo26x-obb.pt）
#   4. cityscapes 消融矩阵（GPU 全精度版：yolo26x 检测侧 + sam3）
#
# 用法（任意工作目录可运行，脚本自动切回仓库根）:
#   bash auto2dlabel/benchmarks/run_gpu_all.sh
#
# 结果全部落在 benchmarks_outputs/（时间戳命名），对照基线见
# auto2dlabel/tests/test-v0.3.md「GPU 复测清单」。
# ============================================================
set -u

cd "$(dirname "$0")/../.."

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 若 GPU 不可用则只跑检查并退出（防误在 CPU 服务器上启动全量）
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null; then
    echo -e "${RED}❌ 未检测到 NVIDIA GPU——本脚本仅用于 GPU 服务器。${NC}"
    echo -e "  CPU 服务器请使用: bash auto2dlabel/benchmarks/run_benchmarks.sh --extra \"--model yolo11n.pt\""
    exit 1
fi

declare -a LABELS=() CMDS=() RCS=()

LABELS+=("环境自检");          CMDS+=("bash auto2dlabel/benchmarks/run_benchmarks.sh --check-only")
LABELS+=("全量 11 数据集");    CMDS+=("bash auto2dlabel/benchmarks/run_benchmarks.sh")
LABELS+=("OBB 458 图全量");    CMDS+=("bash auto2dlabel/benchmarks/run_benchmarks.sh obb --extra \"--model yolo26x-obb.pt --max-images 0\"")
LABELS+=("cityscapes 消融");   CMDS+=("bash auto2dlabel/benchmarks/run_cityscapes_ablation.sh")

for i in "${!LABELS[@]}"; do
    echo ""
    echo -e "${BOLD}== [$((i+1))/${#LABELS[@]}] ${LABELS[$i]} ==${NC}"
    eval "${CMDS[$i]}"
    RCS+=($?)
    [[ ${RCS[$i]} -eq 0 ]] \
        && echo -e "${GREEN}✓ ${LABELS[$i]} 完成${NC}" \
        || echo -e "${RED}✗ ${LABELS[$i]} 失败（退出码 ${RCS[$i]}，继续下一步）${NC}"
done

echo ""
echo -e "${BOLD}===== 汇总 ====${NC}"
ok=0
for i in "${!LABELS[@]}"; do
    if [[ ${RCS[$i]} -eq 0 ]]; then
        echo -e "  ${GREEN}✓${NC} ${LABELS[$i]}"
        ok=$((ok+1))
    else
        echo -e "  ${RED}✗${NC} ${LABELS[$i]}（退出码 ${RCS[$i]}）"
    fi
done
echo -e "${BOLD}${ok}/${#LABELS[@]} 步成功。产物: benchmarks_outputs/（对照 test-v0.3.md GPU 复测清单）${NC}"
exit $(( ${#LABELS[@]} - ok ))
