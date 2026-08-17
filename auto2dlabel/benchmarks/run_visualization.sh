#!/usr/bin/env bash
# ============================================================
# run_visualization.sh — 11 数据集全量可视化一键总入口
#
# 按数据集对应任务全量推理（检测 bbox / 分割 mask / 分类 top-K /
# OBB 旋转框），渲染结果保存到项目同级的 Visualization/<数据集名>/
# （镜像原数据集相对路径）。评测 JSON/MD 照常写 benchmarks_outputs/。
#
# 串行执行（GPU 峰值约束），单步失败不阻断后续；已渲染产物自动跳过，
# 可断点续跑。预估全量 7-9GB 产物 / 1.5-2h，磁盘预留 ≥10GB。
#
# 用法（任意工作目录可运行，脚本自动切回仓库根）:
#   bash auto2dlabel/benchmarks/run_visualization.sh
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
    exit 1
fi

declare -a LABELS=() CMDS=() RCS=()

# 目标检测 5 集（yolo26x.pt 通用域）
LABELS+=("coco2017 检测");   CMDS+=("python3 -m auto2dlabel.benchmarks.coco_benchmark --viz --max-images 0 --conf 0.3 --model yolo26x.pt")
LABELS+=("voc2007 检测");    CMDS+=("python3 -m auto2dlabel.benchmarks.voc_benchmark --viz --max-images 0 --conf 0.3 --model yolo26x.pt")
LABELS+=("kitti 检测");      CMDS+=("python3 -m auto2dlabel.benchmarks.kitti_benchmark --viz --max-images 0 --conf 0.3")
LABELS+=("dota 检测");       CMDS+=("python3 -m auto2dlabel.benchmarks.dota_benchmark --viz --max-images 0 --conf 0.3")
LABELS+=("mot 检测");        CMDS+=("python3 -m auto2dlabel.benchmarks.mot_benchmark --viz --max-images 0 --conf 0.3")

# OBB 旋转框（yolo11n-obb.pt）
LABELS+=("dota_obb 旋转框"); CMDS+=("python3 -m auto2dlabel.benchmarks.dota_obb_benchmark --viz --max-images 0 --conf 0.3")

# 实例分割 4 集（sam2_l.pt；cityscapes 用域内 maskrcnn_r50_cityscapes 一步到位）
LABELS+=("coco_seg 分割");   CMDS+=("python3 -m auto2dlabel.benchmarks.coco_seg_benchmark --viz --max-images 0 --conf 0.3 --seg-model sam2_l.pt")
LABELS+=("cityscapes 分割"); CMDS+=("python3 -m auto2dlabel.benchmarks.cityscapes_benchmark --viz --max-images 0 --seg-model maskrcnn_r50_cityscapes")
LABELS+=("nuimages 分割");   CMDS+=("python3 -m auto2dlabel.benchmarks.nuimages_benchmark --viz --max-images 0 --conf 0.3")
LABELS+=("d2sa 分割");       CMDS+=("python3 -m auto2dlabel.benchmarks.d2sa_benchmark --viz --max-images 0 --seg-model sam2_l.pt")

# 图像分类（resnet18 监督 top-K 文本条）
LABELS+=("imagenet100 分类"); CMDS+=("python3 -m auto2dlabel.benchmarks.classification_benchmark --viz --max-images 0 --dataset imagenet100 --per-class 50 --model resnet18")

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
echo -e "${BOLD}${ok}/${#LABELS[@]} 步成功。产物: ../Visualization/<数据集名>/（镜像原相对路径）${NC}"
exit $(( ${#LABELS[@]} - ok ))
