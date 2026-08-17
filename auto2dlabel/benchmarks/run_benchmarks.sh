#!/usr/bin/env bash
# ============================================================
# run_benchmarks.sh — Benchmark 一键运行包装（纯 Bash）
#
# 职责边界：只做 环境检查 + 分组 + 透传。
# 子进程调度 / 超时 / 汇总 / 退出码统计全部归 run_all.py。
#
# 用法（任意工作目录可运行，脚本自动切回仓库根）:
#   bash run_benchmarks.sh                        # 全部 11 数据集
#   bash run_benchmarks.sh detection              # 仅检测组
#   bash run_benchmarks.sh segmentation           # 仅分割组
#   bash run_benchmarks.sh classification         # 仅分类组
#   bash run_benchmarks.sh obb                    # 仅旋转框组
#   bash run_benchmarks.sh --only voc2007,kitti   # 指定数据集（覆盖组）
#   bash run_benchmarks.sh detection --extra "--conf 0.1"
#   bash run_benchmarks.sh --check-only           # 只检查，不实跑
#
# 注意（仅 CPU 服务器）：检测/分割默认模型 yolo26x.pt 在 CPU 上极慢，
# 全组运行请加 --extra "--model yolo11n.pt" 覆盖。
# ============================================================
set -euo pipefail

# ── CWD 自定位：脚本在 auto2dlabel/benchmarks/ 下，仓库根为其上两级 ──
cd "$(dirname "$0")/../.."

# ── 颜色 ──
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ── 分组（key 与 run_all.py BENCHMARKS 一致） ──
GROUP_DETECTION="coco2017,voc2007,kitti,dota,mot"
GROUP_SEGMENTATION="coco_seg,cityscapes,nuimages,d2sa"
GROUP_CLASSIFICATION="imagenet100"
GROUP_OBB="dota_obb"

usage() {
    cat <<EOF
${BOLD}用法:${NC} bash run_benchmarks.sh [all|detection|segmentation|classification|obb] [--only DATASET,...] [--extra "ARGS"] [--check-only]

${BOLD}分组:${NC}
  all            全部 11 数据集（默认）
  detection      ${GROUP_DETECTION//,/, }
  segmentation   ${GROUP_SEGMENTATION//,/, }
  classification ${GROUP_CLASSIFICATION//,/, }
  obb            ${GROUP_OBB//,/, }

${BOLD}选项:${NC}
  --only DATASET,...   仅跑指定数据集（逗号分隔，覆盖分组选择）
  --extra "ARGS"       透传给 run_all 的额外 CLI 参数（如 "--conf 0.1"）
  --check-only         只做环境与数据集检查，不实跑
  -h, --help           显示本帮助
EOF
}

# ── 参数解析 ──
GROUP="all"
ONLY=""
EXTRA=""
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --only)
            [[ -z "${2:-}" ]] && { echo -e "${RED}❌ --only 缺少参数${NC}"; exit 1; }
            ONLY="$2"; shift 2 ;;
        --extra)
            [[ -z "${2:-}" ]] && { echo -e "${RED}❌ --extra 缺少参数${NC}"; exit 1; }
            EXTRA="$2"; shift 2 ;;
        all|detection|segmentation|classification|obb) GROUP="$1"; shift ;;
        *) echo -e "${RED}❌ 未知参数: $1${NC}"; usage; exit 1 ;;
    esac
done

# ── 环境检查 ──
echo -e "${BOLD}== 环境检查 ==${NC}"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌ 未找到 python3${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} python3: $(python3 --version 2>&1)"

if ! python3 -c "import auto2dlabel" &>/dev/null; then
    echo -e "${RED}❌ auto2dlabel 无法导入，请安装依赖或检查仓库结构${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} auto2dlabel 可导入"

if command -v nvidia-smi &>/dev/null; then
    gpu_line="$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo -e "  ${GREEN}✓${NC} GPU: ${gpu_line:-检测到 nvidia-smi 但读取失败}"
else
    echo -e "  ${YELLOW}⚠${NC} 未找到 nvidia-smi，将用 CPU 推理（较慢，不阻断）"
fi

# ── 数据集目录探测（warn 不阻断：ensure_* 会按需解压） ──
DATASETS_ROOT="$(python3 -c "from auto2dlabel.benchmarks import datasets; print(datasets.DATASETS_ROOT)" 2>/dev/null || echo "")"
if [[ -n "$DATASETS_ROOT" ]]; then
    echo -e "  数据集根目录: ${CYAN}${DATASETS_ROOT}${NC}"
    missing=()
    for d in COCO2017 VOCdevkit KITTI cityscapes nuImages DOTA D2SA MOT17 MOT20 imagenet100; do
        [[ -e "${DATASETS_ROOT}/${d}" ]] || missing+=("$d")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠${NC} 未就绪的数据集目录: ${missing[*]}（首次运行 benchmark 时 ensure_* 会自动从归档解压；MOT 需预先解压）"
    else
        echo -e "  ${GREEN}✓${NC} 10 个数据集目录均已就绪"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} 无法读取 DATASETS_ROOT，跳过数据集目录探测"
fi

[[ "$CHECK_ONLY" == "1" ]] && { echo -e "\n${GREEN}检查完成（--check-only，未运行 benchmark）。${NC}"; exit 0; }

# ── 分组 → --only 列表 ──
if [[ -n "$ONLY" ]]; then
    DATASETS="$ONLY"
elif [[ "$GROUP" == "detection" ]]; then
    DATASETS="$GROUP_DETECTION"
elif [[ "$GROUP" == "segmentation" ]]; then
    DATASETS="$GROUP_SEGMENTATION"
elif [[ "$GROUP" == "classification" ]]; then
    DATASETS="$GROUP_CLASSIFICATION"
elif [[ "$GROUP" == "obb" ]]; then
    DATASETS="$GROUP_OBB"
else
    DATASETS="${GROUP_DETECTION},${GROUP_SEGMENTATION},${GROUP_CLASSIFICATION},${GROUP_OBB}"
fi

# ── 执行（调度归 run_all.py） ──
echo ""
echo -e "${BOLD}== 运行 Benchmark ==${NC}"
echo -e "  数据集: ${CYAN}${DATASETS}${NC}"
[[ -n "$EXTRA" ]] && echo -e "  额外参数: ${CYAN}${EXTRA}${NC}"

python3 -m auto2dlabel.benchmarks.run_all --only "$DATASETS" --extra "${EXTRA:-}"
exit $?
