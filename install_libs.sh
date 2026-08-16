#!/bin/bash
# AutoLabel 依赖安装脚本
# 用法: bash install_libs.sh [2d|3d|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_2D="$SCRIPT_DIR/auto2dlabel/requirements.txt"
REQ_3D="$SCRIPT_DIR/auto3dlabel/requirements.txt"

echo "========================================="
echo "  AutoLabel 依赖安装"
echo "========================================="
echo "  1) auto2dlabel  (2D 图像标注)"
echo "  2) auto3dlabel  (3D 点云标注，待实现)"
echo "  3) 全部安装"
echo "========================================="

# 如果有命令行参数，直接使用；否则交互选择
CHOICE="${1:-}"

if [[ -z "$CHOICE" ]]; then
    read -r -p "请选择 [1/2/3]: " CHOICE
fi

install_2d() {
    echo ""
    echo ">>> 安装 auto2dlabel 依赖..."
    if [[ -f "$REQ_2D" ]]; then
        pip install -r "$REQ_2D"
        echo ">>> auto2dlabel 依赖安装完成 ✓"
    else
        echo "!!! 错误: 未找到 $REQ_2D"
        return 1
    fi
}

install_3d() {
    echo ""
    echo ">>> 安装 auto3dlabel 依赖..."
    if [[ -f "$REQ_3D" ]]; then
        pip install -r "$REQ_3D"
        echo ">>> auto3dlabel 依赖安装完成 ✓"
    else
        echo "!!! 错误: 未找到 $REQ_3D"
        return 1
    fi
}

case "$CHOICE" in
    1|2d)
        install_2d
        ;;
    2|3d)
        install_3d
        ;;
    3|all|both)
        install_2d && install_3d
        ;;
    *)
        echo "无效选项: $CHOICE (请选 1/2/3 或 2d/3d/all)"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "  安装完成"
echo "========================================="
echo "  可用检测模型:"
echo "    python3 -c 'from auto2dlabel.models.model_catalog import ALL_DETECTION_MODELS; print(ALL_DETECTION_MODELS)'"
echo "  验证:"
echo "    auto2dlabel run --help"
