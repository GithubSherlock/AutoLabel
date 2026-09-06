"""route_domain 路由判据单测（docs/Agentic_UI_plan.md §9.3 单一事实源）。

边界用例锁死（企划评审记录 + 实测修正）：
- 「KITTI」不能当 3D 强特征词（kitti_finetune 是 2D 模型）
- 「000860.png」等 2D 图像文件名含 6 位数字，不得误判 3D
- 四名单（DETECTOR3D/MONO3D/BEVFUSION/FCOS3D）命中强制 3D
"""

from __future__ import annotations

import pytest

from autolabel.route import ENGINE3D_NAMES, route_domain


@pytest.mark.parametrize(
    ("instruction", "det_model", "expected"),
    [
        # ---- 2D 默认（宁 2D 勿错）----
        ("检测 000860.png 中的汽车", None, "2d"),  # 图像文件名 6 位数字不误判
        ("检测汽车和行人", None, "2d"),
        ("检测 KITTI 数据集的汽车", None, "2d"),  # kitti 域 2D 不误判（边界）
        ("检测 KITTI 数据集的汽车", "kitti_finetune", "2d"),  # 2D 模型名不误判
        ("检测汽车", "yolo26x.pt", "2d"),
        # ---- 3D 强特征词 ----
        ("标注点云中的汽车", None, "3d"),
        ("处理 lidar 数据", None, "3d"),
        ("velodyne 点云检测", None, "3d"),
        ("检测 3d框 中的车辆", None, "3d"),
        ("3d 检测汽车", None, "3d"),
        ("3d检测 汽车", None, "3d"),
        ("检测 nuScenes 中的车", None, "3d"),
        # ---- 6 位帧号（独立 token）----
        ("标注 KITTI 帧 000123 中的汽车和行人", None, "3d"),
        ("检测 000860 图像中的汽车", None, "3d"),  # 无扩展名帧号 → 3D（宁 3D 勿漏）
        # ---- 12 位编号图像（回归：2026-09-02 实测曾误判 3D）----
        (
            "检测 /root/autodl-tmp/Documents/datasets/COCO2017/val2017/000000000139.jpg"
            " 中的行人",
            None,
            "2d",
        ),
        ("检测 000000000139.jpg 中的汽车", None, "2d"),  # 裸 12 位编号+扩展名
        ("标注 000000000139 帧", None, "2d"),  # 12 位纯数字（非 6 位帧号）
        # ---- det_model ∈ 3D 引擎名（四名单并集）强制 3D ----
        ("检测汽车", "pointpillars_kitti", "3d"),
        ("检测汽车", "centerpoint_nus", "3d"),
        ("检测汽车", "pvrcnn_kitti", "3d"),
        ("检测汽车", "pgd_kitti", "3d"),  # MONO3D 名单
        ("检测汽车", "bevfusion_nus", "3d"),  # BEVFUSION 名单
        ("检测汽车", "fcos3d_nus", "3d"),  # FCOS3D 名单
        ("检测汽车", "POINTPILLARS_KITTI", "3d"),  # 大小写不敏感
    ],
)
def test_route_domain(instruction: str, det_model: str | None, expected: str) -> None:
    assert route_domain(instruction, det_model) == expected


def test_engine3d_names_covers_all_catalogs() -> None:
    """四名单并集完整性：ENGINE3D_NAMES 覆盖 model_catalog 全部 3D 引擎名。"""
    from auto3dlabel.configs.model_catalog import (
        BEVFUSION_NAMES,
        DETECTOR3D_NAMES,
        FCOS3D_NAMES,
        MONO3D_NAMES,
    )

    assert ENGINE3D_NAMES == frozenset(
        {*DETECTOR3D_NAMES, *MONO3D_NAMES, *BEVFUSION_NAMES, *FCOS3D_NAMES}
    )


def test_route_domain_unknown_det_model_defaults_2d() -> None:
    """未知/未注册模型名不强制 3D（宁 2D 勿错；2D 侧自行报未知模型）。"""
    assert route_domain("检测汽车", "not_a_real_model.pt") == "2d"
