"""test_fit：合成矩形 yaw 恢复（0/30/90° ±0.05）+ 180° 模糊 + 退化丢弃。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from auto3dlabel.configs.kitti import MIN_FIT_POINTS_ABS
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tests.helpers.synth import rect_points
from auto3dlabel.tools.fit import _fit_yaw, fit_box3d


def _pts(yaw: float, n: int = 500, cx: float = 10.0, cz: float = 20.0) -> np.ndarray:
    return rect_points(cx, cz, 4.0, 2.0, yaw, n=n, y_lo=-1.5, y_hi=-0.3)


@pytest.mark.parametrize("yaw", [0.0, 0.52, 1.571])
def test_yaw_recovery(yaw: Any) -> None:
    """矩形 yaw 恢复 ±0.05（含 90°）。"""
    box = fit_box3d(_pts(yaw), "Car", 0.9, 100.0, 100.0, 300.0, 300.0)
    assert box is not None
    d = abs(box.yaw_bev - yaw)
    assert min(d, abs(d - np.pi)) < 0.05  # 180° 模糊等价


def test_yaw_180_ambiguous() -> None:
    """长边方向 ±v 等价：yaw 与 yaw+π 的框 IoU=1（180° 模糊交 HITL）。"""
    box = fit_box3d(_pts(0.3), "Car", 0.9, 0.0, 0.0, 100.0, 100.0)
    assert box is not None
    flipped = Box3D(
        label="Car", cx=box.cx, cy=box.cy, cz=box.cz, h=box.h, w=box.w, l=box.l,
        yaw_bev=box.yaw_bev + np.pi,
    )
    from auto3dlabel.tools.geometry import bev_iou

    assert abs(bev_iou(box, flipped) - 1.0) < 1e-6


def test_size_recovery() -> None:
    box = fit_box3d(_pts(0.0), "Car", 0.9, 0.0, 0.0, 100.0, 100.0)
    assert box is not None
    assert abs(box.l - 4.0) < 0.15 and abs(box.w - 2.0) < 0.15
    assert 0.9 < box.h < 1.5  # y_lo=-1.5 y_hi=-0.3 → 5-95 分位略窄于 1.2


def test_center_recovery() -> None:
    box = fit_box3d(_pts(0.0), "Car", 0.9, 0.0, 0.0, 100.0, 100.0)
    assert box is not None
    assert abs(box.cx - 10.0) < 0.1 and abs(box.cz - 20.0) < 0.1


def test_center_uses_cx_mean() -> None:
    """cx/cz = 点均值（非 minAreaRect 中心），点云稀疏不对称时更稳。"""
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=300)
    pts[:, 0] += 0.5  # 偏置后均值仍追踪
    box = fit_box3d(pts, "Car", 0.9, 0.0, 0.0, 100.0, 100.0)
    assert box is not None
    assert abs(box.cx - 10.5) < 0.15


def test_degenerate_too_few_points_none() -> None:
    assert fit_box3d(_pts(0.0, n=MIN_FIT_POINTS_ABS - 1), "Car", 0.9, 0, 0, 10, 10) is None


def test_degenerate_thin_sheet_none() -> None:
    """y 高度 < 0.2（地面薄片）→ None。"""
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=200, y_lo=-1.0, y_hi=-0.95)
    assert fit_box3d(pts, "Car", 0.9, 0, 0, 10, 10) is None


def test_low_fit_points_review_flag() -> None:
    """fit_points < MIN_FIT_POINTS(20) → review_flag。"""
    pts = rect_points(10.0, 20.0, 4.0, 2.0, 0.0, n=12, y_lo=-1.5, y_hi=-0.3)
    box = fit_box3d(pts, "Car", 0.9, 0, 0, 10, 10)
    assert box is not None and box.review_flag and box.fit_points == 12


def test_adhesion_sets_review() -> None:
    box = fit_box3d(_pts(0.0), "Car", 0.9, 0, 0, 100, 100, adhesion_flag=True)
    assert box is not None and box.review_flag


def test_oversize_dropped() -> None:
    """尺寸超 MAX_SIZE_FACTOR × 类上限 → None（背景混入宁缺勿假）。"""
    pts = rect_points(10.0, 20.0, 12.0, 5.0, 0.0, n=500)  # Car 上限 l=7 w=3.2
    assert fit_box3d(pts, "Car", 0.9, 0, 0, 100, 100) is None


def test_near_square_review_flag() -> None:
    """长宽比 < 1.4 → review_flag（yaw 方向不可靠）。"""
    pts = rect_points(10.0, 20.0, 1.5, 1.4, 0.0, n=500)
    box = fit_box3d(pts, "Pedestrian", 0.9, 0, 0, 100, 100)
    assert box is not None and box.review_flag


def test_healthy_box_no_review() -> None:
    box = fit_box3d(_pts(0.0), "Car", 0.95, 0, 0, 100, 100)
    assert box is not None and not box.review_flag
    assert box.fit_points == 500


def test_fit_yaw_direct() -> None:
    """_fit_yaw 直接调用：返回 (yaw, l, w, aspect)，l ≥ w。"""
    yaw, l, w, aspect = _fit_yaw(rect_points(0, 0, 4.0, 2.0, 0.3, n=300)[:, [0, 2]])
    assert l >= w and aspect >= 1.0
