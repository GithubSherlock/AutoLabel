"""test_geometry：yaw↔rotation_y 转换（红线唯一转换点）+ BEV/3D IoU 已知值。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tools.geometry import (
    bev_iou,
    bev_iou_quad,
    iou3d,
    iou3d_list,
    points_in_box,
    rotation_y_to_yaw,
    wrap_pi,
    yaw_to_rotation_y,
)


def _box(
    cx: float = 0.0, cz: float = 0.0, l: float = 4.0, w: float = 2.0,
    yaw: float = 0.0, cy: float = 0.0, h: float = 1.5, label: str = "Car",
) -> Box3D:
    return Box3D(label=label, cx=cx, cy=cy, cz=cz, h=h, w=w, l=l, yaw_bev=yaw)


# ── yaw ↔ rotation_y ─────────────────────────────────────────

def test_yaw_zero_maps_to_minus_pi_over_2() -> None:
    """yaw=0（车头朝 +z）→ rotation_y = -π/2（KITTI：ry=-π/2 车头正对相机）。"""
    assert abs(yaw_to_rotation_y(0.0) + np.pi / 2) < 1e-9


def test_rotation_y_zero_maps_to_pi_over_2() -> None:
    """GT 000000 ry≈0 → 车头朝 +x（相机系），yaw = +π/2。"""
    assert abs(rotation_y_to_yaw(0.0) - np.pi / 2) < 1e-9


@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.7, np.pi / 2, -np.pi / 2, 2.8, -3.0])
def test_roundtrip(yaw: Any) -> None:
    assert abs(rotation_y_to_yaw(yaw_to_rotation_y(yaw)) - yaw) < 1e-9


@pytest.mark.parametrize("ry", [0.0, 0.3, -0.7, np.pi / 2, -np.pi / 2, 2.8, -3.0])
def test_roundtrip_inverse(ry: Any) -> None:
    assert abs(yaw_to_rotation_y(rotation_y_to_yaw(ry)) - ry) < 1e-9


def test_wrap_pi() -> None:
    assert abs(wrap_pi(np.pi + 0.5) - (-np.pi + 0.5)) < 1e-9
    assert abs(wrap_pi(-np.pi - 0.5) - (np.pi - 0.5)) < 1e-9


def test_box_rotation_y_property() -> None:
    b = _box(yaw=np.pi / 2)
    assert abs(b.rotation_y - 0.0) < 1e-9


# ── corners / BEV IoU ─────────────────────────────────────────

def test_corners_bev_shape_and_order() -> None:
    b = _box(yaw=0.0)
    corners = b.corners_bev()
    assert corners.shape == (4, 2)
    # yaw=0：长边沿 z，半长 2；顺序 右前/右后/左后/左前
    assert np.allclose(corners[0], [1.0, 2.0], atol=1e-9)  # 右前
    assert np.allclose(corners[1], [1.0, -2.0], atol=1e-9)  # 右后
    assert np.allclose(corners[2], [-1.0, -2.0], atol=1e-9)  # 左后
    assert np.allclose(corners[3], [-1.0, 2.0], atol=1e-9)  # 左前


def test_bev_iou_identical() -> None:
    assert abs(bev_iou(_box(), _box()) - 1.0) < 1e-9


def test_bev_iou_half_overlap() -> None:
    a = _box(cx=0.0, l=4.0, w=2.0)  # x∈[-1,1], z∈[-2,2]
    b = _box(cx=1.0, l=4.0, w=2.0)  # x∈[0,2], z∈[-2,2] → 交 0.5
    assert abs(bev_iou(a, b) - 1.0 / 3.0) < 1e-9


def test_bev_iou_disjoint() -> None:
    a = _box(cx=0.0)
    b = _box(cx=20.0)
    assert bev_iou(a, b) == 0.0


def test_bev_iou_quad_known() -> None:
    """8 值四边形：单位正方形 vs 右移 0.5 → 交 0.5/并 1.5 = 1/3。"""
    unit = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    shifted = [0.5, 0.0, 1.5, 0.0, 1.5, 1.0, 0.5, 1.0]
    assert abs(bev_iou_quad(unit, shifted) - 1.0 / 3.0) < 1e-9


# ── 3D IoU ────────────────────────────────────────────────────

def test_iou3d_identical() -> None:
    assert abs(iou3d(_box(), _box()) - 1.0) < 1e-9


def test_iou3d_no_y_overlap() -> None:
    a = _box(cy=0.0)
    b = _box(cy=10.0)
    assert iou3d(a, b) == 0.0


def test_iou3d_list_signature() -> None:
    """7 值 [h,w,l,x,y,z,ry] 注入签名（y 为底部中心口径）。"""
    a = [1.5, 2.0, 4.0, 0.0, -0.75, 0.0, -np.pi / 2]  # y=-0.75 底部 → cy=0
    b = [1.5, 2.0, 4.0, 0.0, -0.75, 0.0, -np.pi / 2]
    assert abs(iou3d_list(a, b) - 1.0) < 1e-9


def test_iou3d_list_rotated_yaw_differs() -> None:
    """ry 差 π/2 的同一中心框（4×2 非正方形）：BEV 交 2×2=4，并 8+8-4=12 → 1/3。"""
    a = [1.5, 2.0, 4.0, 0.0, -0.75, 0.0, 0.0]  # yaw=π/2 → 4×2
    b = [1.5, 2.0, 4.0, 0.0, -0.75, 0.0, np.pi / 2]  # yaw=π → 2×4
    assert abs(iou3d_list(a, b) - 1.0 / 3.0) < 1e-6


# ── points_in_box（v0.2 LiDAR 引擎 fit_points 替代）─────────────

def test_points_in_box_counts_inside() -> None:
    """框内点计数：相机系 4×2×1.2 框（yaw=π/2 车头 +x），点云半在 z 窗内。"""
    rng = np.random.default_rng(0)
    pts = np.stack([
        rng.uniform(0.0, 2.0, 400),   # x_cam
        rng.uniform(0.3, 1.5, 400),   # y_cam
        rng.uniform(8.0, 12.0, 400),  # z_cam
    ], axis=1)
    box = Box3D(label="Car", cx=1.0, cy=0.9, cz=10.0, h=1.2, w=2.0, l=4.0, yaw_bev=np.pi / 2)
    n = points_in_box(pts, box)
    assert 150 < n < 250  # z∈[9,11] 约 200/400，x/y 全部在框内


def test_points_in_box_none_outside() -> None:
    """框远离点云 → 0。"""
    pts = np.zeros((10, 3), dtype=np.float64)
    box = Box3D(label="Car", cx=100.0, cy=0.0, cz=100.0, h=1.0, w=1.0, l=1.0)
    assert points_in_box(pts, box) == 0


def test_points_in_box_y_threshold() -> None:
    """y 超出高度区间不计（相机系 y 向下）。"""
    pts = np.zeros((5, 3), dtype=np.float64)
    pts[:, 1] = np.linspace(-5.0, 5.0, 5)
    box = Box3D(label="Car", cx=0.0, cy=0.0, cz=0.0, h=2.0, w=2.0, l=2.0)
    assert points_in_box(pts, box) == 1  # y∈[-5,-2.5,0,2.5,5] 仅 0 落在 y∈[-1,1]
