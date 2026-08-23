"""3D 几何：yaw↔rotation_y 唯一转换点 + BEV/3D IoU（shapely）。

yaw 约定（红线）：
- 内部 yaw_bev：车头方向相对 +z 轴、向 +x 为正，[-π, π]
- KITTI rotation_y：绕相机 y 轴，从 +x 向 +z 为正（官方 computeBoxCorners：
  x' = x·cos ry + z·sin ry, z' = -x·sin ry + z·cos ry，车头向量 (cos ry, -sin ry)）
- 由 (sin yaw, cos yaw) = (cos ry, -sin ry) ⇒ ry = yaw_bev - π/2（已单测锁定）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from auto3dlabel.schema.box3d import Box3D


def wrap_pi(angle: float) -> float:
    """归一化到 [-π, π]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_to_rotation_y(yaw_bev: float) -> float:
    """内部 yaw_bev → KITTI rotation_y（唯一转换点）。"""
    return wrap_pi(yaw_bev - np.pi / 2)


def rotation_y_to_yaw(rotation_y: float) -> float:
    """KITTI rotation_y → 内部 yaw_bev（反转换）。"""
    return wrap_pi(rotation_y + np.pi / 2)


def _bev_polygon_from_corners(corners: np.ndarray) -> Polygon:
    """(4,2) 鸟瞰角点（x,z）→ shapely Polygon。"""
    return Polygon(np.asarray(corners, dtype=np.float64))


def bev_polygon_of_box(box: Box3D) -> Polygon:
    """Box3D → 鸟瞰 Polygon（x,z 平面）。"""
    return _bev_polygon_from_corners(box.corners_bev())


def bev_iou(a: Box3D, b: Box3D) -> float:
    """两 Box3D 鸟瞰矩形 IoU（shapely 交集；退化/无交返回 0）。"""
    pa, pb = bev_polygon_of_box(a), bev_polygon_of_box(b)
    if pa.is_empty or pb.is_empty or pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def iou3d(a: Box3D, b: Box3D) -> float:
    """两 Box3D（绕同一相机 y 轴棱柱）3D IoU = BEV 交集 × y 区间交 ÷ 体积并。"""
    inter_area = bev_polygon_of_box(a).intersection(bev_polygon_of_box(b)).area
    if inter_area <= 0:
        return 0.0
    ya_lo, ya_hi = a.cy - a.h / 2, a.cy + a.h / 2
    yb_lo, yb_hi = b.cy - b.h / 2, b.cy + b.h / 2
    y_inter = max(0.0, min(ya_hi, yb_hi) - max(ya_lo, yb_lo))
    if y_inter <= 0:
        return 0.0
    vol_a, vol_b = a.h * a.w * a.l, b.h * b.w * b.l
    vol_inter = inter_area * y_inter
    union = vol_a + vol_b - vol_inter
    return float(vol_inter / union) if union > 0 else 0.0


def iou3d_list(a: list[float], b: list[float]) -> float:
    """评测注入签名（common.evaluate_per_class iou_fn）：7 值 [h,w,l,x,y,z,ry] → 3D IoU。"""
    from auto3dlabel.schema.box3d import Box3D

    ba = Box3D.from_gt_row("x", a[0], a[1], a[2], a[3], a[4], a[5], a[6])
    bb = Box3D.from_gt_row("x", b[0], b[1], b[2], b[3], b[4], b[5], b[6])
    return iou3d(ba, bb)


def bev_iou_quad(a: list[float], b: list[float]) -> float:
    """评测注入签名：8 值 BEV 四边形 [x1,z1,x2,z2,x3,z3,x4,z4] → 多边形 IoU。"""
    pa = _bev_polygon_from_corners(np.asarray(a, dtype=np.float64).reshape(4, 2))
    pb = _bev_polygon_from_corners(np.asarray(b, dtype=np.float64).reshape(4, 2))
    if pa.is_empty or pb.is_empty or pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0
