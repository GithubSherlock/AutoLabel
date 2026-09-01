"""3D 几何：yaw↔rotation_y 唯一转换点 + BEV/3D IoU（shapely）+ nuScenes 四元数转换。

yaw 约定（红线）：
- 内部 yaw_bev：车头方向相对 +z 轴、向 +x 为正，[-π, π]
- KITTI rotation_y：绕相机 y 轴，从 +x 向 +z 为正（官方 computeBoxCorners：
  x' = x·cos ry + z·sin ry, z' = -x·sin ry + z·cos ry，车头向量 (cos ry, -sin ry)）
- 由 (sin yaw, cos yaw) = (cos ry, -sin ry) ⇒ ry = yaw_bev - π/2（已单测锁定）
- nuScenes 全局系 yaw（绕 z 轴，x 前 y 左）→ 四元数 (w,x,y,z)：
  车头向量 (cos yaw, sin yaw, 0) ⇒ quat = (cos(yaw/2), 0, 0, sin(yaw/2))
  ——唯一转换点 yaw_to_quat / quat_to_yaw（v0.2 P3）
- nuScenes 全局系 → ego 局部「相机式」帧（P2 Web 渲染，x 右/y 下/z 前）：
  GLOBAL_TO_CAM_LIKE 矩阵 + global_to_cam_like/cam_like_to_global（见下）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from auto3dlabel.schema.box3d import Box3D


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """nuScenes 全局系 yaw（绕 z 轴）→ 四元数 (w,x,y,z)。"""
    half = yaw / 2
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def quat_to_yaw(quat: tuple[float, float, float, float]) -> float:
    """四元数 (w,x,y,z) → 绕 z 轴 yaw（忽略 x/y 分量，nuScenes 目标框仅绕 z 旋转）。"""
    w, _x, _y, z = quat
    return wrap_pi(2 * float(np.arctan2(z, w)))


def wrap_pi(angle: float) -> float:
    """归一化到 [-π, π]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_to_rotation_y(yaw_bev: float) -> float:
    """内部 yaw_bev → KITTI rotation_y（唯一转换点）。"""
    return wrap_pi(yaw_bev - np.pi / 2)


def rotation_y_to_yaw(rotation_y: float) -> float:
    """KITTI rotation_y → 内部 yaw_bev（反转换）。"""
    return wrap_pi(rotation_y + np.pi / 2)


# ── nuScenes 全局系 ↔ ego 局部「相机式」帧（P2 Web 复核渲染，唯一转换点）────────
#
# 全局系（x 前 / y 左 / z 上）→ 相机式帧（x 右 / y 下 / z 前）的旋转矩阵 M：
#   p_camlike = M @ (p_global − t_ego)
# M = [[0,-1,0],[0,0,-1],[1,0,0]]（正交矩阵，M⁻¹ = Mᵀ，单测锚点锁定）。
# 相机式帧 yaw_bev = −yaw_g（车头 (cos y_g, sin y_g) → (−sin y_g, cos y_g)，
# 代入「车头相对 +z、向 +x 为正」定义即 atan2(−sin, cos) = −y_g）。

GLOBAL_TO_CAM_LIKE = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def global_to_cam_like(
    points: np.ndarray, ego_translation: tuple[float, float, float] | list[float] | np.ndarray
) -> np.ndarray:
    """全局系点 (N,3) → ego 局部相机式帧 (x 右/y 下/z 前)（P2 Web 渲染）。

    ego_translation = devkit ego_pose 全局位置 (x,y,z)（data/nuscenes.py 查表传入）。
    """
    t = np.asarray(ego_translation, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    return (pts - t) @ GLOBAL_TO_CAM_LIKE.T


def cam_like_to_global(
    points: np.ndarray, ego_translation: tuple[float, float, float] | list[float] | np.ndarray
) -> np.ndarray:
    """相机式帧点 (N,3) → 全局系（global_to_cam_like 逆变换，M 正交 ⇒ 转置即逆）。"""
    t = np.asarray(ego_translation, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    return (pts @ GLOBAL_TO_CAM_LIKE) + t


def nus_yaw_to_rotation_y(yaw_g: float) -> float:
    """nuScenes 全局系 yaw → KITTI 相机式 rotation_y：ry = −yaw_g − π/2。

    推导：相机式帧 yaw_bev = −yaw_g，代入唯一转换点 yaw_to_rotation_y。
    """
    return wrap_pi(-yaw_g - np.pi / 2)


def rotation_y_to_nus_yaw(rotation_y: float) -> float:
    """KITTI 相机式 rotation_y → nuScenes 全局系 yaw（反转换）。"""
    return wrap_pi(-rotation_y - np.pi / 2)


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


def points_in_box(points: np.ndarray, box: Box3D) -> int:
    """3D 框内 LiDAR 点计数（LiDAR 观测性 → v0.2 LiDAR 引擎的 fit_points 替代）。

    相机系 (N,3) 点云 → box 局部系（沿车头 u / 沿右 v / y 高度）矩形过滤。
    triage_3d 的 fit_points < MIN_FIT_POINTS 强制 review 逻辑零改动复用。
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    dx, dz = np.sin(box.yaw_bev), np.cos(box.yaw_bev)  # 车头单位向量
    px, pz = dz, -dx  # 垂直向量（右）
    rel_x = pts[:, 0] - box.cx
    rel_y = pts[:, 1] - box.cy
    rel_z = pts[:, 2] - box.cz
    u = rel_x * dx + rel_z * dz
    v = rel_x * px + rel_z * pz
    inside = (
        (np.abs(u) <= box.l / 2) & (np.abs(v) <= box.w / 2) & (np.abs(rel_y) <= box.h / 2)
    )
    return int(inside.sum())


def bev_iou_quad(a: list[float], b: list[float]) -> float:
    """评测注入签名：8 值 BEV 四边形 [x1,z1,x2,z2,x3,z3,x4,z4] → 多边形 IoU。"""
    pa = _bev_polygon_from_corners(np.asarray(a, dtype=np.float64).reshape(4, 2))
    pb = _bev_polygon_from_corners(np.asarray(b, dtype=np.float64).reshape(4, 2))
    if pa.is_empty or pb.is_empty or pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0
