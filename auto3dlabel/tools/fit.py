"""主簇 → 3D bbox 拟合（Box3D 单一产出源，穿透点纪律见 schema/box3d.py）。

yaw：minAreaRect 角点法（boxPoints → 最长边向量 v，yaw_bev = atan2(vx, vz)）——
绕开 cv2 4.x angle 字段语义坑；方向规则「车头 z 分量 ≥ 0」，180° 模糊对 IoU 无影响，
仅近正方形（长宽比 < 1.4）目标标记 review_flag 交 HITL。
高度：相机 y 5-95 分位 → cy/h（剔除地面/顶部噪点）。
"""

from __future__ import annotations

import cv2
import numpy as np

from auto3dlabel.configs.kitti import (
    CLASS_MAX_DIMS,
    MAX_SIZE_FACTOR,
    MIN_FIT_POINTS,
    MIN_FIT_POINTS_ABS,
    Z_PERCENT_HIGH,
    Z_PERCENT_LOW,
)
from auto3dlabel.schema.box3d import Box3D


def _fit_yaw(pts_bev: np.ndarray) -> tuple[float, float, float, float]:
    """BEV 点 → (yaw_bev, l, w, aspect)。minAreaRect 角点法 + 长边方向。"""
    rect = cv2.minAreaRect(pts_bev.astype(np.float32))
    box = cv2.boxPoints(rect)  # (4,2) 固定顺序（从 angle 0° 顶点绕行）
    v1 = box[1] - box[0]
    v2 = box[2] - box[1]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    v, _n = (v1, n1) if n1 >= n2 else (v2, n2)  # 长边
    l, w = max(n1, n2), min(n1, n2)
    yaw = float(np.arctan2(v[0], v[1]))  # atan2(vx, vz)：车头相对 +z、向 +x 为正
    if np.cos(yaw) < 0:  # 确定性方向规则：车头 z 分量 ≥ 0（180° 翻转等价框）
        yaw = float(np.arctan2(-v[0], -v[1]))
    aspect = l / max(w, 1e-6)
    return yaw, float(l), float(w), float(aspect)


def fit_box3d(
    points_cam: np.ndarray,
    label: str,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    adhesion_flag: bool = False,
) -> Box3D | None:
    """主簇相机系点 → Box3D；退化（< MIN_FIT_POINTS_ABS 点/高度无效）返回 None。

    review_flag 置位条件：fit_points < MIN_FIT_POINTS / 粘连嫌疑 / 尺寸超类上限 / 近正方形。
    """
    if len(points_cam) < MIN_FIT_POINTS_ABS:
        return None
    pts_bev = points_cam[:, [0, 2]]
    yaw, l, w, aspect = _fit_yaw(pts_bev)
    y_pts = points_cam[:, 1]
    y_low = float(np.percentile(y_pts, Z_PERCENT_LOW))
    y_high = float(np.percentile(y_pts, Z_PERCENT_HIGH))
    h = y_high - y_low
    if h < 0.2:  # 地面薄片/无效高度
        return None
    cy = (y_low + y_high) / 2
    cx, cz = float(pts_bev[:, 0].mean()), float(pts_bev[:, 1].mean())

    review = bool(adhesion_flag or len(points_cam) < MIN_FIT_POINTS)
    max_l, max_w = CLASS_MAX_DIMS.get(label, (7.0, 3.2))
    if l > MAX_SIZE_FACTOR * max_l or w > MAX_SIZE_FACTOR * max_w:
        return None  # 超类上限太多 = 背景混入，宁缺勿假直接丢弃
    if l > max_l or w > max_w:
        review = True
    if aspect < 1.4:  # 近正方形：yaw 方向不可靠（180° 模糊），交 HITL
        review = True

    return Box3D(
        label=label,
        confidence=confidence,
        cx=cx,
        cy=cy,
        cz=cz,
        h=h,
        w=w,
        l=l,
        yaw_bev=yaw,
        fit_points=len(points_cam),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        review_flag=review,
    )
