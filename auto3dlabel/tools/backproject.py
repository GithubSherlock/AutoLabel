"""反投影语义点云（红线：用 mask 不用 bbox——bbox 边缘把背景点投进语义点云，mask 才是像素级边界）。

流程：SAM2 mask polygon → cv2.fillPoly 栅格化 label 图（每实例一个 ID）→ 投影点花式索引查表。
115k 点一次矩阵乘 + 一次查表毫秒级；mask 内零 LiDAR 点的实例丢弃并计数（宁缺勿假）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from auto3dlabel.schema.box3d import KittiFrame


@dataclass
class BackprojectResult:
    """语义点云（仅命中实例的点；背景点已剔除）。"""

    points_cam: np.ndarray  # (M,3) rectified cam0 相机系
    instance_ids: np.ndarray  # (M,) int32，对应 polygons 下标
    dropped: int  # mask 内零 LiDAR 点的实例数（宁缺勿假）

    def points_of(self, instance: int) -> np.ndarray:
        """实例子点云（空实例返回 (0,3)）。"""
        return cast(np.ndarray, self.points_cam[self.instance_ids == instance])


def rasterize_masks(
    polygons: list[list[list[float]]], img_w: int, img_h: int
) -> np.ndarray:
    """COCO polygon 列表 → (img_h, img_w) label 图，每实例一个 ID（-1 背景）。

    polygons[i] = 第 i 个实例的轮廓列表（每轮廓 [[x1,y1,x2,y2,...]] 平铺）。
    """
    label_map = np.full((img_h, img_w), -1, dtype=np.int32)
    for inst_id, polys in enumerate(polygons):
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
            pts_int = pts.astype(np.int32)
            # cv2.typing.Scalar = Sequence[float]，int 不匹配需转浮点
            cv2.fillPoly(label_map, [pts_int], (float(inst_id),))
    return label_map


def backproject_semantics(
    frame: KittiFrame,
    polygons: list[list[list[float]]],
    max_points: int = 0,
) -> BackprojectResult:
    """帧 + mask polygon 列表 → 语义点云（投影查表，一次向量化）。

    max_points > 0 时按比例抽样点云（调试/大帧提速）。
    """
    calib = frame.load_calib()
    img = frame.load_image()
    img_h, img_w = img.shape[:2]
    pts = frame.load_points()[:, :3]
    if max_points > 0 and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    u, v, valid = calib.project_velo_to_image(pts, img_w, img_h)
    label_map = rasterize_masks(polygons, img_w, img_h)
    # 无效点 u=v=0 哨兵，先置背景再查表
    hit_ids = cast(np.ndarray, np.where(valid, label_map[v, u], -1))
    m = hit_ids >= 0
    points_cam = calib.velo_to_cam(pts)[m]
    instance_ids = hit_ids[m].astype(np.int32)
    dropped = sum(1 for i in range(len(polygons)) if not (instance_ids == i).any())
    return BackprojectResult(points_cam=points_cam, instance_ids=instance_ids, dropped=dropped)
