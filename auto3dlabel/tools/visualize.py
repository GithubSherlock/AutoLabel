"""cv2 渲染可视化（零 open3d 依赖）：投影自检图 / 语义点云 BEV 彩图 / 3D bbox vs GT BEV 对比图。

投影自检图是标定链红线的验收图：LiDAR 点投影回图像，颜色按深度渐变——
投影正确则散点贴附在物体轮廓上；标定错则整体偏移/无点（一眼可辨）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from auto3dlabel.schema.box3d import Box3D, KittiFrame

_BEV_SIZE = 800
_BEV_RANGE = 60.0  # 鸟瞰视野：前后各 60m（车头 z 正向）
_BEV_MAX_POINTS = 60000  # BEV 散点抽样上限（过密不增信息量，纯省时）

# KITTI 3 类框配色（BGR）：Car 绿 / Pedestrian 青 / Cyclist 品红；未知类兜底绿
_CLASS_COLORS = {
    "Car": (0, 255, 0),
    "Pedestrian": (255, 255, 0),
    "Cyclist": (255, 0, 255),
}


def _depth_color(depth: np.ndarray, dmax: float = 80.0) -> np.ndarray:
    """深度 → BGR 色（近暖远冷）。"""
    d = np.clip(depth, 0, dmax) / dmax
    h = (1.0 - d) * 120.0
    hsv = np.zeros((len(d), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = h.astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


def draw_projection_check(frame: KittiFrame, out_path: str | Path, max_points: int = 20000) -> str:
    """点云投影回图像散点自检图（每点 1px，深度着色；按比例抽样防过密）。"""
    img = frame.load_image()
    pts = frame.load_points()
    calib = frame.load_calib()
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    u, v, valid = calib.project_velo_to_image(pts, img.shape[1], img.shape[0])
    cam = calib.velo_to_cam(pts)
    depth = np.linalg.norm(cam[:, [0, 2]], axis=1)
    colors = _depth_color(depth)
    for i in range(len(pts)):
        if valid[i]:
            cv2.circle(img, (int(u[i]), int(v[i])), 1, colors[i].tolist(), -1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(out)


def draw_bev(
    frame: KittiFrame,
    out_path: str | Path,
    points_cam: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    boxes: list[Box3D] | None = None,
    gt_boxes: list[Box3D] | None = None,
) -> str:
    """鸟瞰图（800×800，视野 120m）：可选点云散点 + 预测 3D 框（类别色+标签）+ GT 框（蓝）。

    points_cam: (N,3) 相机系点；colors: (N,3) BGR 色（None 时深度着色）。
    预测框按 KITTI 类着色并标注类别名（未知类兜底绿）。
    """
    canvas = np.full((_BEV_SIZE, _BEV_SIZE, 3), 32, dtype=np.uint8)

    def to_px(x: float, z: float) -> tuple[int, int]:
        px = int((x + _BEV_RANGE) / (2 * _BEV_RANGE) * _BEV_SIZE)
        py = int((_BEV_RANGE - z) / (2 * _BEV_RANGE) * _BEV_SIZE)  # z 前向上
        return px, py

    if points_cam is not None and len(points_cam) > 0:
        if colors is None:
            colors = _depth_color(np.linalg.norm(points_cam[:, [0, 2]], axis=1))
        pts = np.clip(np.stack([to_px(x, z) for x, z in points_cam[:, [0, 2]]]), 0, _BEV_SIZE - 1)
        canvas[pts[:, 1], pts[:, 0]] = colors

    def draw_box(b: Box3D, color: tuple[int, int, int], thickness: int = 2) -> None:
        corners = b.corners_bev()
        for i in range(4):
            p1 = to_px(*corners[i])
            p2 = to_px(*corners[(i + 1) % 4])
            cv2.line(canvas, p1, p2, color, thickness)

    if gt_boxes:
        for b in gt_boxes:
            draw_box(b, (255, 128, 0), 1)  # 蓝 GT
    if boxes:
        for b in boxes:
            color = _CLASS_COLORS.get(b.label, (0, 255, 0))
            draw_box(b, color, 2)
            tx, ty = to_px(*b.corners_bev()[0])
            cv2.putText(
                canvas, b.label, (tx, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return str(out)


def draw_lidar_bev_predictions(
    frame: KittiFrame,
    out_path: str | Path,
    boxes: list[Box3D],
    gt_boxes: list[Box3D] | None = None,
) -> str:
    """LiDAR 点云 BEV 预测图：点云散点（相机系，深度着色）+ 预测 3D 框（类别色）+ GT（蓝）。

    cli run（pipeline._draw_lidar_bev）与 chat（tools3d 的 detect/visualize 工具）
    共用单一事实源——点云线「BEV 点云预测图」验收产物；散点比例抽样防过密
    （≤ _BEV_MAX_POINTS，与投影自检图同款抽样）。
    """
    pts = frame.load_calib().velo_to_cam(frame.load_points())
    if len(pts) > _BEV_MAX_POINTS:
        idx = np.linspace(0, len(pts) - 1, _BEV_MAX_POINTS).astype(int)
        pts = pts[idx]
    return draw_bev(frame, out_path, points_cam=pts, boxes=boxes, gt_boxes=gt_boxes)


def draw_bev_semantic(
    frame: KittiFrame,
    out_path: str | Path,
    points_cam: np.ndarray,
    instance_ids: np.ndarray,
    boxes: list[Box3D] | None = None,
    gt_boxes: list[Box3D] | None = None,
) -> str:
    """语义点云 BEV 彩图：按实例 ID 上色（Phase 2 验收图）。"""
    rng = np.random.RandomState(42)
    n = int(instance_ids.max()) + 1 if len(instance_ids) else 0
    palette = rng.randint(60, 255, size=(max(n, 1), 3)).astype(np.uint8)
    colors = np.zeros((len(points_cam), 3), dtype=np.uint8)
    valid = instance_ids >= 0
    colors[valid] = palette[instance_ids[valid]]
    return draw_bev(frame, out_path, points_cam, colors, boxes, gt_boxes)
