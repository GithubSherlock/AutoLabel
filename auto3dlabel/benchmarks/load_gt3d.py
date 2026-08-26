"""KITTI label_2 GT 3D 加载（评测注入格式）。

7 值 bbox = [h,w,l,x,y,z,ry]（KITTI 原始：y 为底部中心，iou3d_list 经 from_gt_row 归一）；
8 值 quad = corners_bev 展平 [x1,z1,x2,z2,x3,z3,x4,z4]（bev_iou_quad 消费）。
difficulty 复用 auto2dlabel kitti_difficulty（truncated/occluded/2D 框高度判据）。
"""

from __future__ import annotations

from typing import Any

from auto2dlabel.benchmarks.kitti_benchmark import kitti_difficulty

from auto3dlabel.schema.box3d import Box3D, KittiFrame


def gt_frame(frame: KittiFrame) -> dict[str, Any]:
    """帧 GT → {"objects": [{"name", "bbox"(7 值), "quad"(8 值), "bbox2d", "difficulty"}],
    "dontcares": [[x1,y1,x2,y2], ...]}。"""
    objects: list[dict[str, Any]] = []
    dontcares: list[list[float]] = []
    for line in frame.label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 15:
            continue
        name = parts[0]
        bbox2d = [float(v) for v in parts[4:8]]
        if name in ("DontCare", "Misc"):
            dontcares.append(bbox2d)
            continue
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])
        truncated = float(parts[1])
        occluded = int(parts[2])
        b2d_h = float(parts[7]) - float(parts[5])
        difficulty = kitti_difficulty(truncated, occluded, b2d_h)
        if difficulty is None:
            continue
        box = Box3D.from_gt_row(name, h, w, l, x, y, z, ry)
        corners = box.corners_bev()  # (4,2) (x,z)
        objects.append(
            {
                "name": name,
                "bbox": [h, w, l, x, y, z, ry],
                "quad": corners.reshape(-1).tolist(),
                "bbox2d": bbox2d,
                "difficulty": difficulty,
            }
        )
    return {"objects": objects, "dontcares": dontcares}


def pred_frame(boxes: list[Box3D]) -> list[dict[str, Any]]:
    """预测 Box3D → evaluate_per_class 的 pred 格式（bbox 7 值 y 回写底部中心 + quad 8 值）。"""
    out: list[dict[str, Any]] = []
    for b in boxes:
        corners = b.corners_bev()
        out.append(
            {
                "name": b.label,
                "bbox": [
                    b.h,
                    b.w,
                    b.l,
                    b.cx,
                    b.cy + b.h / 2,  # 内部中心 → KITTI 底部口径
                    b.cz,
                    b.rotation_y,
                ],
                "quad": corners.reshape(-1).tolist(),
                "bbox2d": [b.x1, b.y1, b.x2, b.y2],
                "conf": b.confidence,
            }
        )
    return out
