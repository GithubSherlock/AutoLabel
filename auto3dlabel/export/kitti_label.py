"""KITTI label_2 导出（15 字段行）。

红线：Box3D 内部 cy 是体积中心，label_2 的 y 是底部中心——
line_from_box3d 是 y = cy + h/2 回写的唯一出口（对称于 Box3D.from_gt_row 的 y - h/2）。
"""

from __future__ import annotations

from pathlib import Path

from auto3dlabel.schema.box3d import Box3D


def line_from_box3d(box: Box3D) -> str:
    """Box3D → label_2 15 字段行。

    "Car 0.00 0 0.00 x1 y1 x2 y2 h w l x y z ry"
    truncated/occluded/alpha：v0.1 由 2D 检测不可得，恒 0（评测不计）。
    """
    y_bottom = box.cy + box.h / 2  # 内部中心 → KITTI 底部中心（唯一出口）
    return (
        f"{box.label} 0.00 0 0.00 "
        f"{box.x1:.2f} {box.y1:.2f} {box.x2:.2f} {box.y2:.2f} "
        f"{box.h:.2f} {box.w:.2f} {box.l:.2f} "
        f"{box.cx:.2f} {y_bottom:.2f} {box.cz:.2f} {box.rotation_y:.2f}"
    )


def build_label_file(
    frame_id: str, boxes: list[Box3D], out_dir: str | Path
) -> Path:
    """帧 → outputs/labels/{frame_id}.txt（按 conf 降序）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{frame_id}.txt"
    lines = [line_from_box3d(b) for b in sorted(boxes, key=lambda b: -b.confidence)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
