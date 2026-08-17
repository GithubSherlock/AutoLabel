"""DOTA 旋转框格式导出。

每张图像对应一个同名的 .txt 文件，每行一个标注：
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> 0
4 个角点由 center + R(angle)·(±w/2, ±h/2) 计算，按中心角 atan2 排序
（图像坐标系 y 向下，atan2 递增即顺时针）；末尾 0 为 difficulty（恒 0）。
同时生成 classes.txt 记录类别名 → class_id 映射。
"""

from __future__ import annotations

import math

from auto2dlabel.export import Path
from auto2dlabel.schema.annotation import Annotation


def rotated_corners(
    cx: float, cy: float, width: float, height: float, angle: float,
) -> list[tuple[float, float]]:
    """旋转框 4 角点（顺时针）。

    width 轴相对 x 轴旋转 angle（弧度，y 向下坐标系正角为顺时针），
    与 ultralytics xywhr 约定一致；角点按中心角排序保证确定性。
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    offsets = [
        (width / 2, height / 2),
        (width / 2, -height / 2),
        (-width / 2, -height / 2),
        (-width / 2, height / 2),
    ]
    corners = [
        (cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a)
        for hw, hh in offsets
    ]
    corners.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return corners


def export_dota(annotations: list[Annotation], output_dir: Path) -> None:
    """将旋转框标注导出为 DOTA 格式。

    angle=0 时退化为轴对齐框（4 角点即外接矩形角点）。

    Args:
        annotations: Annotation 对象列表（bboxes 带 angle）。
        output_dir: 输出目录（每图一个 txt 直接写在目录根）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集类别名称 → 类别 ID
    class_names: list[str] = []
    class_name_to_id: dict[str, int] = {}
    for ann in annotations:
        for bbox in ann.bboxes:
            if bbox.label not in class_name_to_id:
                class_name_to_id[bbox.label] = len(class_names)
                class_names.append(bbox.label)

    # 写入 classes.txt
    (output_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    for ann in annotations:
        stem = Path(ann.image_path).stem
        txt_path = output_dir / f"{stem}.txt"

        lines = []
        for bbox in ann.bboxes:
            cls_id = class_name_to_id[bbox.label]
            cx = bbox.x + bbox.width / 2
            cy = bbox.y + bbox.height / 2
            corners = rotated_corners(cx, cy, bbox.width, bbox.height, bbox.angle)
            coords = " ".join(f"{c:.4f}" for pt in corners for c in pt)
            lines.append(f"{cls_id} {coords} 0")

        txt_path.write_text("\n".join(lines), encoding="utf-8")
