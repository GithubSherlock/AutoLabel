"""LabelMe JSON 格式导出。"""

from __future__ import annotations

from typing import Any

from auto2dlabel.export import Path, json
from auto2dlabel.schema.annotation import Annotation

LABELME_VERSION = "4.5.6"


def export_labelme(annotations: list[Annotation], output_dir: Path) -> None:
    """将标注列表导出为 LabelMe JSON 格式。

    每张图像对应一个同名的 .json 文件，直接写在 output_dir 下：
    - Bbox → shape_type "rectangle"，points 为 [x1,y1] / [x2,y2] 两角点
    - Mask → shape_type "polygon"，COCO 扁平点列按 2 个一组配对
    imageData 为空串（labelme --nodata 惯例，读取端回退到 imagePath）。

    注意事项：
    - 同名 stem 冲突时后者覆盖前者（与 yolo/voc 语义一致）
    - 坐标保留浮点不取整（labelme 规范不要求 points 为整数）
    - 多段 polygon（COCO [[p1],[p2],...]）全部展平进一个 shape 的点列
    - 少于 3 点的退化 polygon、空 segmentation 原样输出，读取端可能拒绝

    Args:
        annotations: Annotation 对象列表。
        output_dir: 输出目录（JSON 文件直接写在目录根）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        shapes: list[dict[str, Any]] = []  # 先 bbox 后 mask，保持输入顺序
        for bbox in ann.bboxes:
            x1, y1, x2, y2 = bbox.xyxy
            shapes.append({
                "label": bbox.label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": bbox.id,
                "shape_type": "rectangle",
                "flags": {},
            })
        for mask in ann.masks:
            points = [
                [flat[i], flat[i + 1]]
                for flat in mask.segmentation
                for i in range(0, len(flat) - 1, 2)
            ]
            shapes.append({
                "label": mask.bbox.label,
                "points": points,
                "group_id": mask.bbox.id,
                "shape_type": "polygon",
                "flags": {},
            })

        width, height = ann.image_size  # (0, 0) 时输出 0，与现有三器兜底一致
        data = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": Path(ann.image_path).name,
            "imageData": "",
            "imageHeight": height,
            "imageWidth": width,
        }

        stem = Path(ann.image_path).stem
        (output_dir / f"{stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )
