"""COCO JSON 格式导出。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from auto2dlabel.export import Path, json
from auto2dlabel.schema.annotation import Annotation


def build_coco_dict(annotations: list[Annotation]) -> dict[str, Any]:
    """构建 COCO JSON 字典（纯函数，无 I/O）。

    供 export_coco 写文件与 Web /api/export-coco 端点共用，
    保证导出逻辑单一事实源。

    Args:
        annotations: Annotation 对象列表。

    Returns:
        COCO 格式 dict。
    """
    coco: dict[str, Any] = {
        "info": {
            "description": "AutoLabel exported annotations",
            "version": "0.1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_name_to_id: dict[str, int] = {}
    next_category_id = 1
    next_ann_id = 1

    for image_id, ann in enumerate(annotations, start=1):
        img_w, img_h = ann.image_size if ann.image_size != (0, 0) else (0, 0)

        # 图像信息
        coco["images"].append({
            "id": image_id,
            "file_name": Path(ann.image_path).name,
            "width": img_w,
            "height": img_h,
        })

        # 收集所有类别
        for bbox in ann.bboxes:
            if bbox.label not in category_name_to_id:
                category_name_to_id[bbox.label] = next_category_id
                coco["categories"].append({
                    "id": next_category_id,
                    "name": bbox.label,
                    "supercategory": "",
                })
                next_category_id += 1

        # Bbox annotations
        for bbox in ann.bboxes:
            cat_id = category_name_to_id[bbox.label]
            coco_ann: dict[str, Any] = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                "area": bbox.area(),
                "iscrowd": 0,
                "score": bbox.confidence,
            }
            if bbox.track_id is not None:
                coco_ann["track_id"] = bbox.track_id
            coco["annotations"].append(coco_ann)
            next_ann_id += 1

        # Mask annotations (使用 polygon 格式)
        for mask in ann.masks:
            cat_id = category_name_to_id.get(mask.bbox.label, 1)
            coco_ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [mask.bbox.x, mask.bbox.y, mask.bbox.width, mask.bbox.height],
                "area": mask.area,
                "iscrowd": 0,
                "segmentation": mask.segmentation,
                "score": mask.bbox.confidence,
            }
            if mask.bbox.track_id is not None:
                coco_ann["track_id"] = mask.bbox.track_id
            coco["annotations"].append(coco_ann)
            next_ann_id += 1

    return coco


def export_coco(annotations: list[Annotation], output_path: Path) -> None:
    """将标注列表导出为 COCO JSON 文件。

    Args:
        annotations: Annotation 对象列表。
        output_path: 输出 JSON 文件路径。
    """
    coco = build_coco_dict(annotations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False))
