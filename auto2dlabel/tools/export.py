"""导出 Tool —— 将标注结果导出为标准格式（COCO / YOLO / VOC / LabelMe）。"""

from __future__ import annotations

from typing import Any

from auto2dlabel.schema.annotation import Annotation
from auto2dlabel.tools import Path
from auto2dlabel.tools.base import Tool


class ExportTool(Tool):
    """标注导出 Tool。

    接受 Annotation 列表，输出为标准训练格式文件。
    """

    name = "export_annotations"
    description = (
        "Export accumulated annotations to a standard format file. "
        "Supports COCO JSON (detection/segmentation), YOLO txt, Pascal VOC XML, "
        "and LabelMe JSON. "
        "Call this at the end of an annotation session to persist results."
    )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "description": "List of annotation dicts accumulated during the session.",
                },
                "format": {
                    "type": "string",
                    "enum": ["coco", "yolo", "voc", "labelme"],
                    "default": "coco",
                    "description": "Export format.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Output file path (COCO) or output directory "
                    "(YOLO/VOC/LabelMe).",
                },
            },
            "required": ["annotations", "output_path"],
        }

    def forward(
        self,
        annotations: list[dict],
        output_path: str,
        format: str = "coco",
        **kwargs,  # 忽略 LLM 可能误传的额外参数 (如 image_path)
    ) -> str:
        """执行导出。

        Args:
            annotations: Annotation dict 列表，或 LLM 直接传入的 bbox dict 列表。
            output_path: 输出文件路径。
            format: 导出格式 (coco/yolo/voc/labelme)。

        Returns:
            输出文件路径。
        """
        # 兼容两种输入格式：
        # 1. 标准格式: [{"image_path": "...", "bboxes": [...], ...}]
        # 2. LLM 直接传 bbox 列表: [{"label": "car", "bbox": [...], "conf": 0.5}, ...]
        if annotations and isinstance(annotations[0], dict):
            first = annotations[0]
            if "bboxes" not in first and ("label" in first or "bbox" in first):
                # LLM 直接传了 bbox 列表，包装成标准格式
                annotations = [{
                    "image_path": kwargs.get("image_path", "unknown.png"),
                    "bboxes": annotations,
                }]

        ann_objects = []
        for a in annotations:
            # 如果没提供 image_size，尝试从文件读取
            image_size = tuple(a.get("image_size", (0, 0)))
            img_path = a.get("image_path", "unknown.png")
            if image_size == (0, 0) and img_path != "unknown.png":
                from PIL import Image as _Image

                try:
                    im = _Image.open(img_path)
                    image_size = (im.width, im.height)
                except Exception:
                    pass

            ann_objects.append(Annotation(
                image_path=img_path,
                image_size=image_size,
                bboxes=[self._make_bbox(b) for b in a.get("bboxes", [])],
                masks=[self._make_mask(m) for m in a.get("masks", [])],
                labels=[self._make_label(lab) for lab in a.get("labels", [])],
                review_flags=a.get("review_flags", []),
                metadata=a.get("metadata", {}),
            ))

        from auto2dlabel.export.cls import export_cls
        from auto2dlabel.export.coco import export_coco
        from auto2dlabel.export.dota import export_dota
        from auto2dlabel.export.labelme import export_labelme
        from auto2dlabel.export.mot import export_mot
        from auto2dlabel.export.voc import export_voc
        from auto2dlabel.export.yolo import export_yolo, export_yolo_obb

        exporters = {
            "coco": export_coco,
            "cls": export_cls,
            "dota": export_dota,
            "labelme": export_labelme,
            "mot": export_mot,
            "yolo": export_yolo,
            "yolo_obb": export_yolo_obb,
            "voc": export_voc,
        }

        exporter = exporters[format]
        exporter(ann_objects, Path(output_path))

        return str(Path(output_path).resolve())

    @staticmethod
    def _make_bbox(d: dict) -> Any:
        from auto2dlabel.schema.annotation import Bbox

        return Bbox(
            x=d["x"], y=d["y"],
            width=d["width"], height=d["height"],
            label=d.get("label", ""),
            confidence=d.get("confidence", 1.0),
            angle=d.get("angle", 0.0),
            track_id=d.get("track_id"),
            keypoints=[
                (float(k[0]), float(k[1]), float(k[2]))
                for k in d.get("keypoints", [])
            ],
        )

    @staticmethod
    def _make_mask(d: dict) -> Any:
        from auto2dlabel.schema.annotation import Bbox, Mask

        b = d["bbox"]
        bbox = Bbox(
            x=b["x"], y=b["y"],
            width=b["width"], height=b["height"],
            label=b.get("label", ""),
            confidence=b.get("confidence", 1.0),
        )
        return Mask(bbox=bbox, segmentation=d.get("segmentation", []), area=d.get("area", 0.0))

    @staticmethod
    def _make_label(d: dict) -> Any:
        from auto2dlabel.schema.annotation import ImageLabel

        return ImageLabel(label=d.get("label", ""), score=d.get("score", 0.0))


def register(registry=None) -> None:
    """向全局注册表注册此 Tool。"""
    from auto2dlabel.tools.registry import registry as reg

    target = registry or reg
    target.register(ExportTool())
