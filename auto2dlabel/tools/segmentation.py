"""分割 Tool —— 接受边界框，调用 SAM 模型生成实例分割 mask。"""

from __future__ import annotations

from auto2dlabel.models.segmentation import SegmentationModel
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.base import Tool


class SegmentationTool(Tool):
    """实例分割 Tool。

    接受 bbox（或未来扩展 point）作为 prompt，调用 SAM 类模型生成 mask。
    """

    name = "segment_mask"
    description = (
        "Generate instance segmentation masks for specified regions in an image. "
        "Takes bounding boxes as input prompts and returns precise pixel-level masks "
        "for each object. Use this after detect_objects to get fine-grained masks "
        "for objects of interest."
    )

    def __init__(self, model: SegmentationModel | None = None) -> None:
        self._model = model

    @property
    def model(self) -> SegmentationModel:
        if self._model is None:
            from auto2dlabel.models.segmentation import FastSAMModel

            self._model = FastSAMModel()
        return self._model

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image.",
                },
                "bboxes": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of bounding boxes from detect_objects output.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["box", "point"],
                    "default": "box",
                    "description": "SAM prompt mode: 'box' uses bbox corners, 'point' uses center points.",
                },
            },
            "required": ["image_path", "bboxes"],
        }

    def forward(
        self,
        image_path: str,
        bboxes: list[dict],
        mode: str = "box",
    ) -> list[dict]:
        """执行分割。

        Args:
            image_path: 图像路径。
            bboxes: bbox dict 列表（来自 DetectionTool 输出）。
            mode: SAM prompt 模式。

        Returns:
            Mask dict 列表。
        """
        # 从 dict 重建 Bbox 对象
        input_bboxes = [
            Bbox(
                x=b["x"],
                y=b["y"],
                width=b["width"],
                height=b["height"],
                label=b.get("label", ""),
                confidence=b.get("confidence", 1.0),
            )
            for b in bboxes
        ]

        masks = self.model.generate(image_path, input_bboxes, mode=mode)

        return [m.to_dict() for m in masks]


def register(registry=None) -> None:
    """向全局注册表注册此 Tool。"""
    from auto2dlabel.tools.registry import registry as reg

    target = registry or reg
    target.register(SegmentationTool())
