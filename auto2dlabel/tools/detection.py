"""检测 Tool —— 调用检测模型（Grounding DINO / Ultralytics），返回边界框列表。"""

from __future__ import annotations

import os

from auto2dlabel.models.detection import (
    DetectionModel,
    create_detection_model,
    detect_image_sahi,
    detect_with_retry,
)
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.base import Tool


class DetectionTool(Tool):
    """目标检测 Tool。

    支持 Grounding DINO（开放词汇）与 Ultralytics 全系列 YOLO 模型。
    通过 DetectionModel 抽象层自动适配。
    可选启用 SAHI 切片推理以处理大分辨率图像。
    """

    name = "detect_objects"
    description = (
        "Detect objects in an image. Returns a list of bounding boxes with class "
        "labels and confidence scores. Use this tool when you need to find objects "
        "in an image by describing them in natural language (e.g., 'car', 'person'). "
        "You can pass multiple prompts at once."
    )

    def __init__(
        self,
        model: DetectionModel | None = None,
        model_name: str | None = None,
        use_sahi: bool = False,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self.use_sahi = use_sahi
        # 最近一次 forward 是否降阈值重试（供 orchestrator 质量评估读取）
        self.last_retried = False
        self.last_retry_threshold: float | None = None

    @property
    def model(self) -> DetectionModel:
        if self._model is None:
            name = self._model_name or os.environ.get("DETECTION_MODEL")
            self._model = create_detection_model(name)
        return self._model

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image (JPG/PNG/TIFF).",
                },
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of object descriptions to detect, e.g. ['car', 'person'].",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Minimum confidence score (0-1).",
                },
            },
            "required": ["image_path", "prompts"],
        }

    def forward(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[Bbox]:
        """执行检测。use_sahi=True 时启用切片推理。

        0 框时自动降阈值（×0.5）重试一次；重试发生在 tool 内部，
        对 Agent Loop 完全透明，不增加迭代次数。
        """
        if self.use_sahi:
            dets, self.last_retried, self.last_retry_threshold = detect_with_retry(
                lambda conf: detect_image_sahi(
                    self.model, image_path, prompts,
                    confidence_threshold=conf,
                ),
                confidence_threshold,
            )
            bboxes = [
                Bbox(x=d["bbox"][0], y=d["bbox"][1],
                     width=d["bbox"][2] - d["bbox"][0],
                     height=d["bbox"][3] - d["bbox"][1],
                     label=d["name"], confidence=d["conf"])
                for d in dets
            ]
        else:
            results, self.last_retried, self.last_retry_threshold = detect_with_retry(
                lambda conf: self.model.detect(image_path, prompts, conf),
                confidence_threshold,
            )
            bboxes = [
                Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                     label=r.label, confidence=r.confidence)
                for r in results
            ]
        bboxes.sort(key=lambda b: b.confidence, reverse=True)
        return bboxes


def register(registry=None) -> None:
    from auto2dlabel.tools.registry import registry as reg
    target = registry or reg
    target.register(DetectionTool())
