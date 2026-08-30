"""Fake 注入（零真实权重铁律：Protocol duck-typed，不经 create_*_model 工厂）。

FakeDetection/FakeSegmentation 满足 auto2dlabel 的 DetectionModel/SegmentationModel
Protocol（pipeline 消费 .detect / .generate）；FakeLLM 脚本化 LLM 响应队列。
"""

from __future__ import annotations

from typing import Any

from auto2dlabel.agent.llm import LLMResponse
from auto2dlabel.models.detection import DetectionResult
from auto3dlabel.models.detection3d import Det3DResult


class FakeDetection:
    """按 conf 过滤返回预设 DetectionResult，记录每次调用 (image, prompts, conf)。"""

    def __init__(self, dets: list[DetectionResult]) -> None:
        self.dets = list(dets)
        self.calls: list[tuple[str, list[str], float]] = []

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        self.calls.append((image_path, list(prompts), confidence_threshold))
        return [d for d in self.dets if d.confidence >= confidence_threshold]


class FakeMask:
    """SAM2 风格 mask：segmentation = COCO polygon 平铺列表 [[x1,y1,x2,y2,...]]。"""

    def __init__(self, polygon: list[float]) -> None:
        self.segmentation = [list(polygon)]


class FakeSegmentation:
    """按 bboxes 顺序返回预设 polygon 列表，记录调用。"""

    def __init__(self, polygons: list[list[float]]) -> None:
        self.polygons = [list(p) for p in polygons]
        self.calls: list[tuple[str, int, str]] = []

    def generate(
        self,
        image_path: str,
        bboxes: list[Any],
        mode: str = "box",
    ) -> list[Any]:
        self.calls.append((image_path, len(bboxes), mode))
        # 预设 polygon 不足时补全图矩形（不抛，测试按需断言）
        n = len(bboxes)
        polys = self.polygons[:n] + [
            [0.0, 0.0, float(1242), 0.0, float(1242), float(375), 0.0, float(375)]
        ] * max(0, n - len(self.polygons))
        return [FakeMask(p) for p in polys]


class FakeDetector3D:
    """LiDAR 3D 检测器 Fake（Detector3D Protocol duck-typed，零真实权重铁律）。

    按 conf_threshold 过滤返回预设 Det3DResult，记录每次调用 (frame_id, conf)。
    """

    def __init__(self, dets: list[Det3DResult]) -> None:
        self.dets = list(dets)
        self.calls: list[tuple[str, float | None]] = []
        self.batch_calls: list[tuple[list[str], float | None]] = []
        self.class_names: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")

    def detect(
        self, frame: Any, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        self.calls.append((frame.frame_id, conf_threshold))
        threshold = conf_threshold if conf_threshold is not None else 0.0
        return [d for d in self.dets if d.confidence >= threshold]

    def detect_batch(
        self, frames: list[Any], conf_threshold: float | None = None
    ) -> list[list[Det3DResult]]:
        """批量接口（parity 双接口断言铁律）：记录整批调用，逐帧 conf 过滤。"""
        self.batch_calls.append(([f.frame_id for f in frames], conf_threshold))
        threshold = conf_threshold if conf_threshold is not None else 0.0
        return [[d for d in self.dets if d.confidence >= threshold] for _ in frames]


class FakeLLM:
    """脚本化 LLM：按队列返回响应（content/tool_calls），记录每轮 tools。"""

    model = "fake-llm"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.tools_seen: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    **kwargs: Any,
    ) -> LLMResponse:
        self.tools_seen.append(list(tools or []))
        assert self.responses, "LLM 脚本耗尽"
        r = self.responses.pop(0)
        return LLMResponse(content=r.get("content"), tool_calls=r.get("tool_calls"))


def det(
    x: float, y: float, width: float, height: float, label: str, confidence: float,
) -> DetectionResult:
    return DetectionResult(x=x, y=y, width=width, height=height, label=label, confidence=confidence)


def det3d(
    label: str = "Car",
    confidence: float = 0.9,
    bbox: tuple[float, ...] = (8.0, -0.9, 18.0, 3.9, 1.5, 1.6, 0.0),
) -> Det3DResult:
    """快速构造 Det3DResult（bbox = [x,y,z,l,h,w,ry] 底面中心相机系）。"""
    return Det3DResult(label=label, confidence=confidence, bbox=list(bbox))


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    from auto2dlabel.tests import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
