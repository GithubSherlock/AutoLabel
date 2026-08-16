"""代码级质量评估单元测试 — 0 框重试 / 类别覆盖 / 默认模型一致性。

全部 mock 推理，不加载权重。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from auto2dlabel.agent.evaluate import evaluate_detections
from auto2dlabel.agent.llm import LLMResponse
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.models.detection import (
    DetectionResult,
    UltralyticsModel,
    create_detection_model,
    detect_with_retry,
)
from auto2dlabel.schema.task_plan import DEFAULT_MODEL
from auto2dlabel.tools.detection import DetectionTool
from auto2dlabel.tools.registry import ToolRegistry


def _det(label: str, conf: float = 0.9) -> DetectionResult:
    return DetectionResult(x=0, y=0, width=10, height=10, label=label, confidence=conf)


class MockModel:
    """mock 检测模型：按调用次数依次返回预设结果。"""

    def __init__(self, results: list[list[DetectionResult]] | None = None) -> None:
        self._results = results or []
        self.calls: list[float] = []

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        self.calls.append(confidence_threshold)
        if not self._results:
            return []
        idx = min(len(self.calls), len(self._results)) - 1
        return self._results[idx]


class TestEvaluateDetections:
    def test_zero_boxes_warns(self) -> None:
        q = evaluate_detections([], ["car"], image_path="a.png")
        assert "0 框" in q.warnings
        assert q.total_boxes == 0
        assert q.missing_prompts == ["car"]
        assert not q.ok

    def test_missing_prompt(self) -> None:
        q = evaluate_detections([_det("car")], ["car", "person"])
        assert q.covered_prompts == ["car"]
        assert q.missing_prompts == ["person"]
        assert not q.ok

    def test_too_many_boxes_warns(self) -> None:
        q = evaluate_detections([_det("car")] * 201, ["car"], max_box_warning=200)
        assert any("201" in w for w in q.warnings)
        assert not q.ok

    def test_normal_ok(self) -> None:
        q = evaluate_detections([_det("car"), _det("person")], ["car", "person"])
        assert q.ok
        assert not q.warnings
        assert q.summary_line() == ""

    def test_sahi_dict_input(self) -> None:
        q = evaluate_detections(
            [{"name": "car", "bbox": [0, 0, 10, 10], "conf": 0.9}],
            ["car"],
        )
        assert q.covered_prompts == ["car"]
        assert q.ok

    def test_bbox_input(self) -> None:
        from auto2dlabel.schema.annotation import Bbox

        q = evaluate_detections(
            [Bbox(x=0, y=0, width=10, height=10, label="car", confidence=0.9)],
            ["car"],
        )
        assert q.ok

    def test_empty_prompts(self) -> None:
        q = evaluate_detections([_det("car")], [])
        assert q.missing_prompts == []
        assert q.covered_prompts == []
        assert q.ok

    def test_summary_line_retry_zero(self) -> None:
        q = evaluate_detections([], ["car"], retried=True, retry_threshold=0.15)
        assert "已降阈值重试" in q.summary_line()
        assert "0 框" in q.summary_line()

    def test_to_dict(self) -> None:
        q = evaluate_detections([_det("car")], ["car"], image_path="a.png")
        d = q.to_dict()
        assert d["total_boxes"] == 1
        assert d["image_path"] == "a.png"
        assert d["missing_prompts"] == []


class TestDetectWithRetry:
    def test_retry_on_empty(self) -> None:
        calls: list[float] = []

        def fn(conf: float) -> list[Any]:
            calls.append(conf)
            return [] if len(calls) == 1 else [_det("car", conf)]

        dets, retried, final = detect_with_retry(fn, 0.3)
        assert retried is True
        assert final == pytest.approx(0.15)
        assert calls == pytest.approx([0.3, 0.15])
        assert len(dets) == 1

    def test_no_retry_when_boxes(self) -> None:
        calls: list[float] = []

        def fn(conf: float) -> list[Any]:
            calls.append(conf)
            return [_det("car", conf)]

        dets, retried, final = detect_with_retry(fn, 0.3)
        assert retried is False
        assert final == 0.3
        assert len(calls) == 1

    def test_double_empty_retries_once(self) -> None:
        calls: list[float] = []

        def fn(conf: float) -> list[Any]:
            calls.append(conf)
            return []

        dets, retried, _final = detect_with_retry(fn, 0.3)
        assert retried is True
        assert dets == []
        assert len(calls) == 2  # 只重试一次，不无限重试


class TestDetectionToolRetry:
    def test_forward_retries(self) -> None:
        model = MockModel(results=[[], [_det("car", 0.2)]])
        tool = DetectionTool(model=model)
        boxes = tool.forward("a.png", ["car"], confidence_threshold=0.3)
        assert len(boxes) == 1
        assert tool.last_retried is True
        assert tool.last_retry_threshold == pytest.approx(0.15)
        assert model.calls == pytest.approx([0.3, 0.15])

    def test_forward_no_retry_when_boxes(self) -> None:
        model = MockModel(results=[[_det("car", 0.9)]])
        tool = DetectionTool(model=model)
        boxes = tool.forward("a.png", ["car"], confidence_threshold=0.3)
        assert len(boxes) == 1
        assert tool.last_retried is False
        assert len(model.calls) == 1

    def test_forward_sahi_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[float] = []
        sahi_results: list[list[dict[str, Any]]] = [
            [],
            [{"name": "car", "bbox": [0, 0, 10, 10], "conf": 0.2}],
        ]

        def fake_sahi(
            model: Any,
            image_path: str,
            prompts: list[str],
            confidence_threshold: float,
        ) -> list[dict[str, Any]]:
            calls.append(confidence_threshold)
            return sahi_results[len(calls) - 1]

        monkeypatch.setattr(
            "auto2dlabel.benchmarks.common.detect_image_sahi", fake_sahi
        )
        tool = DetectionTool(model=MockModel(), use_sahi=True)
        boxes = tool.forward("a.png", ["car"], confidence_threshold=0.3)
        assert len(boxes) == 1
        assert tool.last_retried is True
        assert calls == pytest.approx([0.3, 0.15])


class TestOrchestratorQualityEval:
    """回归：质量评估接入 Agent Loop。

    外部预置 registry 场景——此前 _detect_tool 为 None 时 retried 状态丢失。
    """

    def test_quality_report_in_metadata(self) -> None:
        class MockLLM:
            model = "mock"

            def __init__(self) -> None:
                self.n_calls = 0

            def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                temperature: float = 0.1,
            ) -> LLMResponse:
                self.n_calls += 1
                if self.n_calls == 1:
                    return LLMResponse(tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "detect_objects",
                            "arguments": json.dumps(
                                {"prompts": ["car"], "confidence_threshold": 0.3}
                            ),
                        },
                    }])
                return LLMResponse(content="car: 1")

        registry = ToolRegistry()
        model = MockModel(results=[[], [_det("car", 0.2)]])
        registry.register(DetectionTool(model=model))
        orch = AgentOrchestrator(
            llm_client=MockLLM(), tool_registry=registry  # type: ignore[arg-type]
        )
        state = orch.run("fake.png", "检测汽车", confidence_threshold=0.3)

        qr = state.metadata.get("quality_report")
        assert qr is not None
        assert qr["retried"] is True
        assert qr["retry_threshold"] == pytest.approx(0.15)
        assert qr["missing_prompts"] == []
        # 重试在 tool 内部完成，Agent Loop 迭代数不回退
        assert state.iteration <= 2
        assert state.done is True
        # quality 进 tool 摘要（LLM 可见）
        assert "quality" in state.tool_calls[-1]["result"]


class TestDefaultModelConsistency:
    def test_default_model_value(self) -> None:
        assert DEFAULT_MODEL == "yolo26x.pt"

    def test_factory_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DETECTION_MODEL", raising=False)
        model = create_detection_model(None)
        assert isinstance(model, UltralyticsModel)
        assert model._model_name == DEFAULT_MODEL

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DETECTION_MODEL", "yolov8n.pt")
        model = create_detection_model(None)
        assert isinstance(model, UltralyticsModel)
        assert model._model_name == "yolov8n.pt"

    def test_planner_prompt_mentions_default(self) -> None:
        from auto2dlabel.agent import planner

        assert "yolo26x.pt" in planner._PLANNER_SYSTEM_PROMPT

    def test_visualize_default(self) -> None:
        from auto2dlabel.tools.visualize import detect_and_visualize

        defaults = detect_and_visualize.__defaults__
        assert defaults is not None and DEFAULT_MODEL in defaults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
