"""3b 模型级重试测试（pick_alternate_model 规则 / retry_swap_model 动作 / orchestrator e2e）。

零真实权重：换模型检测经 monkeypatch create_detection_model → FakeAltModel 注入；
orchestrator 用脚本化 LLM + Fake 检测 Tool（沿用 test_orchestrator_evaluate 模式）。
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from auto2dlabel.agent.evaluate import (
    QualityReport,
    apply_evaluate_action,
    pick_alternate_model,
)
from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.schema.task_plan import DEFAULT_MODEL
from auto2dlabel.tests import json
from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry

# ---------- pick_alternate_model 规则 ----------

@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("yolo26x.pt", "fasterrcnn_resnet50_fpn_v2"),
        ("yolo11n.pt", "fasterrcnn_resnet50_fpn_v2"),
        ("yolo12s.pt", "fasterrcnn_resnet50_fpn_v2"),
        ("fasterrcnn_resnet50_fpn_v2", DEFAULT_MODEL),
        ("fasterrcnn_resnet50_fpn", DEFAULT_MODEL),
        ("IDEA-Research/grounding-dino-tiny", DEFAULT_MODEL),
        ("grounding-dino-base", DEFAULT_MODEL),
        ("some_unknown_model", DEFAULT_MODEL),
    ],
)
def test_pick_alternate_model_rules(current: str, expected: str) -> None:
    """备选模型规则：yolo→高召回 fasterrcnn；fasterrcnn/g-dino/未知→yolo26x.pt。"""
    assert pick_alternate_model(current) == expected


# ---------- apply_evaluate_action: retry_swap_model ----------

def _report() -> QualityReport:
    return QualityReport(
        image_path="a.jpg", total_boxes=0, prompts=["car"], warnings=["0 框"],
    )


def test_apply_swap_calls_swap_fn_once() -> None:
    """retry_swap_model：swap_fn 恰调用一次，结果携带 detections。"""
    calls: list[int] = []
    bbox = Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.8)

    def swap_fn() -> list[Bbox]:
        calls.append(1)
        return [bbox]

    result = apply_evaluate_action(
        "retry_swap_model", report=_report(), retry_used=False,
        base_threshold=0.3, swap_fn=swap_fn,
    )
    assert calls == [1]
    assert result["retried_swap"] is True
    assert result["detections"] == [bbox]


def test_apply_swap_rejected_when_model_retried() -> None:
    """已换过模型（model_retried=True）→ 拒绝重换，转为 accept，swap_fn 不调用。"""

    def swap_fn() -> list[Any]:
        raise AssertionError("不应调用 swap_fn")

    result = apply_evaluate_action(
        "retry_swap_model", report=_report(), retry_used=False,
        base_threshold=0.3, model_retried=True, swap_fn=swap_fn,
    )
    assert result["accepted"] is True
    assert "已换过模型" in result["note"]
    assert "detections" not in result


def test_apply_swap_without_swap_fn_raises() -> None:
    """无 swap_fn（如外部注入 registry 未提供模型）→ ValueError。"""
    with pytest.raises(ValueError, match="swap_fn"):
        apply_evaluate_action(
            "retry_swap_model", report=_report(), retry_used=False, base_threshold=0.3,
        )


def test_evaluate_tool_schema_includes_swap_action() -> None:
    """EvaluateTool schema enum 含 retry_swap_model（LLM 可见）。"""
    from auto2dlabel.tools.evaluate import EvaluateTool

    tool = EvaluateTool(report=_report(), retry_used=False, base_threshold=0.3)
    enum = tool.input_schema["properties"]["action"]["enum"]
    assert "retry_swap_model" in enum


# ---------- orchestrator e2e ----------

class _FakeDetectTool(Tool):
    """按队列返回检测结果（0 框触发质量失败），记录调用。"""

    name = "detect_objects"
    description = "fake detection for tests"

    def __init__(self, results: list[list[Bbox]]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, float]] = []
        self.last_retried = False
        self.last_retry_threshold: float | None = None
        self.use_sahi = False

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompts": {"type": "array", "items": {"type": "string"}},
                "confidence_threshold": {"type": "number"},
            },
            "required": ["image_path", "prompts"],
        }

    def forward(self, **kwargs: Any) -> list[Bbox]:
        image_path = str(kwargs["image_path"])
        confidence_threshold = float(kwargs.get("confidence_threshold", 0.3))
        self.calls.append((image_path, confidence_threshold))
        return self.results.pop(0)


class _FakeAltModel:
    """备选检测模型 Fake（monkeypatch create_detection_model 注入，零权重）。

    results 为每次 detect 调用的返回队列（每个元素是一次调用的完整结果列表）。
    """

    def __init__(self, results: list[list[DetectionResult]]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[str], float]] = []

    def detect(
        self, image_path: str, prompts: list[str], confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        self.calls.append((image_path, list(prompts), confidence_threshold))
        return self.results.pop(0)


class _ScriptedLLM:
    """按固定序列返回响应（耗尽即断言失败）。"""

    model = "fake-llm"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.1, **kwargs: Any) -> LLMResponse:
        assert self.responses, "LLM 脚本耗尽"
        r = self.responses.pop(0)
        return LLMResponse(content=r.get("content"), tool_calls=r.get("tool_calls"))


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def _make_orchestrator(llm: _ScriptedLLM, detect: _FakeDetectTool) -> AgentOrchestrator:
    reg = ToolRegistry()  # 全新实例，避免全局单例污染
    reg.register(detect)
    orch = AgentOrchestrator(llm_client=cast(LLMClient, llm), tool_registry=reg, max_iterations=3)
    orch._detect_tool = detect
    return orch


_DETECT_ARGS = {"image_path": "a.jpg", "prompts": ["car"], "confidence_threshold": 0.3}


def test_orchestrator_swap_model_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 框 → 质量失败 → LLM 选 retry_swap_model → 备选模型同阈值检测一次，
    结果并入 annotations + metadata 记录换模型；检测工具不重复调用。"""
    detect = _FakeDetectTool([[]])  # 首检 0 框 → 触发 Evaluate 节点
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("evaluate_quality", {"action": "retry_swap_model"}, "c2")]},
        {"content": "完成: car × 1"},
    ])
    alt_model = _FakeAltModel([[
        DetectionResult(x=10, y=20, width=30, height=40, label="car", confidence=0.85),
    ]])
    monkeypatch.setattr(
        "auto2dlabel.models.detection.create_detection_model",
        lambda name=None, **kw: alt_model,
    )
    orch = _make_orchestrator(llm, detect)
    orch.detection_model = "yolo11n.pt"

    out = orch.run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert detect.calls == [("a.jpg", 0.3)]  # 检测工具只调一次
    assert alt_model.calls == [("a.jpg", ["car"], 0.3)]  # 备选模型同阈值检测一次
    assert out.metadata["model_swapped"] == "fasterrcnn_resnet50_fpn_v2"
    assert len(out.annotations) == 1
    bboxes = out.annotations[0].bboxes
    assert len(bboxes) == 1
    assert (bboxes[0].x, bboxes[0].y) == (10, 20)
    assert bboxes[0].label == "car"
