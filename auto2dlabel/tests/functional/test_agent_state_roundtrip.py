"""AgentState 序列化 round-trip 与续跑测试（v0.4 Phase 3，零模型加载）。

覆盖：全字段 round-trip（含 max_iterations）、annotations 完整恢复
（angle/track_id/labels/review_flags）、旧快照向后兼容默认值、
orchestrator initial_state 续跑（不重复检测 + 重复调用跳过）。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.agent.state import AgentState
from auto2dlabel.schema.annotation import Annotation, Bbox, ImageLabel
from auto2dlabel.tests import json
from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry


class _FakeDetectTool(Tool):
    """按队列返回检测结果，记录每次调用阈值。"""

    name = "detect_objects"
    description = "fake detection for tests"

    def __init__(self, results: list[list[Bbox]]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, float]] = []
        self.last_retried = False
        self.last_retry_threshold: float | None = None

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


def _full_state() -> AgentState:
    """构造含全部字段与 annotation 细节的 state（angle/track_id/labels/review_flags）。"""
    state = AgentState(
        image_path="a.jpg",
        user_instruction="检测汽车",
        confidence_threshold=0.25,
        max_iterations=5,
    )
    state.add_message("system", state.system_prompt)
    state.add_message("user", "Image: a.jpg")
    state.tool_calls.append({"tool_name": "detect_objects", "tool_call_id": "c1",
                             "result": {"success": True, "count": 1}})
    state.iteration = 3
    state.done = True
    state.metadata["quality_report"] = {"ok": True}
    ann = Annotation(image_path="a.jpg", image_size=(640, 480))
    b = Bbox(x=10.0, y=20.0, width=30.0, height=40.0, label="car",
             confidence=0.9, angle=0.5, track_id=3)
    ann.add_bbox(b)
    assert b.id is not None
    ann.flag_for_review(b.id)
    ann.labels = [ImageLabel(label="car", score=0.9)]
    state.annotations.append(ann)
    return state


# ---------- 纯 round-trip ----------

def test_roundtrip_all_fields() -> None:
    """全字段 round-trip 相等（含 max_iterations/iteration/done/messages/tool_calls）。"""
    s = _full_state()
    s2 = AgentState.from_dict(json.loads(json.dumps(s.to_dict())))
    assert s2.image_path == s.image_path
    assert s2.user_instruction == s.user_instruction
    assert s2.confidence_threshold == 0.25
    assert s2.max_iterations == 5
    assert s2.iteration == 3
    assert s2.done is True
    assert s2.messages == s.messages
    assert s2.tool_calls == s.tool_calls
    assert s2.metadata == s.metadata


def test_roundtrip_annotations_full() -> None:
    """annotations 完整恢复：angle/track_id/labels/review_flags 不丢。"""
    s = _full_state()
    s2 = AgentState.from_dict(json.loads(json.dumps(s.to_dict())))
    assert len(s2.annotations) == 1
    ann = s2.annotations[0]
    assert ann.image_size == (640, 480)
    assert len(ann.bboxes) == 1
    b = ann.bboxes[0]
    assert (b.x, b.y, b.width, b.height) == (10.0, 20.0, 30.0, 40.0)
    assert b.label == "car" and b.confidence == 0.9
    assert b.angle == 0.5
    assert b.track_id == 3
    assert ann.review_flags == [0]
    assert ann.labels == [ImageLabel(label="car", score=0.9)]


def test_roundtrip_defaults() -> None:
    """空 dict → 全部默认值（max_iterations=10 与构造默认一致）。"""
    s = AgentState.from_dict({})
    assert s.image_path == ""
    assert s.user_instruction == ""
    assert s.confidence_threshold == 0.3
    assert s.max_iterations == 10
    assert s.annotations == [] and s.messages == [] and s.tool_calls == []


def test_roundtrip_legacy_snapshot_no_max_iterations() -> None:
    """旧快照（无 max_iterations 字段）→ 默认 10，向后兼容。"""
    legacy = _full_state().to_dict()
    legacy.pop("max_iterations")
    s = AgentState.from_dict(legacy)
    assert s.max_iterations == 10


# ---------- orchestrator 续跑 ----------

def _first_run() -> tuple[AgentState, _FakeDetectTool, _ScriptedLLM]:
    """跑一轮完整 Agent Loop（detect → 总结）得可序列化 state。"""
    detect = _FakeDetectTool([[Bbox(x=1, y=2, width=3, height=4, label="car",
                                   confidence=0.9)]])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"content": "完成: car × 1"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)
    return state, detect, llm


def test_orchestrator_resume_continues_without_redetect() -> None:
    """快照续跑：跳过已完成迭代直接总结，detect 不重复调用，annotations 保留。"""
    state, detect, _ = _first_run()
    assert state.iteration == 2
    assert len(state.annotations[0].bboxes) == 1
    state.done = False  # 模拟中断快照（失败时循环未走完，done=False）

    resumed = AgentState.from_dict(json.loads(json.dumps(state.to_dict())))
    llm2 = _ScriptedLLM([{"content": "总结: car × 1"}])
    out = _make_orchestrator(llm2, detect).run(
        "a.jpg", "检测 car", confidence_threshold=0.3, initial_state=resumed
    )

    assert detect.calls == [("a.jpg", 0.3)]  # 续跑不重复调用检测
    assert out.iteration == 3  # 在恢复点继续（2 → 3）
    assert len(out.annotations[0].bboxes) == 1  # 标注保留


def test_orchestrator_resume_skips_duplicate_detect_call() -> None:
    """续跑时 LLM 再次调 detect → 被防重复红线跳过（记录 skipped 结果）。"""
    state, detect, _ = _first_run()
    state.done = False  # 模拟中断快照
    resumed = AgentState.from_dict(json.loads(json.dumps(state.to_dict())))
    llm2 = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c2")]},
        {"content": "总结: car × 1"},
    ])
    out = _make_orchestrator(llm2, detect).run(
        "a.jpg", "检测 car", confidence_threshold=0.3, initial_state=resumed
    )

    assert detect.calls == [("a.jpg", 0.3)]  # 第二次调用被跳过
    skipped = [tc for tc in out.tool_calls if tc["result"].get("skipped")]
    assert len(skipped) == 1
    assert len(out.annotations[0].bboxes) == 1


def test_orchestrator_resume_done_state_short_circuits() -> None:
    """done=True 的恢复状态直接返回，不进入循环。"""
    state, detect, _ = _first_run()
    state.done = True
    resumed = AgentState.from_dict(json.loads(json.dumps(state.to_dict())))
    llm2 = _ScriptedLLM([])  # 空脚本：若进入循环会断言失败
    out = _make_orchestrator(llm2, detect).run(
        "a.jpg", "检测 car", confidence_threshold=0.3, initial_state=resumed
    )
    assert out is resumed
