"""Agent Loop LLM Evaluate 节点集成测试（脚本化 LLM + Fake 检测 Tool，零模型加载）。

覆盖：质量通过时零暴露；质量未通过时条件暴露 evaluate_quality；
flag 写 metadata、retry 结果并入 _sync_annotations、max_iterations=3 红线、
同批重复调用只执行一次、重复调用消息配对（DeepSeek 400 回归）。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.schema.annotation import Bbox
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
    """按固定序列返回响应，并记录每轮收到的 tools。"""

    model = "fake-llm"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.tools_seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.1) -> LLMResponse:
        self.tools_seen.append(list(tools or []))
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


def test_quality_ok_no_evaluate_tool_exposed() -> None:
    """检测正常（1 框覆盖类别）→ 任何一轮 tools 都不含 evaluate_quality。"""
    detect = _FakeDetectTool([[Bbox(x=1, y=2, width=3, height=4, label="car",
                                   confidence=0.9)]])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"content": "完成: car × 1"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert len(state.annotations) == 1
    assert state.annotations[0].bboxes[0].label == "car"
    for tools in llm.tools_seen:
        assert all(t["function"]["name"] != "evaluate_quality" for t in tools)


def test_zero_boxes_exposes_evaluate_and_flag() -> None:
    """0 框 → 第 2 轮暴露 evaluate_quality；LLM 选 flag_for_review → metadata 标记。"""
    detect = _FakeDetectTool([[]])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("evaluate_quality", {"action": "flag_for_review"}, "c2")]},
        {"content": "已标记复核"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    names = [t["function"]["name"] for t in llm.tools_seen[1]]
    assert "evaluate_quality" in names  # 条件暴露
    assert state.metadata["llm_review_flagged"] is True
    assert state.iteration == 3  # iter1 detect → iter2 evaluate → iter3 总结（红线内）


def test_retry_lower_threshold_merges_annotations() -> None:
    """0 框 → LLM 选 retry_lower_threshold → 降阈值检测结果并入 annotations。"""
    detect = _FakeDetectTool([
        [],  # 第一次 detect_objects: 0 框
        [Bbox(x=10, y=20, width=30, height=40, label="car", confidence=0.05)],
    ])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("evaluate_quality",
                                   {"action": "retry_lower_threshold"}, "c2")]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert detect.calls[1][1] == 0.3 * 0.25  # LLM_RETRY_FACTOR 降阈值
    assert len(state.annotations) == 1
    assert len(state.annotations[0].bboxes) == 1
    bbox = state.annotations[0].bboxes[0]
    assert bbox.label == "car" and bbox.confidence == 0.05
    assert 0 in state.annotations[0].review_flags  # 低于阈值 → 复核标记
    assert state.iteration == 3  # max_iterations 红线


def test_duplicate_evaluate_in_same_batch_runs_once() -> None:
    """同批两次 evaluate_quality → 只执行一次，第二次跳过。"""
    detect = _FakeDetectTool([[]])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [
            _tool_call("evaluate_quality", {"action": "flag_for_review"}, "c2"),
            _tool_call("evaluate_quality", {"action": "retry_lower_threshold"}, "c3"),
        ]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert state.metadata["llm_review_flagged"] is True
    assert detect.calls == [("a.jpg", 0.3)]  # retry 未执行（重复调用被跳过）


def test_duplicate_detect_keeps_tool_result_paired() -> None:
    """重复 detect_objects：assistant 消息保留完整 tool_calls，tool 结果紧随配对。

    回归（2026-08-28 chat 实测）：曾从 assistant 消息删除重复调用条目 → 孤儿
    tool 消息（前置无对应 tool_calls）→ DeepSeek 严格 API 下一轮请求 400。
    """
    detect = _FakeDetectTool([
        [Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.9)],
        [Bbox(x=5, y=6, width=7, height=8, label="car", confidence=0.9)],
    ])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c2")]},
        {"content": "完成: car × 1"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert detect.calls == [("a.jpg", 0.3)]  # 重复调用未执行（防重复红线）
    for i, m in enumerate(state.messages):
        if m.get("role") != "tool":
            continue
        prev = state.messages[i - 1]
        assert prev.get("role") == "assistant", "tool 结果必须紧随 assistant 消息"
        ids = [tc["id"] for tc in prev.get("tool_calls", [])]
        assert m["tool_call_id"] in ids, "tool 结果必须有前置 tool_calls 配对条目"
    assert state.iteration == 3  # detect → 重复跳过 → 总结（红线内）
