"""tool_calls arguments 规整化测试（v0.6 Phase 4，零真实权重）。

覆盖：_normalize_tool_calls 纯函数（紧凑化/非法保持原样/原对象不变/字段保留）
+ 编排器集成零行为回归（带冗余空白的 arguments 经状态回传后紧凑化，
执行仍用原值解析，行为不变）。
"""

from __future__ import annotations

from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.orchestrator import (
    AgentOrchestrator,
    _normalize_tool_calls,
)
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import json
from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry


def test_normalize_compacts_whitespace() -> None:
    """带冗余空白/换行的 arguments → 紧凑 JSON（键序保留、语义不变）。"""
    args = '{\n  "image_path": "a.jpg",\n  "prompts": ["car"],\n  "confidence_threshold": 0.3\n}'
    normalized = _normalize_tool_calls([
        {"id": "c1", "type": "function", "function": {"name": "detect_objects", "arguments": args}},
    ])
    out = normalized[0]["function"]["arguments"]
    assert out == '{"image_path":"a.jpg","prompts":["car"],"confidence_threshold":0.3}'
    assert json.loads(out) == json.loads(args)  # 语义不变


def test_normalize_invalid_json_kept_as_is() -> None:
    """非法 JSON arguments → 原样保留（执行处 json.loads 再报错，行为不变）。"""
    normalized = _normalize_tool_calls([
        {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "not-json"}},
    ])
    assert normalized[0]["function"]["arguments"] == "not-json"


def test_normalize_does_not_mutate_original() -> None:
    """返回新列表/新 dict，原 tool_calls 对象不变（执行循环仍用原值）。"""
    original: list[dict[str, Any]] = [
        {"id": "c1", "type": "function",
         "function": {"name": "x", "arguments": '{"a": 1, "b": 2}'}},
    ]
    normalized = _normalize_tool_calls(original)
    assert normalized is not original
    assert original[0]["function"]["arguments"] == '{"a": 1, "b": 2}'
    assert normalized[0]["function"]["arguments"] == '{"a":1,"b":2}'


def test_normalize_preserves_extra_fields() -> None:
    """id/type/function 之外字段与 function 子键完整保留。"""
    normalized = _normalize_tool_calls([
        {"id": "c1", "type": "function", "extra": 42,
         "function": {"name": "x", "arguments": "{}", "custom": True}},
    ])
    out = normalized[0]
    assert out["id"] == "c1" and out["type"] == "function" and out["extra"] == 42
    assert out["function"]["name"] == "x" and out["function"]["custom"] is True


# ============ 编排器集成（零行为回归） ============


class _FakeDetectTool(Tool):
    name = "detect_objects"
    description = "fake detection for tests"

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
        return [Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.9)]


class _ScriptedLLM:
    model = "fake-llm"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.1, **kwargs: Any) -> LLMResponse:
        assert self.responses, "LLM 脚本耗尽"
        r = self.responses.pop(0)
        return LLMResponse(content=r.get("content"), tool_calls=r.get("tool_calls"))


def test_orchestrator_normalizes_arguments_into_state() -> None:
    """集成：冗余空白 arguments 执行无误，入 state 的 assistant 消息已紧凑化。"""
    # LLM 原样写回带换行/缩进的 arguments（模拟真实 LLM 冗余回传）
    messy_args = (
        '{\n  "image_path": "a.jpg",\n  "prompts": ["car"],\n'
        '  "confidence_threshold": 0.3\n}'
    )
    llm = _ScriptedLLM([
        {"tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "detect_objects", "arguments": messy_args}}]},
        {"content": "完成: car × 1"},
    ])
    reg = ToolRegistry()
    reg.register(_FakeDetectTool())
    orch = AgentOrchestrator(
        llm_client=cast(LLMClient, llm), tool_registry=reg, max_iterations=3
    )
    state = orch.run("a.jpg", "检测 car", confidence_threshold=0.3)

    # 执行零回归：检测结果入 annotation
    assert len(state.annotations) == 1
    assert state.annotations[0].bboxes[0].label == "car"

    # state 中 assistant tool_calls 已紧凑化（后续轮次消息体积缩小）
    assistant_msgs = [m for m in state.messages if m.get("role") == "assistant"
                      and m.get("tool_calls")]
    assert len(assistant_msgs) == 1
    stored_args = assistant_msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert stored_args == (
        '{"image_path":"a.jpg","prompts":["car"],"confidence_threshold":0.3}'
    )
