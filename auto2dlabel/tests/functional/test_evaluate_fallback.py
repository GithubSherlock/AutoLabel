"""LLM Evaluate 代码级降级测试（v0.6 Phase 4，零真实权重）。

覆盖：deterministic_disposition 纯函数规则；evaluate_mode="code" 直走规则
（零 evaluate_quality 暴露、0 框自动降阈值重检、缺类转 review）；
"auto" 下 LLM Evaluate 失败 → 确定性规则兜底不中断；mode 解析链
（构造参数 > env AUTOLABEL_EVALUATE_MODE > auto，非法值回退）。
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from auto2dlabel.agent.evaluate import (
    QualityReport,
    deterministic_disposition,
)
from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import json
from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry

# ============ 规则纯函数 ============


def test_disposition_zero_boxes_not_retried() -> None:
    """0 框且未降阈值重试过 → retry_lower_threshold（一次重检机会）。"""
    report = QualityReport(total_boxes=0, retried=False)
    assert deterministic_disposition(report) == "retry_lower_threshold"


def test_disposition_zero_boxes_retried() -> None:
    """已重试仍 0 框 → flag_for_review（宁缺勿假，不无限重试）。"""
    report = QualityReport(total_boxes=0, retried=True)
    assert deterministic_disposition(report) == "flag_for_review"


def test_disposition_missing_prompts() -> None:
    """有框但缺类 → flag_for_review（非 0 框不触发重检）。"""
    report = QualityReport(total_boxes=3, missing_prompts=["person"])
    assert deterministic_disposition(report) == "flag_for_review"


def test_disposition_warnings() -> None:
    """超框数警告 → flag_for_review。"""
    report = QualityReport(total_boxes=300, warnings=["框数异常"])
    assert deterministic_disposition(report) == "flag_for_review"


# ============ 编排器集成 ============


class _FakeDetectTool(Tool):
    """按队列返回检测结果，记录每次调用阈值；可选第 N 次调用抛异常。"""

    name = "detect_objects"
    description = "fake detection for tests"

    def __init__(self, results: list[list[Bbox]], raise_at: int | None = None) -> None:
        self.results = list(results)
        self.raise_at = raise_at  # 第 N 次调用（1-based）抛异常
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
        if self.raise_at is not None and len(self.calls) >= self.raise_at:
            raise RuntimeError("detect boom")
        return self.results.pop(0)


class _ScriptedLLM:
    """按固定序列返回响应，并记录每轮收到的 tools。"""

    model = "fake-llm"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.tools_seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.1, **kwargs: Any) -> LLMResponse:
        self.tools_seen.append(list(tools or []))
        assert self.responses, "LLM 脚本耗尽"
        r = self.responses.pop(0)
        return LLMResponse(content=r.get("content"), tool_calls=r.get("tool_calls"))


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


_DETECT_ARGS = {"image_path": "a.jpg", "prompts": ["car"], "confidence_threshold": 0.3}


def _make_orchestrator(
    llm: _ScriptedLLM,
    detect: _FakeDetectTool,
    evaluate_mode: str | None = None,
) -> AgentOrchestrator:
    reg = ToolRegistry()  # 全新实例，避免全局单例污染
    reg.register(detect)
    orch = AgentOrchestrator(
        llm_client=cast(LLMClient, llm),
        tool_registry=reg,
        max_iterations=3,
        evaluate_mode=evaluate_mode,
    )
    orch._detect_tool = detect
    return orch


def test_code_mode_zero_boxes_auto_retry() -> None:
    """evaluate_mode="code" + 0 框 → 自动降阈值重检（LLM_RETRY_FACTOR），零工具暴露。"""
    detect = _FakeDetectTool([
        [],  # 第一次 detect_objects: 0 框
        [Bbox(x=10, y=20, width=30, height=40, label="car", confidence=0.05)],
    ])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect, evaluate_mode="code").run(
        "a.jpg", "检测 car", confidence_threshold=0.3
    )

    assert detect.calls[1][1] == pytest.approx(0.3 * 0.25)  # 代码级降阈值重检
    for tools in llm.tools_seen:
        assert all(t["function"]["name"] != "evaluate_quality" for t in tools)
    assert len(state.annotations) == 1
    assert len(state.annotations[0].bboxes) == 1  # 重检结果并入
    assert "llm_review_flagged" not in state.metadata  # 走了 retry，非 flag


def test_code_mode_missing_prompt_flags_review() -> None:
    """evaluate_mode="code" + 缺类 → 直接 flag_for_review，零重检、零工具暴露。"""
    detect = _FakeDetectTool([
        [Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.9)],
    ])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call(
            "detect_objects",
            {"image_path": "a.jpg", "prompts": ["car", "person"], "confidence_threshold": 0.3},
            "c1",
        )]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect, evaluate_mode="code").run(
        "a.jpg", "检测 car person", confidence_threshold=0.3
    )

    assert len(detect.calls) == 1  # 非 0 框 → 无重检
    assert state.metadata["llm_review_flagged"] is True
    for tools in llm.tools_seen:
        assert all(t["function"]["name"] != "evaluate_quality" for t in tools)
    assert state.iteration == 2  # detect → 总结（代码处置不追加迭代）


def test_auto_mode_llm_evaluate_failure_falls_back() -> None:
    """"auto" 下 LLM Evaluate 执行失败 → 代码级规则兜底，流程不中断。"""
    # 第 2 次调用（evaluate 的 retry detect）起抛异常：LLM 处置与代码兜底都失败
    detect = _FakeDetectTool([[]], raise_at=2)
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("evaluate_quality",
                                   {"action": "retry_lower_threshold"}, "c2")]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert state.metadata["llm_review_flagged"] is True  # 兜底 flag_for_review
    assert state.iteration == 3  # 流程不中断，正常走完
    # tool 结果带 fallback 标记（可诊断）
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    fallback_msg = next(
        m for m in tool_msgs if isinstance(m.get("content"), str)
        and "fallback" in m.get("content", "")
    )
    assert "fallback" in fallback_msg["content"]


def test_evaluate_without_pending_skipped_gracefully() -> None:
    """防御：质量通过（无挂起资产）时 LLM 误调 evaluate_quality → 跳过不崩。"""
    detect = _FakeDetectTool([
        [Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.9)],
    ])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"tool_calls": [_tool_call("evaluate_quality", {"action": "accept"}, "c2")]},
        {"content": "完成"},
    ])
    state = _make_orchestrator(llm, detect).run("a.jpg", "检测 car", confidence_threshold=0.3)

    assert state.iteration == 3
    assert "llm_review_flagged" not in state.metadata
    tool_msgs = [str(m.get("content")) for m in state.messages if m.get("role") == "tool"]
    assert any("skipped" in c for c in tool_msgs)


def test_mode_resolution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """解析链：构造参数 None + env AUTOLABEL_EVALUATE_MODE → 生效。"""
    monkeypatch.setenv("AUTOLABEL_EVALUATE_MODE", "code")
    detect = _FakeDetectTool([[]])
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("detect_objects", _DETECT_ARGS, "c1")]},
        {"content": "完成"},
    ])
    orch = _make_orchestrator(llm, detect, evaluate_mode=None)
    assert orch.evaluate_mode == "code"


def test_mode_resolution_param_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造参数优先于 env。"""
    monkeypatch.setenv("AUTOLABEL_EVALUATE_MODE", "code")
    detect = _FakeDetectTool([[]])
    llm = _ScriptedLLM([{"content": "完成"}])
    orch = _make_orchestrator(llm, detect, evaluate_mode="auto")
    assert orch.evaluate_mode == "auto"


def test_mode_resolution_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """非法值 → 回退 auto（不静默启用错误模式）。"""
    monkeypatch.setenv("AUTOLABEL_EVALUATE_MODE", "bogus")
    detect = _FakeDetectTool([[]])
    llm = _ScriptedLLM([{"content": "完成"}])
    orch = _make_orchestrator(llm, detect, evaluate_mode=None)
    assert orch.evaluate_mode == "auto"
