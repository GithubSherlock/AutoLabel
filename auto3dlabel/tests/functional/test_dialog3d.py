"""v0.3 P1 对话式 Planner（3D）测试（脚本化 FakeLLM，零 mmdet3d/零网络）。

覆盖：对话闭环（frame_id/prompts 问出）、questions 空直行、轮次耗尽、
ask 放弃、非法响应降级传播、_dict_to_plan3d 读取 questions（致命回归：
LLM JSON → Plan3D 不经 TaskPlan.from_dict）。
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.schema.task_plan import PlanQuestion
from auto2dlabel.tests import json
from auto3dlabel.agent.planner3d import TaskPlanner3D, _dict_to_plan3d
from auto3dlabel.tests.helpers.fakes import FakeLLM


class _ScriptedLLM(FakeLLM):
    """FakeLLM + 记录完整 messages（本地子类，不改共享 fakes.py 契约）。"""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(responses)
        self.messages_seen: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        self.messages_seen.append(messages)
        return super().chat(messages, tools=tools, temperature=temperature)


def _plan3d_json(
    frame_id: str = "",
    prompts: list[str] | None = None,
    questions: Any = None,
) -> str:
    """构造 Plan3D JSON（questions 为 None 时省略键）。"""
    data: dict[str, Any] = {
        "frame_id": frame_id,
        "prompts": prompts or [],
        "confidence_threshold": 0.3,
        "det_model": "IDEA-Research/grounding-dino-tiny",
        "seg_model": "sam2_l.pt",
    }
    if questions is not None:
        data["questions"] = questions
    return json.dumps(data)


def _make_planner(responses: list[dict[str, Any]]) -> tuple[TaskPlanner3D, _ScriptedLLM]:
    llm = _ScriptedLLM(responses)
    return TaskPlanner3D(llm_client=cast(LLMClient, llm)), llm


# ── 对话闭环 ────────────────────────────────────────────────

def test_dialog3d_roundtrip_collects_answers() -> None:
    """缺 frame_id → 问出 → 第 2 轮补全 prompts；回喂文本带 id。"""
    planner, llm = _make_planner([
        {"content": _plan3d_json(questions=[{"id": "frame_id", "question": "哪个帧？"}])},
        {"content": _plan3d_json(frame_id="000123", prompts=["car"])},
    ])
    asked: list[list[PlanQuestion]] = []

    def ask(questions: list[PlanQuestion]) -> str:
        asked.append(list(questions))
        return "000123，检测汽车"

    plan = planner.parse_dialog("标注 KITTI 帧中的汽车", ask_fn=ask)

    assert len(llm.messages_seen) == 2
    fed = llm.messages_seen[1][-1]["content"]
    assert "[frame_id]" in fed and "000123，检测汽车" in fed
    assert asked == [[PlanQuestion(id="frame_id", question="哪个帧？")]]
    assert plan.frame_id == "000123"
    assert plan.prompts == ["car"]


def test_dialog3d_no_questions_single_round() -> None:
    """完整指令 → questions 空 → 1 次解析直接返回，ask 不触发。"""
    planner, llm = _make_planner([
        {"content": _plan3d_json(frame_id="000123", prompts=["car"])},
    ])

    def ask(questions: list[PlanQuestion]) -> str:  # pragma: no cover - 不应触发
        raise AssertionError("ask 不应被调用")

    plan = planner.parse_dialog("标注 KITTI 帧 000123 中的汽车", ask_fn=ask)

    assert len(llm.messages_seen) == 1
    assert plan.frame_id == "000123"


def test_dialog3d_max_rounds_exhaustion() -> None:
    """恒返回 questions + 恒回答 → 3 次解析后返回缺参 plan（CLI BadParameter 兜底）。"""
    q = [{"id": "prompts", "question": "类别？"}]
    planner, llm = _make_planner([
        {"content": _plan3d_json(frame_id="000123", questions=q)},
        {"content": _plan3d_json(frame_id="000123", questions=q)},
        {"content": _plan3d_json(frame_id="000123", questions=q)},
    ])
    asks: list[int] = []

    def ask(questions: list[PlanQuestion]) -> str:
        asks.append(len(questions))
        return "car"

    plan = planner.parse_dialog("标注帧 000123", ask_fn=ask)

    assert len(llm.messages_seen) == 3
    assert len(asks) == 2
    assert plan.prompts == []  # 末轮缺参返回 → 代码兜底


def test_dialog3d_ask_abandon_returns_incomplete() -> None:
    """用户放弃（ask 返回空白）→ 1 轮后返回缺参 plan。"""
    planner, llm = _make_planner([
        {"content": _plan3d_json(questions=[{"id": "frame_id", "question": "哪个帧？"}])},
        {"content": _plan3d_json(frame_id="000123")},  # 不应被消费
    ])

    def ask(questions: list[PlanQuestion]) -> str:
        return ""

    plan = planner.parse_dialog("标注帧", ask_fn=ask)

    assert len(llm.messages_seen) == 1
    assert plan.frame_id == ""


# ── 异常与解析 ──────────────────────────────────────────────

def test_dialog3d_parse_error_propagates() -> None:
    """LLM 非法响应 → parse_plan3d_json ValueError 传播（CLI 降级链处理）。"""
    planner, _ = _make_planner([{"content": "not json at all"}])
    with pytest.raises(ValueError):
        planner.parse_dialog("标注帧", ask_fn=lambda q: "x")


def test_dict_to_plan3d_reads_questions() -> None:
    """致命回归：LLM JSON → Plan3D 走 _dict_to_plan3d（不经 TaskPlan.from_dict）。"""
    plan = _dict_to_plan3d(json.loads(_plan3d_json(
        questions=[{"id": "frame_id", "question": "哪个帧？"}, {"id": 7, "question": "类别？"}],
    )))
    assert plan.questions == [
        PlanQuestion(id="frame_id", question="哪个帧？"),
        PlanQuestion(id="7", question="类别？"),
    ]
    # 无 questions 键 → 兼容旧 JSON
    assert _dict_to_plan3d(json.loads(_plan3d_json(frame_id="000123"))).questions == []
