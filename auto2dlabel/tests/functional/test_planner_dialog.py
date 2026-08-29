"""v0.6 P1 对话式 Planner 测试（脚本化 LLM + Fake ask_fn，零网络零模型）。

覆盖：对话闭环（缺参问 → 回答 → 补全）、零 questions 直行、轮次上限、
ask 放弃走代码兜底、questions 垃圾容错、parse 异常传播、raw_instruction
防污染（骨架兜底）、dialog_context 累积、_dict_to_plan 接入 questions
（致命回归：不经 TaskPlan.from_dict）、schema 往返兼容旧 JSON、
ask_text 无词典语义、cli 降级链、G3 确认修改重解析、chat 命令串联、
无 key 零影响（代码兜底计划 + 代码级补参，零 LLM 调用）。
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from auto2dlabel import cli_commands
from auto2dlabel.agent.dialog import _answer_section
from auto2dlabel.agent.llm import LLMClient, LLMResponse, create_client
from auto2dlabel.agent.planner import TaskPlanner, _dict_to_plan
from auto2dlabel.cli_commands import (
    _fill_missing_params,
    _parse_plan_dialog,
    _reparse_confirm_edit,
)
from auto2dlabel.schema.task_plan import PlanQuestion, TaskPlan
from auto2dlabel.tests import json
from auto2dlabel.tools.confirm import ConfirmResult, ask_text


class _ScriptedLLM:
    """按固定序列返回响应，并记录每轮收到的完整 messages。"""

    model = "fake-llm"
    has_credentials = True  # 默认有 key（无 key 用例按需覆盖为 False）

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.messages_seen: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        self.messages_seen.append(messages)
        assert self.responses, "LLM 脚本耗尽"
        r = self.responses.pop(0)
        return LLMResponse(content=r.get("content"), tool_calls=r.get("tool_calls"))


def _plan_json(
    source: str = "",
    prompts: list[str] | None = None,
    questions: Any = None,
    conf: float = 0.1,
    iou: float = 0.3,
    model: str = "yolo26x.pt",
) -> str:
    """构造 TaskPlan JSON（questions 为 None 时省略键）。"""
    step = {
        "step_id": 1,
        "task_type": "object_detection",
        "source": source,
        "prompts": prompts or [],
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "model_name": model,
        "export_format": "coco",
        "sahi": False,
        "num_workers": None,
        "batch_size": None,
    }
    data: dict[str, Any] = {"steps": [step], "confirm_timeout": 30}
    if questions is not None:
        data["questions"] = questions
    return json.dumps(data)


def _ask_recorder(answers: list[str]) -> tuple[Any, list[list[PlanQuestion]]]:
    """Fake ask_fn：按序返回预设回答，记录收到的问题列表。"""
    seen: list[list[PlanQuestion]] = []
    it = iter(answers)

    def ask(questions: list[PlanQuestion]) -> str:
        seen.append(list(questions))
        return next(it)

    return ask, seen


def _make_planner(responses: list[dict[str, Any]]) -> tuple[TaskPlanner, _ScriptedLLM]:
    llm = _ScriptedLLM(responses)
    return TaskPlanner(llm_client=cast(LLMClient, llm)), llm


# ── 对话闭环 ────────────────────────────────────────────────

def test_dialog_roundtrip_collects_answers() -> None:
    """缺参+questions → 收集回答回喂 → 第 2 轮补全；raw_instruction 不被污染。"""
    instruction = "检测图片中的汽车"
    planner, llm = _make_planner([
        {"content": _plan_json(questions=[{"id": "step1.source", "question": "图像目录？"}])},
        {"content": _plan_json(source="/data/images", prompts=["car"])},
    ])
    ask, seen = _ask_recorder(["数据在 /data/images"])
    plan = planner.parse_dialog(instruction, ask_fn=ask)

    assert len(llm.messages_seen) == 2
    # 回喂文本：带 id 的补充信息段 + 用户回答
    fed = llm.messages_seen[1][-1]["content"]
    assert "[step1.source]" in fed and "数据在 /data/images" in fed
    assert seen == [[PlanQuestion(id="step1.source", question="图像目录？")]]
    assert plan.steps[0].source == "/data/images"
    assert plan.steps[0].prompts == ["car"]
    assert plan.raw_instruction == instruction  # 骨架防污染


def test_dialog_no_questions_single_round() -> None:
    """完整指令 → LLM questions 空/省略 → 1 次解析直接返回，ask 不触发。"""
    planner, llm = _make_planner([
        {"content": _plan_json(source="/data/images", prompts=["car"])},
    ])
    ask, seen = _ask_recorder(["不应被问"])
    plan = planner.parse_dialog("检测 /data/images 中的汽车", ask_fn=ask)

    assert len(llm.messages_seen) == 1
    assert seen == []
    assert plan.steps[0].source == "/data/images"


def test_dialog_max_rounds_exhaustion() -> None:
    """恒返回 questions + 恒回答 → 3 次解析后返回（不再问）；ask 只调 2 次。"""
    q = [{"id": "step1.prompts", "question": "类别？"}]
    planner, llm = _make_planner([
        {"content": _plan_json(questions=q)},
        {"content": _plan_json(questions=q)},
        {"content": _plan_json(questions=q)},
    ])
    ask, seen = _ask_recorder(["car", "car"])
    plan = planner.parse_dialog("检测图片", ask_fn=ask)

    assert len(llm.messages_seen) == 3  # ≤3 次 LLM 解析
    assert len(seen) == 2  # ≤2 次用户问答
    assert plan.steps[0].prompts == []  # 末轮缺参返回 → 代码兜底


def test_dialog_ask_abandon_returns_incomplete() -> None:
    """用户放弃（ask 返回空白）→ 1 轮后返回缺参 plan（_fill_missing_params 兜底）。"""
    planner, llm = _make_planner([
        {"content": _plan_json(questions=[{"id": "step1.source", "question": "路径？"}])},
        {"content": _plan_json(source="/data/images")},  # 不应被消费
    ])
    ask, seen = _ask_recorder([""])
    plan = planner.parse_dialog("检测图片", ask_fn=ask)

    assert len(llm.messages_seen) == 1
    assert len(seen) == 1
    assert plan.steps[0].source == ""


# ── questions 容错 ──────────────────────────────────────────

def test_dialog_garbage_questions_tolerated() -> None:
    """questions 非 list / 非 dict / 空 question → 视为无问题，直接执行（零对话）。"""
    planner, llm = _make_planner([
        {"content": _plan_json(
            questions=["nope", {"id": "x", "question": ""}, {"id": 1, "question": "  "}]
        )},
    ])
    ask, seen = _ask_recorder(["x"])
    planner.parse_dialog("检测图片", ask_fn=ask)

    assert len(llm.messages_seen) == 1
    assert seen == []


def test_dialog_numeric_id_coerced() -> None:
    """LLM 输出数字 id → 强转 str，对话正常进行。"""
    planner, llm = _make_planner([
        {"content": _plan_json(questions=[{"id": 123, "question": "路径？"}])},
        {"content": _plan_json(source="/data/images")},
    ])
    ask, seen = _ask_recorder(["/data/images"])
    plan = planner.parse_dialog("检测图片", ask_fn=ask)

    assert len(llm.messages_seen) == 2
    assert seen == [[PlanQuestion(id="123", question="路径？")]]
    assert plan.steps[0].source == "/data/images"


# ── 异常与状态 ──────────────────────────────────────────────

def test_dialog_parse_error_propagates() -> None:
    """parse_fn 抛 ValueError（LLM 非法响应）→ 骨架传播（降级链在调用方）。"""
    planner, _ = _make_planner([{"content": "not json at all"}])
    with pytest.raises(ValueError):
        planner.parse_dialog("检测图片", ask_fn=lambda q: "x")


def test_dialog_context_accumulates_across_rounds() -> None:
    """dialog_context 累积所有回喂段（Step 4 G3 重解析的基座）。"""
    planner, llm = _make_planner([
        {"content": _plan_json(questions=[{"id": "step1.source", "question": "路径？"}])},
        {"content": _plan_json(
            questions=[{"id": "step1.prompts", "question": "类别？"}],
        )},
        {"content": _plan_json(source="/data/images", prompts=["car"])},
    ])
    ask, _ = _ask_recorder(["/data/images", "car"])
    plan = planner.parse_dialog("检测图片", ask_fn=ask)

    assert plan.dialog_context.count("补充信息") == 2
    assert "路径？" in plan.dialog_context and "类别？" in plan.dialog_context
    assert plan.raw_instruction == "检测图片"


def test_dict_to_plan_parses_questions() -> None:
    """致命回归：LLM JSON → plan 走 _dict_to_plan（不经 TaskPlan.from_dict）。"""
    plan = _dict_to_plan(json.loads(_plan_json(
        questions=[{"id": "step1.source", "question": "路径？"}, {"id": 7, "question": "类别？"}],
    )))
    assert plan.questions == [
        PlanQuestion(id="step1.source", question="路径？"),
        PlanQuestion(id="7", question="类别？"),
    ]
    # 无 questions 键 → 兼容旧 JSON
    assert _dict_to_plan(json.loads(_plan_json(source="/d"))).questions == []


def test_plan_questions_roundtrip() -> None:
    """questions/dialog_context to_dict↔from_dict 往返 + 旧 JSON 兼容。"""
    plan = TaskPlan(
        steps=[],
        questions=[PlanQuestion(id="step1.source", question="路径？")],
        dialog_context="补充信息…",
    )
    d = plan.to_dict()
    assert "questions" in d and "dialog_context" in d  # 非空才输出
    back = TaskPlan.from_dict(d)
    assert back.questions == plan.questions and back.dialog_context == plan.dialog_context

    empty = TaskPlan()
    assert "questions" not in empty.to_dict()  # 空则不输出（防 JSON 膨胀）
    assert TaskPlan.from_dict({"steps": [], "confirm_timeout": 30}).questions == []
    assert TaskPlan.from_dict({"steps": [], "confirm_timeout": 30}).dialog_context == ""


def test_answer_section_format() -> None:
    """回喂格式与 prompt 规则描述一致：头 + 带 id 问题 + 用户回答。"""
    section = _answer_section(
        [PlanQuestion(id="step1.source", question="图像目录？")], "在 /data/images"
    )
    assert section == (
        "补充信息（用户在以下问题的回答，请据此更新 JSON）：\n"
        "- [step1.source] 图像目录？\n"
        "用户回答：在 /data/images"
    )


def test_parse_with_dialog_llm_empty_response() -> None:
    """LLM 空响应 → ValueError（与 parse 同语义）。"""
    planner, _ = _make_planner([{"content": None}])
    with pytest.raises(ValueError, match="空响应"):
        planner.parse_dialog("检测图片", ask_fn=lambda q: "x")


# ── ask_text 无词典语义 ─────────────────────────────────────

def test_ask_text_no_wait_returns_none() -> None:
    """timeout<=0 → 跳过对话返回 None（--no-wait）。"""
    assert ask_text("问题", timeout=0) is None
    assert ask_text("问题", timeout=-1) is None


def test_ask_text_collects_plain_words(monkeypatch: Any) -> None:
    """确认词（ok/好）按普通回答返回——不吞答案（对话闭环验收关键）。"""
    monkeypatch.setattr("builtins.input", lambda _: "好")
    assert ask_text("问题", timeout=1) == "好"
    monkeypatch.setattr("builtins.input", lambda _: "ok")
    assert ask_text("问题", timeout=1) == "ok"


def test_ask_text_cancel_and_empty(monkeypatch: Any) -> None:
    """cancel 词 / 空回车 → None（放弃对话走代码兜底）。"""
    monkeypatch.setattr("builtins.input", lambda _: "cancel")
    assert ask_text("问题", timeout=1) is None
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert ask_text("问题", timeout=1) is None
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert ask_text("问题", timeout=1) is None


# ── CLI 层：降级链 + G3 ────────────────────────────────────

def test_parse_plan_dialog_fallback_on_bad_llm(caplog: Any) -> None:
    """对话层 LLM 非法响应 → 降级单轮 parse（v0.5 行为）+ warning 日志。"""
    planner, llm = _make_planner([
        {"content": "garbage"},
        {"content": _plan_json(source="/data/images", prompts=["car"])},
    ])
    plan = _parse_plan_dialog(planner, "检测图片", timeout=30)

    assert len(llm.messages_seen) == 2  # 1 次对话失败 + 1 次单轮兜底
    assert plan.steps[0].source == "/data/images"
    assert "降级为单轮 parse" in caplog.text


def test_reparse_confirm_edit_uses_dialog_context() -> None:
    """G3：对话收集后确认修改 → 重解析基于 dialog_context；raw_instruction 保留原语义。"""
    instruction = "检测图片"
    planner, llm = _make_planner([
        {"content": _plan_json(questions=[{"id": "step1.source", "question": "路径？"}])},
        {"content": _plan_json(source="/data/images", prompts=["car"])},
        # 第 3 次 = Step 4 确认修改重解析（用户改类别）
        {"content": _plan_json(source="/data/images", prompts=["person"])},
    ])
    ask, _ = _ask_recorder(["/data/images"])
    plan = planner.parse_dialog(instruction, ask_fn=ask)

    updated = _reparse_confirm_edit(planner, plan, "改成检测行人", timeout=30)

    # 重解析的消息含 dialog_context（补全字段不丢）
    last_user = llm.messages_seen[-1][-1]["content"]
    assert "补充信息" in last_user and "改成检测行人" in last_user
    assert updated.steps[0].prompts == ["person"]
    # raw_instruction = 原指令 + 用户修改（v0.5 语义，Step 4.5+ 代码级扫描可见）
    assert updated.raw_instruction == f"{instruction} 改成检测行人"
    assert "补充信息" not in updated.raw_instruction  # 对话噪声不入代码扫描面


# ── 无 key 零影响（v0.6 P1 验收）───────────────────────────

def test_parse_plan_dialog_no_key_code_fallback() -> None:
    """无 key → 零 LLM 调用，代码级提取类别 + 全默认参数构建计划。"""
    planner, llm = _make_planner([{"content": _plan_json(source="/d", prompts=["car"])}])
    llm.has_credentials = False

    plan = _parse_plan_dialog(planner, "检测图片中的汽车和行人", timeout=30)

    assert llm.messages_seen == []  # 零 LLM 调用
    assert plan.steps[0].source == ""  # 路径未知：缺参交 _fill_missing_params
    assert plan.steps[0].prompts == ["person", "car"]  # CN_EN_MAP 代码级提取
    assert plan.steps[0].task_type == "object_detection"
    assert plan.raw_instruction == "检测图片中的汽车和行人"


def test_parse_plan_dialog_no_key_no_keywords() -> None:
    """无 key + 无类别关键词 → prompts 留空（不硬造，交推荐/追问接管）。"""
    planner, llm = _make_planner([{"content": _plan_json()}])
    llm.has_credentials = False

    plan = _parse_plan_dialog(planner, "帮我标注一下", timeout=30)

    assert llm.messages_seen == []
    assert plan.steps[0].prompts == []
    assert plan.all_missing_params  # source/prompts 均缺 → 代码级追问兜底


def test_fill_missing_params_no_key_code_mapping(monkeypatch: Any) -> None:
    """无 key 补参：自由文本代码级映射（路径 → source、关键词 → prompts），零 LLM。"""
    planner, llm = _make_planner([{"content": _plan_json(source="/d", prompts=["car"])}])
    llm.has_credentials = False
    plan = _parse_plan_dialog(planner, "检测图片", timeout=30)
    assert plan.all_missing_params

    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_missing_params",
        lambda missing, timeout=30, optional=None: "数据在 /data/images/，检测汽车和行人",
    )
    _fill_missing_params(plan, planner, timeout=30)

    assert llm.messages_seen == []  # 补参也零 LLM
    assert plan.steps[0].source == "/data/images/"
    assert plan.steps[0].prompts == ["person", "car"]
    assert not plan.all_missing_params


def test_fill_missing_params_no_key_partial_mapping(monkeypatch: Any) -> None:
    """无 key 补参提取不到的内容保持缺参（宁缺勿错，不硬造字段）。"""
    planner, llm = _make_planner([{"content": _plan_json()}])
    llm.has_credentials = False
    plan = _parse_plan_dialog(planner, "检测图片", timeout=30)

    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_missing_params",
        lambda missing, timeout=30, optional=None: "随便写点啥，没路径也没类别",
    )
    _fill_missing_params(plan, planner, timeout=30)

    assert llm.messages_seen == []
    assert plan.steps[0].source == ""
    assert plan.steps[0].prompts == []
    assert plan.all_missing_params  # 保持缺参 → 后续步骤接管


def test_fill_missing_params_asks_optional_model_conf(monkeypatch: Any) -> None:
    """追问扩展（2026-08-29）：缺模型/缺超参数 → optional 进入追问，回答重解析提取。"""
    # 第一响应：全默认参数计划（模型/阈值未指定）；第二响应：重解析提取回答
    planner, llm = _make_planner([
        {"content": _plan_json(source="/d", prompts=["car"])},
        {"content": _plan_json(
            source="/d", prompts=["car"],
            conf=0.5, model="fasterrcnn_resnet50_fpn_v2",
        )},
    ])
    plan = _parse_plan_dialog(planner, "检测 /d 中的汽车", timeout=30)
    assert not plan.all_missing_params  # required 已满足

    asked: dict[str, Any] = {}

    def _ask(missing: Any, timeout: int = 30, optional: Any = None) -> str:
        asked["missing"] = missing
        asked["optional"] = optional
        return "模型用 fasterrcnn，置信度 0.5"

    monkeypatch.setattr("auto2dlabel.tools.confirm.ask_missing_params", _ask)
    _fill_missing_params(plan, planner, timeout=30)

    assert asked["missing"] == {}  # 必填无缺失
    step = plan.steps[0]
    # optional 覆盖 model + conf（iou 仍在默认 → 保持默认继续）
    opt_text = "\n".join(asked["optional"][1])
    assert "model_name（模型）（默认 yolo26x.pt）" in opt_text
    assert "confidence_threshold（置信度）（默认 0.1）" in opt_text
    assert step.model_name == "fasterrcnn_resnet50_fpn_v2"
    assert step.confidence_threshold == 0.5
    assert step.iou_threshold == 0.3  # 未回答 → 默认
    assert llm.messages_seen  # 有 key：回答经 LLM 重解析提取


def test_fill_missing_params_optional_abort_keeps_defaults(monkeypatch: Any) -> None:
    """追问扩展：回车/超时（None）→ 可选参数保持默认继续（必填也不硬造）。"""
    planner, llm = _make_planner([{"content": _plan_json(source="/d", prompts=["car"])}])
    plan = _parse_plan_dialog(planner, "检测 /d 中的汽车", timeout=30)

    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_missing_params",
        lambda missing, timeout=30, optional=None: None,  # 用户回车跳过
    )
    _fill_missing_params(plan, planner, timeout=30)

    step = plan.steps[0]
    assert step.model_name == "yolo26x.pt"  # 默认设置继续
    assert step.confidence_threshold == 0.1
    assert len(llm.messages_seen) == 1  # 无重解析（回答为空）


def test_fill_missing_params_appends_dialog_context(monkeypatch: Any) -> None:
    """追问回答进 dialog_context（2026-08-29 实测：确认编辑后 conf 0.3→0.1）。

    回答必须成为后续重解析的基础文本，否则 Step 4 确认时改一句参数，
    追问成果全丢（v0.6 G3 同款教训）。
    """
    planner, llm = _make_planner([
        {"content": _plan_json(source="/d", prompts=["car"])},
        {"content": _plan_json(source="/d", prompts=["car"], conf=0.3, iou=0.3)},
    ])
    plan = _parse_plan_dialog(planner, "检测 /d 中的汽车", timeout=30)

    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_missing_params",
        lambda missing, timeout=30, optional=None: "confidence 和 iou 都设置为 0.3",
    )
    _fill_missing_params(plan, planner, timeout=30)

    assert "confidence 和 iou 都设置为 0.3" in (plan.dialog_context or "")
    assert plan.steps[0].confidence_threshold == 0.3


def test_reparse_confirm_edit_preserves_earlier_answers() -> None:
    """确认编辑不丢前序追问设置（实测洞端到端回归）。

    用户先回答 conf=0.3（追问链），确认时改任务类型/模型；LLM 重解析
    把 conf 回归默认 0.1 → 代码级 carry-over：编辑未提及的 conf 保留 0.3，
    显式提及的 model 信任新解析（sam3.pt）。
    """
    planner, llm = _make_planner([
        # 重解析响应：LLM 只带出模型变更，conf 丢回默认（模拟实测）
        {"content": _plan_json(
            source="llm_test_data/llm_test_data_nuscenes",
            prompts=["car", "person"],
            conf=0.1, iou=0.3, model="sam3.pt",
        )},
    ])
    old = _dict_to_plan(json.loads(_plan_json(
        source="llm_test_data/llm_test_data_nuscenes",
        prompts=["car", "person"],
        conf=0.3, iou=0.3, model="yolo26x.pt",
    )))
    old.raw_instruction = "检测 llm_test_data/llm_test_data_nuscenes 中的汽车和行人"

    updated = _reparse_confirm_edit(
        planner, old, "改为实例分割任务, 模型采用 sam3", timeout=30
    )

    step = updated.steps[0]
    assert step.model_name == "sam3.pt"  # 编辑显式提及 → 新解析胜出
    assert step.confidence_threshold == 0.3  # 未提及 → 前序回答保留（曾回归 0.1）
    assert step.prompts == ["car", "person"]
    assert len(llm.messages_seen) == 1


def test_reparse_confirm_edit_explicit_edit_wins() -> None:
    """编辑显式提及的参数信任新解析（不被 carry-over 回滚）。"""
    planner, llm = _make_planner([
        {"content": _plan_json(
            source="/d", prompts=["car"],
            conf=0.1, iou=0.3, model="yolo12n.pt",
        )},
    ])
    old = _dict_to_plan(json.loads(_plan_json(
        source="/d", prompts=["car"],
        conf=0.5, iou=0.4, model="fasterrcnn_resnet50_fpn_v2",
    )))
    old.raw_instruction = "检测 /d 中的汽车，置信度 0.5"

    updated = _reparse_confirm_edit(
        planner, old, "置信度改为 0.1，模型改为 yolo12n.pt", timeout=30
    )

    step = updated.steps[0]
    # 编辑显式提及 conf/model → 新解析值生效（即使 conf 0.1 == 默认）
    assert step.model_name == "yolo12n.pt"
    assert step.confidence_threshold == 0.1
    # iou 未提及且新解析为默认 → 前序 0.4 保留
    assert step.iou_threshold == 0.4


def test_create_client_has_credentials(monkeypatch: Any) -> None:
    """has_credentials = 创建时解析的 env key（openai/anthropic 同 deepseek 提前到创建）。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert not create_client("deepseek").has_credentials  # 无 env 无显式 key → False
    assert create_client("deepseek", api_key="k").has_credentials  # 显式 key → True

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert create_client("openai").has_credentials  # v0.6：openai 也创建时解析
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert create_client("anthropic").has_credentials


def test_chat_command_no_key_full_chain(monkeypatch: Any) -> None:
    """chat 无 key 全链路：零 LLM 调用 + 代码级计划直达执行（验收「无 key 零影响」）。"""
    llm = _ScriptedLLM([{"content": _plan_json(source="/d", prompts=["car"])}])
    llm.has_credentials = False
    executed: list[TaskPlan] = []

    def _fake_execute(plan: TaskPlan, **_kw: Any) -> None:
        executed.append(plan)

    monkeypatch.setattr(cli_commands, "create_client", lambda provider: llm)
    monkeypatch.setattr(cli_commands, "execute_plan", _fake_execute)
    monkeypatch.setattr(cli_commands, "_fill_batch_params", lambda *a, **kw: None)
    # 无 key 下缺失追问由代码级映射兜底（零 LLM）
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_missing_params",
        lambda missing, timeout=30, optional=None: "数据在 /data/images/，检测汽车",
    )
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_with_timeout",
        lambda *a, **kw: ConfirmResult(confirmed=True),
    )

    cli_commands.chat_command(
        instruction="检测图片中的汽车",
        det_model=None,
        confirm_timeout=30,
        no_wait=False,
        provider="deepseek",
        verbose=False,
        sahi=False,
    )

    assert llm.messages_seen == []  # 全链路零 LLM 调用
    assert len(executed) == 1
    assert executed[0].steps[0].source == "/data/images/"
    assert executed[0].steps[0].prompts == ["car"]
    assert executed[0].raw_instruction == "检测图片中的汽车"


# ── CLI 串联：chat_command 全链路 ──────────────────────────

def test_chat_command_dialog_flow(monkeypatch: Any) -> None:
    """串联闭环：缺参 → ask_text 问出 source → execute_plan 收到补全 plan。"""
    llm = _ScriptedLLM([
        {"content": _plan_json(
            prompts=["car"],
            questions=[{"id": "step1.source", "question": "图像目录？"}],
        )},
        {"content": _plan_json(source="/data/images", prompts=["car"])},
    ])
    executed: list[TaskPlan] = []

    def _fake_execute(plan: TaskPlan, **_kw: Any) -> None:
        executed.append(plan)

    monkeypatch.setattr(cli_commands, "create_client", lambda provider: llm)
    monkeypatch.setattr(cli_commands, "execute_plan", _fake_execute)
    monkeypatch.setattr(cli_commands, "_fill_batch_params", lambda *a, **kw: None)
    # dialog.py 顶层已绑定 ask_text → 打 dialog 命名空间（不是 confirm）
    monkeypatch.setattr(
        "auto2dlabel.agent.dialog.ask_text", lambda prompt, timeout=30: "/data/images"
    )
    # Step 4 确认直接通过（无修改文本 → 不触发重解析）
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_with_timeout",
        lambda *a, **kw: ConfirmResult(confirmed=True),
    )

    cli_commands.chat_command(
        instruction="检测图片中的汽车",
        det_model=None,
        confirm_timeout=30,
        no_wait=False,
        provider="deepseek",
        verbose=False,
        sahi=False,
    )

    assert len(llm.messages_seen) == 2  # 1 轮对话 + 1 轮补全，无多余 LLM 调用
    assert len(executed) == 1
    assert executed[0].steps[0].source == "/data/images"
    assert executed[0].steps[0].prompts == ["car"]
    assert executed[0].raw_instruction == "检测图片中的汽车"  # 防污染穿透到执行层


def test_chat_command_complete_no_wait(monkeypatch: Any) -> None:
    """串联直行：完整指令 + --no-wait → 单轮解析，零对话零确认直达执行。

    组 1 控制变量场景：指令显式指定「模型用 yolo26x.pt，置信度 0.5，
    IoU 0.4」→ 即使 LLM 提取的 model 恰为默认值（yolo26x.pt），
    guard_explicit_params 视为已指定 → optional 追问为空，不误追问。
    """
    llm = _ScriptedLLM([
        {"content": _plan_json(
            source="/data/images", prompts=["car"],
            conf=0.5, iou=0.4, model="yolo26x.pt",
        )},
    ])
    executed: list[TaskPlan] = []

    def _fake_execute(plan: TaskPlan, **_kw: Any) -> None:
        executed.append(plan)

    monkeypatch.setattr(cli_commands, "create_client", lambda provider: llm)
    monkeypatch.setattr(cli_commands, "execute_plan", _fake_execute)
    monkeypatch.setattr(cli_commands, "_fill_batch_params", lambda *a, **kw: None)
    # --no-wait 下任何交互调用都是缺陷 → fail 守卫
    monkeypatch.setattr(
        "auto2dlabel.agent.dialog.ask_text",
        lambda *a, **kw: pytest.fail("--no-wait 不应触发对话"),
    )
    monkeypatch.setattr(
        "auto2dlabel.tools.confirm.ask_with_timeout",
        lambda *a, **kw: pytest.fail("--no-wait 不应触发确认"),
    )

    cli_commands.chat_command(
        instruction="检测 /data/images 中的汽车，模型用 yolo26x.pt，置信度 0.5，IoU 0.4",
        det_model=None,
        confirm_timeout=30,
        no_wait=True,
        provider="deepseek",
        verbose=False,
        sahi=False,
    )

    assert len(llm.messages_seen) == 1  # 单轮解析
    assert len(executed) == 1
    assert executed[0].steps[0].source == "/data/images"
    assert executed[0].steps[0].prompts == ["car"]
