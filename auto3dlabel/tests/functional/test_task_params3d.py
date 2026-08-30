"""3D 任务参数规格表 + 守卫测试（configs/task_params.py，2026-08-29 重构）。

覆盖：规格表字段集与必填、默认值引用 kitti 常量（单一事实源）、
Plan3D.missing_params 表驱动（frame_id "" / prompts None/[] → 缺失，
原 cli 只查 frame_id——prompts 缺失会漏到下游）；守卫层：frame_id 纯数字
校验（垃圾 → "" 缺参，防 resolve_frame 裸 traceback）、conf 值域、
parse 钩子（帧号数字被误提为 conf 的漏洞类）。
"""

from __future__ import annotations

import json
from typing import Any, cast

from auto2dlabel.agent.llm import LLMClient, LLMResponse
from auto2dlabel.configs.task_params import ParamSpec
from auto3dlabel.agent.planner3d import Plan3D, TaskPlanner3D, parse_plan3d_json, sanitize_plan3d
from auto3dlabel.configs.kitti import DEFAULT_CONF, DEFAULT_DET_MODEL, DEFAULT_SEG_MODEL
from auto3dlabel.configs.task_params import PARAM_SPECS_3D


class _FakeLLM:
    """固定内容单次响应（零网络零模型）。"""

    model = "fake-llm"
    has_credentials = True

    def __init__(self, content: str) -> None:
        self.content = content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self.content, tool_calls=None)


def test_specs_3d_fields() -> None:
    """规格表恰好覆盖 Plan3D 的 5 个任务参数；frame_id/prompts 必填。"""
    assert set(PARAM_SPECS_3D) == {
        "frame_id",
        "prompts",
        "confidence_threshold",
        "det_model",
        "seg_model",
    }
    required = {s.key for s in PARAM_SPECS_3D.values() if s.required}
    assert required == {"frame_id", "prompts"}
    for spec in PARAM_SPECS_3D.values():
        assert isinstance(spec, ParamSpec)


def test_defaults_reference_kitti_constants() -> None:
    """可选参数默认值引用 kitti.py 常量（改常量不漏表），与 Plan3D 同源。"""
    assert PARAM_SPECS_3D["confidence_threshold"].default == DEFAULT_CONF
    assert PARAM_SPECS_3D["det_model"].default == DEFAULT_DET_MODEL
    assert PARAM_SPECS_3D["seg_model"].default == DEFAULT_SEG_MODEL
    plan = Plan3D()  # 空计划默认 == 规格表默认（派生零漂移）
    for key in ("confidence_threshold", "det_model", "seg_model"):
        assert getattr(plan, key) == PARAM_SPECS_3D[key].default, key


def test_plan3d_missing_params() -> None:
    """missing 判定表驱动：frame_id "" / prompts None → 缺失。"""
    assert Plan3D().missing_params == [
        "frame_id（KITTI 帧号）",
        "prompts（检测类别）",
    ]
    assert Plan3D(frame_id="000123").missing_params == ["prompts（检测类别）"]
    assert Plan3D(prompts=["car"]).missing_params == ["frame_id（KITTI 帧号）"]
    assert Plan3D(frame_id="000123", prompts=["car"]).missing_params == []


def test_plan3d_missing_params_empty_prompts() -> None:
    """prompts 空列表同样视为缺失（[] falsy）——原 cli 漏检的场景。"""
    assert Plan3D(frame_id="000123", prompts=[]).missing_params == [
        "prompts（检测类别）"
    ]


def test_planner3d_rules_untouched() -> None:
    """planner3d prompt 规则不受规格表影响（回归护栏）。"""
    from auto3dlabel.agent.planner3d import _PLANNER3D_SYSTEM_PROMPT

    assert "frame_id" in _PLANNER3D_SYSTEM_PROMPT
    assert "grounding-dino-tiny" in _PLANNER3D_SYSTEM_PROMPT
    # 帧号/路径数字不提取规则（与代码守卫双保险）
    assert "NEVER hyperparameters" in _PLANNER3D_SYSTEM_PROMPT


# ================================================================
# 3D 守卫（2026-08-29 防御纵深）
# ================================================================


def test_sanitize_plan3d_garbage() -> None:
    """frame_id 垃圾 → ""（缺参走追问/BadParameter，防裸 traceback）；
    prompts 过滤空串；conf 值域回默认。"""
    plan = Plan3D(
        frame_id="None",  # LLM 把 null 转成字符串 "None"
        prompts=["car", "  "],
        confidence_threshold=2017,
        det_model="",
        seg_model=cast(str, None),
    )
    fixes = sanitize_plan3d(plan)
    assert plan.frame_id == ""
    assert plan.prompts == ["car"]
    assert plan.confidence_threshold == DEFAULT_CONF
    assert plan.det_model == DEFAULT_DET_MODEL
    assert plan.seg_model == DEFAULT_SEG_MODEL
    assert plan.missing_params == ["frame_id（KITTI 帧号）"]
    assert fixes  # 守卫触发必须可见


def test_sanitize_plan3d_non_digit_frame_id() -> None:
    """非纯数字帧号（abc123/123abc）→ ""。"""
    plan = Plan3D(frame_id="abc123", prompts=["car"])
    sanitize_plan3d(plan)
    assert plan.frame_id == ""


def test_sanitize_plan3d_keeps_valid() -> None:
    """合法计划零修正（零漂移）。"""
    plan = Plan3D(frame_id="000123", prompts=["car"], confidence_threshold=0.5)
    assert sanitize_plan3d(plan) == []
    assert plan.frame_id == "000123" and plan.confidence_threshold == 0.5


def test_dict_to_plan3d_garbage_conf_no_raise() -> None:
    """conf 垃圾（"abc"/2017）→ 默认，不再抛异常中断解析。"""
    plan = parse_plan3d_json(
        json.dumps({"frame_id": "000123", "prompts": ["car"], "confidence_threshold": "abc"})
    )
    assert plan.confidence_threshold == DEFAULT_CONF
    plan2 = parse_plan3d_json(
        json.dumps({"frame_id": "000123", "prompts": ["car"], "confidence_threshold": 2017})
    )
    assert plan2.confidence_threshold == DEFAULT_CONF


def test_planner3d_parse_sanitizes_garbage() -> None:
    """parse 钩子：LLM 输出垃圾 → 守卫归位（帧号数字误提为 conf 的洞）。"""
    content = json.dumps({
        "frame_id": "abc123",
        "prompts": ["car"],
        "confidence_threshold": 2017,
    })
    planner = TaskPlanner3D(llm_client=cast(LLMClient, _FakeLLM(content)))
    plan = planner.parse("标注 KITTI 帧 000123 中的汽车")
    assert plan.frame_id == ""
    assert plan.confidence_threshold == DEFAULT_CONF
