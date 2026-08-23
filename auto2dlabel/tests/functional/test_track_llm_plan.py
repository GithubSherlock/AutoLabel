"""跟踪 LLM 一次性规划测试 —— resolve_plan 回退链 + _llm_plan_once JSON 解析。

v0.4 3a 起 resolve_plan/_llm_plan_once 返回 ReferentialConstraint
（prompts + 可选 attributes/position + threshold）。
"""

from __future__ import annotations

from typing import Any

import pytest

import auto2dlabel.cli_track as cli_track
from auto2dlabel.agent.llm import LLMResponse
from auto2dlabel.cli_track import _llm_plan_once, resolve_plan
from auto2dlabel.tools.constraints import ReferentialConstraint


class _FakeClient:
    """预设回复的 LLM 客户端桩。"""

    def __init__(self, content: str | None) -> None:
        self._content = content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        return LLMResponse(content=self._content)


def _plan_with_fake(
    content: str, monkeypatch: pytest.MonkeyPatch,
) -> ReferentialConstraint:
    """注入预设回复后调用 _llm_plan_once。"""
    import auto2dlabel.agent.llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "create_client",
        lambda provider=None, model=None, api_key=None, base_url=None: _FakeClient(content),
    )
    return _llm_plan_once("检测行人", "deepseek", None, None, None)


def test_llm_plan_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _plan_with_fake('{"prompts": ["person"], "threshold": 0.3}', monkeypatch)
    assert c.prompts == ["person"]
    assert c.threshold == 0.3
    assert c.is_plain


def test_llm_plan_parses_attributes_position(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _plan_with_fake(
        '{"prompts": ["person"], "threshold": null, "attributes": ["red"], "position": "left"}',
        monkeypatch,
    )
    assert c.attributes == ["red"]
    assert c.position == "left"
    assert not c.is_plain


def test_llm_plan_strips_markdown_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _plan_with_fake(
        '```json\n{"prompts": ["person", "car"], "threshold": null}\n```', monkeypatch)
    assert c.prompts == ["person", "car"]
    assert c.threshold is None


def test_llm_plan_null_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _plan_with_fake('{"prompts": ["person"], "threshold": null}', monkeypatch)
    assert c.threshold is None


def test_llm_plan_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _plan_with_fake("不是 JSON", monkeypatch)


def test_llm_plan_empty_prompts_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _plan_with_fake('{"prompts": [], "threshold": null}', monkeypatch)


def test_llm_plan_threshold_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _plan_with_fake('{"prompts": ["person"], "threshold": 1.5}', monkeypatch)


def test_llm_plan_bool_threshold_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """threshold=True 是 int 子类，须拒绝（布尔阈值无意义）。"""
    with pytest.raises(ValueError):
        _plan_with_fake('{"prompts": ["person"], "threshold": true}', monkeypatch)


def test_llm_plan_invalid_position_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _plan_with_fake(
            '{"prompts": ["person"], "threshold": null, "position": "behind"}', monkeypatch)


def test_llm_plan_non_str_attribute_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _plan_with_fake(
            '{"prompts": ["person"], "threshold": null, "attributes": [1]}', monkeypatch)


def test_resolve_plan_without_llm_uses_parse_referential(monkeypatch: pytest.MonkeyPatch) -> None:
    c = resolve_plan("检测行人", False, "deepseek", None, None, None, 0.1)
    assert c.prompts == ["person"]
    assert c.threshold == 0.1  # 代码级解析沿用 CLI 阈值
    assert c.is_plain


def test_resolve_plan_without_llm_parses_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    c = resolve_plan("检测红色的汽车", False, "deepseek", None, None, None, 0.1)
    assert c.prompts == ["car"]
    assert c.attributes == ["red"]
    assert not c.is_plain


def test_resolve_plan_llm_success_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_track, "_llm_plan_once",
        lambda *a, **k: ReferentialConstraint(prompts=["car"], threshold=0.5),
    )
    c = resolve_plan("复杂指令", True, "deepseek", None, None, None, 0.1)
    assert c.prompts == ["car"]
    assert c.threshold == 0.5


def test_resolve_plan_llm_null_threshold_keeps_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_track, "_llm_plan_once",
        lambda *a, **k: ReferentialConstraint(prompts=["car"], threshold=None),
    )
    c = resolve_plan("复杂指令", True, "deepseek", None, None, None, 0.1)
    assert c.threshold == 0.1


def test_resolve_plan_llm_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> ReferentialConstraint:
        raise RuntimeError("network down")

    monkeypatch.setattr(cli_track, "_llm_plan_once", boom)
    c = resolve_plan("检测行人", True, "deepseek", None, None, None, 0.1)
    assert c.prompts == ["person"]  # 回退 parse_referential
    assert c.threshold == 0.1
