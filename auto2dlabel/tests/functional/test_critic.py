"""v1.1 P2 质检 Agent（Critic）测试 —— critic.py 纯函数 + cli_execute 触发。

零真实 LLM 铁律：FakeChat 注入（记录 call_site + 返回固定 JSON）；Fake
client has_credentials False 走静默跳过路径。Critic 触发 = quality.ok==False。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel.agent import critic as critic_mod
from auto2dlabel.agent.critic import (
    CRITIC_CALL_SITE,
    CRITIC_SYSTEM_PROMPT,
    _parse_critic_json,
    critic_provider,
    criticize,
    should_criticize,
)
from auto2dlabel.agent.evaluate import QualityReport
from auto2dlabel.cli_execute import _run_critic_if_needed, _should_run_critic


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.finish_reason = "stop"


class _FakeChat:
    """记录 call_site + 返回固定 content 的 Fake LLM 客户端（零真实调用）。"""

    def __init__(self, content: str, *, has_credentials: bool = True) -> None:
        self._content = content
        self.has_credentials = has_credentials
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _FakeResponse:
        self.calls.append({"messages": messages, **kwargs})
        return _FakeResponse(self._content)


def _bad_report() -> QualityReport:
    return QualityReport(total_boxes=0, missing_prompts=["person"])


def _ok_report() -> QualityReport:
    return QualityReport(total_boxes=5, prompts=["person"], covered_prompts=["person"])


# ---------- 纯函数 ----------


def test_should_criticize_trigger() -> None:
    """quality.ok == False → 触发；ok == True / None → 不触发。"""
    assert should_criticize(_bad_report()) is True
    assert should_criticize(_ok_report()) is False
    assert should_criticize(None) is False


def test_parse_critic_json_valid() -> None:
    verdict = _parse_critic_json(
        '{"judgment": "fail", "reason": "缺 person", "action": "flag_for_review"}'
    )
    assert verdict["judgment"] == "fail"
    assert verdict["reason"] == "缺 person"
    assert verdict["action"] == "flag_for_review"


def test_parse_critic_json_fenced_and_braced() -> None:
    """容错：markdown 代码块 / 花括号提取。"""
    fenced = '```json\n{"judgment": "pass", "reason": "ok", "action": "accept"}\n```'
    assert _parse_critic_json(fenced)["judgment"] == "pass"
    braced = '前缀 {"judgment": "fail", "reason": "x", "action": "accept"} 后缀'
    assert _parse_critic_json(braced)["judgment"] == "fail"


def test_parse_critic_json_invalid_raises() -> None:
    """非法 judgment/action / 无 JSON → ValueError。"""
    with pytest.raises(ValueError):
        _parse_critic_json("no json at all")
    with pytest.raises(ValueError):
        _parse_critic_json('{"judgment": "maybe", "reason": "x", "action": "accept"}')
    with pytest.raises(ValueError):
        _parse_critic_json('{"judgment": "pass", "reason": "x", "action": "do_something"}')


def test_critic_provider_env_default() -> None:
    assert critic_provider() == "deepseek"  # 默认同 planner；env 覆盖


def test_critic_system_prompt_rendered() -> None:
    """critic.md 渲染（frontmatter profile: planning；独立 prompt 不含 planner 规则）。"""
    assert "independent quality critic" in CRITIC_SYSTEM_PROMPT
    assert "judgment" in CRITIC_SYSTEM_PROMPT and "action" in CRITIC_SYSTEM_PROMPT


# ---------- criticize() ----------


def test_criticize_calls_with_call_site() -> None:
    """criticize → client.chat(call_site=evaluate_critic)；prompt 注入 quality JSON。"""
    fake = _FakeChat('{"judgment": "fail", "reason": "缺 person", "action": "flag_for_review"}')
    verdict = criticize(_bad_report(), fake)  # type: ignore[arg-type]
    assert verdict["judgment"] == "fail"
    assert fake.calls[0]["call_site"] == CRITIC_CALL_SITE
    assert "Quality report" in fake.calls[0]["messages"][1]["content"]
    assert "person" in fake.calls[0]["messages"][1]["content"]  # 缺类进 prompt


def test_criticize_empty_response_raises() -> None:
    fake = _FakeChat("")
    with pytest.raises(ValueError):
        criticize(_bad_report(), fake)  # type: ignore[arg-type]


def test_criticize_non_json_raises() -> None:
    fake = _FakeChat("模型说我也不知道")
    with pytest.raises(ValueError):
        criticize(_bad_report(), fake)  # type: ignore[arg-type]


# ---------- cli_execute 触发（_run_critic_if_needed） ----------


def test_run_critic_when_ok_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """quality.ok == True → 不触发（零 Critic 调用）。"""
    called: list[str] = []

    def _fake(r: Any, c: Any) -> dict[str, Any]:
        called.append("x")
        return {}

    monkeypatch.setattr(critic_mod, "criticize", _fake)
    _run_critic_if_needed(_ok_report(), [])
    assert called == []


def test_run_critic_no_credentials_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """无凭据（has_credentials False）→ 静默跳过零调用。"""
    called: list[str] = []

    def _fake(r: Any, c: Any) -> dict[str, Any]:
        called.append("x")
        return {}

    monkeypatch.setattr(critic_mod, "criticize", _fake)

    class _NoKey:
        has_credentials = False

    monkeypatch.setattr(critic_mod, "build_critic_client", lambda: _NoKey())
    _run_critic_if_needed(_bad_report(), [])
    assert called == []


def test_run_critic_success_annotates_steps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """ok==False + 有凭据 → criticize 调用 + steps_results[-1]['critic'] 写入。"""
    verdict = {"judgment": "fail", "reason": "缺 person", "action": "flag_for_review"}
    monkeypatch.setattr(critic_mod, "criticize", lambda r, c: dict(verdict))

    class _Key:
        has_credentials = True

    monkeypatch.setattr(critic_mod, "build_critic_client", lambda: _Key())
    results: list[dict[str, Any]] = [{"step_id": 1}]
    _run_critic_if_needed(_bad_report(), results)
    assert results[0]["critic"]["judgment"] == "fail"
    assert results[0]["critic"]["call_site"] == CRITIC_CALL_SITE


def test_run_critic_failure_swallows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Critic 调用抛异常 → 静默跳过（不阻断标注路径）。"""
    monkeypatch.setattr(
        critic_mod, "criticize",
        lambda r, c: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    class _Key:
        has_credentials = True

    monkeypatch.setattr(critic_mod, "build_critic_client", lambda: _Key())
    _run_critic_if_needed(_bad_report(), [])  # 不 raise


def test_should_run_critic_wrapper() -> None:
    assert _should_run_critic(_bad_report()) is True
    assert _should_run_critic(_ok_report()) is False
    assert _should_run_critic(None) is False
