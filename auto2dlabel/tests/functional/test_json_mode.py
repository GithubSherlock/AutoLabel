"""json mode / max_tokens 参数穿透测试（v0.6 Phase 4，零真实 API）。

覆盖：OpenAI 兼容路径 response_format={"type":"json_object"} 注入、max_tokens
透传、usage/finish_reason 捕获；Anthropic 路径 json_mode 忽略（无此能力）、
max_tokens 生效、stop_reason=="max_tokens" → "length" 规整。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from auto2dlabel.agent import llm as llm_mod
from auto2dlabel.agent.llm import AnthropicClient, OpenAIClient


class _OpenAIStub:
    """OpenAI SDK 客户端桩：捕获 create kwargs，返回预置响应。"""

    def __init__(self, response: Any):
        self.response = response
        self.kwargs: dict[str, Any] = {}

    @property
    def chat(self) -> _OpenAIStub:
        return self

    @property
    def completions(self) -> _OpenAIStub:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _openai_response(content: str, usage: Any = None, finish_reason: str = "stop") -> Any:
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=None),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _openai_client(response: Any) -> tuple[OpenAIClient, _OpenAIStub]:
    client = OpenAIClient(model="deepseek-chat", api_key="x", base_url="https://x")
    stub = _OpenAIStub(response)
    client._client = stub  # type: ignore[assignment]
    return client, stub


def test_openai_json_mode_injects_response_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_mode=True → response_format={"type":"json_object"} 注入。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, stub = _openai_client(_openai_response("{}"))
    client.chat([{"role": "user", "content": "hi"}], json_mode=True, max_tokens=1024)
    assert stub.kwargs["response_format"] == {"type": "json_object"}
    assert stub.kwargs["max_tokens"] == 1024


def test_openai_json_mode_off_omits_response_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_mode=False → 无 response_format 键（行为与改造前一致）。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, stub = _openai_client(_openai_response("ok"))
    client.chat([{"role": "user", "content": "hi"}])
    assert "response_format" not in stub.kwargs


def test_openai_usage_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI 路径 usage → UsageStats 规整进 LLMResponse（台账落盘）。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    client, _ = _openai_client(_openai_response("{}", usage=usage))
    resp = client.chat([{"role": "user", "content": "hi"}], call_site="planner.parse")
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.finish_reason == "stop"


def test_openai_max_tokens_omitted_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_tokens=None → 不注入（provider 默认行为保留）。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, stub = _openai_client(_openai_response("ok"))
    client.chat([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in stub.kwargs


# ============ Anthropic 路径 ============


class _AnthropicStub:
    """Anthropic SDK 客户端桩：捕获 create kwargs，返回预置响应。"""

    def __init__(self, response: Any):
        self.response = response
        self.kwargs: dict[str, Any] = {}

    @property
    def messages(self) -> _AnthropicStub:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _anthropic_response(content: str, stop_reason: str = "end_turn", usage: Any = None) -> Any:
    block = SimpleNamespace(type="text", text=content)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, usage=usage)


def _anthropic_client(response: Any) -> tuple[AnthropicClient, _AnthropicStub]:
    client = AnthropicClient(model="claude-sonnet-5", api_key="x", base_url="https://x")
    stub = _AnthropicStub(response)
    client._client = stub  # type: ignore[assignment]
    return client, stub


def test_anthropic_max_tokens_parameterized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_tokens 注入（None → 默认 4096 保留）；json_mode 无 response_format。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, stub = _anthropic_client(_anthropic_response("hi"))
    client.chat([{"role": "user", "content": "hi"}], max_tokens=2048, json_mode=True)
    assert stub.kwargs["max_tokens"] == 2048
    assert "response_format" not in stub.kwargs  # Anthropic 无 json_object 能力，忽略

    client2, stub2 = _anthropic_client(_anthropic_response("hi"))
    client2.chat([{"role": "user", "content": "hi"}])
    assert stub2.kwargs["max_tokens"] == 4096


def test_anthropic_stop_reason_max_tokens_mapped_to_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic 截断信号 stop_reason=="max_tokens" → 规整为 "length"（截断告警）。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, _ = _anthropic_client(_anthropic_response("...", stop_reason="max_tokens"))
    resp = client.chat([{"role": "user", "content": "hi"}], call_site="agent.loop")
    assert resp.finish_reason == "length"


def test_anthropic_stop_reason_normal_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正常结束 stop_reason 原样透传。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client, _ = _anthropic_client(_anthropic_response("done", stop_reason="end_turn"))
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.finish_reason == "end_turn"


def test_anthropic_usage_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic usage 三键规整进 LLMResponse。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=30)
    client, _ = _anthropic_client(_anthropic_response("hi", usage=usage))
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.usage is not None
    assert resp.usage.to_dict() == {
        "prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 30,
    }
