"""LLM 流式改造单测（v1.0 P1：chat() stream + on_delta）。

铁律：零真实 LLM 调用——OpenAI/Anthropic SDK 客户端用 Fake 对象注入
（_client 直赋绕过惰性 property）；usage 台账统一出口回归（流式/非流式同落盘）。
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from auto2dlabel.agent import llm as llm_mod
from auto2dlabel.agent.llm import (
    AnthropicClient,
    LLMClient,
    LLMResponse,
    OpenAIClient,
    UsageStats,
    load_usage_logs,
)


class _StreamClient(LLMClient):
    """记录 chat() 透传参数并手动触发增量的 Fake（走统一出口台账）。"""

    def __init__(self) -> None:
        super().__init__(model="deepseek-chat")
        self.last_stream: bool | None = None
        self.last_on_delta: Callable[[str], None] | None = None

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.last_stream = stream
        self.last_on_delta = on_delta
        if stream and on_delta:
            on_delta('{"a":')
            on_delta("1}")
        return LLMResponse(
            content='{"a":1}',
            usage=UsageStats(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
        )


def test_chat_stream_forwards_and_logs_ledger(tmp_path, monkeypatch) -> None:
    """stream=True → _chat 收到 stream/on_delta；增量回调触发；台账照常落盘。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client = _StreamClient()
    deltas: list[str] = []

    resp = client.chat(
        [{"role": "user", "content": "hi"}],
        json_mode=True,
        call_site="test.stream",
        stream=True,
        on_delta=deltas.append,
    )

    assert client.last_stream is True
    assert deltas == ['{"a":', "1}"], "增量回调序列不符"
    assert resp.content == '{"a":1}', "全量响应不变（解析兜底）"
    assert resp.usage and resp.usage.prompt_tokens == 10
    entries = load_usage_logs(tmp_path / "usage.jsonl")
    assert len(entries) == 1 and entries[0]["call_site"] == "test.stream"


def test_chat_stream_false_keeps_old_path() -> None:
    """stream=False（默认）→ 旧路径零变化；on_delta 忽略。"""
    client = _StreamClient()
    client.chat([{"role": "user", "content": "hi"}], on_delta=lambda s: None)
    assert client.last_stream is False


# ---------- OpenAI 流式累积（Fake SDK 注入） ----------


def _chunk(content: str | None, finish: str | None = None, usage: Any = None,
           tool_delta: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_delta),
                                 finish_reason=finish)],
        usage=usage,
    )


def test_openai_stream_accumulates_content_tools_usage() -> None:
    """OpenAI 流式：content/tool_calls 增量累积、末 chunk usage、回调序列。"""
    chunks = [
        _chunk('{"a"'),
        _chunk(
            ":1",
            tool_delta=[
                SimpleNamespace(
                    index=0, id="call_1", function=SimpleNamespace(name="detect", arguments='{"x"')
                )
            ],
        ),
        _chunk(
            "}",
            tool_delta=[
                SimpleNamespace(
                    index=0, id=None, function=SimpleNamespace(name=None, arguments=":1}")
                )
            ],
        ),
        _chunk(None, finish="stop", usage=SimpleNamespace(prompt_tokens=11, completion_tokens=6)),
    ]

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> list[Any]:
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            assert kwargs["response_format"] == {"type": "json_object"}, "json_mode 与 stream 正交"
            return chunks

    client = OpenAIClient(model="gpt-4o", api_key="sk-test")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    deltas: list[str] = []

    resp = client.chat(
        [{"role": "user", "content": "hi"}],
        json_mode=True,
        call_site="test.openai.stream",
        stream=True,
        on_delta=deltas.append,
    )

    assert resp.content == '{"a":1}'
    assert deltas == ['{"a"', ":1", "}"]
    assert resp.finish_reason == "stop"
    assert resp.usage == UsageStats(prompt_tokens=11, completion_tokens=6)
    assert resp.tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "detect", "arguments": '{"x":1}'}}
    ]


def test_openai_stream_no_usage_chunk() -> None:
    """末 chunk 无 usage（provider 不带 stream_options 兼容）→ usage=None 不炸。"""
    chunks = [_chunk("hi", finish="stop")]

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> list[Any]:
            return chunks

    client = OpenAIClient(model="gpt-4o", api_key="sk-test")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    resp = client.chat([{"role": "user", "content": "hi"}], stream=True)
    assert resp.content == "hi" and resp.usage is None and resp.finish_reason == "stop"


# ---------- Anthropic 流式（Fake stream manager 注入） ----------


class _FakeStreamManager:
    def __init__(self, final: Any) -> None:
        self._final = final

    def __enter__(self) -> _FakeStreamManager:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    @property
    def text_stream(self) -> Any:
        return iter(["hel", "lo"])

    def get_final_message(self) -> Any:
        return self._final


def test_anthropic_stream_reuses_final_message_parsing() -> None:
    """Anthropic 流式：text_stream 增量回调 + final message 共用尾部解析。"""
    final = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="tu_1", name="detect", input={"x": 1}),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        stop_reason="end_turn",
    )

    class _FakeMessages:
        def stream(self, **kwargs: Any) -> _FakeStreamManager:
            return _FakeStreamManager(final)

    client = AnthropicClient(model="claude-sonnet-5", api_key="sk-test")
    client._client = SimpleNamespace(messages=_FakeMessages())
    deltas: list[str] = []

    resp = client.chat([{"role": "user", "content": "hi"}], stream=True, on_delta=deltas.append)

    assert deltas == ["hel", "lo"]
    assert resp.content == "hello"
    assert resp.tool_calls == [
        {"id": "tu_1", "type": "function", "function": {"name": "detect", "arguments": '{"x": 1}'}}
    ]
    assert resp.finish_reason == "end_turn"
    assert resp.usage == UsageStats(prompt_tokens=12, completion_tokens=7)


def test_anthropic_stream_stop_reason_max_tokens_maps_to_length() -> None:
    """Anthropic 流式截断信号（max_tokens）→ 规整为 length（台账告警口径一致）。"""
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="truncated")],
        usage=None,
        stop_reason="max_tokens",
    )

    class _FakeMessages:
        def stream(self, **kwargs: Any) -> _FakeStreamManager:
            return _FakeStreamManager(final)

    client = AnthropicClient(model="claude-sonnet-5", api_key="sk-test")
    client._client = SimpleNamespace(messages=_FakeMessages())

    resp = client.chat([{"role": "user", "content": "hi"}], stream=True)
    assert resp.finish_reason == "length"
