"""LLM Harness usage 台账测试（v0.6 Phase 4，零真实 API）。

覆盖：UsageStats 跨 provider 规整、费用折算纯函数、JSONL 台账落盘/读取/聚合、
chat() 统一出口（台账落盘 / 台账失败不阻断 / 截断告警可见）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from auto2dlabel.agent import llm as llm_mod
from auto2dlabel.agent.llm import (
    LLMClient,
    LLMResponse,
    UsageStats,
    aggregate_usage,
    estimate_cost_rmb,
    load_usage_logs,
    log_llm_usage,
)

# ============ UsageStats 规整 ============


def test_usage_from_openai_deepseek_style() -> None:
    """DeepSeek 口径（prompt_cache_hit_tokens）→ cached_tokens 正确提取。"""
    usage = _ns(prompt_tokens=1000, completion_tokens=500, prompt_cache_hit_tokens=400)
    s = UsageStats.from_openai(usage)
    assert s is not None
    assert s.to_dict() == {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 400}


def test_usage_from_openai_nested_cached() -> None:
    """OpenAI 口径（prompt_tokens_details.cached_tokens）→ cached_tokens 提取。"""
    details = _ns(cached_tokens=123)
    usage = _ns(prompt_tokens=1000, completion_tokens=500, prompt_tokens_details=details)
    s = UsageStats.from_openai(usage)
    assert s is not None and s.cached_tokens == 123


def test_usage_from_openai_none() -> None:
    assert UsageStats.from_openai(None) is None


def test_usage_from_anthropic() -> None:
    """Anthropic 三键（input/output/cache_read_input_tokens）规整。"""
    usage = _ns(input_tokens=1000, output_tokens=500, cache_read_input_tokens=400)
    s = UsageStats.from_anthropic(usage)
    assert s is not None
    assert s.to_dict() == {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 400}


def test_usage_from_anthropic_none() -> None:
    assert UsageStats.from_anthropic(None) is None


# ============ 费用折算 ============


def test_estimate_cost_deepseek_chat() -> None:
    """deepseek-chat 单价（2/0.5/8 元每百万）计价：miss 按 prompt、hit 按 cached。"""
    usage = UsageStats(prompt_tokens=1000, completion_tokens=500, cached_tokens=400)
    # (600*2 + 400*0.5 + 500*8) / 1e6 = 5400 / 1e6
    assert estimate_cost_rmb("deepseek-chat", usage) == pytest.approx(0.0054)


def test_estimate_cost_reasoner() -> None:
    usage = UsageStats(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=0)
    assert estimate_cost_rmb("deepseek-reasoner", usage) == pytest.approx(4.0)


def test_estimate_cost_unknown_model_returns_none() -> None:
    """未知模型不臆造单价 → None（台账记 null）。"""
    usage = UsageStats(prompt_tokens=10)
    assert estimate_cost_rmb("unknown-model", usage) is None


def test_estimate_cost_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """env AUTOLABEL_LLM_PRICE="prompt,cached,completion" 覆盖防调价漂移。"""
    monkeypatch.setenv("AUTOLABEL_LLM_PRICE", "1,0.25,4")
    usage = UsageStats(prompt_tokens=1000, completion_tokens=1000, cached_tokens=0)
    # (1000*1 + 1000*4) / 1e6 = 0.005
    assert estimate_cost_rmb("some-future-model", usage) == pytest.approx(0.005)


# ============ 台账落盘 / 读取 / 聚合 ============


def test_log_and_load_roundtrip(tmp_path: Path) -> None:
    """log_llm_usage → load_usage_logs 往返：字段齐全。"""
    path = tmp_path / "usage.jsonl"
    log_llm_usage(
        call_site="planner.parse",
        model="deepseek-chat",
        usage=UsageStats(prompt_tokens=100, completion_tokens=50, cached_tokens=20),
        finish_reason="stop",
        cost_rmb=0.00123,
        path=path,
    )
    entries = load_usage_logs(path)
    assert len(entries) == 1
    e = entries[0]
    assert e["call_site"] == "planner.parse"
    assert e["model"] == "deepseek-chat"
    assert e["finish_reason"] == "stop"
    assert e["prompt_tokens"] == 100
    assert e["completion_tokens"] == 50
    assert e["cached_tokens"] == 20
    assert e["cost_rmb"] == pytest.approx(0.00123)
    assert isinstance(e["ts"], float)


def test_load_skips_corrupted_lines(tmp_path: Path) -> None:
    """损坏行跳过（不阻断报告），空行忽略。"""
    path = tmp_path / "usage.jsonl"
    path.write_text(
        '{"call_site": "a", "model": "m", "prompt_tokens": 1}\n'
        "not-json\n"
        "\n"
        '{"call_site": "b", "model": "m", "prompt_tokens": 2}\n',
        encoding="utf-8",
    )
    entries = load_usage_logs(path)
    assert [e["call_site"] for e in entries] == ["a", "b"]


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_usage_logs(tmp_path / "nope.jsonl") == []


def test_aggregate_usage_groups_and_sorts(tmp_path: Path) -> None:
    """按 (call_site, model) 聚合：调用数/token 求和、命中率、费用降序。"""
    path = tmp_path / "usage.jsonl"
    # planner.parse × 2（费用最高）、agent.loop × 1
    for _ in range(2):
        log_llm_usage(
            call_site="planner.parse",
            model="deepseek-chat",
            usage=UsageStats(prompt_tokens=500, completion_tokens=100, cached_tokens=100),
            cost_rmb=0.005,
            path=path,
        )
    log_llm_usage(
        call_site="agent.loop",
        model="deepseek-chat",
        usage=UsageStats(prompt_tokens=300, completion_tokens=50, cached_tokens=0),
        cost_rmb=0.001,
        path=path,
    )
    rows = aggregate_usage(load_usage_logs(path))
    assert len(rows) == 2
    assert rows[0]["call_site"] == "planner.parse"  # 费用降序
    assert rows[0]["calls"] == 2
    assert rows[0]["prompt_tokens"] == 1000
    assert rows[0]["completion_tokens"] == 200
    assert rows[0]["cached_tokens"] == 200
    assert rows[0]["cache_hit_rate"] == 0.2
    assert rows[1]["call_site"] == "agent.loop"
    assert rows[1]["cache_hit_rate"] == 0.0


# ============ chat() 统一出口 ============


class _UsageClient(LLMClient):
    """返回固定 usage 的 Fake 客户端（走 LLMClient.chat 统一出口）。"""

    def __init__(self, usage: UsageStats | None, finish_reason: str | None = None):
        super().__init__(model="deepseek-chat")
        self._usage = usage
        self._finish_reason = finish_reason

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        return LLMResponse(
            content="{}", usage=self._usage, finish_reason=self._finish_reason
        )


def test_chat_logs_usage_to_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """chat() 带 usage → 台账 JSONL 落盘（含调用点/费用）。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client = _UsageClient(UsageStats(prompt_tokens=1000, completion_tokens=500, cached_tokens=0))
    client.chat([{"role": "user", "content": "hi"}], call_site="test.site")

    entries = load_usage_logs(tmp_path / "usage.jsonl")
    assert len(entries) == 1
    assert entries[0]["call_site"] == "test.site"
    assert entries[0]["model"] == "deepseek-chat"
    # (1000*2 + 500*8) / 1e6 = 0.006
    assert entries[0]["cost_rmb"] == pytest.approx(0.006)


def test_chat_ledger_failure_never_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """台账落盘异常吞掉，业务响应正常返回（台账失败绝不阻断红线）。"""
    def _boom(*args: Any, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(llm_mod, "log_llm_usage", _boom)
    client = _UsageClient(UsageStats(prompt_tokens=1))
    resp = client.chat([{"role": "user", "content": "hi"}], call_site="test.site")
    assert resp.content == "{}"


def test_chat_truncation_warning_visible(caplog: pytest.LogCaptureFixture) -> None:
    """finish_reason=length（max_tokens 截断）→ 告警日志可见（不静默）。"""
    import logging

    client = _UsageClient(None, finish_reason="length")
    with caplog.at_level(logging.WARNING, logger="auto2dlabel"):
        client.chat([{"role": "user", "content": "hi"}], call_site="test.site")
    assert any("截断" in r.message for r in caplog.records)


def test_chat_no_usage_no_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 usage 响应（Fake/部分 provider）→ 零台账写入、零告警。"""
    monkeypatch.setattr(llm_mod, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    client = _UsageClient(None)
    client.chat([{"role": "user", "content": "hi"}], call_site="test.site")
    assert load_usage_logs(tmp_path / "usage.jsonl") == []


def _ns(**kwargs: Any) -> Any:
    """轻量命名空间桩（SimpleNamespace 的省 import 别名）。"""
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)
