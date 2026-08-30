"""LLM 客户端抽象层。

统一封装 OpenAI / Anthropic / DeepSeek 等模型的调用接口。
Agent Loop 通过此模块与 LLM 交互，不直接依赖任何特定 SDK。

v0.6 Phase 4（LLM Harness，2026-08-30）：
- UsageStats：usage 跨 provider 规整（OpenAI/DeepSeek/Anthropic 口径统一）
- chat() 统一出口：usage 台账落盘（logs/llm_usage.jsonl）+ 截断可见性
- max_tokens 分级 / json_mode / call_site（cost-report 按调用点聚合）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto2dlabel.agent import json


def _load_dotenv() -> None:
    """自动加载 config/.env 文件。"""
    env_path = Path(__file__).resolve().parent.parent / "configs" / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass


# 模块导入时自动加载
_load_dotenv()


@dataclass
class UsageStats:
    """跨 provider 规整的 token 用量（零 SDK 依赖的纯数据结构）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # 前缀缓存命中（DeepSeek prompt_cache_hit / Anthropic cache_read）

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
        }

    @classmethod
    def from_openai(cls, usage: Any) -> UsageStats | None:
        """OpenAI 兼容 usage 对象 → UsageStats（None = 无 usage）。

        DeepSeek 返回 prompt_cache_hit_tokens / prompt_cache_miss_tokens；
        OpenAI 返回 prompt_tokens_details.cached_tokens——两者都尝试。
        """
        if usage is None:
            return None
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details else None
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=int(cached or 0),
        )

    @classmethod
    def from_anthropic(cls, usage: Any) -> UsageStats | None:
        """Anthropic usage 对象 → UsageStats（input/output/cache_read 三键）。"""
        if usage is None:
            return None
        return cls(
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        )


@dataclass
class LLMResponse:
    """LLM 回复的标准化结构。"""

    content: str | None = None  # 文本回复
    tool_calls: list[dict[str, Any]] | None = None  # tool-use 请求
    usage: UsageStats | None = None  # token 用量（v0.6 Phase 4 台账）
    finish_reason: str | None = None  # 原生结束原因（"length"=被 max_tokens 截断）

    @property
    def wants_tool_call(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


# ================================================================
# usage 台账（v0.6 Phase 4）：JSONL 落盘 + 费用折算 + 聚合
# ================================================================

USAGE_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "llm_usage.jsonl"

# DeepSeek 官方单价（元/百万 token；env 可覆盖防调价漂移）。
# deepseek-v4-flash 为 2026-08-17 起峰谷计费——本表取空闲档（保守下限），
# 高峰时段（北京 9-12 / 14-18）×2；账单级精度用 env AUTOLABEL_LLM_PRICE 覆盖。
PRICE_RMB_PER_1M: dict[str, dict[str, float]] = {
    "deepseek-chat": {"prompt": 2.0, "cached": 0.5, "completion": 8.0},
    "deepseek-reasoner": {"prompt": 4.0, "cached": 1.0, "completion": 16.0},
    "deepseek-v4-flash": {"prompt": 1.5, "cached": 0.05, "completion": 4.5},
}


def _price_row(model: str) -> dict[str, float] | None:
    """模型 → 单价行；未知模型返回 None（不臆造单价）。"""
    if model in PRICE_RMB_PER_1M:
        return PRICE_RMB_PER_1M[model]
    env_row = os.environ.get("AUTOLABEL_LLM_PRICE")  # "prompt,cached,completion"
    if env_row:
        try:
            parts = [float(x) for x in env_row.split(",")]
            if len(parts) == 3:
                return {"prompt": parts[0], "cached": parts[1], "completion": parts[2]}
        except ValueError:
            pass
    return None


def estimate_cost_rmb(model: str, usage: UsageStats | None) -> float | None:
    """单次调用折算费用（元）；无单价/无 usage → None（台账记 null）。

    计价：未命中缓存的 prompt 按 prompt 价、命中按 cached 价、completion 按输出价。
    """
    row = _price_row(model)
    if row is None or usage is None:
        return None
    prompt_miss = max(0, usage.prompt_tokens - usage.cached_tokens)
    return round(
        prompt_miss * row["prompt"] / 1e6
        + usage.cached_tokens * row["cached"] / 1e6
        + usage.completion_tokens * row["completion"] / 1e6,
        6,
    )


def log_llm_usage(
    call_site: str,
    model: str,
    usage: UsageStats,
    finish_reason: str | None = None,
    cost_rmb: float | None = None,
    path: Path | None = None,
) -> Path:
    """追加一条 usage 记录到 JSONL 台账；返回台账路径（文件不存在则创建）。"""
    target = path or USAGE_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "call_site": call_site,
        "model": model,
        "finish_reason": finish_reason,
        "cost_rmb": cost_rmb,
        **usage.to_dict(),
    }
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return target


def load_usage_logs(path: Path | None = None) -> list[dict[str, Any]]:
    """读台账 JSONL → 记录列表（损坏行跳过，空/缺文件 → []）。"""
    target = path or USAGE_LOG_PATH
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue  # 损坏行不阻断报告
    return entries


def aggregate_usage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 call_site + model 聚合台账 → 排序列表（总费用降序）。

    每行：{call_site, model, calls, prompt_tokens, completion_tokens, cached_tokens,
    cache_hit_rate, cost_rmb}——cost-report 与测试共用单一事实源。
    """
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        key = (str(e.get("call_site", "unknown")), str(e.get("model", "?")))
        row = agg.setdefault(key, {
            "call_site": key[0],
            "model": key[1],
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cost_rmb": 0.0,
        })
        row["calls"] += 1
        row["prompt_tokens"] += int(e.get("prompt_tokens", 0) or 0)
        row["completion_tokens"] += int(e.get("completion_tokens", 0) or 0)
        row["cached_tokens"] += int(e.get("cached_tokens", 0) or 0)
        cost = e.get("cost_rmb")
        row["cost_rmb"] += float(cost) if cost is not None else 0.0
    rows = list(agg.values())
    for row in rows:
        row["cache_hit_rate"] = (
            round(row["cached_tokens"] / row["prompt_tokens"], 4)
            if row["prompt_tokens"] else 0.0
        )
        row["cost_rmb"] = round(row["cost_rmb"], 6)
    rows.sort(key=lambda r: (-r["cost_rmb"], -r["calls"]))
    return rows


class LLMClient:
    """LLM 客户端基类。

    子类实现 _chat 方法，适配不同模型 API。
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = False,
        call_site: str = "unknown",
    ) -> LLMResponse:
        """发送对话请求到 LLM（统一出口：usage 台账 + 截断可见性）。

        Args:
            messages: 对话历史（OpenAI 格式）。
            tools: Tool 列表（OpenAI tool-use 格式）。
            temperature: 采样温度。
            max_tokens: 输出上限（None = provider 默认）；超限截断日志可见。
            json_mode: 请求 provider 的 JSON 输出模式（OpenAI 兼容
                response_format；Anthropic 无此能力则忽略）。
            call_site: 调用点标注（v0.6 Phase 4 台账聚合键）。

        Returns:
            标准化的 LLMResponse（usage 已规整）。
        """
        response = self._chat(messages, tools, temperature, max_tokens, json_mode)
        if response.usage is not None:
            try:
                log_llm_usage(
                    call_site=call_site,
                    model=self.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    cost_rmb=estimate_cost_rmb(self.model, response.usage),
                )
            except Exception:
                pass  # 台账失败绝不阻断业务调用
        if response.finish_reason == "length":
            import logging

            logging.getLogger("auto2dlabel").warning(
                "LLM 输出被 max_tokens 截断（call_site=%s, model=%s）——"
                "JSON 解析可能失败，若降级请检查该调用点上限",
                call_site,
                self.model,
            )
        return response

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        raise NotImplementedError

    @property
    def has_credentials(self) -> bool:
        """是否有可用 API 凭据（v0.6「无 key 零影响」判定点）。

        无 key 时调用方（chat 对话式解析、缺失参数回填）走代码级兜底，
        零 LLM 调用；SDK 客户端不构造。
        """
        return bool(self.api_key)


class OpenAIClient(LLMClient):
    """OpenAI 兼容 API 客户端。

    适用于 OpenAI、Qwen、DeepSeek 等兼容 OpenAI API 的模型。
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import os

            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
            )
        return self._client

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = None
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=UsageStats.from_openai(getattr(response, "usage", None)),
            finish_reason=choice.finish_reason,
        )


class AnthropicClient(LLMClient):
    """Anthropic API 客户端。

    使用原生 tool-use 协议。
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None, base_url: str | None = None):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import os

            import anthropic  # type: ignore  # 可选依赖，未安装时跳过静态解析

            self._client = anthropic.Anthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                base_url=self.base_url or os.environ.get("ANTHROPIC_BASE_URL"),
            )
        return self._client

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        # Anthropic 无 response_format json_object（json_mode 忽略，靠 prompt 约束）
        # Anthropic API 需要提取 system prompt
        system = ""
        anthropic_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system = msg["content"]
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }
                    ],
                })
            elif role == "assistant" and "tool_calls" in msg:
                tool_use_blocks = [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in msg["tool_calls"]
                ]
                anthropic_messages.append({"role": "assistant", "content": tool_use_blocks})
            else:
                anthropic_messages.append({"role": role, "content": msg["content"]})

        # 转换 tools 为 Anthropic 格式
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self.client.messages.create(**kwargs)

        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        # Anthropic 截断信号是 stop_reason == "max_tokens" → 规整为 "length"（与
        # OpenAI 口径一致，chat() 统一出口按此告警）
        stop_reason = getattr(response, "stop_reason", None)
        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            usage=UsageStats.from_anthropic(getattr(response, "usage", None)),
            finish_reason="length" if stop_reason == "max_tokens" else stop_reason,
        )


def create_client(
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """工厂函数：根据 provider 创建 LLM 客户端。

    Args:
        provider: "openai" | "anthropic" | "deepseek"
        model: 模型名称（provider 特定）。
        api_key: API key（可选，默认读环境变量）。
        base_url: API base URL（可选，用于兼容其他 API 服务）。

    Returns:
        LLMClient 实例。
    """
    providers: dict[str, type[LLMClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "deepseek": OpenAIClient,  # DeepSeek 兼容 OpenAI API
    }

    default_models: dict[str, str] = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-5",
        "deepseek": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    }

    # 各 provider 统一在创建时解析环境变量 key（v0.6「无 key 零影响」：
    # has_credentials 判定必须与实际可用 key 同源——openai/anthropic 原在
    # chat 时才读 env，会导致「判定无凭据走代码兜底」与「chat 时 env 注入
    # 其实有 key」两个口径分叉）
    api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
    # DeepSeek 另需 base_url
    if provider == "deepseek":
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(providers.keys())}")

    return cls(
        model=model or default_models[provider],
        api_key=api_key,
        base_url=base_url,
    )
