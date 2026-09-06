"""LLM 客户端抽象层。

统一封装 OpenAI / Anthropic / DeepSeek 等模型的调用接口。
Agent Loop 通过此模块与 LLM 交互，不直接依赖任何特定 SDK。

v0.6 Phase 4（LLM Harness，2026-08-30）：
- UsageStats：usage 跨 provider 规整（OpenAI/DeepSeek/Anthropic 口径统一）
- chat() 统一出口：usage 台账落盘（logs/llm_usage.jsonl）+ 截断可见性
- max_tokens 分级 / json_mode / call_site（cost-report 按调用点聚合）

v1.0 P1（Agentic 交互化，2026-09-02）：
- chat() 加 stream: bool + on_delta 增量回调（流式展示；响应仍全量返回，
  usage 台账统一出口不变，json_mode 与 stream 正交——流式展示 + 全量解析兜底）
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

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
    """模型 → 单价行；未知模型返回 None（不臆造单价）。

    查找顺序（v1.0 P2）：代码精确表 → providers.yaml 中 model 匹配条目的
    price → env AUTOLABEL_LLM_PRICE 兜底 → None（台账记 null）。
    """
    if model in PRICE_RMB_PER_1M:
        return PRICE_RMB_PER_1M[model]
    for spec in list_providers():
        if spec.model == model and spec.price is not None:
            return dict(spec.price)
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
    provider: str = "",
) -> Path:
    """追加一条 usage 记录到 JSONL 台账；返回台账路径（文件不存在则创建）。"""
    target = path or USAGE_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "call_site": call_site,
        "model": model,
        "provider": provider,
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
    """按 call_site + model + provider 聚合台账 → 排序列表（总费用降序）。

    每行：{call_site, model, provider, calls, prompt_tokens, completion_tokens,
    cached_tokens, cache_hit_rate, cost_rmb}——cost-report 与测试共用单一事实源。
    旧行无 provider 字段 → 归入 "-"（不迁移历史，容错展示）。
    """
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in entries:
        key = (
            str(e.get("call_site", "unknown")),
            str(e.get("model", "?")),
            # provider 归一：缺键（旧行）与空串（手构造客户端 v0.6 落盘）同归 "-"
            str(e.get("provider") or "-"),
        )
        row = agg.setdefault(key, {
            "call_site": key[0],
            "model": key[1],
            "provider": key[2],
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
        if cost is not None:
            try:
                row["cost_rmb"] += float(cost)
            except (TypeError, ValueError):
                pass  # 脏行费用（非数字，如字符串）跳过该行 cost，不阻断其余聚合
    rows = list(agg.values())
    for row in rows:
        row["cache_hit_rate"] = (
            round(row["cached_tokens"] / row["prompt_tokens"], 4)
            if row["prompt_tokens"] else 0.0
        )
        row["cost_rmb"] = round(row["cost_rmb"], 6)
    rows.sort(key=lambda r: (-r["cost_rmb"], -r["calls"]))
    return rows


# 推理模型识别标记（模型名子串）：reasoning 与 content 共享 max_tokens 预算，
# 低温下推理链吃满预算产出空 content（实测 deepseek-v4-flash temp=0.1 →
# reasoning 1023/content 0 vs temp=1.0 → 569/834）——统一出口钳制（见 chat()）
_REASONING_MARKERS = ("reasoner", "thinking", "v4-", "o1", "o3", "r1-")
# 推理模型最小输出预算：<4096 时 reasoning 可能占满导致 content 空
REASONING_MIN_MAX_TOKENS = 4096


def _is_reasoning_model(model: str) -> bool:
    """按模型名判定推理模型（DeepSeek reasoner/v4 系、OpenAI o 系等）。"""
    name = model.lower()
    return any(marker in name for marker in _REASONING_MARKERS)


class LLMClient:
    """LLM 客户端基类。

    子类实现 _chat 方法，适配不同模型 API。
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "",
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        # provider 注册名（v1.0 P2 台账维度；默认 "" 向后兼容手构造客户端）
        self.provider_name = provider_name

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = False,
        call_site: str = "unknown",
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
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
            stream: True 时走 provider 流式传输（v1.0 P1）。响应仍全量返回
                （含 usage/tool_calls/finish_reason），台账统一出口不变；
                与 json_mode 正交——流式展示 + 全量解析兜底。
            on_delta: 流式增量回调（仅 stream=True 时触发；False 时忽略）。

        Returns:
            标准化的 LLMResponse（usage 已规整）。
        """
        # 推理模型钳制（v1.0 P1 bugfix，统一出口一处生效）：temperature 显式
        # 传 0 的调用点（planner/dialog 确定性 JSON）在推理模型下低温即空响应；
        # max_tokens 预算 reasoning 与 content 共享，调用点按非推理模型配的
        # 小预算（1024）会被推理链吃满——均在此放大/钳制，非推理模型零影响
        if _is_reasoning_model(self.model):
            temperature = 1.0
            if max_tokens is not None:
                max_tokens = max(max_tokens, REASONING_MIN_MAX_TOKENS)
        response = self._chat(
            messages, tools, temperature, max_tokens, json_mode, stream, on_delta
        )
        if response.usage is not None:
            try:
                log_llm_usage(
                    call_site=call_site,
                    model=self.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    cost_rmb=estimate_cost_rmb(self.model, response.usage),
                    provider=self.provider_name,
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
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @property
    def has_credentials(self) -> bool:
        """是否有可用 API 凭据（v0.6「无 key 零影响」判定点）。

        无 key 时调用方（chat 对话式解析、缺失参数回填）走代码级兜底，
        零 LLM 调用；SDK 客户端不构造。

        占位语义：create_client 对 api_key_env: null 的本地免 key 端点
        （Ollama/vLLM 等，OpenAI/Anthropic SDK 均要求非空 key）注入占位串
        "EMPTY"——本判定对占位 key 恒 True，即本地端点视作「总有凭据
        可用」，不触发代码级兜底。
        """
        return bool(self.api_key)


class OpenAIClient(LLMClient):
    """OpenAI 兼容 API 客户端。

    适用于 OpenAI、Qwen、DeepSeek 等兼容 OpenAI API 的模型。
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "",
    ):
        super().__init__(
            model=model, api_key=api_key, base_url=base_url, provider_name=provider_name
        )
        self._client: Any = None  # 惰性 SDK 客户端（类型 Any：测试 Fake 注入零依赖 stub）

    @property
    def client(self) -> Any:
        if self._client is None:
            import os

            import httpx
            from openai import OpenAI

            # LLM 调用超时（2026-09-06 实测 F6：全量 100 条中 case-092 挂满
            # 600s——SDK 默认 timeout 600s + max_retries 2，DeepSeek API 抖动
            # 时单次调用最长 ~30min，整条命令无感知卡死）。分相超时：
            # connect 10s / read 120s（流式按块间隔重置，不影响流式）/ 重试 1 次。
            # key 的 env 兜底只留手构造客户端（provider_name==""，v0.6 兼容）：
            # registry 创建（provider_name 非空）key 已在 create_client 按
            # spec.api_key_env 解析——此处若再兜底 env 会跨 provider 串扰
            # （DEEPSEEK_API_KEY 缺失而 OPENAI_API_KEY 存在时把 OpenAI key
            # 发给 DeepSeek）。base_url 的 env 兜底保留（无串扰风险）。
            api_key = (
                self.api_key
                if self.provider_name
                else (self.api_key or os.environ.get("OPENAI_API_KEY"))
            )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
                timeout=httpx.Timeout(10.0, read=120.0, write=60.0, pool=10.0),
                max_retries=1,
            )
        return self._client

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

        if stream:
            # 流式：usage 需 stream_options 显式开启（末 chunk 携带）；
            # tool_calls 按 index 增量累积，finish_reason 末 chunk 置位
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            parts: list[str] = []
            tc_acc: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            usage: UsageStats | None = None
            for chunk in self.client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None):
                    usage = UsageStats.from_openai(chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if delta is None:
                    continue
                if delta.content:
                    parts.append(delta.content)
                    if on_delta:
                        on_delta(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tc_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
            tool_calls = [
                {
                    "id": acc["id"],
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
                for acc in tc_acc.values()
            ] or None
            return LLMResponse(
                content="".join(parts) or None,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
            )

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

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "",
    ):
        super().__init__(
            model=model, api_key=api_key, base_url=base_url, provider_name=provider_name
        )
        self._client: Any = None  # 惰性 SDK 客户端（类型 Any：测试 Fake 注入零依赖 stub）

    @property
    def client(self) -> Any:
        if self._client is None:
            import os

            import anthropic

            # key 的 env 兜底语义与 OpenAIClient.client 相同（见上）：只留
            # 手构造客户端（provider_name==""，v0.6 兼容）；registry 创建时
            # key 已在 create_client 解析，再兜底 env 会把 ANTHROPIC_API_KEY
            # 串给同名以外的 provider（如本地 OpenAI 兼容端点）。base_url 保留。
            api_key = (
                self.api_key
                if self.provider_name
                else (self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
            )
            self._client = anthropic.Anthropic(
                api_key=api_key,
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
        stream: bool = False,
        on_delta: Callable[[str], None] | None = None,
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

        if stream:
            # 流式：text_stream 增量回调；get_final_message 与尾部解析共用
            # （Message 同构：content blocks / usage / stop_reason 全保留）
            with self.client.messages.stream(**kwargs) as stream_obj:
                for text in stream_obj.text_stream:
                    if on_delta:
                        on_delta(text)
                response = stream_obj.get_final_message()
        else:
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


# ================================================================
# provider 注册表（v1.0 P2）：providers.yaml 连接配置单一事实源
# ================================================================

PROVIDERS_YAML_PATH = Path(__file__).resolve().parent.parent / "configs" / "providers.yaml"


@dataclass(frozen=True)
class ProviderSpec:
    """provider 连接条目（providers.yaml 或内置兜底表）。

    api_key_env 只引用环境变量名（密钥不入 yaml）；None = 本地免 key 端点。
    """

    name: str
    provider: str  # 客户端实现："openai" | "anthropic"
    model: str
    model_env: str | None = None  # 可选：env 覆盖模型名
    base_url: str | None = None
    base_url_env: str | None = None  # 可选：env 覆盖 base_url
    api_key_env: str | None = None
    price: dict[str, float] | None = None  # {prompt, cached, completion} 元/百万 token


# 内置兜底表（yaml 缺失/损坏时保持可用——安装版漏打 package-data 也功能不崩，
# 只退化到这三条）；与 providers.yaml 条目按 name 去重（yaml 优先）。
_BUILTIN_PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        name="deepseek",
        provider="openai",  # DeepSeek 兼容 OpenAI API
        model="deepseek-chat",
        model_env="DEEPSEEK_MODEL",
        base_url="https://api.deepseek.com",
        base_url_env="DEEPSEEK_BASE_URL",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    ProviderSpec(name="openai", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"),
    ProviderSpec(
        name="anthropic",
        provider="anthropic",
        model="claude-sonnet-5",
        api_key_env="ANTHROPIC_API_KEY",
    ),
]

# name → 内置条目（_merge_spec 的字段级回退基准；list_providers 用）
_BUILTIN_BY_NAME: dict[str, ProviderSpec] = {s.name: s for s in _BUILTIN_PROVIDERS}

_CLIENT_CLASSES: dict[str, type[LLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def _normalize_price(raw: Any) -> dict[str, float] | None:
    """yaml price → 完整三键单价表（缺失键补 0，estimate_cost_rmb 零 KeyError）。

    price: {}（空 dict）→ None——0 元与未知不混同（设计口径「缺省 = 台账记
    null」）；非 dict/坏值 → None（同款容错，绝不 raise）。
    """
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return {
            "prompt": float(raw.get("prompt", 0.0)),
            "cached": float(raw.get("cached", 0.0)),
            "completion": float(raw.get("completion", 0.0)),
        }
    except (TypeError, ValueError):
        return None


def _opt_env_field(raw: Any) -> str | None:
    """yaml env 引用字段（model_env/api_key_env/base_url_env/base_url）类型容错。

    yaml 误写列表/数字（如 `api_key_env: [DEEPSEEK_API_KEY]`）→ None，
    与 price 同款容错绝不 raise；空串同 None（无意义的空 env 名/URL）。
    """
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _load_provider_config(path: Path) -> list[ProviderSpec]:
    """读 providers.yaml → ProviderSpec 列表；缺失/损坏 → []（内置表兜底，绝不 raise）。"""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []
    specs: list[ProviderSpec] = []
    seen: set[str] = set()
    for raw in data.get("providers") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        name = str(raw["name"])
        # 重复 name 静默后者胜 → 显式告警（#21；yaml 笔误不静默）
        if name in seen:
            import logging

            logging.getLogger(__name__).warning(
                "providers.yaml 重复 name=%r——后者覆盖前者，请检查是否笔误", name
            )
        seen.add(name)
        specs.append(
            ProviderSpec(
                name=name,
                provider=str(raw.get("provider", "openai")),
                model=str(raw.get("model") or ""),
                # env 引用字段类型容错（#10）：yaml 误写列表/数字 → None 绝不 raise
                model_env=_opt_env_field(raw.get("model_env")),
                base_url=_opt_env_field(raw.get("base_url")),
                base_url_env=_opt_env_field(raw.get("base_url_env")),
                api_key_env=_opt_env_field(raw.get("api_key_env")),
                price=_normalize_price(raw.get("price")),
            )
        )
    return specs


@lru_cache(maxsize=1)
def _load_default_providers() -> list[ProviderSpec]:
    """默认路径 providers.yaml 加载（缓存；测试经 path 参数注入 tmp 文件）。"""
    return _load_provider_config(PROVIDERS_YAML_PATH)


def load_provider_config(path: Path | None = None) -> list[ProviderSpec]:
    """provider 配置入口：显式 path 注入测试；默认读包内 providers.yaml。"""
    if path is not None:
        return _load_provider_config(path)
    return list(_load_default_providers())


def _merge_spec(yaml_spec: ProviderSpec, builtin_spec: ProviderSpec) -> ProviderSpec:
    """yaml 条目与内置同名时的字段级合并：yaml 中 None/空串字段回退内置值。

    #9（2026-09-07）：整条覆盖会让「只写 name + model 的局部覆盖 yaml」丢掉
    内置的 api_key_env/base_url 等字段——字段级回退后局部覆盖语义才成立
    （yaml 只声明想改的字段，其余继承内置）。ProviderSpec frozen，构造新实例。
    """
    def _pick(yaml_val: Any, builtin_val: Any) -> Any:
        if yaml_val is None or yaml_val == "":
            return builtin_val  # 未声明（None/空串）→ 内置对应字段值
        return yaml_val

    return ProviderSpec(
        name=yaml_spec.name,
        provider=_pick(yaml_spec.provider, builtin_spec.provider),
        model=_pick(yaml_spec.model, builtin_spec.model),
        model_env=_pick(yaml_spec.model_env, builtin_spec.model_env),
        base_url=_pick(yaml_spec.base_url, builtin_spec.base_url),
        base_url_env=_pick(yaml_spec.base_url_env, builtin_spec.base_url_env),
        api_key_env=_pick(yaml_spec.api_key_env, builtin_spec.api_key_env),
        price=_pick(yaml_spec.price, builtin_spec.price),
    )


def list_providers() -> list[ProviderSpec]:
    """可用 provider 列表：providers.yaml 条目 ∪ 内置三条目（按 name 去重）。

    与内置同名的 yaml 条目走字段级合并（_merge_spec，yaml 优先；None/空串
    回退内置）——局部覆盖 yaml 不丢内置的 env/URL 默认值。
    """
    by_name = {s.name: s for s in _BUILTIN_PROVIDERS}
    for spec in load_provider_config():
        builtin = _BUILTIN_BY_NAME.get(spec.name)
        if builtin is not None:
            spec = _merge_spec(spec, builtin)
        by_name[spec.name] = spec
    return list(by_name.values())


def create_client(
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """工厂函数：根据 provider 注册表条目创建 LLM 客户端（v1.0 P2）。

    解析顺序（显式参数 > env > yaml/内置条目默认值，与 v0.6 行为逐字等价）：
    - model: 显式 model > spec.model_env 的 env > spec.model
    - api_key: 显式 > spec.api_key_env 的 env（无 key 且 api_key_env is None
      的本地端点 → 占位 key "EMPTY"，has_credentials 恒 True）
    - base_url: 显式 > spec.base_url_env 的 env > spec.base_url

    Args:
        provider: 注册名（"deepseek" | "openai" | "anthropic" | yaml 自定义条目）。
        model: 模型名称（可选，覆盖注册表条目）。
        api_key: API key（可选，默认读环境变量）。
        base_url: API base URL（可选，用于兼容其他 API 服务）。

    Returns:
        LLMClient 实例（provider_name 置为注册名，台账按 provider 维度记账）。
    """
    spec = next((s for s in list_providers() if s.name == provider), None)
    if spec is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {[s.name for s in list_providers()]}"
        )
    cls = _CLIENT_CLASSES.get(spec.provider)
    if cls is None:
        raise ValueError(
            f"Provider '{spec.name}' 的 provider 实现 '{spec.provider}' 不受支持"
            f"（可用: {list(_CLIENT_CLASSES.keys())}）"
        )

    # 各 provider 统一在创建时解析环境变量 key（v0.6「无 key 零影响」：
    # has_credentials 判定必须与实际可用 key 同源——避免「判定无凭据走代码
    # 兜底」与「chat 时 env 注入其实有 key」两个口径分叉）。
    # 因此 registry 创建的客户端（provider_name 非空）在 client property 中
    # 不再做 env 兜底——key 解析只此一处（单一事实源），防跨 provider 串扰：
    # DEEPSEEK_API_KEY 缺失而 OPENAI_API_KEY 存在时不会把 OpenAI key 发给
    # DeepSeek（env 兜底仅留手构造客户端，见 OpenAIClient/AnthropicClient）
    resolved_model = (
        model or (os.environ.get(spec.model_env) if spec.model_env else None) or spec.model
    )
    resolved_key = api_key or (os.environ.get(spec.api_key_env) if spec.api_key_env else None)
    if resolved_key is None and spec.api_key_env is None:
        resolved_key = "EMPTY"  # 本地免 key 端点（OpenAI SDK 要求非空占位）
    resolved_base = (
        base_url
        or (os.environ.get(spec.base_url_env) if spec.base_url_env else None)
        or spec.base_url
    )
    return cls(
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_base,
        provider_name=spec.name,
    )
