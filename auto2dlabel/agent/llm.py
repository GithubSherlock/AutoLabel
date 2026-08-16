"""LLM 客户端抽象层。

统一封装 OpenAI / Anthropic / DeepSeek 等模型的调用接口。
Agent Loop 通过此模块与 LLM 交互，不直接依赖任何特定 SDK。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
class LLMResponse:
    """LLM 回复的标准化结构。"""

    content: str | None = None  # 文本回复
    tool_calls: list[dict[str, Any]] | None = None  # tool-use 请求

    @property
    def wants_tool_call(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


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
    ) -> LLMResponse:
        """发送对话请求到 LLM。

        Args:
            messages: 对话历史（OpenAI 格式）。
            tools: Tool 列表（OpenAI tool-use 格式）。
            temperature: 采样温度。

        Returns:
            标准化的 LLMResponse。
        """
        return self._chat(messages, tools, temperature)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> LLMResponse:
        raise NotImplementedError


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
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

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
    ) -> LLMResponse:
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
            "max_tokens": 4096,
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

        return LLMResponse(content=content or None, tool_calls=tool_calls)


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

    # DeepSeek 自动从环境变量读取 key 和 base_url
    if provider == "deepseek":
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(providers.keys())}")

    return cls(
        model=model or default_models[provider],
        api_key=api_key,
        base_url=base_url,
    )
