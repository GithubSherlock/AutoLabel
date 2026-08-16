"""Tool 注册表。

全局单例，管理所有已注册的 Tool 实例。Agent 通过注册表获取可用 Tool 列表。
"""

from __future__ import annotations

from typing import Any

from auto2dlabel.tools.base import Tool


class ToolRegistry:
    """Tool 注册表（单例模式）。

    负责 Tool 的注册、查找、以及导出为 LLM tool-use 格式。
    """

    _instance: ToolRegistry | None = None

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """获取全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: Tool) -> None:
        """注册一个 Tool。"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名称获取 Tool。"""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """返回所有已注册 Tool 的名称列表。"""
        return list(self._tools.keys())

    def call(self, name: str, **kwargs) -> Any:
        """调用指定 Tool。"""
        tool = self.get(name)
        return tool(**kwargs)

    def to_openai_tools(self) -> list[dict]:
        """导出为 OpenAI tool-use 格式。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> list[dict]:
        """导出为 Anthropic tool-use 格式。"""
        return [tool.to_anthropic_tool() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def clear(self) -> None:
        self._tools.clear()


# 全局注册表 instance
registry = ToolRegistry.get_instance()
