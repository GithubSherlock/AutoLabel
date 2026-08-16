"""Tool 基类定义。

参考 Anthropic/OpenAI tool-use 协议，所有 Tool 实现统一 schema。
v0.1 阶段保持轻量——不引入外部 Agent 框架依赖，自行实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """标注 Tool 的抽象基类。

    每个 Tool 对应一种标注能力（检测、分割、导出等）。

    Usage:
        class MyTool(Tool):
            name = "my_tool"
            description = "Does something useful."

            @property
            def input_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ],
                }

            def forward(self, **kwargs) -> Any:
                ...
    """

    name: str = ""
    description: str = ""

    @property
    def input_schema(self) -> dict:
        """返回 OpenAI/Anthropic 兼容的 tool input_schema。

        子类覆盖此方法定义输入参数。
        """
        return {"type": "object", "properties": {}, "required": []}

    def to_openai_tool(self) -> dict:
        """导出为 OpenAI tool-use 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic_tool(self) -> dict:
        """导出为 Anthropic tool-use 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @abstractmethod
    def forward(self, **kwargs) -> Any:
        """执行 Tool 的核心逻辑。子类必须实现。"""
        ...

    def __call__(self, **kwargs) -> Any:
        """调用 forward，提供统一的错误处理。"""
        return self.forward(**kwargs)

    def __repr__(self) -> str:
        return f"<Tool name={self.name}>"
