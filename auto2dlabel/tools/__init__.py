"""Tool 模块 —— 可注册的标注工具。"""

from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry, registry

__all__ = ["Tool", "ToolRegistry", "registry"]
