"""Tool 模块 —— 可注册的标注工具。

库载入重构：json / datetime / pathlib.Path 在此集中导入（显式 as 再导出，
mypy no-implicit-reexport 要求），子模块通过
`from auto2dlabel.tools import datetime, Path, json` 引用。
"""

import json as json
from datetime import datetime as datetime
from pathlib import Path as Path

from auto2dlabel.tools.base import Tool
from auto2dlabel.tools.registry import ToolRegistry, registry

__all__ = ["Tool", "ToolRegistry", "registry"]
