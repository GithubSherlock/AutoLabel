"""Benchmark 套件 —— 9 数据集自动标注 vs GT 评测（python -m 运行）。

库载入重构：json / time / datetime / numpy 在此集中导入（显式 as 再导出，
mypy no-implicit-reexport 要求），子模块通过
`from auto2dlabel.benchmarks import datetime, json, np, time` 引用。
sys / pathlib.Path 例外保留在各文件顶部：直接运行引导（PROJECT_ROOT
计算 + sys.path.insert）须先于包导入完成，无法转移。
"""

import json as json
import time as time
from datetime import datetime as datetime

import numpy as np

# __all__ 显式再导出：mypy no-implicit-reexport 不认 `import numpy as np`
# （别名与原名不同），须以 __all__ 声明；ruff F401 同样依据 __all__ 豁免。
__all__ = ["datetime", "json", "np", "time"]
