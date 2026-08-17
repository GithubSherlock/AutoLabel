"""Auto2dLabel 测试。

库载入重构：json / sys / pathlib.Path / numpy 在此集中导入（显式 as 再导出 +
__all__，mypy no-implicit-reexport 要求），子模块通过
`from auto2dlabel.tests import Path, json, np, sys` 引用。
conftest.py 例外保留 sys/Path（pytest 引导 sys.path.insert 须先于包导入完成）。
"""

import json as json
import sys as sys
from pathlib import Path as Path

import numpy as np

# __all__ 显式再导出：mypy 不认 `import numpy as np`（别名与原名不同），
# 须以 __all__ 声明；ruff F401 同样依据 __all__ 豁免。
__all__ = ["Path", "json", "np", "sys"]
