"""auto3dlabel 测试 conftest：把 tests/ 加入 sys.path（包内测试，同 auto2dlabel）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # AutoLabel/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
