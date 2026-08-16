"""pytest 共享配置与 fixtures。

将仓库根加入 sys.path，保证从任意目录运行 pytest 均可导入 auto2dlabel；
提供共享的样例图 fixture（tests/data/sample.png，640×480 合成图）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SAMPLE_IMAGE = Path(__file__).resolve().parent / "data" / "sample.png"


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """共享样例图：640×480 合成图（深色矩形/椭圆目标）。"""
    if not SAMPLE_IMAGE.exists():
        pytest.fail(f"样例图缺失: {SAMPLE_IMAGE}")
    return SAMPLE_IMAGE
