"""common.mask_to_bbox 回归测试 — box-prompted 链路（coco_seg/cityscapes 共用）。"""

from __future__ import annotations

import pytest

from auto2dlabel.benchmarks.common import mask_to_bbox
from auto2dlabel.tests import np


def _mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """构造 (h,w) 全零 mask，仅 [x1:x2, y1:y2) 区域为 True。"""
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class TestMaskToBbox:
    def test_regular_mask_bounds(self) -> None:
        """mask [0:4, 1:4) → bbox [1, 0, 3, 3]（x2/y2 = 最大像素下标，非 +1）。"""
        assert mask_to_bbox(_mask(10, 10, 1, 0, 4, 4)) == [1, 0, 3, 3]

    def test_empty_mask_returns_zero_box(self) -> None:
        """空 mask 不抛异常，返回 [0, 0, 0, 0]（box-prompted 派生 bbox 的退化防御）。"""
        assert mask_to_bbox(np.zeros((5, 5), dtype=bool)) == [0, 0, 0, 0]

    def test_single_pixel_mask(self) -> None:
        """单像素 mask → 零宽 bbox，x1==x2==y1==y2。"""
        assert mask_to_bbox(_mask(10, 10, 3, 3, 4, 4)) == [3, 3, 3, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
