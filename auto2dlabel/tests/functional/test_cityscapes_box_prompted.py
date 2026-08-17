"""cityscapes box-prompted 消融测试 — assign_mask_labels（mask-IoU 重匹配）。

纯合成 mask，无权重依赖。核心契约：对输出顺序/数量零假设，
FastSAM 丢框 / SAM2 丢 prompt / maskrcnn 自检测三种退化免疫。
"""

from __future__ import annotations

import pytest

from auto2dlabel.benchmarks.common import assign_mask_labels
from auto2dlabel.tests import np


def _mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """构造 (h,w) 全零 mask，仅 [x1:x2, y1:y2) 区域为 True。"""
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class TestGtBboxView:
    def test_view_keeps_file_name(self) -> None:
        """回归：_gt_bbox_view 必须保留 file_name（SAHI 路径曾 KeyError）。"""
        from auto2dlabel.benchmarks.cityscapes_benchmark import _gt_bbox_view

        gt = {0: {"file_name": "aachen/img.png", "objects": [
            {"name": "car", "mask": _mask(10, 10, 0, 0, 4, 4)},
        ]}}
        view = _gt_bbox_view(gt)
        assert view[0]["file_name"] == "aachen/img.png"
        assert view[0]["objects"][0]["bbox"] == [0, 0, 3, 3]


class TestAssignMaskLabels:
    def test_permuted_order_still_correct(self) -> None:
        """输出顺序与 GT 顺序打乱 → 命名按 mask-IoU 重匹配，不依赖索引。"""
        gt = [
            {"name": "car", "mask": _mask(10, 10, 0, 0, 4, 4)},
            {"name": "person", "mask": _mask(10, 10, 6, 6, 10, 10)},
        ]
        outputs = [
            {"mask": _mask(10, 10, 6, 6, 10, 10), "conf": 0.9},  # person 在前
            {"mask": _mask(10, 10, 0, 0, 4, 4), "conf": 0.8},
        ]
        named = assign_mask_labels(outputs, gt, 10, 10)
        assert [o["name"] for o in named] == ["person", "car"]
        assert [o["conf"] for o in named] == [0.9, 0.8]

    def test_fewer_outputs_no_drift(self) -> None:
        """输出数 < GT 数（丢框退化）→ 只命名 IoU 匹配到的，无索引漂移。"""
        gt = [
            {"name": "car", "mask": _mask(10, 10, 0, 0, 4, 4)},
            {"name": "bus", "mask": _mask(10, 10, 6, 0, 10, 4)},
        ]
        outputs = [{"mask": _mask(10, 10, 6, 0, 10, 4), "conf": 0.9}]
        named = assign_mask_labels(outputs, gt, 10, 10)
        assert len(named) == 1
        assert named[0]["name"] == "bus"

    def test_more_outputs_each_best_match(self) -> None:
        """输出数 > GT 数：每个输出取 IoU 最大的 GT 命名（可重复）。"""
        gt = [{"name": "car", "mask": _mask(10, 10, 0, 0, 6, 6)}]
        outputs = [
            {"mask": _mask(10, 10, 0, 0, 6, 6), "conf": 0.9},
            {"mask": _mask(10, 10, 1, 1, 5, 5), "conf": 0.7},  # 与 car 部分重叠
        ]
        named = assign_mask_labels(outputs, gt, 10, 10)
        assert [o["name"] for o in named] == ["car", "car"]

    def test_low_iou_discarded(self) -> None:
        """与所有 GT 的 IoU < min_iou 的输出丢弃（离群噪声）。"""
        gt = [{"name": "car", "mask": _mask(10, 10, 0, 0, 4, 4)}]
        outputs = [{"mask": _mask(10, 10, 6, 6, 9, 9), "conf": 0.9}]  # 不相交
        assert assign_mask_labels(outputs, gt, 10, 10) == []

    def test_resolution_guard_skips(self) -> None:
        """mask 分辨率与图像不符 → 跳过（分辨率守卫，防推理缩放坑）。"""
        gt = [{"name": "car", "mask": _mask(10, 10, 0, 0, 4, 4)}]
        outputs = [{"mask": _mask(8, 8, 0, 0, 4, 4), "conf": 0.9}]
        assert assign_mask_labels(outputs, gt, 10, 10) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
