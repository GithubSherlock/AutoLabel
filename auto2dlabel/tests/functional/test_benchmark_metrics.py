"""基准测试指标单元测试 — bbox AP / mask IoU / 重构等价性。"""

from __future__ import annotations

import pytest

from auto2dlabel.benchmarks.common import (
    compute_dice,
    compute_iou,
    compute_mask_iou,
    evaluate_mask_per_class,
    evaluate_per_class,
)
from auto2dlabel.tests import np


class TestComputeIoU:
    def test_no_overlap(self) -> None:
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_full_overlap(self) -> None:
        assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_partial_overlap(self) -> None:
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        # 交集 5*5=25, 并集 100+100-25=175 → IoU = 25/175 = 0.142...
        assert abs(iou - 0.142857) < 0.01

    def test_edge_touching(self) -> None:
        assert compute_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


class TestBboxAP:
    def test_empty_predictions(self) -> None:
        gt = {0: {"objects": [{"name": "car", "bbox": [0, 0, 10, 10]}]}}
        pred: dict[int, list[dict[str, object]]] = {0: []}
        result = evaluate_per_class(gt, pred, "car")
        assert result["ap"] == 0.0
        assert result["gt_count"] == 1
        assert result["pred_count"] == 0

    def test_perfect_detection(self) -> None:
        gt = {0: {"objects": [{"name": "car", "bbox": [0, 0, 10, 10]}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "car", "bbox": [0, 0, 10, 10], "conf": 0.99}],
        }
        result = evaluate_per_class(gt, pred, "car")
        assert result["ap"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_single_fp(self) -> None:
        gt = {0: {"objects": [{"name": "car", "bbox": [0, 0, 10, 10]}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "car", "bbox": [50, 50, 60, 60], "conf": 0.99}],
        }
        result = evaluate_per_class(gt, pred, "car")
        assert result["ap"] == 0.0
        assert result["pred_count"] == 1


class TestMaskMetrics:
    def test_mask_iou_full(self) -> None:
        a = np.ones((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert compute_mask_iou(a, b) == 1.0

    def test_mask_iou_zero(self) -> None:
        a = np.zeros((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert compute_mask_iou(a, b) == 0.0

    def test_dice(self) -> None:
        a = np.ones((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert compute_dice(a, b) == 1.0

    def test_mask_ap_perfect(self) -> None:
        gt = {0: {"objects": [{"name": "person", "mask": np.ones((10, 10), dtype=bool)}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "person", "mask": np.ones((10, 10), dtype=bool), "conf": 0.99}],
        }
        result = evaluate_mask_per_class(gt, pred, "person")
        assert result["ap"] == 1.0
        assert result["miou"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
