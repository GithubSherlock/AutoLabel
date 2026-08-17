"""DOTA OBB benchmark 指标测试 — evaluate_per_class(iou_fn=rotate_iou) + 类名归一。

纯合成数值，无权重依赖。
"""

from __future__ import annotations

import math

import pytest

from auto2dlabel.benchmarks.common import (
    evaluate_per_class,
    normalize_dota_class,
    rotate_iou,
)

# 10×10 轴对齐正方形（顺时针点序）
_SQ = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
# 同中心 45° 旋转正方形（边长 10，顶点距中心 10/√2，对角线沿轴）
_a = math.sqrt(2) / 2 * 10.0
_SQ45 = [5.0 - _a, 5.0, 5.0, 5.0 - _a, 5.0 + _a, 5.0, 5.0, 5.0 + _a]


class TestObbAP:
    def test_perfect_detection(self) -> None:
        """GT 与预测 quad 完全一致 → AP=1.0（rotate_iou 匹配）。"""
        gt = {0: {"objects": [{"name": "plane", "quad": _SQ, "bbox": [0, 0, 10, 10]}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "plane", "quad": _SQ, "conf": 0.99}],
        }
        result = evaluate_per_class(gt, pred, "plane", iou_fn=rotate_iou)
        assert result["ap"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_45deg_partial_overlap_is_tp(self) -> None:
        """GT 轴对齐 vs 预测 45° 同中心（IoU=2√2−2≈0.828 ≥ 0.5）→ 判 TP。"""
        gt = {0: {"objects": [{"name": "plane", "quad": _SQ}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "plane", "quad": _SQ45, "conf": 0.9}],
        }
        result = evaluate_per_class(gt, pred, "plane", iou_fn=rotate_iou)
        assert result["ap"] == 1.0

    def test_shifted_low_iou_is_fp(self) -> None:
        """大幅偏移（IoU≈0.087 < 0.5）→ 判 FP，AP=0。"""
        shifted = [6.0, 0.0, 16.0, 0.0, 16.0, 10.0, 6.0, 10.0]
        gt = {0: {"objects": [{"name": "plane", "quad": _SQ}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "plane", "quad": shifted, "conf": 0.9}],
        }
        result = evaluate_per_class(gt, pred, "plane", iou_fn=rotate_iou)
        assert result["ap"] == 0.0
        assert result["pred_count"] == 1

    def test_gt_bbox_fallback_no_crash(self) -> None:
        """GT 无 quad 键（仅 HBB bbox）→ 几何回退 bbox，rotate_iou 对 4 值退化 0.0。"""
        gt = {0: {"objects": [{"name": "car", "bbox": [0, 0, 10, 10]}]}}
        pred: dict[int, list[dict[str, object]]] = {
            0: [{"name": "car", "quad": _SQ, "conf": 0.9}],
        }
        result = evaluate_per_class(gt, pred, "car", iou_fn=rotate_iou)
        assert result["ap"] == 0.0  # 不崩溃，len<8 容错


class TestNormalizeDotaClass:
    def test_space_hyphen_unified(self) -> None:
        """空格 ↔ 连字符统一（GT 连字符名 vs 模型空格名）。"""
        assert normalize_dota_class("small vehicle") == "small-vehicle"
        assert normalize_dota_class("small-vehicle") == "small-vehicle"
        assert normalize_dota_class("storage tank") == "storage-tank"

    def test_case_and_whitespace(self) -> None:
        assert normalize_dota_class("  Small Vehicle ") == "small-vehicle"

    def test_roundtrip_stable(self) -> None:
        """归一化幂等。"""
        once = normalize_dota_class("large vehicle")
        assert normalize_dota_class(once) == once


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
