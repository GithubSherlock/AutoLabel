"""torchvision 语义分割接入测试（FCN / DeepLabV3 / LRASPP）—— 零权重下载。

覆盖：工厂分发与懒加载、logits→类别图（argmax）、类别图→Mask
（背景跳过 / VOC→COCO 别名 / boundingRect 派生 bbox / prompt 过滤）、
VOC↔COCO 别名匹配。
"""

from __future__ import annotations

from typing import Any

import pytest

from auto2dlabel.configs.model_catalog import VOC_CLASSES
from auto2dlabel.models.segmentation import (
    FastSAMModel,
    TorchVisionSegModel,
    _class_mask_to_masks,
    _logits_to_class_mask,
    _voc_prompt_match,
    create_segmentation_model,
)
from auto2dlabel.tests import np

# ============================================================
# 工厂分发 + 懒加载
# ============================================================

class TestDispatch:
    def test_torchvision_names(self) -> None:
        """6 个目录名全部路由到 TorchVisionSegModel 且名称透传。"""
        for name in (
            "fcn_resnet50", "fcn_resnet101",
            "deeplabv3_resnet50", "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large",
        ):
            model = create_segmentation_model(name)
            assert isinstance(model, TorchVisionSegModel)
            assert model._model_name == name

    def test_prefix_tolerance(self) -> None:
        """fcn_/deeplabv3_/lraspp_ 前缀容错（含未收录后缀）。"""
        for name in ("deeplabv3_resnet101", "fcn_resnet50", "lraspp_mobilenet_v3_large"):
            assert isinstance(create_segmentation_model(name), TorchVisionSegModel)

    def test_unknown_falls_back_to_fastsam(self) -> None:
        """未知名保持原行为：兜底 FastSAM。"""
        assert isinstance(create_segmentation_model("mystery_model"), FastSAMModel)

    def test_lazy_loading(self) -> None:
        """构造零加载（不 import torchvision / 不下载权重）。"""
        model = TorchVisionSegModel()
        assert model._model is None


# ============================================================
# 纯函数：logits → 类别图
# ============================================================

class TestLogitsToClassMask:
    def test_argmax(self) -> None:
        logits = np.zeros((21, 8, 8), dtype=np.float32)
        logits[3] = 2.0  # 类别 3（bird）全域最强
        logits[0] = 1.0  # 背景次强
        mask = _logits_to_class_mask(logits)
        assert mask.shape == (8, 8)
        assert (mask == 3).all()
        assert mask.dtype == np.int32

    def test_pixelwise_argmax(self) -> None:
        logits = np.zeros((21, 2, 2), dtype=np.float32)
        logits[1, 0, 0] = 5.0  # 左上：类别 1
        logits[2, 1, 1] = 5.0  # 右下：类别 2
        mask = _logits_to_class_mask(logits)
        assert mask[0, 0] == 1
        assert mask[1, 1] == 2
        assert mask[0, 1] == 0  # 其余为背景（logits 全 0 → argmax 0）


# ============================================================
# 纯函数：类别图 → Mask 列表
# ============================================================

class TestClassMaskToMasks:
    def _mask(self) -> np.ndarray[Any, Any]:
        return np.zeros((10, 10), dtype=np.int32)

    def test_background_produces_no_mask(self) -> None:
        """全背景 → 空结果（类 0 恒跳过）。"""
        assert _class_mask_to_masks(self._mask(), []) == []

    def test_voc_alias_label_and_bbox(self) -> None:
        """类 1 = aeroplane → 输出 label "airplane"；bbox 由外接矩形派生。"""
        mask = self._mask()
        mask[2:6, 3:8] = 1  # y:2-5, x:3-7
        out = _class_mask_to_masks(mask, [])
        assert len(out) == 1
        m = out[0]
        assert m.bbox.label == "airplane"  # VOC→COCO 别名
        assert (m.bbox.x, m.bbox.y) == (3.0, 2.0)
        assert (m.bbox.width, m.bbox.height) == (5.0, 4.0)
        assert m.area == pytest.approx(20.0)
        assert m.segmentation and len(m.segmentation[0]) >= 8  # polygon 非退化

    def test_multiple_classes_with_background(self) -> None:
        """多类同时输出；无别名类（person/car）label 原样。"""
        mask = self._mask()
        mask[1:4, 1:5] = VOC_CLASSES.index("person")
        mask[5:7, 5:9] = VOC_CLASSES.index("car")
        out = _class_mask_to_masks(mask, [])
        labels = {m.bbox.label for m in out}
        assert labels == {"person", "car"}
        assert len(out) == 2

    def test_prompt_filters_by_voc_alias(self) -> None:
        """prompt 用 VOC 名（aeroplane）→ 仅输出对应类的 COCO 名 label。"""
        mask = self._mask()
        mask[2:4, 2:6] = VOC_CLASSES.index("aeroplane")
        mask[6:8, 6:9] = VOC_CLASSES.index("car")
        out = _class_mask_to_masks(mask, ["aeroplane"])
        assert len(out) == 1
        assert out[0].bbox.label == "airplane"

    def test_prompt_miss_returns_empty(self) -> None:
        mask = self._mask()
        mask[2:4, 2:6] = VOC_CLASSES.index("car")
        assert _class_mask_to_masks(mask, ["dog"]) == []


# ============================================================
# 纯函数：VOC↔COCO prompt 匹配
# ============================================================

class TestVocPromptMatch:
    def test_voc_prompt_matches_coco_label(self) -> None:
        """label（COCO）与 prompt（VOC 名）互为别名。"""
        assert _voc_prompt_match("airplane", ["aeroplane"])
        assert _voc_prompt_match("tv", ["tvmonitor"])

    def test_coco_prompt_matches_voc_alias(self) -> None:
        """prompt 为 COCO 名时命中 label 的 VOC 别名。"""
        assert _voc_prompt_match("motorcycle", ["motorbike"])
        assert _voc_prompt_match("potted plant", ["pottedplant"])

    def test_multiword_alias(self) -> None:
        assert _voc_prompt_match("dining table", ["diningtable"])

    def test_substring_fallback(self) -> None:
        """无别名时双向子串包含兜底。"""
        assert _voc_prompt_match("car", ["racing car"])

    def test_unrelated_negative(self) -> None:
        assert not _voc_prompt_match("airplane", ["car", "person"])
        assert not _voc_prompt_match("car", [])
