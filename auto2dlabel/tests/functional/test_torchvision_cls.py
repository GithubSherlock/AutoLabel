"""torchvision 分类接入测试（ImageNet1K 监督）—— 零权重下载、零 torch import。

覆盖：工厂分发与懒加载、_topk_labels 纯函数（候选子串过滤 /
纯 top-K / 命中不足 / 类名越界回退 / 零命中 / top_k=0 防御）。
"""

from __future__ import annotations

import pytest

from auto2dlabel.models.classification import (
    TorchVisionClassifier,
    _topk_labels,
    create_classification_model,
)

# ============================================================
# 工厂分发 + 懒加载
# ============================================================

class TestDispatch:
    def test_torchvision_names(self) -> None:
        """6 个目录名全部路由到 TorchVisionClassifier 且名称透传。"""
        for name in (
            "convnext_large", "convnext_base",
            "maxvit_t", "swin_b",
            "efficientnet_v2_l", "vit_b_16",
        ):
            model = create_classification_model(name)
            assert isinstance(model, TorchVisionClassifier)
            assert model._model_name == name

    def test_resnet_resnext_names(self) -> None:
        """ResNet / ResNeXt 8 个目录名全部路由且名称透传。"""
        for name in (
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
        ):
            model = create_classification_model(name)
            assert isinstance(model, TorchVisionClassifier)
            assert model._model_name == name

    def test_prefix_tolerance(self) -> None:
        """前缀容错：目录外 torchvision 名仍可运行（泛型 enum 解析不依赖名单）。"""
        for name in ("vit_l_16", "swin_t", "efficientnet_v2_s", "resnet1202"):
            model = create_classification_model(name)
            assert isinstance(model, TorchVisionClassifier)
            assert model._model_name == name

    def test_unknown_raises_value_error(self) -> None:
        """未知名抛 ValueError，文案列目录名单。"""
        with pytest.raises(ValueError, match="无法识别的分类模型"):
            create_classification_model("densenet121")
        with pytest.raises(ValueError, match="convnext_large"):
            create_classification_model("mystery_model")

    def test_lazy_loading(self) -> None:
        """构造零加载（不 import torchvision / 不下载权重）。"""
        model = TorchVisionClassifier(model_name="convnext_large")
        assert model._model is None
        assert model._transform is None
        assert model._class_names == []


# ============================================================
# 纯函数：_topk_labels（ImageNet 全类分数 → top-K ImageLabel）
# ============================================================

class TestTopkLabels:
    def test_candidate_filter_sorted_by_score(self) -> None:
        """candidates 非空：_match_prompt 子串过滤 + 按分降序。"""
        labels = _topk_labels(
            [0.5, 0.3, 0.2],
            ["tabby cat", "dog", "car"],
            top_k=2,
            candidates=["cat", "car"],
        )
        assert [(lab.label, lab.score) for lab in labels] == [
            ("tabby cat", 0.5),
            ("car", 0.2),
        ]

    def test_no_candidates_pure_topk(self) -> None:
        """candidates 为空/None：纯 top-K ImageNet 类名。"""
        labels = _topk_labels(
            [0.1, 0.7, 0.2],
            ["a", "b", "c"],
            top_k=2,
            candidates=None,
        )
        assert [(lab.label, lab.score) for lab in labels] == [("b", 0.7), ("c", 0.2)]
        assert [(lab.label, lab.score) for lab in _topk_labels(
            [0.1, 0.7, 0.2], ["a", "b", "c"], top_k=1, candidates=[],
        )] == [("b", 0.7)]

    def test_fewer_hits_than_topk(self) -> None:
        """命中不足 top_k 时只返回命中数（不填占位）。"""
        labels = _topk_labels(
            [0.5, 0.3, 0.2],
            ["tabby cat", "dog", "car"],
            top_k=3,
            candidates=["cat"],
        )
        assert [lab.label for lab in labels] == ["tabby cat"]

    def test_class_names_shorter_than_scores(self) -> None:
        """类名表短于分数表：越界下标回退 str(i) 标签。"""
        labels = _topk_labels([0.5, 0.3, 0.2], ["a", "b"], top_k=3)
        assert [lab.label for lab in labels] == ["a", "b", "2"]

    def test_zero_hit_returns_empty(self) -> None:
        assert _topk_labels(
            [0.5, 0.3], ["tabby cat", "dog"], top_k=2, candidates=["xyz"],
        ) == []

    def test_topk_zero_returns_empty(self) -> None:
        assert _topk_labels([0.5], ["cat"], top_k=0) == []
