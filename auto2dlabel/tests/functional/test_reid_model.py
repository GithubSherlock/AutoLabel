"""ReID 特征提取层单测 —— 零模型加载、零权重下载（仿 test_classification.py）。

覆盖：工厂分发 / 构造懒加载 / l2_normalize / crop_bbox 边界 /
extract_frame_features 索引对齐 / 目录常量。
推理路径经假模型（Protocol duck typing）注入，不 import torch。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from auto2dlabel.models import Image
from auto2dlabel.models.model_catalog import REID_MODELS, format_catalog_summary
from auto2dlabel.models.tracking import (
    ClipReIDModel,
    SigLIPReIDModel,
    _image_embeds,
    create_reid_model,
    crop_bbox,
    extract_frame_features,
    l2_normalize,
)
from auto2dlabel.schema.annotation import Bbox


def _det(x: float, y: float, conf: float = 0.9) -> Bbox:
    return Bbox(x=x, y=y, width=20.0, height=30.0,
                label="person", confidence=conf)


class FakeReIDModel:
    """返回与输入索引绑定的可辨识特征（每图不同），用于对齐断言。"""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def extract(
        self, images: Sequence[Image.Image],
    ) -> list[np.ndarray[Any, Any]]:
        self.calls.append(len(images))
        return [np.full(4, float(len(self.calls)) * 100 + i, dtype=np.float32)
                for i in range(len(images))]


# ============================================================
# 工厂与懒加载
# ============================================================

def test_create_reid_model_dispatch() -> None:
    assert isinstance(create_reid_model("openai/clip-vit-base-patch32"), ClipReIDModel)
    assert isinstance(create_reid_model("google/siglip-base-patch16-224"),
                      SigLIPReIDModel)
    assert isinstance(create_reid_model("SIGLIP-base"), SigLIPReIDModel), \
        "大小写不敏感路由"


def test_create_reid_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="ReID"):
        create_reid_model("no-such-model")


def test_reid_constructor_is_lazy() -> None:
    """构造零加载（零下载、零 import transformers/torch）。"""
    clip = ClipReIDModel()
    assert clip._model is None
    assert clip._processor is None
    assert clip._model_name == "openai/clip-vit-base-patch32"
    siglip = SigLIPReIDModel()
    assert siglip._model is None
    assert siglip._processor is None
    assert siglip._model_name == "google/siglip-base-patch16-224"


# ============================================================
# 纯函数
# ============================================================

def test_l2_normalize_unit_norm() -> None:
    feats = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
    out = l2_normalize(feats)
    assert out.shape == (2, 2)
    assert out.dtype == np.float64
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 1.0])


def test_l2_normalize_zero_vector_preserved() -> None:
    """零向量行原样保留（不产生 NaN/Inf）。"""
    out = l2_normalize(np.zeros((2, 3), dtype=np.float32))
    assert out.shape == (2, 3)
    assert np.all(out == 0.0)


def test_crop_bbox_rounding_and_clamp() -> None:
    img = Image.new("RGB", (100, 100))
    # 浮点坐标 floor/ceil 取整：x1=10, y1=10, x2=31, y2=41
    crop = crop_bbox(img, _det(10.2, 10.8))
    assert crop.size == (21, 31)
    # 负坐标越界 clamp 到 0
    assert crop_bbox(img, _det(-5.0, -10.0)).size == (15, 20)
    # 右下越界 clamp 到图像边界
    assert crop_bbox(img, _det(95.0, 90.0)).size == (5, 10)


def test_crop_bbox_fully_out_of_bounds_degrades_1x1() -> None:
    """全越界退化边界 1×1，crop 不抛异常。"""
    img = Image.new("RGB", (100, 100))
    crop = crop_bbox(img, _det(200.0, 200.0))
    assert crop.size == (1, 1)


def test_crop_bbox_pad_expands_and_still_clamps() -> None:
    img = Image.new("RGB", (100, 100))
    # pad=5 向四周扩边，但仍 clamp 到图像边界
    assert crop_bbox(img, _det(2.0, 2.0), pad=5).size == (27, 37)
    assert crop_bbox(img, _det(0.0, 0.0), pad=5).size == (25, 35)


# ============================================================
# extract_frame_features 对齐语义
# ============================================================

def test_extract_frame_features_alignment_and_filter() -> None:
    """min_conf 过滤 → 低分项 None、高分项特征、与 bboxes 逐索引对齐。"""
    img = Image.new("RGB", (200, 200))
    fake = FakeReIDModel()
    bboxes = [_det(10.0, 10.0, conf=0.9), _det(50.0, 10.0, conf=0.3),
              _det(90.0, 10.0, conf=0.8)]

    out = extract_frame_features(fake, img, bboxes, min_conf=0.5)

    assert len(out) == 3
    assert fake.calls == [2], "仅高分框（2 个）参与批量提取"
    assert out[1] is None, "低于 min_conf 的框特征为 None"
    assert out[0] is not None and float(out[0][0]) == 100.0, "第 0 框对应第 1 个特征"
    assert out[2] is not None and float(out[2][0]) == 101.0, "第 2 框对应第 2 个特征"


def test_extract_frame_features_empty_not_called() -> None:
    """空 bboxes → 空列表，模型不被调用。"""
    fake = FakeReIDModel()
    out = extract_frame_features(fake, Image.new("RGB", (10, 10)), [])
    assert out == []
    assert fake.calls == []


# ============================================================
# get_image_features 输出兼容（transformers 5.x 输出对象）
# ============================================================

def test_image_embeds_compat_output_object() -> None:
    """5.x 输出对象 → pooler_output / image_embeds 提取（零权重，SimpleNamespace 模拟）。"""
    from types import SimpleNamespace

    out_clip = SimpleNamespace(pooler_output=np.ones((2, 4)))
    assert _image_embeds(out_clip) is out_clip.pooler_output

    out_siglip = SimpleNamespace(pooler_output=None, image_embeds=np.zeros((3, 5)))
    assert _image_embeds(out_siglip) is out_siglip.image_embeds


# ============================================================
# 目录常量
# ============================================================

def test_reid_catalog_constant_and_summary() -> None:
    assert "openai/clip-vit-base-patch32" in REID_MODELS
    assert "google/siglip-base-patch16-224" in REID_MODELS
    summary = format_catalog_summary()
    assert "reid" in summary
    assert "BoT-SORT" in summary
