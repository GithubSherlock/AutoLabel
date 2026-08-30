"""COCO 91→80 类别映射回归测试（v0.6 Phase 3a 实测 bug 修复，零真实权重）。

torchvision COCO_V1 权重（fasterrcnn/retinanet/maskrcnn/ssd/fcos）输出
detectron 91 类 1-based 索引（0=background，10 个占位类）——曾直接
label-1 索引 80 类表：前 11 类两表一致掩盖错位（cat 预测标成 dog），
COCO 全量 fasterrcnn mAP@0.5 0.0998 实测暴露（见 test-v0.6.md）。修复 =
model_catalog.COCO_91_TO_80 单一事实源映射，三处解析点（detection
_parse_output / sahi_infer_torchvision / segmentation MaskRCNNModel.
_parse_output）统一走映射；cityscapes 权重 label 语义不同保持原逻辑。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from auto2dlabel.configs.model_catalog import COCO_91_TO_80, COCO_CLASSES
from auto2dlabel.models.detection import (
    PyTorchVisionModel,
    sahi_infer_torchvision,
)
from auto2dlabel.models.segmentation import CITYSCAPES_THING_NAMES, MaskRCNNModel


def test_coco91_table_anchors() -> None:
    """映射表关键锚点：前 11 类 -1 一致 / cat-dog 错位锚点 / 占位类 None / 表长 90。"""
    assert COCO_91_TO_80[1] == 0  # person
    assert COCO_91_TO_80[11] == 10  # fire hydrant（前 11 类两表一致，错位被掩盖的区间）
    assert COCO_91_TO_80[17] == 15  # cat（曾错位成 dog 的锚点）
    assert COCO_91_TO_80[18] == 16  # dog（曾收 cat 预测的受害者）
    assert COCO_91_TO_80[90] == 79  # toothbrush
    for ph in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83):
        assert COCO_91_TO_80[ph] is None  # 10 个占位类：无 80 类对应
    assert len(COCO_91_TO_80) == 90  # 1..90（0=background 不入表）


def test_parse_output_maps_91_to_80() -> None:
    """回归：label 17/90 → cat/toothbrush；占位类 12 与背景 0 跳过。"""
    model = PyTorchVisionModel(device="cpu")
    outputs = {
        "boxes": torch.tensor([
            [1.0, 2.0, 11.0, 22.0], [5.0, 5.0, 9.0, 9.0],
            [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0],
        ]),
        "labels": torch.tensor([17, 90, 12, 0]),
        "scores": torch.tensor([0.9, 0.9, 0.9, 0.9]),
    }
    results = model._parse_output(outputs, COCO_CLASSES, 0.3)
    assert [(r.label, r.width, r.height) for r in results] == [
        ("cat", 10.0, 20.0),
        ("toothbrush", 4.0, 4.0),
    ]


def test_sahi_infer_maps_91_to_80() -> None:
    """回归：SAHI torchvision 路径同映射（label 17 → cat）。"""

    class _FakeTorchvision:
        def __call__(self, tensors: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            return [{
                "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
                "labels": torch.tensor([17]),
                "scores": torch.tensor([0.9]),
            }]

    dets = sahi_infer_torchvision(
        SimpleNamespace(_model=_FakeTorchvision(), _device="cpu"),
        Image.new("RGB", (64, 64)), ["cat"], 0.3,
    )
    assert len(dets) == 1 and dets[0]["name"] == "cat"


def test_maskrcnn_parse_output_maps_91_to_80() -> None:
    """回归：MaskRCNNModel（COCO 权重）label 18 → dog；bbox/mask 照常产出。"""
    model = MaskRCNNModel(device="cpu")
    mask = torch.zeros((1, 8, 8))
    mask[0, 1:3, 1:3] = 1.0
    outputs = {
        "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        "labels": torch.tensor([18]),  # detectron 91 类 18 = dog
        "scores": torch.tensor([0.9]),
        "masks": mask.unsqueeze(0),
    }
    masks_out = model._parse_output(outputs, COCO_CLASSES)
    assert len(masks_out) == 1
    assert masks_out[0].bbox.label == "dog"
    assert masks_out[0].segmentation  # 2x2 方块 mask → 非空 polygon


def test_maskrcnn_cityscapes_label_scheme_unchanged() -> None:
    """cityscapes 权重 label 语义不同（1-based thing 类），-1 原逻辑行为零变。"""
    model = MaskRCNNModel(device="cpu")
    mask = torch.zeros((1, 8, 8))
    mask[0, 1:3, 1:3] = 1.0
    outputs = {
        "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        "labels": torch.tensor([3]),  # 1-based → cls_id 2 = car
        "scores": torch.tensor([0.9]),
        "masks": mask.unsqueeze(0),
    }
    masks_out = model._parse_output(outputs, CITYSCAPES_THING_NAMES)
    assert len(masks_out) == 1
    assert masks_out[0].bbox.label == "car"
