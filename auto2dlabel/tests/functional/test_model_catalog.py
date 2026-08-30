"""模型目录回归测试 —— 零模型加载、零下载。

覆盖：旧版本模型（YOLOv5-v10/World）已移除、新增模型（YOLO12/26-OBB、
torchvision 语义分割、torchvision 分类）已登记、列表长度、VOC 类表、目录摘要生成。
"""

from __future__ import annotations

from auto2dlabel.configs.model_catalog import (
    CLASSIFICATION_MODELS,
    SEGMENTATION_MODELS,
    TORCHVISION_CLS_MODELS,
    TORCHVISION_SEG_MODELS,
    ULTRALYTICS_MODELS,
    ULTRALYTICS_OBB_MODELS,
    VOC_CLASSES,
    format_catalog_summary,
)

_REMOVED_MODELS = [
    "yolov5nu.pt", "yolov8n.pt", "yolov9t.pt", "yolov10n.pt", "yolov8s-world.pt",
]


def test_catalog_removed_old_yolo() -> None:
    """YOLOv5-v10 与 YOLO-World 已从检测目录移除。"""
    for name in _REMOVED_MODELS:
        assert name not in ULTRALYTICS_MODELS, name
    assert all("world" not in name for name in ULTRALYTICS_MODELS)


def test_catalog_kept_new_yolo() -> None:
    """仅保留 YOLO11/12/26 + RT-DETR，共 17 个。"""
    assert len(ULTRALYTICS_MODELS) == 17
    assert "yolo11n.pt" in ULTRALYTICS_MODELS
    assert "yolo12x.pt" in ULTRALYTICS_MODELS
    assert "yolo26x.pt" in ULTRALYTICS_MODELS
    assert "rtdetr-l.pt" in ULTRALYTICS_MODELS
    assert "rtdetr-x.pt" in ULTRALYTICS_MODELS


def test_obb_catalog_expanded() -> None:
    """OBB 目录含 YOLO11/12/26 三系 15 个，全部 -obb.pt 后缀。"""
    assert len(ULTRALYTICS_OBB_MODELS) == 15
    assert "yolo12n-obb.pt" in ULTRALYTICS_OBB_MODELS
    assert "yolo26x-obb.pt" in ULTRALYTICS_OBB_MODELS
    assert all(name.endswith("-obb.pt") for name in ULTRALYTICS_OBB_MODELS)


def test_seg_catalog_expanded() -> None:
    """分割目录含 torchvision 语义分割 6 个 + cityscapes 域内 Mask R-CNN 1 个（23 + 1 = 24）。"""
    assert len(SEGMENTATION_MODELS) == 24
    assert len(TORCHVISION_SEG_MODELS) == 6
    for name in TORCHVISION_SEG_MODELS:
        assert name in SEGMENTATION_MODELS
    assert "fcn_resnet50" in TORCHVISION_SEG_MODELS
    assert "deeplabv3_resnet101" in TORCHVISION_SEG_MODELS
    assert "lraspp_mobilenet_v3_large" in TORCHVISION_SEG_MODELS
    assert "maskrcnn_r50_cityscapes" in SEGMENTATION_MODELS


def test_voc_classes() -> None:
    """VOC 21 类：0 为背景，其余 20 类。"""
    assert len(VOC_CLASSES) == 21
    assert VOC_CLASSES[0] == "__background__"
    assert "aeroplane" in VOC_CLASSES
    assert "tvmonitor" in VOC_CLASSES
    assert len(set(VOC_CLASSES)) == 21


def test_classification_catalog_expanded() -> None:
    """分类目录：零样本 2 + torchvision 14（含 ResNet/ResNeXt 8），各组不重叠。"""
    assert len(CLASSIFICATION_MODELS) == 2
    assert len(TORCHVISION_CLS_MODELS) == 14
    assert set(TORCHVISION_CLS_MODELS) & set(CLASSIFICATION_MODELS) == set()
    for name in (
        "convnext_large", "convnext_base", "maxvit_t",
        "swin_b", "efficientnet_v2_l", "vit_b_16",
        "resnet18", "resnet50", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
    ):
        assert name in TORCHVISION_CLS_MODELS


def test_format_catalog_summary() -> None:
    """摘要从目录列表生成：含新模型名、不含已移除旧名。"""
    summary = format_catalog_summary()
    assert "yolo12n.pt" in summary
    assert "yolo26x.pt" in summary
    assert "fcn_resnet50" in summary
    assert "yolo26n-obb.pt" in summary
    assert "rtdetr-l.pt" in summary
    assert "convnext_large" in summary
    assert "resnet50" in summary
    assert "ImageNet1K" in summary
    for name in _REMOVED_MODELS:
        assert name not in summary
    # 组齐备
    assert "Grounding DINO" in summary
    assert "PyTorch Vision" in summary
    assert "classification" in summary
