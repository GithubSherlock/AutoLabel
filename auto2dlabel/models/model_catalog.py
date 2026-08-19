"""模型目录。

列出所有可用模型：Grounding DINO、Ultralytics、PyTorch Vision（检测）、
分割（SAM/FastSAM/torchvision 语义分割）、分类（CLIP/SigLIP/torchvision ImageNet1K）、
OBB（YOLO-OBB）。
"""

from pathlib import Path

# ============================================================
# Grounding DINO（HuggingFace transformers，开放词汇）
# ============================================================
GROUNDING_DINO_MODELS = [
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
    "IDEA-Research/grounding-dino-large",
]

# ============================================================
# Ultralytics YOLO 系列（YOLO11/12/26 + RT-DETR，统一 API，自动下载 .pt 权重）
# ============================================================
ULTRALYTICS_MODELS = [
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLO12
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    # YOLO26
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    # RT-DETR（Real-Time Detection Transformer）
    "rtdetr-l.pt", "rtdetr-x.pt",
]

# ============================================================
# PyTorch Vision 检测模型（torchvision.models.detection）
# ============================================================
PYTORCH_DETECTION_MODELS = [
    # Faster R-CNN
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    # RetinaNet
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    # SSD
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    # FCOS
    "fcos_resnet50_fpn",
]

# ============================================================
ALL_DETECTION_MODELS = GROUNDING_DINO_MODELS + ULTRALYTICS_MODELS + PYTORCH_DETECTION_MODELS

# ============================================================
# COCO 80 类别名（标准 YOLO / PyTorch 模型默认训练集）
# ============================================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# ============================================================
# DOTA v1 15 类别（YOLO-OBB 预训练集，ultralytics dota8.yaml 权威顺序）
# ============================================================
DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]

# DOTA 类名 → prompt 别名（含 4 个 COCO 对应名，供用户 prompt 兼容）
DOTA_PROMPT_ALIASES: dict[str, tuple[str, ...]] = {
    "plane": ("airplane",),
    "ship": ("boat",),
    "large vehicle": ("truck",),
    "small vehicle": ("car", "vehicle"),
}

# ============================================================
# torchvision 语义分割模型（VOC 21 类预训练）
# ============================================================
TORCHVISION_SEG_MODELS = [
    "fcn_resnet50", "fcn_resnet101",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large",
]

# VOC 21 类（torchvision 分割权重标签，COCO_WITH_VOC_LABELS_V1，0=背景）
VOC_CLASSES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ============================================================
# 分割模型
# ============================================================
SEGMENTATION_MODELS = [
    # FastSAM
    "FastSAM-s.pt", "FastSAM-x.pt",
    # SAM / SAM2 (Ultralytics)
    "sam_t.pt", "sam_s.pt", "sam_b.pt", "sam_l.pt",
    "sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt",
    "sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt",
    # Mask R-CNN
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    # Cityscapes 域内（mmdet 权重转换，Box 40.9 / Mask 36.4；需先运行转换脚本）
    "maskrcnn_r50_cityscapes",
    # SAM3（需手动下载权重 ~3.45GB，运行 bash download_sam3.sh）
    "sam3.pt",
    # torchvision 语义分割（VOC 21 类预训练，0=__background__）
    "fcn_resnet50", "fcn_resnet101",
    "deeplabv3_resnet50", "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large",
]

# ============================================================
# 分类模型（HuggingFace transformers，零样本）
# ============================================================
CLASSIFICATION_MODELS = [
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
]

# ============================================================
# ReID 特征提取模型（HuggingFace transformers 图像编码器，
# BoT-SORT 外观关联用；与分类共用同一批 CLIP/SigLIP 权重）
# ============================================================
REID_MODELS = [
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
]

# ============================================================
# torchvision 分类模型（ImageNet1K 监督）：
# 前 6 个为性能最强档（top-1 84%±），后 8 个为 ResNet/ResNeXt 经典系
# 权重 URL 见 weights/download_weights.sh case 表（哈希名取自 0.28 枚举）
# ============================================================
TORCHVISION_CLS_MODELS = [
    "convnext_large", "convnext_base",
    "maxvit_t",
    "swin_b",
    "efficientnet_v2_l",
    "vit_b_16",
    # ResNet / ResNeXt 系列（resnet50 通用之选）
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
]

# ============================================================
# 旋转框检测模型（Ultralytics YOLO-OBB）
# ============================================================
ULTRALYTICS_OBB_MODELS = [
    # YOLO11-OBB（DOTAv1 预训练）
    "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt",
    "yolo11l-obb.pt", "yolo11x-obb.pt",
    # YOLO12-OBB（DOTAv1 预训练）
    "yolo12n-obb.pt", "yolo12s-obb.pt", "yolo12m-obb.pt",
    "yolo12l-obb.pt", "yolo12x-obb.pt",
    # YOLO26-OBB（DOTAv1 预训练）
    "yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt",
    "yolo26l-obb.pt", "yolo26x-obb.pt",
]

# ============================================================
# 权重下载目录
# ============================================================
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 目录摘要（供 planner system prompt 注入，单一事实源）
# ============================================================
_CATALOG_SUMMARY_GROUPS: list[tuple[str, list[str], str]] = [
    ("object_detection [Ultralytics, COCO80]", ULTRALYTICS_MODELS,
     "nano 快速预览, x 高精度; rtdetr 精度优先"),
    ("object_detection [Grounding DINO, 开放词汇]", GROUNDING_DINO_MODELS,
     "任意类别文本 prompt"),
    ("object_detection [PyTorch Vision, COCO80]", PYTORCH_DETECTION_MODELS, "高召回"),
    ("obb_detection [YOLO-OBB, DOTAv1]", ULTRALYTICS_OBB_MODELS, "航拍旋转框"),
    ("instance_segmentation", SEGMENTATION_MODELS,
     "sam3/maskrcnn/fcn*/deeplabv3*/lraspp* 自带检测"),
    ("classification [零样本]", CLASSIFICATION_MODELS, "多候选 softmax top-K"),
    ("classification [ImageNet1K, torchvision]", TORCHVISION_CLS_MODELS,
     "监督 top-K，candidates 子串过滤；resnet50 通用之选"),
    ("reid [图像嵌入, 零样本]", REID_MODELS,
     "BoT-SORT 外观关联特征（--track --bot-sort）"),
]


def format_catalog_summary() -> str:
    """生成精简目录摘要（供 planner system prompt 注入，token 可控）。"""
    lines = ["Available models (use exact names):"]
    for group, names, hint in _CATALOG_SUMMARY_GROUPS:
        lines.append(f"- {group}: {', '.join(names)} — {hint}")
    return "\n".join(lines)
