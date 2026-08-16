"""检测模型目录。

列出所有可用的检测模型：Grounding DINO、Ultralytics、PyTorch Vision。
"""

# ============================================================
# Grounding DINO（HuggingFace transformers，开放词汇）
# ============================================================
GROUNDING_DINO_MODELS = [
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
    "IDEA-Research/grounding-dino-large",
]

# ============================================================
# Ultralytics YOLO 全系列（统一 API，自动下载 .pt 权重）
# ============================================================
ULTRALYTICS_MODELS = [
    # YOLOv5
    "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
    # YOLOv8
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    # YOLOv9
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLO12
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    # YOLO-World（开放词汇）
    "yolov8s-world.pt", "yolov8m-world.pt", "yolov8l-world.pt", "yolov8x-world.pt",
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
# 分割模型
# ============================================================
SEGMENTATION_MODELS = [
    # FastSAM
    "FastSAM-s.pt", "FastSAM-x.pt",
    # SAM / SAM2 (Ultralytics)
    "sam_t.pt", "sam_s.pt", "sam_b.pt", "sam_l.pt", "sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt", "sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt",
    # Mask R-CNN
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    # SAM3（需手动下载权重 ~3.45GB，运行 bash download_sam3.sh）
    "sam3.pt",
]

# ============================================================
# 分类模型（HuggingFace transformers，零样本）
# ============================================================
CLASSIFICATION_MODELS = [
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
]

# ============================================================
# 旋转框检测模型（Ultralytics YOLO-OBB）
# ============================================================
ULTRALYTICS_OBB_MODELS = [
    "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt",
    "yolo11l-obb.pt", "yolo11x-obb.pt",
]

# ============================================================
# 权重下载目录
# ============================================================
from pathlib import Path
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
