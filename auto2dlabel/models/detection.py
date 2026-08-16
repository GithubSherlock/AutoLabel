"""检测模型抽象层。

封装 Grounding DINO 与 Ultralytics 全系列检测模型（YOLOv5/v8/v9/v10/v11/v12），
提供统一接口：detect(image, prompts) -> List[DetectionResult]。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from PIL import Image

from auto2dlabel.models.model_catalog import COCO_CLASSES
from auto2dlabel.schema.task_plan import DEFAULT_MODEL

if TYPE_CHECKING:
    # 仅用于类型标注与 cast（字符串前向引用），运行时保持懒加载
    import torch
    from ultralytics.engine.results import Boxes, Results


@dataclass
class DetectionResult:
    """单个检测结果。坐标使用像素坐标系。"""

    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float


class DetectionModel(Protocol):
    """检测模型接口（Protocol，允许 duck typing）。"""

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        ...


# ============================================================
# Grounding DINO（HuggingFace transformers）
# ============================================================

class GroundingDINOModel:
    """Grounding DINO 模型封装（开放词汇）。

    使用 HuggingFace transformers 内置实现，首次调用自动下载权重。
    Apple Silicon 用 MPS，NVIDIA 用 CUDA。
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._device = device or get_device()
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._iou = iou_threshold
        self._model = None
        self._processor = None

    def _load(self) -> Any:
        """加载模型与处理器（幂等）。transformers 为可选依赖，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        try:
            from transformers import (  # type: ignore  # 可选依赖，未安装时跳过静态解析
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR

        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self._model_name)
        self._model.to(self._device)
        return self._model

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor
        image = Image.open(image_path).convert("RGB")
        text_prompt = " . ".join(prompts) + " ."

        import torch

        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results_p = processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes,
            threshold=self._box_threshold, text_threshold=self._text_threshold,
        )[0]

        results = []
        for box, label_id, score in zip(
            results_p["boxes"], results_p["labels"], results_p["scores"]
        ):
            conf = float(score)
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append(DetectionResult(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
                label=str(label_id), confidence=conf,
            ))
        return results


# ============================================================
# Ultralytics 全系列（YOLOv5/v8/v9/v10/v11/v12 + YOLO-World）
# ============================================================

class UltralyticsModel:
    """Ultralytics 全系列检测模型封装。

    支持所有 YOLOv5/v8/v9/v10/v11/v12 及其 n/s/m/l/x 变体，
    以及 YOLO-World 开放词汇模型。首次使用自动下载 .pt 权重。
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str | None = None,
        iou_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._device = device
        self._iou = iou_threshold
        self._model = None
        self._is_world = "world" in model_name.lower()

    def _load(self) -> Any:
        """加载模型（幂等）。ultralytics stub 的 YOLO 导出不稳定，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO, settings
        except ImportError:
            raise ImportError("ultralytics 未安装，请运行: pip install ultralytics")

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR

        # 权重目录和下载目录统一指向 auto2dlabel/weights/
        settings.update({
            "weights_dir": str(WEIGHTS_DIR),
            "datasets_dir": str(WEIGHTS_DIR / "datasets"),
        })

        # 优先从本地 weights/ 加载，没有则自动下载
        local_path = WEIGHTS_DIR / self._model_name
        if local_path.exists():
            self._model = YOLO(str(local_path))
        else:
            self._model = YOLO(self._model_name)
        return self._model

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        model = self._load()

        if self._is_world:
            # YOLO-World：开放词汇，直接设类别
            model.set_classes(prompts)

        # YOLO.__call__ stub 为 Results | Tensor 联合，运行时恒为 list[Results]
        preds = cast("list[Results]", model(
            image_path, conf=confidence_threshold, iou=self._iou,
            device=self._device, verbose=False,
        ))

        results = []
        for pred in preds:
            if pred.boxes is None:
                continue
            for box_data in pred.boxes:
                box_data = cast("Boxes", box_data)  # stub 迭代推断为 BaseTensor，实际是 Boxes
                x1, y1, x2, y2 = box_data.xyxy[0].tolist()
                conf = float(box_data.conf[0])
                cls_id = int(box_data.cls[0])

                if self._is_world:
                    label = prompts[cls_id] if cls_id < len(prompts) else str(cls_id)
                else:
                    # 标准 YOLO：用 COCO 类别名，按 prompt 过滤
                    if cls_id < len(COCO_CLASSES):
                        label = COCO_CLASSES[cls_id]
                        if not _match_prompt(label, prompts):
                            continue  # 跳过不匹配的类别
                    else:
                        label = str(cls_id)

                results.append(DetectionResult(
                    x=x1, y=y1,
                    width=x2 - x1, height=y2 - y1,
                    label=label, confidence=conf,
                ))

        return results


# ============================================================
# PyTorch Vision 检测模型（torchvision.models.detection）
# ============================================================

class PyTorchVisionModel:
    """PyTorch Vision 检测模型封装。

    支持 Faster R-CNN、RetinaNet、SSD、FCOS 等 COCO 预训练模型。
    权重自动下载到 auto2dlabel/weights/ 目录。
    非开放词汇——使用 COCO 80 类，按 prompt 过滤。
    """

    # 映射模型名 → torchvision 构建函数
    _MODEL_BUILDERS: dict[str, tuple[str, str]] = {
        "fasterrcnn_resnet50_fpn":                ("detection", "fasterrcnn_resnet50_fpn"),
        "fasterrcnn_resnet50_fpn_v2":             ("detection", "fasterrcnn_resnet50_fpn_v2"),
        "fasterrcnn_mobilenet_v3_large_fpn":      (
            "detection", "fasterrcnn_mobilenet_v3_large_fpn"
        ),
        "fasterrcnn_mobilenet_v3_large_320_fpn":  (
            "detection", "fasterrcnn_mobilenet_v3_large_320_fpn"
        ),
        "retinanet_resnet50_fpn":                 ("detection", "retinanet_resnet50_fpn"),
        "retinanet_resnet50_fpn_v2":              ("detection", "retinanet_resnet50_fpn_v2"),
        "ssd300_vgg16":                           ("detection", "ssd300_vgg16"),
        "ssdlite320_mobilenet_v3_large":          ("detection", "ssdlite320_mobilenet_v3_large"),
        "fcos_resnet50_fpn":                      ("detection", "fcos_resnet50_fpn"),
    }

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        device: str | None = None,
        iou_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._device = device or get_device()
        self._iou = iou_threshold
        self._model = None

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        import os as _os

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR

        # 权重下载目录
        _os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))

        try:
            import torchvision  # noqa: F401, F811
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_320_fpn,
                fasterrcnn_mobilenet_v3_large_fpn,
                fasterrcnn_resnet50_fpn,
                fasterrcnn_resnet50_fpn_v2,
                fcos_resnet50_fpn,
                retinanet_resnet50_fpn,
                retinanet_resnet50_fpn_v2,
                ssd300_vgg16,
                ssdlite320_mobilenet_v3_large,
            )
        except ImportError:
            raise ImportError("torchvision 未安装，请运行: pip install torchvision")

        builders = {
            "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
            "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
            "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
            "retinanet_resnet50_fpn": retinanet_resnet50_fpn,
            "retinanet_resnet50_fpn_v2": retinanet_resnet50_fpn_v2,
            "ssd300_vgg16": ssd300_vgg16,
            "ssdlite320_mobilenet_v3_large": ssdlite320_mobilenet_v3_large,
            "fcos_resnet50_fpn": fcos_resnet50_fpn,
        }

        builder = builders.get(self._model_name)
        if builder is None:
            raise ValueError(
                f"未知 PyTorch 模型: {self._model_name}。可用: {list(builders.keys())}"
            )

        self._model = builder(weights="DEFAULT")
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def detect(
        self,
        image_path: str,
        prompts: list[str],
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        model = self._load()

        import torch
        from torchvision.transforms import functional as F  # noqa: N812

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(self._device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        from auto2dlabel.models.model_catalog import COCO_CLASSES

        results = []
        for box, label_idx, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            conf = float(score)
            if conf < confidence_threshold:
                continue

            # PyTorch 使用 1-based 索引（0=background），做 -1 偏移
            cls_id = int(label_idx) - 1
            if cls_id == -1:  # 跳过背景
                continue
            if 0 <= cls_id < len(COCO_CLASSES):
                label = COCO_CLASSES[cls_id]
            else:
                label = str(cls_id)

            if not _match_prompt(label, prompts):
                continue

            x1, y1, x2, y2 = box.tolist()
            results.append(DetectionResult(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
                label=label, confidence=conf,
            ))

        return results


# ============================================================
# 辅助函数
# ============================================================

from auto2dlabel.tools.device import get_device  # noqa: E402


def _match_prompt(label: str, prompts: list[str]) -> bool:
    """检查检测到的标签是否匹配用户指定的任意 prompt。"""
    label_lower = label.lower()
    for p in prompts:
        p_lower = p.lower()
        if p_lower in label_lower or label_lower in p_lower:
            return True
    return False  # 不匹配则过滤掉


def detect_with_retry(
    detect_fn: Callable[[float], list[Any]],
    confidence_threshold: float,
    retry_factor: float = 0.5,
) -> tuple[list[Any], bool, float]:
    """调用 detect_fn(conf) 检测；若 0 框则以 conf*retry_factor 重试一次（仅一次）。

    模型/iou 不变（detect_fn 闭包内固化）。返回 (dets, retried, final_threshold)。
    与结果格式无关（DetectionResult / SAHI dict / Bbox 列表），统一只看 len。
    """
    dets = detect_fn(confidence_threshold)
    if dets:
        return dets, False, confidence_threshold
    retry_conf = confidence_threshold * retry_factor
    dets = detect_fn(retry_conf)
    return dets, True, retry_conf


def create_detection_model(model_name: str | None = None, **kwargs) -> DetectionModel:
    """工厂函数：根据模型名创建合适的检测模型实例。

    自动识别模型类型：
    - Grounding DINO：含 "/"（HuggingFace ID）
    - Ultralytics YOLO：以 .pt 结尾
    - PyTorch Vision：前缀 "fasterrcnn_" | "retinanet_" | "ssd" | "fcos_"

    Args:
        model_name: 模型名。为 None 时按优先级:
                    env DETECTION_MODEL > DEFAULT_MODEL（yolo26x.pt）。
        **kwargs: 透传给具体模型类的参数。
    """
    if model_name is None:
        model_name = os.environ.get("DETECTION_MODEL", DEFAULT_MODEL)

    kwargs.setdefault("iou_threshold", 0.5)
    if "/" in model_name:
        return GroundingDINOModel(model_name=model_name, **kwargs)
    elif model_name.endswith(".pt"):
        return UltralyticsModel(model_name=model_name, **kwargs)
    elif _is_pytorch_model(model_name):
        return PyTorchVisionModel(model_name=model_name, **kwargs)
    else:
        from auto2dlabel.models.model_catalog import ALL_DETECTION_MODELS

        raise ValueError(
            f"无法识别的模型名: '{model_name}'。\n"
            f"可用: {', '.join(ALL_DETECTION_MODELS[:5])} ..."
        )


def _is_pytorch_model(name: str) -> bool:
    """检查是否为 PyTorch Vision 检测模型名。"""
    return any(
        name.startswith(prefix)
        for prefix in ("fasterrcnn_", "retinanet_", "ssd", "fcos_")
    )
