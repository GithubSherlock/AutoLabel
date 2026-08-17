"""检测模型抽象层。

封装 Grounding DINO 与 Ultralytics 检测模型（YOLO11/12/26），
提供统一接口：detect(image, prompts) -> List[DetectionResult]。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from auto2dlabel.models import Image, os
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
# Ultralytics 系列（YOLO11/12/26 + RT-DETR）
# ============================================================

class UltralyticsModel:
    """Ultralytics 检测模型封装。

    支持 YOLO11/12/26 及其 n/s/m/l/x 变体与 RT-DETR。
    首次使用自动下载 .pt 权重（目录外的自定义 .pt 亦可加载）。
    """

    def __init__(
        self,
        model_name: str = "yolo26x.pt",
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

        return self._parse_pred(preds[0], prompts)

    def detect_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
        confidence_threshold: float = 0.3,
        num_workers: int = 0,
    ) -> list[list[DetectionResult]]:
        """批量检测：ultralytics 原生 batch 推理（model(paths, batch=, workers=)）。

        每图解析与 detect 完全一致（同一 _parse_pred）；len>1 时传 batch=len。
        """
        model = self._load()

        if self._is_world:
            model.set_classes(prompts)

        kwargs: dict[str, Any] = {
            "conf": confidence_threshold, "iou": self._iou,
            "device": self._device, "verbose": False,
        }
        if num_workers:
            kwargs["workers"] = num_workers
        if len(image_paths) > 1:
            kwargs["batch"] = len(image_paths)

        preds = cast("list[Results]", model(image_paths, **kwargs))
        return [self._parse_pred(p, prompts) for p in preds]

    def _parse_pred(self, pred: "Results", prompts: list[str]) -> list[DetectionResult]:
        """单个 ultralytics Results → DetectionResult 列表（单图/批量共用）。"""
        results = []
        if pred.boxes is None:
            return results
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

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR

        # 权重下载目录
        os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))

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

        return self._parse_output(outputs, prompts, confidence_threshold)

    def detect_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
        confidence_threshold: float = 0.3,
        num_workers: int = 0,
    ) -> list[list[DetectionResult]]:
        """批量检测：torchvision 原生 list-of-tensors 推理。

        每图解析与 detect 完全一致（同一 _parse_output）；num_workers 仅
        ultralytics 引擎生效，torchvision 前向为同步张量推理，保留参数对齐接口。
        """
        model = self._load()

        import torch
        from torchvision.transforms import functional as F  # noqa: N812

        tensors = [
            F.to_tensor(Image.open(p).convert("RGB")).to(self._device)
            for p in image_paths
        ]
        with torch.no_grad():
            outputs = model(tensors)

        return [self._parse_output(o, prompts, confidence_threshold) for o in outputs]

    def _parse_output(
        self,
        outputs: dict[str, Any],
        prompts: list[str],
        confidence_threshold: float,
    ) -> list[DetectionResult]:
        """torchvision 检测输出 dict → DetectionResult 列表（单图/批量共用）。"""
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


# ============================================================
# SAHI 切片推理（单图；原 benchmarks/common.py 迁移至此，
# 供标注 pipeline 复用，benchmarks 的 run_detection_sahi 反向引用）
# ============================================================


def sahi_infer_yolo(
    model: Any, tile: Image.Image, prompts: list[str], conf: float
) -> list[dict[str, Any]]:
    """YOLO SAHI 推理：PIL Image → YOLO → dets。

    model 标注为 Any：访问私有 _model/_device（Protocol 不可见），
    与下方 cast(Any, model)._load() 同一惯例。
    """
    results = model._model(tile, conf=conf, iou=0.5, device=model._device, verbose=False)
    return _yolo_results_to_dets(results, prompts)


def sahi_infer_torchvision(
    model: Any, tile: Image.Image, prompts: list[str], conf: float
) -> list[dict[str, Any]]:
    """Torchvision SAHI 推理：PIL Image → Tensor → Faster R-CNN → dets。"""
    import torch
    from torchvision.transforms import functional as F  # noqa: N812

    tile_tensor = F.to_tensor(tile).to(model._device)
    with torch.no_grad():
        outputs = model._model([tile_tensor])[0]

    dets: list[dict[str, Any]] = []
    for box, label_idx, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        conf_val = float(score)
        if conf_val < conf:
            continue
        cls_id = int(label_idx) - 1  # 1-based → 0-based
        if cls_id < 0 or cls_id >= len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[cls_id]
        # prompt 过滤
        label_lower = label.lower()
        if not any(p.lower() in label_lower or label_lower in p.lower() for p in prompts):
            continue
        x1, y1, x2, y2 = box.tolist()
        dets.append({
            "name": label,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": conf_val,
        })
    return dets


def _yolo_results_to_dets(results: list[Any], prompts: list[str]) -> list[dict[str, Any]]:
    """将 Ultralytics YOLO 推理结果转为检测字典列表。"""
    dets: list[dict[str, Any]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id >= len(COCO_CLASSES):
                continue
            label = COCO_CLASSES[cls_id]
            # 按 prompt 过滤
            label_lower = label.lower()
            if not any(p.lower() in label_lower or label_lower in p.lower() for p in prompts):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "name": label,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(box.conf[0]),
            })
    return dets


def nms_per_class(dets: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """按类别分组执行 IoU NMS，返回去重后的检测列表。"""
    if not dets:
        return []

    # 按类别分组
    by_class: dict[str, list[dict[str, Any]]] = {}
    for d in dets:
        by_class.setdefault(d["name"], []).append(d)

    kept: list[dict[str, Any]] = []
    for cls_name, cls_dets in by_class.items():
        # 按置信度降序
        cls_dets.sort(key=lambda d: d["conf"], reverse=True)
        boxes = [d["bbox"] for d in cls_dets]

        # 贪心 NMS
        suppressed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            for j in range(i + 1, len(cls_dets)):
                if suppressed[j]:
                    continue
                if _box_iou(boxes[i], boxes[j]) > iou_threshold:
                    suppressed[j] = True

    return kept


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """两个 bbox 的 IoU（[x1,y1,x2,y2] 格式）。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def detect_image_sahi(
    model: DetectionModel,
    image_path: str,
    prompts: list[str],
    confidence_threshold: float = 0.3,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
) -> list[dict[str, Any]]:
    """SAHI 切片推理单张图像，返回检测 dict 列表。

    将大图切为重叠的 slice_size × slice_size 小块，分别推理后
    跨切片 NMS 合并。适用于航拍/高分辨率图中 YOLO resize 导致
    小物体丢失的场景。

    Args:
        model: 已加载的 DetectionModel 实例（需有 _model 属性）。
        image_path: 图像路径。
        prompts: 检测类别（英文名）。
        confidence_threshold: 置信度阈值。
        slice_size: 切片尺寸（正方形，像素）。
        overlap_ratio: 相邻切片重叠比例。

    Returns:
        [{"name": str, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
    """
    # 确保模型已加载（_load 为具体模型类方法，Protocol 不可见）
    if hasattr(model, "_load"):
        cast(Any, model)._load()

    # 根据模型类型选择推理后端
    if isinstance(model, PyTorchVisionModel):
        _infer_fn = sahi_infer_torchvision
    elif isinstance(model, UltralyticsModel):
        _infer_fn = sahi_infer_yolo
    else:
        # 回退：直接调 model.detect()
        results = model.detect(image_path, prompts, confidence_threshold)
        return [
            {
                "name": r.label,
                "bbox": [r.x, r.y, r.x + r.width, r.y + r.height],
                "conf": r.confidence,
            }
            for r in results
        ]

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 小图直接推理
    if w <= slice_size and h <= slice_size:
        return _infer_fn(model, img, prompts, confidence_threshold)

    # 切片推理
    step = int(slice_size * (1 - overlap_ratio))
    all_dets: list[dict[str, Any]] = []
    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))

    for y in y_starts:
        for x in x_starts:
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)

            tile = img.crop((x1, y1, x2, y2))
            for det in _infer_fn(model, tile, prompts, confidence_threshold):
                bx1, by1, bx2, by2 = det["bbox"]
                all_dets.append({
                    "name": det["name"],
                    "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                    "conf": det["conf"],
                })

    # 跨切片 NMS 合并（用 model 的 iou 阈值）
    iou_threshold = getattr(model, "_iou", 0.5)
    return nms_per_class(all_dets, iou_threshold)
