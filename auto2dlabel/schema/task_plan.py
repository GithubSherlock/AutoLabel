"""TaskPlan / TaskStep 数据结构。

一个 TaskPlan 由 LLM 从自然语言指令解析生成，包含多个 TaskStep。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """支持的标注任务类型。"""
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    IMAGE_CLASSIFICATION = "classification"
    POSE_ESTIMATION = "pose_estimation"
    OBB_DETECTION = "obb_detection"
    TRACKING = "tracking"


# 默认值（标注工具）
DEFAULT_CONFIDENCE = 0.1  # 标注工具低阈值，宁多勿漏
DEFAULT_IOU = 0.3
DEFAULT_MODEL = "yolo26x.pt"  # 统一默认检测模型（VOC 实测 2.4× fasterrcnn）
DEFAULT_EXPORT = "coco"
DEFAULT_TIMEOUT = 30

# 默认值（benchmark）
BENCHMARK_DEFAULT_CONF = 0.3
BENCHMARK_DEFAULT_IOU = 0.5
BENCHMARK_DEFAULT_MODEL = "yolov8x.pt"
BENCHMARK_DEFAULT_SEG_MODEL = "FastSAM-s.pt"
BENCHMARK_DEFAULT_MAX_IMAGES = 50

# 可用数据集（用于 LLM prompt + 验证）
BENCHMARK_DATASETS: dict[str, dict[str, str]] = {
    # 目标检测
    "coco":       {"script": "coco_benchmark.py",       "task_type": "detection"},
    "voc2007":    {"script": "voc_benchmark.py",        "task_type": "detection"},
    "kitti":      {"script": "kitti_benchmark.py",      "task_type": "detection"},
    "dota":       {"script": "dota_benchmark.py",       "task_type": "detection"},
    "mot":        {"script": "mot_benchmark.py",        "task_type": "detection"},
    # 实例分割
    "coco_seg":   {"script": "coco_seg_benchmark.py",   "task_type": "segmentation"},
    "cityscapes": {"script": "cityscapes_benchmark.py", "task_type": "segmentation"},
    "nuimages":   {"script": "nuimages_benchmark.py",   "task_type": "segmentation"},
    "d2sa":       {"script": "d2sa_benchmark.py",       "task_type": "segmentation"},
}

# 数据集中文名 → key 映射
DATASET_CN_MAP: dict[str, str] = {
    "coco": "coco", "coco2017": "coco", "coco val": "coco",
    "voc": "voc2007", "voc2007": "voc2007", "voc07": "voc2007", "pascal voc": "voc2007",
    "kitti": "kitti",
    "dota": "dota", "航拍": "dota",
    "mot": "mot", "mot17": "mot", "mot20": "mot", "行人检测": "mot", "密集行人": "mot",
    "coco分割": "coco_seg", "coco seg": "coco_seg", "coco 分割": "coco_seg",
    "cityscapes": "cityscapes", "城市街景": "cityscapes",
    "nuimages": "nuimages", "nu": "nuimages",
    "d2sa": "d2sa", "零售": "d2sa", "货架": "d2sa", "密集零售": "d2sa",
}

# 必填参数集合
REQUIRED_PARAMS = {"source", "prompts"}

# 可选参数默认值
OPTIONAL_PARAMS = {
    "confidence_threshold": DEFAULT_CONFIDENCE,
    "iou_threshold": DEFAULT_IOU,
    "model_name": DEFAULT_MODEL,
    "export_format": DEFAULT_EXPORT,
}


@dataclass
class TaskStep:
    """单个标注步骤的完整参数。

    所有字段均由 LLM 从自然语言中提取，未指定的字段使用默认值。
    """

    step_id: int
    task_type: str = "object_detection"
    source: str = ""                     # 数据路径
    prompts: list[str] = field(default_factory=list)  # 检测类别
    confidence_threshold: float = DEFAULT_CONFIDENCE
    iou_threshold: float = DEFAULT_IOU
    model_name: str = DEFAULT_MODEL
    export_format: str = DEFAULT_EXPORT
    sahi: bool = False                   # SAHI 切片推理

    @property
    def missing_params(self) -> list[str]:
        """返回缺失的必填参数名列表。"""
        missing = []
        if not self.source:
            missing.append("source（数据路径）")
        if not self.prompts:
            missing.append("prompts（检测类别）")
        return missing

    @property
    def is_complete(self) -> bool:
        return len(self.missing_params) == 0

    @property
    def summary(self) -> str:
        """一行摘要。"""
        return (
            f"Step {self.step_id}: {self.task_type} → {self.source} → "
            f"[{', '.join(self.prompts)}] → conf={self.confidence_threshold} "
            f"iou={self.iou_threshold} → {self.model_name} → {self.export_format}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_type": self.task_type,
            "source": self.source,
            "prompts": self.prompts,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_name": self.model_name,
            "export_format": self.export_format,
            "sahi": self.sahi,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskStep:
        return cls(
            step_id=d.get("step_id", 1),
            task_type=d.get("task_type", "object_detection"),
            source=d.get("source", ""),
            prompts=d.get("prompts", []),
            confidence_threshold=d.get("confidence_threshold", DEFAULT_CONFIDENCE),
            iou_threshold=d.get("iou_threshold", DEFAULT_IOU),
            model_name=d.get("model_name", DEFAULT_MODEL),
            export_format=d.get("export_format", DEFAULT_EXPORT),
            sahi=d.get("sahi", False),
        )


@dataclass
class TaskPlan:
    """多步标注计划。

    由 LLM 从用户自然语言指令中解析生成。
    """

    steps: list[TaskStep] = field(default_factory=list)
    confirm_timeout: int = DEFAULT_TIMEOUT  # 确认等待秒数
    raw_instruction: str = ""               # 原始用户指令

    @property
    def summary(self) -> str:
        """多行摘要，用于展示给用户确认。"""
        lines = [f"共 {len(self.steps)} 步任务："]
        for s in self.steps:
            lines.append(f"  {s.summary}")
        return "\n".join(lines)

    @property
    def first_incomplete(self) -> TaskStep | None:
        """返回第一个参数不完整的步骤。"""
        for s in self.steps:
            if not s.is_complete:
                return s
        return None

    @property
    def all_missing_params(self) -> dict[int, list[str]]:
        """返回所有步骤的缺失参数。"""
        return {s.step_id: s.missing_params for s in self.steps if not s.is_complete}

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "confirm_timeout": self.confirm_timeout,
            "raw_instruction": self.raw_instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskPlan:
        return cls(
            steps=[TaskStep.from_dict(s) for s in d.get("steps", [])],
            confirm_timeout=d.get("confirm_timeout", DEFAULT_TIMEOUT),
            raw_instruction=d.get("raw_instruction", ""),
        )


# ================================================================
# Benchmark 请求（Chat → Benchmark）
# ================================================================


@dataclass
class BenchmarkRequest:
    """Benchmark 请求，由 LLM 从自然语言中解析。

    包含运行 benchmark 所需的全部参数。Chat 命令中缺失参数时追问用户。
    """

    dataset: str = ""  # 数据集 key: coco/voc2007/kitti/dota/mot/coco_seg/cityscapes/nuimages/d2sa
    task_type: str = "detection"      # detection | segmentation
    model: str = BENCHMARK_DEFAULT_MODEL
    seg_model: str = BENCHMARK_DEFAULT_SEG_MODEL  # 仅 segmentation 使用
    conf: float = BENCHMARK_DEFAULT_CONF
    iou: float = BENCHMARK_DEFAULT_IOU
    max_images: int = BENCHMARK_DEFAULT_MAX_IMAGES
    top_classes: int = 20
    sahi: bool = False

    @property
    def missing_params(self) -> list[str]:
        """返回缺失的必填参数名列表。"""
        missing: list[str] = []
        if not self.dataset:
            missing.append("dataset（数据集，如 COCO / DOTA / MOT / cityscapes）")
        return missing

    @property
    def is_complete(self) -> bool:
        return len(self.missing_params) == 0

    @property
    def summary(self) -> str:
        """一行摘要，用于展示给用户确认。"""
        parts = [
            f"数据集: {self.dataset}",
            f"任务: {'实例分割' if self.task_type == 'segmentation' else '目标检测'}",
            f"模型: {self.model}",
        ]
        if self.task_type == "segmentation":
            parts.append(f"分割模型: {self.seg_model}")
        parts.append(f"conf={self.conf}  iou={self.iou}")
        parts.append(f"图像数: {self.max_images if self.max_images > 0 else '全部'}")
        if self.sahi:
            parts.append("SAHI: ✓")
        return " | ".join(parts)

    def to_cli_args(self) -> list[str]:
        """转换为 CLI 参数列表，用于 subprocess 执行 benchmark 脚本。"""
        args = [
            "--max-images", str(self.max_images),
            "--conf", str(self.conf),
            "--model", self.model,
            "--iou", str(self.iou),
            "--top-classes", str(self.top_classes),
        ]
        if self.sahi:
            args.append("--sahi")
        if self.task_type == "segmentation":
            args.extend(["--seg-model", self.seg_model])
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task_type": self.task_type,
            "model": self.model,
            "seg_model": self.seg_model,
            "conf": self.conf,
            "iou": self.iou,
            "max_images": self.max_images,
            "top_classes": self.top_classes,
            "sahi": self.sahi,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkRequest:
        return cls(
            dataset=d.get("dataset", ""),
            task_type=d.get("task_type", "detection"),
            model=d.get("model", BENCHMARK_DEFAULT_MODEL),
            seg_model=d.get("seg_model", BENCHMARK_DEFAULT_SEG_MODEL),
            conf=d.get("conf", BENCHMARK_DEFAULT_CONF),
            iou=d.get("iou", BENCHMARK_DEFAULT_IOU),
            max_images=d.get("max_images", BENCHMARK_DEFAULT_MAX_IMAGES),
            top_classes=d.get("top_classes", 20),
            sahi=d.get("sahi", False),
        )
