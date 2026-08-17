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
BENCHMARK_DEFAULT_MODEL = "yolo26x.pt"
BENCHMARK_DEFAULT_SEG_MODEL = "sam2_l.pt"  # GPU 复测：box-prompted 0.9278 vs FastSAM 0.4917（coco_seg）
BENCHMARK_DEFAULT_MAX_IMAGES = 50

# 可用数据集（用于 LLM prompt + 验证）
BENCHMARK_DATASETS: dict[str, dict[str, str]] = {
    # 目标检测
    "coco":       {"script": "coco_benchmark.py",       "task_type": "detection"},
    "voc2007":    {"script": "voc_benchmark.py",        "task_type": "detection"},
    "kitti":      {"script": "kitti_benchmark.py",      "task_type": "detection"},
    "dota":       {"script": "dota_benchmark.py",       "task_type": "detection"},
    "dota_obb":   {"script": "dota_obb_benchmark.py",   "task_type": "obb_detection"},
    "mot":        {"script": "mot_benchmark.py",        "task_type": "detection"},
    # 实例分割
    "coco_seg":   {"script": "coco_seg_benchmark.py",   "task_type": "segmentation"},
    "cityscapes": {"script": "cityscapes_benchmark.py", "task_type": "segmentation"},
    "nuimages":   {"script": "nuimages_benchmark.py",   "task_type": "segmentation"},
    "d2sa":       {"script": "d2sa_benchmark.py",       "task_type": "segmentation"},
    # 图像分类
    "imagenet100": {"script": "classification_benchmark.py", "task_type": "classification"},
}

# 数据集中文名 → key 映射
DATASET_CN_MAP: dict[str, str] = {
    "coco": "coco", "coco2017": "coco", "coco val": "coco",
    "voc": "voc2007", "voc2007": "voc2007", "voc07": "voc2007", "pascal voc": "voc2007",
    "kitti": "kitti",
    "dota": "dota", "航拍": "dota",
    "dota_obb": "dota_obb", "obb": "dota_obb", "旋转框": "dota_obb",
    "rotated": "dota_obb", "oriented": "dota_obb",
    "mot": "mot", "mot17": "mot", "mot20": "mot", "行人检测": "mot", "密集行人": "mot",
    "coco分割": "coco_seg", "coco seg": "coco_seg", "coco 分割": "coco_seg",
    "cityscapes": "cityscapes", "城市街景": "cityscapes",
    "nuimages": "nuimages", "nu": "nuimages",
    "d2sa": "d2sa", "零售": "d2sa", "货架": "d2sa", "密集零售": "d2sa",
    "imagenet100": "imagenet100", "imagenet": "imagenet100", "image net": "imagenet100",
}

# 必填参数集合
REQUIRED_PARAMS = {"source", "prompts"}


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
    model_hint: str = ""                 # LLM 选型理由（仅展示，不影响执行）
    export_format: str = DEFAULT_EXPORT
    sahi: bool = False                   # SAHI 切片推理
    num_workers: int | None = None       # DataLoader 子进程数（None = 按 GPU 推荐/逐图）
    batch_size: int | None = None        # 批量推理每批图像数（1 = 逐图）

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
        batch_note = (
            f" batch={self.batch_size}/{self.num_workers}w"
            if self.batch_size is not None or self.num_workers is not None
            else ""
        )
        return (
            f"Step {self.step_id}: {self.task_type} → {self.source} → "
            f"[{', '.join(self.prompts)}] → conf={self.confidence_threshold} "
            f"iou={self.iou_threshold} → {self.model_name} → {self.export_format}"
            f"{batch_note}"
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
            "model_hint": self.model_hint,
            "export_format": self.export_format,
            "sahi": self.sahi,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
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
            model_hint=d.get("model_hint", ""),
            export_format=d.get("export_format", DEFAULT_EXPORT),
            sahi=d.get("sahi", False),
            num_workers=d.get("num_workers"),
            batch_size=d.get("batch_size"),
        )


# ================================================================
# 批量推理超参数（batch_size / num_workers）推荐规则
# ================================================================


def detect_gpu_memory_gb() -> int | None:
    """检测首块 GPU 显存（GB）；无 CUDA GPU 返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory / (1024 ** 3))


def recommend_batch_params(task_type: str, gpu_memory_gb: int | None) -> tuple[int, int]:
    """按 GPU 显存档位推荐 batch_size——LLM prompt 与代码兜底共用。

    静态兜底（规划阶段用，模型尚未加载无法实测）；执行阶段由
    tools/device.resolve_batch_params 动态实测修正。
    num_workers 按 CPU 核数推荐（DataLoader 并行度与显存无关，
    tools/device.recommend_num_workers），不再随显存档位变化。

    显存预算依据（RTX 4090 24GB 实测量级）：yolo26x 640 fp32 ~14GB/批、
    maskrcnn 1024 输入 ~2GB/张、resnet18 ~0.1GB/张。推荐值保守留余量，
    用户显式指定的参数优先。
    """
    from auto2dlabel.tools.device import recommend_num_workers

    if gpu_memory_gb is None:
        return 1, recommend_num_workers()
    if task_type in ("classification", "image_classification"):
        bs = 16 if gpu_memory_gb >= 20 else 8 if gpu_memory_gb >= 10 else 4
    elif task_type in ("instance_segmentation", "semantic_segmentation"):
        bs = 4 if gpu_memory_gb >= 20 else 2 if gpu_memory_gb >= 10 else 1
    else:  # object_detection / obb_detection
        bs = 8 if gpu_memory_gb >= 20 else 4 if gpu_memory_gb >= 10 else 2
    return bs, recommend_num_workers()


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

    dataset: str = ""  # 数据集 key: coco/voc2007/kitti/dota/dota_obb/mot/coco_seg/cityscapes/nuimages/d2sa/imagenet100
    task_type: str = "detection"      # detection | segmentation | classification | obb_detection
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
        if self.task_type == "segmentation":
            task_cn = "实例分割"
        elif self.task_type == "classification":
            task_cn = "图像分类"
        elif self.task_type == "obb_detection":
            task_cn = "旋转框检测"
        else:
            task_cn = "目标检测"
        parts = [
            f"数据集: {self.dataset}",
            f"任务: {task_cn}",
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
        if self.task_type == "classification":
            args.extend(["--dataset", self.dataset])
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
