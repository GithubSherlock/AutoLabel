"""TaskPlan / TaskStep 数据结构。

一个 TaskPlan 由 LLM 从自然语言指令解析生成，包含多个 TaskStep。
"""

from __future__ import annotations

import re
from collections.abc import Set
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from auto2dlabel.configs.task_params import (
    TASK_PARAM_SPECS,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str,
    coerce_strs,
    guard_batch_size,
    guard_num_workers,
)


class TaskType(str, Enum):
    """支持的标注任务类型。"""
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    IMAGE_CLASSIFICATION = "classification"
    POSE_ESTIMATION = "pose_estimation"
    OBB_DETECTION = "obb_detection"
    TRACKING = "tracking"


# 默认值（标注工具）——单一事实源在 configs/task_params.py 规格表（default 字段派生）
DEFAULT_CONFIDENCE = float(TASK_PARAM_SPECS["confidence_threshold"].default)  # 低阈值，宁多勿漏
DEFAULT_IOU = float(TASK_PARAM_SPECS["iou_threshold"].default)
# 统一默认检测模型（VOC 实测 2.4× fasterrcnn）
DEFAULT_MODEL = str(TASK_PARAM_SPECS["model_name"].default)
DEFAULT_EXPORT = str(TASK_PARAM_SPECS["export_format"].default)
DEFAULT_TIMEOUT = 30

# 默认值（benchmark）
BENCHMARK_DEFAULT_CONF = 0.3
BENCHMARK_DEFAULT_IOU = 0.5
BENCHMARK_DEFAULT_MODEL = "yolo26x.pt"
BENCHMARK_DEFAULT_SEG_MODEL = "sam2_l.pt"  # GPU 复测：box-prompted 0.9278 vs FastSAM
# 0.4917（coco_seg）
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
    "imagenet1k": {"script": "classification_benchmark.py", "task_type": "classification"},
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
    "imagenet1k": "imagenet1k", "ilsvrc": "imagenet1k", "ilsvrc2012": "imagenet1k",
    "image net 1k": "imagenet1k",
}

# 必填参数集合（由规格表派生——required 字段单一事实源）
REQUIRED_PARAMS = {s.key for s in TASK_PARAM_SPECS.values() if s.required}

# 合法导出格式（tools/export.py exporters 注册表键；新增格式须同步该处）
EXPORT_FORMATS = frozenset(
    {"coco", "cls", "dota", "labelme", "mot", "yolo", "yolo_obb", "voc"}
)


@dataclass
class PlanQuestion:
    """对话式规划问题（v0.6）：LLM 输出，代码只收集回答并回喂，不解释。

    id 定位字段：2D 为 ``step{step_id}.{field}``（如 step1.source），
    3D 为裸字段名（如 frame_id）。LLM 输出类型不稳定 → from_dict 强转 str。
    """

    id: str
    question: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question}

    @classmethod
    def from_list(cls, raw: Any) -> list[PlanQuestion]:
        """LLM 输出容错：非 list → []；非 dict 元素 / 空 question 跳过。

        单一过滤入口（`_dict_to_plan` / `_dict_to_plan3d` 共用）。
        """
        if not isinstance(raw, list):
            return []
        questions: list[PlanQuestion] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            questions.append(cls(id=str(item.get("id", "")), question=question))
        return questions


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
        """返回缺失的必填参数名列表（规格表驱动：required 字段值 falsy 即缺失）。"""
        return [
            spec.label
            for spec in TASK_PARAM_SPECS.values()
            if spec.required and not getattr(self, spec.key)
        ]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_params) == 0

    def optional_missing(self, explicit: Set[str] = frozenset()) -> list[str]:
        """未指定的可选参数（规格表 ask 标记；2026-08-29 追问扩展）。

        值为默认值即视为「未指定」（LLM 未提取）→ 进入追问链；
        文案带默认提醒——回车/超时按默认继续。
        explicit: 指令文本显式提及的 key（guard_explicit_params）——
            「模型用 yolo26x.pt」值恰为默认也视为已指定，不误追问。
        """
        return [
            f"{spec.label}（默认 {spec.default}）"
            for spec in TASK_PARAM_SPECS.values()
            if spec.ask
            and getattr(self, spec.key) == spec.default
            and spec.key not in explicit
        ]

    @property
    def summary(self) -> str:
        """一行摘要。

        注：prompts 用全角括号（markup 安全）——summary 会经 console.print
        渲染，半角方括号 [car, person] 会被 Rich 当作无效标签吞掉（实测）。
        """
        batch_note = (
            f" batch={self.batch_size}/{self.num_workers}w"
            if self.batch_size is not None or self.num_workers is not None
            else ""
        )
        return (
            f"Step {self.step_id}: {self.task_type} → {self.source} → "
            f"（{', '.join(self.prompts)}） → conf={self.confidence_threshold} "
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
    questions: list[PlanQuestion] = field(default_factory=list)  # v0.6 对话问题
    dialog_context: str = ""  # v0.6 对话累积文本（Step 4 重解析用，不入 raw_instruction）

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
        """返回所有步骤的缺失参数（required，必问）。"""
        return {s.step_id: s.missing_params for s in self.steps if not s.is_complete}

    def all_optional_missing(
        self, explicit: Set[str] = frozenset()
    ) -> dict[int, list[str]]:
        """返回所有步骤未指定的可选参数（ask 标记，2026-08-29 追问扩展）。

        explicit: 指令文本显式提及的 key 集合（guard_explicit_params），
            透传给每步 optional_missing 过滤——显式指定默认值不误追问。
        """
        return {
            s.step_id: s.optional_missing(explicit)
            for s in self.steps
            if s.optional_missing(explicit)
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "steps": [s.to_dict() for s in self.steps],
            "confirm_timeout": self.confirm_timeout,
            "raw_instruction": self.raw_instruction,
        }
        # 非空才输出（防 JSON 膨胀惯例同 edited_by_human）；旧消费者零破坏
        if self.questions:
            out["questions"] = [q.to_dict() for q in self.questions]
        if self.dialog_context:
            out["dialog_context"] = self.dialog_context
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskPlan:
        return cls(
            steps=[TaskStep.from_dict(s) for s in d.get("steps", [])],
            confirm_timeout=d.get("confirm_timeout", DEFAULT_TIMEOUT),
            raw_instruction=d.get("raw_instruction", ""),
            questions=PlanQuestion.from_list(d.get("questions")),
            dialog_context=str(d.get("dialog_context", "")),
        )


# ================================================================
# LLM 输出代码级守卫（防御纵深，2026-08-29）
# ================================================================

_TASK_TYPES = frozenset({t.value for t in TaskType})


def sanitize_task_plan(plan: TaskPlan) -> list[str]:
    """LLM 输出校验/强转（LLM 不可全信——实测「COCO2017」→ batch_size=2017）。

    - batch_size/num_workers：以指令文本代码级提取为准（guard_*），
      LLM 值不采信；无显式表述 → None（自动实测/交互）
    - conf/iou：值域 (0,1] 外回默认（路径数字误提 → 全零静默失败的根治）
    - 类型强转：prompts 单字符串 → 列表（防逐字符 join）、
      sahi "false" 字符串 → False（防 Python truthy 坑）、
      confirm_timeout 垃圾 → 默认（防 thread.join TypeError）
    - 枚举白名单：task_type/export_format 垃圾 → 默认（防 KeyError/静默误路由）

    Returns:
        修正记录（"key: old -> new"），供调用方日志——守卫触发必须可见。
    """
    fixes: list[str] = []
    # 对话回喂可能含用户补充的超参（如回答里说「批量4」），一并纳入提取文本
    text = f"{plan.dialog_context} {plan.raw_instruction}"

    def _fix(step: TaskStep, key: str, new: Any) -> None:
        old = getattr(step, key)
        if old != new:
            fixes.append(f"{key}: {old!r} -> {new!r}")
            setattr(step, key, new)

    for i, step in enumerate(plan.steps):
        _fix(step, "source", coerce_str(step.source))
        _fix(step, "prompts", coerce_strs(step.prompts))
        _fix(
            step,
            "confidence_threshold",
            coerce_float(step.confidence_threshold, DEFAULT_CONFIDENCE, lo=0.0, hi=1.0),
        )
        _fix(
            step,
            "iou_threshold",
            coerce_float(step.iou_threshold, DEFAULT_IOU, lo=0.0, hi=1.0),
        )
        _fix(step, "batch_size", guard_batch_size(text))
        _fix(step, "num_workers", guard_num_workers(text))
        _fix(step, "sahi", coerce_bool(step.sahi))
        _fix(step, "model_name", coerce_str(step.model_name, DEFAULT_MODEL) or DEFAULT_MODEL)
        _fix(step, "model_hint", coerce_str(step.model_hint))
        fmt = coerce_str(step.export_format).replace("-", "_")  # yolo-obb → yolo_obb
        _fix(step, "export_format", fmt if fmt in EXPORT_FORMATS else DEFAULT_EXPORT)
        if step.task_type not in _TASK_TYPES:
            _fix(step, "task_type", "object_detection")
        if not isinstance(step.step_id, int) or isinstance(step.step_id, bool):
            _fix(step, "step_id", i + 1)  # 枚举索引保唯一，防字典键覆盖

    ct = coerce_int(plan.confirm_timeout)
    new_ct = ct if ct is not None and 0 <= ct <= 3600 else DEFAULT_TIMEOUT
    if plan.confirm_timeout != new_ct:
        fixes.append(f"confirm_timeout: {plan.confirm_timeout!r} -> {new_ct!r}")
        plan.confirm_timeout = new_ct
    return fixes


# ================================================================
# Benchmark 请求（Chat → Benchmark）
# ================================================================


@dataclass
class BenchmarkRequest:
    """Benchmark 请求，由 LLM 从自然语言中解析。

    包含运行 benchmark 所需的全部参数。Chat 命令中缺失参数时追问用户。
    """

    dataset: str = ""  # 数据集 key: coco/voc2007/kitti/dota/dota_obb/mot/coco_seg/
    # cityscapes/nuimages/d2sa/imagenet100
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


# 图片数显式表述（代码级提取为准：LLM 把路径里的 2017 误提为 max_images
# → 全量长跑几小时，必须守卫）
_IMAGE_COUNT_RE = re.compile(
    r"(?:(\d+)\s*(?:张|幅|images?|imgs?|frames?|张图))|(?:max(?:[_-]?images?)?\s*[=:：]?\s*(\d+))",
    re.IGNORECASE,
)
_ALL_IMAGES_RE = re.compile(r"全部|所有|\ball\b", re.IGNORECASE)


def _guard_max_images(instruction: str) -> int:
    """max_images 守卫：「全部/所有/all」→ 0（全量）；「N张/N images/max=N」→ N；
    无表述 → 默认（LLM 值不采信——路径数字 2017 会被误提为图片数）。"""
    if _ALL_IMAGES_RE.search(instruction):
        return 0
    m = _IMAGE_COUNT_RE.search(instruction)
    if m:
        return int(next(g for g in m.groups() if g))
    return BENCHMARK_DEFAULT_MAX_IMAGES


def sanitize_benchmark_request(request: BenchmarkRequest, instruction: str) -> list[str]:
    """Benchmark 参数守卫（与 sanitize_task_plan 同漏洞类，2026-08-29）。

    - dataset 未知（非 BENCHMARK_DATASETS 键）→ ""（缺参追问，防下游 KeyError）
    - conf/iou 值域外回默认；max_images 以指令文本代码级提取为准
    - top_classes/model/sahi 类型强转 + 值域白名单
    """
    fixes: list[str] = []

    def _fix(key: str, new: Any) -> None:
        old = getattr(request, key)
        if old != new:
            fixes.append(f"{key}: {old!r} -> {new!r}")
            setattr(request, key, new)

    _fix("dataset", request.dataset if request.dataset in BENCHMARK_DATASETS else "")
    _fix("conf", coerce_float(request.conf, BENCHMARK_DEFAULT_CONF, lo=0.0, hi=1.0))
    _fix("iou", coerce_float(request.iou, BENCHMARK_DEFAULT_IOU, lo=0.0, hi=1.0))
    _fix("max_images", _guard_max_images(instruction))
    tc = coerce_int(request.top_classes)
    _fix("top_classes", tc if tc is not None and 1 <= tc <= 1000 else 20)
    _fix("sahi", coerce_bool(request.sahi))
    _fix("model", coerce_str(request.model, BENCHMARK_DEFAULT_MODEL) or BENCHMARK_DEFAULT_MODEL)
    _fix(
        "seg_model",
        coerce_str(request.seg_model, BENCHMARK_DEFAULT_SEG_MODEL) or BENCHMARK_DEFAULT_SEG_MODEL,
    )
    if request.task_type not in {"detection", "segmentation", "classification", "obb_detection"}:
        _fix("task_type", "detection")
    return fixes
