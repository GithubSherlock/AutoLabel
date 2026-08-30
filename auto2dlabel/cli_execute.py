"""TaskPlan 执行器 —— chat 命令 Step 5 的实现（检测/分割/分类/OBB/姿态/跟踪六类任务）。"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import datetime as _datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from auto2dlabel.agent.state import AgentState
from auto2dlabel.cli_common import collect_images, console, display_results, triage_and_export
from auto2dlabel.configs.model_catalog import SEGMENTATION_MODELS, TORCHVISION_SEG_MODELS
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import DEFAULT_MODEL, TaskPlan, TaskStep
from auto2dlabel.tools.constraints import ReferentialConstraint
from auto2dlabel.tools.tracking import DEFAULT_REID_MODEL

if TYPE_CHECKING:
    from auto2dlabel.agent.evaluate import QualityReport
    from auto2dlabel.models.classification import ClassificationModel
    from auto2dlabel.models.detection import DetectionModel
    from auto2dlabel.models.obb import OBBModel
    from auto2dlabel.models.pose import PoseModel


def execute_plan(
    plan: TaskPlan,
    sahi: bool = False,
    explicit_batch_size: int | None = None,
    explicit_num_workers: int | None = None,
    use_bot_sort: bool = False,
    reid_model_name: str = DEFAULT_REID_MODEL,
    output_dir: str = "outputs",
    viz: bool = True,
    constraint: ReferentialConstraint | None = None,
    batch_strategy: bool = False,
    llm: Any | None = None,
    refer_l2: bool = False,
    refer_l3: bool = False,
) -> None:
    """顺序执行 TaskPlan 的每个步骤。

    explicit_batch_size/explicit_num_workers：CLI 显式指定的批量推理超参数
    （优先于交互询问值与动态实测推荐）。
    use_bot_sort/reid_model_name/output_dir：tracking 步骤的跟踪器选择、
    ReID 特征模型与输出目录（run --track 与 chat 共用 TrackingTool 管线）。
    viz：tracking 步骤是否输出逐帧 PNG 可视化（--no-viz 关闭，省磁盘）。
    constraint：tracking 步骤的指代约束（v0.4 3a——chat 从 raw_instruction
    代码级解析；None = 纯类别跟踪）。
    batch_strategy：多图检测步骤抽样统计 + LLM 一次性调参（3b；LLM 每批
    1 次调用，逐图仍零 LLM）；llm 为 chat 已建的客户端（None = 无 key，跳过）。
    refer_l2：tracking 步骤启用 v0.5 指代 L2（Florence-2 首帧解析锁定目标；
    chat 由关系词自动触发或 --refer-l2 显式启用）。
    refer_l3：tracking 步骤直用 v0.5 指代 L3（Qwen2-VL-7B，GPU）；refer_l2
    路径默认阶梯升级（L2 失败自动升级 L3）。
    """
    _plan_t0 = _time.time()
    steps_results: list[dict[str, Any]] = []

    # 启动显存体检（2026-08-29）：上次运行未正常退出（Ctrl+Z 挂起/终端未关）
    # 残留进程占显存 → 开跑前黄字提醒（不阻断，单图 OOM 时另有跳过保护）
    from auto2dlabel.tools.device import check_gpu_headroom

    check_gpu_headroom()

    for step in plan.steps:
        _print_step_header(step)

        _t0 = _time.time()
        _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        source_path = Path(step.source)

        # 序列跟踪（视频/帧目录）：走 TrackingTool 管线（与 run --track 共用，
        # 不进逐图 collect_images 循环）
        if step.task_type == "tracking":
            _execute_tracking_step(
                step,
                use_bot_sort=use_bot_sort,
                reid_model_name=reid_model_name,
                output_dir=output_dir,
                explicit_batch_size=explicit_batch_size,
                explicit_num_workers=explicit_num_workers,
                steps_results=steps_results,
                viz=viz,
                constraint=constraint,
                refer_l2=refer_l2,
                refer_l3=refer_l3,
                raw_instruction=plan.raw_instruction,
            )
            continue

        # 收集图像
        images = collect_images(source_path, batch=True if source_path.is_dir() else False)
        if not images:
            console.print(f"[red]未找到图像: {step.source}[/red]")
            continue

        # 路由检测/分割模型名（分割模型名不能用于检测——目录派生 + 前缀容错）
        det_name, seg_name = _route_model(step)

        model: DetectionModel | None = None
        if det_name:
            from auto2dlabel.models.detection import create_detection_model

            model = create_detection_model(det_name, iou_threshold=step.iou_threshold)

        # 分割模型步骤级复用（原每图重建——权重 IO/加载只发生一次）
        seg_model: Any | None = None
        if seg_name:
            from auto2dlabel.models.segmentation import create_segmentation_model

            seg_model = create_segmentation_model(seg_name)

        # OBB/分类/姿态模型步骤级复用（原每块重建——同上，权重加载只发生一次）
        obb_model: Any | None = None
        if step.task_type == "obb_detection":
            from auto2dlabel.models.obb import create_obb_model

            obb_model = create_obb_model(step.model_name, iou_threshold=step.iou_threshold)
        cls_model: Any | None = None
        if step.task_type == "classification":
            from auto2dlabel.models.classification import create_classification_model

            cls_model = create_classification_model(step.model_name)
        pose_model: Any | None = None
        if step.task_type == "pose_estimation":
            from auto2dlabel.models.pose import create_pose_model

            pose_model = create_pose_model(step.model_name, iou_threshold=step.iou_threshold)

        # 批次级策略（3b）：多图检测步骤抽样统计 + LLM 一次性调参（失败降级零影响）
        if batch_strategy and step.task_type == "object_detection" and det_name and len(images) > 1:
            _apply_batch_strategy(step, images, model, llm, plan.raw_instruction)

        # 批量推理超参数：显式 > 动态实测（模型加载后探针测单图峰值）> 静态表
        batch_size, num_workers = _resolve_step_batch(
            step, images, det_name, seg_name, model, seg_model, obb_model, cls_model,
            pose_model, sahi, explicit_batch_size, explicit_num_workers,
        )
        console.print(f"  批量: batch_size={batch_size}  num_workers={num_workers}")
        # 分块执行；批量路径 OOM 时剩余图降级逐图（batch=1 历史已验证路径）
        idx = 0
        try:
            for i in range(0, len(images), batch_size):
                _execute_chunk(
                    step, images[i:i + batch_size], det_name, seg_name, model, seg_model,
                    sahi, _t0, _ts, steps_results,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                    num_workers=num_workers,
                )
                idx = i + batch_size
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            console.print("[yellow]⚠ 批量推理 OOM，剩余图像降级逐图执行[/yellow]")
            import torch

            torch.cuda.empty_cache()
            # 单图仍 OOM（如大模型加载都装不下，2026-08-29 SAM3 实测）：不崩整批，
            # 跳图带指引继续（其余图像照常产出，失败可见不静默）
            oom_hint_shown = False
            for img in images[idx:]:
                try:
                    _execute_chunk(
                        step, [img], det_name, seg_name, model, seg_model,
                        sahi, _t0, _ts, steps_results,
                        obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                        num_workers=num_workers,
                    )
                except RuntimeError as e2:
                    if "out of memory" not in str(e2).lower():
                        raise
                    torch.cuda.empty_cache()
                    if not oom_hint_shown:
                        console.print(
                            "[yellow]⚠ 单图仍 OOM：GPU 显存不足（可能被其他进程占用，"
                            "请用 nvidia-smi 检查并清理后重跑，或换更小的模型"
                            "（如分割 SAM3 → sam2_l.pt）。剩余图像继续尝试...[/yellow]"
                        )
                        oom_hint_shown = True
                    console.print(f"[yellow]跳过: {img}（显存不足）[/yellow]")

    _log_plan(plan, steps_results, _plan_t0)


def _execute_tracking_step(
    step: TaskStep,
    use_bot_sort: bool,
    reid_model_name: str,
    output_dir: str,
    explicit_batch_size: int | None,
    explicit_num_workers: int | None,
    steps_results: list[dict[str, Any]],
    viz: bool = True,
    constraint: ReferentialConstraint | None = None,
    refer_l2: bool = False,
    refer_l3: bool = False,
    raw_instruction: str = "",
) -> None:
    """tracking 步骤：委托 TrackingTool 序列管线（与 run --track 同一实现）。

    batch/workers 取「CLI 显式 > step 字段」（与其余任务一致的优先级）；
    逐帧 JSON 用通用检测格式（step.export_format=mot 时映射回 coco——
    MOT 恒为序列级独立合并导出）；失败（ValueError）红字提示后跳过本步。
    constraint：指代约束（chat 从 raw_instruction 解析，含属性/方位；--roi
    由 run 路径注入）→ TrackingTool 过滤层。refer_l2/refer_l3 启用 v0.5
    指代 L2/L3（首帧解析锁定，替代属性/方位链、ROI 仍叠加）；refer_l3
    直用 L3（GPU），refer_l2 为阶梯升级（L2 失败自动升级 L3）。
    """
    from auto2dlabel.tools.tracking import TrackingTool

    batch_size = explicit_batch_size if explicit_batch_size is not None else step.batch_size
    num_workers = explicit_num_workers if explicit_num_workers is not None else step.num_workers
    per_frame_format = "coco" if step.export_format == "mot" else step.export_format

    # 指代 L2/L3：解析器 + 英文短语（chat 从原始指令构造）
    referential = None
    referential_phrase: str | None = None
    if (refer_l2 or refer_l3) and constraint is not None:
        from auto2dlabel.tools.constraints import build_referential_phrase

        referential_phrase = build_referential_phrase(raw_instruction, constraint)
        if refer_l3:
            from auto2dlabel.models.referential_l3 import (
                create_referential_l3_resolver,
            )

            referential = create_referential_l3_resolver()
        else:
            from auto2dlabel.models.referential_l3 import (
                create_cascade_referential_resolver,
            )

            referential = create_cascade_referential_resolver()

    tool = TrackingTool(
        model_name=step.model_name,
        iou_threshold=step.iou_threshold,
        use_sahi=step.sahi,
        use_bot_sort=use_bot_sort,
        reid_model_name=reid_model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        export_format=per_frame_format,
        viz=viz,
        constraint=constraint,
        referential=referential,
        referential_phrase=referential_phrase,
    )
    try:
        summary = tool.forward(
            step.source, step.prompts, confidence_threshold=step.confidence_threshold
        )
    except ValueError as e:
        console.print(f"[red]跟踪失败: {e}[/red]")
        return
    steps_results.append(
        {
            "step_id": step.step_id,
            "source": step.source,
            "model": step.model_name,
            "prompts": step.prompts,
            "bbox_count": summary["bboxes"],
            "triage_summary": {},
            "track_ids": summary["track_ids"],
            "mot_path": summary["mot_path"],
            "video_path": summary.get("video_path"),
        }
    )


def _apply_batch_strategy(
    step: Any,  # TaskStep（测试可用 duck-typed SimpleNamespace）
    images: list[Path],
    model: Any,
    llm: Any,
    instruction: str,
) -> dict[str, Any] | None:
    """批次级策略三步：抽样统计 → LLM 单轮调参 → 覆写步骤参数。

    任何异常（无 LLM / 无 key / 输出非法）黄字降级代码级默认参数，返回 None。
    抽样复用本步已加载的检测模型（model 可能为 None → 按步名自建）。
    """
    from auto2dlabel.agent.batch_strategy import (
        apply_strategy,
        llm_tune_strategy,
        sample_stats,
    )

    if llm is None:
        console.print("[dim]批次策略: 无 LLM 客户端，跳过（沿用代码级默认参数）[/dim]")
        return None
    try:
        stats = sample_stats(
            images, step.prompts, step.confidence_threshold,
            model=model, det_model_name=step.model_name,
        )
        strategy = llm_tune_strategy(llm, instruction, stats)
        applied = apply_strategy(step, strategy)
    except Exception as e:
        console.print(f"[yellow]批次策略失败，沿用代码级默认参数: {e}[/yellow]")
        return None

    console.print(
        f"[dim]批次策略: conf {applied['old_confidence_threshold']}"
        f"→{applied['confidence_threshold']}"
        + (f"  建议模型: {applied['suggest_model']}" if applied.get("suggest_model") else "")
        + (f"  （{applied['note']}）" if applied.get("note") else "")
        + "[/dim]"
    )
    return applied


def _print_step_header(step: TaskStep) -> None:
    console.print(f"\n[bold]━━━ Step {step.step_id}: {step.task_type} ━━━[/bold]")
    console.print(f"  数据: {step.source}")
    console.print(f"  类别: {step.prompts}")
    console.print(
        f"  模型: {step.model_name}  conf={step.confidence_threshold}  iou={step.iou_threshold}"
    )
    if step.model_hint:
        console.print(f"  [dim]模型建议: {step.model_hint}[/dim]")


def _is_seg_name(n: str) -> bool:
    """模型名是否指向分割模型（目录精确名 + 前缀容错）。"""
    lower = n.lower()
    return (
        lower in {"sam3", "sam", "fastsam", "maskrcnn", "sam2"}
        or n in SEGMENTATION_MODELS
        or lower in TORCHVISION_SEG_MODELS
        or lower.startswith(("sam", "fastsam", "fcn_", "deeplabv3_", "lraspp_"))
    )


def _is_self_detect_seg(n: str) -> bool:
    """自带检测的分割模型（sam3/maskrcnn/torchvision 语义分割）。"""
    lower = n.lower()
    return (
        "sam3" in lower
        or "maskrcnn" in lower
        or lower in TORCHVISION_SEG_MODELS
        or lower.startswith(("fcn_", "deeplabv3_", "lraspp_"))
    )


def _route_model(step: TaskStep) -> tuple[str | None, str | None]:
    """路由检测/分割模型名。返回 (det_name, seg_name)；det_name=None 表示跳过检测。

    分割模型名（如 sam3）不能用于检测——目录派生 + 前缀容错
    （planner 可能输出目录精确名如 sam2_t.pt，不得误入检测分支）。
    """
    det_name: str | None = step.model_name
    seg_name: str | None = None
    if _is_seg_name(step.model_name):
        det_name = DEFAULT_MODEL
        seg_name = step.model_name

    # 自带检测的分割模型跳过单独的检测步骤
    if _is_self_detect_seg(step.model_name):
        det_name = None  # 信号：跳过检测

    # 分类/旋转框/姿态任务不创建普通检测模型
    if step.task_type in ("classification", "obb_detection", "pose_estimation"):
        det_name = None

    return det_name, seg_name


def _build_tune_infer_fn(
    step: TaskStep,
    det_name: str | None,
    seg_name: str | None,
    model: Any,
    seg_model: Any,
    obb_model: Any,
    cls_model: Any,
    pose_model: Any,
    sahi: bool,
) -> Callable[[list[str]], Any] | None:
    """按任务构建动态实测闭包（批量推理探针）；无批量能力 → None（回退静态表）。

    SAHI 切片推理保持逐图；SAM 系列/语义分割无 *_batch → None。
    探针内 num_workers=0（DataLoader 并行度不影响显存）。
    """
    if sahi:
        return None

    if step.task_type == "classification":
        if cls_model is None or not hasattr(cls_model, "classify_batch"):
            return None
        return lambda paths: cls_model.classify_batch(paths, step.prompts, top_k=5)
    if step.task_type == "obb_detection":
        if obb_model is None or not hasattr(obb_model, "detect_obb_batch"):
            return None
        return lambda paths: obb_model.detect_obb_batch(
            paths, step.prompts, step.confidence_threshold, 0,
        )
    if step.task_type == "pose_estimation":
        if pose_model is None or not hasattr(pose_model, "detect_pose_batch"):
            return None
        return lambda paths: pose_model.detect_pose_batch(
            paths, step.prompts, step.confidence_threshold, 0,
        )
    if step.task_type in ("instance_segmentation", "semantic_segmentation"):
        # 自带检测分割（maskrcnn）有 generate_batch；两段式受益在检测步
        if seg_name and _is_self_detect_seg(seg_name):
            if seg_model is None or not hasattr(seg_model, "generate_batch"):
                return None
            prompt_bboxes = [
                Bbox(x=0, y=0, width=1, height=1, label=p, confidence=1.0)
                for p in step.prompts
            ]
            return lambda paths: seg_model.generate_batch(paths, prompt_bboxes)
        det_name = det_name or DEFAULT_MODEL
    if det_name and model is not None and hasattr(model, "detect_batch"):
        return lambda paths: model.detect_batch(
            paths, step.prompts, step.confidence_threshold, 0,
        )
    return None


def _resolve_step_batch(
    step: TaskStep,
    images: list[Path],
    det_name: str | None,
    seg_name: str | None,
    model: Any,
    seg_model: Any,
    obb_model: Any,
    cls_model: Any,
    pose_model: Any,
    sahi: bool,
    explicit_batch_size: int | None,
    explicit_num_workers: int | None,
) -> tuple[int, int]:
    """解析本步批量超参数：显式 CLI > 动态实测 > 静态表。

    显式 CLI 值由 execute_plan 参数透传恒优先；step 字段值（LLM 规划 / 交互询问）
    仅作规划阶段兜底，执行阶段以动态实测修正。
    """
    from auto2dlabel.tools.device import recommend_num_workers, resolve_batch_params

    batch = explicit_batch_size if explicit_batch_size is not None else step.batch_size
    workers = explicit_num_workers if explicit_num_workers is not None else step.num_workers

    nw = workers if workers is not None else recommend_num_workers()
    if batch is not None:
        # CLI 显式 / 交互询问 / LLM 规划已定 batch → 尊重（workers 缺省按 CPU 公式）
        return max(1, batch), nw

    # batch 未定：动态实测（模型已加载；无批量能力/无 CUDA 回退静态表）
    infer_fn = _build_tune_infer_fn(
        step, det_name, seg_name, model, seg_model, obb_model, cls_model, pose_model, sahi,
    )
    bs, _ = resolve_batch_params(
        step.task_type, infer_fn, [str(p) for p in images[:20]],
        explicit_workers=workers,
    )
    return bs, nw


def _execute_image(
    step: TaskStep,
    img_path: Path,
    det_name: str | None,
    seg_name: str | None,
    model: DetectionModel | None,
    seg_model: Any | None,
    sahi: bool,
    step_t0: float,
    _ts: str,
    steps_results: list[dict[str, Any]],
    det_results: list[Any] | None = None,
    det_quality: QualityReport | None = None,
    cls_labels: list[Any] | None = None,
    obb_results: list[Any] | None = None,
    pose_results: list[Any] | None = None,
    seg_masks: list[Any] | None = None,
    cls_model: Any = None,
    obb_model: Any = None,
    pose_model: Any = None,
) -> None:
    """执行单图：分类/OBB/姿态分支直接产出；其余走 检测 → 可选分割 → 后处理。

    批量路径经 det_results/det_quality/cls_labels/obb_results/pose_results/seg_masks
    注入预计算结果，对应模型调用在块级完成（None = 本图自行推理）。
    cls_model/obb_model/pose_model 步骤级复用（None 时各步自行创建，旧行为）。
    """
    ann = Annotation(image_path=str(img_path))
    try:
        from PIL import Image as _Image

        im = _Image.open(img_path)
        ann.image_size = (im.width, im.height)
    except Exception:
        pass

    # 分类任务：零样本分类 → labels → 导出（跳过检测/分割/可视化/三档分流）
    if step.task_type == "classification":
        _classification_step(
            step, img_path, ann, _ts, step_t0, steps_results,
            cls_labels=cls_labels, cls_model=cls_model,
        )
        return

    # 旋转框检测任务：YOLO-OBB → Bbox(angle) → dota/yolo_obb 导出（三档分流照旧）
    if step.task_type == "obb_detection":
        _obb_step(
            step, img_path, ann, _ts, step_t0, steps_results,
            obb_results=obb_results, obb_model=obb_model,
        )
        return

    # 姿态估计任务：YOLO-pose → Bbox(keypoints) → coco 导出（keypoints 内嵌）
    if step.task_type == "pose_estimation":
        _pose_step(
            step, img_path, ann, _ts, step_t0, steps_results,
            pose_results=pose_results, pose_model=pose_model,
        )
        return

    # 检测（自带检测的分割模型跳过）；批量路径直接注入块级结果
    if det_results is not None:
        results = det_results
        quality = det_quality
        # 块级推理跳过了单图重试分支的 bbox 填充，这里补齐
        for r in results:
            ann.add_bbox(Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                              label=r.label, confidence=r.confidence))
            if r.confidence < step.confidence_threshold:
                ann.flag_for_review(len(ann.bboxes) - 1)
    else:
        results, quality = _detection_step(step, img_path, ann, det_name, model, sahi)

    # 分割：自带检测的模型（sam3/maskrcnn/torchvision 语义分割）一步到位，其他需要 bbox prompt
    if "segmentation" in step.task_type:
        _segmentation_step(step, img_path, ann, seg_name, seg_model, seg_masks)

    # 展示/导出/可视化/HITL/日志
    _finalize_step(step, img_path, ann, results, quality, _ts, step_t0, steps_results)


def _execute_chunk(
    step: TaskStep,
    chunk: list[Path],
    det_name: str | None,
    seg_name: str | None,
    model: DetectionModel | None,
    seg_model: Any | None,
    sahi: bool,
    step_t0: float,
    _ts: str,
    steps_results: list[dict[str, Any]],
    obb_model: Any = None,
    cls_model: Any = None,
    pose_model: Any = None,
    num_workers: int = 0,
) -> None:
    """执行一个图像块：支持批量推理的任务走模型 *_batch 方法，否则逐图回退。

    batch_size=1 时 chunk 恒为单图，自然落入逐图路径（与旧行为一致）。
    obb_model/cls_model/pose_model 步骤级复用（execute_plan 已创建）；None 时块内创建。
    """
    num_workers = max(0, num_workers)
    batch = len(chunk) > 1

    # 分类批量（CLIP/SigLIP/torchvision 原生多图）
    if step.task_type == "classification" and batch:
        labels_per_img = _classification_batch(step, chunk, cls_model)
        if labels_per_img is not None:
            for img_path, labels in zip(chunk, labels_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, cls_labels=labels,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                )
            return

    # OBB 批量（ultralytics 原生 batch）
    if step.task_type == "obb_detection" and batch:
        obb_per_img = _obb_batch(step, chunk, num_workers, obb_model)
        if obb_per_img is not None:
            for img_path, obb_results in zip(chunk, obb_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, obb_results=obb_results,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                )
            return

    # 姿态批量（ultralytics 原生 batch）
    if step.task_type == "pose_estimation" and batch:
        pose_per_img = _pose_batch(step, chunk, num_workers, pose_model)
        if pose_per_img is not None:
            for img_path, pose_results in zip(chunk, pose_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, pose_results=pose_results,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                )
            return

    # 检测批量（SAHI 保持逐图——切片推理按单图编排）
    _use_sahi = sahi or getattr(step, "sahi", False)
    if det_name and model is not None and batch and not _use_sahi:
        det_map = _detection_batch(step, chunk, model, num_workers)
        if det_map is not None:
            for img_path in chunk:
                det_results, det_quality = det_map[img_path]
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results,
                    det_results=det_results, det_quality=det_quality,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                )
            return

    # 自带检测的分割批量（torchvision maskrcnn 原生 list-of-tensors；SAM 系列逐图）
    if seg_name and _is_self_detect_seg(seg_name) and seg_model is not None and batch:
        masks_per_img = _segmentation_batch(seg_model, chunk, step)
        if masks_per_img is not None:
            for img_path, masks in zip(chunk, masks_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, seg_masks=masks,
                    obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
                )
            return

    # 回退：逐图执行（batch_size=1 / 模型不支持 batch / SAHI）
    for img_path in chunk:
        _execute_image(
            step, img_path, det_name, seg_name, model, seg_model, sahi,
            step_t0, _ts, steps_results,
            obb_model=obb_model, cls_model=cls_model, pose_model=pose_model,
        )


def _detection_batch(
    step: TaskStep,
    chunk: list[Path],
    model: Any,
    num_workers: int,
) -> dict[Path, tuple[list[Any], Any]] | None:
    """块内检测批量推理 + 逐图代码级评估；模型无 detect_batch 能力返回 None（逐图回退）。"""
    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.detection import detect_with_retry

    detect_batch = cast(
        "Callable[..., list[list[Any]]] | None", getattr(model, "detect_batch", None)
    )
    if detect_batch is None:
        return None
    results_per_img = detect_batch(
        [str(p) for p in chunk], step.prompts, step.confidence_threshold, num_workers,
    )
    out: dict[Path, tuple[list[Any], Any]] = {}
    for img_path, results in zip(chunk, results_per_img):
        retried, retry_threshold = False, None
        if not results:
            # 0 框 → 单图降阈值重试一次（与逐图路径行为一致）
            results, retried, retry_threshold = detect_with_retry(
                lambda conf: model.detect(str(img_path), step.prompts, conf),
                step.confidence_threshold,
            )
        quality = evaluate_detections(
            results, step.prompts, step.confidence_threshold, str(img_path),
            retried=retried, retry_threshold=retry_threshold,
        )
        out[img_path] = (results, quality)
    return out


def _classification_batch(
    step: TaskStep, chunk: list[Path], cls_model: Any = None,
) -> list[list[Any]] | None:
    """块内分类批量推理；模型无 classify_batch 返回 None（逐图回退）。

    cls_model 步骤级复用（execute_plan 已创建）；None 时块内创建（向后兼容）。
    """
    if cls_model is None:
        from auto2dlabel.models.classification import create_classification_model

        cls_model = create_classification_model(step.model_name)
    classify_batch = cast(
        "Callable[..., list[list[Any]]] | None",
        getattr(cls_model, "classify_batch", None),
    )
    if classify_batch is None:
        return None
    return classify_batch([str(p) for p in chunk], step.prompts, top_k=5)


def _obb_batch(
    step: TaskStep, chunk: list[Path], num_workers: int, obb_model: Any = None,
) -> list[list[Any]] | None:
    """块内旋转框批量推理；模型无 detect_obb_batch 返回 None（逐图回退）。

    obb_model 步骤级复用（execute_plan 已创建）；None 时块内创建（向后兼容）。
    """
    if obb_model is None:
        from auto2dlabel.models.obb import create_obb_model

        obb_model = create_obb_model(step.model_name, iou_threshold=step.iou_threshold)
    detect_obb_batch = cast(
        "Callable[..., list[list[Any]]] | None",
        getattr(obb_model, "detect_obb_batch", None),
    )
    if detect_obb_batch is None:
        return None
    return detect_obb_batch(
        [str(p) for p in chunk], step.prompts, step.confidence_threshold, num_workers,
    )


def _pose_batch(
    step: TaskStep, chunk: list[Path], num_workers: int, pose_model: Any = None,
) -> list[list[Any]] | None:
    """块内姿态批量推理；模型无 detect_pose_batch 返回 None（逐图回退）。

    pose_model 步骤级复用（execute_plan 已创建）；None 时块内创建（向后兼容）。
    """
    if pose_model is None:
        from auto2dlabel.models.pose import create_pose_model

        pose_model = create_pose_model(step.model_name, iou_threshold=step.iou_threshold)
    detect_pose_batch = cast(
        "Callable[..., list[list[Any]]] | None",
        getattr(pose_model, "detect_pose_batch", None),
    )
    if detect_pose_batch is None:
        return None
    return detect_pose_batch(
        [str(p) for p in chunk], step.prompts, step.confidence_threshold, num_workers,
    )


def _segmentation_batch(
    seg_model: Any, chunk: list[Path], step: TaskStep
) -> list[list[Any]] | None:
    """块内自带检测分割批量推理；模型无 generate_batch 返回 None（SAM 系列逐图回退）。"""
    generate_batch = cast(
        "Callable[..., list[list[Any]]] | None",
        getattr(seg_model, "generate_batch", None),
    )
    if generate_batch is None:
        return None
    prompt_bboxes = [
        Bbox(x=0, y=0, width=1, height=1, label=p, confidence=1.0)
        for p in step.prompts
    ]
    return generate_batch([str(p) for p in chunk], prompt_bboxes)


def _classification_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    _ts: str,
    step_t0: float,
    steps_results: list[dict[str, Any]],
    cls_labels: list[Any] | None = None,
    cls_model: ClassificationModel | None = None,
) -> None:
    """分类任务：零样本分类 → labels → 导出（cls 为目录型格式：每图写 outputs/{stem}.json）。

    cls_labels 非 None 时跳过推理（批量路径块级预计算结果）。
    cls_model 步骤级复用（execute_plan 已创建）；None 时每图创建（旧行为）。
    """
    if cls_labels is None:
        if cls_model is None:
            from auto2dlabel.models.classification import create_classification_model

            cls_model = create_classification_model(step.model_name)
        cls_labels = cls_model.classify(str(img_path), step.prompts, top_k=5)
    ann.labels = cls_labels
    ann.metadata["model"] = step.model_name
    console.print(f"[dim]分类完成: {len(ann.labels)} 个候选标签[/dim]")

    state = AgentState(image_path=str(img_path))
    state.annotations = [ann]
    display_results(state)

    from auto2dlabel.tools.export import ExportTool

    export_tool = ExportTool()
    # cls 为目录型格式：每图写 outputs/{stem}.json（forward 返回的是 outputs 目录，故不接返回值）
    export_tool.forward(
        annotations=[ann.to_dict()],
        output_path="outputs",
        format="cls",
    )
    console.print(f"[green]✓ 导出: outputs/{img_path.stem}.json[/green]")

    from auto2dlabel.tools.log import log_python_api_call

    log_python_api_call(
        image_path=str(img_path),
        prompts=step.prompts,
        results=[lab.to_dict() for lab in ann.labels],
        elapsed=round(_time.time() - step_t0, 3),
        model_name=step.model_name,
        confidence_threshold=step.confidence_threshold,
        iou_threshold=step.iou_threshold,
        annotation_type="classification",
        timestamp=_ts,
    )
    steps_results.append({
        "step_id": step.step_id,
        "source": step.source,
        "model": step.model_name,
        "prompts": step.prompts,
        "labels": [{"label": lab.label, "score": lab.score} for lab in ann.labels],
        "label_count": len(ann.labels),
    })


def _obb_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    _ts: str,
    step_t0: float,
    steps_results: list[dict[str, Any]],
    obb_results: list[Any] | None = None,
    obb_model: OBBModel | None = None,
) -> None:
    """旋转框检测：YOLO-OBB → Bbox(angle) → dota/yolo_obb 导出（三档分流照旧）。

    obb_results 非 None 时跳过推理（批量路径块级预计算结果）。
    obb_model 步骤级复用（execute_plan 已创建）；None 时每图创建（旧行为）。
    """
    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.obb import _match_obb_prompt

    if obb_results is None:
        if obb_model is None:
            from auto2dlabel.models.obb import create_obb_model

            obb_model = create_obb_model(step.model_name, iou_threshold=step.iou_threshold)
        obb_results = obb_model.detect_obb(
            str(img_path), step.prompts,
            confidence_threshold=step.confidence_threshold,
        )
    for r in obb_results:
        ann.add_bbox(Bbox(
            x=r.cx - r.width / 2, y=r.cy - r.height / 2,
            width=r.width, height=r.height,
            label=r.label, confidence=r.confidence, angle=r.angle,
        ))
        if r.confidence < step.confidence_threshold:
            ann.flag_for_review(len(ann.bboxes) - 1)
    console.print(f"[dim]旋转框检测: {len(obb_results)} 个目标[/dim]")

    quality = evaluate_detections(
        obb_results, step.prompts, step.confidence_threshold, str(img_path),
        prompt_matcher=_match_obb_prompt,
    )
    if quality.warnings:
        console.print(f"[yellow]质量警告: {'; '.join(quality.warnings)}[/yellow]")

    state = AgentState(image_path=str(img_path))
    state.annotations = [ann]
    if quality:
        state.metadata["quality_report"] = quality.to_dict()
    display_results(state)

    from auto2dlabel.tools.export import ExportTool

    export_tool = ExportTool()
    fmt = step.export_format.replace("-", "_")  # yolo-obb → yolo_obb
    out_json = export_tool.forward(
        annotations=[ann.to_dict()],
        output_path="outputs",
        format=fmt,
    )
    console.print(f"[green]✓ 导出: {out_json}[/green]")

    from auto2dlabel.tools.visualize import visualize_annotation

    vis_dir = Path("vis_outputs")
    vis_dir.mkdir(exist_ok=True)
    vis_path = vis_dir / f"vis_{img_path.stem}_{_ts}.png"
    visualize_annotation(img_path, ann, vis_path, draw_mask=False)
    console.print(f"[green]✓ 可视化: {vis_path}[/green]")

    triage_and_export(state, img_path, step.confidence_threshold, _ts)

    from auto2dlabel.tools.log import log_python_api_call

    log_python_api_call(
        image_path=str(img_path),
        prompts=step.prompts,
        results=[{
            "label": r.label, "conf": r.confidence,
            "cx": r.cx, "cy": r.cy,
            "width": r.width, "height": r.height, "angle": r.angle,
        } for r in obb_results],
        elapsed=round(_time.time() - step_t0, 3),
        model_name=step.model_name,
        confidence_threshold=step.confidence_threshold,
        iou_threshold=step.iou_threshold,
        annotation_type="obb_detection",
        timestamp=_ts,
        quality=quality.to_dict(),
    )
    steps_results.append({
        "step_id": step.step_id,
        "source": step.source,
        "model": step.model_name,
        "prompts": step.prompts,
        "bbox_count": len(ann.bboxes),
        "obb_count": len(obb_results),
        "quality": quality.to_dict(),
    })


def _pose_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    _ts: str,
    step_t0: float,
    steps_results: list[dict[str, Any]],
    pose_results: list[Any] | None = None,
    pose_model: PoseModel | None = None,
) -> None:
    """姿态估计：YOLO-pose → Bbox(keypoints) → coco 导出（keypoints 内嵌，三档分流照旧）。

    pose_results 非 None 时跳过推理（批量路径块级预计算结果）。
    pose_model 步骤级复用（execute_plan 已创建）；None 时每图创建（旧行为）。
    """
    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.detection import _match_prompt

    if pose_results is None:
        if pose_model is None:
            from auto2dlabel.models.pose import create_pose_model

            pose_model = create_pose_model(step.model_name, iou_threshold=step.iou_threshold)
        pose_results = pose_model.detect_pose(
            str(img_path), step.prompts,
            confidence_threshold=step.confidence_threshold,
        )
    for r in pose_results:
        ann.add_bbox(Bbox(
            x=r.x, y=r.y, width=r.width, height=r.height,
            label=r.label, confidence=r.confidence, keypoints=r.keypoints,
        ))
        if r.confidence < step.confidence_threshold:
            ann.flag_for_review(len(ann.bboxes) - 1)
    console.print(f"[dim]姿态估计: {len(pose_results)} 个人体（含 keypoints）[/dim]")

    quality = evaluate_detections(
        pose_results, step.prompts, step.confidence_threshold, str(img_path),
        prompt_matcher=_match_prompt,
    )
    if quality.warnings:
        console.print(f"[yellow]质量警告: {'; '.join(quality.warnings)}[/yellow]")

    state = AgentState(image_path=str(img_path))
    state.annotations = [ann]
    if quality:
        state.metadata["quality_report"] = quality.to_dict()
    display_results(state)

    # keypoints 语义只在 COCO 有意义 → 恒 coco 导出（防 planner 误给其他格式丢字段）
    from auto2dlabel.tools.export import ExportTool

    export_tool = ExportTool()
    out_json = export_tool.forward(
        annotations=[ann.to_dict()],
        output_path=f"outputs/{img_path.stem}_{_ts}.json",
        format="coco",
    )
    console.print(f"[green]✓ 导出: {out_json}[/green]")

    from auto2dlabel.tools.visualize import visualize_annotation

    vis_dir = Path("vis_outputs")
    vis_dir.mkdir(exist_ok=True)
    vis_path = vis_dir / f"vis_{img_path.stem}_{_ts}.png"
    # visualize_annotation 已内置 draw_keypoints（bboxes 含 keypoints 自动绘骨架）
    visualize_annotation(img_path, ann, vis_path, draw_mask=False)
    console.print(f"[green]✓ 可视化: {vis_path}[/green]")

    triage_and_export(state, img_path, step.confidence_threshold, _ts)

    from auto2dlabel.tools.log import log_python_api_call

    log_python_api_call(
        image_path=str(img_path),
        prompts=step.prompts,
        results=[{
            "label": r.label, "conf": r.confidence,
            "bbox": [r.x, r.y, r.width, r.height],
            "num_keypoints": sum(1 for k in r.keypoints if k[2] > 0),
        } for r in pose_results],
        elapsed=round(_time.time() - step_t0, 3),
        model_name=step.model_name,
        confidence_threshold=step.confidence_threshold,
        iou_threshold=step.iou_threshold,
        annotation_type="pose_estimation",
        timestamp=_ts,
        quality=quality.to_dict(),
    )
    steps_results.append({
        "step_id": step.step_id,
        "source": step.source,
        "model": step.model_name,
        "prompts": step.prompts,
        "bbox_count": len(ann.bboxes),
        "keypoint_count": sum(b.num_keypoints for b in ann.bboxes),
        "quality": quality.to_dict(),
    })


def _detection_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    det_name: str | None,
    model: DetectionModel | None,
    sahi: bool,
) -> tuple[list[Any], QualityReport | None]:
    """检测（自带检测的分割模型跳过）。返回 (results, quality)。"""
    if not det_name:
        return [], None

    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.detection import DetectionResult, detect_image_sahi, detect_with_retry

    assert model is not None  # det_name 非空时已创建
    det_model = model  # 局部别名：lambda 闭包内 pyright 不保留 assert 收窄
    # SAHI 切片推理（来自 CLI --sahi 或 NL 中的 sahi 参数）
    _use_sahi = sahi or getattr(step, "sahi", False)
    results: list[Any] = []
    if _use_sahi:
        console.print("[dim]SAHI 切片推理...[/dim]")
        dets, retried, retry_threshold = detect_with_retry(
            lambda conf: detect_image_sahi(
                det_model, str(img_path), step.prompts,
                confidence_threshold=conf,
            ),
            step.confidence_threshold,
        )
        results = [
            DetectionResult(
                x=d["bbox"][0], y=d["bbox"][1],
                width=d["bbox"][2] - d["bbox"][0],
                height=d["bbox"][3] - d["bbox"][1],
                label=d["name"], confidence=d["conf"],
            )
            for d in dets
        ]
    else:
        results, retried, retry_threshold = detect_with_retry(
            lambda conf: det_model.detect(str(img_path), step.prompts, conf),
            step.confidence_threshold,
        )
    for r in results:
        ann.add_bbox(Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                          label=r.label, confidence=r.confidence))
        if r.confidence < step.confidence_threshold:
            ann.flag_for_review(len(ann.bboxes) - 1)

    # 代码级质量评估（与 Agent Loop / no-LLM baseline 共用判据）
    quality = evaluate_detections(
        results, step.prompts, step.confidence_threshold, str(img_path),
        retried=retried, retry_threshold=retry_threshold,
    )
    if quality.warnings:
        console.print(f"[yellow]质量警告: {'; '.join(quality.warnings)}[/yellow]")

    return results, quality


def _segmentation_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    seg_name: str | None,
    seg_model: Any | None = None,
    seg_masks: list[Any] | None = None,
) -> None:
    """分割：自带检测的模型一步到位（用 prompt 作文本提示），其他需要 bbox prompt。

    seg_model 由 execute_plan 步骤级创建复用（原每图重建）；seg_masks 非 None 时
    跳过推理（批量路径块级预计算结果）。
    """
    seg_model_name = seg_name or "sam2_l.pt"
    console.print(f"[dim]正在生成分割 mask（{seg_model_name}）...[/dim]")
    if seg_model is None:
        from auto2dlabel.models.segmentation import create_segmentation_model

        seg_model = create_segmentation_model(seg_model_name)

    if _is_self_detect_seg(seg_model_name):
        # 这些模型自带检测——用用户 prompt 作为文本提示
        if seg_masks is not None:
            masks = seg_masks
        else:
            prompt_bboxes = [
                Bbox(x=0, y=0, width=1, height=1, label=p, confidence=1.0)
                for p in step.prompts
            ]
            if not prompt_bboxes:
                # SAM3 等文本驱动模型无类别无法检测（曾静默零输出，2026-08-29）
                console.print(
                    "[yellow]未指定检测类别（prompts 为空）——SAM3 文本驱动分割"
                    "无法运行，分割跳过（检测结果照常导出）。重新执行时请指定类别，"
                    "如「检测 KITTI 中的汽车、行人」[/yellow]"
                )
                return
            masks = seg_model.generate(str(img_path), prompt_bboxes)
        if masks:
            ann.bboxes = []  # 用分割模型的检测结果替换
            ann.masks = []
            for m in masks:
                ann.add_bbox(m.bbox)
                ann.add_mask(m)
            console.print(
                f"[dim]已生成 {len(masks)} 个检测+mask"
                f"（{seg_model_name} 一步完成）[/dim]"
            )
    else:
        high_conf_bboxes = [b for b in ann.bboxes if b.confidence >= 0.5]
        if high_conf_bboxes:
            masks = seg_model.generate(str(img_path), high_conf_bboxes)
            for m in masks:
                ann.add_mask(m)
            console.print(f"[dim]已生成 {len(masks)} 个 mask[/dim]")


def _finalize_step(
    step: TaskStep,
    img_path: Path,
    ann: Annotation,
    results: list[Any],
    quality: QualityReport | None,
    _ts: str,
    step_t0: float,
    steps_results: list[dict[str, Any]],
) -> None:
    """展示/导出/可视化/HITL 分流/日志。"""
    from auto2dlabel.tools.export import ExportTool
    from auto2dlabel.tools.hitl import triage_annotations
    from auto2dlabel.tools.log import log_python_api_call
    from auto2dlabel.tools.visualize import visualize_annotation

    elapsed = round(_time.time() - step_t0, 3)

    # 显示结果
    state = AgentState(image_path=str(img_path))
    state.annotations = [ann]
    if quality:
        state.metadata["quality_report"] = quality.to_dict()
    display_results(state)

    # 导出
    export_tool = ExportTool()
    out_json = export_tool.forward(
        annotations=[ann.to_dict()],
        output_path=f"outputs/{img_path.stem}_{_ts}.json",
        format=step.export_format,
    )
    console.print(f"[green]✓ 导出: {out_json}[/green]")

    # 可视化
    vis_dir = Path("vis_outputs")
    vis_dir.mkdir(exist_ok=True)
    vis_path = vis_dir / f"vis_{img_path.stem}_{_ts}.png"
    visualize_annotation(img_path, ann, vis_path, draw_mask=bool(ann.masks))
    console.print(f"[green]✓ 可视化: {vis_path}[/green]")

    # HITL 分流
    triage_and_export(state, img_path, step.confidence_threshold, _ts)

    # 日志
    log_python_api_call(
        image_path=str(img_path),
        prompts=step.prompts,
        results=results,
        elapsed=elapsed,
        model_name=step.model_name,
        confidence_threshold=step.confidence_threshold,
        iou_threshold=step.iou_threshold,
        timestamp=_ts,
        quality=quality.to_dict() if quality else None,
    )

    # 收集步骤结果用于顶层 chat 日志
    triage = triage_annotations(
        [ann], tau_high=0.7, tau_low=max(0.1, step.confidence_threshold * 0.5),
    )
    steps_results.append({
        "step_id": step.step_id,
        "source": step.source,
        "model": step.model_name,
        "prompts": step.prompts,
        "bbox_count": len(ann.bboxes),
        "triage_summary": triage.summary,
        "quality": quality.to_dict() if quality else None,
    })


def _log_plan(plan: TaskPlan, steps_results: list[dict[str, Any]], plan_t0: float) -> None:
    """记录顶层 Chat 日志。"""
    from auto2dlabel.tools.log import log_chat_call

    plan_summary = {
        "total_steps": len(plan.steps),
        "steps": [s.to_dict() for s in plan.steps],
    }
    log_chat_call(
        instruction=plan.raw_instruction,
        plan_summary=plan_summary,
        steps_results=steps_results,
        elapsed=round(_time.time() - plan_t0, 3),
        timestamp=_datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
