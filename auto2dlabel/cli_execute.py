"""TaskPlan 执行器 —— chat 命令 Step 5 的实现（检测/分割/分类/OBB 四类任务）。"""

from __future__ import annotations

import time as _time
from datetime import datetime as _datetime
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from auto2dlabel.agent.state import AgentState
from auto2dlabel.cli_common import collect_images, console, display_results, triage_and_export
from auto2dlabel.models.model_catalog import SEGMENTATION_MODELS, TORCHVISION_SEG_MODELS
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import DEFAULT_MODEL, TaskPlan, TaskStep

if TYPE_CHECKING:
    from auto2dlabel.agent.evaluate import QualityReport
    from auto2dlabel.models.detection import DetectionModel


def execute_plan(plan: TaskPlan, sahi: bool = False) -> None:
    """顺序执行 TaskPlan 的每个步骤。"""
    _plan_t0 = _time.time()
    steps_results: list[dict[str, Any]] = []

    for step in plan.steps:
        _print_step_header(step)

        _t0 = _time.time()
        _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        source_path = Path(step.source)

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

        # 批量推理：按 batch_size 分块（1 = 逐图，与旧行为一致）
        batch_size = max(1, step.batch_size or 1)
        console.print(f"  批量: batch_size={batch_size}  num_workers={step.num_workers or 0}")
        for i in range(0, len(images), batch_size):
            _execute_chunk(
                step, images[i:i + batch_size], det_name, seg_name, model, seg_model,
                sahi, _t0, _ts, steps_results,
            )

    _log_plan(plan, steps_results, _plan_t0)


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

    # 分类/旋转框任务不创建普通检测模型
    if step.task_type in ("classification", "obb_detection"):
        det_name = None

    return det_name, seg_name


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
    seg_masks: list[Any] | None = None,
) -> None:
    """执行单图：分类/OBB 分支直接产出；其余走 检测 → 可选分割 → 后处理。

    批量路径经 det_results/det_quality/cls_labels/obb_results/seg_masks 注入预计算结果，
    对应模型调用在块级完成（None = 本图自行推理）。
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
        _classification_step(step, img_path, ann, _ts, step_t0, steps_results, cls_labels)
        return

    # 旋转框检测任务：YOLO-OBB → Bbox(angle) → dota/yolo_obb 导出（三档分流照旧）
    if step.task_type == "obb_detection":
        _obb_step(step, img_path, ann, _ts, step_t0, steps_results, obb_results)
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
) -> None:
    """执行一个图像块：支持批量推理的任务走模型 *_batch 方法，否则逐图回退。

    batch_size=1 时 chunk 恒为单图，自然落入逐图路径（与旧行为一致）。
    """
    num_workers = max(0, step.num_workers or 0)
    batch = len(chunk) > 1

    # 分类批量（CLIP/SigLIP/torchvision 原生多图）
    if step.task_type == "classification" and batch:
        labels_per_img = _classification_batch(step, chunk)
        if labels_per_img is not None:
            for img_path, labels in zip(chunk, labels_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, cls_labels=labels,
                )
            return

    # OBB 批量（ultralytics 原生 batch）
    if step.task_type == "obb_detection" and batch:
        obb_per_img = _obb_batch(step, chunk, num_workers)
        if obb_per_img is not None:
            for img_path, obb_results in zip(chunk, obb_per_img):
                _execute_image(
                    step, img_path, det_name, seg_name, model, seg_model, sahi,
                    step_t0, _ts, steps_results, obb_results=obb_results,
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
                )
            return

    # 回退：逐图执行（batch_size=1 / 模型不支持 batch / SAHI）
    for img_path in chunk:
        _execute_image(
            step, img_path, det_name, seg_name, model, seg_model, sahi,
            step_t0, _ts, steps_results,
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


def _classification_batch(step: TaskStep, chunk: list[Path]) -> list[list[Any]] | None:
    """块内分类批量推理；模型无 classify_batch 返回 None（逐图回退）。"""
    from auto2dlabel.models.classification import create_classification_model

    cls_model = create_classification_model(step.model_name)
    classify_batch = cast(
        "Callable[..., list[list[Any]]] | None",
        getattr(cls_model, "classify_batch", None),
    )
    if classify_batch is None:
        return None
    return classify_batch([str(p) for p in chunk], step.prompts, top_k=5)


def _obb_batch(step: TaskStep, chunk: list[Path], num_workers: int) -> list[list[Any]] | None:
    """块内旋转框批量推理；模型无 detect_obb_batch 返回 None（逐图回退）。"""
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
) -> None:
    """分类任务：零样本分类 → labels → 导出（cls 为目录型格式：每图写 outputs/{stem}.json）。

    cls_labels 非 None 时跳过推理（批量路径块级预计算结果）。
    """
    if cls_labels is None:
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
) -> None:
    """旋转框检测：YOLO-OBB → Bbox(angle) → dota/yolo_obb 导出（三档分流照旧）。

    obb_results 非 None 时跳过推理（批量路径块级预计算结果）。
    """
    from auto2dlabel.agent.evaluate import evaluate_detections
    from auto2dlabel.models.obb import _match_obb_prompt

    if obb_results is None:
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
