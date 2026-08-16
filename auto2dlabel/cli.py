"""CLI 入口 —— 命令行标注工具。

Usage:
    auto2dlabel run image.jpg "detect all cars and pedestrians"
    auto2dlabel run image.jpg "detect all cars" --threshold 0.5 --export coco
    auto2dlabel run ./images/ "detect cars, bikes, people" --batch --export yolo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from auto2dlabel.agent.llm import create_client
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.agent.planner import TaskPlanner
from auto2dlabel.agent.state import AgentState
from auto2dlabel.schema.task_plan import DEFAULT_MODEL, TaskPlan

app = typer.Typer(
    name="auto2dlabel",
    help="Agentic 2D image annotation — AI-first, human-in-the-loop.",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


@app.command()
def run(
    image: str = typer.Argument("", help="图像路径或目录（支持 JPG/PNG/TIFF）；--resume 时忽略"),
    instruction: str = typer.Argument(
        "", help="标注指令，用自然语言描述要标注什么；--resume 时从清单读取"
    ),
    threshold: float = typer.Option(0.1, "--threshold", "-t", help="置信度阈值 (0-1)"),
    iou: float = typer.Option(0.3, "--iou", help="IoU 阈值 (0-1)，NMS 去重力度"),
    export: str = typer.Option(
        "coco", "--export", "-e", help="导出格式: coco, yolo, voc, labelme"
    ),
    output: str = typer.Option("outputs", "--output", "-o", help="输出目录"),
    batch: bool = typer.Option(False, "--batch", "-b", help="如果是目录, 批量处理所有图像"),
    resume: str = typer.Option(
        "", "--resume", help="从批次清单续跑（跳过 ok，重跑 failed/pending）"
    ),
    provider: str = typer.Option(
        "deepseek", "--provider", "-p", help="LLM provider: openai, anthropic, deepseek"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM 模型名"),
    det_model: str | None = typer.Option(
        None,
        "--det-model",
        "-d",
        help="检测模型: grounding-dino-tiny | yolov8n.pt | fasterrcnn_resnet50_fpn | ...",
    ),
    api_key: str = typer.Option(None, "--api-key", help="LLM API key"),
    base_url: str = typer.Option(None, "--base-url", help="LLM API base URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志输出"),
    sahi: bool = typer.Option(False, "--sahi", help="启用 SAHI 切片推理（大分辨率图像）"),
) -> None:
    """对图像运行 Agentic 标注。"""
    _setup_logging(verbose)

    # ---- 续跑模式：清单为单一事实源（图像列表/指令/参数均从清单读取） ----
    if resume:
        from auto2dlabel.tools.batch import load_manifest, resume_targets

        manifest = load_manifest(resume)
        instruction = manifest.instruction
        cfg = manifest.config
        threshold = float(cfg.get("threshold", threshold))
        iou = float(cfg.get("iou", iou))
        export = str(cfg.get("export", export))
        output = str(cfg.get("output", output))
        det_model = cfg.get("det_model", det_model) or None
        provider = str(cfg.get("provider", provider))
        model = cfg.get("model", model) or None
        sahi = bool(cfg.get("sahi", sahi))
        image_files = [Path(e.path) for e in resume_targets(manifest) if Path(e.path).exists()]
        if not image_files:
            console.print("[green]清单中无待重跑图像（全部 ok 或文件已不存在）[/green]")
            return
        console.print(f"[dim]续跑模式: 清单 {resume} → 重跑 {len(image_files)} 张[/dim]")
        manifest_path = resume
    else:
        if not image:
            console.print("[red]错误: 请提供图像路径或目录[/red]")
            raise typer.Exit(code=1)
        if not instruction:
            console.print("[red]错误: 请提供标注指令[/red]")
            raise typer.Exit(code=1)
        image_path = Path(image)
        if not image_path.exists():
            console.print(f"[red]错误: 路径不存在: {image}[/red]")
            raise typer.Exit(code=1)

        # 确认图像文件列表
        image_files = _collect_images(image_path, batch)
        if not image_files:
            console.print(f"[red]错误: 未找到图像文件: {image}[/red]")
            raise typer.Exit(code=1)

        # 新建批次清单（成功/失败逐图更新，中断后可 --resume 续跑）
        from auto2dlabel.tools.batch import new_manifest

        manifest = new_manifest(
            instruction,
            config={
                "threshold": threshold, "iou": iou, "export": export, "output": output,
                "det_model": det_model, "provider": provider, "model": model, "sahi": sahi,
            },
            image_paths=[str(p.resolve()) for p in image_files],
        )
        manifest_path = f"{output}/batch_manifest.json"
        from auto2dlabel.tools.batch import save_manifest

        save_manifest(manifest, manifest_path)

    from auto2dlabel.tools.batch import STATUS_FAILED, STATUS_OK, save_manifest, update_entry

    # 创建 LLM 客户端
    llm = create_client(provider=provider, model=model or None, api_key=api_key, base_url=base_url)
    # 解析检测模型名（显示与实跑一致：orchestrator 收到 None 后工厂兜底即 DEFAULT_MODEL）
    det_name = det_model or DEFAULT_MODEL
    from auto2dlabel.tools.device import get_device_info
    console.print(f"[dim]设备: {get_device_info()}[/dim]")
    console.print(f"[dim]LLM: {llm.model} ({provider})[/dim]")
    console.print(f"[dim]检测模型: {det_name}[/dim]")
    console.print(f"[dim]置信度: {threshold}  IoU: {iou}  SAHI: {sahi}[/dim]")
    console.print(f"[dim]导出格式: {export}[/dim]")

    # 创建编排器并运行
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        detection_model=det_model,
        iou_threshold=iou,
        use_sahi=sahi,
    )

    failed_count = 0
    for img_path in image_files:
        console.print(f"\n[bold]━━━ 标注: {img_path.name} ━━━[/bold]")

        import time as _time
        from datetime import datetime as _datetime

        _t0 = _time.time()
        _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # 统一时间戳
        error_msg = None

        try:
            state = orchestrator.run(
                image_path=str(img_path),
                instruction=instruction,
                confidence_threshold=threshold,
            )
        except Exception as e:
            error_msg = str(e)
            from auto2dlabel.tools.log import log_llm_call

            log_llm_call(
                image_path=str(img_path),
                instruction=instruction,
                state_summary={"error": error_msg},
                elapsed=_time.time() - _t0,
                llm_model=llm.model,
                detection_model=det_model or "default",
                confidence_threshold=threshold,
                iou_threshold=iou,
                timestamp=_ts,
                error=error_msg,
            )
            # 失败隔离：记录清单后继续下一图（不中断整批）
            update_entry(
                manifest, str(img_path.resolve()),
                status=STATUS_FAILED, error=error_msg, elapsed=_time.time() - _t0,
            )
            save_manifest(manifest, manifest_path)
            failed_count += 1
            console.print(f"[red]✗ 失败: {img_path.name} — {error_msg}[/red]")
            continue

        # 写日志
        from auto2dlabel.tools.log import log_llm_call

        log_llm_call(
            image_path=str(img_path),
            instruction=instruction,
            state_summary={
                "iterations": state.iteration,
                "annotations": len(state.annotations),
                "bbox_count": sum(len(a.bboxes) for a in state.annotations),
            },
            elapsed=_time.time() - _t0,
            llm_model=llm.model,
            detection_model=det_model or "default",
            confidence_threshold=threshold,
            iou_threshold=iou,
            timestamp=_ts,
        )

        # 显示结果
        _display_results(state)

        # 导出 JSON
        if state.annotations:
            ann_dicts = [a.to_dict() for a in state.annotations]
            from auto2dlabel.tools.export import ExportTool

            export_tool = ExportTool()
            out_path = export_tool.forward(
                annotations=ann_dicts,
                output_path=f"{output}/{img_path.stem}_{_ts}.json",
                format=export,
            )
            console.print(f"[green]✓ 已导出: {out_path}[/green]")

        # 可视化输出到 vis_outputs/
        _visualize_results(state, img_path, timestamp=_ts)

        # HITL 置信度分流
        _triage_and_export(state, img_path, threshold, _ts)

        # 写 AgentState 快照 + 更新清单（快照仅供调试/审计，from_dict 恢复未实现）
        state_file = f"{output}/{img_path.stem}_{_ts}_state.json"
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(state_file).write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
        bbox_count = sum(len(a.bboxes) for a in state.annotations)
        update_entry(
            manifest, str(img_path.resolve()),
            status=STATUS_OK, elapsed=_time.time() - _t0,
            bbox_count=bbox_count, state_file=state_file,
        )
        save_manifest(manifest, manifest_path)

    # 批次结束汇总（多图或续跑才打印）
    if len(image_files) > 1 or resume:
        if failed_count:
            console.print(
                f"\n[red]批次完成: {len(image_files) - failed_count} 成功, "
                f"{failed_count} 失败[/red]"
            )
            console.print(f"[dim]重跑失败项: auto2dlabel run --resume {manifest_path}[/dim]")
        else:
            console.print(f"\n[green]✓ 批次完成: 全部 {len(image_files)} 张成功[/green]")


@app.command()
def sample(
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回前 K 个最不确定的样本"),
    review_dir: str = typer.Option("outputs", "--review-dir", help="复核队列目录"),
    output: str = typer.Option(
        "outputs/sampling_manifest.json", "--output", "-o", help="采样清单输出路径"
    ),
) -> None:
    """主动学习采样：从 HITL 复核队列挑选信息量最高的样本。"""
    from auto2dlabel.tools.sampling import (
        collect_review_pool,
        sample_priority,
        write_priority_manifest,
    )

    pool = collect_review_pool(review_dir)
    if not pool:
        console.print(f"[yellow]复核队列为空（{review_dir}）[/yellow]")
        return
    selected = sample_priority(pool, top_k=top_k)
    out = write_priority_manifest(selected, output)
    console.print(f"[green]✓ 采样清单: {out}[/green]（{len(selected)}/{len(pool)} 张）")
    for i, item in enumerate(selected, 1):
        console.print(
            f"  {i}. {item.file}  score={item.score:.3f}  "
            f"unc={item.mean_uncertainty:.3f}  classes={item.class_count}"
        )


@app.command()
def tools() -> None:
    """列出所有可用的标注 Tool。"""
    from auto2dlabel.tools.detection import register as reg_detect
    from auto2dlabel.tools.export import register as reg_export

    reg_detect()
    reg_export()

    # Lazy import to avoid requiring SAM2 for CLI help
    try:
        from auto2dlabel.tools.segmentation import register as reg_seg

        reg_seg()
    except ImportError:
        pass

    from auto2dlabel.tools.registry import registry

    table = Table(title="Registered Annotation Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    for name in registry.list_tools():
        tool = registry.get(name)
        desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
        table.add_row(name, desc)

    console.print(table)


@app.command()
def chat(
    instruction: str = typer.Argument(None, help="自然语言标注指令（省略则进入交互模式）"),
    det_model: str = typer.Option(None, "--det-model", "-d", help="覆盖检测模型（如 rtdetr-l.pt）"),
    confirm_timeout: int = typer.Option(30, "--timeout", help="确认等待秒数（0 = 跳过确认）"),
    no_wait: bool = typer.Option(False, "--no-wait", help="跳过所有确认，直接执行"),
    provider: str = typer.Option("deepseek", "--provider", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    sahi: bool = typer.Option(False, "--sahi", help="启用 SAHI 切片推理（大分辨率图像）"),
) -> None:
    """自然语言驱动的 Agentic 标注——无需记忆 CLI 参数。

    示例:
        auto2dlabel chat "检测 000860.png 中的汽车和行人，conf=0.5"
        auto2dlabel chat "检测汽车和行人" -d rtdetr-x.pt --no-wait
        auto2dlabel chat  →  进入交互对话模式
    """
    _setup_logging(verbose)

    from auto2dlabel.tools.confirm import ask_with_timeout

    llm = create_client(provider=provider)
    planner = TaskPlanner(llm_client=llm)
    timeout = 0 if no_wait else confirm_timeout

    # ---- Step 0: 设备信息 ----
    from auto2dlabel.tools.device import print_device
    print_device()

    # ---- Step 1: 获取指令 ----
    if instruction:
        user_text = instruction
    else:
        console.print("[bold]AutoLabel Chat 模式[/bold]")
        console.print("[dim]输入自然语言指令，例如: 检测 /data/images/ 中的汽车和行人[/dim]")
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出[/dim]")
            return
        if not user_text:
            console.print("[red]指令为空，退出[/red]")
            return

    console.print("\n[dim]正在解析指令...[/dim]")

    # ---- Step 2: LLM 解析 → TaskPlan ----
    try:
        plan = planner.parse(user_text, confirm_timeout=timeout)
    except Exception as e:
        console.print(f"[red]解析失败: {e}[/red]")
        raise typer.Exit(code=1)

    # 如果 CLI 指定了 -d，覆盖所有步骤的模型
    if det_model:
        for step in plan.steps:
            step.model_name = det_model

    # ---- Step 3: 缺失参数追问 ----
    _fill_missing_params(plan, planner, timeout)

    # ---- Step 3.5: 类别推荐 ----
    _maybe_recommend_classes(plan, timeout, no_wait)

    # ---- Step 4: 展示计划 + 确认 ----
    if not no_wait:
        confirm_msg = f"[bold]执行计划如下：[/bold]\n{plan.summary}"
        result = ask_with_timeout(confirm_msg, timeout=timeout)
        if not result.confirmed:
            console.print("[yellow]用户取消[/yellow]")
            return
        if result.user_input:
            try:
                plan = planner.parse(
                    plan.raw_instruction + " " + result.user_input,
                    confirm_timeout=timeout,
                )
                console.print(f"[dim]参数已更新: {plan.summary}[/dim]")
            except Exception:
                console.print("[yellow]无法解析修改，使用原计划[/yellow]")

    # ---- Step 5: 执行 TaskPlan ----
    console.print(f"\n[bold]开始执行 {len(plan.steps)} 步任务...[/bold]\n")
    _execute_plan(plan, sahi=sahi)

    console.print("\n[bold green]✓ 全部任务完成[/bold green]")


def _maybe_recommend_classes(plan: TaskPlan, timeout: int, no_wait: bool) -> None:
    """如果 TaskPlan 中有步骤缺少 prompts，运行类别推荐。"""
    from pathlib import Path

    from auto2dlabel.tools.confirm import ask_with_timeout
    from auto2dlabel.tools.recommend import format_recommendation, recommend_classes, scan_classes

    for step in plan.steps:
        if step.prompts:
            continue  # 用户已指定类别，不需要推荐

        source = Path(step.source)
        if not source.exists():
            continue

        # 如果是目录，取第一张图做扫描
        images = _collect_images(source, batch=True) if source.is_dir() else [source]
        if not images:
            continue

        console.print("\n[bold]类别推荐模式[/bold] — 扫描图像以检测可标注的类别...")
        stats = scan_classes(str(images[0]))

        recommended, skipped, _ = recommend_classes(stats)

        if not recommended:
            console.print("[yellow]未检出足够置信度的类别。建议手动指定。[/yellow]")
            continue

        msg = format_recommendation(recommended, skipped, stats)

        if no_wait:
            step.prompts = recommended
            console.print(f"[dim]--no-wait: 自动选择推荐类别 {recommended}[/dim]")
            continue

        result = ask_with_timeout(msg, timeout=timeout)
        if result.user_input and result.user_input.lower() != "ok":
            # 用户自定义类别
            user_classes = [
                c.strip() for c in result.user_input.replace("，", ",").split(",") if c.strip()
            ]
            step.prompts = user_classes
            console.print(f"[dim]已设置类别: {step.prompts}[/dim]")
        else:
            step.prompts = recommended
            console.print(f"[dim]已自动选择: {recommended}[/dim]")


def _fill_missing_params(plan: TaskPlan, planner: TaskPlanner, timeout: int) -> None:
    """如果 TaskPlan 有缺失参数，追问用户。"""
    from auto2dlabel.tools.confirm import ask_missing_params

    missing = plan.all_missing_params
    if not missing:
        return

    reply = ask_missing_params(missing, timeout=timeout)
    if reply:
        try:
            updated = planner.parse(
                plan.raw_instruction + " " + reply,
                confirm_timeout=timeout,
            )
            plan.steps = updated.steps
        except Exception:
            pass


def _execute_plan(plan: TaskPlan, sahi: bool = False) -> None:
    """顺序执行 TaskPlan 的每个步骤。"""
    import time as _time
    from datetime import datetime as _datetime
    from pathlib import Path

    from auto2dlabel.agent.evaluate import QualityReport, evaluate_detections
    from auto2dlabel.models.detection import (
        DetectionResult,
        create_detection_model,
        detect_with_retry,
    )
    from auto2dlabel.schema.annotation import Annotation, Bbox
    from auto2dlabel.tools.export import ExportTool
    from auto2dlabel.tools.hitl import triage_annotations
    from auto2dlabel.tools.log import log_chat_call, log_python_api_call
    from auto2dlabel.tools.visualize import visualize_annotation

    _plan_t0 = _time.time()
    steps_results: list[dict[str, Any]] = []

    for step in plan.steps:
        console.print(f"\n[bold]━━━ Step {step.step_id}: {step.task_type} ━━━[/bold]")
        console.print(f"  数据: {step.source}")
        console.print(f"  类别: {step.prompts}")
        console.print(
            f"  模型: {step.model_name}  conf={step.confidence_threshold}  iou={step.iou_threshold}"
        )

        _t0 = _time.time()
        _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        source_path = Path(step.source)

        # 收集图像
        images = _collect_images(source_path, batch=True if source_path.is_dir() else False)
        if not images:
            console.print(f"[red]未找到图像: {step.source}[/red]")
            continue

        # 分割模型名（如 sam3）不能用于检测——分割任务用默认检测模型
        _seg_only = {"sam3", "sam", "fastsam", "maskrcnn", "sam2"}
        _seg_self_detect = {"sam3", "maskrcnn"}  # 自带检测，不需要外部检测模型
        det_name: str | None = step.model_name
        seg_name = None
        if step.model_name in _seg_only:
            det_name = DEFAULT_MODEL
            seg_name = step.model_name

        # 自带检测的分割模型跳过单独的检测步骤
        if step.model_name in _seg_self_detect:
            det_name = None  # 信号：跳过检测

        # 分类/旋转框任务不创建普通检测模型
        if step.task_type in ("classification", "obb_detection"):
            det_name = None

        model = None
        if det_name:
            model = create_detection_model(det_name, iou_threshold=step.iou_threshold)

        for img_path in images:
            ann = Annotation(image_path=str(img_path))
            try:
                from PIL import Image as _Image
                im = _Image.open(img_path)
                ann.image_size = (im.width, im.height)
            except Exception:
                pass

            # 分类任务：零样本分类 → labels → 导出（跳过检测/分割/可视化/三档分流）
            if step.task_type == "classification":
                from auto2dlabel.models.classification import create_classification_model

                cls_model = create_classification_model(step.model_name)
                ann.labels = cls_model.classify(str(img_path), step.prompts, top_k=5)
                ann.metadata["model"] = step.model_name
                console.print(f"[dim]分类完成: {len(ann.labels)} 个候选标签[/dim]")

                from auto2dlabel.agent.state import AgentState

                state = AgentState(image_path=str(img_path))
                state.annotations = [ann]
                _display_results(state)

                export_tool = ExportTool()
                # cls 为目录型格式：每图写 outputs/{stem}.json
                out_json = export_tool.forward(
                    annotations=[ann.to_dict()],
                    output_path="outputs",
                    format="cls",
                )
                console.print(f"[green]✓ 导出: outputs/{img_path.stem}.json[/green]")

                log_python_api_call(
                    image_path=str(img_path),
                    prompts=step.prompts,
                    results=[lab.to_dict() for lab in ann.labels],
                    elapsed=round(_time.time() - _t0, 3),
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
                continue

            # 旋转框检测任务：YOLO-OBB → Bbox(angle) → dota/yolo_obb 导出（三档分流照旧）
            if step.task_type == "obb_detection":
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

                # obb_quality 命名避免与下方检测分支的 quality 冲突（mypy no-redef）
                obb_quality = evaluate_detections(
                    obb_results, step.prompts, step.confidence_threshold, str(img_path),
                )
                if obb_quality.warnings:
                    console.print(f"[yellow]质量警告: {'; '.join(obb_quality.warnings)}[/yellow]")

                from auto2dlabel.agent.state import AgentState

                state = AgentState(image_path=str(img_path))
                state.annotations = [ann]
                if obb_quality:
                    state.metadata["quality_report"] = obb_quality.to_dict()
                _display_results(state)

                export_tool = ExportTool()
                fmt = step.export_format.replace("-", "_")  # yolo-obb → yolo_obb
                out_json = export_tool.forward(
                    annotations=[ann.to_dict()],
                    output_path="outputs",
                    format=fmt,
                )
                console.print(f"[green]✓ 导出: {out_json}[/green]")

                vis_dir = Path("vis_outputs")
                vis_dir.mkdir(exist_ok=True)
                vis_path = vis_dir / f"vis_{img_path.stem}_{_ts}.png"
                visualize_annotation(img_path, ann, vis_path, draw_mask=False)
                console.print(f"[green]✓ 可视化: {vis_path}[/green]")

                _triage_and_export(state, img_path, step.confidence_threshold, _ts)

                log_python_api_call(
                    image_path=str(img_path),
                    prompts=step.prompts,
                    results=[{
                        "label": r.label, "conf": r.confidence,
                        "cx": r.cx, "cy": r.cy,
                        "width": r.width, "height": r.height, "angle": r.angle,
                    } for r in obb_results],
                    elapsed=round(_time.time() - _t0, 3),
                    model_name=step.model_name,
                    confidence_threshold=step.confidence_threshold,
                    iou_threshold=step.iou_threshold,
                    annotation_type="obb_detection",
                    timestamp=_ts,
                    quality=obb_quality.to_dict(),
                )
                steps_results.append({
                    "step_id": step.step_id,
                    "source": step.source,
                    "model": step.model_name,
                    "prompts": step.prompts,
                    "bbox_count": len(ann.bboxes),
                    "obb_count": len(obb_results),
                    "quality": obb_quality.to_dict(),
                })
                continue

            # 检测步骤（自带检测的分割模型跳过）
            results: list[Any] = []
            quality: QualityReport | None = None
            if det_name:
                assert model is not None  # det_name 非空时已创建
                det_model = model  # 局部别名：lambda 闭包内 pyright 不保留 assert 收窄
                # SAHI 切片推理（来自 CLI --sahi 或 NL 中的 sahi 参数）
                _use_sahi = sahi or getattr(step, "sahi", False)
                if _use_sahi:
                    from auto2dlabel.benchmarks.common import detect_image_sahi
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

            # 分割：自带检测的模型（sam3/maskrcnn）一步到位，其他需要 bbox prompt
            if "segmentation" in step.task_type:
                seg_model_name = seg_name or "fastsam"
                console.print(f"[dim]正在生成分割 mask（{seg_model_name}）...[/dim]")
                from auto2dlabel.models.segmentation import create_segmentation_model
                seg_model = create_segmentation_model(seg_model_name)

                if seg_model_name in ("sam3", "maskrcnn"):
                    # 这些模型自带检测——用用户 prompt 作为文本提示
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

            elapsed = round(_time.time() - _t0, 3)

            # 显示结果
            from auto2dlabel.agent.state import AgentState
            state = AgentState(image_path=str(img_path))
            state.annotations = [ann]
            if quality:
                state.metadata["quality_report"] = quality.to_dict()
            _display_results(state)

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
            _triage_and_export(state, img_path, step.confidence_threshold, _ts)

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

    # 记录顶层 Chat 日志
    plan_summary = {
        "total_steps": len(plan.steps),
        "steps": [s.to_dict() for s in plan.steps],
    }
    log_chat_call(
        instruction=plan.raw_instruction,
        plan_summary=plan_summary,
        steps_results=steps_results,
        elapsed=round(_time.time() - _plan_t0, 3),
        timestamp=_datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )


def _collect_images(path: Path, batch: bool) -> list[Path]:
    """收集图像文件列表。"""
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    if path.is_dir() and batch:
        return sorted([p for p in path.iterdir() if p.suffix.lower() in extensions])
    if path.is_dir():
        console.print("[yellow]目录模式需要 --batch 参数[/yellow]")
        return []
    return []


def _visualize_results(state: AgentState, img_path: Path, timestamp: str = "") -> None:
    """将标注结果可视化保存到 vis_outputs/ 目录。

    Args:
        state: AgentState。
        img_path: 原始图像路径。
        timestamp: 统一时间戳（与日志共用），格式 YYYY-MM-DD-HH-MM-SS。
    """
    from auto2dlabel.tools.visualize import visualize_annotation

    vis_dir = Path("vis_outputs")
    vis_dir.mkdir(exist_ok=True)

    for ann in state.annotations:
        stem = Path(ann.image_path).stem
        if timestamp:
            out_name = f"vis_{stem}_{timestamp}.png"
        else:
            out_name = f"vis_{stem}.png"
        out_path = vis_dir / out_name
        visualize_annotation(img_path, ann, out_path, draw_mask=bool(ann.masks))
        console.print(f"[green]✓ 可视化: {out_path}[/green]")


def _display_results(state: AgentState) -> None:
    """格式化显示标注结果。"""
    if not state.annotations:
        console.print("[yellow]未找到标注结果[/yellow]")
        return

    for ann in state.annotations:
        # 分类结果（labels）单独展示，与 bbox 表格互斥
        if ann.labels and not ann.bboxes:
            label_table = Table(title=f"分类结果: {Path(ann.image_path).name}")
            label_table.add_column("#", style="dim")
            label_table.add_column("Label", style="cyan")
            label_table.add_column("Score", style="yellow")
            for i, lab in enumerate(ann.labels, 1):
                label_table.add_row(str(i), lab.label, f"{lab.score:.2%}")
            console.print(label_table)
            continue

        table = Table(title=f"结果: {Path(ann.image_path).name}")
        table.add_column("#", style="dim")
        table.add_column("Label", style="cyan")
        table.add_column("Bbox [x,y,w,h]", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Status", style="red")

        for i, bbox in enumerate(ann.bboxes, 1):
            bbox_str = f"[{bbox.x:.0f}, {bbox.y:.0f}, {bbox.width:.0f}, {bbox.height:.0f}]"
            conf_str = f"{bbox.confidence:.2%}"
            status = "⚠ REVIEW" if i - 1 in ann.review_flags else "✓"
            table.add_row(str(i), bbox.label, bbox_str, conf_str, status)

        console.print(table)

        if ann.labels:
            console.print(
                "[cyan]分类: "
                + ", ".join(f"{lab.label}={lab.score:.2%}" for lab in ann.labels)
                + "[/cyan]"
            )

        if ann.masks:
            console.print(f"[dim]{len(ann.masks)} segmentation masks[/dim]")
        if ann.review_flags:
            console.print(f"[yellow]⚠ {len(ann.review_flags)} items flagged for review[/yellow]")


def _triage_and_export(
    state: AgentState, img_path: Path, conf_threshold: float, timestamp: str
) -> None:
    """HITL 置信度三档分流：直接采纳 / 人工复核 / 困难样本。"""
    if not state.annotations:
        return

    tau_high = 0.7  # 可配置
    tau_low = max(0.1, conf_threshold * 0.5)

    from auto2dlabel.tools.hitl import export_triage, triage_annotations

    triage = triage_annotations(state.annotations, tau_high=tau_high, tau_low=tau_low)

    # LLM Evaluate 判定整图待复核：全量并入 review 档强制进队列
    if state.metadata.get("llm_review_flagged"):
        triage.review = triage.accepted + triage.review + triage.hard
        triage.accepted = []
        triage.hard = []

    files = export_triage(
        triage,
        output_dir="outputs",
        image_stem=img_path.stem,
        timestamp=timestamp,
        # Web 复核队列需要原图路径与尺寸（消费方显示原图）
        image_path=str(img_path.resolve()),
        image_size=state.annotations[0].image_size if state.annotations else (0, 0),
    )

    console.print(
        f"[dim]HITL 分流: {triage.summary['accepted']} 直接采纳 | "
        f"{triage.summary['review']} 待复核 | {triage.summary['hard']} 困难样本[/dim]"
    )
    if triage.review or triage.hard:
        console.print(
            f"[yellow]  复核队列 → {files['review']}[/yellow]"
        )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
