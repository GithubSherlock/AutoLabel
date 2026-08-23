"""`run` 命令实现 —— 批次标注：新建/续跑清单 → 逐图编排 → 导出/可视化/HITL 分流。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from auto2dlabel.agent.llm import create_client
from auto2dlabel.agent.orchestrator import AgentOrchestrator
from auto2dlabel.agent.state import AgentState
from auto2dlabel.cli_common import (
    collect_images,
    console,
    display_results,
    setup_logging,
    triage_and_export,
    visualize_results,
)
from auto2dlabel.schema.task_plan import DEFAULT_MODEL


def run_command(
    image: str,
    instruction: str,
    threshold: float,
    iou: float,
    export: str,
    output: str,
    batch: bool,
    resume: str,
    provider: str,
    model: str | None,
    det_model: str | None,
    api_key: str,
    base_url: str,
    verbose: bool,
    sahi: bool,
    track: bool = False,
    use_llm: bool = False,
    bot_sort: bool = False,
    reid_model: str = "openai/clip-vit-base-patch32",
    viz: bool = True,
    roi: str | None = None,
    refer_l2: bool = False,
    refer_l3: bool = False,
) -> None:
    """`run` 命令实现：续跑/新建清单 → 逐图编排 → 导出/可视化/HITL 分流。"""
    setup_logging(verbose)

    # ---- 跟踪模式：帧序列检测 + ByteTrack ID 维持 + MOT 导出（--llm 仅一次性规划） ----
    if track:
        if resume:
            console.print("[yellow]跟踪模式不支持 --resume（帧序列可整段重跑）[/yellow]")
        from auto2dlabel.cli_track import run_tracking

        run_tracking(
            source=image,
            instruction=instruction,
            threshold=threshold,
            iou=iou,
            export=export,
            output=output,
            det_model=det_model,
            sahi=sahi,
            verbose=verbose,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_llm=use_llm,
            use_bot_sort=bot_sort,
            reid_model_name=reid_model,
            viz=viz,
            roi=roi,
            refer_l2=refer_l2,
            refer_l3=refer_l3,
        )
        return

    # ---- 续跑模式：清单为单一事实源（图像列表/指令/参数均从清单读取） ----
    restored: dict[str, AgentState] = {}  # 快照恢复表（仅 resume 分支填充）
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
        targets = [e for e in resume_targets(manifest) if Path(e.path).exists()]
        if not targets:
            console.print("[green]清单中无待重跑图像（全部 ok 或文件已不存在）[/green]")
            return

        # 快照恢复（v0.4 Phase 3）：带 state_file 的条目从 AgentState 快照续跑
        # （跳过已完成的 Agent Loop 迭代，防重复检测红线不破）；损坏快照回退整图重跑
        restored = {}
        for e in targets:
            if not e.state_file or not Path(e.state_file).is_file():
                continue
            try:
                st = AgentState.from_dict(
                    json.loads(Path(e.state_file).read_text(encoding="utf-8"))
                )
                restored[e.path] = st
            except Exception:
                console.print(f"[yellow]快照损坏，将整图重跑: {e.state_file}[/yellow]")

        image_files = [Path(e.path) for e in targets]
        console.print(
            f"[dim]续跑模式: 清单 {resume} → 重跑 {len(image_files)} 张"
            f"（快照续跑 {len(restored)} 张）[/dim]"
        )
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
        image_files = collect_images(image_path, batch)
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
                initial_state=restored.get(str(img_path.resolve())),
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
        display_results(state)

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
        visualize_results(state, img_path, timestamp=_ts)

        # HITL 置信度分流
        triage_and_export(state, img_path, threshold, _ts)

        # 写 AgentState 快照 + 更新清单（快照供 --resume 的 from_dict 续跑与调试审计）
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
