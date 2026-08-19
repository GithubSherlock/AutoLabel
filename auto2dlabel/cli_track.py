"""`run --track` 流程 —— 帧序列检测 + ByteTrack/BoT-SORT ID 维持 + MOT 导出。

与 run 的 Agentic 主流程不同：跟踪模式不走逐帧 LLM 编排（每帧 LLM 调用成本
与收益不匹配），而是代码级直连检测路径（与 no-LLM baseline 同构）：
    帧序列（视频/帧目录）→ 批量检测 → ByteTracker/BoT-SORT → track_id → 导出/HITL
检测步复用现成检测器（默认 DEFAULT_MODEL，v0.4 Phase 1 验收口径）；
--bot-sort 启用精度档（ReID 外观关联 + ECC 相机运动补偿）。
"""

from __future__ import annotations

import time as _time
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any

import typer

from auto2dlabel.cli_common import console, display_results, triage_and_export
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import DEFAULT_MODEL

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_frames(source: Path, output: Path) -> list[Path]:
    """帧序列收集：视频 → cv2 抽帧到 {output}/track_frames/<stem>/；目录 → 排序图像。

    已抽取的帧不重复抽取（幂等，中断可重跑）。
    """
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            console.print(f"[red]错误: 无法打开视频: {source}[/red]")
            return []
        frame_dir = Path(output) / "track_frames" / source.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]视频 {source.name} → 抽帧 {frame_dir}/[/dim]")
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            out_p = frame_dir / f"frame_{idx:06d}.jpg"
            if not out_p.exists():
                cv2.imwrite(str(out_p), img)
            idx += 1
        cap.release()
        console.print(f"[dim]共 {idx} 帧[/dim]")
        return sorted(frame_dir.glob("frame_*.jpg"))

    from auto2dlabel.cli_common import collect_images

    return collect_images(source, batch=source.is_dir())


def resolve_plan(
    instruction: str,
    use_llm: bool,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    threshold: float,
) -> tuple[list[str], float]:
    """跟踪规划入口：--llm 时序列级一次性 LLM 解析（失败回退），否则代码级解析。

    LLM 覆盖 extract_prompts 处理不了的复杂指令（如「跟踪穿红衣服的人」）；
    每序列仅 1 次调用，绝不逐帧。extract_prompts 的 ValueError 上抛（调用方转 Exit）。
    """
    if use_llm:
        try:
            prompts, llm_threshold = _llm_plan_once(
                instruction, provider, model, api_key, base_url)
            console.print(f"[dim]LLM 规划: 类别 {', '.join(prompts)}[/dim]")
            return prompts, llm_threshold if llm_threshold is not None else threshold
        except Exception as e:
            console.print(f"[yellow]LLM 规划失败，回退 extract_prompts: {e}[/yellow]")
    from auto2dlabel.tools.prompts import extract_prompts

    return extract_prompts(instruction), threshold


def _llm_plan_once(
    instruction: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> tuple[list[str], float | None]:
    """一次性 LLM 指令解析 → (prompts, threshold 建议；None=沿用 CLI 阈值)。

    单轮 tool-less 调用，JSON 输出。prompts 字段对齐 TaskPlan schema（为其子集，
    未来升级完整 planner 编排时无损）。输出无效时抛 ValueError/JSONDecodeError。
    """
    import json

    from auto2dlabel.agent.llm import create_client

    client = create_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是自动标注系统的跟踪指令解析器。把用户指令解析为 JSON：\n"
                    '{"prompts": ["COCO 英文类别名"], "threshold": 0到1的浮点数或 null}\n'
                    "threshold=null 表示沿用 CLI 阈值。只输出 JSON，不要其他文字。"
                ),
            },
            {"role": "user", "content": instruction},
        ],
        temperature=0.0,
    )
    content = (response.content or "").strip()
    # 剥离可能的 markdown 代码块
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)

    raw_prompts = data.get("prompts") or []
    if not raw_prompts or not all(isinstance(p, str) for p in raw_prompts):
        raise ValueError(f"LLM 规划输出无效 prompts: {data}")
    prompts = [str(p) for p in raw_prompts]

    raw_threshold = data.get("threshold")
    llm_threshold: float | None
    if raw_threshold is None:
        llm_threshold = None
    elif (
        isinstance(raw_threshold, (int, float))
        and not isinstance(raw_threshold, bool)
        and 0 <= raw_threshold <= 1
    ):
        llm_threshold = float(raw_threshold)
    else:
        raise ValueError(f"LLM 规划输出无效 threshold: {data}")
    return prompts, llm_threshold


def run_tracking(
    source: str,
    instruction: str,
    threshold: float,
    iou: float,
    export: str,
    output: str,
    det_model: str | None,
    sahi: bool,
    verbose: bool,
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    use_llm: bool = False,
    use_bot_sort: bool = False,
    reid_model_name: str = "openai/clip-vit-base-patch32",
) -> None:
    """`run --track` 实现：帧序列 → 检测（批量）→ ByteTracker/BoT-SORT → MOT 导出。

    --llm 时启用序列级一次性 LLM 指令解析（resolve_plan），失败回退代码级
    extract_prompts；无论是否用 LLM，全程无逐帧 LLM 调用。
    use_bot_sort 启用精度档：ReID 特征（CLIP/SigLIP，懒加载）注入融合关联
    + ECC 相机运动补偿；PIL 打开失败自动降级纯 ByteTrack 语义。
    """
    from auto2dlabel.cli_common import setup_logging

    setup_logging(verbose)

    frames = collect_frames(Path(source), Path(output))
    if not frames:
        console.print(f"[red]错误: 未找到可跟踪的帧/视频: {source}[/red]")
        raise typer.Exit(code=1)
    try:
        prompts, threshold = resolve_plan(
            instruction, use_llm, provider, model, api_key, base_url, threshold)
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1)

    det_name = det_model or DEFAULT_MODEL
    console.print(f"[dim]跟踪模式: {len(frames)} 帧  类别: {', '.join(prompts)}[/dim]")
    console.print(f"[dim]检测模型: {det_name}  置信度: {threshold}  IoU: {iou}[/dim]")

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.tracking import (
        BotSORTTracker,
        ByteTracker,
        ReIDModel,
        create_reid_model,
        extract_frame_features,
    )

    det = create_detection_model(det_name, iou_threshold=iou)
    tracker: ByteTracker
    reid_model_inst: ReIDModel | None = None
    if use_bot_sort:
        reid_model_inst = create_reid_model(reid_model_name)
        tracker = BotSORTTracker()
        console.print(f"[dim]跟踪器: BoT-SORT  ReID 特征: {reid_model_name}[/dim]")
    else:
        tracker = ByteTracker()

    # 批量推理超参（同 v0.3 四档：resolve_batch_params 内 disable_tf32 + 动态实测；
    # 无 CUDA/无批量能力回退逐图）
    det_batch = getattr(det, "detect_batch", None)
    infer_fn = (lambda ps: det_batch(ps, prompts, threshold, 0)) if det_batch else None
    from auto2dlabel.tools.device import resolve_batch_params

    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, [str(p) for p in frames[:20]],
    )
    console.print(f"  批量: batch_size={batch_size}  num_workers={num_workers}")

    _t0 = _time.time()
    _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    dets_per_frame = _detect_frames(
        det, frames, prompts, threshold, batch_size, num_workers, sahi)

    annotations: list[Annotation] = []
    vis_dir = Path("vis_outputs")
    vis_dir.mkdir(exist_ok=True)

    from auto2dlabel.agent.state import AgentState
    from auto2dlabel.tools.export import ExportTool
    from auto2dlabel.tools.log import log_python_api_call
    from auto2dlabel.tools.visualize import visualize_annotation

    export_tool = ExportTool()

    for frame_path, dets in zip(frames, dets_per_frame):
        bboxes = [
            Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                 label=r.label, confidence=r.confidence)
            for r in dets
        ]

        ann = Annotation(image_path=str(frame_path))
        im: Any | None = None  # PIL Image，BoT-SORT 分支复用；打开失败 None
        try:
            from PIL import Image as _Image

            im = _Image.open(frame_path)
            ann.image_size = (im.width, im.height)
        except Exception:
            pass

        # BoT-SORT：复用上面打开的帧图（RGB→BGR 供 ECC），高分框批量提 ReID
        # 特征注入融合关联；任一前置缺失（PIL 失败等）降级纯 ByteTrack 语义
        if use_bot_sort and reid_model_inst is not None and im is not None:
            assert isinstance(tracker, BotSORTTracker)
            import numpy as np

            feats = extract_frame_features(
                reid_model_inst, im, bboxes,
                min_conf=tracker.track_high_thresh)
            tracker.update(bboxes, image=np.asarray(im)[:, :, ::-1], features=feats)
        else:
            tracker.update(bboxes)
        for b in bboxes:
            ann.add_bbox(b)
            if b.confidence < threshold:
                ann.flag_for_review(len(ann.bboxes) - 1)
        annotations.append(ann)

        state = AgentState(image_path=str(frame_path))
        state.annotations = [ann]
        display_results(state)

        # 导出 JSON（track_id 随 Bbox.to_dict 透出）
        out_json = export_tool.forward(
            annotations=[ann.to_dict()],
            output_path=f"{output}/{frame_path.stem}_{_ts}.json",
            format=export,
        )
        console.print(f"[green]✓ 已导出: {out_json}[/green]")

        # 可视化（轨迹线 + ID 角标）
        vis_path = vis_dir / f"vis_{frame_path.stem}_{_ts}.png"
        trajectories = {
            t.track_id: t.trajectory()
            for t in tracker.tracks() if len(t.trajectory()) >= 2
        }
        visualize_annotation(
            frame_path, ann, vis_path, draw_mask=False, trajectories=trajectories)
        console.print(f"[green]✓ 可视化: {vis_path}[/green]")

        # HITL 置信度分流（与 run 路径一致）
        triage_and_export(state, frame_path, threshold, _ts)

        log_python_api_call(
            image_path=str(frame_path),
            prompts=prompts,
            results=dets,
            elapsed=round(_time.time() - _t0, 3),
            model_name=det_name,
            confidence_threshold=threshold,
            iou_threshold=iou,
            annotation_type="object_detection_tracking",
            timestamp=_ts,
        )

    # MOT 导出（全帧合并，frame 从 1 起）
    from auto2dlabel.export.mot import export_mot

    mot_path = Path(output) / f"{Path(source).stem}_mot.txt"
    export_mot(annotations, mot_path)
    console.print(f"[green]✓ MOT 导出: {mot_path}[/green]")

    tracked_ids = {
        b.track_id for ann in annotations for b in ann.bboxes if b.track_id is not None
    }
    console.print(
        f"\n[green]✓ 跟踪完成: {len(annotations)} 帧, {len(tracked_ids)} 条轨迹[/green]"
    )


def _detect_frames(
    model: Any,
    frames: list[Path],
    prompts: list[str],
    threshold: float,
    batch_size: int,
    num_workers: int,
    sahi: bool,
) -> list[list[Any]]:
    """逐帧检测（分块批量 + 0 框降阈值重试 + OOM 降级逐图），返回帧序对齐的结果。"""
    from collections.abc import Callable

    det_batch: (
        Callable[[list[str], list[str], float, int], list[list[Any]]] | None
    ) = getattr(model, "detect_batch", None)
    out: list[list[Any]] = []

    if det_batch is not None and batch_size > 1 and not sahi:
        idx = 0
        try:
            for i in range(0, len(frames), batch_size):
                chunk = frames[i:i + batch_size]
                per = det_batch([str(p) for p in chunk], prompts, threshold, num_workers)
                for frame_path, results in zip(chunk, per):
                    if results:
                        out.append(results)
                    else:
                        out.append(_detect_one(
                            model, frame_path, prompts, threshold, sahi))
                idx = i + batch_size
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            console.print("[yellow]⚠ 批量推理 OOM，剩余帧降级逐图[/yellow]")
            import torch

            torch.cuda.empty_cache()
            for frame_path in frames[idx:]:
                out.append(_detect_one(model, frame_path, prompts, threshold, sahi))
        return out

    return [_detect_one(model, f, prompts, threshold, sahi) for f in frames]


def _detect_one(
    model: Any, frame_path: Path, prompts: list[str], threshold: float, sahi: bool,
) -> list[Any]:
    """单帧检测（0 框降阈值重试一次，与批量路径行为一致）。"""
    from auto2dlabel.models.detection import (
        DetectionResult,
        detect_image_sahi,
        detect_with_retry,
    )

    if sahi:
        dets, _, _ = detect_with_retry(
            lambda conf: detect_image_sahi(
                model, str(frame_path), prompts, confidence_threshold=conf),
            threshold,
        )
        return [
            DetectionResult(
                x=d["bbox"][0], y=d["bbox"][1],
                width=d["bbox"][2] - d["bbox"][0],
                height=d["bbox"][3] - d["bbox"][1],
                label=d["name"], confidence=d["conf"],
            )
            for d in dets
        ]
    results, _, _ = detect_with_retry(
        lambda conf: model.detect(str(frame_path), prompts, conf), threshold)
    return results
