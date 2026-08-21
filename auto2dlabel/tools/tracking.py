"""跟踪 Tool —— 视频/帧序列固定类别跟踪（ByteTrack/BoT-SORT）+ MOT 导出。

Tool 封装层语义与 detection.py/segmentation.py 一致：TrackingTool 是 Tool 子类，
管线实现收敛在此；`run --track`（cli_track.py 薄壳）与 chat 的 execute_plan
track 分支共用同一实现。与 run 的 Agentic 主流程不同：跟踪模式不走逐帧 LLM
编排（每帧 LLM 调用成本与收益不匹配），而是代码级直连检测路径
（与 no-LLM baseline 同构）：
    帧序列（视频/帧目录）→ 批量检测 → ByteTracker/BoT-SORT → track_id → 导出/HITL
use_bot_sort 启用精度档：ReID 外观关联（CLIP/SigLIP 懒加载）+ ECC 相机运动补偿。
跟踪算法本体在 models/tracking.py（零权重依赖），本模块负责管线编排。

注意：TrackingTool 不注册 LLM registry（orchestrator 单图循环调不动序列工具），
调用方是 execute_plan track 分支与 cli_track 薄壳；错误只抛 ValueError，
typer.Exit 收敛在 CLI 层。
"""

from __future__ import annotations

import os
import time as _time
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any

from auto2dlabel.cli_common import console, display_results, triage_and_export
from auto2dlabel.models.detection import DetectionModel, create_detection_model
from auto2dlabel.models.tracking import (
    BotSORTTracker,
    ByteTracker,
    ReIDModel,
    create_reid_model,
    extract_frame_features,
)
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.schema.task_plan import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
)
from auto2dlabel.tools.base import Tool

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_REID_MODEL = "openai/clip-vit-base-patch32"


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


def detect_tracker_kind(text: str) -> str:
    """代码级扫描跟踪器选择（LLM 不参与）：BoT-SORT 别名 → "bot_sort"，否则 "bytetrack"。"""
    lower = text.lower()
    for alias in ("bot-sort", "botsort", "bot_sort", "bot sort"):
        if alias in lower:
            return "bot_sort"
    return "bytetrack"


class TrackingTool(Tool):
    """序列跟踪 Tool —— 帧序列检测 + ByteTrack/BoT-SORT ID 维持 + MOT 导出。

    序列级操作（视频 → 逐帧 → MOT），不注册 LLM registry：
    orchestrator 单图循环无法调用，调用方为 execute_plan track 分支与
    cli_track 薄壳。模型/ReID 懒加载（仿 DetectionTool），测试可注入
    Fake 检测模型与 FakeReIDModel（零真实权重）。
    """

    name = "track_sequence"
    description = (
        "Track fixed object classes across a video or image sequence "
        "(ByteTrack or BoT-SORT), writing per-frame detection JSON and a "
        "combined MOT-format file with track IDs."
    )

    def __init__(
        self,
        model: DetectionModel | None = None,
        model_name: str | None = None,
        iou_threshold: float = DEFAULT_IOU,
        use_sahi: bool = False,
        use_bot_sort: bool = False,
        reid_model_name: str = DEFAULT_REID_MODEL,
        reid_model: ReIDModel | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        output_dir: str = "outputs",
        export_format: str = "coco",
        viz: bool = True,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._iou_threshold = iou_threshold
        self.use_sahi = use_sahi
        self.use_bot_sort = use_bot_sort
        self._reid_model_name = reid_model_name
        self._reid_model = reid_model
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._output_dir = output_dir
        self.export_format = export_format
        # viz=False：跳过逐帧 PNG 可视化（vis_outputs 大头）；MOT/JSON/HITL/成片视频不受影响
        self.viz = viz

    @property
    def model(self) -> DetectionModel:
        if self._model is None:
            name = self._model_name or os.environ.get("DETECTION_MODEL")
            self._model = create_detection_model(name, iou_threshold=self._iou_threshold)
        return self._model

    @property
    def reid_model(self) -> ReIDModel:
        if self._reid_model is None:
            self._reid_model = create_reid_model(self._reid_model_name)
        return self._reid_model

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "视频文件（mp4/avi/mov/mkv）或帧图像目录路径。",
                },
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待跟踪的固定类别列表（英文 COCO 类名）。",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": DEFAULT_CONFIDENCE,
                    "description": "检测置信度阈值（标注工具低阈值，宁多勿漏）。",
                },
            },
            "required": ["source", "prompts"],
        }

    def forward(  # type: ignore[override]  # Tool.__call__ 传 **kwargs，窄签名属预期
        self,
        source: str,
        prompts: list[str],
        confidence_threshold: float = DEFAULT_CONFIDENCE,
    ) -> dict[str, Any]:
        """执行序列跟踪 → 逐帧检测 JSON + 可视化 + HITL + MOT 合并导出。

        Args:
            source: 视频文件或帧图像目录路径。
            prompts: 固定类别列表（英文 COCO 类名）。
            confidence_threshold: 检测置信度阈值。

        Returns:
            {"frames", "bboxes", "track_ids", "mot_path", "video_path"} 汇总；
            视频源附带标注成片（源视频同目录 output_<原名>.mp4），帧目录为 None。
            viz=False 时跳过逐帧 PNG 可视化（vis_outputs）——测试/省磁盘场景；
            逐帧 JSON、MOT、HITL 复核与成片视频不受影响。

        Raises:
            ValueError: 未找到可跟踪的帧/视频。
        """
        threshold = confidence_threshold
        frames = collect_frames(Path(source), Path(self._output_dir))
        if not frames:
            raise ValueError(f"未找到可跟踪的帧/视频: {source}")

        det_name = self._model_name or DEFAULT_MODEL
        console.print(f"[dim]跟踪模式: {len(frames)} 帧  类别: {', '.join(prompts)}[/dim]")
        console.print(
            f"[dim]检测模型: {det_name}  置信度: {threshold}  IoU: {self._iou_threshold}[/dim]"
        )

        det = self.model
        tracker: ByteTracker
        reid_model_inst: ReIDModel | None = None
        if self.use_bot_sort:
            reid_model_inst = self.reid_model
            tracker = BotSORTTracker()
            console.print(f"[dim]跟踪器: BoT-SORT  ReID 特征: {self._reid_model_name}[/dim]")
        else:
            tracker = ByteTracker()

        # 批量推理超参（同 v0.3 四档：resolve_batch_params 内 disable_tf32 + 动态实测；
        # 无 CUDA/无批量能力回退逐图；显式 batch_size/num_workers 恒优先）
        det_batch = getattr(det, "detect_batch", None)
        infer_fn = (lambda ps: det_batch(ps, prompts, threshold, 0)) if det_batch else None
        from auto2dlabel.tools.device import resolve_batch_params

        batch_size, num_workers = resolve_batch_params(
            "object_detection",
            infer_fn,
            [str(p) for p in frames[:20]],
            explicit_batch=self._batch_size,
            explicit_workers=self._num_workers,
        )
        console.print(f"  批量: batch_size={batch_size}  num_workers={num_workers}")

        _t0 = _time.time()
        _ts = _datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        dets_per_frame = _detect_frames(
            det, frames, prompts, threshold, batch_size, num_workers, self.use_sahi
        )

        annotations: list[Annotation] = []
        vis_dir = Path("vis_outputs")
        if self.viz:
            vis_dir.mkdir(exist_ok=True)

        from auto2dlabel.agent.state import AgentState
        from auto2dlabel.tools.export import ExportTool
        from auto2dlabel.tools.log import log_python_api_call
        from auto2dlabel.tools.visualize import draw_bboxes, draw_trajectories, visualize_annotation

        export_tool = ExportTool()

        # 标注成片视频输出（仅视频源；帧目录不产片——避免污染数据集目录）。
        # 位置：源视频同目录 output_<原名>.mp4；帧率取源视频（异常回退 30）。
        video_writer: Any | None = None
        video_path: Path | None = None
        source_path = Path(source)
        if source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTS:
            import cv2

            fps = 30.0
            try:
                cap = cv2.VideoCapture(str(source_path))
                fps_raw = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if 0 < fps_raw <= 240:
                    fps = fps_raw
            except Exception:
                pass
            first = cv2.imread(str(frames[0]))
            if first is not None:
                h, w = first.shape[:2]
                video_path = source_path.parent / f"output_{source_path.stem}.mp4"
                try:
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                        fps,
                        (w, h),
                    )
                except Exception as e:
                    console.print(f"[yellow]⚠ 视频输出不可用: {e}[/yellow]")
                    video_path = None
                if video_writer is not None and not video_writer.isOpened():
                    console.print("[yellow]⚠ 视频编码器 mp4v 不可用，跳过视频输出[/yellow]")
                    video_writer = None
                    video_path = None

        try:
            for frame_path, dets in zip(frames, dets_per_frame):
                bboxes = [
                    Bbox(
                        x=r.x,
                        y=r.y,
                        width=r.width,
                        height=r.height,
                        label=r.label,
                        confidence=r.confidence,
                    )
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
                if self.use_bot_sort and reid_model_inst is not None and im is not None:
                    assert isinstance(tracker, BotSORTTracker)
                    import numpy as np

                    feats = extract_frame_features(
                        reid_model_inst, im, bboxes, min_conf=tracker.track_high_thresh
                    )
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
                    output_path=f"{self._output_dir}/{frame_path.stem}_{_ts}.json",
                    format=self.export_format,
                )
                console.print(f"[green]✓ 已导出: {out_json}[/green]")

                # 轨迹线（成片视频同源）；--no-viz 时仅跳过 PNG 落盘
                trajectories = {
                    t.track_id: t.trajectory() for t in tracker.tracks() if len(t.trajectory()) >= 2
                }
                if self.viz:
                    # 可视化（轨迹线 + ID 角标）
                    vis_path = vis_dir / f"vis_{frame_path.stem}_{_ts}.png"
                    visualize_annotation(
                        frame_path, ann, vis_path, draw_mask=False, trajectories=trajectories
                    )
                    console.print(f"[green]✓ 可视化: {vis_path}[/green]")

                # 标注成片帧写入（与 PNG 同一绘制管线：框 + ID 角标 + 轨迹线）
                if video_writer is not None:
                    import cv2

                    frame_bgr = cv2.imread(str(frame_path))
                    if frame_bgr is not None:
                        drawn = draw_bboxes(frame_bgr, ann.bboxes)
                        drawn = draw_trajectories(drawn, trajectories)
                        video_writer.write(drawn)

                # HITL 置信度分流（与 run 路径一致）
                triage_and_export(state, frame_path, threshold, _ts)

                log_python_api_call(
                    image_path=str(frame_path),
                    prompts=prompts,
                    results=dets,
                    elapsed=round(_time.time() - _t0, 3),
                    model_name=det_name,
                    confidence_threshold=threshold,
                    iou_threshold=self._iou_threshold,
                    annotation_type="object_detection_tracking",
                    timestamp=_ts,
                )
        finally:
            if video_writer is not None:
                video_writer.release()

        # MOT 导出（全帧合并，frame 从 1 起）
        from auto2dlabel.export.mot import export_mot

        mot_path = Path(self._output_dir) / f"{Path(source).stem}_mot.txt"
        export_mot(annotations, mot_path)
        console.print(f"[green]✓ MOT 导出: {mot_path}[/green]")
        if video_path is not None:
            console.print(f"[green]✓ 视频输出: {video_path}[/green]")

        tracked_ids = {
            b.track_id for ann in annotations for b in ann.bboxes if b.track_id is not None
        }
        console.print(
            f"\n[green]✓ 跟踪完成: {len(annotations)} 帧, {len(tracked_ids)} 条轨迹[/green]"
        )
        return {
            "frames": len(annotations),
            "bboxes": sum(len(a.bboxes) for a in annotations),
            "track_ids": sorted(tracked_ids),
            "mot_path": str(mot_path),
            "video_path": str(video_path) if video_path is not None else None,
        }


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

    det_batch: Callable[[list[str], list[str], float, int], list[list[Any]]] | None = getattr(
        model, "detect_batch", None
    )
    out: list[list[Any]] = []

    if det_batch is not None and batch_size > 1 and not sahi:
        idx = 0
        try:
            for i in range(0, len(frames), batch_size):
                chunk = frames[i : i + batch_size]
                per = det_batch([str(p) for p in chunk], prompts, threshold, num_workers)
                for frame_path, results in zip(chunk, per):
                    if results:
                        out.append(results)
                    else:
                        out.append(_detect_one(model, frame_path, prompts, threshold, sahi))
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
    model: Any,
    frame_path: Path,
    prompts: list[str],
    threshold: float,
    sahi: bool,
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
                model, str(frame_path), prompts, confidence_threshold=conf
            ),
            threshold,
        )
        return [
            DetectionResult(
                x=d["bbox"][0],
                y=d["bbox"][1],
                width=d["bbox"][2] - d["bbox"][0],
                height=d["bbox"][3] - d["bbox"][1],
                label=d["name"],
                confidence=d["conf"],
            )
            for d in dets
        ]
    results, _, _ = detect_with_retry(
        lambda conf: model.detect(str(frame_path), prompts, conf), threshold
    )
    return results
