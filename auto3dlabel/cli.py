"""Auto3dLabel CLI：run（代码级直跑，零 LLM）+ chat（LLM planner → 3D agent loop → HITL）。

用法：
    auto3dlabel run 000123 "检测汽车和行人" --det-model kitti_finetune
    auto3dlabel run 003712-003731 "检测汽车" --batch --resume
    auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from auto3dlabel.configs.kitti import DEFAULT_CONF, DEFAULT_KITTI_ROOT
from auto3dlabel.configs.model_catalog import DETECTOR3D_NAMES
from auto3dlabel.configs.nuscenes import DEFAULT_NUSCENES_OUT
from auto3dlabel.data.kitti import normalize_frame_id, resolve_frame

app = typer.Typer(help="Agentic 3D 标注（KITTI 单帧 → 3D bbox 初稿 + HITL 三档）")
console = Console()


def _require_auto2dlabel() -> None:
    """auto2dlabel 为 auto3dlabel 硬依赖（复用 agent/模型/工具骨架）。

    install_libs.sh 3d 已连带安装 2D 依赖，但包本体须在 import 路径中
    （仓库根 `pip install -e .` 同时安装 auto2dlabel + auto3dlabel）；
    此处兜底给出可操作提示，而非裸 ImportError 堆栈。
    """
    try:
        import auto2dlabel  # noqa: F401
    except ImportError:
        console.print(
            "[red]未找到 auto2dlabel 包[/red]（auto3dlabel 复用其 agent/模型/工具骨架）。\n"
            "请在仓库根目录执行: pip install -e .\n"
            "或确认 auto2dlabel 在 PYTHONPATH 中后重试。"
        )
        raise typer.Exit(code=1) from None


def _resolve_det_model(name: str | None) -> str:
    """快捷名映射：kitti_finetune/yolo11s_kitti → KITTI 微调权重完整路径
    （.pt 后缀路由到 UltralyticsModel）。"""
    if name in ("kitti_finetune", "yolo11s_kitti"):
        path = Path("auto2dlabel/weights/kitti_finetune/yolo11s_kitti/weights/best.pt")
        if path.is_file():
            return str(path)
        console.print(f"[yellow]KITTI 微调权重不存在: {path}，回退默认模型[/yellow]")
        return "IDEA-Research/grounding-dino-tiny"
    return name or "IDEA-Research/grounding-dino-tiny"


def _parse_frame_ids(spec: str) -> list[str]:
    """'000123' / '123' / '003712-003731' / 'image_2/' → 帧 ID 列表。"""
    from auto3dlabel.data.kitti import frame_ids_by_range, frame_ids_from_dir

    if "-" in spec and spec.split("-")[0].isdigit() and spec.split("-")[1].isdigit():
        start, end = (int(x) for x in spec.split("-"))
        return frame_ids_by_range(start, end + 1)  # 含两端（CLI 语义）
    if spec.isdigit():
        return [normalize_frame_id(spec)]
    path = Path(spec)
    if path.is_dir():
        return frame_ids_from_dir(path, root=DEFAULT_KITTI_ROOT)
    raise typer.BadParameter(f"无法识别的帧 ID/目录: {spec}")


def _make_models(
    det_model: str | None, seg_model: str | None
) -> tuple[object, object, object]:
    """双引擎模型工厂：det_model 命中 DETECTOR3D_NAMES → LiDAR 直检（seg 不实例化）。

    Returns: (det, seg, det3d)（det3d 非 None 时 det/seg 为 None）
    """
    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto3dlabel.models.detection3d import create_detector3d

    det3d = create_detector3d(det_model)
    if det3d is not None:
        return None, None, det3d
    return (
        create_detection_model(_resolve_det_model(det_model)),
        create_segmentation_model(seg_model or "sam2_l.pt"),
        None,
    )


def _run_frame(
    frame_id: str,
    prompts: list[str],
    det: Any,
    seg: Any,
    det3d: Any,
    conf: float,
    out_dir: Path,
    viz: bool,
) -> tuple[int, int, int]:
    """单帧跑管线 + 导出产物 → (accepted, review, hard)。

    模型实例由调用方注入（批量路径只加载一次，不再逐帧重建）。
    """
    from auto3dlabel.export.kitti_label import build_label_file
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue
    from auto3dlabel.tools.pipeline import annotate_frame

    frame = resolve_frame(frame_id)
    result = annotate_frame(
        frame,
        prompts,
        det_model=det,
        seg_model=seg,
        det3d=det3d,
        confidence=conf,
        viz=viz,
        out_dir=out_dir,
    )
    triage = triage_3d(result.boxes3d)
    build_label_file(frame.frame_id, result.boxes3d, out_dir / "labels")
    write_review_queue(frame, triage, out_dir / "reviews")
    return len(triage.accepted), len(triage.review), len(triage.hard)


def _is_oom(e: BaseException) -> bool:
    """CUDA OOM 判定（RuntimeError "out of memory"）。"""
    return "out of memory" in str(e).lower()


def _measure_batch_size(
    det3d: Any,
    probe_frames: list[Any],
    prompts: list[str],
    conf: float,
    viz: bool,
    out_dir: Path,
) -> int:
    """自动实测批大小：batch=1 峰值显存增量 → 空闲 × 0.92 ÷ 单帧峰值。

    热身后测增量（首帧前向含模型加载/权重，不得计入单帧峰值）；
    安全系数 0.92（目标「跑满 GPU」≈92% 显存；估算失真由整批 OOM 减半吸收，
    故可高于 2D 的 0.85）；CUDA 不可用 → 1（逐帧，等同 2D 兜底）。
    """
    import torch

    from auto3dlabel.models.detection3d import suggest_batch_size
    from auto3dlabel.tools.pipeline import annotate_frames_lidar_batch

    if not probe_frames or not torch.cuda.is_available():
        return 1
    annotate_frames_lidar_batch(probe_frames, prompts, det3d, conf, viz, out_dir)  # 热身
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated(0)  # 模型常驻显存（探针前已加载）
    annotate_frames_lidar_batch(probe_frames, prompts, det3d, conf, viz, out_dir)
    per_frame = max(1, torch.cuda.max_memory_allocated(0) - base)  # 单帧激活峰值
    free = torch.cuda.mem_get_info(0)[0]
    return suggest_batch_size(int(per_frame), int(free), safety_factor=0.92)


def _run_batch_lidar(
    frame_ids: list[str],
    prompts: list[str],
    det3d: Any,
    conf: float,
    out_dir: Path,
    viz: bool,
    resume: bool,
    batch_size: int | None,
) -> dict[str, int]:
    """LiDAR 引擎批量路径：分批 detect_batch（一次 forward 整批，显存换吞吐）。

    batch_size None → 自动实测（跑满 GPU：空闲 × 0.85 ÷ 单帧峰值）；
    整批 OOM 减半重试；非 OOM 异常逐帧降级单帧（失败隔离口径同 2D 批量）。
    """
    from auto3dlabel.export.kitti_label import build_label_file
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue
    from auto3dlabel.tools.pipeline import annotate_frames_lidar_batch

    stat = {"accepted": 0, "review": 0, "hard": 0, "failed": 0}
    todo = [
        fid for fid in frame_ids
        if not (resume and (out_dir / "labels" / f"{fid}.txt").is_file())
    ]
    if not todo:
        return stat
    frames = [resolve_frame(fid) for fid in todo]
    batch = batch_size or _measure_batch_size(
        det3d, frames[:1], prompts, conf, viz, out_dir
    )
    console.print(f"[dim]batch size = {batch}（{'实测' if batch_size is None else '显式'}）[/dim]")

    i = 0
    while i < len(frames):
        chunk = frames[i : i + batch]
        try:
            results = annotate_frames_lidar_batch(chunk, prompts, det3d, conf, viz, out_dir)
        except RuntimeError as e:
            if _is_oom(e) and batch > 1:
                console.print(f"[yellow]batch={batch} 显存不足（OOM），减半重试[/yellow]")
                batch = max(1, batch // 2)
                continue
            console.print(
                f"[red]批次 {chunk[0].frame_id}-{chunk[-1].frame_id} 失败（{e}），逐帧降级[/red]"
            )
            for frame in chunk:
                try:
                    a, r, h = _run_frame(
                        frame.frame_id, prompts, None, None, det3d, conf, out_dir, viz
                    )
                    stat["accepted"] += a
                    stat["review"] += r
                    stat["hard"] += h
                    console.print(f"  {frame.frame_id}: 采纳 {a} / 复核 {r} / 困难 {h}")
                except Exception as e2:  # 失败隔离：单帧异常不中断批量
                    stat["failed"] += 1
                    console.print(f"  [red]{frame.frame_id}: 失败 {e2}[/red]")
            i += len(chunk)
            continue
        for frame, result in zip(chunk, results, strict=True):
            triage = triage_3d(result.boxes3d)
            build_label_file(frame.frame_id, result.boxes3d, out_dir / "labels")
            write_review_queue(frame, triage, out_dir / "reviews")
            stat["accepted"] += len(triage.accepted)
            stat["review"] += len(triage.review)
            stat["hard"] += len(triage.hard)
            console.print(
                f"  {frame.frame_id}: 采纳 {len(triage.accepted)} / 复核 {len(triage.review)} / "
                f"困难 {len(triage.hard)}"
            )
        i += len(chunk)
    return stat


def _run_tracked_sequence(
    frame_ids: list[str],
    prompts: list[str],
    det_model: str | None,
    seg_model: str | None,
    conf: float,
    out_dir: Path,
    viz: bool,
) -> dict[str, int]:
    """多帧跟踪：逐帧 annotate → Tracker3D 回写 track_id → label/复核队列/序列 tracks.json。

    Returns: 统计 {accepted, review, hard, failed}（同 run batch 汇总口径）。
    """
    import json

    from auto3dlabel.export.kitti_label import build_label_file
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue
    from auto3dlabel.tools.pipeline import annotate_frame
    from auto3dlabel.tools.track3d import Tracker3D

    det, seg, det3d = _make_models(det_model, seg_model)
    tracker = Tracker3D()
    stat = {"accepted": 0, "review": 0, "hard": 0, "failed": 0}
    track_frames: dict[int, list[str]] = {}
    for fid in frame_ids:
        try:
            frame = resolve_frame(fid)
            result = annotate_frame(
                frame,
                prompts,
                det_model=det,  # type: ignore[arg-type]
                seg_model=seg,  # type: ignore[arg-type]
                det3d=det3d,  # type: ignore[arg-type]
                confidence=conf,
                viz=viz,
                out_dir=out_dir,
            )
            track_ids = tracker.update(result.boxes3d)
            for box, tid in zip(result.boxes3d, track_ids, strict=True):
                box.track_id = tid
                track_frames.setdefault(tid, []).append(fid)
            triage = triage_3d(result.boxes3d)
            build_label_file(frame.frame_id, result.boxes3d, out_dir / "labels")
            write_review_queue(frame, triage, out_dir / "reviews")
            stat["accepted"] += len(triage.accepted)
            stat["review"] += len(triage.review)
            stat["hard"] += len(triage.hard)
            console.print(f"  {fid}: 采纳 {len(triage.accepted)} / 复核 {len(triage.review)} / "
                          f"困难 {len(triage.hard)}")
        except Exception as e:  # 失败隔离：单帧异常不中断序列
            stat["failed"] += 1
            console.print(f"  [red]{fid}: 失败 {e}[/red]")
    # 序列轨迹导出（track_id → 帧/速度历史）
    tracks_json = {
        "sequence": f"{frame_ids[0]}-{frame_ids[-1]}",
        "tracks": [
            {
                "track_id": t.track_id,
                "label": t.label,
                "frames": track_frames.get(t.track_id, []),
                "velocities": t.velocities,
            }
            for t in tracker.tracks() + tracker.completed()
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tracks.json"
    path.write_text(json.dumps(tracks_json, ensure_ascii=False, indent=2))
    console.print(f"[bold]产物[/bold] {path}")
    return stat


dataset_app = typer.Typer(help="管理自建 3D 数据集注册（供 LLM 数据集识别）")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("add")
def dataset_add(
    name: str = typer.Argument(..., help="数据集名（chat 指令中引用的名字）"),
    path: str = typer.Argument(..., help="数据集根目录（必须存在；须符 KITTI 目录结构）"),
    subdirs: str = typer.Option(
        "", "--subdirs", help="关键子目录，逗号分隔（如 training,testing）"
    ),
    task: str = typer.Option("", "--task", help="任务描述（如 3D 检测标注）"),
    note: str = typer.Option("", "--note", help="备注"),
) -> None:
    """注册自建 3D 数据集 → configs/user_datasets.yaml（LLM 数据集识别用）。"""
    from auto3dlabel.configs.datasets import USER_DATASETS_FILE, register_user_dataset

    info = register_user_dataset(
        name,
        path,
        subdirs=[s.strip() for s in subdirs.split(",") if s.strip()],
        task=task,
        note=note,
    )
    console.print(
        f"[green]✓ 已注册[/green] {name} → {info.path}（写入 {USER_DATASETS_FILE}；"
        "须符 KITTI 目录结构 training/{calib,image_2,label_2,velodyne}）"
    )


@dataset_app.command("list")
def dataset_list() -> None:
    """列出已注册的自建 3D 数据集。"""
    from auto3dlabel.configs.datasets import USER_DATASETS_FILE, load_user_datasets

    datasets = load_user_datasets(USER_DATASETS_FILE)
    if not datasets:
        console.print("[yellow]尚未注册自建 3D 数据集（dataset add <name> <path>）[/yellow]")
        return
    table = Table(title="用户自建 3D 数据集")
    table.add_column("名称", style="cyan")
    table.add_column("路径")
    table.add_column("任务")
    table.add_column("备注", style="dim")
    for name, info in sorted(datasets.items()):
        table.add_row(name, str(info.path), info.task, info.note)
    console.print(table)


@dataset_app.command("remove")
def dataset_remove(
    name: str = typer.Argument(..., help="数据集名"),
) -> None:
    """删除已注册的自建 3D 数据集。"""
    from auto3dlabel.configs.datasets import remove_user_dataset

    if remove_user_dataset(name):
        console.print(f"[green]✓ 已删除[/green] {name}")
    else:
        console.print(f"[yellow]数据集 {name} 未注册[/yellow]")


@app.command()
def run(
    target: Annotated[str, typer.Argument(help="帧 ID（000123/123）、范围（003712-003731）或目录")],
    instruction: Annotated[str, typer.Argument(help="自然语言检测指令，如「检测汽车和行人」")],
    det_model: Annotated[
        str | None,
        typer.Option(
            "--det-model",
            "-d",
            help=(
                "检测模型：2D（kitti_finetune/gdino…）或 3D LiDAR"
                f"（{'/'.join(DETECTOR3D_NAMES)}）"
            ),
        ),
    ] = None,
    seg_model: Annotated[
        str | None, typer.Option("--seg-model", "-s", help="分割模型")
    ] = "sam2_l.pt",
    conf: Annotated[float, typer.Option("--conf", "-c", help="置信度阈值")] = 0.3,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="输出目录")] = Path(
        "outputs/kitti3d"
    ),
    batch: Annotated[bool, typer.Option("--batch", help="批量模式（目录/范围输入）")] = False,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="LiDAR 引擎批量帧数（默认自动实测：空闲显存 × 0.92 ÷ 单帧峰值，跑满 GPU）",
        ),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", help="跳过已产出 label 的帧")] = False,
    no_viz: Annotated[bool, typer.Option("--no-viz", help="不生成自检/BEV 图")] = False,
    track3d: Annotated[
        bool,
        typer.Option(
            "--track3d",
            help="多帧跟踪（batch 模式；导出 tracks.json + label 回写 track_id）",
        ),
    ] = False,
) -> None:
    """代码级直跑：检测 → SAM2 → 反投影 → 聚类拟合 → KITTI label + 复核队列（零 LLM）。"""
    _require_auto2dlabel()
    from auto2dlabel.tools.prompts import extract_prompts

    prompts = extract_prompts(instruction) or ["car"]
    console.print(f"[bold]prompts:[/bold] {prompts}  det={det_model or '默认'}  seg={seg_model}")

    if batch and track3d:
        frame_ids = _parse_frame_ids(target)
        console.print(f"跟踪序列 {len(frame_ids)} 帧 → {out_dir}")
        stat = _run_tracked_sequence(
            frame_ids, prompts, det_model, seg_model, conf, out_dir, not no_viz
        )
        console.print(
            f"[bold]汇总:[/bold] 采纳 {stat['accepted']} / 复核 {stat['review']} / "
            f"困难 {stat['hard']} / 失败 {stat['failed']}"
        )
        return

    if batch:
        frame_ids = _parse_frame_ids(target)
        console.print(f"批量 {len(frame_ids)} 帧 → {out_dir}")
        # 模型只加载一次（批量铁律）：LiDAR 引擎走整批 forward，2D 链逐帧复用实例
        det, seg, det3d = _make_models(det_model, seg_model)
        if det3d is not None:
            stat = _run_batch_lidar(
                frame_ids, prompts, det3d, conf, out_dir, not no_viz, resume, batch_size
            )
        else:
            stat = {"accepted": 0, "review": 0, "hard": 0, "failed": 0}
            for fid in frame_ids:
                if resume and (out_dir / "labels" / f"{fid}.txt").is_file():
                    continue
                try:
                    a, r, h = _run_frame(
                        fid, prompts, det, seg, None, conf, out_dir, not no_viz
                    )
                    stat["accepted"] += a
                    stat["review"] += r
                    stat["hard"] += h
                    console.print(f"  {fid}: 采纳 {a} / 复核 {r} / 困难 {h}")
                except Exception as e:  # 失败隔离：单帧异常不中断批量
                    stat["failed"] += 1
                    console.print(f"  [red]{fid}: 失败 {e}[/red]")
        console.print(
            f"[bold]汇总:[/bold] 采纳 {stat['accepted']} / 复核 {stat['review']} / "
            f"困难 {stat['hard']} / 失败 {stat['failed']}"
        )
        return

    frame_id = normalize_frame_id(target)
    det, seg, det3d = _make_models(det_model, seg_model)
    a, r, h = _run_frame(frame_id, prompts, det, seg, det3d, conf, out_dir, not no_viz)
    console.print(f"[bold]{frame_id}:[/bold] 采纳 {a} / 复核 {r} / 困难 {h}")
    console.print(
        f"产物: {out_dir / 'labels' / (frame_id + '.txt')} + "
        f"{out_dir / 'reviews' / (frame_id + '_review.json')}"
    )


@app.command()
def chat(
    instruction: Annotated[
        str, typer.Argument(help="自然语言指令，如「标注 KITTI 帧 000123 中的汽车和行人」")
    ],
    det_model: Annotated[str | None, typer.Option("--det-model", "-d")] = None,
    seg_model: Annotated[str | None, typer.Option("--seg-model", "-s")] = "sam2_l.pt",
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("outputs/kitti3d"),
    provider: Annotated[str, typer.Option("--provider", "-p", help="LLM 服务商")] = "deepseek",
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", help="Agent Loop 最大迭代")
    ] = 3,
    timeout: Annotated[
        int, typer.Option("--timeout", help="对话等待秒数（0 = 不等待）")
    ] = 30,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="跳过对话，缺参直接报错（等价 --timeout 0）")
    ] = False,
) -> None:
    """LLM Agent 闭环：planner 解析 → 3D agent loop → 质量评估 → HITL 三档 → 导出。"""
    _require_auto2dlabel()
    from auto2dlabel.agent.dialog import ask_questions
    from auto2dlabel.agent.llm import create_client
    from auto3dlabel.agent.orchestrator3d import run_3d_agent
    from auto3dlabel.agent.planner3d import TaskPlanner3D
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue
    from auto3dlabel.schema.box3d import Box3D
    from auto3dlabel.tools.device import print_device
    from auto3dlabel.tools.export import ExportTool
    from auto3dlabel.tools.log import log_chat_call

    print_device()  # 设备横幅 + disable_tf32（批量/逐图确定性红线，幂等）
    client = create_client(provider=provider)
    planner = TaskPlanner3D(client)
    wait = 0 if no_wait else timeout
    # v0.3 P1 对话式解析：缺参多轮追问；异常降级单轮 parse（零行为回退）；
    # 单轮仍失败 → 红字 + 退出（对齐 2D chat Step 2）
    try:
        plan = planner.parse_dialog(instruction, ask_fn=partial(ask_questions, timeout=wait))
    except Exception as e:
        console.print(f"[red]对话解析失败，降级单轮解析: {e}[/red]")
        try:
            plan = planner.parse(instruction)
        except Exception as e2:
            console.print(f"[red]解析失败: {e2}[/red]")
            raise typer.Exit(code=1)
    console.print(f"[dim]plan: {plan.summary}[/dim]")
    missing = plan.missing_params  # 规格表驱动（frame_id + prompts，缺参清单直显）
    if missing:
        raise typer.BadParameter(f"缺少参数: {', '.join(missing)}")
    try:
        frame = resolve_frame(plan.frame_id)
    except (ValueError, FileNotFoundError) as e:
        # 帧号非法/帧不存在 → 干净报错，不裸 traceback
        raise typer.BadParameter(str(e))
    det_name = _resolve_det_model(det_model or plan.det_model)

    t0 = time.perf_counter()
    boxes, state = run_3d_agent(
        frame,
        instruction,
        llm_client=client,
        det_model_name=det_name,
        seg_model_name=seg_model or plan.seg_model,
        out_dir=out_dir,
        max_iterations=max_iterations,
    )
    elapsed = time.perf_counter() - t0
    triage = triage_3d(boxes)
    ExportTool().forward(
        annotations=[b.to_dict() for b in boxes],
        output_path=str(out_dir / "labels" / f"{frame.frame_id}.txt"),
        format="kitti",
    )
    write_review_queue(frame, triage, out_dir / "reviews")

    table = Table(title=f"帧 {frame.frame_id} 3D 标注结果")
    table.add_column("档位")
    table.add_column("数量")
    table.add_column("说明")
    table.add_row("采纳", str(len(triage.accepted)), "conf 高且拟合点数充足")
    table.add_row("复核", str(len(triage.review)), "中置信度/拟合质量不足")
    table.add_row("困难", str(len(triage.hard)), "低置信度，建议换模型/人工重标")
    console.print(table)
    if state.metadata.get("quality_report"):
        console.print(f"[dim]质量报告: {state.metadata['quality_report']}[/dim]")
    console.print(f"迭代 {state.iteration} 次，产物: {out_dir / 'labels'} + {out_dir / 'reviews'}")

    # 调用日志（logs/Chat_*.log）：按 detect_objects 工具调用逐条重建（通常 1 条）
    steps_results = []
    for tc in state.tool_calls:
        if tc.get("tool_name") != "detect_objects":
            continue
        result = tc.get("result")
        if not isinstance(result, dict):
            continue
        step_boxes = [
            Box3D.from_dict(item) for item in (result.get("objects") or result.get("data") or [])
            if isinstance(item, dict) and item.get("label")
        ]
        step_triage = triage_3d(step_boxes)
        steps_results.append({
            "step_id": tc.get("tool_call_id"),
            "model": det_name,
            "boxes3d_count": len(step_boxes),
            "triage_summary": {
                "accepted": len(step_triage.accepted),
                "review": len(step_triage.review),
                "hard": len(step_triage.hard),
            },
        })
    log_chat_call(
        instruction=instruction,
        plan_summary={"frame_id": frame.frame_id, "det_model": det_name,
                      "seg_model": seg_model or plan.seg_model},
        steps_results=steps_results,
        elapsed=round(elapsed, 3),
        llm_model=provider,
        annotation_type="kitti_3d",
    )


@app.command("nuscenes-queue")
def nuscenes_queue(
    det_model: Annotated[
        str,
        typer.Option(
            "-d",
            "--det-model",
            help=(
                "3D 检测引擎（LiDAR {'/'.join(DETECTOR3D_NAMES)} / "
                "融合 bevfusion / 单目 fcos3d，三协议自动分派）"
            ),
        ),
    ] = "pointpillars_nus",
    conf: Annotated[float, typer.Option("-c", "--conf", help="置信度阈值")] = DEFAULT_CONF,
    out_dir: Annotated[
        Path, typer.Option("-o", "--out-dir", help="复核队列输出目录")
    ] = DEFAULT_NUSCENES_OUT / "reviews",
    dataroot: Annotated[
        str | None, typer.Option("--dataroot", help="nuScenes dataroot（默认 configs 常量）")
    ] = None,
    version: Annotated[str, typer.Option("--version", help="devkit 版本表")] = "v1.0-mini",
) -> None:
    """nuScenes 端到端复核队列（v0.4 P2）：val 场景逐 sample 检测 → 三档分流。

    产物 {sample_token}_review.json 后接 Web 复核：
    REVIEW3D_DIR=<out_dir> python3 -m auto3dlabel.web.server
    保存即导出 labels/{sample_token}.json（smoke_nuscenes 回灌评测验收）。
    """
    from auto3dlabel.models.detection3d import create_detector3d_any
    from auto3dlabel.tools.nuscenes_pipeline import generate_review_queue

    det = create_detector3d_any(det_model)
    if det is None:
        console.print(f"[red]未知名 3D 模型: {det_model}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]nuScenes 复核队列[/bold] 引擎={det_model} conf={conf} → {out_dir}")
    written = generate_review_queue(
        det, out_dir, conf=conf, dataroot=Path(dataroot) if dataroot else None,
        version=version,
    )
    console.print(
        f"[bold]汇总[/bold] 新生成 {len(written)} 个队列文件（resume：已存在跳过）"
    )


if __name__ == "__main__":
    app()
