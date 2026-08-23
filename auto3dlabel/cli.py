"""Auto3dLabel CLI：run（代码级直跑，零 LLM）+ chat（LLM planner → 3D agent loop → HITL）。

用法：
    auto3dlabel run 000123 "检测汽车和行人" --det-model kitti_finetune
    auto3dlabel run 003712-003731 "检测汽车" --batch --resume
    auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from auto3dlabel.configs.kitti import DEFAULT_KITTI_ROOT
from auto3dlabel.data.kitti import normalize_frame_id, resolve_frame

app = typer.Typer(help="Agentic 3D 标注（KITTI 单帧 → 3D bbox 初稿 + HITL 三档）")
console = Console()


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


def _run_frame(
    frame_id: str,
    prompts: list[str],
    det_model: str | None,
    seg_model: str | None,
    conf: float,
    out_dir: Path,
    viz: bool,
) -> tuple[int, int, int]:
    """单帧跑管线 + 导出产物 → (accepted, review, hard)。"""
    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model

    from auto3dlabel.export.kitti_label import build_label_file
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue
    from auto3dlabel.tools.pipeline import annotate_frame

    frame = resolve_frame(frame_id)
    det = create_detection_model(_resolve_det_model(det_model))
    seg = create_segmentation_model(seg_model or "sam2_l.pt")
    result = annotate_frame(
        frame, prompts, det_model=det, seg_model=seg, confidence=conf, viz=viz, out_dir=out_dir
    )
    triage = triage_3d(result.boxes3d)
    build_label_file(frame.frame_id, result.boxes3d, out_dir / "labels")
    write_review_queue(frame, triage, out_dir / "reviews")
    return len(triage.accepted), len(triage.review), len(triage.hard)


@app.command()
def run(
    target: Annotated[str, typer.Argument(help="帧 ID（000123/123）、范围（003712-003731）或目录")],
    instruction: Annotated[str, typer.Argument(help="自然语言检测指令，如「检测汽车和行人」")],
    det_model: Annotated[
        str | None,
        typer.Option("--det-model", "-d", help="2D 检测模型（kitti_finetune=KITTI 微调权重）"),
    ] = None,
    seg_model: Annotated[
        str | None, typer.Option("--seg-model", "-s", help="分割模型")
    ] = "sam2_l.pt",
    conf: Annotated[float, typer.Option("--conf", "-c", help="置信度阈值")] = 0.3,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="输出目录")] = Path(
        "outputs/kitti3d"
    ),
    batch: Annotated[bool, typer.Option("--batch", help="批量模式（目录/范围输入）")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="跳过已产出 label 的帧")] = False,
    no_viz: Annotated[bool, typer.Option("--no-viz", help="不生成自检/BEV 图")] = False,
) -> None:
    """代码级直跑：检测 → SAM2 → 反投影 → 聚类拟合 → KITTI label + 复核队列（零 LLM）。"""
    from auto2dlabel.tools.prompts import extract_prompts

    prompts = extract_prompts(instruction) or ["car"]
    console.print(f"[bold]prompts:[/bold] {prompts}  det={det_model or '默认'}  seg={seg_model}")

    if batch:
        frame_ids = _parse_frame_ids(target)
        console.print(f"批量 {len(frame_ids)} 帧 → {out_dir}")
        stat = {"accepted": 0, "review": 0, "hard": 0, "failed": 0}
        for fid in frame_ids:
            if resume and (out_dir / "labels" / f"{fid}.txt").is_file():
                continue
            try:
                a, r, h = _run_frame(fid, prompts, det_model, seg_model, conf, out_dir, not no_viz)
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
    a, r, h = _run_frame(frame_id, prompts, det_model, seg_model, conf, out_dir, not no_viz)
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
) -> None:
    """LLM Agent 闭环：planner 解析 → 3D agent loop → 质量评估 → HITL 三档 → 导出。"""
    from auto2dlabel.agent.llm import create_client

    from auto3dlabel.agent.orchestrator3d import run_3d_agent
    from auto3dlabel.agent.planner3d import TaskPlanner3D
    from auto3dlabel.export.kitti_label import build_label_file
    from auto3dlabel.export.review_queue import triage_3d, write_review_queue

    client = create_client(provider=provider)
    planner = TaskPlanner3D(client)
    plan = planner.parse(instruction)
    console.print(f"[dim]plan: {plan.summary}[/dim]")
    if not plan.frame_id:
        raise typer.BadParameter("指令中未指定 KITTI 帧 ID")
    frame = resolve_frame(plan.frame_id)
    det_name = _resolve_det_model(det_model or plan.det_model)

    boxes, state = run_3d_agent(
        frame,
        instruction,
        llm_client=client,
        det_model_name=det_name,
        seg_model_name=seg_model or plan.seg_model,
        out_dir=out_dir,
        max_iterations=max_iterations,
    )
    triage = triage_3d(boxes)
    build_label_file(frame.frame_id, boxes, out_dir / "labels")
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


if __name__ == "__main__":
    app()
