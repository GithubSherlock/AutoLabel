"""CLI 公共层 —— console/logger 与 run/chat 命令共用的辅助函数。"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from auto2dlabel.agent.state import AgentState

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def collect_images(path: Path, batch: bool) -> list[Path]:
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


def visualize_results(state: AgentState, img_path: Path, timestamp: str = "") -> None:
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


def display_results(state: AgentState) -> None:
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
        has_track = any(b.track_id is not None for b in ann.bboxes)
        if has_track:
            table.add_column("Track", style="magenta")
        table.add_column("Status", style="red")

        for i, bbox in enumerate(ann.bboxes, 1):
            bbox_str = f"[{bbox.x:.0f}, {bbox.y:.0f}, {bbox.width:.0f}, {bbox.height:.0f}]"
            conf_str = f"{bbox.confidence:.2%}"
            status = "⚠ REVIEW" if i - 1 in ann.review_flags else "✓"
            row = [str(i), bbox.label, bbox_str, conf_str]
            if has_track:
                row.append(str(bbox.track_id) if bbox.track_id is not None else "-")
            row.append(status)
            table.add_row(*row)

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


def triage_and_export(
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
