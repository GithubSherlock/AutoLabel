"""KITTI 20 帧冒烟：LiDAR 直检引擎 → 双口径 AP 出表（M3 执行器，真实权重）。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_kitti [start-end] [model] [conf] [config] [checkpoint]

官方口径（40-point + 每类 IoU）为主表，11-point 口径（IoU 0.5）为对照表
（与 v0.1 反投影基线同口径可比）。仅做整体评测，不改任何测试代码。

P3 微调对比：config + checkpoint 提供时加载自定义权重（kitti3d_finetune
微调产物），与官方权重同 spec 同 conf 同函数出表——提升/持平/退化如实记录。
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from auto3dlabel.benchmarks.kitti3d_benchmark import (
    format_benchmark_table,
    run_kitti3d_benchmark,
)
from auto3dlabel.benchmarks.kitti_official_ap import run_kitti_official
from auto3dlabel.configs.kitti import DEFAULT_CONF, KITTI_EVAL_CLASSES
from auto3dlabel.data.kitti import frame_ids_by_range, resolve_frame
from auto3dlabel.models.detection3d import Detector3D, Mmdet3dDetector, create_detector3d
from auto3dlabel.tools.pipeline import annotate_frame


def _format_official_table(result: dict[str, dict[str, dict[str, Any]]]) -> str:
    """官方 40-point 口径表格渲染（难度 × 类 → AP %）。"""
    buf = io.StringIO()
    console = Console(file=buf, record=True, width=110, force_terminal=False)
    table = Table(title="KITTI 官方口径（40-point AP %，每类官方 IoU）", show_lines=True)
    table.add_column("难度", style="bold")
    for cls in KITTI_EVAL_CLASSES:
        table.add_column(cls, justify="right")
    for diff, classes in result.items():
        row = [diff]
        for cls in KITTI_EVAL_CLASSES:
            stats = classes[cls]
            ap = float(stats["ap"]) * 100
            row.append(f"{ap:5.1f} ({stats['gt_count']})")
        table.add_row(*row)
    console.print(table)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    spec = argv[0] if len(argv) > 0 else "003712-003731"
    model = argv[1] if len(argv) > 1 else "pointpillars_kitti"
    conf = float(argv[2]) if len(argv) > 2 else DEFAULT_CONF
    config_path = argv[3] if len(argv) > 3 else ""
    checkpoint_path = argv[4] if len(argv) > 4 else ""

    det3d: Detector3D | None
    if config_path:
        # P3 微调对比：自定义 config/checkpoint（kitti3d_finetune 产物），
        # Mmdet3dDetector 直构（detect_points 协议同 catalog 模型）
        if not checkpoint_path:
            raise SystemExit("提供 config 时必须提供 checkpoint")
        det3d = Mmdet3dDetector(config_path, checkpoint_path)
        tag = "finetune"
    else:
        det3d = create_detector3d(model)
        if det3d is None:
            raise SystemExit(f"未知名 3D 模型：{model}")
        tag = model
    start_s, end_s = (int(p) for p in spec.split("-"))
    frames = [resolve_frame(fid) for fid in frame_ids_by_range(start_s, end_s)]
    console = Console()
    console.print(f"[bold]冒烟[/bold] {spec}（{len(frames)} 帧）model={tag} conf={conf}")

    predictions: dict[str, list[object]] = {}
    t0 = time.perf_counter()
    n_boxes = 0
    for frame in frames:
        result = annotate_frame(
            frame, ["car", "person", "bicycle"], det3d=det3d, confidence=conf,
            viz=False,
        )
        predictions[frame.frame_id] = list(result.boxes3d)
        n_boxes += len(result.boxes3d)
    elapsed = time.perf_counter() - t0
    console.print(
        f"[bold]耗时[/bold] {elapsed:.1f}s（{elapsed / len(frames):.2f}s/帧）"
        f"共 {n_boxes} 个 3D 框"
    )

    official = run_kitti_official(frames, predictions)
    legacy = run_kitti3d_benchmark(frames, predictions)
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / f"smoke_{tag}_{spec}.txt"
    text = _format_official_table(official) + "\n" + format_benchmark_table(
        legacy, title="KITTI 11-point 对照口径（AP %，IoU 0.5，v0.1 基线同口径）"
    )
    out.write_text(text)
    console.print(text)
    console.print(f"[bold]产物[/bold] {out}")


if __name__ == "__main__":
    main()
