"""KITTI 20 帧跟踪冒烟（M5 执行器，真实权重）：PointPillars 检测 → Tracker3D 序列跟踪。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_track3d [start-end] [model] [conf]

统计：每帧检测框数 / 匈牙利匹配率 / 轨迹数 / 轨迹长分布 / velocity 输出。
IDSW 不可真算（KITTI object 无 GT track id，tracking 版 label_02 才有）——如实记录。
仅做整体统计，不改任何测试代码。
"""

from __future__ import annotations

import io
import sys
import time
from collections import Counter

from rich.console import Console
from rich.table import Table

from auto3dlabel.configs.kitti import DEFAULT_CONF
from auto3dlabel.data.kitti import frame_ids_by_range, resolve_frame
from auto3dlabel.models.detection3d import create_detector3d
from auto3dlabel.tools.track3d import Tracker3D


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    spec = argv[0] if len(argv) > 0 else "003712-003731"
    model = argv[1] if len(argv) > 1 else "pointpillars_kitti"
    conf = float(argv[2]) if len(argv) > 2 else DEFAULT_CONF

    det3d = create_detector3d(model)
    if det3d is None:
        raise SystemExit(f"未知名 3D 模型：{model}")
    start_s, end_s = (int(p) for p in spec.split("-"))
    frames = [resolve_frame(fid) for fid in frame_ids_by_range(start_s, end_s)]
    console = Console()
    console.print(f"[bold]跟踪冒烟[/bold] {spec}（{len(frames)} 帧）model={model} conf={conf}")

    tracker = Tracker3D()
    total_boxes = 0
    total_matched = 0
    t0 = time.perf_counter()
    for frame in frames:
        results = det3d.detect(frame, conf_threshold=conf)
        boxes = [r.to_box3d() for r in results]
        active_before = {t.track_id for t in tracker.tracks()}
        track_ids = tracker.update(boxes)
        # 匹配 = 关联到已有轨迹（含 missed 未删轨迹）；新 id = 本帧新建
        n_matched = sum(1 for tid in track_ids if tid in active_before)
        total_boxes += len(boxes)
        total_matched += n_matched
    elapsed = time.perf_counter() - t0

    tracks = tracker.tracks() + tracker.completed()  # 全量轨迹（含超龄删除）
    length_counter: Counter[int] = Counter(len(t.history) for t in tracks)
    n_with_velocity = sum(1 for t in tracks if t.velocities)

    buf = io.StringIO()
    table_console = Console(file=buf, record=True, width=110, force_terminal=False)
    table = Table(title="KITTI 20 帧跟踪冒烟（BEV IoU 匈牙利 + 恒速 Kalman）", show_lines=True)
    table.add_column("统计项", style="bold")
    table.add_column("数值", justify="right")
    table.add_row("帧数", str(len(frames)))
    table.add_row("总检测框", str(total_boxes))
    table.add_row(
        "匹配率",
        f"{total_matched}/{total_boxes}（{total_matched / max(total_boxes, 1) * 100:.1f}%）",
    )
    table.add_row("轨迹总数", str(len(tracks)))
    table.add_row(
        "轨迹长分布", ", ".join(f"长度 {k} × {v}" for k, v in sorted(length_counter.items()))
    )
    table.add_row("有速度输出轨迹", str(n_with_velocity))
    table.add_row("总耗时", f"{elapsed:.1f}s（{elapsed / max(len(frames), 1):.2f}s/帧）")
    table_console.print(table)
    console.print(buf.getvalue())


if __name__ == "__main__":
    main()
