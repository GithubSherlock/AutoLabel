"""KITTI 20 帧冒烟：单目引擎（pgd）→ 双口径 AP 出表（P3 执行器，真实权重）。

用法：
    python3 -m auto3dlabel.benchmarks.smoke_kitti_mono [start-end] [model] [conf]
        [--lidar pointpillars_kitti]

单目（默认，LiDAR 不可用降级方案 + 交叉验证基准）：pgd_kitti；
--lidar 时同帧并列跑 LiDAR 引擎（annotate_frame 管线），两套表对照。
官方口径（40-point + 每类 IoU）为主表，11-point 口径（IoU 0.5）为对照表；
另附 L-W sanity（单目预测 vs GT 的 h/w/l 中位数，检测 coder 顺序反转）。
仅做整体评测，不改任何测试代码。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from auto3dlabel.benchmarks.kitti3d_benchmark import (
    format_benchmark_table,
    run_kitti3d_benchmark,
)
from auto3dlabel.benchmarks.kitti_official_ap import run_kitti_official
from auto3dlabel.benchmarks.smoke_kitti import _format_official_table
from auto3dlabel.configs.kitti import DEFAULT_CONF, KITTI_EVAL_CLASSES
from auto3dlabel.data.kitti import frame_ids_by_range, resolve_frame
from auto3dlabel.models.detection3d import create_detector3d
from auto3dlabel.models.mono3d import create_mono3d_detector
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tools.pipeline import annotate_frame


def _dims_summary(boxes: list[Box3D], cls: str) -> str | None:
    """该类 h/w/l 中位数摘要（L-W sanity：coder 顺序反转时与 GT 明显相悖）。"""
    vals = np.asarray([[b.h, b.w, b.l] for b in boxes if b.label == cls])
    if len(vals) == 0:
        return None
    med = np.median(vals, axis=0)
    return f"h={med[0]:.2f} w={med[1]:.2f} l={med[2]:.2f} (n={len(vals)})"


def _format_sanity(
    frames: list[Any], mono_pred: dict[str, list[Box3D]]
) -> str:
    """L-W sanity 表：GT vs 单目预测的每类 h/w/l 中位数。"""
    lines = ["L-W sanity（h/w/l 中位数，米）：", "  类          GT                     mono"]
    for cls in KITTI_EVAL_CLASSES:
        gt_boxes = [b for f in frames for b in f.load_gt3d() if b.label == cls]
        pred_boxes = [b for boxes in mono_pred.values() for b in boxes]
        gt_s = _dims_summary(gt_boxes, cls)
        pd_s = _dims_summary(pred_boxes, cls)
        lines.append(
            f"  {cls:<10} {gt_s or '—':<24} {pd_s or '—'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="?", default="003712-003731")
    parser.add_argument("model", nargs="?", default="pgd_kitti")
    parser.add_argument("conf", nargs="?", type=float, default=DEFAULT_CONF)
    parser.add_argument(
        "--lidar", default=None, help="同帧并列 LiDAR 引擎（如 pointpillars_kitti）"
    )
    args = parser.parse_args(argv)

    det = create_mono3d_detector(args.model)
    if det is None:
        raise SystemExit(f"未知名单目 3D 模型：{args.model}")
    det3d = create_detector3d(args.lidar) if args.lidar else None
    if args.lidar and det3d is None:
        raise SystemExit(f"未知名 LiDAR 3D 模型：{args.lidar}")

    start_s, end_s = (int(p) for p in args.spec.split("-"))
    frames = [resolve_frame(fid) for fid in frame_ids_by_range(start_s, end_s)]
    console = Console()
    console.print(
        f"[bold]冒烟[/bold] {args.spec}（{len(frames)} 帧）"
        f"mono={args.model} conf={args.conf}"
        + (f" lidar={args.lidar}" if args.lidar else "")
    )

    mono_pred: dict[str, list[Box3D]] = {}
    lidar_pred: dict[str, list[Box3D]] = {}
    t0 = time.perf_counter()
    n_mono = 0
    for frame in frames:
        dets = det.detect(frame, conf_threshold=args.conf)
        mono_pred[frame.frame_id] = [d.to_box3d() for d in dets]
        n_mono += len(dets)
        if det3d is not None:
            result = annotate_frame(
                frame, ["car", "person", "bicycle"], det3d=det3d,
                confidence=args.conf, viz=False,
            )
            lidar_pred[frame.frame_id] = list(result.boxes3d)
    elapsed = time.perf_counter() - t0
    console.print(
        f"[bold]耗时[/bold] {elapsed:.1f}s（{elapsed / len(frames):.2f}s/帧）"
        f"共 {n_mono} 个单目 3D 框"
    )

    mono_official = run_kitti_official(frames, mono_pred)
    mono_legacy = run_kitti3d_benchmark(frames, mono_pred)
    text = _format_official_table(mono_official) + "\n" + format_benchmark_table(
        mono_legacy, title="KITTI 11-point 对照口径（AP %，IoU 0.5，v0.1 基线同口径）"
    )
    if lidar_pred:
        lidar_official = run_kitti_official(frames, lidar_pred)
        lidar_legacy = run_kitti3d_benchmark(frames, lidar_pred)
        text += (
            "\n\n===== LiDAR 引擎同帧对照 =====\n"
            + _format_official_table(lidar_official)
            + "\n"
            + format_benchmark_table(
                lidar_legacy,
                title="KITTI 11-point 对照口径（AP %，IoU 0.5）",
            )
        )
    text += "\n\n" + _format_sanity(frames, mono_pred)

    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / f"smoke_{args.model}_{args.spec}.txt"
    out.write_text(text)
    console.print(text)
    console.print(f"[bold]产物[/bold] {out}")


if __name__ == "__main__":
    main()
