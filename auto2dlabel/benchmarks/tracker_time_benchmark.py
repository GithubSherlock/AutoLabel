#!/usr/bin/env python3
"""ByteTracker 纯跟踪耗时微基准 —— 密集场景单帧 update 耗时（不含检测）。

mot_tracking_benchmark 的 0.1 img/s 是检测主导，跟踪器自身成本被掩盖；
本基准隔离跟踪器：合成随机游走检测框（确定性 seed）直喂 ByteTracker，
密度扫 10/25/50/100/200 框/帧——匈牙利匹配（scipy linear_sum_assignment，O(n³)）
是主导项，MOT20 量级密集场景（100+/帧）是否可承受由本基准定量回答。

用法:
    python3 auto2dlabel/benchmarks/tracker_time_benchmark.py               # 默认全密度档
    python3 auto2dlabel/benchmarks/tracker_time_benchmark.py --boxes 10 100
    python3 auto2dlabel/benchmarks/tracker_time_benchmark.py --frames 100
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import time  # noqa: E402
from auto2dlabel.models.tracking import ByteTracker  # noqa: E402
from auto2dlabel.schema.annotation import Bbox  # noqa: E402


def synthetic_frames(
    n_boxes: int, n_frames: int, seed: int = 0,
) -> list[list[Bbox]]:
    """合成随机游走检测帧序列（确定性：固定 seed 同输出）。

    n_boxes 个目标在 1000×1000 画布上随机游走（≤5px/帧），
    10% 帧间消失（模拟遮挡，激活 track_buffer 恢复路径），
    置信度均匀 [0.4, 1.0]（高低分两段关联均激活）。
    """
    import random

    rng = random.Random(seed)
    objects = [
        [rng.uniform(50.0, 950.0), rng.uniform(50.0, 950.0)]
        for _ in range(n_boxes)
    ]
    frames: list[list[Bbox]] = []
    for _ in range(n_frames):
        boxes: list[Bbox] = []
        for obj in objects:
            if rng.random() < 0.1:  # 模拟遮挡：本帧消失
                continue
            obj[0] += rng.uniform(-5.0, 5.0)
            obj[1] += rng.uniform(-5.0, 5.0)
            boxes.append(Bbox(
                x=obj[0] - 25.0, y=obj[1] - 25.0, width=50.0, height=50.0,
                label="person", confidence=rng.uniform(0.4, 1.0),
            ))
        frames.append(boxes)
    return frames


def measure_update_time(
    frames: list[list[Bbox]], repeats: int = 3,
) -> tuple[float, float, int]:
    """逐帧 update 计时（每轮新 tracker，取最短一轮抗调度噪声）。

    Returns:
        (总秒数, 均值 ms/帧, 总框数)。
    """
    best = float("inf")
    for _ in range(repeats):
        tracker = ByteTracker()
        t0 = time.perf_counter()
        for boxes in frames:
            tracker.update(boxes)
        best = min(best, time.perf_counter() - t0)
    return best, best / len(frames) * 1000.0, sum(len(b) for b in frames)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ByteTracker 纯跟踪耗时微基准（合成检测，不含检测推理）")
    parser.add_argument("--boxes", type=int, nargs="+",
                        default=[10, 25, 50, 100, 200],
                        help="每帧目标数档位（默认 10 25 50 100 200）")
    parser.add_argument("--frames", type=int, default=200,
                        help="每档帧数（默认 200）")
    parser.add_argument("--seed", type=int, default=0,
                        help="合成数据随机种子（默认 0，确定性可复现）")
    args = parser.parse_args()

    print(f"ByteTracker 纯跟踪耗时微基准（seed={args.seed}, {args.frames} 帧/档, "
          f"3 轮取最短）")
    print(f"{'目标数/帧':>10} {'总框数':>8} {'均值 ms/帧':>12} {'µs/框':>10}")
    for n_boxes in args.boxes:
        frames = synthetic_frames(n_boxes, args.frames, seed=args.seed)
        total, ms_per_frame, n_total = measure_update_time(frames)
        print(f"{n_boxes:>10} {n_total:>8} {ms_per_frame:>12.3f} "
              f"{total / n_total * 1e6:>10.2f}", flush=True)
    print("（匈牙利匹配 O(n³) 主导；MOT20 量级密集场景 100+/帧 见最右两档）",
          flush=True)


if __name__ == "__main__":
    main()
