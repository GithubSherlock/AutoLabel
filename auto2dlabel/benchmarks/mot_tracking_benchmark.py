#!/usr/bin/env python3
"""MOT Tracking Benchmark — 检测 → ByteTrack 完整管线的 CLEAR MOT 评测。

与 mot_benchmark.py（纯检测口径）互补：本脚本走检测 → ByteTrack 管线，
评测 MOTA/IDF1/IDSW/MT/ML，并同报检测 recall/precision（v0.4 红线：
跟踪评测必须与检测口径同报，避免「跟踪器背检测的锅」）。

GT: gt/gt.txt → frame, id, x, y, w, h, conf, class, visibility
    仅 class=1 (pedestrian) + class=7 (static person)
默认 yolo12n.pt + 单序列连续 20 帧窗口，CPU 可行冒烟：
    python3 auto2dlabel/benchmarks/mot_tracking_benchmark.py
    python3 auto2dlabel/benchmarks/mot_tracking_benchmark.py --max-frames 20 --conf 0.3
GT-as-detection 上界模式（跳过检测，GT 框直喂跟踪器，分离检测/跟踪责任）：
    python3 auto2dlabel/benchmarks/mot_tracking_benchmark.py --gt-as-detection
BoT-SORT 精度档（ReID 外观关联 + ECC 相机运动补偿，CLIP/SigLIP 特征）：
    python3 auto2dlabel/benchmarks/mot_tracking_benchmark.py --bot-sort
    python3 auto2dlabel/benchmarks/mot_tracking_benchmark.py --gt-as-detection --bot-sort
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, time  # noqa: E402
from auto2dlabel.benchmarks.common import (  # noqa: E402
    OUTPUT_DIR,
    build_parser,
    detect_batch_or_fallback,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_mot17_frcnn, ensure_mot20  # noqa: E402
from auto2dlabel.benchmarks.track_eval import (  # noqa: E402
    TrackBox,
    evaluate_tracking,
    load_track_boxes,
)

# MOT class → COCO class（仅行人）
MOT_CLASS_MAP = {1: "person", 7: "person"}


def build_seq_registry() -> dict[str, Path]:
    """{序列名: 序列目录}，MOT17 FRCNN 7 序列 + MOT20 4 序列。"""
    registry: dict[str, Path] = {}
    for seq in sorted(ensure_mot17_frcnn().glob("MOT17-*-FRCNN")):
        registry[seq.name] = seq
    for seq in sorted(ensure_mot20().glob("MOT20-*")):
        registry[seq.name] = seq
    return registry


def consecutive_frames(seq_dir: Path, max_frames: int) -> list[int]:
    """取序列前 max_frames 个连续标注帧号（跟踪需时序连续性，不能均匀采样）。"""
    frames = sorted(load_track_boxes(seq_dir / "gt" / "gt.txt",
                                     keep_classes=set(MOT_CLASS_MAP)).keys())
    return frames[:max_frames] if max_frames > 0 else frames


def run_detection_tracking(
    seq_dir: Path,
    frames: list[int],
    model_name: str,
    conf: float,
    iou: float,
    batch: int | None,
    workers: int | None,
    use_bot_sort: bool = False,
    reid_model_name: str = "openai/clip-vit-base-patch32",
) -> tuple[dict[int, list[TrackBox]], float]:
    """检测（批量）→ ByteTracker/BoT-SORT → {frame: [TrackBox]}，返回 (pred, elapsed)。"""
    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.tracking import (
        BotSORTTracker,
        ByteTracker,
        ReIDModel,
        create_reid_model,
        extract_frame_features,
    )
    from auto2dlabel.schema.annotation import Bbox
    from auto2dlabel.tools.device import resolve_batch_params

    img_paths = [seq_dir / "img1" / f"{f:06d}.jpg" for f in frames]
    missing = [p for p in img_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少帧图像: {missing[0]}")

    model = create_detection_model(model_name, iou_threshold=iou)
    all_cats = ["person"]

    # 批量推理超参：CLI 显式 > 动态实测 > 静态表
    det_batch = getattr(model, "detect_batch", None)
    infer_fn = (
        (lambda ps: det_batch(ps, all_cats, conf, 0))
        if det_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, [str(p) for p in img_paths[:20]],
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    tracker: ByteTracker
    reid_model_inst: ReIDModel | None = None
    if use_bot_sort:
        reid_model_inst = create_reid_model(reid_model_name)
        tracker = BotSORTTracker()
        print(f"跟踪器: BoT-SORT  ReID: {reid_model_name}")
    else:
        tracker = ByteTracker()
    pred: dict[int, list[TrackBox]] = {}

    _t0 = time.time()
    for start in range(0, len(frames), batch_size):
        chunk_paths = [str(p) for p in img_paths[start:start + batch_size]]
        results_per_img = detect_batch_or_fallback(
            model, chunk_paths, all_cats, conf, num_workers,
        )
        for frame, results in zip(frames[start:start + batch_size], results_per_img):
            bboxes = [
                Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                     label=r.label, confidence=r.confidence)
                for r in results
            ]
            if use_bot_sort and reid_model_inst is not None:
                assert isinstance(tracker, BotSORTTracker)
                import numpy as np
                from PIL import Image

                im = Image.open(seq_dir / "img1" / f"{frame:06d}.jpg").convert("RGB")
                feats = extract_frame_features(
                    reid_model_inst, im, bboxes,
                    min_conf=tracker.track_high_thresh)
                tracker.update(
                    bboxes, image=np.asarray(im)[:, :, ::-1], features=feats)
            else:
                tracker.update(bboxes)
            pred[frame] = [
                TrackBox(frame=frame, track_id=b.track_id,
                         x=b.x, y=b.y, w=b.width, h=b.height, conf=b.confidence)
                for b in bboxes if b.track_id is not None
            ]
    elapsed = time.time() - _t0
    return pred, elapsed


def run_gt_tracking(
    gt: dict[int, list[TrackBox]],
    frames: list[int],
    seq_dir: Path | None = None,
    use_bot_sort: bool = False,
    reid_model_name: str = "openai/clip-vit-base-patch32",
) -> tuple[dict[int, list[TrackBox]], float]:
    """GT-as-detection 上界模式：GT 框直接喂跟踪器（conf=1.0），跳过检测。

    理想检测（recall=1、FP=0）下评测 MOTA/IDF1——只反映跟踪器关联能力；
    与 run_detection_tracking（检测+跟踪混合口径）对比可分离「检测的锅/跟踪的锅」。
    use_bot_sort 时从 seq_dir/img1 取帧图注入 ReID 特征 + ECC（seq_dir=None 降级纯 IoU）。
    """
    from auto2dlabel.models.tracking import (
        BotSORTTracker,
        ByteTracker,
        ReIDModel,
        create_reid_model,
        extract_frame_features,
    )
    from auto2dlabel.schema.annotation import Bbox

    tracker: ByteTracker
    reid_model_inst: ReIDModel | None = None
    if use_bot_sort:
        reid_model_inst = create_reid_model(reid_model_name)
        tracker = BotSORTTracker()
        print(f"跟踪器: BoT-SORT  ReID: {reid_model_name}")
    else:
        tracker = ByteTracker()
    pred: dict[int, list[TrackBox]] = {}
    _t0 = time.time()
    for frame in frames:
        bboxes = [
            Bbox(x=b.x, y=b.y, width=b.w, height=b.h,
                 label="person", confidence=1.0)
            for b in gt.get(frame, [])
        ]
        if use_bot_sort and reid_model_inst is not None and seq_dir is not None:
            assert isinstance(tracker, BotSORTTracker)
            import numpy as np
            from PIL import Image

            im = Image.open(seq_dir / "img1" / f"{frame:06d}.jpg").convert("RGB")
            feats = extract_frame_features(
                reid_model_inst, im, bboxes,
                min_conf=tracker.track_high_thresh)
            tracker.update(
                bboxes, image=np.asarray(im)[:, :, ::-1], features=feats)
        else:
            tracker.update(bboxes)
        pred[frame] = [
            TrackBox(frame=frame, track_id=b.track_id,
                     x=b.x, y=b.y, w=b.width, h=b.height, conf=b.confidence)
            for b in bboxes if b.track_id is not None
        ]
    elapsed = time.time() - _t0
    return pred, elapsed


def main() -> None:
    parser = build_parser("MOT Tracking Benchmark (ByteTrack, MOT17+MOT20)")
    parser.set_defaults(model="yolo12n.pt", conf=0.3)
    parser.add_argument("--max-frames", type=int, default=20,
                        help="每序列连续帧窗口（跟踪需时序连续性；0=整序列）")
    parser.add_argument("--seq", type=str, default="MOT17-02-FRCNN",
                        help="序列名（见 registry 打印；MOT17-*-FRCNN / MOT20-*）")
    parser.add_argument("--gt-as-detection", action="store_true",
                        help="上界模式：GT 框直喂跟踪器（跳过检测，分离检测/跟踪责任）")
    parser.add_argument("--bot-sort", action="store_true",
                        help="BoT-SORT 精度档（ReID 外观关联 + ECC 相机运动补偿）")
    parser.add_argument("--reid-model", type=str, default="openai/clip-vit-base-patch32",
                        help="BoT-SORT ReID 特征模型（CLIP/SigLIP，见 model_catalog.REID_MODELS）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    registry = build_seq_registry()
    print(f"MOT 序列注册表 ({len(registry)}): {', '.join(sorted(registry))}")
    seq_dir = registry.get(args.seq)
    if seq_dir is None:
        raise SystemExit(f"未知序列: {args.seq}（可用: {', '.join(sorted(registry))}）")

    frames = consecutive_frames(seq_dir, args.max_frames)
    if not frames:
        raise SystemExit(f"序列 {args.seq} 无有效标注帧")
    print(f"\n序列: {args.seq}  帧窗口: {len(frames)} 帧 (frame {frames[0]}..{frames[-1]})")

    # GT
    gt = load_track_boxes(seq_dir / "gt" / "gt.txt",
                          keep_classes=set(MOT_CLASS_MAP))
    gt = {f: boxes for f, boxes in gt.items() if f in set(frames)}
    num_gt_boxes = sum(len(v) for v in gt.values())
    num_gt_tracks = len({b.track_id for v in gt.values() for b in v})
    print(f"GT: {num_gt_boxes} 个行人框, {num_gt_tracks} 条轨迹")

    # 检测 → ByteTracker/BoT-SORT（或 GT-as-detection 上界模式）
    tracker_name = "BoT-SORT" if args.bot_sort else "ByteTrack"
    if args.gt_as_detection:
        print(f"\n[GT-as-detection 上界模式] 检测=GT（recall=1, FP=0）——"
              f"MOTA/IDF1 只反映跟踪关联能力（{tracker_name}）")
        pred, elapsed = run_gt_tracking(
            gt, frames, seq_dir=seq_dir,
            use_bot_sort=args.bot_sort, reid_model_name=args.reid_model,
        )
    else:
        pred, elapsed = run_detection_tracking(
            seq_dir, frames, args.model, args.conf, args.iou, args.batch, args.workers,
            use_bot_sort=args.bot_sort, reid_model_name=args.reid_model,
        )
    print(f"完成！{elapsed:.1f}s, {len(frames) / elapsed:.1f} frame/s")

    # 评测（红线：MOTA/IDF1 与检测 recall 同报）
    metrics = evaluate_tracking(gt, pred, iou_threshold=0.5)
    print("\n═══ CLEAR MOT 评测 ═══")
    print(f"  {metrics.summary_line()}")
    print("  （recall/precision 为检测口径：recall 低 = 检测的锅；IDSW 高 = 跟踪的锅）")

    source = "gt" if args.gt_as_detection else args.model
    if args.bot_sort:
        source = f"{source}-botsort"
    result_data = {
        "timestamp": ts, "dataset": "MOT17+MOT20 (pedestrian tracking)",
        "model": source, "confidence_threshold": args.conf,
        "detection_source": "gt (GT-as-detection)" if args.gt_as_detection else "model",
        "iou_match_threshold": 0.5, "sequence": args.seq,
        "frame_count": len(frames), "tracker": tracker_name,
        "metrics": metrics.to_dict(),
    }
    json_path, md_path = save_results(result_data, "mot_track", source, ts)
    det_desc = ("GT-as-detection 上界模式" if args.gt_as_detection
                else f"模型 `{args.model}`")
    md_path.write_text("\n".join([
        f"# MOT Tracking Benchmark ({tracker_name})",
        f"- **检测**: {det_desc} | **conf**: {args.conf} | **序列**: {args.seq}",
        f"- **{len(frames)} 帧连续窗口, GT {num_gt_boxes} 框**",
        f"```\n{metrics.summary_line()}\n```",
        "- 检测口径（recall/precision）与跟踪口径（MOTA/IDF1/IDSW/MT/ML）同报",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
