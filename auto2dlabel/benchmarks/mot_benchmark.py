#!/usr/bin/env python3
"""MOT Detection Benchmark — 密集行人检测（MOT17 FRCNN + MOT20）。

MOT（Multiple Object Tracking）是多目标跟踪基准数据集，
取其 GT 标注用于评估 YOLO 在密集人群场景中的行人检测能力。

MOT17: 7 个 FRCNN 序列（商场/街道），~5300 张图
MOT20: 4 个序列（极密集场景），~8900 张图

GT: gt/gt.txt → frame, id, x, y, w, h, conf, class, visibility
     仅 class=1 (pedestrian) + class=7 (static person) 映射为 "person"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    build_parser,
    detect_batch_or_fallback,
    evaluate_per_class,
    format_result_table,
    run_detection_sahi,
    sample_image_paths,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_mot17_frcnn, ensure_mot20

# ── 配置 ──────────────────────────────────────────────────────
# MOT class → COCO class（仅行人）
MOT_CLASS_MAP = {1: "person", 7: "person"}


# ================================================================
# 1. GT 加载
# ================================================================

def load_mot_ground_truth(
    seq_dirs: list[Path], max_images: int = 0,
) -> dict[int, dict[str, Any]]:
    """加载 MOT GT，跨序列均匀采样。

    GT 格式: frame, id, x, y, w, h, conf, class, visibility
    过滤: conf=0（非活跃标注）、非行人类别。

    Args:
        seq_dirs: 序列目录列表。
        max_images: 总图片数上限（0 = 全部）。

    Returns:
        {idx: {"file_name": "MOT17-02-FRCNN/000001.jpg", "objects": [...]}}
    """
    # 收集所有 (seq_dir, frame) 对
    all_frames: list[tuple[Path, int, int]] = []  # [(seq_dir, frame_num, num_objects), ...]
    for seq_dir in sorted(seq_dirs):
        gt_file = seq_dir / "gt" / "gt.txt"
        if not gt_file.exists():
            continue
        img_dir = seq_dir / "img1"
        # 统计每帧的标注数
        frame_counts: dict[int, int] = {}
        for line in gt_file.read_text().strip().splitlines():
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame = int(parts[0])
            cls_id = int(parts[7])
            conf = int(parts[6])
            if conf == 0:
                continue
            if cls_id not in MOT_CLASS_MAP:
                continue
            frame_counts[frame] = frame_counts.get(frame, 0) + 1

        # 筛选有标注且图片存在的帧
        for frame in sorted(frame_counts.keys()):
            img_name = f"{frame:06d}.jpg"
            if (img_dir / img_name).exists():
                all_frames.append((seq_dir, frame, frame_counts[frame]))

    if not all_frames:
        raise RuntimeError("未找到任何有效帧")

    # 均匀采样
    if max_images > 0 and len(all_frames) > max_images:
        step = len(all_frames) / max_images
        sampled = [all_frames[int(i * step)] for i in range(max_images)]
    else:
        sampled = all_frames

    # 加载 GT
    gt: dict[int, dict[str, Any]] = {}
    skipped = 0

    for i, (seq_dir, frame, _) in enumerate(sampled):
        img_name = f"{frame:06d}.jpg"
        img_path = seq_dir / "img1" / img_name
        if not img_path.exists():
            skipped += 1
            continue

        gt_file = seq_dir / "gt" / "gt.txt"
        objects = []
        for line in gt_file.read_text().strip().splitlines():
            parts = line.split(",")
            if len(parts) < 9:
                continue
            f = int(parts[0])
            if f != frame:
                continue
            cls_id = int(parts[7])
            conf = int(parts[6])
            if conf == 0:
                continue
            coco_name = MOT_CLASS_MAP.get(cls_id)
            if coco_name is None:
                continue

            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            objects.append({"name": coco_name, "bbox": [x, y, x + w, y + h]})

        # 相对于 DATASETS_ROOT 的相对路径
        rel_path = str(img_path.relative_to(Path.home() / "autodl-tmp" / "Documents" / "datasets"))
        gt[i] = {"file_name": rel_path, "objects": objects}

    if skipped:
        print(f"跳过 {skipped} 张缺少图片的标注")
    return gt


# ================================================================
# 2. 检测
# ================================================================

def run_detection(
    gt: dict[int, dict[str, Any]],
    model_name: str,
    conf: float,
    iou: float,
    batch: int | None = None,
    workers: int | None = None,
) -> dict[int, list[dict]]:
    """对 GT 中所有图像运行检测（批量推理 + OOM 降级逐图）。"""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.tools.device import resolve_batch_params

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"检测类别 ({len(all_cats)}): {', '.join(all_cats)}")

    datasets_root = Path.home() / "autodl-tmp" / "Documents" / "datasets"
    model = create_detection_model(model_name, iou_threshold=iou)
    predictions: dict[int, list[dict]] = {}

    # 批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    # （gt.file_name 为相对 DATASETS_ROOT 的路径，探针基目录用 datasets_root）
    _det_batch = getattr(model, "detect_batch", None)
    infer_fn = (
        (lambda ps: _det_batch(ps, all_cats, conf, 0))
        if _det_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, sample_image_paths(gt, datasets_root),
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(gt)
    img_ids = list(gt.keys())
    n_batches = (total + batch_size - 1) // batch_size
    iterator = tqdm(range(0, total, batch_size), total=n_batches,
                    desc=f"检测（{model_name} b={batch_size}）", unit="batch") \
        if has_tqdm else range(0, total, batch_size)

    _t0 = time.time()
    for start in iterator:
        # 分块（缺失文件过滤，chunk_ids 与 chunk_paths 保序对齐）
        chunk_ids: list[int] = []
        chunk_paths: list[str] = []
        for img_id in img_ids[start:start + batch_size]:
            img_path = datasets_root / gt[img_id]["file_name"]
            if not img_path.exists():
                continue
            chunk_ids.append(img_id)
            chunk_paths.append(str(img_path))
        if not chunk_paths:
            continue

        results_per_img = detect_batch_or_fallback(
            model, chunk_paths, all_cats, conf, num_workers,
        )
        for img_id, results in zip(chunk_ids, results_per_img):
            predictions[img_id] = [
                {"name": r.label, "bbox": [r.x, r.y, r.x + r.width, r.y + r.height], "conf": r.confidence}
                for r in results
            ]

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


# ================================================================
# 3. 主流程
# ================================================================

def main():
    parser = build_parser("MOT Pedestrian Detection Benchmark (MOT17+MOT20)")
    parser.set_defaults(model="yolo26x.pt")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    mot17_root = ensure_mot17_frcnn()
    mot20_root = ensure_mot20()

    mot17_seqs = sorted(mot17_root.glob("MOT17-*-FRCNN"))
    mot20_seqs = sorted(mot20_root.glob("MOT20-*"))
    all_seqs = mot17_seqs + mot20_seqs
    print(f"MOT17 FRCNN: {len(mot17_seqs)} 个序列")
    print(f"MOT20: {len(mot20_seqs)} 个序列")

    # 加载 GT
    print("加载 MOT Ground Truth (仅行人: class 1 + 7) ...")
    gt = load_mot_ground_truth(all_seqs, max_images=args.max_images or 0)

    # 统计
    total_persons = sum(len(g["objects"]) for g in gt.values())
    print(f"已加载 {len(gt)} 张图像, {total_persons} 个行人标注\n")

    # 检测（SAHI 切片路径不支持批量，仅标准路径接入）
    datasets_root = Path.home() / "autodl-tmp" / "Documents" / "datasets"
    if args.sahi:
        predictions = run_detection_sahi(gt, datasets_root, args.model, args.conf, args.iou)
    else:
        predictions = run_detection(gt, args.model, args.conf, args.iou,
                                    batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 bbox，序列子目录保留）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("mot", gt, predictions,
                          lambda img_id, info: datasets_root / info["file_name"])
        print()

    # 评估
    print(f"计算指标（IoU@{IOU_MATCH_THRESHOLD}）...\n")
    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    results = {}
    for cls in all_cats:
        results[cls] = evaluate_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD)

    summary = format_result_table(results, all_cats, top_n=args.top_classes)
    print(summary)
    print()

    # 保存
    mAP = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))
    result_data = {
        "timestamp": ts, "dataset": "MOT17+MOT20 (pedestrian detection)",
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "person_count": total_persons,
        "sequences": {
            "mot17_frcnn": [s.name for s in mot17_seqs],
            "mot20": [s.name for s in mot20_seqs],
        },
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "mot", args.model, ts)
    md_path.write_text("\n".join([
        "# MOT Pedestrian Detection Benchmark",
        f"- **模型**: {args.model} | **conf**: {args.conf} | **mAP@0.5**: {mAP:.4f}",
        f"- **{len(gt)} 张图, {total_persons} 个行人实例**",
        f"- **MOT17 FRCNN ({len(mot17_seqs)} 序列) + MOT20 ({len(mot20_seqs)} 序列)**",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
