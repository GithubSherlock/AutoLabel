#!/usr/bin/env python3
"""KITTI object Detection Benchmark — 自动标注 vs Ground Truth。

KITTI 自动驾驶场景：8 类目标（Car, Pedestrian, Cyclist 等），
纯文本标注格式，7,481 张训练图（默认用前 300 张）。

GT: training/label_2/*.txt → {name, bbox: [x1,y1,x2,y2]}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, json, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    build_parser,
    compute_iou,
    detect_batch_or_fallback,
    evaluate_per_class,
    format_result_table,
    sample_image_paths,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_kitti, DATASETS_ROOT

# ── 配置 ──────────────────────────────────────────────────────
KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Tram", "Person_sitting"]
KITTI_TO_COCO = {
    "Car": "car", "Van": "car",
    "Pedestrian": "person", "Person_sitting": "person",
    "Cyclist": "bicycle",
    "Truck": "truck",
    "Tram": "train",
}
# Misc / DontCare → 忽略


# ================================================================
# 1. GT 加载
# ================================================================

def kitti_difficulty(truncated: float, occluded: int, height: float) -> str | None:
    """KITTI 官方 difficulty 判据（height = bbox 像素高度 y2-y1）。

    - easy:     h ≥ 40 且 trunc ≤ 0.15 且 occ == 0
    - moderate: h ≥ 25 且 trunc ≤ 0.3  且 occ ≤ 1
    - hard:     h ≥ 25 且 trunc ≤ 0.5  且 occ ≤ 2
    - 其余（过小/遮挡过重/截断过重）→ None（不参与评测）
    """
    if height >= 40 and truncated <= 0.15 and occluded == 0:
        return "easy"
    if height >= 25 and truncated <= 0.3 and occluded <= 1:
        return "moderate"
    if height >= 25 and truncated <= 0.5 and occluded <= 2:
        return "hard"
    return None


def filter_gt_by_difficulty(
    gt: dict[int, dict[str, Any]], difficulty: str,
) -> dict[int, dict[str, Any]]:
    """按难度档过滤 GT（对象须带 difficulty 标签；无该档对象的图保留空列表）。

    预测不按难度过滤（检测结果无难度语义）——分层评测 = 分层 GT × 全量预测，
    与 KITTI 官方口径一致。
    """
    return {
        idx: {
            "file_name": info["file_name"],
            "objects": [o for o in info["objects"] if o.get("difficulty") == difficulty],
        }
        for idx, info in gt.items()
    }


def load_kitti_ground_truth(
    image_dir: Path,
    label_dir: Path,
    max_images: int = 0,
    difficulty: str | None = None,
) -> dict[int, dict[str, Any]]:
    """加载 KITTI 标注。

    KITTI txt 格式每行:
      class truncated occluded alpha x1 y1 x2 y2 h w l x y z rot_y

    difficulty: None = 全量（每个对象带 difficulty 标签）；指定 easy/moderate/hard
    时只保留该档对象（其余档与 ignore 类对象剔除）。

    Returns:
        {idx: {"file_name": str, "objects": [{"name": "car", "bbox": [x1,y1,x2,y2],
               "difficulty": "easy"}]}}
    """
    label_files = sorted(label_dir.glob("*.txt"))
    if max_images > 0:
        label_files = label_files[:max_images]

    gt: dict[int, dict[str, Any]] = {}
    skipped = 0

    for i, lf in enumerate(label_files):
        img_name = f"{lf.stem}.png"
        img_path = image_dir / img_name
        if not img_path.exists():
            skipped += 1
            continue

        objects = []
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            cls_name = parts[0]
            if cls_name in ("Misc", "DontCare"):
                continue
            coco_name = KITTI_TO_COCO.get(cls_name)
            if coco_name is None:
                continue

            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            truncated = float(parts[1])
            occluded = int(parts[2])
            diff = kitti_difficulty(truncated, occluded, y2 - y1)
            if diff is None:
                continue  # 不满足任何难度档（ignore 语义），不参与评测
            objects.append({
                "name": coco_name, "bbox": [x1, y1, x2, y2],
                "difficulty": diff,
            })

        gt[i] = {"file_name": img_name, "objects": objects}

    if difficulty:
        gt = filter_gt_by_difficulty(gt, difficulty)

    if skipped:
        print(f"跳过 {skipped} 张缺少图片的标注")
    return gt


# ================================================================
# 2. 检测
# ================================================================

def run_detection(
    gt: dict[int, dict[str, Any]],
    image_dir: Path,
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

    model = create_detection_model(model_name, iou_threshold=iou)
    predictions: dict[int, list[dict]] = {}

    # 批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    _det_batch = getattr(model, "detect_batch", None)
    infer_fn = (
        (lambda ps: _det_batch(ps, all_cats, conf, 0))
        if _det_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, sample_image_paths(gt, image_dir),
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
            img_path = image_dir / gt[img_id]["file_name"]
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
    parser = build_parser("KITTI object Detection Benchmark")
    parser.set_defaults(model="yolo26x.pt")
    parser.add_argument(
        "--difficulty", choices=["all", "easy", "moderate", "hard"], default="all",
        help="难度分层：all=overall+三档同报；easy/moderate/hard=单档（KITTI 官方判据）",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    kitty_root = ensure_kitti(max_images=args.max_images or 300)
    image_dir = kitty_root / "training" / "image_2"
    label_dir = kitty_root / "training" / "label_2"

    # 加载 GT
    print("加载 KITTI Ground Truth ...")
    gt = load_kitti_ground_truth(image_dir, label_dir, max_images=args.max_images or 0)
    print(f"已加载 {len(gt)} 张图像\n")

    # 检测
    predictions = run_detection(gt, image_dir, args.model, args.conf, args.iou,
                                batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 bbox）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("kitti", gt, predictions,
                          lambda img_id, info: image_dir / info["file_name"])
        print()

    # 评估（--difficulty all = overall + 三档分层同报；分层 = 分层 GT × 全量预测）
    print(f"计算指标（IoU@{IOU_MATCH_THRESHOLD}）...\n")
    if args.difficulty == "all":
        tiers = [("overall", gt)] + [
            (t, filter_gt_by_difficulty(gt, t)) for t in ("easy", "moderate", "hard")
        ]
    else:
        tiers = [(args.difficulty, filter_gt_by_difficulty(gt, args.difficulty))]

    per_tier: dict[str, dict[str, Any]] = {}
    primary_summary = ""
    for tier_name, tier_gt in tiers:
        tier_cats = sorted(set(o["name"] for g in tier_gt.values() for o in g["objects"]))
        tier_results = {
            cls: evaluate_per_class(tier_gt, predictions, cls, IOU_MATCH_THRESHOLD)
            for cls in tier_cats
        }
        map_tier = float(np.mean([tier_results[c]["ap"] for c in tier_cats])) if tier_cats else 0.0
        per_tier[tier_name] = {
            "mAP@0.5": round(map_tier, 4),
            "gt_objects": sum(len(g["objects"]) for g in tier_gt.values()),
            "per_class": tier_results,
        }
        if tier_name == tiers[0][0]:  # 主表：overall 或指定单档
            primary_summary = format_result_table(tier_results, tier_cats, top_n=args.top_classes)

    print(primary_summary)
    print()
    if args.difficulty == "all":
        for tier_name in ("easy", "moderate", "hard"):
            t = per_tier[tier_name]
            print(f"  {tier_name:>8}: mAP@0.5 = {t['mAP@0.5']:.4f}（{t['gt_objects']} GT 目标）")
        print()

    # 保存
    primary = per_tier[tiers[0][0]]
    result_data = {
        "timestamp": ts, "dataset": "KITTI object",
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "difficulty": args.difficulty,
        "summary": {
            "mAP@0.5": primary["mAP@0.5"],
            "per_difficulty": {t: v["mAP@0.5"] for t, v in per_tier.items()},
        },
        "per_class": primary["per_class"],
    }

    json_path, md_path = save_results(result_data, "kitti", args.model, ts)
    diff_line = " | ".join(f"{t} {v['mAP@0.5']:.4f}" for t, v in per_tier.items())
    md_path.write_text("\n".join([
        f"# KITTI object Benchmark",
        f"- **模型**: {args.model} | **conf**: {args.conf} | **mAP@0.5**: {primary['mAP@0.5']:.4f}",
        f"- **难度分层**: {diff_line}",
        f"```\n{primary_summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
