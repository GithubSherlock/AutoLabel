#!/usr/bin/env python3
"""DOTA Aerial Detection Benchmark — 航拍图像目标检测 + 实例分割。

DOTA 是航拍视角目标检测数据集（15 类），使用 v1.0 HBB（水平框）标注。
由于 yolo26x.pt（COCO 预训练）仅支持 80 个通用类别，本 benchmark 仅评估
4 个可映射的类别：plane→airplane, ship→boat, large-vehicle→truck, small-vehicle→car。
其余 11 类在航拍场景中无 COCO 对应类，跳过评估。

GT: labels/*.txt → {name: coco_name, bbox: [x1,y1,x2,y2]}
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
from auto2dlabel.benchmarks.datasets import ensure_dota_val

# ── 配置 ──────────────────────────────────────────────────────
# DOTA 15 类 → COCO 80 类映射（仅 4 类有合理对应）
DOTA_TO_COCO = {
    "plane": "airplane",
    "ship": "boat",
    "large-vehicle": "truck",
    "small-vehicle": "car",
}
# 以下 DOTA 类别无 COCO 对应，跳过评估:
#   storage-tank, harbor, tennis-court, bridge, swimming-pool,
#   baseball-diamond, roundabout, soccer-ball-field, ground-track-field,
#   basketball-court, helicopter

SKIPPED_CLASSES = frozenset([
    "storage-tank", "harbor", "tennis-court", "bridge", "swimming-pool",
    "baseball-diamond", "roundabout", "soccer-ball-field", "ground-track-field",
    "basketball-court", "helicopter",
])


# ================================================================
# 1. GT 加载
# ================================================================

def load_dota_ground_truth(
    image_dir: Path, label_dir: Path, max_images: int = 0,
) -> dict[int, dict[str, Any]]:
    """加载 DOTA HBB 标注。

    HBB Task2 格式每行（无 header，CRLF 换行）:
      x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
    其中 (x1,y1)=左上, (x3,y3)=右下（水平矩形）。

    Returns:
        {idx: {"file_name": str, "objects": [{"name": "airplane", "bbox": [x1,y1,x2,y2]}]}}
    """
    label_files = sorted(label_dir.glob("*.txt"))
    if max_images > 0:
        label_files = label_files[:max_images]

    gt: dict[int, dict[str, Any]] = {}
    skipped_img = 0
    skipped_obj = 0

    for i, lf in enumerate(label_files):
        img_name = f"{lf.stem}.png"
        img_path = image_dir / img_name
        if not img_path.exists():
            skipped_img += 1
            continue

        objects = []
        for line in lf.read_text().strip().splitlines():
            line = line.strip()
            parts = line.split()
            if len(parts) < 9:
                continue
            cls_name = parts[8]
            if cls_name in SKIPPED_CLASSES:
                skipped_obj += 1
                continue
            coco_name = DOTA_TO_COCO.get(cls_name)
            if coco_name is None:
                continue

            # HBB: 4 个角点 → 取 min/max 得水平矩形
            xs = [float(parts[i]) for i in range(0, 8, 2)]
            ys = [float(parts[i]) for i in range(1, 8, 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            objects.append({"name": coco_name, "bbox": [x1, y1, x2, y2]})

        gt[i] = {"file_name": img_name, "objects": objects}

    if skipped_img:
        print(f"  跳过 {skipped_img} 张缺少图片的标注")
    if skipped_obj:
        print(f"  跳过 {skipped_obj} 个无 COCO 映射的标注对象")
    return gt


# ================================================================
# 2. 检测
# ================================================================

def run_detection(
    gt: dict[int, dict[str, Any]], image_dir: Path,
    model_name: str, conf: float, iou: float,
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
    parser = build_parser("DOTA Aerial Detection Benchmark")
    parser.set_defaults(model="yolo26x.pt")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    dota_root = ensure_dota_val()
    image_dir = dota_root / "images"
    label_dir = dota_root / "labels"

    # 加载 GT
    print("加载 DOTA Ground Truth (HBB, 4 类可映射至 COCO) ...")
    gt = load_dota_ground_truth(image_dir, label_dir, max_images=args.max_images or 0)
    print(f"已加载 {len(gt)} 张图像\n")

    # 检测（SAHI 切片路径不支持批量，仅标准路径接入）
    if args.sahi:
        predictions = run_detection_sahi(gt, image_dir, args.model, args.conf, args.iou)
    else:
        predictions = run_detection(gt, image_dir, args.model, args.conf, args.iou,
                                    batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 bbox）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("dota", gt, predictions,
                          lambda img_id, info: image_dir / info["file_name"])
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
    mapped_note = " | ".join(f"{d}={c}" for d, c in DOTA_TO_COCO.items())
    result_data = {
        "timestamp": ts, "dataset": "DOTA val (aerial, 4/15 classes mapped to COCO)",
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "class_mapping": DOTA_TO_COCO,
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "dota", args.model, ts)
    md_path.write_text("\n".join([
        "# DOTA Aerial Detection Benchmark",
        f"- **模型**: {args.model} | **conf**: {args.conf} | **mAP@0.5**: {mAP:.4f}",
        f"- **类映射**: {mapped_note}",
        "  (其余 11 类无 COCO 对应，跳过评估)",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
