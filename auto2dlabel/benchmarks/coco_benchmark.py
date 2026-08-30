#!/usr/bin/env python3
"""COCO val 2017 Benchmark — 自动标注 vs Ground Truth。

COCO 是业界标准检测 benchmark，80 类，5,000 张 val 图像。
GT 和预测均为 COCO JSON 格式，可直接用 pycocotools 或自实现评估。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, json, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    detect_batch_or_fallback,
    evaluate_per_class,
    format_result_table,
    sample_image_paths,
    save_results,
)

# ---- 配置 ----
COCO_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets" / "COCO2017"
IMAGE_DIR = COCO_ROOT / "val2017"
GT_JSON = COCO_ROOT / "annotations" / "instances_val2017.json"

CONFIG = {
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "model_name": "yolo26x.pt",
}

IOU_MATCH_THRESHOLD = 0.5


# ============================================================
# 1. 加 GT
# ============================================================

def load_coco_ground_truth(max_images: int = 0) -> dict[int, dict[str, Any]]:
    """加 COCO GT。

    Returns:
        {image_id: {"file_name": str, "objects": [{"name": str, "bbox": [x,y,w,h]}]}}

    COCO bbox 格式: [x, y, width, height]（绝对像素）
    """
    data = json.loads(GT_JSON.read_text())

    # 建立 image_id → file_name 映射
    id_to_filename: dict[int, str] = {}
    for img in data["images"]:
        id_to_filename[img["id"]] = img["file_name"]

    # 建立 category_id → name 映射
    id_to_cat: dict[int, str] = {}
    for cat in data["categories"]:
        id_to_cat[cat["id"]] = cat["name"]

    # 按 image_id 分组标注
    gt: dict[int, dict[str, Any]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = id_to_cat.get(ann["category_id"], str(ann["category_id"]))
        if img_id not in gt:
            gt[img_id] = {
                "file_name": id_to_filename.get(img_id, f"{img_id}.jpg"),
                "objects": [],
            }
        # COCO: [x, y, w, h] → 转为 [x1, y1, x2, y2] 方计算 IoU
        x, y, w, h = ann["bbox"]
        gt[img_id]["objects"].append({
            "name": cat_name,
            "bbox": [x, y, x + w, y + h],
        })

    if max_images > 0:
        keys = list(gt.keys())[:max_images]
        gt = {k: gt[k] for k in keys}

    return gt


# ============================================================
# 2. 运行检测
# ============================================================

def run_detection(
    gt: dict[int, dict[str, Any]],
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

    # 收所有 GT 类别作为检测 prompt
    all_cats = sorted(set(
        o["name"] for g in gt.values() for o in g["objects"]
    ))
    print(f"检测类别: {len(all_cats)} 种（{', '.join(all_cats[:10])}...）")

    model = create_detection_model(CONFIG["model_name"], iou_threshold=CONFIG["iou_threshold"])
    predictions: dict[int, list[dict]] = {}

    # 批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    _det_batch = getattr(model, "detect_batch", None)
    infer_fn = (
        (lambda ps: _det_batch(ps, all_cats, CONFIG["confidence_threshold"], 0))
        if _det_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, sample_image_paths(gt, IMAGE_DIR),
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(gt)
    img_ids = list(gt.keys())
    n_batches = (total + batch_size - 1) // batch_size
    iterator = tqdm(range(0, total, batch_size), total=n_batches,
                    desc=f"检测（{CONFIG['model_name']} b={batch_size}）", unit="batch") \
        if has_tqdm else range(0, total, batch_size)

    _t0 = time.time()
    for start in iterator:
        # 分块（缺失文件过滤，chunk_ids 与 chunk_paths 保序对齐）
        chunk_ids: list[int] = []
        chunk_paths: list[str] = []
        for img_id in img_ids[start:start + batch_size]:
            img_path = IMAGE_DIR / gt[img_id]["file_name"]
            if not img_path.exists():
                continue
            chunk_ids.append(img_id)
            chunk_paths.append(str(img_path))
        if not chunk_paths:
            continue

        results_per_img = detect_batch_or_fallback(
            model, chunk_paths, all_cats, CONFIG["confidence_threshold"], num_workers,
        )
        # COCO 格式: [x, y, w, h]
        for img_id, results in zip(chunk_ids, results_per_img):
            predictions[img_id] = [
                {"name": r.label, "bbox": [r.x, r.y, r.x + r.width, r.y + r.height], "conf": r.confidence}
                for r in results
            ]

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total/elapsed:.1f} img/s")
    return predictions


# ============================================================
# 3. 主流程（compute_iou / evaluate_per_class / format / save 已移至 common.py）
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="COCO val 2017 Detection Benchmark")
    parser.add_argument("--max-images", type=int, default=0, help="最多图像数（0=全部 5000）")
    parser.add_argument("--conf", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument("--model", type=str, default=CONFIG["model_name"])
    parser.add_argument("--iou", type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--top-classes", type=int, default=20, help="展示前 N 类的详细结果")
    parser.add_argument("--batch", type=int, default=None,
                        help="批量推理每批图像数（默认 None=模型加载后自动实测最大 batch，1=逐图）")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 子进程数（默认 None=按 CPU 核数自动）")
    parser.add_argument("--viz", action="store_true",
                        help="渲染预测结果到项目同级 Visualization/coco2017/（评测协议不变）")

    args = parser.parse_args()

    CONFIG["confidence_threshold"] = args.conf
    CONFIG["model_name"] = args.model
    CONFIG["iou_threshold"] = args.iou

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from auto2dlabel.tools.device import print_device
    print_device()
    print()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 加 GT
    print("加载 COCO val 2017 Ground Truth...")
    gt = load_coco_ground_truth(max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")

    # 运行检测
    predictions = run_detection(gt, batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 bbox）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("coco2017", gt, predictions,
                          lambda img_id, info: IMAGE_DIR / info["file_name"])
        print()

    # 计算指标
    print(f"计算指标（IoU@{IOU_MATCH_THRESHOLD}）...\n")
    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    results = {}
    for cls in all_cats:
        results[cls] = evaluate_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD)

    summary = format_result_table(results, all_cats, top_n=args.top_classes)
    print(summary)
    print()

    all_aps = [results[cls]["ap"] for cls in all_cats if cls in results]
    mAP = float(np.mean(all_aps)) if all_aps else 0.0

    # 保存
    result_data = {
        "timestamp": timestamp, "dataset": "COCO val 2017",
        "model": CONFIG["model_name"], "confidence_threshold": CONFIG["confidence_threshold"],
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "coco2017", CONFIG["model_name"], timestamp)
    md_path.write_text("\n".join([
        "# COCO val 2017 Benchmark",
        f"- **模型**: {CONFIG['model_name']} | **conf**: {CONFIG['confidence_threshold']} | **mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
