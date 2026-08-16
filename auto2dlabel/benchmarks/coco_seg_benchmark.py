#!/usr/bin/env python3
"""COCO val 2017 Instance Segmentation Benchmark — mask 预测 vs GT。

使用 FastSAM（box-prompted）对检测结果运行分割，评估 mask 级指标。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks.common import (
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    build_parser,
    coco_seg_to_mask,
    evaluate_mask_per_class,
    format_mask_result_table,
    polygon_to_mask,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_coco_val

# ── 配置 ──────────────────────────────────────────────────────
DET_MODEL = "yolov8x.pt"  # 检测 backbone（为 FastSAM 提供 bbox）


# ================================================================
# 1. GT 加载（mask 版本）
# ================================================================

def load_coco_seg_ground_truth(gt_json: Path, max_images: int = 0) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[int, int]]]:
    """加载 COCO GT（含 polygon mask）。

    Returns:
        gt: {image_id: {"objects": [{"name": str, "mask": ndarray (H,W) bool}, ...]}}
        sizes: {image_id: (h, w)}
    """
    data = json.loads(gt_json.read_text())

    id_to_filename: dict[int, str] = {}
    id_to_size: dict[int, tuple[int, int]] = {}
    for img in data["images"]:
        id_to_filename[img["id"]] = img["file_name"]
        id_to_size[img["id"]] = (img["height"], img["width"])

    id_to_cat: dict[int, str] = {}
    for cat in data["categories"]:
        id_to_cat[cat["id"]] = cat["name"]

    # 按 image_id 分组
    gt: dict[int, dict[str, Any]] = {}
    for ann in data["annotations"]:
        if "segmentation" not in ann or not ann["segmentation"]:
            continue  # 跳过无 mask 标注

        img_id = ann["image_id"]
        cat_name = id_to_cat.get(ann["category_id"], str(ann["category_id"]))

        h, w = id_to_size.get(img_id, (0, 0))
        mask = coco_seg_to_mask(ann["segmentation"], h, w)

        if img_id not in gt:
            gt[img_id] = {"file_name": id_to_filename.get(img_id, f"{img_id}.jpg"), "objects": []}
        gt[img_id]["objects"].append({"name": cat_name, "mask": mask})

    if max_images > 0:
        keys = sorted(gt.keys())[:max_images]
        gt = {k: gt[k] for k in keys}

    return gt, id_to_size


# ================================================================
# 2. 检测 + 分割
# ================================================================

def run_segmentation(
    gt: dict[int, dict[str, Any]],
    image_dir: Path,
    conf: float,
    iou: float,
    seg_model_name: str,
) -> dict[int, list[dict]]:
    """检测 → 分割，返回 mask 预测。"""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"分割类别 ({len(all_cats)}): {', '.join(all_cats[:10])}...")

    det_model = create_detection_model(DET_MODEL, iou_threshold=iou)
    seg_model = create_segmentation_model(seg_model_name)

    predictions: dict[int, list[dict]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = tqdm(enumerate(img_ids), total=total, desc=f"分割（{seg_model_name}）", unit="img") if has_tqdm else enumerate(img_ids)

    _t0 = time.time()
    for i, img_id in iterator:
        filename = gt[img_id]["file_name"]
        img_path = image_dir / filename
        if not img_path.exists():
            continue

        # Step 1: 检测
        det_results = det_model.detect(str(img_path), all_cats, confidence_threshold=conf)
        high_conf = [r for r in det_results if r.confidence >= 0.5]
        if not high_conf:
            predictions[img_id] = []
            continue

        # Step 2: 分割（用高置信度 bbox 做 prompt）
        bboxes = [Bbox(x=r.x, y=r.y, width=r.width, height=r.height, label=r.label, confidence=r.confidence)
                  for r in high_conf]
        masks = seg_model.generate(str(img_path), bboxes)

        # 获取图像尺寸用于 polygon → mask 转换
        from PIL import Image as _PILImage
        im = _PILImage.open(img_path)
        h, w = im.height, im.width

        predictions[img_id] = []
        for mask_obj in masks:
            mask_bool = polygon_to_mask(mask_obj.segmentation, h, w)
            predictions[img_id].append({
                "name": mask_obj.bbox.label,
                "conf": mask_obj.bbox.confidence,
                "mask": mask_bool,
                "bbox": mask_obj.bbox,
            })

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


# ================================================================
# 3. 主流程
# ================================================================

def main():
    parser = build_parser("COCO val 2017 Instance Segmentation Benchmark")
    parser.add_argument("--seg-model", type=str, default="FastSAM-s.pt", help="分割模型")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    coco_root = ensure_coco_val()
    image_dir = coco_root / "val2017"
    gt_json = coco_root / "annotations" / "instances_val2017.json"

    # 加载 GT
    print("加载 COCO val 2017 Ground Truth (polygon masks) ...")
    gt, sizes = load_coco_seg_ground_truth(gt_json, max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")

    # 分割
    predictions = run_segmentation(gt, image_dir, args.conf, args.iou, args.seg_model)
    print()

    # 评估
    print(f"计算 mask 指标（mask IoU@{IOU_MATCH_THRESHOLD}）...\n")
    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    results = {}
    for cls in all_cats:
        results[cls] = evaluate_mask_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD)

    summary = format_mask_result_table(results, all_cats, top_n=args.top_classes)
    print(summary)
    print()

    # 保存
    mAP = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))
    result_data = {
        "timestamp": ts, "dataset": "COCO val 2017 (seg)",
        "model": args.seg_model, "det_model": DET_MODEL,
        "confidence_threshold": args.conf,
        "mask_iou_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "coco_seg", args.seg_model, ts)
    md_path.write_text("\n".join([
        f"# COCO val 2017 Instance Segmentation Benchmark",
        f"- **分割模型**: {args.seg_model} | **检测 backbone**: {DET_MODEL} | **mask mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
