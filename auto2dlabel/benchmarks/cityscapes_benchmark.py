#!/usr/bin/env python3
"""Cityscapes val Instance Segmentation Benchmark — mask 预测 vs GT。

cityscapes 的 instanceIds.png 编码了实例 mask（像素值 = instance_id * 1000 + class_id）。
对 thing 类（8 类）评估 mask 指标。
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
    evaluate_mask_per_class,
    format_mask_result_table,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_cityscapes_val

# ── 配置 ──────────────────────────────────────────────────────
DET_MODEL = "yolov8x.pt"
SEG_MODEL = "FastSAM-s.pt"

# cityscapes thing 类: class_id → COCO name
THING_CLASSES = {
    24: "person",      # person
    25: "person",      # rider → person
    26: "car",
    27: "truck",
    28: "bus",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}


# ================================================================
# 1. GT 加载
# ================================================================

def load_cityscapes_ground_truth(
    gt_dir: Path,
    img_dir: Path,
    max_images: int = 0,
) -> dict[int, dict[str, Any]]:
    """从 instanceIds.png 加载 cityscapes GT。

    instanceIds.png 编码: pixel_value = instance_id * 1000 + class_id
    class_id ∈ {24..28, 31..33} 为 thing 类（有实例 mask）。

    Returns:
        {idx: {"file_name": str, "objects": [{"name": "car", "mask": ndarray (H,W) bool}]}}
    """
    gt_files = sorted(gt_dir.rglob("*_gtFine_instanceIds.png"))
    if max_images > 0:
        gt_files = gt_files[:max_images]

    gt: dict[int, dict[str, Any]] = {}
    for i, gf in enumerate(gt_files):
        # 对应的原始图像
        city_name = gf.name.split("_")[0]
        img_name = gf.name.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")
        img_path = img_dir / city_name / img_name
        if not img_path.exists():
            continue

        # 读取 instance IDs
        instance_ids = np.array(Image.open(gf), dtype=np.int32)
        h, w = instance_ids.shape

        objects = []
        unique_instances = np.unique(instance_ids)
        for inst_id in unique_instances:
            if inst_id == 0:
                continue
            class_id = inst_id % 1000
            coco_name = THING_CLASSES.get(class_id)
            if coco_name is None:
                continue

            mask = (instance_ids == inst_id)
            objects.append({"name": coco_name, "mask": mask})

        gt[i] = {"file_name": f"{city_name}/{img_name}", "objects": objects}

    return gt


# ================================================================
# 2. 检测 + 分割
# ================================================================

def run_segmentation(
    gt: dict[int, dict[str, Any]],
    img_dir: Path,
    conf: float,
    iou: float,
) -> dict[int, list[dict]]:
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox
    from auto2dlabel.benchmarks.common import polygon_to_mask

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"分割类别 ({len(all_cats)}): {', '.join(all_cats)}")

    det_model = create_detection_model(DET_MODEL, iou_threshold=iou)
    seg_model = create_segmentation_model(SEG_MODEL)

    predictions: dict[int, list[dict]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = tqdm(enumerate(img_ids), total=total, desc=f"分割（{SEG_MODEL}）", unit="img") if has_tqdm else enumerate(img_ids)

    _t0 = time.time()
    for i, img_id in iterator:
        filename = gt[img_id]["file_name"]
        img_path = img_dir / filename
        if not img_path.exists():
            continue

        # Step 1: 检测
        det_results = det_model.detect(str(img_path), all_cats, confidence_threshold=conf)
        high_conf = [r for r in det_results if r.confidence >= 0.5]
        if not high_conf:
            predictions[img_id] = []
            continue

        # Step 2: 分割
        bboxes = [Bbox(x=r.x, y=r.y, width=r.width, height=r.height, label=r.label, confidence=r.confidence)
                  for r in high_conf]
        masks = seg_model.generate(str(img_path), bboxes)

        # 读取原图尺寸
        from PIL import Image as _Image
        im = _Image.open(img_path)
        h, w = im.height, im.width

        predictions[img_id] = []
        for mask_obj in masks:
            mask_bool = polygon_to_mask(mask_obj.segmentation, h, w)
            predictions[img_id].append({
                "name": mask_obj.bbox.label,
                "conf": mask_obj.bbox.confidence,
                "mask": mask_bool,
            })

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


# ================================================================
# 3. 主流程
# ================================================================

def main():
    parser = build_parser("Cityscapes Instance Segmentation Benchmark")
    parser.set_defaults(model=DET_MODEL)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    cs_root = ensure_cityscapes_val()
    gt_dir = cs_root / "gtFine" / "val"
    img_dir = cs_root / "leftImg8bit" / "val"

    # 加载 GT
    print("加载 cityscapes Ground Truth (instanceIds.png) ...")
    gt = load_cityscapes_ground_truth(gt_dir, img_dir, max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")

    # 分割
    predictions = run_segmentation(gt, img_dir, args.conf, args.iou)
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
        "timestamp": ts, "dataset": "cityscapes val",
        "seg_model": SEG_MODEL, "det_model": DET_MODEL,
        "confidence_threshold": args.conf,
        "mask_iou_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "cityscapes", SEG_MODEL, ts)
    md_path.write_text("\n".join([
        f"# Cityscapes Instance Segmentation Benchmark",
        f"- **分割模型**: {SEG_MODEL} | **检测 backbone**: {DET_MODEL} | **mask mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    from PIL import Image
    main()
