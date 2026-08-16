#!/usr/bin/env python3
"""D2SA Instance Segmentation Benchmark — 密集零售货架。

D2SA 是密集零售货架商品实例分割数据集（60 SKU 类，27 超类，COCO JSON）。
由于商品 SKU 类名无 COCO 对应，本 benchmark 采用 **box-prompted 分割** 模式：
跳过 YOLO 检测步骤，直接用 GT bbox 作为 FastSAM 的 box prompt，
评估 FastSAM 在给定正确位置下的原始分割质量。

GT: annotations/D2S_amodal_validation.json → RLE mask + bbox
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks.common import (
    OUTPUT_DIR,
    build_parser,
    evaluate_mask_per_class,
    format_mask_result_table,
    polygon_to_mask,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_d2sa_val

# ── 配置 ──────────────────────────────────────────────────────
MASK_IOU_THRESHOLD = 0.5


# ================================================================
# 1. GT 加载（使用 COCO API 的 annToMask，兼容 D2SA RLE 格式）
# ================================================================

def load_d2sa_ground_truth(
    gt_json: Path, max_images: int = 0,
) -> dict[int, dict[str, Any]]:
    """加载 D2SA GT（RLE mask + bbox）。

    使用 pycocotools.coco.COCO.annToMask() 处理 RLE 解码，
    该格式含 backslash 字符，frPyObjects() 无法直接处理。

    Returns:
        gt: {image_id: {"file_name": str, "objects": [{"name": str, "mask": ndarray, "bbox": [x,y,w,h]}]}}
    """
    from pycocotools.coco import COCO

    coco = COCO(str(gt_json))

    # 类别索引: cat_id → supercategory
    cat_to_supercat: dict[int, str] = {}
    for cat in coco.loadCats(coco.getCatIds()):
        cat_to_supercat[cat["id"]] = cat.get("supercategory", "unknown")

    # 图片索引
    img_id_to_filename: dict[int, str] = {}
    for img in coco.loadImgs(coco.getImgIds()):
        img_id_to_filename[img["id"]] = img["file_name"]

    # 限制图片数
    img_ids = sorted(coco.getImgIds())
    if max_images > 0:
        img_ids = img_ids[:max_images]

    gt: dict[int, dict[str, Any]] = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        if not ann_ids:
            continue

        objects = []
        for ann in coco.loadAnns(ann_ids):
            mask = coco.annToMask(ann)  # (H, W) bool
            if mask.sum() == 0:
                continue
            supercat = cat_to_supercat.get(ann["category_id"], "unknown")
            bbox = ann["bbox"]  # [x, y, w, h]
            objects.append({
                "name": supercat,
                "mask": mask,
                "bbox": bbox,
            })

        if objects:
            gt[img_id] = {
                "file_name": img_id_to_filename.get(img_id, f"{img_id}.jpg"),
                "objects": objects,
            }

    return gt


# ================================================================
# 2. Box-prompted 分割（跳过检测，直接用 GT bbox）
# ================================================================

def run_box_prompted_segmentation(
    gt: dict[int, dict[str, Any]],
    image_dir: Path,
    seg_model_name: str,
) -> dict[int, list[dict]]:
    """用 GT bbox 作为 prompt 运行 FastSAM 分割。

    不依赖检测模型，直接以 GT 的 bbox 位置作为分割提示。
    评估纯分割质量。
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox

    seg_model = create_segmentation_model(seg_model_name)

    predictions: dict[int, list[dict]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = tqdm(enumerate(img_ids), total=total, desc=f"分割（{seg_model_name}）", unit="img") if has_tqdm else enumerate(img_ids)

    _t0 = time.time()
    for i, img_id in iterator:
        info = gt[img_id]
        img_path = image_dir / info["file_name"]
        if not img_path.exists():
            continue

        objects = info["objects"]
        if not objects:
            predictions[img_id] = []
            continue

        # 用 GT bbox 构建 box prompts
        bboxes = []
        for obj in objects:
            x, y, w, h = obj["bbox"]
            bboxes.append(Bbox(
                x=x, y=y, width=w, height=h,
                label=obj["name"], confidence=1.0,
            ))

        # FastSAM box-prompted 分割
        masks = seg_model.generate(str(img_path), bboxes)

        # 获取图像尺寸用于 polygon → mask 转换
        from PIL import Image as _PILImage
        im = _PILImage.open(img_path)
        img_h, img_w = im.height, im.width

        predictions[img_id] = []
        for j, mask_obj in enumerate(masks):
            mask_bool = polygon_to_mask(mask_obj.segmentation, img_h, img_w)
            # 用 j 关联回原始 GT object 的 label
            label = objects[j]["name"] if j < len(objects) else "unknown"
            predictions[img_id].append({
                "name": label,
                "conf": 1.0,
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
    parser = build_parser("D2SA Instance Segmentation Benchmark (box-prompted)")
    parser.add_argument("--seg-model", type=str, default="FastSAM-s.pt", help="分割模型")
    parser.set_defaults(model="FastSAM-s.pt")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    d2sa_root = ensure_d2sa_val()
    image_dir = d2sa_root / "images"
    gt_json = d2sa_root / "annotations" / "D2S_amodal_validation.json"

    if not gt_json.exists():
        raise FileNotFoundError(f"D2SA val JSON 未找到: {gt_json}")

    # 加载 GT
    print("加载 D2SA validation Ground Truth (RLE masks, 27 超类) ...")
    gt = load_d2sa_ground_truth(gt_json, max_images=args.max_images)
    all_supercats = sorted(set(
        o["name"] for g in gt.values() for o in g["objects"]
    ))
    total_instances = sum(len(g["objects"]) for g in gt.values())
    print(f"已加载 {len(gt)} 张图像, {total_instances} 个实例, {len(all_supercats)} 个超类\n")

    # 分割（box-prompted，无检测步骤）
    print("运行 box-prompted 分割（使用 GT bbox）...")
    predictions = run_box_prompted_segmentation(gt, image_dir, args.seg_model)
    print()

    # 评估 — 按超类分组
    print(f"计算 mask 指标（mask IoU@{MASK_IOU_THRESHOLD}）...\n")
    results = {}
    for cls in all_supercats:
        results[cls] = evaluate_mask_per_class(gt, predictions, cls, MASK_IOU_THRESHOLD)

    summary = format_mask_result_table(results, all_supercats, top_n=args.top_classes)
    print(summary)
    print()

    # 保存
    valid_cls = [c for c in all_supercats if c in results]
    mAP = float(np.mean([results[cls]["ap"] for cls in valid_cls])) if valid_cls else 0.0
    mIoU = float(np.mean([results[cls]["miou"] for cls in valid_cls])) if valid_cls else 0.0
    mDice = float(np.mean([results[cls]["mdice"] for cls in valid_cls])) if valid_cls else 0.0

    result_data = {
        "timestamp": ts, "dataset": "D2SA val (box-prompted seg, 27 supercategories)",
        "seg_model": args.seg_model, "mode": "box-prompted (no detection)",
        "mask_iou_threshold": MASK_IOU_THRESHOLD, "image_count": len(gt),
        "instance_count": total_instances, "supercategory_count": len(all_supercats),
        "summary": {
            "mask mAP@0.5": round(mAP, 4),
            "mIoU": round(mIoU, 4),
            "mDice": round(mDice, 4),
        },
        "per_class": {cls: results[cls] for cls in valid_cls},
    }

    json_path, md_path = save_results(result_data, "d2sa", args.seg_model, ts)
    md_path.write_text("\n".join([
        "# D2SA Instance Segmentation Benchmark (box-prompted)",
        f"- **分割模型**: {args.seg_model} | **模式**: GT bbox prompt（无检测步骤）",
        f"- **mask mAP@0.5**: {mAP:.4f} | **mIoU**: {mIoU:.4f} | **mDice**: {mDice:.4f}",
        f"- **{len(gt)} 张图, {total_instances} 个实例, {len(all_supercats)} 个超类**",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
