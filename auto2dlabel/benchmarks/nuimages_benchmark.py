#!/usr/bin/env python3
"""nuImages Mini Instance Segmentation Benchmark — mask 预测 vs GT。

nuImages 使用 RLE 编码的实例 mask（通过 nuscenes-devkit 解码）。
Mini 版仅 50 张图，适合快速 smoke test。
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
    evaluate_mask_per_class,
    format_mask_result_table,
    sample_image_paths,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_nuimages_mini

# ── 配置 ──────────────────────────────────────────────────────
DET_MODEL = "yolo26x.pt"
SEG_MODEL = "sam2_l.pt"

# nuImages 类别 → COCO 映射（nuImages 使用层级名称）
NUIMAGES_TO_COCO = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.trailer": None,
    "vehicle.construction": None,
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.personal_mobility": "person",
    "human.pedestrian.police_officer": "person",
    "human.pedestrian.stroller": "person",
    "human.pedestrian.wheelchair": "person",
    "movable_object.barrier": None,
    "movable_object.trafficcone": None,
    "movable_object.debris": None,
    "movable_object.pushable_pullable": None,
    "static_object.bicycle_rack": None,
}


# ================================================================
# 1. GT 加载
# ================================================================

def load_nuimages_ground_truth(nuim_root: Path, max_images: int = 0) -> dict[int, dict[str, Any]]:
    """通过 nuscenes-devkit 加载 nuImages Mini 的 GT mask。

    每个 object_ann 自带了 mask（RLE 编码），无需额外查找。

    Returns:
        {idx: {"file_name": str, "objects": [{"name": "car", "mask": ndarray (H,W) bool}]}}
    """
    from nuimages import NuImages
    from nuimages.nuimages import mask_decode

    dataroot = str(nuim_root)
    nuim = NuImages(version="v1.0-mini", dataroot=dataroot, verbose=False)

    # sample_data_token → filename
    sd_token_to_filename: dict[str, str] = {
        sd["token"]: sd["filename"] for sd in nuim.sample_data
    }

    gt: dict[int, dict[str, Any]] = {}

    for obj_ann in nuim.object_ann:
        sd_token = obj_ann["sample_data_token"]
        if sd_token not in sd_token_to_filename:
            continue

        # 类别映射（使用层级名称）
        cat = nuim.get("category", obj_ann["category_token"])
        nu_name = cat["name"]
        coco_name = NUIMAGES_TO_COCO.get(nu_name)
        if coco_name is None:
            continue

        # 解码 nuImages 自定义 RLE mask（base64 编码）
        rle = obj_ann["mask"]
        if rle is None:
            continue
        try:
            mask = mask_decode(rle).astype(bool)
        except Exception:
            continue

        if mask is None or mask.sum() == 0:
            continue

        if sd_token not in gt:
            if max_images > 0 and len(gt) >= max_images:
                break
            gt[sd_token] = {
                "file_name": sd_token_to_filename[sd_token],
                "objects": [],
            }

        gt[sd_token]["objects"].append({"name": coco_name, "mask": mask})

    # 重索引为连续整数
    return {i: v for i, (k, v) in enumerate(gt.items())}


# ================================================================
# 2. 检测 + 分割
# ================================================================

def run_segmentation(
    gt: dict[int, dict[str, Any]],
    nuim_root: Path,
    conf: float,
    iou: float,
    batch: int | None = None,
    workers: int | None = None,
) -> dict[int, list[dict]]:
    """检测 → 分割，返回 mask 预测。

    检测步分块批量（detect_batch_or_fallback + OOM 降级）；
    分割步逐图（每图 bbox prompt 不同，SAM2/FastSAM 无批量能力）。
    GT RLE 解码在加载层，预测转换（polygon → mask）不变。
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from PIL import Image as _Image

    from auto2dlabel.benchmarks.common import polygon_to_mask
    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox
    from auto2dlabel.tools.device import resolve_batch_params

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"分割类别 ({len(all_cats)}): {', '.join(all_cats)}")

    det_model = create_detection_model(DET_MODEL, iou_threshold=iou)
    seg_model = create_segmentation_model(SEG_MODEL)

    predictions: dict[int, list[dict]] = {}

    # 检测步批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    _det_batch = getattr(det_model, "detect_batch", None)
    infer_fn = (
        (lambda ps: _det_batch(ps, all_cats, conf, 0))
        if _det_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, sample_image_paths(gt, nuim_root),
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(gt)
    img_ids = sorted(gt.keys())
    n_batches = (total + batch_size - 1) // batch_size
    iterator = tqdm(range(0, total, batch_size), total=n_batches,
                    desc=f"分割（{SEG_MODEL} b={batch_size}）", unit="batch") \
        if has_tqdm else range(0, total, batch_size)

    _t0 = time.time()
    for start in iterator:
        # 分块（缺失文件过滤，chunk_ids 与 chunk_paths 保序对齐）
        chunk_ids: list[int] = []
        chunk_paths: list[str] = []
        for img_id in img_ids[start:start + batch_size]:
            img_path = nuim_root / gt[img_id]["file_name"]
            if not img_path.exists():
                continue
            chunk_ids.append(img_id)
            chunk_paths.append(str(img_path))
        if not chunk_paths:
            continue

        # Step 1: 检测（批量）
        det_results_per_img = detect_batch_or_fallback(
            det_model, chunk_paths, all_cats, conf, num_workers,
        )
        for img_id, det_results in zip(chunk_ids, det_results_per_img):
            high_conf = [r for r in det_results if r.confidence >= 0.3]
            if not high_conf:
                predictions[img_id] = []
                continue

            # Step 2: 分割（逐图，用该图高置信度 bbox 做 prompt）
            img_path = nuim_root / gt[img_id]["file_name"]
            bboxes = [Bbox(x=r.x, y=r.y, width=r.width, height=r.height, label=r.label, confidence=r.confidence)
                      for r in high_conf]
            masks = seg_model.generate(str(img_path), bboxes)

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
    parser = build_parser("nuImages Mini Instance Segmentation Benchmark")
    parser.set_defaults(model=DET_MODEL, conf=0.3)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    nuim_root = ensure_nuimages_mini()

    # 加载 GT
    print("加载 nuImages Mini Ground Truth (RLE masks) ...")
    gt = load_nuimages_ground_truth(nuim_root, max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")

    # 分割
    predictions = run_segmentation(gt, nuim_root, args.conf, args.iou,
                                   batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 mask）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("nuimages", gt, predictions,
                          lambda img_id, info: nuim_root / info["file_name"])
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
        "timestamp": ts, "dataset": "nuImages Mini",
        "seg_model": SEG_MODEL, "det_model": DET_MODEL,
        "confidence_threshold": args.conf,
        "mask_iou_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "nuimages", SEG_MODEL, ts)
    md_path.write_text("\n".join([
        "# nuImages Mini Instance Segmentation Benchmark",
        f"- **分割模型**: {SEG_MODEL} | **检测 backbone**: {DET_MODEL} | **mask mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
