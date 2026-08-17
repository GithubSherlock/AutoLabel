#!/usr/bin/env python3
"""PASCAL VOC 2007 Benchmark — 自动标注 vs Ground Truth。"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, json, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    compute_iou,
    detect_batch_or_fallback,
    evaluate_per_class,
    format_result_table,
    save_results,
)

# ---- 配置 ----
VOC_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets" / "VOCdevkit" / "VOC2007"
ANNO_DIR = VOC_ROOT / "Annotations"
IMAGE_DIR = VOC_ROOT / "JPEGImages"
TEST_LIST = VOC_ROOT / "ImageSets" / "Main" / "test.txt"

CONFIG = {
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "model_name": "yolo26x.pt",
}

# VOC → COCO 类别别名映射（部分 VOC 类别名与 COCO 不同）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_TO_COCO = {
    "aeroplane": "airplane", "diningtable": "dining table",
    "motorbike": "motorcycle", "pottedplant": "potted plant",
    "sofa": "couch", "tvmonitor": "tv",
}


# ============================================================
# GT 加载
# ============================================================

def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename", "")
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", 0))
        ymin = float(bndbox.findtext("ymin", 0))
        xmax = float(bndbox.findtext("xmax", 0))
        ymax = float(bndbox.findtext("ymax", 0))
        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
    return {"filename": filename, "objects": objects}


def load_voc_ground_truth(image_ids: list[str]) -> dict[str, dict[str, Any]]:
    gt = {}
    for img_id in image_ids:
        xml_path = ANNO_DIR / f"{img_id}.xml"
        if xml_path.exists():
            gt[img_id] = parse_voc_xml(xml_path)
    return gt


# ============================================================
# 检测
# ============================================================

def run_detection(
    image_ids: list[str],
    batch: int | None = None,
    workers: int | None = None,
) -> dict[str, list[dict]]:
    """对 GT 中所有图像运行检测（批量推理 + OOM 降级逐图）。"""
    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.tools.device import resolve_batch_params

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    model = create_detection_model(CONFIG["model_name"], iou_threshold=CONFIG["iou_threshold"])
    predictions: dict[str, list[dict]] = {}

    # 批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    _det_batch = getattr(model, "detect_batch", None)
    infer_fn = (
        (lambda ps: _det_batch(ps, VOC_CLASSES, CONFIG["confidence_threshold"], 0))
        if _det_batch is not None else None
    )
    probe_paths: list[str] = []
    for img_id in image_ids:
        p = IMAGE_DIR / f"{img_id}.jpg"
        if p.exists():
            probe_paths.append(str(p))
        if len(probe_paths) >= 20:
            break
    batch_size, num_workers = resolve_batch_params(
        "object_detection", infer_fn, probe_paths,
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(image_ids)
    n_batches = (total + batch_size - 1) // batch_size
    iterator = tqdm(range(0, total, batch_size), total=n_batches,
                    desc=f"检测（{CONFIG['model_name']} b={batch_size}）", unit="batch") \
        if has_tqdm else range(0, total, batch_size)

    _t0 = time.time()
    for start in iterator:
        # 分块（缺失文件过滤，chunk_ids 与 chunk_paths 保序对齐）
        chunk_ids: list[str] = []
        chunk_paths: list[str] = []
        for img_id in image_ids[start:start + batch_size]:
            img_path = IMAGE_DIR / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            chunk_ids.append(img_id)
            chunk_paths.append(str(img_path))
        if not chunk_paths:
            continue

        results_per_img = detect_batch_or_fallback(
            model, chunk_paths, VOC_CLASSES, CONFIG["confidence_threshold"], num_workers,
        )
        for img_id, results in zip(chunk_ids, results_per_img):
            predictions[img_id] = [
                {"name": r.label, "bbox": [r.x, r.y, r.x + r.width, r.y + r.height], "conf": r.confidence}
                for r in results
            ]

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PASCAL VOC 2007 Benchmark")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument("--model", type=str, default=CONFIG["model_name"])
    parser.add_argument("--iou", type=float, default=CONFIG["iou_threshold"])
    parser.add_argument("--batch", type=int, default=None,
                        help="批量推理每批图像数（默认 None=模型加载后自动实测最大 batch，1=逐图）")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 子进程数（默认 None=按 CPU 核数自动）")
    parser.add_argument("--viz", action="store_true",
                        help="渲染预测结果到项目同级 Visualization/voc2007/（评测协议不变）")
    args = parser.parse_args()

    CONFIG["confidence_threshold"] = args.conf
    CONFIG["model_name"] = args.model
    CONFIG["iou_threshold"] = args.iou

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 加载图像列表
    if TEST_LIST.exists():
        image_ids = [line.strip() for line in TEST_LIST.read_text().splitlines() if line.strip()]
        image_ids = [line.split()[0] if " " in line else line for line in image_ids]
    else:
        image_ids = sorted([p.stem for p in IMAGE_DIR.glob("*.jpg")])

    if args.max_images > 0 and args.max_images < len(image_ids):
        image_ids = image_ids[:args.max_images]

    print(f"测试图像: {len(image_ids)} 张")

    # 加载 GT
    print("加载 Ground Truth...")
    gt = load_voc_ground_truth(image_ids)
    print(f"已加载 {len(gt)} 张图像\n")

    # 检测
    predictions = run_detection(image_ids, batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染 bbox）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("voc2007", gt, predictions,
                          lambda img_id, info: IMAGE_DIR / f"{img_id}.jpg",
                          rel_name_fn=lambda img_id, info: f"{img_id}.jpg")
        print()

    # 评估
    print(f"计算指标（IoU@{IOU_MATCH_THRESHOLD}）...\n")
    results = {}
    for cls in VOC_CLASSES:
        results[cls] = evaluate_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD)

    summary = format_result_table(results, VOC_CLASSES, top_n=20)
    print(summary)
    print()

    all_aps = [results[cls]["ap"] for cls in VOC_CLASSES if cls in results]
    mAP = float(np.mean(all_aps)) if all_aps else 0.0

    result_data = {
        "timestamp": timestamp, "dataset": "PASCAL VOC 2007",
        "model": CONFIG["model_name"], "confidence_threshold": CONFIG["confidence_threshold"],
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(image_ids),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in VOC_CLASSES},
    }

    json_path, md_path = save_results(result_data, "voc2007", CONFIG["model_name"], timestamp)
    md_path.write_text("\n".join([
        "# PASCAL VOC 2007 Benchmark",
        f"- **模型**: {CONFIG['model_name']} | **conf**: {CONFIG['confidence_threshold']} | **mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    import argparse
    main()
