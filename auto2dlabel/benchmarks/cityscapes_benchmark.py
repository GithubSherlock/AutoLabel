#!/usr/bin/env python3
"""Cityscapes val Instance Segmentation Benchmark — mask 预测 vs GT。

cityscapes 的 instanceIds.png 编码了实例 mask（像素值 = instance_id * 1000 + class_id）。
对 thing 类（8 类）评估 mask 指标。

三种运行模式（消融矩阵，定位 mAP 根因）：
- 默认 full pipeline：检测 conf≥prompt-conf → seg_model box-prompt 两段式
- --box-prompted：GT mask 派生 bbox 直接作 seg_model prompt（检测零误差）
- --det-only：仅检测 bbox mAP（GT mask 派生 bbox，隔离检测器质量）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (  # noqa: E402
    IOU_MATCH_THRESHOLD,
    OUTPUT_DIR,
    assign_mask_labels,
    build_parser,
    evaluate_mask_per_class,
    evaluate_per_class,
    format_mask_result_table,
    format_result_table,
    polygon_to_mask,
    run_detection_sahi,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_cityscapes_val  # noqa: E402
from auto2dlabel.cli_execute import _is_self_detect_seg  # noqa: E402
from PIL import Image  # noqa: E402  # load_cityscapes_ground_truth 用（原靠 __main__ 块注入）

# ── 配置 ──────────────────────────────────────────────────────
DET_MODEL = "yolo26x.pt"
SEG_MODEL = "sam2_l.pt"

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


def _mask_bbox(mask: np.ndarray) -> list[int]:
    """实例 mask → bbox [x1, y1, x2, y2]（numpy，无 cv2 依赖）。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _gt_bbox_view(gt: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """GT 视图：objects 的 mask → bbox（供 evaluate_per_class / SAHI 使用）。"""
    return {
        img_id: {
            "file_name": g["file_name"],
            "objects": [
                {"name": o["name"], "bbox": _mask_bbox(o["mask"])}
                for o in g["objects"]
            ],
        }
        for img_id, g in gt.items()
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

    instanceIds.png 编码（官方）: pixel_value = label_id * 1000 + instance_idx
    （如 24000 = person 第 0 个实例）；stuff 类像素为其 labelId（1..23 等）。
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
            class_id = inst_id // 1000
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
    seg_model_name: str,
    det_model_name: str,
    prompt_conf: float = 0.3,
) -> dict[int, list[dict[str, Any]]]:
    """两段式：检测 conf≥prompt_conf → seg_model box-prompt。"""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from PIL import Image as _Image

    from auto2dlabel.models.detection import create_detection_model
    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"分割类别 ({len(all_cats)}): {', '.join(all_cats)}")

    det_model = create_detection_model(det_model_name, iou_threshold=iou)
    seg_model = create_segmentation_model(seg_model_name)

    predictions: dict[int, list[dict[str, Any]]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = (
        tqdm(enumerate(img_ids), total=total, desc=f"分割（{seg_model_name}）", unit="img")
        if has_tqdm else enumerate(img_ids)
    )

    _t0 = time.time()
    for i, img_id in iterator:
        filename = gt[img_id]["file_name"]
        img_path = img_dir / filename
        if not img_path.exists():
            continue

        # Step 1: 检测
        det_results = det_model.detect(
            str(img_path), all_cats, confidence_threshold=conf,
        )
        high_conf = [r for r in det_results if r.confidence >= prompt_conf]
        if not high_conf:
            predictions[img_id] = []
            continue

        # Step 2: 分割
        bboxes = [
            Bbox(x=r.x, y=r.y, width=r.width, height=r.height,
                 label=r.label, confidence=r.confidence)
            for r in high_conf
        ]
        masks = seg_model.generate(str(img_path), bboxes)

        # 读取原图尺寸
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


def run_box_prompted_segmentation(
    gt: dict[int, dict[str, Any]],
    img_dir: Path,
    seg_model_name: str,
) -> dict[int, list[dict[str, Any]]]:
    """box-prompted：GT mask 派生 bbox 直接作 prompt（跳过检测，隔离分割器质量）。

    输出 mask 经 assign_mask_labels 按 mask-IoU 重匹配 GT 命名（对输出
    顺序/数量零假设，FastSAM 丢框 / SAM2 丢 prompt 均免疫）。
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from PIL import Image as _Image

    from auto2dlabel.models.segmentation import create_segmentation_model
    from auto2dlabel.schema.annotation import Bbox

    seg_model = create_segmentation_model(seg_model_name)

    predictions: dict[int, list[dict[str, Any]]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = (
        tqdm(enumerate(img_ids), total=total, desc=f"box-prompted（{seg_model_name}）", unit="img")
        if has_tqdm else enumerate(img_ids)
    )

    _t0 = time.time()
    for i, img_id in iterator:
        filename = gt[img_id]["file_name"]
        img_path = img_dir / filename
        if not img_path.exists():
            continue

        objects = gt[img_id]["objects"]
        bboxes = [Bbox(
            x=b[0], y=b[1], width=b[2] - b[0], height=b[3] - b[1],
            label=o["name"], confidence=1.0,
        ) for o, b in ((o, _mask_bbox(o["mask"])) for o in objects)]
        masks = seg_model.generate(str(img_path), bboxes)

        im = _Image.open(img_path)
        h, w = im.height, im.width

        # 未命名输出 mask → mask-IoU 重匹配 GT 实例命名
        raw_masks = []
        for mask_obj in masks:
            mask_bool = polygon_to_mask(mask_obj.segmentation, h, w)
            raw_masks.append({
                "mask": mask_bool,
                "conf": getattr(mask_obj.bbox, "confidence", 1.0),
            })
        predictions[img_id] = assign_mask_labels(raw_masks, objects, h, w)

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


def run_detection_only(
    gt: dict[int, dict[str, Any]],
    img_dir: Path,
    conf: float,
    iou: float,
    det_model_name: str,
) -> dict[int, list[dict[str, Any]]]:
    """det-only：仅检测 bbox mAP（GT mask 派生 bbox，隔离检测器质量）。"""
    from auto2dlabel.models.detection import create_detection_model

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"检测类别 ({len(all_cats)}): {', '.join(all_cats)}")

    det_model = create_detection_model(det_model_name, iou_threshold=iou)

    predictions: dict[int, list[dict[str, Any]]] = {}
    _t0 = time.time()
    for img_id in sorted(gt.keys()):
        img_path = img_dir / gt[img_id]["file_name"]
        if not img_path.exists():
            continue
        det_results = det_model.detect(
            str(img_path), all_cats, confidence_threshold=conf,
        )
        predictions[img_id] = [
            {
                "name": r.label,
                "bbox": [r.x, r.y, r.x + r.width, r.y + r.height],
                "conf": r.confidence,
            }
            for r in det_results
        ]

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {len(gt) / elapsed:.1f} img/s")
    return predictions


def run_self_detect_segmentation(
    gt: dict[int, dict[str, Any]],
    img_dir: Path,
    seg_model_name: str,
) -> dict[int, list[dict[str, Any]]]:
    """self-detect：分割模型自带检测（maskrcnn/sam3 等），跳过外部检测步。

    与 CLI 路由（cli_execute._is_self_detect_seg）语义一致：maskrcnn 类
    generate 忽略 bbox 坐标自行检测，输出 label 由模型类别映射决定。
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from PIL import Image as _Image

    from auto2dlabel.models.segmentation import create_segmentation_model

    seg_model = create_segmentation_model(seg_model_name)

    predictions: dict[int, list[dict[str, Any]]] = {}
    total = len(gt)
    img_ids = sorted(gt.keys())
    iterator = (
        tqdm(enumerate(img_ids), total=total, desc=f"self-detect（{seg_model_name}）", unit="img")
        if has_tqdm else enumerate(img_ids)
    )

    _t0 = time.time()
    for i, img_id in iterator:
        img_path = img_dir / gt[img_id]["file_name"]
        if not img_path.exists():
            continue

        masks = seg_model.generate(str(img_path), [])

        # 读取原图尺寸
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

def main() -> None:
    parser = build_parser("Cityscapes Instance Segmentation Benchmark")
    parser.set_defaults(model=DET_MODEL)
    parser.add_argument("--seg-model", type=str, default=SEG_MODEL,
                        help="分割模型（FastSAM-s.pt / sam2_l.pt / sam3.pt 等）")
    parser.add_argument("--prompt-conf", type=float, default=0.3,
                        help="两段式检测框过滤阈值（0.0 = 不过滤）")
    parser.add_argument("--box-prompted", action="store_true",
                        help="GT bbox 直接作分割 prompt（跳过检测步）")
    parser.add_argument("--det-only", action="store_true",
                        help="仅检测 bbox mAP 诊断模式（跳过分割步）")
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

    # 推理（三种模式）
    if args.det_only:
        mode = "det-only"
        print(f"det-only 模式（检测模型 {args.model}，conf={args.conf}）...")
        if args.sahi:
            # SAHI 切片推理（大图小目标：2048×1024 resize 640 后小目标消失）
            predictions = run_detection_sahi(
                _gt_bbox_view(gt), img_dir, args.model, args.conf, args.iou,
            )
        else:
            predictions = run_detection_only(gt, img_dir, args.conf, args.iou, args.model)
    elif args.box_prompted:
        mode = "box-prompted"
        print(f"box-prompted 模式（分割模型 {args.seg_model}，GT bbox 直接 prompt）...")
        predictions = run_box_prompted_segmentation(gt, img_dir, args.seg_model)
    elif _is_self_detect_seg(args.seg_model):
        mode = "self-detect"
        print(f"self-detect 模式（分割模型 {args.seg_model} 自带检测，跳过检测步）...")
        predictions = run_self_detect_segmentation(gt, img_dir, args.seg_model)
    else:
        mode = "two-stage"
        print(f"两段式（检测 {args.model} conf≥{args.prompt_conf} → 分割 {args.seg_model}）...")
        predictions = run_segmentation(
            gt, img_dir, args.conf, args.iou, args.seg_model, args.model, args.prompt_conf,
        )
    print()

    # 可视化（--viz：镜像相对路径渲染；det-only 画 bbox，其余画 mask）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("cityscapes", gt, predictions,
                          lambda img_id, info: img_dir / info["file_name"])
        print()

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))

    if args.det_only:
        # bbox 指标（GT mask 派生 bbox）
        print(f"计算 bbox 指标（IoU@{IOU_MATCH_THRESHOLD}）...\n")
        gt_bbox = _gt_bbox_view(gt)
        results = {
            cls: evaluate_per_class(gt_bbox, predictions, cls, IOU_MATCH_THRESHOLD)
            for cls in all_cats
        }
        summary = format_result_table(results, all_cats, top_n=args.top_classes)
        print(summary)
        print()
        mAP = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))
        result_data = {
            "timestamp": ts, "dataset": "cityscapes val",
            "mode": mode, "det_model": args.model, "sahi": args.sahi,
            "confidence_threshold": args.conf,
            "iou_match_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
            "summary": {"mAP@0.5": round(mAP, 4)},
            "per_class": {cls: results[cls] for cls in all_cats},
        }
        save_model = args.model
    else:
        print(f"计算 mask 指标（mask IoU@{IOU_MATCH_THRESHOLD}）...\n")
        results = {
            cls: evaluate_mask_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD)
            for cls in all_cats
        }
        summary = format_mask_result_table(results, all_cats, top_n=args.top_classes)
        print(summary)
        print()
        mAP = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))
        result_data = {
            "timestamp": ts, "dataset": "cityscapes val",
            "mode": mode, "seg_model": args.seg_model,
            "det_model": "" if args.box_prompted else args.model,
            "prompt_conf": args.prompt_conf,
            "confidence_threshold": args.conf,
            "mask_iou_threshold": IOU_MATCH_THRESHOLD, "image_count": len(gt),
            "summary": {"mAP@0.5": round(mAP, 4)},
            "per_class": {cls: results[cls] for cls in all_cats},
        }
        save_model = args.seg_model

    json_path, md_path = save_results(result_data, "cityscapes", save_model, ts)
    md_path.write_text("\n".join([
        f"# Cityscapes Instance Segmentation Benchmark（{mode}）",
        f"- **分割模型**: {args.seg_model} | **检测 backbone**: {args.model}"
        f" | **mask mAP@0.5**: {mAP:.4f}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
