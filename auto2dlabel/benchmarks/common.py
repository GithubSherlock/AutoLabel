"""共享基准测试工具 — bbox / mask 指标 + 报告生成。

提取自 coco_benchmark.py / voc_benchmark.py 中的公共代码，
消除跨脚本的复制粘贴。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from PIL import Image

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmarks_outputs"
DATASETS_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets"
IOU_MATCH_THRESHOLD = 0.5


# ================================================================
# bbox 指标
# ================================================================

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """计算两个 box 的 IoU。box 格式: [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    a_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    b_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(a_area + b_area - inter)


def rotate_iou(quad_a: list[float], quad_b: list[float]) -> float:
    """计算两个旋转四边形（DOTA 8 值: x1,y1,...,x4,y4）的 IoU（shapely）。

    点序任意（Polygon 按几何求交）；退化/无效多边形返回 0.0。

    Args:
        quad_a: 四边形 A 的 8 个坐标值。
        quad_b: 四边形 B 的 8 个坐标值。

    Returns:
        IoU ∈ [0, 1]。
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("shapely 未安装，请运行: pip install shapely")

    if len(quad_a) < 8 or len(quad_b) < 8:
        return 0.0
    try:
        poly_a = Polygon([(quad_a[i], quad_a[i + 1]) for i in range(0, 8, 2)])
        poly_b = Polygon([(quad_b[i], quad_b[i + 1]) for i in range(0, 8, 2)])
    except ValueError:
        return 0.0

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    union = poly_a.union(poly_b).area
    if union == 0.0:
        return 0.0
    return float(poly_a.intersection(poly_b).area / union)


_ImageKey = TypeVar("_ImageKey")  # 图像 ID 键类型：coco 等为 int，voc 为 str


def evaluate_per_class(
    gt: Mapping[_ImageKey, dict[str, Any]],
    pred: Mapping[_ImageKey, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """逐类别计算 bbox 检测指标（11-point interpolated AP）。

    Args:
        gt: {image_id: {"objects": [{"name": str, "bbox": [x1,y1,x2,y2]}]}}
        pred: {image_id: [{"name": str, "bbox": [x1,y1,x2,y2], "conf": float}]}
        class_name: 当前类别名。
        iou_threshold: 匹配 IoU 阈值。

    Returns:
        {"precision", "recall", "f1", "ap", "gt_count", "pred_count"}
    """
    tp_list: list[float] = []
    fp_list: list[float] = []
    scores_list: list[float] = []
    n_gt = 0

    for img_id, gt_data in gt.items():
        gt_boxes = [o["bbox"] for o in gt_data["objects"] if o["name"] == class_name]
        n_gt += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)

        pred_boxes = [p for p in pred.get(img_id, []) if p["name"] == class_name]
        pred_boxes.sort(key=lambda p: p["conf"], reverse=True)

        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = compute_iou(pb["bbox"], gb)
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp_list.append(1.0)
                fp_list.append(0.0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0.0)
                fp_list.append(1.0)
            scores_list.append(pb["conf"])

    if not scores_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "gt_count": n_gt, "pred_count": 0}

    indices = np.argsort(scores_list)[::-1]
    tp_arr = np.array(tp_list)[indices]
    fp_arr = np.array(fp_list)[indices]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    prec = cum_tp / (cum_tp + cum_fp + 1e-10)
    rec = cum_tp / max(n_gt, 1)

    # 11-point interpolated AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        if mask.any():
            ap += float(np.max(prec[mask]))
    ap /= 11.0

    return {
        "precision": round(float(prec[-1]) if len(prec) > 0 else 0.0, 4),
        "recall": round(float(rec[-1]) if len(rec) > 0 else 0.0, 4),
        "f1": round(
            2 * float(prec[-1]) * float(rec[-1]) / (float(prec[-1]) + float(rec[-1]) + 1e-10)
            if len(prec) > 0 else 0.0, 4),
        "ap": round(ap, 4),
        "gt_count": n_gt,
        "pred_count": int(cum_tp[-1] + cum_fp[-1]) if len(cum_tp) > 0 else 0,
    }


# ================================================================
# mask 指标（光栅化方案）
# ================================================================

def polygon_to_mask(polygon: list[list[float]], h: int, w: int) -> np.ndarray:
    """将 COCO polygon 转为二进制 mask (H, W) bool。

    Args:
        polygon: [[x1,y1, x2,y2, ...], ...] 多轮廓。
        h, w: mask 尺寸（图像高度、宽度）。
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for contour_flat in polygon:
        if len(contour_flat) < 6:
            continue
        pts = np.array(contour_flat, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], (1, 1, 1, 1))  # cv2 stub 要求 color 为 Scalar，标量 1 不被接受
    return mask.astype(bool)


def coco_seg_to_mask(seg: list | dict, h: int, w: int) -> np.ndarray:
    """将 COCO segmentation 转为二进制 mask。

    COCO segmentation 有两种格式:
      - polygon: [[x1,y1, x2,y2, ...], ...]
      - RLE: {"counts": "...", "size": [h, w]}
    """
    if isinstance(seg, dict):
        return rle_to_mask(seg, h, w)
    elif isinstance(seg, list):
        return polygon_to_mask(seg, h, w)
    else:
        raise TypeError(f"不支持的 segmentation 类型: {type(seg)}")


def rle_to_mask(rle: dict, h: int, w: int) -> np.ndarray:
    """将 COCO RLE 转为二进制 mask (H, W) bool。

    COCO RLE 的 counts 有两种编码:
      - 压缩:  base64 字符串
      - 未压缩: 整数列表
    frPyObjects() 统一转换为 pycocotools 内部格式后才能 decode。
    """
    try:
        from pycocotools.mask import frPyObjects, decode as _rle_decode
    except ImportError:
        raise ImportError("pycocotools 未安装。请运行: pip install pycocotools")
    rle_obj = frPyObjects(cast(Any, rle), h, w)  # stub 未覆盖运行时合法的 dict RLE 格式
    return _rle_decode(rle_obj).astype(bool)


def nuscenes_rle_to_mask(token: str, nuimg) -> np.ndarray:
    """通过 nuscenes-devkit 将 nuImages RLE token 转为 mask。"""
    try:
        return nuimg.get_segmentation(token)
    except Exception:
        # 回退：直接从数据中解码
        raise


def compute_mask_iou(maskA: np.ndarray, maskB: np.ndarray) -> float:
    """计算两个二进制 mask 的 IoU。"""
    inter = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def compute_dice(maskA: np.ndarray, maskB: np.ndarray) -> float:
    """计算两个二进制 mask 的 Dice 系数。"""
    inter = np.logical_and(maskA, maskB).sum()
    total = maskA.sum() + maskB.sum()
    if total == 0:
        return 0.0
    return float(2 * inter / total)


def evaluate_mask_per_class(
    gt: dict[int, dict[str, Any]],
    pred: dict[int, list[dict]],
    class_name: str,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """逐类别计算实例分割 mask 指标。

    GT 格式: {image_id: {"objects": [{"name": str, "mask": np.ndarray (H,W) bool}, ...]}}
    Pred 格式: {image_id: [{"name": str, "mask": np.ndarray (H,W) bool, "conf": float}, ...]}

    每个预测匹配最高 mask-IoU 的未匹配 GT（大于阈值 = TP，否则 FP）。
    使用 11-point interpolated AP（与 bbox 一致）。
    """
    tp_list: list[float] = []
    fp_list: list[float] = []
    scores_list: list[float] = []
    all_ious: list[float] = []
    all_dices: list[float] = []
    n_gt = 0

    for img_id, gt_data in gt.items():
        gt_objs = [o for o in gt_data["objects"] if o["name"] == class_name]
        n_gt += len(gt_objs)
        gt_matched = [False] * len(gt_objs)

        pred_objs = [p for p in pred.get(img_id, []) if p["name"] == class_name]
        pred_objs.sort(key=lambda p: p.get("conf", 1.0), reverse=True)

        for pb in pred_objs:
            best_iou, best_idx = 0.0, -1
            for j, go in enumerate(gt_objs):
                if not gt_matched[j]:
                    iou = compute_mask_iou(pb["mask"], go["mask"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp_list.append(1.0)
                fp_list.append(0.0)
                gt_matched[best_idx] = True
                all_ious.append(best_iou)
                all_dices.append(compute_dice(pb["mask"], gt_objs[best_idx]["mask"]))
            else:
                tp_list.append(0.0)
                fp_list.append(1.0)
            scores_list.append(pb.get("conf", 1.0))

    if not scores_list:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0,
            "miou": 0.0, "mdice": 0.0,
            "gt_count": n_gt, "pred_count": 0, "matched_count": 0,
        }

    indices = np.argsort(scores_list)[::-1]
    tp_arr = np.array(tp_list)[indices]
    fp_arr = np.array(fp_list)[indices]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    prec = cum_tp / (cum_tp + cum_fp + 1e-10)
    rec = cum_tp / max(n_gt, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        if mask.any():
            ap += float(np.max(prec[mask]))
    ap /= 11.0

    return {
        "precision": round(float(prec[-1]) if len(prec) > 0 else 0.0, 4),
        "recall": round(float(rec[-1]) if len(rec) > 0 else 0.0, 4),
        "f1": round(
            2 * float(prec[-1]) * float(rec[-1]) / (float(prec[-1]) + float(rec[-1]) + 1e-10)
            if len(prec) > 0 else 0.0, 4),
        "ap": round(ap, 4),
        "miou": round(float(np.mean(all_ious)) if all_ious else 0.0, 4),
        "mdice": round(float(np.mean(all_dices)) if all_dices else 0.0, 4),
        "gt_count": n_gt,
        "pred_count": int(cum_tp[-1] + cum_fp[-1]) if len(cum_tp) > 0 else 0,
        "matched_count": int(sum(tp_list)),
    }


# ================================================================
# 报告生成
# ================================================================


def format_result_table(
    results: dict[str, dict[str, Any]],
    all_classes: list[str],
    top_n: int = 20,
) -> str:
    """将逐类结果格式化为表格字符串。"""
    class_ap = {cls: results[cls]["ap"] for cls in all_classes if cls in results}
    top_classes = sorted(class_ap, key=lambda k: class_ap[k], reverse=True)[:top_n]

    lines = [f"{'Class':>20s} {'GT':>7s} {'Pred':>7s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'AP':>8s}"]
    lines.append("-" * 74)
    for cls in top_classes:
        r = results[cls]
        lines.append(
            f"{cls:>20s} {r['gt_count']:>7d} {r['pred_count']:>7d} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['ap']:>8.4f}"
        )
    lines.append("-" * 74)

    all_aps = [results[cls]["ap"] for cls in all_classes if cls in results]
    mAP = float(np.mean(all_aps)) if all_aps else 0.0
    avg_r = float(np.mean([results[cls]["recall"] for cls in all_classes if cls in results])) if all_aps else 0.0
    avg_p = float(np.mean([results[cls]["precision"] for cls in all_classes if cls in results])) if all_aps else 0.0
    total_cls = len(all_classes)
    lines.append(
        f"{f'mAP@0.5 (all {total_cls} cls)':>20s} {'':>7s} {'':>7s} "
        f"{avg_p:>8.4f} {avg_r:>8.4f} {'':>8s} {mAP:>8.4f}"
    )

    return "\n".join(lines)


def format_mask_result_table(
    results: dict[str, dict[str, Any]],
    all_classes: list[str],
    top_n: int = 20,
) -> str:
    """将逐类分割结果格式化为表格字符串（含 mIoU / mDice）。"""
    class_ap = {cls: results[cls]["ap"] for cls in all_classes if cls in results}
    top_classes = sorted(class_ap, key=lambda k: class_ap[k], reverse=True)[:top_n]

    lines = [
        f"{'Class':>20s} {'GT':>7s} {'Pred':>7s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'AP':>8s} {'mIoU':>8s}"
    ]
    lines.append("-" * 90)
    for cls in top_classes:
        r = results[cls]
        lines.append(
            f"{cls:>20s} {r['gt_count']:>7d} {r['pred_count']:>7d} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
            f"{r['ap']:>8.4f} {r['miou']:>8.4f}"
        )
    lines.append("-" * 90)

    all_aps = [results[cls]["ap"] for cls in all_classes if cls in results]
    mAP = float(np.mean(all_aps)) if all_aps else 0.0
    mIoU = float(np.mean([results[cls]["miou"] for cls in all_classes if cls in results])) if all_aps else 0.0
    mDice = float(np.mean([results[cls]["mdice"] for cls in all_classes if cls in results])) if all_aps else 0.0
    total_cls = len(all_classes)
    lines.append(
        f"{f'mAP@0.5 (all {total_cls} cls)':>20s} {'':>7s} {'':>7s} "
        f"{'':>8s} {'':>8s} {'':>8s} {mAP:>8.4f} {mIoU:>8.4f}"
    )
    lines.append(f"  mIoU: {mIoU:.4f}  |  mDice: {mDice:.4f}")

    return "\n".join(lines)


def save_results(
    result_data: dict,
    dataset: str,
    model: str,
    ts: str | None = None,
) -> tuple[Path, Path]:
    """保存 JSON + Markdown 报告到 benchmarks_outputs/。

    Returns:
        (json_path, md_path)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    prefix = f"{dataset}_{model}_{ts}"

    json_path = OUTPUT_DIR / f"{prefix}.json"
    json_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))

    md_path = OUTPUT_DIR / f"{prefix}.md"
    return json_path, md_path


def build_parser(description: str) -> argparse.ArgumentParser:
    """构建标准的 CLI argument parser。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max-images", type=int, default=0, help="最多图像数（0=全部）")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="检测模型")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--top-classes", type=int, default=20, help="展示前 N 类结果")
    parser.add_argument("--sahi", action="store_true", help="启用 SAHI 切片推理（大图检测）")
    return parser


# ================================================================
# SAHI 切片推理（Slicing Aided Hyper Inference）
# ================================================================


def run_detection_sahi(
    gt: dict[int, dict[str, Any]],
    image_dir: Path,
    model_name: str,
    conf: float,
    iou: float,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
) -> dict[int, list[dict]]:
    """SAHI 切片推理：将大图切为小块分别检测后合并。

    解决航拍/大分辨率图中 YOLO resize 导致小物体丢失的问题。
    每一切片用模型独立推理，坐标映射回原图，跨切片 NMS 去重。

    Args:
        gt: GT 字典（仅用于获取文件列表和类别）。
        image_dir: 图像目录。
        model_name: 检测模型名。
        conf: 置信度阈值。
        iou: NMS IoU 阈值。
        slice_size: 切片尺寸（正方形）。
        overlap_ratio: 切片重叠比例。
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.detection import (
        create_detection_model, UltralyticsModel, PyTorchVisionModel,
    )
    from auto2dlabel.models.model_catalog import COCO_CLASSES

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"SAHI 切片推理: slice={slice_size}px, overlap={overlap_ratio:.0%}")
    print(f"检测类别 ({len(all_cats)}): {', '.join(all_cats)}")

    model = create_detection_model(model_name, iou_threshold=iou)
    # 预加载权重，避免首帧推理耗时；_load 为具体模型类方法，Protocol 不可见
    cast(Any, model)._load()

    # 根据模型类型选择推理后端
    if isinstance(model, PyTorchVisionModel):
        _sahi_infer_fn = _sahi_infer_torchvision
        print(f"  使用 torchvision SAHI 后端")
    elif isinstance(model, UltralyticsModel):
        _sahi_infer_fn = _sahi_infer_yolo
    else:
        print("  ⚠ SAHI 不支持此模型类型，回退到标准推理")
        return _run_detection_fallback(gt, image_dir, model, all_cats, conf)

    step = int(slice_size * (1 - overlap_ratio))
    predictions: dict[int, list[dict]] = {}

    total = len(gt)
    img_ids = list(gt.keys())
    iterator = tqdm(enumerate(img_ids), total=total, desc=f"SAHI 检测（{model_name}）", unit="img") if has_tqdm else enumerate(img_ids)

    import time as _time
    _t0 = _time.time()

    for i, img_id in iterator:
        img_path = image_dir / gt[img_id]["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # 如果图片小于 slice_size，直接推理
        if W <= slice_size and H <= slice_size:
            all_dets = _sahi_infer_fn(model, img, all_cats, conf)
        else:
            all_dets = []
            y_starts = list(range(0, H, step))
            x_starts = list(range(0, W, step))

            for y in y_starts:
                for x in x_starts:
                    x2 = min(x + slice_size, W)
                    y2 = min(y + slice_size, H)
                    x1 = max(0, x2 - slice_size)
                    y1 = max(0, y2 - slice_size)

                    tile = img.crop((x1, y1, x2, y2))
                    for det in _sahi_infer_fn(model, tile, all_cats, conf):
                        bx1, by1, bx2, by2 = det["bbox"]
                        all_dets.append({
                            "name": det["name"],
                            "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                            "conf": det["conf"],
                        })

        # 跨切片 NMS 合并
        predictions[img_id] = _nms_per_class(all_dets, iou)

    elapsed = _time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


def _sahi_infer_yolo(model, tile: Image.Image, prompts: list[str], conf: float) -> list[dict]:
    """YOLO SAHI 推理：PIL Image → YOLO → dets。"""
    results = model._model(tile, conf=conf, iou=0.5, device=model._device, verbose=False)
    return _yolo_results_to_dets(results, prompts)


def _sahi_infer_torchvision(model, tile: Image.Image, prompts: list[str], conf: float) -> list[dict]:
    """Torchvision SAHI 推理：PIL Image → Tensor → Faster R-CNN → dets。"""
    import torch
    from torchvision.transforms import functional as F

    tile_tensor = F.to_tensor(tile).to(model._device)
    with torch.no_grad():
        outputs = model._model([tile_tensor])[0]

    from auto2dlabel.models.model_catalog import COCO_CLASSES

    dets = []
    for box, label_idx, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        conf_val = float(score)
        if conf_val < conf:
            continue
        cls_id = int(label_idx) - 1  # 1-based → 0-based
        if cls_id < 0 or cls_id >= len(COCO_CLASSES):
            continue
        label = COCO_CLASSES[cls_id]
        # prompt 过滤
        label_lower = label.lower()
        if not any(p.lower() in label_lower or label_lower in p.lower() for p in prompts):
            continue
        x1, y1, x2, y2 = box.tolist()
        dets.append({
            "name": label,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": conf_val,
        })
    return dets


def _yolo_results_to_dets(results: list, prompts: list[str]) -> list[dict]:
    """将 Ultralytics YOLO 推理结果转为检测字典列表。"""
    from auto2dlabel.models.model_catalog import COCO_CLASSES

    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id >= len(COCO_CLASSES):
                continue
            label = COCO_CLASSES[cls_id]
            # 按 prompt 过滤
            label_lower = label.lower()
            if not any(p.lower() in label_lower or label_lower in p.lower() for p in prompts):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "name": label,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(box.conf[0]),
            })
    return dets


def _nms_per_class(dets: list[dict], iou_threshold: float) -> list[dict]:
    """按类别分组执行 IoU NMS，返回去重后的检测列表。"""
    if not dets:
        return []

    # 按类别分组
    by_class: dict[str, list[dict]] = {}
    for d in dets:
        by_class.setdefault(d["name"], []).append(d)

    kept = []
    for cls_name, cls_dets in by_class.items():
        # 按置信度降序
        cls_dets.sort(key=lambda d: d["conf"], reverse=True)
        boxes = [d["bbox"] for d in cls_dets]

        # 贪心 NMS
        suppressed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            for j in range(i + 1, len(cls_dets)):
                if suppressed[j]:
                    continue
                if _box_iou(boxes[i], boxes[j]) > iou_threshold:
                    suppressed[j] = True

    return kept


def _box_iou(boxA: list[float], boxB: list[float]) -> float:
    """两个 bbox 的 IoU（[x1,y1,x2,y2] 格式）。"""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def _run_detection_fallback(
    gt: dict[int, dict[str, Any]], image_dir: Path,
    model, all_cats: list[str], conf: float,
) -> dict[int, list[dict]]:
    """标准检测（非 SAHI 回退）。"""
    predictions: dict[int, list[dict]] = {}
    for img_id, info in gt.items():
        img_path = image_dir / info["file_name"]
        if not img_path.exists():
            continue
        results = model.detect(str(img_path), all_cats, confidence_threshold=conf)
        predictions[img_id] = [
            {"name": r.label, "bbox": [r.x, r.y, r.x + r.width, r.y + r.height], "conf": r.confidence}
            for r in results
        ]
    return predictions


# ================================================================
# SAHI 单图推理（供标注 pipeline 复用）
# ================================================================


def detect_image_sahi(
    model,
    image_path: str,
    prompts: list[str],
    confidence_threshold: float = 0.3,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
) -> list[dict]:
    """SAHI 切片推理单张图像，返回检测 dict 列表。

    将大图切为重叠的 slice_size × slice_size 小块，分别推理后
    跨切片 NMS 合并。适用于航拍/高分辨率图中 YOLO resize 导致
    小物体丢失的场景。

    Args:
        model: 已加载的 DetectionModel 实例（需有 _model 属性）。
        image_path: 图像路径。
        prompts: 检测类别（英文名）。
        confidence_threshold: 置信度阈值。
        slice_size: 切片尺寸（正方形，像素）。
        overlap_ratio: 相邻切片重叠比例。

    Returns:
        [{"name": str, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
    """
    from auto2dlabel.models.detection import UltralyticsModel, PyTorchVisionModel

    # 确保模型已加载
    if hasattr(model, "_load"):
        model._load()

    # 根据模型类型选择推理后端
    if isinstance(model, PyTorchVisionModel):
        _infer_fn = _sahi_infer_torchvision
    elif isinstance(model, UltralyticsModel):
        _infer_fn = _sahi_infer_yolo
    else:
        # 回退：直接调 model.detect()
        from auto2dlabel.models.detection import DetectionResult
        results = model.detect(image_path, prompts, confidence_threshold)
        return [
            {"name": r.label, "bbox": [r.x, r.y, r.x + r.width, r.y + r.height], "conf": r.confidence}
            for r in results
        ]

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # 小图直接推理
    if W <= slice_size and H <= slice_size:
        return _infer_fn(model, img, prompts, confidence_threshold)

    # 切片推理
    step = int(slice_size * (1 - overlap_ratio))
    all_dets: list[dict] = []
    y_starts = list(range(0, H, step))
    x_starts = list(range(0, W, step))

    for y in y_starts:
        for x in x_starts:
            x2 = min(x + slice_size, W)
            y2 = min(y + slice_size, H)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)

            tile = img.crop((x1, y1, x2, y2))
            for det in _infer_fn(model, tile, prompts, confidence_threshold):
                bx1, by1, bx2, by2 = det["bbox"]
                all_dets.append({
                    "name": det["name"],
                    "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                    "conf": det["conf"],
                })

    # 跨切片 NMS 合并（用 model 的 iou 阈值）
    iou_threshold = getattr(model, "_iou", 0.5)
    return _nms_per_class(all_dets, iou_threshold)


# 延迟导入（避免循环依赖）
import cv2
