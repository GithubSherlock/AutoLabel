"""共享基准测试工具 — bbox / mask 指标 + 报告生成。

提取自 coco_benchmark.py / voc_benchmark.py 中的公共代码，
消除跨脚本的复制粘贴。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

from PIL import Image

from auto2dlabel.benchmarks import datetime, json, np
from auto2dlabel.models.detection import (
    PyTorchVisionModel,
    UltralyticsModel,
    create_detection_model,
    nms_per_class,
    sahi_infer_torchvision,
    sahi_infer_yolo,
)
from auto2dlabel.models.detection import (
    # SAHI 实现已迁移至 models/detection.py，此处保留引用（显式别名再导出）
    detect_image_sahi as detect_image_sahi,
)

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


def normalize_dota_class(name: str) -> str:
    """DOTA 类名归一：空格 ↔ 连字符统一（GT 连字符名 vs 模型空格名）。

    例: "small vehicle" ↔ "small-vehicle" → "small-vehicle"。
    """
    return name.strip().lower().replace(" ", "-")


_ImageKey = TypeVar("_ImageKey")  # 图像 ID 键类型：coco 等为 int，voc 为 str


def evaluate_per_class(
    gt: Mapping[_ImageKey, dict[str, Any]],
    pred: Mapping[_ImageKey, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    iou_fn: Callable[[list[float], list[float]], float] = compute_iou,
) -> dict[str, Any]:
    """逐类别计算检测指标（11-point interpolated AP）。

    Args:
        gt: {image_id: {"objects": [{"name": str, "bbox": [x1,y1,x2,y2] | "quad": 8 值}]}}
        pred: {image_id: [{"name": str, "bbox": ... | "quad": ..., "conf": float}]}
        class_name: 当前类别名。
        iou_threshold: 匹配 IoU 阈值。
        iou_fn: IoU 函数（HBB 默认 compute_iou；OBB 传 rotate_iou）。
            几何取对象 "quad" 键（旋转框 8 值四边形），缺省回退 "bbox"。

    Returns:
        {"precision", "recall", "f1", "ap", "gt_count", "pred_count"}
    """
    tp_list: list[float] = []
    fp_list: list[float] = []
    scores_list: list[float] = []
    n_gt = 0

    for img_id, gt_data in gt.items():
        gt_boxes = [
            (o["quad"] if "quad" in o else o["bbox"])
            for o in gt_data["objects"] if o["name"] == class_name
        ]
        n_gt += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)

        pred_boxes = [p for p in pred.get(img_id, []) if p["name"] == class_name]
        pred_boxes.sort(key=lambda p: p["conf"], reverse=True)

        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = iou_fn(
                        (pb["quad"] if "quad" in pb else pb["bbox"]), gb
                    )
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
        from pycocotools.mask import decode as _rle_decode
        from pycocotools.mask import frPyObjects
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


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """实例 mask → bbox [x1, y1, x2, y2]（numpy，无 cv2 依赖）。空 mask 返回 [0,0,0,0]。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def assign_mask_labels(
    output_masks: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    img_h: int,
    img_w: int,
    min_iou: float = 0.01,
) -> list[dict[str, Any]]:
    """box-prompted 分割输出 mask 按 mask-IoU 重匹配 GT 实例命名。

    对输出顺序/数量零假设（FastSAM 丢框、SAM2 丢 prompt、maskrcnn 自检测
    三种退化全部免疫——不依赖输入 bbox 与输出 mask 的一一对应）。

    Args:
        output_masks: [{"mask": (H,W) bool, "conf": float}, ...]（未命名输出）。
        gt_instances: [{"name": str, "mask": (H,W) bool}, ...]（实例 mask 互不重叠）。
        img_h, img_w: 图像尺寸（分辨率守卫基准）。
        min_iou: 命名阈值，best IoU 低于此值的输出 mask 丢弃（离群噪声）。

    Returns:
        [{"name": str, "mask": (H,W) bool, "conf": float}, ...]（已命名，顺序不变）。
    """
    named: list[dict[str, Any]] = []
    for om in output_masks:
        mask = om["mask"]
        if mask.shape[0] != img_h or mask.shape[1] != img_w:
            # 分辨率守卫：模型输出 mask 与图像尺寸不符时跳过（可能是推理分辨率缩放坑）
            print(f"[assign_mask_labels] 警告: mask 分辨率 {mask.shape} "
                  f"!= 图像 ({img_h}, {img_w})，已跳过")
            continue
        best_iou, best_name = 0.0, ""
        for g in gt_instances:
            iou = compute_mask_iou(mask, g["mask"])
            if iou > best_iou:
                best_iou, best_name = iou, str(g["name"])
        if best_iou >= min_iou and best_name:
            named.append({"name": best_name, "mask": mask, "conf": om.get("conf", 1.0)})
    return named


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
# 分类指标
# ================================================================


def evaluate_classification(
    gt: Mapping[str, dict[str, Any]],
    pred: Mapping[str, list[dict[str, Any]]],
    top_k: int = 5,
) -> dict[str, Any]:
    """逐类分类指标（top-1 / top-K 准确率）。

    Args:
        gt: {image_id: {"file_name": str, "label": str}}（单标签，英文类名）
        pred: {image_id: [{"label": str, "score": float}, ...]}（score 降序 top-K 列表）
        top_k: 评测的 top-K 值。

    Returns:
        {"top1", "top_k"(键名 f"top{top_k}"), "per_class":
         {cls: {"correct": int, "total": int, "accuracy": float}}}
        缺少 pred 键或 pred 为空一律计 miss（top1/topK 均不命中）。
    """
    top1_correct = 0
    topk_correct = 0
    per_class: dict[str, dict[str, Any]] = {}

    for img_id, gt_data in gt.items():
        label = gt_data["label"]
        stat = per_class.setdefault(label, {"correct": 0, "total": 0, "accuracy": 0.0})
        stat["total"] += 1
        labels = [p["label"] for p in pred.get(img_id, [])]
        if not labels:
            continue
        if labels[0] == label:
            top1_correct += 1
        if label in labels[:top_k]:
            topk_correct += 1
            stat["correct"] += 1

    total = len(gt)
    for stat in per_class.values():
        stat["accuracy"] = round(stat["correct"] / stat["total"], 4) if stat["total"] else 0.0

    return {
        "top1": round(top1_correct / total, 4) if total else 0.0,
        f"top{top_k}": round(topk_correct / total, 4) if total else 0.0,
        "per_class": per_class,
    }


def format_classification_table(
    results: dict[str, Any],
    all_classes: list[str],
    top_n: int = 20,
) -> str:
    """将逐类分类结果格式化为表格字符串（准确率口径）。"""
    class_acc = {cls: results["per_class"][cls]["accuracy"] for cls in all_classes
                 if cls in results["per_class"]}
    top_classes = sorted(class_acc, key=lambda k: class_acc[k], reverse=True)[:top_n]

    lines = [f"{'Class':>24s} {'Total':>7s} {'Correct':>9s} {'Acc':>8s}"]
    lines.append("-" * 52)
    for cls in top_classes:
        r = results["per_class"][cls]
        lines.append(
            f"{cls:>24s} {r['total']:>7d} {r['correct']:>9d} {r['accuracy']:>8.4f}"
        )
    lines.append("-" * 52)

    total = sum(r["total"] for r in results["per_class"].values())
    correct = sum(r["correct"] for r in results["per_class"].values())
    acc = correct / total if total else 0.0
    top1 = results["top1"]
    topk_key = next((k for k in results if k.startswith("top") and k != "top1"), "top5")
    topk = results[topk_key]
    lines.append(f"{f'总体 ({len(all_classes)} cls)':>24s} {total:>7d} {correct:>9d} {acc:>8.4f}")
    lines.append(f"  top-1: {top1:.4f}  |  {topk_key}: {topk:.4f}")

    return "\n".join(lines)


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
    parser.add_argument("--model", type=str, default="yolo26x.pt", help="检测模型")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--top-classes", type=int, default=20, help="展示前 N 类结果")
    parser.add_argument("--sahi", action="store_true", help="启用 SAHI 切片推理（大图检测）")
    parser.add_argument("--viz", action="store_true",
                        help="渲染预测结果到项目同级 Visualization/<数据集名>/（评测协议不变）")
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

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"SAHI 切片推理: slice={slice_size}px, overlap={overlap_ratio:.0%}")
    print(f"检测类别 ({len(all_cats)}): {', '.join(all_cats)}")

    model = create_detection_model(model_name, iou_threshold=iou)
    # 预加载权重，避免首帧推理耗时；_load 为具体模型类方法，Protocol 不可见
    cast(Any, model)._load()

    # 根据模型类型选择推理后端
    if isinstance(model, PyTorchVisionModel):
        _sahi_infer_fn = sahi_infer_torchvision
        print("  使用 torchvision SAHI 后端")
    elif isinstance(model, UltralyticsModel):
        _sahi_infer_fn = sahi_infer_yolo
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
        predictions[img_id] = nms_per_class(all_dets, iou)

    elapsed = _time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


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


# 延迟导入（避免循环依赖）
import cv2
