#!/usr/bin/env python3
"""COCO val 2017 姿态基准 — OKS 评测协议（v0.6 Phase 3b）。

协议（COCO 17 点标准口径，11-point 插值与项目检测协议同款）：
- GT = person_keypoints_val2017.json（iscrowd=1 排除；num_keypoints=0 保留——
  永远不能 TP，与 COCO 官方口径一致）
- 匹配：预测按 conf 降序贪心，与 GT person bbox IoU ≥ 0.5，每个 GT 至多匹配一次
- TP 判定：OKS > t（t ∈ [.50:.95:.05] 10 阈值）；OKS 用标准 σ 表（OKS_SIGMAS）
- 汇总：mAP = 10 阈值 11-point AP 平均 + AP50/AP75
- 分层：small（GT bbox 面积 < 32²）/ dense（图内 GT person ≥ 8）——
  YOLO-pose 已知弱点在 small/dense，P3b 验收要求分层出表

用法：
    python3 -m auto2dlabel.benchmarks.pose_benchmark --model yolo11n-pose.pt
    python3 -m auto2dlabel.benchmarks.pose_benchmark --model rtmpose_l --max-images 500
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from auto2dlabel.benchmarks import datetime, json, np, time  # noqa: E402
from auto2dlabel.benchmarks.common import (  # noqa: E402
    OUTPUT_DIR,
    _batch_or_fallback,
    compute_iou,
    sample_image_paths,
    save_results,
)

# ---- 配置 ----
COCO_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets" / "COCO2017"
IMAGE_DIR = COCO_ROOT / "val2017"
GT_JSON = COCO_ROOT / "annotations" / "person_keypoints_val2017.json"

# COCO 官方 17 点 σ（鼻子/眼/耳/肩/肘/腕/髋/膝/踝）
OKS_SIGMAS = np.array(
    [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]
)
OKS_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
IOU_MATCH_THRESHOLD = 0.5
SMALL_AREA = 32.0 * 32.0  # 小目标分层阈值（GT bbox 面积，像素²）
DENSE_PERSONS = 8         # 密集分层阈值（图内 GT person 数）


# ============================================================
# 1. 协议纯函数（可单测）
# ============================================================

def compute_oks(
    gt_kpts: list[tuple[float, float, float]],
    pred_kpts: list[tuple[float, float, float]],
    gt_area: float,
) -> float:
    """标准 COCO OKS：Σ_i exp(-d_i²/(2σ_i²·area))·δ(v_i>0) / Σ δ(v_i>0)。

    只看 GT 可见点（v>0）；预测缺 17 点时缺位按 (0,0) 计（宁多勿漏不崩溃）。
    """
    if gt_area <= 0:
        return 0.0
    visible = [i for i in range(17) if gt_kpts[i][2] > 0]
    if not visible:
        return 0.0
    total = 0.0
    for i in visible:
        gx, gy, _ = gt_kpts[i]
        px, py = pred_kpts[i][:2] if i < len(pred_kpts) else (0.0, 0.0)
        d2 = (gx - px) ** 2 + (gy - py) ** 2
        total += math.exp(-d2 / (2.0 * OKS_SIGMAS[i] ** 2 * gt_area))
    return total / len(visible)


def match_predictions(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> list[tuple[int, float]]:
    """贪心匹配：pred 按 conf 降序，每 pred 取 IoU 最大的未用 GT。

    Returns: [(gt_idx, oks)]——GT 至多出现一次；oks 在匹配时即算好（纯函数单测点）。
    """
    matches: list[tuple[int, float]] = []
    gt_used: set[int] = set()
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    for pi in order:
        best_j, best_iou = -1, 0.0
        for j, gt in enumerate(gts):
            if j in gt_used:
                continue
            iou = compute_iou(preds[pi]["bbox"], gt["bbox"])
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= iou_threshold:
            gt_used.add(best_j)
            oks = compute_oks(gts[best_j]["keypoints"], preds[pi]["keypoints"],
                              gts[best_j]["area"])
            matches.append((best_j, oks))
    return matches


def ap_11point(oks_scores: list[float], gt_total: int, t: float) -> float:
    """单一阈值 t 的 11-point 插值 AP（OKS > t 为 TP；与检测协议同款）。"""
    if gt_total <= 0 or not oks_scores:
        return 0.0
    scores = sorted(oks_scores, reverse=True)
    tp = np.cumsum([1.0 if s > t else 0.0 for s in scores])
    recall = tp / gt_total
    precision = tp / np.arange(1, len(scores) + 1)
    ap = 0.0
    for r in np.arange(0.0, 1.01, 0.1):
        above = precision[recall >= r]
        ap += float(above.max()) if len(above) else 0.0
    return ap / 11.0


def summarize_pose(
    gts: list[dict[str, Any]],
    matches: list[tuple[int, float]],
) -> dict[str, dict[str, Any]]:
    """分层汇总：all / small(<32²) / dense(≥8 人/图)。

    gts 每项需含 area 与 img_gt_count；matches 为 [(gt_idx, oks)]。
    返回 {子集名: {gt, matched, mAP, AP50, AP75, per_threshold}}。
    """
    subsets = [
        ("all", "全部", [True] * len(gts)),
        ("small", "small(<32²)", [g["area"] < SMALL_AREA for g in gts]),
        ("dense", "dense(≥8人/图)", [g["img_gt_count"] >= DENSE_PERSONS for g in gts]),
    ]
    results: dict[str, dict[str, Any]] = {}
    for name, _label, keep in subsets:
        idx_set = {j for j, k in enumerate(keep) if k}
        oks = [o for j, o in matches if j in idx_set]
        aps = {t: ap_11point(oks, len(idx_set), t) for t in OKS_THRESHOLDS}
        results[name] = {
            "gt": len(idx_set),
            "matched": len(oks),
            "mAP": float(np.mean(list(aps.values()))),
            "AP50": aps[0.5],
            "AP75": aps[0.75],
            "per_threshold": {str(t): aps[t] for t in OKS_THRESHOLDS},
        }
    return results


def format_pose_table(results: dict[str, dict[str, Any]]) -> str:
    """分层结果表格（单类 person，无 per-class 表）。"""
    names = [("all", "全部"), ("small", "small(<32²)"), ("dense", "dense(≥8人/图)")]
    lines = [f"{'子集':<16s} {'GT':>6s} {'匹配':>6s} {'mAP':>8s} {'AP50':>8s} {'AP75':>8s}"]
    lines.append("-" * 58)
    for key, label in names:
        r = results[key]
        lines.append(
            f"{label:<16s} {r['gt']:>6d} {r['matched']:>6d} "
            f"{r['mAP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}"
        )
    return "\n".join(lines)


# ============================================================
# 2. 数据加载与推理
# ============================================================

def load_pose_ground_truth(max_images: int = 0) -> dict[int, dict[str, Any]]:
    """加载 person_keypoints_val2017.json。

    Returns:
        {image_id: {"file_name", "persons": [{"bbox": [x1,y1,x2,y2], "area",
                     "keypoints": [(x,y,v)*17], "img_gt_count": int}]}}
    """
    data = json.loads(GT_JSON.read_text())
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # 每图 person 数（dense 分层用）
    per_img: dict[int, int] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue  # crowd 排除（COCO 官方口径）
        per_img[ann["image_id"]] = per_img.get(ann["image_id"], 0) + 1

    gt: dict[int, dict[str, Any]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in gt:
            gt[img_id] = {
                "file_name": id_to_filename.get(img_id, f"{img_id}.jpg"),
                "persons": [],
            }
        x, y, w, h = ann["bbox"]
        flat = ann["keypoints"]  # [x,y,v]*17
        gt[img_id]["persons"].append({
            "bbox": [x, y, x + w, y + h],
            "area": w * h,
            "keypoints": [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, 51, 3)],
            "img_gt_count": per_img[img_id],
        })

    if max_images > 0:
        keys = list(gt.keys())[:max_images]
        gt = {k: gt[k] for k in keys}
    return gt


def run_pose_inference(
    gt: dict[int, dict[str, Any]],
    model_name: str,
    conf: float,
    iou: float,
    batch: int | None = None,
    workers: int | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], float]:
    """对 GT 图像逐块跑姿态推理（批量能力自动 + OOM 降级逐图）。

    Returns: (predictions, elapsed_seconds)；predictions 值为
    [{"bbox": [x1,y1,x2,y2], "conf", "keypoints": [(x,y,v)*17]}]
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.pose import create_pose_model
    from auto2dlabel.tools.device import resolve_batch_params

    model = create_pose_model(model_name, iou_threshold=iou)
    # PoseModel Protocol 只承诺 detect_pose；批量能力经 getattr 探测（同检测工厂模式）
    _pose_batch = getattr(model, "detect_pose_batch", None)
    batch_fn = (
        (lambda ps: _pose_batch(ps, ["person"], conf, num_workers=workers or 0))
        if _pose_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "pose_estimation", batch_fn, sample_image_paths(gt, IMAGE_DIR),
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(gt)
    img_ids = list(gt.keys())
    iterator = tqdm(range(0, total, batch_size), total=(total + batch_size - 1) // batch_size,
                    desc=f"姿态（{model_name} b={batch_size}）", unit="batch") \
        if has_tqdm else range(0, total, batch_size)

    predictions: dict[int, list[dict[str, Any]]] = {}
    _t0 = time.time()
    for start in iterator:
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

        results_per_img = _batch_or_fallback(
            batch_fn,
            lambda p: model.detect_pose(p, ["person"], conf),
            chunk_paths,
        )
        for img_id, results in zip(chunk_ids, results_per_img):
            predictions[img_id] = [
                {
                    "bbox": [r.x, r.y, r.x + r.width, r.y + r.height],
                    "conf": r.confidence,
                    "keypoints": list(r.keypoints),
                }
                for r in results
            ]

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total/elapsed:.1f} img/s")
    return predictions, elapsed


# ============================================================
# 3. 主流程
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="COCO val 2017 姿态基准（OKS 协议）")
    parser.add_argument("--max-images", type=int, default=0, help="最多图像数（0=全部）")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt",
                        help="姿态模型（yolo11n-pose.pt / rtmpose_l 等）")
    parser.add_argument("--iou", type=float, default=0.5, help="bbox 匹配 IoU 阈值")
    parser.add_argument("--batch", type=int, default=None,
                        help="批量推理每批图像数（默认 None=自动实测，1=逐图）")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 子进程数（默认 None=按 CPU 核数自动）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from auto2dlabel.tools.device import print_device
    print_device()
    print()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print("加载 person_keypoints_val2017 Ground Truth...")
    gt = load_pose_ground_truth(max_images=args.max_images)
    print(f"已加载 {len(gt)} 张图像\n")

    predictions, elapsed = run_pose_inference(
        gt, args.model, args.conf, args.iou, batch=args.batch, workers=args.workers,
    )
    print()

    # 逐图匹配 → 全局分层汇总
    print(f"计算指标（OKS 11-point，bbox IoU@{IOU_MATCH_THRESHOLD}）...\n")
    all_gts: list[dict[str, Any]] = []
    all_matches: list[tuple[int, float]] = []
    for img_id, info in gt.items():
        base = len(all_gts)
        all_gts.extend(info["persons"])
        preds = predictions.get(img_id, [])
        for j, oks in match_predictions(preds, info["persons"]):
            all_matches.append((base + j, oks))

    results = summarize_pose(all_gts, all_matches)
    table = format_pose_table(results)
    print(table)
    print()

    result_data = {
        "timestamp": timestamp, "dataset": "COCO val 2017 person keypoints",
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "image_count": len(gt), "gt_count": len(all_gts),
        "elapsed_seconds": round(elapsed, 1),
        "imgs_per_sec": round(len(gt) / elapsed, 2) if elapsed > 0 else 0.0,
        "ok_sigmas": OKS_SIGMAS.tolist(),
        "summary": {k: {
            "gt": v["gt"], "matched": v["matched"],
            "mAP": round(v["mAP"], 4),
            "AP50": round(v["AP50"], 4), "AP75": round(v["AP75"], 4),
        } for k, v in results.items()},
        "per_subset": results,
    }

    json_path, md_path = save_results(result_data, "coco_pose", args.model, timestamp)
    md_path.write_text("\n".join([
        "# COCO val 2017 姿态基准（OKS 协议）",
        f"- **模型**: {args.model} | **conf**: {args.conf} | "
        f"**图像**: {len(gt)} | **速度**: {result_data['imgs_per_sec']} img/s",
        f"```\n{table}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
