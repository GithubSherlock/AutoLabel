#!/usr/bin/env python3
"""DOTA OBB (Oriented Bounding Box) Benchmark — 航拍旋转框检测。

DOTA v1.0 Task1 旋转框标注（15 类全评），模型 yolo11n-obb.pt（DOTAv1
训练权重，类名与 GT 原生对齐，无 COCO 映射损耗）。IoU 采用 rotate_iou
（shapely 四边形求交），复用 evaluate_per_class 的贪心匹配 + 11 点插值
AP 骨架（iou_fn 注入）。

GT: labels_obb/*.txt（前 2 行 imagesource:/gsd: 头，数据行 10 列）→
    {name: 连字符类名, quad: [x1..y4] 8 值, bbox: min/max 派生}
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
    build_parser,
    evaluate_per_class,
    format_result_table,
    normalize_dota_class,
    obb_batch_or_fallback,
    rotate_iou,
    sample_image_paths,
    save_results,
)
from auto2dlabel.benchmarks.datasets import ensure_dota_val  # noqa: E402
from auto2dlabel.export.dota import rotated_corners  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────
OBB_MODEL = "yolo11n-obb.pt"


# ================================================================
# 1. GT 加载
# ================================================================

def load_dota_obb_ground_truth(
    image_dir: Path,
    label_dir: Path,
    max_images: int = 0,
    include_difficult: bool = False,
) -> dict[int, dict[str, Any]]:
    """加载 DOTA Task1 OBB 旋转框标注。

    Task1 格式（前 2 行为 imagesource:/gsd: 头，数据行 10 列）:
      x1 y1 x2 y2 x3 y3 x4 y4 classname difficulty
    类名为连字符命名（small-vehicle）；difficulty ∈ {0, 1}。

    Args:
        image_dir: 图像目录（labels_obb 同名 PNG 校验存在性）。
        label_dir: labels_obb/ 标注目录。
        max_images: 最多图像数（0=全部）。
        include_difficult: 是否纳入 difficulty=1 难例（默认剔除，官方惯例）。

    Returns:
        {idx: {"file_name": str, "objects": [{"name": "small-vehicle",
               "quad": [8 值], "bbox": [x1,y1,x2,y2]}]}}
    """
    label_files = sorted(label_dir.glob("*.txt"))
    if max_images > 0:
        label_files = label_files[:max_images]

    gt: dict[int, dict[str, Any]] = {}
    skipped_img = 0
    skipped_diff = 0

    for i, lf in enumerate(label_files):
        img_name = f"{lf.stem}.png"
        img_path = image_dir / img_name
        if not img_path.exists():
            skipped_img += 1
            continue

        objects = []
        for line in lf.read_text().strip().splitlines():
            line = line.strip()
            parts = line.split()
            # 头行（imagesource:/gsd:）不足 10 列，防御跳过
            if len(parts) < 10:
                continue
            if not include_difficult and parts[9] != "0":
                skipped_diff += 1
                continue
            quad = [float(v) for v in parts[:8]]
            xs = quad[0::2]
            ys = quad[1::2]
            objects.append({
                "name": parts[8],
                "quad": quad,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })

        gt[i] = {"file_name": img_name, "objects": objects}

    if skipped_img:
        print(f"  跳过 {skipped_img} 张缺少图片的标注")
    if skipped_diff:
        print(f"  剔除 {skipped_diff} 个 difficulty=1 难例标注")
    return gt


# ================================================================
# 2. 检测
# ================================================================

def run_obb_detection(
    gt: dict[int, dict[str, Any]],
    image_dir: Path,
    model_name: str,
    conf: float,
    iou: float,
    batch: int | None = None,
    workers: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """对 GT 中所有图像运行 OBB 旋转框检测（批量推理 + OOM 降级逐图）。"""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from auto2dlabel.models.obb import create_obb_model
    from auto2dlabel.tools.device import resolve_batch_params

    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    print(f"检测类别 ({len(all_cats)}): {', '.join(all_cats)}")

    # 提示直接传连字符 GT 名（_match_obb_prompt 内部空格↔连字符归一 + 别名兼容）
    model = create_obb_model(model_name, iou_threshold=iou)
    predictions: dict[int, list[dict[str, Any]]] = {}

    # 批量推理超参：CLI 显式 > 动态实测（模型加载后测单图显存）> 静态表
    _obb_batch = getattr(model, "detect_obb_batch", None)
    infer_fn = (
        (lambda ps: _obb_batch(ps, all_cats, conf, 0))
        if _obb_batch is not None else None
    )
    batch_size, num_workers = resolve_batch_params(
        "obb_detection", infer_fn, sample_image_paths(gt, image_dir),
        explicit_batch=batch, explicit_workers=workers,
    )
    print(f"批量推理: batch_size={batch_size}  num_workers={num_workers}")

    total = len(gt)
    img_ids = list(gt.keys())
    n_batches = (total + batch_size - 1) // batch_size
    iterator = (
        tqdm(range(0, total, batch_size), total=n_batches,
             desc=f"OBB 检测（{model_name} b={batch_size}）", unit="batch")
        if has_tqdm else range(0, total, batch_size)
    )

    _t0 = time.time()
    for start in iterator:
        # 分块（缺失文件过滤，chunk_ids 与 chunk_paths 保序对齐）
        chunk_ids: list[int] = []
        chunk_paths: list[str] = []
        for img_id in img_ids[start:start + batch_size]:
            img_path = image_dir / gt[img_id]["file_name"]
            if not img_path.exists():
                continue
            chunk_ids.append(img_id)
            chunk_paths.append(str(img_path))
        if not chunk_paths:
            continue

        results_per_img = obb_batch_or_fallback(
            model, chunk_paths, all_cats, conf, num_workers,
        )
        for img_id, results in zip(chunk_ids, results_per_img):
            predictions[img_id] = []
            for r in results:
                corners = rotated_corners(r.cx, r.cy, r.width, r.height, r.angle)
                predictions[img_id].append({
                    "name": normalize_dota_class(r.label),
                    "quad": [c for pt in corners for c in pt],
                    "conf": r.confidence,
                })

    elapsed = time.time() - _t0
    print(f"完成！{elapsed:.1f}s, {total / elapsed:.1f} img/s")
    return predictions


# ================================================================
# 3. 主流程
# ================================================================

def main() -> None:
    parser = build_parser("DOTA OBB (Oriented Bounding Box) Benchmark")
    parser.set_defaults(model=OBB_MODEL)
    parser.add_argument("--include-difficult", action="store_true",
                        help="纳入 difficulty=1 难例（默认剔除，DOTA 官方惯例）")
    args = parser.parse_args()

    if args.sahi:
        parser.error("--sahi 不支持 OBB 评测（SAHI 切片仅支持水平框）")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from auto2dlabel.tools.device import print_device
    print_device()
    print()

    # 数据集
    dota_root = ensure_dota_val()
    image_dir = dota_root / "images"
    label_dir = dota_root / "labels_obb"

    # 加载 GT
    print("加载 DOTA Ground Truth (Task1 OBB, 15 类) ...")
    gt = load_dota_obb_ground_truth(
        image_dir, label_dir,
        max_images=args.max_images or 0,
        include_difficult=args.include_difficult,
    )
    print(f"已加载 {len(gt)} 张图像\n")

    # 检测
    predictions = run_obb_detection(gt, image_dir, args.model, args.conf, args.iou,
                                    batch=args.batch, workers=args.workers)
    print()

    # 可视化（--viz：镜像相对路径渲染旋转框 quad）
    if args.viz:
        from auto2dlabel.benchmarks.viz import visualize_dataset
        visualize_dataset("dota_obb", gt, predictions,
                          lambda img_id, info: image_dir / info["file_name"])
        print()

    # 评估（旋转 IoU）
    print(f"计算指标（rotate IoU@{IOU_MATCH_THRESHOLD}）...\n")
    all_cats = sorted(set(o["name"] for g in gt.values() for o in g["objects"]))
    results = {
        cls: evaluate_per_class(gt, predictions, cls, IOU_MATCH_THRESHOLD, iou_fn=rotate_iou)
        for cls in all_cats
    }

    summary = format_result_table(results, all_cats, top_n=args.top_classes)
    print(summary)
    print()

    # 保存
    mAP = float(np.mean([results[cls]["ap"] for cls in all_cats if cls in results]))
    result_data = {
        "timestamp": ts, "dataset": "DOTA val (aerial, Task1 OBB, 15 classes)",
        "model": args.model, "confidence_threshold": args.conf,
        "iou_match_threshold": IOU_MATCH_THRESHOLD, "iou_fn": "rotate_iou",
        "include_difficult": args.include_difficult, "image_count": len(gt),
        "summary": {"mAP@0.5": round(mAP, 4)},
        "per_class": {cls: results[cls] for cls in all_cats},
    }

    json_path, md_path = save_results(result_data, "dota_obb", args.model, ts)
    md_path.write_text("\n".join([
        "# DOTA OBB (Oriented Bounding Box) Benchmark",
        f"- **模型**: {args.model} | **conf**: {args.conf} | **mAP@0.5**: {mAP:.4f}",
        f"- **IoU**: rotate_iou（四边形求交）@{IOU_MATCH_THRESHOLD} | 15 类全评",
        f"- **难例**: {'纳入' if args.include_difficult else '剔除（默认）'}",
        f"```\n{summary}\n```",
    ]))
    print(f"结果: {json_path}")
    print(f"报告: {md_path}")


if __name__ == "__main__":
    main()
