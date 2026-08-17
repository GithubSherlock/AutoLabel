"""Benchmark 可视化 —— 预测结果渲染为图像，保存到项目同级 Visualization/。

复用 tools/visualize.py 的渲染原语（颜色 / bbox / mask）；分类任务画文本条
（GT + top-K）。仅额外写可视化产物，不改评测协议。输出路径镜像原数据集
相对路径：`Visualization/<数据集名>/<原相对路径>`。已存在的产物跳过（断点续跑）。

预测 dict 分派规则（按 benchmark 脚本现有结构）：
- {"quad": [x1..y4 8 点], "name", "conf"}        → OBB 旋转框折线
- {"mask": (H,W) bool, "name", "conf"}           → mask 轮廓叠加（含 bbox 时同框）
- {"polygon": [[x,y,...],...], "bbox", "name"}   → polygon 直接填充（内存友好，免解码）
- {"bbox": [x1,y1,x2,y2] 或 Bbox, "name", "conf"} → 水平框
- {"label", "score"} + gt_label_fn               → 分类文本条（GT + top-K）
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from auto2dlabel.benchmarks.common import mask_to_bbox
from auto2dlabel.schema.annotation import Bbox, Mask
from auto2dlabel.tools.visualize import _get_color, draw_bboxes, draw_masks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIS_ROOT = PROJECT_ROOT.parent / "Visualization"


def _mask_polygons(mask_bool: np.ndarray[Any, Any]) -> list[list[float]]:
    """bool mask → 外轮廓多边形列表（tools.draw_masks 的 segmentation 格式）。"""
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c.astype(float).flatten().tolist() for c in contours if len(c) >= 3]


def _draw_quads(
    image: np.ndarray[Any, Any],
    quads: list[tuple[str, list[float], float]],
) -> np.ndarray[Any, Any]:
    """旋转框：quad 8 点闭合折线 + 标签（颜色与 bbox 渲染一致）。"""
    img = image.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    for name, quad, conf in quads:
        color = _get_color(name)
        pts = np.array(quad, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick)
        text = f"{name} {conf:.0%}"
        x1, y1 = int(pts[0][0][0]), int(pts[0][0][1])
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return img


def _draw_classification_strip(
    image: np.ndarray[Any, Any],
    gt_label: str | None,
    preds: list[dict[str, Any]],
) -> np.ndarray[Any, Any]:
    """分类文本条：顶部半透明黑带，GT 行（白）+ top-5 预测行（类别色）。"""
    rows: list[tuple[str, tuple[int, int, int]]] = []
    if gt_label:
        rows.append((f"GT: {gt_label}", (255, 255, 255)))
    for i, p in enumerate(preds[:5], start=1):
        rows.append((f"{i}. {p['label']} {p['score']:.2%}", _get_color(str(p["label"]))))
    if not rows:
        return image

    img = image.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    line_h = 30
    band_h = line_h * len(rows) + 12
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], band_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for i, (line, color) in enumerate(rows):
        cv2.putText(img, line, (8, 24 + i * line_h), font, scale, color, thick, cv2.LINE_AA)
    return img


def visualize_predictions(
    image_path: Path,
    preds: list[dict[str, Any]],
    out_path: Path,
    gt_label: str | None = None,
) -> None:
    """单图渲染：quad → 旋转框、mask → mask 叠加、bbox → 水平框；gt_label 非空 → 分类文本条。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if gt_label is not None:
        image = _draw_classification_strip(image, gt_label, preds)
    else:
        quads: list[tuple[str, list[float], float]] = []
        masks: list[Mask] = []
        bboxes: list[Bbox] = []
        for p in preds:
            if "quad" in p:
                quads.append((str(p["name"]), [float(c) for c in p["quad"]],
                              float(p.get("conf", 1.0))))
            elif "mask" in p:
                mask_bool = p["mask"]
                x1, y1, x2, y2 = mask_to_bbox(mask_bool)
                masks.append(Mask(
                    bbox=Bbox(x=float(x1), y=float(y1), width=float(x2 - x1),
                              height=float(y2 - y1), label=str(p["name"]),
                              confidence=float(p.get("conf", 1.0))),
                    segmentation=_mask_polygons(mask_bool),
                ))
            elif "polygon" in p:
                # polygon 压缩存储（如 d2sa box-prompted SAM2 输出）：免解码直接填充
                b = p.get("bbox")
                if isinstance(b, Bbox):
                    bb = b
                elif isinstance(b, (list, tuple)):
                    x, y, w, h = (float(c) for c in b)
                    bb = Bbox(x=x, y=y, width=w, height=h, label=str(p["name"]),
                              confidence=float(p.get("conf", 1.0)))
                else:
                    continue  # 无 bbox 的 polygon 无法定位（跳过）
                masks.append(Mask(bbox=bb, segmentation=p["polygon"]))
            elif "bbox" in p:
                b = p["bbox"]
                if isinstance(b, Bbox):
                    bboxes.append(b)
                else:
                    x1, y1, x2, y2 = (float(c) for c in b)
                    bboxes.append(Bbox(
                        x=x1, y=y1, width=x2 - x1, height=y2 - y1,
                        label=str(p["name"]), confidence=float(p.get("conf", 1.0)),
                    ))
        if masks:
            image = draw_masks(image, masks)
        if bboxes:
            image = draw_bboxes(image, bboxes)
        if quads:
            image = _draw_quads(image, quads)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def visualize_dataset(
    dataset_name: str,
    gt: dict[Any, dict[str, Any]],
    predictions: dict[Any, list[dict[str, Any]]],
    image_path_fn: Callable[[Any, dict[str, Any]], Path],
    rel_name_fn: Callable[[Any, dict[str, Any]], str] | None = None,
    gt_label_fn: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[int, int]:
    """渲染整个数据集 → Visualization/<dataset_name>/（镜像原相对路径）。

    Args:
        dataset_name: 输出文件夹名（与数据集同名）。
        gt: {img_id: info}。
        predictions: {img_id: [pred dict]}（结构与 visualize_predictions 分派规则一致）。
        image_path_fn(img_id, info) → 原图路径。
        rel_name_fn(img_id, info) → 输出相对名（默认 info["file_name"]）。
        gt_label_fn(info) → 分类 GT 标签（传入即画分类文本条）。

    Returns:
        (渲染数, 跳过数)。已存在产物或原图缺失时跳过（断点续跑）。
    """
    out_root = VIS_ROOT / dataset_name
    rendered = skipped = 0
    for img_id, info in gt.items():
        rel = (rel_name_fn(img_id, info) if rel_name_fn
               else str(info.get("file_name", f"{img_id}.jpg")))
        out_path = out_root / rel
        if out_path.exists():
            skipped += 1
            continue
        img_path = image_path_fn(img_id, info)
        if not img_path.exists():
            skipped += 1
            continue
        gt_label = gt_label_fn(info) if gt_label_fn else None
        visualize_predictions(img_path, predictions.get(img_id, []), out_path, gt_label=gt_label)
        rendered += 1
    print(f"可视化: {rendered} 张已渲染, {skipped} 张跳过 → {out_root}")
    return rendered, skipped
