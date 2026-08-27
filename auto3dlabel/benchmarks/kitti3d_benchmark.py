"""KITTI 3D 评测（双层：3D IoU + BEV IoU，难度分层 × 3 类）。

复用 auto2dlabel evaluate_per_class（贪心 conf 降序 + 11-point AP，iou_fn 注入）：
- 3D 层：iou_fn=iou3d_list，键 "bbox" 7 值 [h,w,l,x,y,z,ry]
- BEV 层：iou_fn=bev_iou_quad，键 "quad" 8 值 BEV 四边形
难度分层复用 kitti_difficulty/filter_gt_by_difficulty（KITTI 官方口径）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.table import Table

from auto2dlabel.benchmarks.common import evaluate_per_class
from auto3dlabel.benchmarks.load_gt3d import gt_frame, pred_frame
from auto3dlabel.configs.kitti import KITTI_EVAL_CLASSES
from auto3dlabel.schema.box3d import KittiFrame
from auto3dlabel.tools.geometry import bev_iou_quad, iou3d_list

IOU_3D_THRESHOLD = 0.5
IOU_BEV_THRESHOLD = 0.5


def _evaluate_layer(
    gt: dict[str, dict[str, Any]],
    pred: dict[str, list[dict[str, Any]]],
    difficulty: str,
    iou_fn: Callable[[list[float], list[float]], float],
    iou_threshold: float,
    classes: list[str],
) -> dict[str, dict[str, Any]]:
    """单层（3D 或 BEV）× 单难度 × 3 类 → {类: evaluate_per_class 返回值}。"""
    # 难度过滤同 filter_gt_by_difficulty 口径（KITTI 官方：分层 GT × 全量预测）；
    # 自写避免其 dict[int, ...] 键注解与 str frame_id 冲突
    gt_f: dict[str, dict[str, Any]] = {
        k: {"objects": [o for o in v["objects"] if o.get("difficulty") == difficulty]}
        for k, v in gt.items()
    }
    out: dict[str, dict[str, Any]] = {}
    for cls in classes:
        out[cls] = evaluate_per_class(
            gt_f, pred, cls, iou_threshold=iou_threshold, iou_fn=iou_fn
        )
    return out


def _without_quad_gt(gt: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """GT 对象去 quad 键 → evaluate_per_class 回退取 "bbox"（7 值 3D 签名）。"""
    return {
        k: {
            "objects": [
                {key: val for key, val in o.items() if key != "quad"} for o in v["objects"]
            ]
        }
        for k, v in gt.items()
    }


def _without_quad_pred(pred: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """预测去 quad 键（同上；pred_frame 产物含 quad 供 BEV 层使用）。"""
    return {
        k: [{key: val for key, val in p.items() if key != "quad"} for p in ps]
        for k, ps in pred.items()
    }


def run_kitti3d_benchmark(
    frames: list[KittiFrame],
    predictions: dict[str, list[Any]],
    difficulties: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """帧列表 + {frame_id: [Box3D]} → 双层 × 难度 × 类 AP 表。

    Returns: {"3d"|"bev": {difficulty: {class: {ap, gt_count, pred_count, ...}}}}
    """
    difficulties = difficulties or ["easy", "moderate", "hard"]
    gt: dict[str, dict[str, Any]] = {f.frame_id: gt_frame(f) for f in frames}
    pred: dict[str, list[dict[str, Any]]] = {
        fid: pred_frame(boxes) for fid, boxes in predictions.items()
    }
    layers = {
        "3d": (iou3d_list, IOU_3D_THRESHOLD),
        "bev": (bev_iou_quad, IOU_BEV_THRESHOLD),
    }
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for layer, (iou_fn, thr) in layers.items():
        # 3D 层必须喂 "bbox" 7 值：evaluate_per_class 有 quad 优先取 quad，
        # 否则 iou3d_list 拿 8 值 quad 的前 7 个值当 [h,w,l,x,y,z,ry] 解包（错误 IoU）
        if layer == "3d":
            layer_gt, layer_pred = _without_quad_gt(gt), _without_quad_pred(pred)
        else:
            layer_gt, layer_pred = gt, pred
        result[layer] = {}
        for diff in difficulties:
            result[layer][diff] = _evaluate_layer(
                layer_gt, layer_pred, diff, iou_fn, thr, KITTI_EVAL_CLASSES
            )
    return result


def format_benchmark_table(
    result: dict[str, dict[str, dict[str, dict[str, Any]]]],
    title: str = "KITTI 3D 评测（AP %）",
) -> str:
    """rich 表格渲染文本（layer × difficulty × class → AP/GT 数），由调用方输出。"""
    import io

    buf = io.StringIO()
    console = Console(file=buf, record=True, width=110, force_terminal=False)
    table = Table(title=title, show_lines=True)
    table.add_column("层次", style="bold")
    table.add_column("难度")
    for cls in KITTI_EVAL_CLASSES:
        table.add_column(cls, justify="right")
    for layer, diffs in result.items():
        for diff, classes in diffs.items():
            row = [layer.upper(), diff]
            for cls in KITTI_EVAL_CLASSES:
                stats = classes[cls]
                ap = stats.get("ap", 0.0) * 100
                row.append(f"{ap:5.1f} ({stats.get('gt_count', 0)})")
            table.add_row(*row)
    console.print(table)
    return buf.getvalue()
