"""类别推荐器 — 首轮全类检测，统计后建议标注哪些类别。

v0.2 Phase 2：让 Agent 主动告诉用户"哪些类值得标、哪些不值得"。
"""

from __future__ import annotations

from auto2dlabel.models.model_catalog import COCO_CLASSES
from auto2dlabel.schema.task_plan import DEFAULT_MODEL


def scan_classes(
    image_path: str,
    model=None,
    confidence_threshold: float = 0.1,
) -> dict[str, dict]:
    """对单张图像用 COCO 80 类做全量检测，统计每类表现。

    Args:
        image_path: 图像路径。
        model: 检测模型实例（如为 None 则自动创建 yolo26x）。
        confidence_threshold: 检测置信度阈值。

    Returns:
        {class_name: {"count": int, "avg_conf": float, "min_conf": float, "max_conf": float}}
    """
    if model is None:
        from auto2dlabel.models.detection import create_detection_model
        model = create_detection_model(DEFAULT_MODEL, iou_threshold=0.3)

    results = model.detect(image_path, COCO_CLASSES, confidence_threshold=confidence_threshold)

    stats: dict[str, list[float]] = {}
    for r in results:
        name = r.label
        if name not in stats:
            stats[name] = []
        stats[name].append(r.confidence)

    summary = {}
    for name, confs in stats.items():
        summary[name] = {
            "count": len(confs),
            "avg_conf": round(sum(confs) / len(confs), 3),
            "min_conf": round(min(confs), 3),
            "max_conf": round(max(confs), 3),
        }

    return summary


def recommend_classes(
    stats: dict[str, dict],
    min_count: int = 3,
    min_avg_conf: float = 0.3,
    top_k: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    """根据检测统计，推荐哪些类值得标注。

    Args:
        stats: scan_classes 的返回值。
        min_count: 最少检出数，低于此值不建议。
        min_avg_conf: 最低平均置信度。
        top_k: 最多推荐类别数。

    Returns:
        (recommended, skipped, rejected) — 推荐、跳过、拒绝三类列表。
    """
    # 按 count × avg_conf 排序
    scored = [
        (name, info["count"] * info["avg_conf"], info["count"], info["avg_conf"])
        for name, info in stats.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    recommended: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []

    for name, score, count, avg_conf in scored:
        if count >= min_count and avg_conf >= min_avg_conf and len(recommended) < top_k:
            recommended.append(name)
        elif count > 0:
            skipped.append(name)
        else:
            rejected.append(name)

    return recommended, skipped, rejected


def format_recommendation(
    recommended: list[str],
    skipped: list[str],
    stats: dict[str, dict],
) -> str:
    """将推荐结果格式化为 Agent 可展示的文本。"""
    lines = ["首轮扫描完成。类别推荐：", ""]

    if recommended:
        lines.append("✅ 建议标注（检出多、置信度高）：")
        for name in recommended:
            s = stats[name]
            lines.append(f"   {name}: {s['count']} 框, avg {s['avg_conf']:.0%}")
        lines.append("")

    if skipped:
        lines.append("⚠️ 可跳过（检出少或置信度低）：")
        for name in skipped[:10]:
            s = stats[name]
            lines.append(f"   {name}: {s['count']} 框, avg {s['avg_conf']:.0%}")
        lines.append("")

    lines.append("回复 'ok' 按推荐类别标注，或输入你想标注的类别名。")
    lines.append("30 秒后自动按推荐类别执行。")

    return "\n".join(lines)
