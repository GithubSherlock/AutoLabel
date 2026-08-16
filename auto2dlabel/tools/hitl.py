"""HITL（人机协同）分流逻辑。

按置信度三档分流标注结果：
  - conf >= τ_high → 直接采纳，进正式导出
  - τ_low <= conf < τ_high → 写入 review_queue.json（人工复核）
  - conf < τ_low → 写入 hard_samples.json（困难样本）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto2dlabel.schema.annotation import Annotation, Bbox

# 默认阈值（可在 CLI 中覆盖）
DEFAULT_TAU_HIGH = 0.7
DEFAULT_TAU_LOW = 0.3


class TriageResult:
    """三档分流结果。"""

    def __init__(self) -> None:
        self.accepted: list[Bbox] = []       # 直接采纳
        self.review: list[Bbox] = []          # 人工复核
        self.hard: list[Bbox] = []            # 困难样本

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.review) + len(self.hard)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "accepted": len(self.accepted),
            "review": len(self.review),
            "hard": len(self.hard),
            "total": self.total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [b.to_dict() for b in self.accepted],
            "review": [b.to_dict() for b in self.review],
            "hard_samples": [b.to_dict() for b in self.hard],
            "summary": self.summary,
        }


def triage_annotations(
    annotations: list[Annotation],
    tau_high: float = DEFAULT_TAU_HIGH,
    tau_low: float = DEFAULT_TAU_LOW,
) -> TriageResult:
    """对所有标注按置信度三档分流。

    Args:
        annotations: Annotation 列表。
        tau_high: 直接采纳阈值（>= 此值直接采纳）。
        tau_low: 人工复核下限（< 此值为困难样本）。

    Returns:
        TriageResult。
    """
    result = TriageResult()
    for ann in annotations:
        for bbox in ann.bboxes:
            if bbox.confidence >= tau_high:
                result.accepted.append(bbox)
            elif bbox.confidence >= tau_low:
                result.review.append(bbox)
            else:
                result.hard.append(bbox)
    return result


def export_triage(
    triage: TriageResult,
    output_dir: str | Path,
    image_stem: str = "output",
    timestamp: str = "",
    image_path: str = "",
    image_size: tuple[int, int] = (0, 0),
) -> dict[str, Path]:
    """将分流结果导出到文件。

    Args:
        triage: 分流结果。
        output_dir: 输出目录。
        image_stem: 图像名（不含扩展名）。
        timestamp: 统一时间戳。
        image_path: 原图绝对路径（供 Web 复核队列消费方显示原图；旧调用可缺省）。
        image_size: 原图尺寸 (width, height)。

    Returns:
        {"review": Path, "hard": Path}。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_suffix = f"_{timestamp}" if timestamp else ""

    review_path = out_dir / f"{image_stem}{ts_suffix}_review.json"
    hard_path = out_dir / f"{image_stem}{ts_suffix}_hard.json"

    # review queue: 包含 review 和 hard 两档（都需人工处理）
    review_data = {
        "image": image_stem,
        "image_path": image_path,
        "image_size": list(image_size),
        "description": "人工复核队列 — 中置信度样本需确认，低置信度样本需重标",
        "annotations": triage.to_dict()["review"] + triage.to_dict()["hard_samples"],
        "summary": {
            "review_count": len(triage.review),
            "hard_count": len(triage.hard),
            "total_need_review": len(triage.review) + len(triage.hard),
        },
    }
    review_path.write_text(json.dumps(review_data, indent=2, ensure_ascii=False))

    # hard samples: 仅困难样本
    hard_data = {
        "image": image_stem,
        "image_path": image_path,
        "image_size": list(image_size),
        "description": "困难样本 — 置信度过低，建议换模型或人工重标",
        "annotations": triage.to_dict()["hard_samples"],
        "summary": {"hard_count": len(triage.hard)},
    }
    if triage.hard:
        hard_path.write_text(json.dumps(hard_data, indent=2, ensure_ascii=False))

    return {"review": review_path, "hard": hard_path}
