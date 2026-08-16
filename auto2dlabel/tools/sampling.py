"""主动学习采样 —— 从 HITL 复核队列挑选最有信息量的样本。

只聚合 *_review.json（复核队列已含 review + hard 两档，无需单独读 hard 文件）。
单图得分 = 平均不确定性 + 类别多样性：

    score = mean(1 - 2*|conf - 0.5|) + 0.1 * 类别数

置信度越接近 0.5 越不确定（1 - 2|conf-0.5| 在 0~1 之间），类别越多覆盖越广。
排序确定性（score 降序，同分按文件名升序），便于复现。不做训练闭环（只出清单）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReviewItem:
    """队列文件中一张图的聚合条目。"""

    file: str = ""              # 队列文件名
    image_path: str = ""        # 原图路径
    mean_uncertainty: float = 0.0
    class_count: int = 0
    bbox_count: int = 0
    score: float = 0.0


def score_image(confs: list[float], class_count: int) -> float:
    """单图得分（纯函数）。空 confs 返回 0。"""
    if not confs:
        return 0.0
    mean_unc = sum(1 - 2 * abs(c - 0.5) for c in confs) / len(confs)
    return mean_unc + 0.1 * class_count


def collect_review_pool(review_dir: str | Path) -> list[ReviewItem]:
    """扫描 *_review.json（跳过 .reviewed），损坏 JSON / 空 annotations 容错。"""
    d = Path(review_dir)
    if not d.is_dir():
        return []
    items: list[ReviewItem] = []
    for f in sorted(d.glob("*_review.json")):
        if ".reviewed" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # 损坏文件跳过
        anns = [a for a in data.get("annotations", []) if isinstance(a, dict)]
        confs = [float(a.get("confidence", 0.5)) for a in anns]
        if not confs:
            continue  # 空队列文件不计入
        labels = {str(a.get("label", "")) for a in anns}
        item = ReviewItem(
            file=f.name,
            image_path=str(data.get("image_path", "")),
            mean_uncertainty=sum(1 - 2 * abs(c - 0.5) for c in confs) / len(confs),
            class_count=len(labels),
            bbox_count=len(confs),
        )
        item.score = score_image(confs, len(labels))
        items.append(item)
    return items


def sample_priority(pool: list[ReviewItem], top_k: int) -> list[ReviewItem]:
    """确定性 top-K：score 降序，同分按文件名升序。"""
    ordered = sorted(pool, key=lambda x: (-x.score, x.file))
    return ordered[:top_k]


def write_priority_manifest(items: list[ReviewItem], output_path: str | Path) -> Path:
    """写采样清单 JSON（按排名）。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "description": "主动学习采样清单（按信息量降序）",
        "count": len(items),
        "items": [
            {
                "rank": i + 1,
                "file": it.file,
                "image_path": it.image_path,
                "score": round(it.score, 4),
                "mean_uncertainty": round(it.mean_uncertainty, 4),
                "class_count": it.class_count,
                "bbox_count": it.bbox_count,
            }
            for i, it in enumerate(items)
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
