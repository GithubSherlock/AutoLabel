"""HITL 三档分流统计纯函数（v1.0 P5）。

扫描 2D/3D 复核队列产物做计数汇总（无缓存——每次 /review 重扫，
回流后刷新天然成立）。口径与两侧 web server 队列扫描一致
（`*_review.json` 未复核队列、`.reviewed` 侧车排除、`*_reviewed.json`
已复核；损坏 JSON 跳过——同 collect_review_pool 容错模式，不复用其
代码：彼处语义是评分排序，此处是计数汇总）。

2D 档位（outputs/*_review.json summary 无 accepted 字段，如实两档）：
  待复核 = Σ review_count；困难 = Σ hard_count；已复核 = *_reviewed.json 数
3D 档位（outputs/kitti3d/reviews/*_review.json summary 含 accepted）：
  待复核 = Σ review_count；困难 = Σ hard_count；已复核 = *_reviewed.json 数
  （与 2D 同口径）；accepted（直接采纳）单列、不混入已复核（#16）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewStats:
    """单个域的复核统计快照（/review 展示与断言用）。"""

    domain: str  # "2d" | "3d"
    dir: str  # 扫描目录（展示可溯源）
    pending_review: int
    hard: int
    accepted: int  # 2D 无此档恒 0（summary 无 accepted 字段）
    reviewed_files: int
    total_files: int  # 未复核队列文件数


def _safe_count(value: Any) -> int:
    """计数容错：summary 字段可能缺失/None/非数字（手改或旧版本产物），
    /review 展示绝不因此崩（#14）。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _scan_one(domain: str, d: Path) -> ReviewStats:
    """扫一个域：未复核队列计数 + 已复核文件数；目录缺失/损坏 JSON 容错。"""
    pending = hard = accepted = total_files = 0
    if d.is_dir():
        for f in sorted(d.glob("*_review.json")):
            if ".reviewed" in f.name:
                continue  # 已复核侧车（2D 保存时 rename 标记）
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # 损坏文件跳过（collect_review_pool 同款容错）
            summary = data.get("summary")
            if not isinstance(summary, dict):
                continue
            pending += _safe_count(summary.get("review_count"))
            hard += _safe_count(summary.get("hard_count"))
            accepted += _safe_count(summary.get("accepted"))
            total_files += 1
    reviewed = len(list(d.glob("*_reviewed.json"))) if d.is_dir() else 0
    return ReviewStats(
        domain=domain,
        dir=str(d),
        pending_review=pending,
        hard=hard,
        accepted=accepted,
        reviewed_files=reviewed,
        total_files=total_files,
    )


def scan_review_stats(
    review_dir_2d: Path = Path("outputs"),
    review_dir_3d: Path = Path("outputs/kitti3d/reviews"),
) -> list[ReviewStats]:
    """2D + 3D 统计（顺序恒定 [2d, 3d]，/review 逐行渲染）。"""
    return [
        _scan_one("2d", review_dir_2d),
        _scan_one("3d", review_dir_3d),
    ]
