"""KITTI difficulty 分层测试（kitti_difficulty 判据 / GT 标签与过滤 / 分层指标合成）。

零真实权重：全部纯函数 + 合成 GT/预测，不加载任何模型。
"""

from __future__ import annotations

import pytest
from PIL import Image

from auto2dlabel.benchmarks.common import evaluate_per_class
from auto2dlabel.benchmarks.kitti_benchmark import (
    filter_gt_by_difficulty,
    kitti_difficulty,
    load_kitti_ground_truth,
)
from auto2dlabel.tests import Path

# ---------- kitti_difficulty 官方判据 ----------

@pytest.mark.parametrize(
    ("truncated", "occluded", "height", "expected"),
    [
        (0.0, 0, 60, "easy"),
        (0.15, 0, 60, "easy"),      # trunc 边界（含）
        (0.0, 0, 40, "easy"),       # h 边界（含）
        (0.16, 0, 60, "moderate"),  # trunc 超 easy → 落 moderate
        (0.0, 1, 60, "moderate"),   # occ=1 非 easy
        (0.3, 1, 25, "moderate"),   # moderate 双边界（含）
        (0.5, 2, 25, "hard"),       # hard 双边界（含）
        (0.51, 0, 60, None),        # trunc 超 hard → ignore
        (0.0, 3, 60, None),         # occ 超 hard → ignore
        (0.0, 0, 24.9, None),       # h 不足 → ignore
    ],
)
def test_kitti_difficulty_boundaries(
    truncated: float, occluded: int, height: float, expected: str | None,
) -> None:
    assert kitti_difficulty(truncated, occluded, height) == expected


# ---------- GT 标签与过滤 ----------

def _make_kitti_scene(tmp_path: Path) -> tuple[Path, Path]:
    """1 张 8x8 图 + 5 行标注：easy/moderate/hard 各一、过小 ignore、DontCare。"""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "000001.png")
    (label_dir / "000001.txt").write_text("\n".join([
        "Car 0.00 0 0.0 10.0 0.0 100.0 40.0 1.5 1.5 4.0 0 0 20 0",         # easy (h=40)
        "Car 0.10 1 0.0 10.0 50.0 100.0 75.0 1.5 1.5 4.0 0 0 20 0",        # moderate (h=25)
        "Pedestrian 0.20 2 0.0 10.0 100.0 60.0 125.0 0.5 0.5 1.0 0 0 20 0",  # hard (h=25)
        "Car 0.60 3 0.0 0.0 0.0 10.0 20.0 1.5 1.5 4.0 0 0 20 0",          # 过小+遮挡过重 → ignore
        "DontCare 0.00 0 0.0 0.0 0.0 10.0 10.0 0.0 0.0 0.0 0 0 0 0",      # DontCare → 忽略
    ]))
    return image_dir, label_dir


def test_load_tags_difficulty(tmp_path: Path) -> None:
    """GT 对象带 difficulty 标签；ignore（过小/遮挡过重/DontCare）剔除。"""
    image_dir, label_dir = _make_kitti_scene(tmp_path)
    gt = load_kitti_ground_truth(image_dir, label_dir)
    assert len(gt) == 1
    objs = gt[0]["objects"]
    assert [o["difficulty"] for o in objs] == ["easy", "moderate", "hard"]
    assert [o["name"] for o in objs] == ["car", "car", "person"]


def test_load_with_difficulty_filter(tmp_path: Path) -> None:
    """difficulty="easy" → 只保留 easy 档对象。"""
    image_dir, label_dir = _make_kitti_scene(tmp_path)
    gt = load_kitti_ground_truth(image_dir, label_dir, difficulty="easy")
    objs = gt[0]["objects"]
    assert len(objs) == 1
    assert objs[0]["difficulty"] == "easy"


def test_filter_gt_by_difficulty() -> None:
    """分层过滤保留文件结构（无该档对象的图留空列表），原 GT 不变。"""
    gt = {
        0: {"file_name": "a.png", "objects": [
            {"name": "car", "bbox": [0, 0, 10, 10], "difficulty": "easy"},
            {"name": "car", "bbox": [1, 1, 11, 11], "difficulty": "hard"},
        ]},
        1: {"file_name": "b.png", "objects": [
            {"name": "person", "bbox": [2, 2, 12, 12], "difficulty": "moderate"},
        ]},
    }
    easy = filter_gt_by_difficulty(gt, "easy")
    assert [o["difficulty"] for o in easy[0]["objects"]] == ["easy"]
    assert easy[1]["objects"] == []
    assert len(gt[0]["objects"]) == 2  # 原 GT 不变


# ---------- 分层指标合成 ----------

def test_per_tier_metrics_composition() -> None:
    """分层 = 分层 GT × 全量预测：easy 框满分命中 → AP 1.0；hard 无命中 → 0.0；
    overall 含两者（2 GT，1 TP 无 FP：recall=0.5 处 prec=1.0 → 11-point AP = 6/11）。"""
    gt = {
        0: {"file_name": "a.png", "objects": [
            {"name": "car", "bbox": [0, 0, 10, 40], "difficulty": "easy"},
            {"name": "car", "bbox": [20, 20, 30, 50], "difficulty": "hard"},
        ]},
    }
    predictions = {0: [{"name": "car", "bbox": [0, 0, 10, 40], "conf": 0.9}]}

    easy_res = evaluate_per_class(
        filter_gt_by_difficulty(gt, "easy"), predictions, "car", 0.5,
    )
    hard_res = evaluate_per_class(
        filter_gt_by_difficulty(gt, "hard"), predictions, "car", 0.5,
    )
    overall_res = evaluate_per_class(gt, predictions, "car", 0.5)

    assert easy_res["ap"] == 1.0
    assert easy_res["gt_count"] == 1
    assert hard_res["ap"] == 0.0
    assert hard_res["gt_count"] == 1
    assert overall_res["ap"] == pytest.approx(6 / 11, abs=1e-3)  # 2 GT、1 TP 无 FP，11-point 插值
