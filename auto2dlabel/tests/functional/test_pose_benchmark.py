"""姿态基准协议纯函数测试（v0.6 Phase 3b，零真实权重）。

覆盖 OKS 评测协议：compute_oks（完美/偏移/可见点过滤/零面积/缺位补齐）、
match_predictions（conf 降序贪心 + IoU 门 + 每 GT 一次）、ap_11point
（全 TP / 无匹配 / 插值 / 空 GT）、summarize_pose（small/dense 分层边界）、
load_pose_ground_truth（crowd 排除 / bbox xyxy 换算 / keypoints 三元组）。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import auto2dlabel.benchmarks.pose_benchmark as pb
from auto2dlabel.tests import Path, json, np


def _kpts(n: int = 17, x0: float = 0.0, y0: float = 0.0) -> list[tuple[float, float, float]]:
    return [(x0 + i, y0 + i, 1.0) for i in range(n)]


def _gt(bbox: tuple[float, float, float, float], area: float | None = None,
         img_count: int = 1) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "bbox": [x1, y1, x2, y2],
        "area": (x2 - x1) * (y2 - y1) if area is None else area,
        "keypoints": _kpts(),
        "img_gt_count": img_count,
    }


def _pred(bbox: tuple[float, float, float, float], conf: float = 0.9) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {"bbox": [x1, y1, x2, y2], "conf": conf, "keypoints": _kpts()}


# ============ compute_oks ============


def test_oks_perfect_match_is_one() -> None:
    """预测与 GT 完全一致 → OKS = 1.0。"""
    assert pb.compute_oks(_kpts(), _kpts(), 10000.0) == pytest.approx(1.0)


def test_oks_offset_value() -> None:
    """已知偏移：全部 17 点平移 d=26 → 逐点 exp(-d²/(2σ²A)) 的均值（对表复算）。"""
    gt = _kpts(x0=0.0)
    pred = _kpts(x0=26.0)
    oks = pb.compute_oks(gt, pred, 10000.0)
    expected = float(np.mean([
        math.exp(-26.0 ** 2 / (2 * s ** 2 * 10000.0)) for s in pb.OKS_SIGMAS
    ]))
    assert oks == pytest.approx(expected)


def test_oks_only_counts_visible_points() -> None:
    """GT v=0 的点不参与分子分母（官方口径）。"""
    gt = [(0.0, 0.0, 0.0)] * 17  # 全不可见
    assert pb.compute_oks(gt, _kpts(), 10000.0) == 0.0

    gt2 = [(0.0, 0.0, 1.0)] + [(99.0, 99.0, 0.0)] * 16  # 只有 nose 可见
    pred2 = [(26.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 16
    assert pb.compute_oks(gt2, pred2, 10000.0) == pytest.approx(math.exp(-0.5))


def test_oks_zero_area_and_short_pred() -> None:
    """面积 ≤0 → 0；预测不足 17 点按 (0,0) 补位不崩溃。"""
    assert pb.compute_oks(_kpts(), _kpts(), 0.0) == 0.0
    oks = pb.compute_oks(_kpts(), _kpts(n=5), 10000.0)
    assert 0.0 < oks < 1.0  # 前 5 点完美，其余按 (0,0) 计


# ============ match_predictions ============


def test_match_greedy_by_conf() -> None:
    """conf 降序贪心：高 conf 预测优先拿走 GT，低 conf 另匹配。"""
    gts = [_gt((0, 0, 100, 100)), _gt((200, 200, 300, 300))]
    preds = [
        _pred((0, 0, 100, 100), conf=0.6),  # 与 pred[1] 竞争同一 GT
        _pred((0, 0, 100, 100), conf=0.9),
        _pred((200, 200, 300, 300), conf=0.5),
    ]
    matches = pb.match_predictions(preds, gts)
    assert sorted(m[0] for m in matches) == [0, 1]  # 两个 GT 各匹配一次
    assert [m[1] for m in matches] == [1.0, 1.0]  # 完美 OKS


def test_match_iou_gating() -> None:
    """无重叠（IoU=0）→ 不匹配；完全重叠 → 匹配。"""
    gts = [_gt((0, 0, 100, 100))]
    assert pb.match_predictions([_pred((500, 500, 600, 600))], gts) == []
    assert len(pb.match_predictions([_pred((0, 0, 100, 100))], gts)) == 1


def test_match_each_gt_once() -> None:
    """同一 GT 至多匹配一次（第二预测落入已用 GT 不得重复）。"""
    gts = [_gt((0, 0, 100, 100))]
    preds = [_pred((0, 0, 100, 100), conf=0.9), _pred((0, 0, 100, 100), conf=0.8)]
    matches = pb.match_predictions(preds, gts)
    assert len(matches) == 1
    assert matches[0][0] == 0


# ============ ap_11point ============


def test_ap_all_true_positive() -> None:
    """全部匹配且 OKS 超阈 → AP = 1.0。"""
    assert pb.ap_11point([1.0, 1.0], 2, 0.5) == pytest.approx(1.0)


def test_ap_empty_and_zero_gt() -> None:
    """无匹配 / GT 为 0 → AP = 0。"""
    assert pb.ap_11point([], 3, 0.5) == 0.0
    assert pb.ap_11point([1.0], 0, 0.5) == 0.0


def test_ap_threshold_boundary() -> None:
    """OKS 严格大于阈值才算 TP（=0.75 不算）。"""
    assert pb.ap_11point([0.6], 1, 0.5) == pytest.approx(1.0)
    assert pb.ap_11point([0.6], 1, 0.75) == 0.0


def test_ap_interpolation() -> None:
    """11-point 插值：2 GT、1 TP → 6/11（recall=0.5 以下全为 precision 1.0）。"""
    assert pb.ap_11point([1.0, 0.4], 2, 0.5) == pytest.approx(6.0 / 11.0)


# ============ summarize_pose ============


def test_summarize_subsets_boundary() -> None:
    """分层边界：area<32² 进 small（1024 不进）；img_gt_count≥8 进 dense。"""
    gts = [
        _gt((0, 0, 31, 33), area=31 * 33, img_count=1),     # 1023 < 1024 → small
        _gt((0, 0, 32, 32), area=1024, img_count=1),        # 1024 → 非 small
        _gt((0, 0, 50, 50), area=2500, img_count=8),        # dense
    ]
    matches = [(0, 1.0), (1, 1.0), (2, 0.6)]
    r = pb.summarize_pose(gts, matches)

    assert r["all"]["gt"] == 3 and r["all"]["matched"] == 3
    assert r["small"]["gt"] == 1 and r["small"]["matched"] == 1
    assert r["dense"]["gt"] == 1 and r["dense"]["matched"] == 1
    # small 子集只含完美匹配 → AP50 = 1.0；dense 子集 OKS 0.6 < 0.75
    assert r["small"]["AP50"] == pytest.approx(1.0)
    assert r["dense"]["AP75"] == 0.0
    # mAP 是 10 阈值均值，介于 AP75 与 AP50 之间
    assert r["all"]["AP75"] <= r["all"]["mAP"] <= r["all"]["AP50"]
    # 键齐全 + per_threshold 10 项
    assert len(r["all"]["per_threshold"]) == 10


def test_summarize_no_matches() -> None:
    """零匹配：各子集 GT 计数保留，AP 全 0。"""
    r = pb.summarize_pose([_gt((0, 0, 10, 10)), _gt((0, 0, 100, 100))], [])
    assert r["all"]["gt"] == 2 and r["all"]["matched"] == 0
    assert r["all"]["mAP"] == 0.0 and r["small"]["mAP"] == 0.0


# ============ load_pose_ground_truth ============


def test_load_pose_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GT 加载：crowd 排除、bbox→xyxy、keypoints 三元组、img_gt_count 去 crowd。"""
    fake = tmp_path / "person_keypoints_val2017.json"
    kpts_flat = [float(i) for i in range(51)]
    fake.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [10, 20, 30, 40], "keypoints": kpts_flat, "iscrowd": 0},
            {"image_id": 1, "bbox": [0, 0, 5, 5], "keypoints": kpts_flat, "iscrowd": 1},
            {"image_id": 2, "bbox": [1, 2, 3, 4], "keypoints": kpts_flat, "iscrowd": 0},
        ],
    }))
    monkeypatch.setattr(pb, "GT_JSON", fake)

    gt = pb.load_pose_ground_truth()

    assert set(gt.keys()) == {1, 2}
    assert gt[1]["file_name"] == "a.jpg"
    assert len(gt[1]["persons"]) == 1  # crowd 排除
    p = gt[1]["persons"][0]
    assert p["bbox"] == [10.0, 20.0, 40.0, 60.0]  # xywh → xyxy
    assert p["area"] == pytest.approx(1200.0)
    assert p["keypoints"][0] == (0.0, 1.0, 2.0)
    assert len(p["keypoints"]) == 17
    assert p["img_gt_count"] == 1  # 只计非 crowd
    assert gt[2]["persons"][0]["img_gt_count"] == 1


def test_load_pose_gt_max_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """max_images 裁剪。"""
    fake = tmp_path / "person_keypoints_val2017.json"
    fake.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [
            {"image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": [0.0] * 51, "iscrowd": 0},
            {"image_id": 2, "bbox": [0, 0, 1, 1], "keypoints": [0.0] * 51, "iscrowd": 0},
        ],
    }))
    monkeypatch.setattr(pb, "GT_JSON", fake)
    assert len(pb.load_pose_ground_truth(max_images=1)) == 1


def test_format_pose_table_rows() -> None:
    """表格三行齐备（all/small/dense），数值格式 4 位小数。"""
    gts = [_gt((0, 0, 100, 100), area=10000, img_count=8)]
    r = pb.summarize_pose(gts, [(0, 1.0)])
    table = pb.format_pose_table(r)
    assert "全部" in table and "small" in table and "dense" in table
    assert "1.0000" in table
