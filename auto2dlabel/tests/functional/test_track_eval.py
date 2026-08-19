"""track_eval 单测 —— CLEAR MOT 指标数值正确性（MOTA/IDF1/IDSW/MT/ML）。

红线验证：评测必须同报 MOTA/IDF1 与检测 recall/IDSW——
完美匹配、ID 交换、FN/FP、MT/ML 边界各一例，数值手算对照。
"""

from __future__ import annotations

import tempfile

import pytest

from auto2dlabel.benchmarks.track_eval import (
    TrackBox,
    evaluate_tracking,
    load_track_boxes,
)
from auto2dlabel.tests import Path


def _tb(frame: int, tid: int, x: float = 0.0, y: float = 0.0) -> TrackBox:
    return TrackBox(frame=frame, track_id=tid, x=x, y=y, w=50.0, h=50.0)


# ============================================================
# 基准场景
# ============================================================

def test_perfect_tracking() -> None:
    """逐帧逐框一致：MOTA/IDF1=1.0，IDSW/FN/FP=0，全部 MT。"""
    gt = {f: [_tb(f, 1, 0.0, 0.0), _tb(f, 2, 100.0, 0.0)] for f in range(1, 4)}
    pred = {f: [_tb(f, 1, 0.0, 0.0), _tb(f, 2, 100.0, 0.0)] for f in range(1, 4)}

    m = evaluate_tracking(gt, pred)
    assert m.mota == pytest.approx(1.0)
    assert m.idf1 == pytest.approx(1.0)
    assert m.idsw == 0
    assert m.fn == 0 and m.fp == 0
    assert m.recall == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.mt == 2 and m.ml == 0 and m.num_gt_tracks == 2


def test_id_switch() -> None:
    """帧 2 两个 ID 互换：IDSW=2，MOTA=1-2/4=0.5，IDF1=0.5。"""
    gt = {
        1: [_tb(1, 1, 0.0, 0.0), _tb(1, 2, 100.0, 0.0)],
        2: [_tb(2, 1, 0.0, 0.0), _tb(2, 2, 100.0, 0.0)],
    }
    pred = {
        1: [_tb(1, 1, 0.0, 0.0), _tb(1, 2, 100.0, 0.0)],
        2: [_tb(2, 1, 100.0, 0.0), _tb(2, 2, 0.0, 0.0)],  # 互换
    }

    m = evaluate_tracking(gt, pred)
    assert m.idsw == 2, f"两轨迹同时换 ID 应计 2 次，实际 {m.idsw}"
    assert m.fn == 0 and m.fp == 0, "ID 交换不产生 FN/FP（位置正确）"
    assert m.mota == pytest.approx(1 - 2 / 4)
    assert m.idf1 == pytest.approx(0.5)
    assert m.recall == pytest.approx(1.0), "检测口径不受 ID 交换影响"


def test_fn_fp_and_negative_mota() -> None:
    """漏检 + 误检同现：FN/FP 计入 MOTA（可负），recall/precision 正确。"""
    gt = {
        1: [_tb(1, 5, 0.0, 0.0)],
        2: [_tb(2, 5, 0.0, 0.0)],
    }
    pred = {
        2: [_tb(2, 0, 0.0, 0.0), _tb(2, 1, 300.0, 0.0)],  # 匹配 + 误检
    }

    m = evaluate_tracking(gt, pred)
    assert m.fn == 1 and m.fp == 1
    assert m.num_gt == 2 and m.num_pred == 2
    assert m.mota == pytest.approx(1 - 2 / 2)
    assert m.recall == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.5)


def test_loose_iou_no_match() -> None:
    """IoU=0 的框不算匹配：各计 FN/FP。"""
    gt = {1: [_tb(1, 1, 0.0, 0.0)]}
    pred = {1: [_tb(1, 9, 100.0, 0.0)]}

    m = evaluate_tracking(gt, pred)
    assert m.fn == 1 and m.fp == 1
    assert m.recall == pytest.approx(0.0)


def test_mt_ml_boundaries() -> None:
    """MT/ML 按轨迹匹配占比 ≥0.8 / ≤0.2 判定。

    构造 3 条 10 帧轨迹：A 全匹配（MT）、B 只匹配 1 帧（ML）、C 匹配 5 帧（居中）。
    """
    frames = range(1, 11)
    gt = {f: [_tb(f, 1, 0.0, 0.0), _tb(f, 2, 100.0, 0.0), _tb(f, 3, 200.0, 0.0)]
          for f in frames}
    pred: dict[int, list[TrackBox]] = {}
    for f in frames:
        boxes = [_tb(f, 1, 0.0, 0.0)]  # A 全程匹配
        if f == 1:
            boxes.append(_tb(f, 2, 100.0, 0.0))  # B 仅首帧
        if f <= 5:
            boxes.append(_tb(f, 3, 200.0, 0.0))  # C 前 5 帧
        pred[f] = boxes

    m = evaluate_tracking(gt, pred)
    assert m.num_gt_tracks == 3
    assert m.mt == 1, f"A(10/10) 应为 MT，实际 MT={m.mt}"
    assert m.ml == 1, f"B(1/10) 应为 ML，实际 ML={m.ml}"
    assert m.idsw == 0
    # 手算：FN = 9(B) + 5(C) = 14；num_gt = 30 → MOTA = 1 - 14/30
    assert m.mota == pytest.approx(1 - 14 / 30)
    # IDF1：IDTP = 10 + 1 + 5 = 16；num_pred = 16 → IDFN = 14, IDFP = 0
    #       → 2·16/(2·16+14+0) = 32/46
    assert m.idf1 == pytest.approx(32 / 46)


def test_summary_line_and_to_dict() -> None:
    """summary_line 含红线要求的两口径（MOTA/IDF1 + recall）。"""
    gt = {1: [_tb(1, 1)]}
    m = evaluate_tracking(gt, gt)
    line = m.summary_line()
    assert "MOTA" in line and "IDF1" in line and "recall" in line
    assert m.to_dict()["mota"] == pytest.approx(1.0)


# ============================================================
# GT txt 解析
# ============================================================

def test_load_track_boxes() -> None:
    """MOT txt 解析：conf=0 跳过、类别过滤、frame 分组。"""
    txt = "\n".join([
        "1,1,10,20,100,200,1,1,0.9",   # frame1 person 活跃
        "1,2,300,400,50,60,1,1,0.8",   # frame1 person
        "1,3,500,400,50,60,0,1,0.9",   # conf=0 非活跃 → 跳过
        "2,1,12,22,100,200,1,7,0.9",   # frame2 class=7（static person）
        "2,9,0,0,10,10,1,3,0.9",       # class=3 车 → 过滤
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "gt.txt"
        p.write_text(txt)

        boxes = load_track_boxes(p, keep_classes={1, 7})
        assert set(boxes) == {1, 2}
        assert len(boxes[1]) == 2, "conf=0 行应跳过"
        assert {b.track_id for b in boxes[1]} == {1, 2}
        assert [b.track_id for b in boxes[2]] == [1], "class=3 应被类别过滤"

        all_boxes = load_track_boxes(p)
        assert len(all_boxes[2]) == 2, "不过滤类别时 class=3 保留"
