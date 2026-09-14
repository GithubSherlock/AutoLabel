"""test_mapvec_compare：Chamfer 匹配（复制版）自证——TP/FP/FN、CD 分布、越窗计数。

交叉验证（复制 vs 产出方原版）见 test_mapvec_crosscheck.py——本文件只测复制版
自身语义（依赖 AutoLabel 内部纯值，零 AutoDriveData import）。
"""

from __future__ import annotations

import numpy as np

from auto3dlabel.schema.mapvec import MapVecFramePred, make_instance
from auto3dlabel.tools.mapvec_compare import (
    chamfer_cost_matrix,
    chamfer_distance,
    compare_frame,
    match_greedy,
)


def _line(x0: float, y0: float, n: int = 20, step: float = 1.0) -> np.ndarray:
    """一条 20 点水平折线：(x0+i, y0)。默认在 BEV 窗口 x∈[-15,15] 内。"""
    return np.column_stack([x0 + step * np.arange(n), np.full(n, y0)])


def _rec(preds: list, gts: list, frame: int = 200) -> MapVecFramePred:
    return MapVecFramePred(
        frame=frame,
        token=f"{frame:06d}",
        score_thr=0.2,
        ckpt="ckpt",
        preds=tuple(preds),
        gts=tuple(gts),
    )


def test_chamfer_distance_identical() -> None:
    """同线 CD = 0；错开 2m 的线 CD 由双向平均控制（≤2 且 >0）。"""
    a = _line(0, 0)
    assert chamfer_distance(a, a) == 0.0
    b = _line(2, 0)  # x 整体 +2
    d = chamfer_distance(a, b)
    assert 0.0 < d <= 2.0


def test_chamfer_cost_matrix_matches_bruteforce() -> None:
    preds = [_line(0, 0), _line(5, 1, 15), _line(0, 0)]  # 重复折线
    gts = [_line(0, 0), _line(6, 2, 12)]
    cost = chamfer_cost_matrix(preds, gts)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            assert abs(cost[i, j] - chamfer_distance(p, g)) < 1e-3


def test_match_greedy_counts() -> None:
    """3 pred 2 gt(1 近 1 远)：阈值 0.5 → 1 TP、2 FP、1 FN。"""
    preds = [_line(0, 0), _line(10, 0), _line(20, 0)]
    gts = [_line(0, 0), _line(50, 0)]  # 第二条远离所有 pred
    tp, fp, fn = match_greedy(preds, gts, thr=0.5)
    assert (tp, fp, fn) == (1, 2, 1)


def test_match_greedy_one_to_one() -> None:
    """GT 禁止复用：两条 pred 同贴一个 GT → 只有一条匹配（TP=1）。"""
    preds = [_line(0, 0), _line(0.2, 0)]  # 都贴 GT
    gts = [_line(0, 0)]
    tp, fp, fn = match_greedy(preds, gts, thr=0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_compare_frame_counts_and_hist() -> None:
    """compare_frame：逐类计数 + CD 直方（仅匹配对）+ 越窗计数。"""
    # divider 两线大部分在窗内（x∈[-10,9], y 贴近 30 界），仅尾部 1 点越窗
    preds = [
        make_instance("centerline", _line(0, 0), score=0.9),
        make_instance("divider", _line(-10, 30.5), score=0.8),
    ]
    gts = [make_instance("centerline", _line(0, 0)), make_instance("divider", _line(-10, 30.5))]
    rec = _rec(preds, gts)
    r = compare_frame(rec, thr=0.5)
    assert r.tp["centerline"] == 1 and r.tp["divider"] == 1
    assert r.fp["centerline"] == 0 and r.fn["divider"] == 0
    assert len(r.cd_dist_hist) == 2  # 两匹配对
    assert r.cd_dist_hist[0] <= 0.5
    # 越窗：GT y=30.5 > 30 界 → 20 点全越窗；pred 同 → 越窗计数=20
    assert r.gt_out_of_window >= 20
    assert r.out_of_window >= 20


def test_compare_frame_empty_pred_side() -> None:
    """pred 空 gt 有 → 全 FN；gt 空 pred 有 → 全 FP。"""
    rec = _rec([], [make_instance("boundary", _line(0, 0))])
    r = compare_frame(rec)
    assert r.fn["boundary"] == 1 and r.tp["boundary"] == 0
    rec2 = _rec([make_instance("boundary", _line(0, 0), score=0.5)], [])
    r2 = compare_frame(rec2)
    assert r2.fp["boundary"] == 1 and r2.fn["boundary"] == 0
    assert r2.cd_dist_hist == []


def test_compare_frame_duplicate_point_no_interference() -> None:
    """复制版 CD 语义：相同坐标多线不干扰 min（一条 GT 只被一条 pred 匹配）。"""
    preds = [make_instance("centerline", _line(0, 0), score=0.9) for _ in range(3)]
    gts = [make_instance("centerline", _line(0, 0))]
    r = compare_frame(_rec(preds, gts), thr=0.5)
    assert r.tp["centerline"] == 1 and r.fp["centerline"] == 2
