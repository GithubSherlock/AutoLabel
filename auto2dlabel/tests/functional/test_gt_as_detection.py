"""GT-as-detection 上界模式测试 —— GT 框直喂 ByteTracker 的理想检测上界语义。

上界模式（mot_tracking_benchmark --gt-as-detection）将 GT 框作为完美检测
（conf=1.0、recall=1、FP=0）喂入跟踪器，MOTA/IDF1 只反映跟踪关联能力——
本文件验证该语义：理想场景 MOTA/IDF1=1.0；FN/FP 结构性为零
（pred 由 GT 派生），MOTA 损失唯一来源是 IDSW。
"""

from __future__ import annotations

from auto2dlabel.benchmarks.mot_tracking_benchmark import run_gt_tracking
from auto2dlabel.benchmarks.track_eval import TrackBox, evaluate_tracking


def _gt_box(frame: int, track_id: int, x: float, y: float) -> TrackBox:
    """50×50 行人 GT 框（x,y 为左上角）。"""
    return TrackBox(frame=frame, track_id=track_id, x=x, y=y, w=50.0, h=50.0)


def test_perfect_detection_upper_bound_mota_idf1_one() -> None:
    """理想场景（检测=GT、匀速直线、无遮挡）：MOTA=IDF1=1.0、IDSW=0。

    上界语义验证：检测完美时跟踪器应零损失——逐帧全 ID、
    两目标 ID 帧间维持、recall/precision=1.0。
    """
    gt: dict[int, list[TrackBox]] = {
        f: [_gt_box(f, 0, 100.0 + 5.0 * f, 100.0),
            _gt_box(f, 1, 300.0 + 5.0 * f, 100.0)]
        for f in range(1, 11)
    }
    frames = sorted(gt)

    pred, elapsed = run_gt_tracking(gt, frames)

    assert elapsed >= 0
    for f in frames:
        assert len(pred[f]) == 2, "conf=1.0 高分框应逐帧全 ID 输出"
        assert {b.track_id for b in pred[f]} == {0, 1}, "两目标 ID 应帧间维持"

    metrics = evaluate_tracking(gt, pred)
    assert metrics.mota == 1.0
    assert metrics.idf1 == 1.0
    assert metrics.idsw == 0
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.mt == 2
    assert metrics.ml == 0


def test_occlusion_recovery_keeps_id_lossless() -> None:
    """GT 遮挡（MOT 约定 conf=0 不活跃框，load_track_boxes 剔除）后恢复：

    跟踪器遮挡期间无检测可见（conf=1 框从 GT 消失）——
    静止目标恢复后同 ID → 零 IDSW → 上界模式 MOTA 仍 1.0（FN/FP 结构性为零）。
    """
    gt: dict[int, list[TrackBox]] = {}
    for f in range(1, 11):
        boxes = [_gt_box(f, 0, 100.0 + 5.0 * f, 100.0)]
        if f not in (5, 6):  # 静止目标 1 遮挡（模拟 conf=0 被剔除）
            boxes.append(_gt_box(f, 1, 300.0, 100.0))
        gt[f] = boxes
    frames = sorted(gt)

    pred, _ = run_gt_tracking(gt, frames)

    assert len(pred[5]) == 1 and len(pred[6]) == 1, "遮挡帧目标 1 无检测输出"
    assert {b.track_id for b in pred[7]} == {0, 1}, "恢复后同 ID（无切换）"

    metrics = evaluate_tracking(gt, pred)
    assert metrics.fn == 0 and metrics.fp == 0, "上界模式 FN/FP 结构性为零"
    assert metrics.idsw == 0
    assert metrics.mota == 1.0
