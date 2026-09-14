"""test_mapvec_schema：契约消费侧硬校验（坏 schema 拒收；与产出方一致）。"""

from __future__ import annotations

import pytest

from auto3dlabel.schema.mapvec import (
    BEV_RANGE,
    MAPTR_CLASSES,
    NUM_POINTS,
    SCHEMA_ID,
    frame_from_dict,
    frame_to_dict,
    out_of_window,
    validate_frame,
)

C = MAPTR_CLASSES


def _pts(x0: float = 1.0) -> list[list[float]]:
    return [[x0 + i, 0.5] for i in range(NUM_POINTS)]


def _frame_dict(**over) -> dict:
    d = {
        "schema": SCHEMA_ID,
        "frame": 200,
        "token": "000200",
        "classes": list(C),
        "coord": "ego",
        "bev_range": list(BEV_RANGE),
        "num_points": NUM_POINTS,
        "score_thr": 0.2,
        "ckpt": "outputs/maptr_ep512.pt",
        "preds": [{"class": "centerline", "points": _pts(), "score": 0.9}],
        "gts": [{"class": "centerline", "points": _pts()}],
    }
    d.update(over)
    return d


def test_roundtrip() -> None:
    rec = frame_from_dict(_frame_dict())
    assert rec.frame == 200 and rec.token == "000200"
    assert len(rec.preds) == 1 and rec.preds[0].cls == "centerline"
    back = frame_from_dict(frame_to_dict(rec))
    assert back.score_thr == rec.score_thr and back.ckpt == rec.ckpt


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "mapvec_pred/2"),
        ("classes", list(C)[::-1]),  # 类序不符
        ("coord", "global"),
        ("num_points", NUM_POINTS + 1),
    ],
)
def test_hard_reject(key: str, value) -> None:
    """契约 ID/类序/坐标系/点数不符 → 拒收（不静默）。"""
    with pytest.raises(ValueError):
        frame_from_dict(_frame_dict(**{key: value}))


def test_bad_instance_geometry() -> None:
    """坏几何：未知类 / 点数不足 / 非有限数 → validate_frame 抛。"""
    bad_cls = _frame_dict(preds=[{"class": "lane", "points": _pts(), "score": 0.5}])
    with pytest.raises(ValueError):
        frame_from_dict(bad_cls)
    bad_pts = _frame_dict(preds=[{"class": "centerline", "points": _pts()[:5], "score": 0.5}])
    with pytest.raises(ValueError):
        frame_from_dict(bad_pts)
    bad_nan = _frame_dict(
        preds=[{"class": "centerline", "points": [[float("nan"), 0.0]] * NUM_POINTS, "score": 0.5}]
    )
    with pytest.raises(ValueError):
        frame_from_dict(bad_nan)


def test_out_of_window_counts_but_not_rejected() -> None:
    """pred 越窗（x=19.75）只计数不拒收（越窗诊断，非错误）。"""
    rec = frame_from_dict(
        _frame_dict(
            preds=[
                {"class": "divider", "points": _pts(19.0), "score": 0.3},
                {"class": "divider", "points": _pts(1.0), "score": 0.3},
            ]
        )
    )
    n = out_of_window(rec)
    assert n >= NUM_POINTS  # 越窗线整条计（x=19.0+i 超出 15 界）
    # 但能正常构造通过校验
    validate_frame(rec)


def test_frame_without_score_is_gt() -> None:
    rec = frame_from_dict(_frame_dict())
    assert rec.preds[0].score is not None
    assert rec.gts[0].score is None
