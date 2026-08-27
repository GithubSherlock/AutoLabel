"""test_nuscenes_export_bench：提交 JSON 自检 + 简化评测精确值（零真实数据）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto3dlabel.benchmarks.nuscenes_benchmark import nus_iou, nuscenes_ap, run_nuscenes_benchmark
from auto3dlabel.export.nuscenes_json import (
    build_submission_json,
    validate_submission,
    write_submission,
)
from auto3dlabel.schema.nuscenes_box import NusBox


def _box(
    x: float = 0.0,
    y: float = 0.0,
    label: str = "car",
    conf: float = 0.9,
    w: float = 2.0,
    l: float = 4.0,
    yaw: float = 0.0,
) -> NusBox:
    from auto3dlabel.tools.geometry import yaw_to_quat

    return NusBox(
        label=label, confidence=conf, translation=(x, y, 0.0),
        size=(w, l, 1.5), quaternion=yaw_to_quat(yaw),
    )


def test_nus_iou_anchors() -> None:
    """IoU 锚点：全等=1；半长平移=1/3；垂直交叉=1/3。"""
    a = _box()
    assert nus_iou(a, _box()) == 1.0
    half = nus_iou(a, _box(x=2.0))  # 沿车头平移 l/2
    assert abs(half - 1 / 3) < 1e-9
    crossed = nus_iou(a, _box(yaw=3.141592653589793 / 2))  # 2×4 垂直交叉
    assert abs(crossed - 1 / 3) < 1e-9


def test_nuscenes_ap_perfect_and_empty_and_half() -> None:
    """AP 锚点：完美=1 / 无预测=0 / 1 命中 1 误检（conf 降序）=0.75。"""
    gt = [_box(x=0.0), _box(x=10.0)]
    assert nuscenes_ap(gt, [_box(x=0.0), _box(x=10.0)], "car")["ap"] == 1.0
    assert nuscenes_ap(gt, [], "car")["ap"] == 0.0
    pred = [_box(x=0.0, conf=0.9), _box(x=50.0, conf=0.8)]  # 命中 + 误检
    # tp=[1,0] fp=[0,1]：recall≤0.5 处 precision=1，recall>0.5 无 → AP=(1×5+0×5)/10=0.5
    assert nuscenes_ap(gt, pred, "car")["ap"] == 0.5


def test_run_benchmark_distance_bins() -> None:
    """距离分桶：近距命中、远距误检分桶统计。"""
    gt = {"s0": [_box(x=5.0), _box(x=30.0)]}  # 近 5m / 中 30m
    pred = {"s0": [_box(x=5.0), _box(x=50.0, conf=0.7)]}
    result = run_nuscenes_benchmark(gt, pred)
    assert result["overall"]["car"]["ap"] == 0.5
    assert result["distance"]["0-25m"]["car"]["ap"] == 1.0
    assert result["distance"]["25-50m"]["car"]["ap"] == 0.0
    assert 0.0 <= result["mAP"] <= 1.0


def test_run_benchmark_distance_bins_ego_relative() -> None:
    """距离分桶相对自车（egos）：全局 x=40 目标距 ego(30,0) 仅 10m → 落 0-25m 桶。

    缺省 egos 相对全局原点（nus city 系目标距原点常 >500m，全落桶外）——
    两者在此用例下分桶相反，同时断言锁定口径。
    """
    gt = {"s0": [_box(x=40.0)]}
    pred = {"s0": [_box(x=40.0)]}
    result = run_nuscenes_benchmark(gt, pred, egos={"s0": (30.0, 0.0)})
    assert result["overall"]["car"]["ap"] == 1.0
    assert result["distance"]["0-25m"]["car"]["ap"] == 1.0  # 相对原点 40m 会错落 25-50m
    assert result["distance"]["0-25m"]["car"]["gt_count"] == 1
    assert result["distance"]["25-50m"]["car"]["ap"] == 0.0
    origin_rel = run_nuscenes_benchmark(gt, pred)  # 缺省 egos = 全局原点
    assert origin_rel["distance"]["25-50m"]["car"]["ap"] == 1.0


def test_build_and_validate_submission() -> None:
    """提交 JSON：官方 box 字段 + sample_token 补入 + 自检通过。"""
    boxes = [_box(x=1.0, conf=0.8), _box(x=2.0, label="pedestrian")]
    sub = build_submission_json({"tok1": boxes})
    assert sub["meta"]["use_lidar"] is True
    r = sub["results"]["tok1"]
    assert r[0]["sample_token"] == "tok1"
    assert r[0]["detection_name"] == "car" and r[0]["velocity"] == [0.0, 0.0]
    assert r[0]["attribute_name"] == "vehicle.moving"
    assert validate_submission(sub) == []


def test_validate_catches_bad_class_and_score() -> None:
    """自检拦截：未知类 / score 越界 / rotation 长度错（完整 box 基底经 build_submission_json）。"""
    good = build_submission_json({"t": [_box()]})["results"]["t"][0]
    bad_cls = {**good, "detection_name": "ufo"}
    bad_score = {**good, "detection_score": 1.5}
    bad_rot = {**good, "rotation": [1.0, 0.0, 0.0]}
    problems = validate_submission({"results": {"t": [bad_cls, bad_score, bad_rot]}})
    assert len(problems) == 3


def test_write_submission_rejects_invalid(tmp_path: Path) -> None:
    """自检未通过 → 抛 ValueError 不落盘（宁缺勿假）。"""
    sub = {"results": {"t": [{"detection_name": "ufo"}]}}
    with pytest.raises(ValueError, match="自检未通过"):
        write_submission(sub, tmp_path / "sub.json")
    valid = build_submission_json({"t": [_box()]})
    out = write_submission(valid, tmp_path / "sub.json")
    assert json.loads(out.read_text(encoding="utf-8"))["results"]["t"]
