"""自动车道 ROI（v0.5 C2）测试 —— UFLD 解码/自车车道选择/ROI 路由纯函数。

零真实权重铁律：FakeLaneModel duck typing 注入（同 ReID/FakeScorer 模式）；
decode_lane_output 吃合成 (GRIDING_NUM+1, NUM_ROW, NUM_LANES) 数组。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from auto2dlabel.models.lane import (
    GRIDING_NUM,
    NUM_LANES,
    NUM_ROW,
    TUSIMPLE_ROW_ANCHOR,
    UltraFastLaneONNXModel,
    create_lane_model,
    decode_lane_output,
    points_to_polygon,
    resolve_auto_roi,
    select_ego_lane_pair,
)
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tools.constraints import filter_by_spatial

# ---------- 合成输出构造 ----------

def _lane_logits(
    lane: int,
    rows: list[int],
    cls: int = 5,
    peak: float = 10.0,
    no_lane_peak: float = 0.0,
) -> NDArray[np.float32]:
    """(GRIDING_NUM+1, NUM_ROW, NUM_LANES) 合成 logits。

    指定行在 cls 列放峰值（软 argmax 期望列 ≈ cls+1）；其余行无车道类
    （第 GRIDING_NUM 类）为峰值且压过车道类默认填充 → 解码丢弃。
    """
    a = np.full((GRIDING_NUM + 1, NUM_ROW, NUM_LANES), -10.0, dtype=np.float32)
    a[GRIDING_NUM, :, :] = no_lane_peak  # 无车道类默认峰（> -10 车道类填充）
    for k in rows:
        a[GRIDING_NUM, k, lane] = -10.0
        a[cls, k, lane] = peak
    return a


# ---------- 纯函数：多边形 / 自车车道对 ----------

def test_points_to_polygon_closes_loop() -> None:
    """左线正序 + 右线反序闭合（顶点可追踪一圈）。"""
    left = [(0.0, 100.0), (10.0, 200.0), (20.0, 300.0)]
    right = [(100.0, 100.0), (110.0, 200.0), (120.0, 300.0)]
    poly = points_to_polygon(left, right)
    assert poly == [
        (0.0, 100.0), (10.0, 200.0), (20.0, 300.0),  # 左线 上→下
        (120.0, 300.0), (110.0, 200.0), (100.0, 100.0),  # 右线 下→上
    ]


def test_select_ego_lane_pair_straddles_center() -> None:
    """4 线场景：取底端夹住画面中线的相邻对（lane1/lane2）。"""
    lanes = [
        [(0.0, 0.0), (0.0, 100.0)],          # 左左
        [(200.0, 0.0), (200.0, 100.0)],      # 左
        [(500.0, 0.0), (500.0, 100.0)],      # 右
        [(900.0, 0.0), (900.0, 100.0)],      # 右右
    ]
    pair = select_ego_lane_pair(lanes, img_width=1000)
    assert pair == [lanes[1], lanes[2]]


def test_select_ego_lane_pair_all_one_side_fallback() -> None:
    """全部在中线一侧（弯道）：退化为最靠近中线的两线。"""
    lanes = [
        [(50.0, 0.0), (50.0, 100.0)],
        [(100.0, 0.0), (100.0, 100.0)],
        [(150.0, 0.0), (150.0, 100.0)],
    ]
    pair = select_ego_lane_pair(lanes, img_width=1000)
    assert pair == [lanes[2], lanes[1]]  # 150 与 100 最靠近中线 500


def test_select_ego_lane_pair_too_few() -> None:
    """不足 2 线 → 空（调用方回退 None）。"""
    lanes = [[(100.0, 0.0), (100.0, 100.0)]]
    assert select_ego_lane_pair(lanes, img_width=1000) == []


def test_select_ego_lane_pair_exact_two() -> None:
    """恰 2 线：直接取（含夹线判定同值）。"""
    lanes = [[(300.0, 0.0), (300.0, 100.0)], [(700.0, 0.0), (700.0, 100.0)]]
    assert select_ego_lane_pair(lanes, img_width=1000) == lanes


# ---------- 纯函数：UFLD 输出解码 ----------

def test_decode_lane_output_maps_xy_and_flips_rows() -> None:
    """rel 定位 + ONNX 行序反转 + 原图坐标反变换（squash 线性映射）。"""
    a = _lane_logits(lane=0, rows=[0, 1, 2, 3], cls=5)
    lanes = decode_lane_output(a, orig_w=1280, orig_h=720)
    assert len(lanes) == 1  # 仅 lane0 有效（其余全无车道）
    pts = lanes[0]
    # 软 argmax 期望列 ≈ cls+1=6 → x = 6 * (799/99) * 1280/800 ≈ 77.5
    assert pts[0][0] == pytest.approx(6 * (799 / 99) * 1280 / 800, abs=1.0)
    # raw 行 r ↔ anchor[r]（翻转 + anchor[55-k] 与 ailia 后处理逐行等价，
    # raw 行 0 = 顶部 anchor[0]=64 → y=160）
    assert pts[0][1] == pytest.approx(TUSIMPLE_ROW_ANCHOR[0] * 720 / 288, abs=1.0)
    assert pts[3][1] == pytest.approx(TUSIMPLE_ROW_ANCHOR[3] * 720 / 288, abs=1.0)
    assert pts[0][1] < pts[3][1]  # 上→下：y 递增


def test_decode_lane_output_drops_short_lane() -> None:
    """有效点 <3 的车道整体丢弃；无车道行不产出点。"""
    a = _lane_logits(lane=1, rows=[10, 11])  # 仅 2 行有效
    lanes = decode_lane_output(a, orig_w=1280, orig_h=720)
    assert lanes == []


def test_decode_lane_output_no_lane_sentinel() -> None:
    """无车道类峰值 → 该行丢弃（不产出垃圾点）。"""
    a = _lane_logits(lane=0, rows=[0, 1, 2, 3])
    a[:, 4, 0] = -10.0  # 抹掉第 4 行车道信号
    a[GRIDING_NUM, 4, 0] = 10.0  # 无车道类接管
    lanes = decode_lane_output(a, orig_w=1280, orig_h=720)
    assert len(lanes) == 1
    assert all(abs(p[1] - 710) > 1 for p in lanes[0])  # 第 4 行（y≈700）被丢弃


# ---------- FakeLaneModel + resolve_auto_roi / 过滤链集成 ----------

class _FakeLaneModel:
    """伪车道模型：返回固定车道点列（零权重）。"""

    def __init__(self, lanes: list[list[tuple[float, float]]]) -> None:
        self._lanes = lanes

    def detect_lanes(self, image_path: str) -> list[list[tuple[float, float]]]:
        return self._lanes


def _make_img(tmp_path: Path, w: int = 1000, h: int = 600) -> str:
    from PIL import Image

    p = tmp_path / "frame.jpg"
    Image.new("RGB", (w, h)).save(p)
    return str(p)


def test_resolve_auto_roi_two_lanes(tmp_path: Path) -> None:
    """2 车道 → 闭合多边形（夹线对 + 正反序拼接）。"""
    lanes = [
        [(300.0, 100.0), (300.0, 500.0)],
        [(700.0, 100.0), (700.0, 500.0)],
    ]
    poly = resolve_auto_roi(_FakeLaneModel(lanes), _make_img(tmp_path))
    assert poly is not None
    assert poly[0] == (300.0, 100.0)
    assert poly[-1] == (700.0, 100.0)  # 右线反序回顶 → 闭合


def test_resolve_auto_roi_ego_pair_selection(tmp_path: Path) -> None:
    """4 车道：只取夹住中线（500）的相邻两线成多边形。"""
    lanes = [
        [(0.0, 100.0), (0.0, 500.0)],
        [(400.0, 100.0), (400.0, 500.0)],
        [(600.0, 100.0), (600.0, 500.0)],
        [(990.0, 100.0), (990.0, 500.0)],
    ]
    poly = resolve_auto_roi(_FakeLaneModel(lanes), _make_img(tmp_path))
    assert poly is not None
    xs = [p[0] for p in poly]
    assert 0.0 not in xs and 990.0 not in xs  # 外侧两线被排除
    assert 400.0 in xs and 600.0 in xs


def test_resolve_auto_roi_single_lane_none(tmp_path: Path) -> None:
    """<2 车道 → None（调用方黄字 + 全图保留，宁多勿漏）。"""
    lanes = [[(300.0, 100.0), (300.0, 500.0)]]
    assert resolve_auto_roi(_FakeLaneModel(lanes), _make_img(tmp_path)) is None


def test_resolve_auto_roi_filters_spatially(tmp_path: Path) -> None:
    """车道多边形与 3a 过滤链集成：车道内框保留、车道外框剔除。"""
    lanes = [
        [(300.0, 100.0), (300.0, 500.0)],
        [(700.0, 100.0), (700.0, 500.0)],
    ]
    poly = resolve_auto_roi(_FakeLaneModel(lanes), _make_img(tmp_path))
    assert poly is not None
    inside = Bbox(x=450.0, y=250.0, width=100.0, height=100.0, label="car", confidence=0.9)
    outside = Bbox(x=50.0, y=250.0, width=80.0, height=80.0, label="car", confidence=0.9)
    kept = filter_by_spatial([inside, outside], roi=poly)
    assert kept == [inside]


# ---------- 工厂 / 懒加载 ----------

def test_create_lane_model_factory() -> None:
    """工厂返回 UltraFastLaneONNXModel（Protocol duck typing）。"""
    m = create_lane_model()
    assert isinstance(m, UltraFastLaneONNXModel)
    assert m._session is None  # 构造零加载（onnxruntime 不参与）


# ---------- --roi auto CLI 路由（不落真实权重） ----------

def test_auto_roi_routing_success(tmp_path: Path, monkeypatch: Any) -> None:
    """--roi auto 走 UFLD 首帧解析（collect_frames + create_lane_model 均为 Fake）。"""
    from PIL import Image

    from auto2dlabel import cli_track

    frame = tmp_path / "frame_0.jpg"
    Image.new("RGB", (1000, 600)).save(frame)
    lanes = [[(300.0, 100.0), (300.0, 500.0)], [(700.0, 100.0), (700.0, 500.0)]]
    monkeypatch.setattr(
        cli_track, "collect_frames",
        lambda source, output: [frame],
    )
    monkeypatch.setattr(
        "auto2dlabel.models.lane.create_lane_model",
        lambda: _FakeLaneModel(lanes),
    )
    poly = cli_track._resolve_auto_roi(str(tmp_path), str(tmp_path))
    assert poly is not None
    assert poly[0] == (300.0, 100.0)


def test_auto_roi_routing_failure_degrades(tmp_path: Path, monkeypatch: Any) -> None:
    """模型失败 → 黄字 + None（全图保留，不阻断跟踪）。"""
    from auto2dlabel import cli_track

    class _Boom:
        def detect_lanes(self, image_path: str) -> list[list[tuple[float, float]]]:
            raise RuntimeError("no weights")

    monkeypatch.setattr(
        cli_track, "collect_frames",
        lambda source, output: [tmp_path / "frame_0.jpg"],
    )
    monkeypatch.setattr("auto2dlabel.models.lane.create_lane_model", lambda: _Boom())
    assert cli_track._resolve_auto_roi(str(tmp_path), str(tmp_path)) is None


def test_auto_roi_routing_no_frames(tmp_path: Path, monkeypatch: Any) -> None:
    """源无帧 → 黄字 + None。"""
    from auto2dlabel import cli_track

    monkeypatch.setattr(cli_track, "collect_frames", lambda source, output: [])
    assert cli_track._resolve_auto_roi(str(tmp_path), str(tmp_path)) is None
