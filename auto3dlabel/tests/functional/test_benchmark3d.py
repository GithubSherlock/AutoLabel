"""test_benchmark3d：GT 7 值解析 + iou_fn 注入评测 + 难度分层 + 表格渲染。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto3dlabel.benchmarks.kitti3d_benchmark import (
    format_benchmark_table,
    run_kitti3d_benchmark,
)
from auto3dlabel.benchmarks.load_gt3d import gt_frame, pred_frame
from auto3dlabel.schema.box3d import Box3D, KittiFrame
from auto3dlabel.tests.helpers.synth import gt_line, write_frame


def _gt_frame(tmp_path: Path, lines: list[str], frame_id: str = "000000") -> KittiFrame:
    return write_frame(tmp_path, frame_id=frame_id, label_lines=lines)


def test_gt_frame_parse(tmp_path: Any) -> None:
    """真实口径 label_2 行 → objects（7 值 bbox + 8 值 quad + difficulty）。"""
    frame = _gt_frame(tmp_path, [gt_line(), gt_line(name="DontCare", x=1.0)])
    data = gt_frame(frame)
    objs = data["objects"]
    assert len(objs) == 1  # DontCare 剔除
    o = objs[0]
    assert o["name"] == "Car"
    assert len(o["bbox"]) == 7 and len(o["quad"]) == 8
    assert abs(o["bbox"][3] - 8.0) < 1e-9  # x
    assert o["difficulty"] in ("easy", "moderate", "hard")


def test_gt_frame_difficulty_hard_when_truncated(tmp_path: Any) -> None:
    """truncated=0.4（>0.3 超 moderate 上限、≤0.5 hard 上限）→ hard。"""
    frame = _gt_frame(tmp_path, [gt_line(truncated=0.4, y2=180.0)])
    objs = gt_frame(frame)["objects"]
    assert objs[0]["difficulty"] == "hard"


def test_gt_frame_truncation_too_heavy_excluded(tmp_path: Any) -> None:
    """truncated > 0.5 → 不参与评测（KITTI 官方口径，整对象剔除）。"""
    frame = _gt_frame(tmp_path, [gt_line(truncated=0.9, y2=260.0)])
    assert gt_frame(frame)["objects"] == []


def test_pred_frame_y_back_to_bottom() -> None:
    box = Box3D(
        label="Car", cx=8.0, cy=1.4, cz=18.0, h=1.5, w=1.6, l=3.9, yaw_bev=0.0, confidence=0.9,
    )
    p = pred_frame([box])[0]
    assert abs(p["bbox"][4] - (1.4 + 0.75)) < 1e-9  # y = cy + h/2
    assert len(p["quad"]) == 8
    assert p["conf"] == 0.9


def test_perfect_match_ap_100(tmp_path: Any) -> None:
    """预测 = GT → 3D/BEV AP 100。GT 只属 easy 档（每对象唯一难度）；其余档 gt_count=0。"""
    box = Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0)
    frame = _gt_frame(tmp_path, [gt_line()])
    result = run_kitti3d_benchmark([frame], {frame.frame_id: [box]})
    for layer in ("3d", "bev"):
        easy = result[layer]["easy"]["Car"]
        assert easy["gt_count"] == 1 and abs(easy["ap"] - 1.0) < 1e-6, (layer, easy)
        for diff in ("moderate", "hard"):
            stats = result[layer][diff]["Car"]
            assert stats["gt_count"] == 0 and abs(stats["ap"] - 0.0) < 1e-9, (layer, diff, stats)


def test_wrong_class_zero_ap(tmp_path: Any) -> None:
    """预测 Pedestrian vs GT Car → 0 匹配（Car 无预测 AP 0；Pedestrian 无 GT 全 FP AP 0）。"""
    box = Box3D.from_gt_row("Pedestrian", 1.7, 0.6, 0.8, 8.0, -0.9, 18.0, 0.0)
    frame = _gt_frame(tmp_path, [gt_line()])
    result = run_kitti3d_benchmark([frame], {frame.frame_id: [box]})
    assert abs(result["3d"]["easy"]["Car"]["ap"] - 0.0) < 1e-9
    assert abs(result["3d"]["easy"]["Pedestrian"]["ap"] - 0.0) < 1e-9
    assert result["3d"]["easy"]["Pedestrian"]["gt_count"] == 0


def test_offset_box_partial_ap(tmp_path: Any) -> None:
    """同帧两 GT 只命中其一 → 0 < AP < 1（单框命中 1/3 GT → 11 点插值 4/11）。"""
    frame = _gt_frame(tmp_path, [gt_line(), gt_line(x=30.0), gt_line(x=50.0)])
    box = Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0)
    result = run_kitti3d_benchmark([frame], {frame.frame_id: [box]})
    ap = result["3d"]["easy"]["Car"]["ap"]
    # recall=1/3 → 11 点插值 t≤0.3 共 4 档 → 4/11；evaluate_per_class 对 ap 保留 4 位小数
    assert ap == round(4.0 / 11.0, 4), ap
    assert result["3d"]["easy"]["Car"]["gt_count"] == 3


def test_multiple_frames_and_gt_difficulty_split(tmp_path: Any) -> None:
    """两帧：一帧匹配一帧无预测 → 11 点插值 AP = 6/11（recall=0.5 层 t≤0.5 共 6 档）。"""
    f1 = _gt_frame(tmp_path, [gt_line()], frame_id="000000")
    f2 = _gt_frame(tmp_path, [gt_line(x=20.0)], frame_id="000001")
    box = Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0)
    result = run_kitti3d_benchmark(
        [f1, f2], {f1.frame_id: [box], f2.frame_id: []}
    )
    assert result["3d"]["easy"]["Car"]["ap"] == round(6.0 / 11.0, 4)
    assert result["3d"]["easy"]["Car"]["gt_count"] == 2


def test_table_renders(tmp_path: Any) -> None:
    frame = _gt_frame(tmp_path, [gt_line()])
    box = Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0)
    result = run_kitti3d_benchmark([frame], {frame.frame_id: [box]})
    text = format_benchmark_table(result)
    assert "3D" in text and "BEV" in text and "Car" in text and "100.0" in text
