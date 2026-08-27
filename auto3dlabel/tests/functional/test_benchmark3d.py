"""test_benchmark3d：GT 7 值解析 + iou_fn 注入评测 + 难度分层 + 表格渲染。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auto3dlabel.benchmarks.kitti3d_benchmark import (
    format_benchmark_table,
    run_kitti3d_benchmark,
)
from auto3dlabel.benchmarks.kitti_official_ap import run_kitti_official
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


# ── 官方 40-point 口径（kitti_official_ap，与模型 zoo 对表）──────────────


def _with_bbox2d(
    box: Box3D, x1: float = 500.0, y1: float = 150.0, x2: float = 700.0, y2: float = 300.0
) -> Box3D:
    """给预测框附官方豁免所需的 2D 外接框（高 150px ≥ MIN_HEIGHT，不会被 ignored_dt）。"""
    box.x1, box.y1, box.x2, box.y2 = x1, y1, x2, y2
    return box


def test_official_ap_perfect_match_sparse_penalty(tmp_path: Any) -> None:
    """预测 = GT → precision/recall 100 但 AP40 = 0（官方稀疏惩罚）。

    阈值采样只取 1 个 TP 分数 → 40 个采样点中仅 1 个非零 → AP = 0/40。
    这是官方 eval 对极小样本的真实行为（对照口径回归锚点，非缺陷）。
    """
    box = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0))
    frame = _gt_frame(tmp_path, [gt_line()])
    result = run_kitti_official([frame], {frame.frame_id: [box]})
    for diff in ("easy", "moderate", "hard"):
        stats = result[diff]["Car"]
        assert stats["gt_count"] == 1 and stats["pred_count"] == 1, (diff, stats)
        assert stats["precision"] == 1.0 and stats["recall"] == 1.0, (diff, stats)
        assert stats["ap"] == 0.0, (diff, stats)


def test_official_ap_partial_recall(tmp_path: Any) -> None:
    """3 GT 命中 1 → precision 1.0 / recall 1/3 / AP 0（同上稀疏惩罚）。"""
    box = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0))
    frame = _gt_frame(tmp_path, [gt_line(), gt_line(x=30.0), gt_line(x=50.0)])
    result = run_kitti_official([frame], {frame.frame_id: [box]})
    stats = result["easy"]["Car"]
    assert stats["gt_count"] == 3 and stats["pred_count"] == 1
    assert stats["precision"] == 1.0 and abs(stats["recall"] - 1.0 / 3.0) < 1e-9
    assert stats["ap"] == 0.0


def test_official_ap_car_iou_threshold_07(tmp_path: Any) -> None:
    """x+1m 偏移（3D IoU≈0.59）：0.5 < IoU < 0.7 → Car 官方阈值 0.7 下不匹配 → AP 0。

    同框在现有 11-point 口径（IoU 0.5）下会命中——双口径差异的实证用例。
    """
    box = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 9.0, -0.9, 18.0, 0.0))
    frame = _gt_frame(tmp_path, [gt_line()])
    result = run_kitti_official([frame], {frame.frame_id: [box]})
    stats = result["easy"]["Car"]
    assert stats["gt_count"] == 1 and stats["pred_count"] == 1
    assert abs(stats["ap"] - 0.0) < 1e-9
    # 对照组：11-point + IoU 0.5 口径同框命中 → AP 100（差异即口径差）
    legacy = run_kitti3d_benchmark([frame], {frame.frame_id: [box]})
    assert abs(legacy["3d"]["easy"]["Car"]["ap"] - 1.0) < 1e-6


def test_official_ap_no_pred_keeps_gt_count(tmp_path: Any) -> None:
    """无预测 → AP 0 但 gt_count 保留（表内显零）。"""
    frame = _gt_frame(tmp_path, [gt_line()])
    result = run_kitti_official([frame], {frame.frame_id: []})
    stats = result["easy"]["Car"]
    assert stats["gt_count"] == 1 and stats["pred_count"] == 0
    assert abs(stats["ap"] - 0.0) < 1e-9


def test_official_ignored_gt_absorbs_pred(tmp_path: Any) -> None:
    """超 hard 范围 GT（truncated=0.9）留在匹配池吸收预测 → 不算 FP（官方豁免）。

    旧近似口径将超 hard 对象整剔 → 命中它的预测变 FP → precision 0.5；
    官方口径吸收后 precision 1.0。难度累积：hard 档同样包含 easy GT（gt_count=1）。
    """
    box1 = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0))
    box2 = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 30.0, -0.9, 18.0, 0.0))
    frame = _gt_frame(tmp_path, [gt_line(), gt_line(truncated=0.9, x=30.0, y2=260.0)])
    result = run_kitti_official([frame], {frame.frame_id: [box1, box2]})
    for diff in ("easy", "moderate", "hard"):
        stats = result[diff]["Car"]
        assert stats["gt_count"] == 1, (diff, stats)  # 超 hard 对象不计入有效 GT
        assert stats["precision"] == 1.0, (diff, stats)  # 匹配超 hard GT 的预测被吸收
        assert stats["pred_count"] == 2, (diff, stats)


def test_official_dontcare_criterion0_exempts_fp(tmp_path: Any) -> None:
    """DontCare 豁免判据 = 交/预测面积 > 0.7（官方 criterion=0，非并集 IoU）。

    预测 2D 框完全落入 DontCare 内（交/预测面积 = 1.0）→ 豁免 FP → precision 1.0。
    旧近似口径用并集 IoU（此处 = 0.25 < 0.7）不豁免 → precision 0.5——判据差异的实证用例。
    """
    box1 = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 8.0, -0.9, 18.0, 0.0))
    box2 = _with_bbox2d(Box3D.from_gt_row("Car", 1.5, 1.6, 3.9, 31.0, -0.9, 18.0, 0.0))
    frame = _gt_frame(
        tmp_path,
        [
            gt_line(),
            gt_line(x=30.0),
            gt_line(name="DontCare", x1=400.0, y1=100.0, x2=800.0, y2=400.0),
        ],
    )
    result = run_kitti_official([frame], {frame.frame_id: [box1, box2]})
    stats = result["easy"]["Car"]
    # box2 命中不了任何 GT（IoU 0.59），但其 2D 框在 DontCare 内 → 官方豁免不罚 FP
    assert stats["gt_count"] == 2
    assert stats["precision"] == 1.0, stats
    assert abs(stats["recall"] - 0.5) < 1e-9
