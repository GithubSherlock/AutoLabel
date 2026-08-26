"""test_cli_track3d：_run_tracked_sequence 集成（Fake 注入，零真实权重）。

覆盖：track_id 回写 → tracks.json（帧列表）/ 复核队列 to_dict 穿透 / KITTI label 15 字段纯净。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from auto3dlabel.cli import _run_tracked_sequence
from auto3dlabel.schema.box3d import Box3D, FrameResult, KittiFrame
from auto3dlabel.tests.helpers.synth import write_frame


def _box(x: float, z: float, conf: float) -> Box3D:
    return Box3D(
        label="Car", cx=x, cy=0.0, cz=z, h=1.5, w=1.6, l=3.9,
        yaw_bev=0.0, confidence=conf, fit_points=50,
    )


def test_run_tracked_sequence_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 帧：静止车 + 复核车连续、第 2 帧新车 → tracks.json 3 轨迹 + label 15 字段纯净。"""
    frames: dict[str, KittiFrame] = {}
    for fid in ("000000", "000001"):
        frames[fid] = write_frame(tmp_path / "kitti", frame_id=fid)

    stationary = _box(0.0, 10.0, 0.9)
    review_car = _box(5.0, 20.0, 0.5)  # conf 0.5 → review 档（穿透点验证）
    newcomer = _box(30.0, 40.0, 0.9)

    def fake_annotate(
        frame: KittiFrame,
        prompts: list[str],
        det_model: object | None = None,
        seg_model: object | None = None,
        det3d: object | None = None,
        confidence: float = 0.3,
        viz: bool = True,
        out_dir: str | Path | None = None,
    ) -> FrameResult:
        boxes = [stationary, review_car] if frame.frame_id == "000000" else [
            stationary, review_car, newcomer,
        ]
        return FrameResult(frame_id=frame.frame_id, boxes3d=boxes)

    monkeypatch.setattr("auto3dlabel.cli._make_models", lambda dm, sm: (None, None, None))
    monkeypatch.setattr("auto3dlabel.cli.resolve_frame", lambda fid: frames[fid])
    monkeypatch.setattr("auto3dlabel.tools.pipeline.annotate_frame", fake_annotate)

    out_dir = tmp_path / "out"
    stat = _run_tracked_sequence(
        ["000000", "000001"], ["car"], "pointpillars_kitti", None, 0.3, out_dir, False
    )

    assert stat == {"accepted": 3, "review": 2, "hard": 0, "failed": 0}

    # tracks.json：3 条轨迹，帧列表正确
    tracks = json.loads((out_dir / "tracks.json").read_text(encoding="utf-8"))
    assert tracks["sequence"] == "000000-000001"
    by_frames = {tuple(t["frames"]): t for t in tracks["tracks"]}
    assert ("000000", "000001") in by_frames  # 静止车 + 复核车（连续）
    assert ("000001",) in by_frames  # 第 2 帧新车
    assert len(tracks["tracks"]) == 3

    # KITTI label 15 字段纯净（track_id 不入 label）
    for fid, n_lines in (("000000", 2), ("000001", 3)):
        lines = (out_dir / "labels" / f"{fid}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_lines
        assert all(len(line.split()) == 15 for line in lines)

    # 复核队列：to_dict 自动穿透 track_id（review 档 2 帧同 id）
    review0 = json.loads((out_dir / "reviews" / "000000_review.json").read_text(encoding="utf-8"))
    review1 = json.loads((out_dir / "reviews" / "000001_review.json").read_text(encoding="utf-8"))
    anns0, anns1 = review0["annotations"], review1["annotations"]
    assert len(anns0) == 1 and len(anns1) == 1
    assert anns0[0]["track_id"] == anns1[0]["track_id"]
