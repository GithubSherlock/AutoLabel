"""test_nuscenes_queue：triage_nus（fit_points 实测）+ 队列 JSON 协议（cam_like 渲染 dict）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auto3dlabel.configs.nuscenes import NUSCENES_CAMERAS
from auto3dlabel.export.nuscenes_queue import build_nuscenes_review_queue, triage_nus
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import yaw_to_quat


def _dense_box_points(rng: np.random.Generator, n: int = 100) -> np.ndarray:
    """全局系点云：围绕 (12,2,0.5)、w2×l4×h1.5 框内部的稠密簇（全部落框内）。"""
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = rng.uniform(11.2, 12.8, n)  # x ∈ (11,13)
    pts[:, 1] = rng.uniform(1.0, 3.0, n)    # y ∈ (0,4)
    pts[:, 2] = rng.uniform(0.2, 0.8, n)    # z ∈ (-0.25,1.25)
    return pts


def test_triage_nus_tiers_by_conf_and_fit_points() -> None:
    """三档：稠密高 conf → accepted；远框中 conf → review（fit_points<10 强制）；低 conf → hard。"""
    rng = np.random.default_rng(0)
    pts = _dense_box_points(rng)
    boxes = [
        NusBox(label="car", confidence=0.9, translation=(12.0, 2.0, 0.5),
               size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0)),
        NusBox(label="truck", confidence=0.5, translation=(40.0, 5.0, 0.0),
               size=(2.5, 8.0, 3.0), quaternion=(1.0, 0.0, 0.0, 0.0)),
        NusBox(label="pedestrian", confidence=0.1, translation=(30.0, 0.0, 0.0),
               size=(0.6, 0.7, 1.8), quaternion=(1.0, 0.0, 0.0, 0.0)),
    ]
    triage = triage_nus(boxes, pts, ego_translation=(10.0, 0.0, 0.0))
    assert [b.label for b in triage.accepted] == ["car"]
    assert [b.label for b in triage.review] == ["truck"]  # conf 0.5 + fit_points=0 强制
    assert [b.label for b in triage.hard] == ["pedestrian"]
    assert triage.accepted[0].fit_points == 100  # 稠密簇全部落框内


def test_triage_nus_min_fit_points_threshold() -> None:
    """NUSCENES_MIN_FIT_POINTS=10 门槛：恰好 10 点不强制 review，9 点强制 review。"""
    box = NusBox(label="car", confidence=0.9, translation=(12.0, 2.0, 0.5),
                 size=(2.0, 4.0, 1.5), quaternion=(1.0, 0.0, 0.0, 0.0))
    pts9 = _dense_box_points(np.random.default_rng(1), n=9)
    triage9 = triage_nus([box], pts9, ego_translation=(10.0, 0.0, 0.0))
    assert len(triage9.review) == 1 and triage9.review[0].fit_points == 9
    pts10 = _dense_box_points(np.random.default_rng(2), n=10)
    triage10 = triage_nus([box], pts10, ego_translation=(10.0, 0.0, 0.0))
    assert len(triage10.accepted) == 1 and triage10.accepted[0].fit_points == 10


class _FakeQueueNusc:
    """假 devkit 表（队列 JSON 用）：sample_data = LIDAR_TOP + 6 相机。"""

    version = "v1.0-mini"

    def __init__(self) -> None:
        self._sd = {
            "sd-lidar": {"filename": "samples/LIDAR_TOP/n008.bin", "token": "lidar-t"},
        }
        for i, name in enumerate(NUSCENES_CAMERAS):
            self._sd[f"sd-{name}"] = {"filename": f"samples/{name}/c{i}.jpg", "token": f"t{i}"}

    def get(self, table: str, token: str) -> dict:
        assert table == "sample_data"
        return self._sd[token]


def _review_sample() -> dict:
    return {
        "token": "tok1",
        "data": {"LIDAR_TOP": "sd-lidar", **{n: f"sd-{n}" for n in NUSCENES_CAMERAS}},
    }


def test_build_nuscenes_review_queue_protocol(tmp_path: Path) -> None:
    """队列 JSON：dataset=="nuscenes" + 渲染 dict（velocity/track_id 透传）+ summary 计数。"""
    nusc = _FakeQueueNusc()
    sample = _review_sample()
    rng = np.random.default_rng(0)
    pts = _dense_box_points(rng)
    boxes = [
        NusBox(label="car", confidence=0.9, translation=(12.0, 2.0, 0.5),
               size=(2.0, 4.0, 1.5), quaternion=yaw_to_quat(0.3),
               velocity=(1.0, 0.0), track_id="inst-7"),  # accepted → 不进队列
        NusBox(label="truck", confidence=0.4, translation=(40.0, 5.0, 0.0),
               size=(2.5, 8.0, 3.0), quaternion=(1.0, 0.0, 0.0, 0.0),
               velocity=(0.0, 2.0), track_id="inst-8"),  # 中 conf → review
    ]
    path = build_nuscenes_review_queue(
        scene_name="scene-A", sample=sample, pred_boxes=boxes,
        points_global=pts, ego_translation=(10.0, 0.0, 0.0),
        nusc=nusc, dataroot=tmp_path, out_dir=tmp_path / "reviews",
    )
    assert path == tmp_path / "reviews" / "tok1_review.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset"] == "nuscenes"
    assert data["version"] == "v1.0-mini"
    assert data["scene_name"] == "scene-A" and data["sample_token"] == "tok1"
    assert data["dataroot"] == str(tmp_path)
    assert data["ego_translation"] == [10.0, 0.0, 0.0]
    assert data["pcd_path"] == str(tmp_path / "samples" / "LIDAR_TOP" / "n008.bin")
    assert [c["name"] for c in data["cameras"]] == list(NUSCENES_CAMERAS)
    assert data["summary"] == {
        "accepted": 1, "review_count": 1, "hard_count": 0, "total_need_review": 1,
    }
    # annotations：仅 review+hard 档；渲染 dict 键 + velocity/track_id 透传
    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert ann["label"] == "truck" and ann["velocity"] == [0.0, 2.0]
    assert ann["track_id"] == "inst-8"
    # cam_like 渲染锚点：translation (40,5,0) − ego (10,0,0) → (cx,cy,cz)=(−5,0,30)
    assert (ann["cx"], ann["cy"], ann["cz"]) == (-5.0, 0.0, 30.0)


def test_build_nuscenes_review_queue_empty_pred(tmp_path: Path) -> None:
    """零预测 → 空队列文件仍写出（Web 扫描协议与 KITTI 一致）。"""
    nusc = _FakeQueueNusc()
    path = build_nuscenes_review_queue(
        scene_name="scene-A", sample=_review_sample(), pred_boxes=[],
        points_global=np.zeros((1, 5)), ego_translation=(0.0, 0.0, 0.0),
        nusc=nusc, dataroot=tmp_path, out_dir=tmp_path / "reviews",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["annotations"] == []
    assert data["summary"]["total_need_review"] == 0
