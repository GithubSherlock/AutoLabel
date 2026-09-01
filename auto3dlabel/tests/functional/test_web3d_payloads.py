"""test_web3d_payloads：四视图 payload 纯函数（v0.3 P4；零权重零网络）。

velo_to_cam 数值断言用 synth.py 真实 calib 锚点（velodyne (8.752,-1.800,-1.546)
→ 相机系 (1.7834,1.4169,8.4302)，投影链已数值验证）——payload 契约 = 相机系点
+ corners_cam 8x3 角点（yaw 数学留在 Python 侧，前端零 calib 依赖）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.tests.helpers.synth import ANCHOR_CAM, VELO_ANCHOR, write_calib
from auto3dlabel.web.payloads import (
    MAX_POINTS,
    downsample_points,
    frame_payload,
    validate_queue_name,
)


def _write_frame(tmp_path: Path) -> tuple[Path, Path]:
    """合成 pcd（真实锚点 + 一散点）+ 真实 calib 文件。"""
    pcd = tmp_path / "000123.pcd.bin"
    np.asarray(
        [[*VELO_ANCHOR, 0.5], [1.0, 2.0, 3.0, 0.1]], dtype=np.float32
    ).tofile(pcd)
    calib = tmp_path / "000123.txt"
    write_calib(calib)
    return pcd, calib


def _queue(pcd_path: Path, calib_path: Path) -> dict:
    box = Box3D(
        label="Car", cx=1.0, cy=-1.0, cz=8.0, h=1.5, w=1.6, l=3.9, yaw_bev=0.3
    ).to_dict()
    return {
        "image": "000123",
        "image_path": "",
        "bev_path": "",
        "pcd_path": str(pcd_path),
        "calib_path": str(calib_path),
        "annotations": [box, "not-a-dict", 42],  # 坏框跳过（宁缺勿假）
    }


def test_downsample_points_identity_and_seed() -> None:
    """不超限原样返回同一对象；超限 seed 确定性 + 原数组子集。"""
    pts = np.arange(30, dtype=np.float32).reshape(10, 3)
    assert downsample_points(pts) is pts

    big = np.random.default_rng(0).uniform(size=(150, 3))
    out1 = downsample_points(big, max_points=50, seed=7)
    out2 = downsample_points(big, max_points=50, seed=7)
    assert out1.shape == (50, 3)
    np.testing.assert_array_equal(out1, out2)  # 同 seed 同结果
    # 每行都是原数组的某一行（子集）
    match = np.all(np.any(np.all(out1[:, None, :] == big[None, :, :], axis=2), axis=1))
    assert match
    # idx 升序 → 保序：行唯一递增的整数点阵上，抽样结果行序与原序一致
    int_pts = np.arange(450, dtype=np.float32).reshape(150, 3)
    int_out = downsample_points(int_pts, max_points=50, seed=7)
    assert np.all(np.diff(int_out[:, 0]) > 0)
    assert MAX_POINTS == 100_000  # 交付上限单一事实源


def test_validate_queue_name() -> None:
    """白名单：后缀 + 路径遍历拒绝。"""
    assert validate_queue_name("000123_review.json")
    assert validate_queue_name("000123_reviewed.json")
    assert not validate_queue_name("../etc_review.json")
    assert not validate_queue_name("a/b_review.json")
    assert not validate_queue_name(r"a\b_review.json")
    assert not validate_queue_name("x.txt")
    assert not validate_queue_name("")


def test_frame_payload_contract(tmp_path: Path) -> None:
    """payload 契约：相机系点（真实 calib 锚点）+ corners_cam 8x3 往返一致 + 坏框跳过。"""
    pcd, calib = _write_frame(tmp_path)
    queue = tmp_path / "000123_review.json"
    queue.write_text(json.dumps(_queue(pcd, calib)), encoding="utf-8")

    payload = frame_payload("000123_review.json", tmp_path)
    assert payload is not None
    assert payload["image"] == "000123"
    points = np.asarray(payload["points"])
    assert points.shape == (2, 3)
    # 锚点：velodyne (8.752,-1.800,-1.546) → 相机系 (1.7834,1.4169,8.4302)
    np.testing.assert_allclose(points[0], ANCHOR_CAM, atol=1e-5)
    assert len(payload["objects"]) == 1  # 非 dict/42 跳过
    obj = payload["objects"][0]
    assert obj["index"] == 0  # 原 annotations 下标（表格/保存回写对齐）
    assert obj["label"] == "Car" and obj["fit_points"] == 0
    corners = np.asarray(obj["corners"])
    assert corners.shape == (8, 3)
    # 与 Box3D 直接计算一致（from_dict 往返 + corners 透传）
    expected = Box3D.from_dict(_queue(pcd, calib)["annotations"][0]).corners_cam()
    np.testing.assert_allclose(corners, expected, atol=1e-9)


def test_frame_payload_damaged_or_missing(tmp_path: Path) -> None:
    """损坏 JSON / 缺 pcd / 缺 calib / 标定不完整 → None。"""
    (tmp_path / "bad_review.json").write_text("{corrupt", encoding="utf-8")
    assert frame_payload("bad_review.json", tmp_path) is None

    pcd, calib = _write_frame(tmp_path)
    q = _queue(pcd, calib)
    no_pcd = dict(q, pcd_path="/tmp/nope.bin")
    (tmp_path / "a_review.json").write_text(json.dumps(no_pcd), encoding="utf-8")
    assert frame_payload("a_review.json", tmp_path) is None

    bad_calib = tmp_path / "bad.txt"
    bad_calib.write_text("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")  # 缺 R0_rect
    q2 = dict(q, calib_path=str(bad_calib))
    (tmp_path / "b_review.json").write_text(json.dumps(q2), encoding="utf-8")
    assert frame_payload("b_review.json", tmp_path) is None


def _nus_queue(tmp_path: Path, pcd_path: Path) -> dict:
    """nuScenes 队列 JSON（零 devkit）：pcd 绝对路径 + ego_translation + 6 相机。"""
    box = Box3D(
        label="car", cx=-5.0, cy=0.2, cz=5.5, h=1.5, w=2.0, l=4.0, yaw_bev=0.3
    ).to_dict()
    return {
        "dataset": "nuscenes",
        "version": "v1.0-mini",
        "dataroot": str(tmp_path),
        "scene_name": "scene-A",
        "sample_token": "tok1",
        "image": "tok1",
        "pcd_path": str(pcd_path),
        "ego_translation": [4.5, -3.0, 0.7],
        "cameras": [
            {"name": "CAM_FRONT", "filename": "samples/CAM_FRONT/n008.jpg"},
            {"name": "CAM_FRONT_LEFT", "filename": "samples/CAM_FRONT_LEFT/n008.jpg"},
            {"name": "CAM_FRONT_RIGHT", "filename": "samples/CAM_FRONT_RIGHT/n008.jpg"},
            {"name": "CAM_BACK", "filename": "samples/CAM_BACK/n008.jpg"},
            {"name": "CAM_BACK_LEFT", "filename": "samples/CAM_BACK_LEFT/n008.jpg"},
            {"name": "CAM_BACK_RIGHT", "filename": "samples/CAM_BACK_RIGHT/n008.jpg"},
            {"name": "CAM_BROKEN", "filename": ""},  # 缺 filename 跳过
            "not-a-dict",
        ],
        "annotations": [box, "bad", 42],  # 坏框跳过（宁缺勿假）
    }


def test_frame_payload_nuscenes_cam_like_and_cameras(tmp_path: Path) -> None:
    """nus 分支：cam_like 点（ego 锚点）+ NaN 剔除 + 6 相机路径 + 坏框跳过。"""
    pcd = tmp_path / "samples" / "LIDAR_TOP" / "n008.bin"
    pcd.parent.mkdir(parents=True)
    np.asarray(
        [[10.0, 2.0, 0.5, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0],
         [np.nan, np.nan, np.nan, np.nan, np.nan]],  # NaN 行剔除
        dtype=np.float32,
    ).tofile(pcd)
    queue = tmp_path / "tok1_review.json"
    queue.write_text(json.dumps(_nus_queue(tmp_path, pcd)), encoding="utf-8")

    payload = frame_payload("tok1_review.json", tmp_path)
    assert payload is not None
    assert payload["dataset"] == "nuscenes"
    points = np.asarray(payload["points"])
    assert points.shape == (2, 3)
    # 锚点：p − ego = (5.5, 5.0, −0.2) → cam_like (−5.0, 0.2, 5.5)
    np.testing.assert_allclose(points[0], (-5.0, 0.2, 5.5), atol=1e-6)
    np.testing.assert_allclose(points[1], (-3.0, 0.7, -4.5), atol=1e-6)
    assert [c["name"] for c in payload["cameras"]] == [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    # image_path = dataroot 绝对路径（review-image 端点读）
    assert payload["cameras"][0]["image_path"] == str(
        tmp_path / "samples" / "CAM_FRONT" / "n008.jpg"
    )
    assert len(payload["objects"]) == 1  # 非 dict/42 跳过
    assert payload["objects"][0]["index"] == 0
    assert payload["objects"][0]["label"] == "car"
    # nus 无 image_path/bev_path（相机图走 cameras 分支，前端渲染 6 图）
    assert payload["image_path"] == "" and payload["bev_path"] == ""


def test_frame_payload_nuscenes_damaged_pcd(tmp_path: Path) -> None:
    """nus 分支 pcd 缺失 → None（端点 400）；dataset 键缺失走 KITTI 分支回归不变。"""
    queue = tmp_path / "tok1_review.json"
    queue.write_text(
        json.dumps(_nus_queue(tmp_path, tmp_path / "nope.bin")), encoding="utf-8"
    )
    assert frame_payload("tok1_review.json", tmp_path) is None
