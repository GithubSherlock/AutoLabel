"""test_web3d：复核队列四端点 + 安全模式 + 3D save 直通（TestClient，零权重）。

REVIEW3D_DIR 环境变量在 fixture 内设置并 reload server 模块（模块级共享）；
恢复时再次 reload 还原默认目录绑定。
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from auto3dlabel.tests.helpers.synth import ANCHOR_CAM, VELO_ANCHOR, write_calib, write_frame

BOX = {
    "label": "Car", "confidence": 0.62, "cx": 8.0, "cy": 1.4, "cz": 18.0,
    "h": 1.5, "w": 1.6, "l": 3.9, "rotation_y": 0.05, "fit_points": 31,
    "truncated": 0.0, "occluded": 0, "alpha": 0.0,
    "x1": 580.0, "y1": 170.0, "x2": 700.0, "y2": 230.0, "review_flag": False,
}
BOX2 = dict(BOX, label="Pedestrian", cx=3.0, cz=7.0, confidence=0.4, fit_points=9)


@pytest.fixture()
def web(tmp_path: Path) -> Iterator[tuple[Any, Path]]:
    """REVIEW3D_DIR → 每用例独立临时目录 + reload server（app/目录常量重新绑定）。"""
    review_dir = tmp_path / "review3d"
    review_dir.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("REVIEW3D_DIR")
    os.environ["REVIEW3D_DIR"] = str(review_dir)
    from auto3dlabel.web import server

    importlib.reload(server)
    yield server, review_dir
    if old is None:
        os.environ.pop("REVIEW3D_DIR", None)
    else:
        os.environ["REVIEW3D_DIR"] = old
    importlib.reload(server)


@pytest.fixture()
def client(web: tuple[Any, Path]) -> TestClient:
    server, _ = web
    return TestClient(server.app)


def _queue(image_path: str) -> dict:
    return {
        "image": "000000",
        "image_path": image_path,
        "image_size": [1242, 375],
        "pcd_path": "/tmp/velodyne/000000.bin",
        "calib_path": "/tmp/calib/000000.txt",
        "bev_path": "",
        "annotations": [BOX, BOX2],
    }


def _write_queue(web: tuple[Any, Path], name: str, data: dict[str, Any]) -> None:
    server, review_dir = web
    (review_dir / name).write_text(json.dumps(data), encoding="utf-8")


def test_index_html(client: Any) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Auto3dLabel" in res.text


def test_list_empty(client: Any) -> None:
    res = client.get("/api/review-files").json()
    assert res["files"] == [] and res["reviewed"] == []


def test_list_files_counts_and_corrupt_skip(web: Any, client: Any) -> None:
    server, review_dir = web
    _write_queue(web, "000000_review.json", _queue("/tmp/a.png"))
    (review_dir / "000001_review.json").write_text("{corrupt", encoding="utf-8")  # 损坏跳过
    _write_queue(web, "000002_reviewed.json", _queue("/tmp/b.png"))
    res = client.get("/api/review-files").json()
    assert [f["name"] for f in res["files"]] == ["000000_review.json"]
    assert res["files"][0]["count"] == 2
    assert [f["name"] for f in res["reviewed"]] == ["000002_reviewed.json"]
    assert "review3d" in res["dir"]


def test_get_review_file_3d_fields(web: Any, client: Any) -> None:
    _write_queue(web, "000000_review.json", _queue("/tmp/a.png"))
    data = client.get("/api/review-file", params={"name": "000000_review.json"}).json()
    assert len(data["annotations"]) == 2
    assert data["annotations"][0]["cx"] == 8.0  # 3D 字段直读
    assert data["annotations"][0]["fit_points"] == 31
    assert data["pcd_path"].endswith("000000.bin")


def test_get_review_file_security(web: Any, client: Any) -> None:
    _write_queue(web, "000000_review.json", _queue("/tmp/a.png"))
    assert client.get("/api/review-file", params={"name": "../etc/passwd"}).status_code == 400
    assert client.get("/api/review-file", params={"name": "x.txt"}).status_code == 400
    assert client.get("/api/review-file", params={"name": "000999_review.json"}).status_code == 404
    (web[1] / "000003_review.json").write_text("{bad", encoding="utf-8")
    assert client.get("/api/review-file", params={"name": "000003_review.json"}).status_code == 400


def test_get_review_image_whitelist(client: Any, tmp_path: Any) -> None:
    frame = write_frame(tmp_path)  # 真实 PNG
    res = client.get("/api/review-image", params={"path": str(frame.image_path)})
    assert res.status_code == 200 and len(res.content) > 100
    # 非图片后缀 / 不存在的图片 → 404
    assert client.get("/api/review-image", params={"path": "/etc/passwd"}).status_code == 404
    assert client.get("/api/review-image", params={"path": "/tmp/nope.png"}).status_code == 404


def test_save_delete_mode_exports_label(web: Any, client: Any, tmp_path: Any) -> None:
    """删除第 2 框 → 3D _reviewed.json + KITTI label 15 字段（y 底部回写）。"""
    frame = write_frame(tmp_path)
    _write_queue(web, "000000_review.json", _queue(str(frame.image_path)))
    res = client.post(
        "/api/review-save",
        json={"queue_file": "000000_review.json", "deleted_indices": [1]},
    ).json()
    assert res["ok"] and res["kept"] == 1 and res["deleted"] == 1, res

    server, review_dir = web
    saved = json.loads((review_dir / "000000_reviewed.json").read_text())
    assert saved["annotations"][0]["label"] == "Car"
    assert saved["annotations"][0]["fit_points"] == 31  # 3D 字段直通
    assert not (review_dir / "000000_review.json").exists()  # 源文件标记 .reviewed

    parts = (server.LABELS_DIR / "000000.txt").read_text().split()
    assert parts[0] == "Car" and len(parts) == 15
    assert (float(parts[8]), float(parts[9]), float(parts[10])) == (1.5, 1.6, 3.9)
    assert abs(float(parts[12]) - 2.15) < 1e-6  # y = cy + h/2 底部中心


def test_save_edited_mode_full_rebuild(web: Any, client: Any, tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    _write_queue(web, "000000_review.json", _queue(str(frame.image_path)))
    edited = [dict(BOX, confidence=0.9, edited_by_human=True)]
    res = client.post(
        "/api/review-save",
        json={"queue_file": "000000_review.json", "edited": edited},
    ).json()
    assert res["ok"] and res["kept"] == 1 and res["deleted"] == 1

    server, review_dir = web
    saved = json.loads((review_dir / "000000_reviewed.json").read_text())
    assert saved["annotations"][0]["confidence"] == 0.9
    assert saved["annotations"][0]["edited_by_human"] is True  # 人工修正标记


def test_save_security_and_missing(web: Any, client: Any) -> None:
    _write_queue(web, "000000_review.json", _queue("/tmp/a.png"))
    assert client.post("/api/review-save", json={"queue_file": "../x.json"}).status_code == 400
    assert client.post(
        "/api/review-save", json={"queue_file": "000999_review.json"},
    ).status_code == 404
    (web[1] / "000004_review.json").write_text("{bad", encoding="utf-8")
    assert client.post(
        "/api/review-save", json={"queue_file": "000004_review.json"},
    ).status_code == 400


def test_save_delete_all_no_label(web: Any, client: Any, tmp_path: Any) -> None:
    frame = write_frame(tmp_path)
    _write_queue(web, "000000_review.json", _queue(str(frame.image_path)))
    res = client.post(
        "/api/review-save",
        json={"queue_file": "000000_review.json", "deleted_indices": [0, 1]},
    ).json()
    assert res["ok"] and res["kept"] == 0 and res["deleted"] == 2
    server, review_dir = web
    saved = json.loads((review_dir / "000000_reviewed.json").read_text())
    assert saved["annotations"] == []
    assert not (server.LABELS_DIR / "000000.txt").exists()  # 全删不导出 label


def test_frame_data_endpoint(web: Any, client: Any, tmp_path: Any) -> None:
    """frame-data 端点（v0.3 P4）：相机系点（真实 calib 锚点）+ 8x3 角点 + 安全态。"""
    pcd = tmp_path / "000000.pcd.bin"
    np.asarray([[*(VELO_ANCHOR), 0.5]], dtype=np.float32).tofile(pcd)
    calib = tmp_path / "000000.txt"
    write_calib(calib)
    data = _queue("/tmp/a.png")
    data["pcd_path"] = str(pcd)
    data["calib_path"] = str(calib)
    _write_queue(web, "000000_review.json", data)

    res = client.get("/api/frame-data", params={"name": "000000_review.json"})
    assert res.status_code == 200
    payload = res.json()
    points = np.asarray(payload["points"])
    assert points.shape == (1, 3)
    # velo_to_cam 数值：velodyne 锚点 → 相机系 (1.7834,1.4169,8.4302)
    np.testing.assert_allclose(points[0], ANCHOR_CAM, atol=1e-5)
    corners = np.asarray(payload["objects"][0]["corners"])
    assert corners.shape == (8, 3)

    # 安全态：非法名 400 / 不存在 404 / pcd 缺失 400
    assert client.get("/api/frame-data", params={"name": "../x"}).status_code == 400
    assert client.get("/api/frame-data", params={"name": "000999_review.json"}).status_code == 404
    no_pcd = dict(_queue("/tmp/a.png"), pcd_path="/tmp/nope.bin")
    _write_queue(web, "000001_review.json", no_pcd)
    assert client.get("/api/frame-data", params={"name": "000001_review.json"}).status_code == 400


def test_reopen_reviewed_and_resave(web: Any, client: Any, tmp_path: Any) -> None:
    """已复核文件重开保存：源为 _reviewed.json 不再 rename。"""
    frame = write_frame(tmp_path)
    _write_queue(web, "000000_review.json", _queue(str(frame.image_path)))
    client.post(
        "/api/review-save", json={"queue_file": "000000_review.json", "deleted_indices": [1]},
    )
    res = client.post(
        "/api/review-save",
        json={"queue_file": "000000_reviewed.json", "edited": [BOX]},
    ).json()
    assert res["ok"] and res["kept"] == 1
    _, review_dir = web
    saved = json.loads((review_dir / "000000_reviewed.json").read_text())
    assert saved["annotations"][0]["label"] == "Car"  # 原位更新（源为 _reviewed 不再 rename）
