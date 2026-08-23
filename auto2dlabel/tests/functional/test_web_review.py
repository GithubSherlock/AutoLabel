"""Web 复核队列端点测试（fastapi TestClient + tmp_path，零模型加载）。

覆盖：队列扫描（排除已复核/损坏文件、旧字段容错）、路径越界与后缀拒绝、
保存修正（过滤→COCO→重命名）。
（/api/export-coco 的 build_coco_dict 已在 test_coco_builder.py 单一验证。）
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from auto2dlabel.tests import Path, json
from auto2dlabel.web import server
from auto2dlabel.web.server import app

client = TestClient(app)


@pytest.fixture()
def review_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """将 REVIEW_DIR 指向临时目录，隔离真实 outputs/。"""
    monkeypatch.setattr(server, "REVIEW_DIR", tmp_path)
    return tmp_path


def _write_review_file(d: Path, name: str, data: dict[str, Any]) -> None:
    (d / name).write_text(json.dumps(data), encoding="utf-8")


def _sample_queue() -> dict[str, Any]:
    return {
        "image": "img_01",
        "image_path": "/data/img_01.png",
        "image_size": [640, 480],
        "annotations": [
            {"x": 10, "y": 20, "width": 100, "height": 200, "label": "car", "confidence": 0.5},
            {"x": 300, "y": 400, "width": 50, "height": 60, "label": "person", "confidence": 0.4},
        ],
        "summary": {},
    }


# ── 队列扫描 ──────────────────────────────────────────────────

def test_review_files_scan(review_dir: Path) -> None:
    """扫描仅列出待复核文件：排除 .reviewed 与损坏 JSON。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    _write_review_file(review_dir, "img_02_review.json.reviewed", _sample_queue())  # 已复核
    (review_dir / "img_03_review.json").write_text("{broken", encoding="utf-8")  # 损坏

    resp = client.get("/api/review-files")
    assert resp.status_code == 200
    files = resp.json()["files"]
    assert [f["name"] for f in files] == ["img_01_review.json"]
    assert files[0]["image_path"] == "/data/img_01.png"
    assert files[0]["image_stem"] == "img_01"
    assert files[0]["count"] == 2


def test_review_files_legacy_fields_tolerated(review_dir: Path) -> None:
    """旧版队列文件缺 image_path/image_size 时置空容错。"""
    _write_review_file(review_dir, "old_review.json", {"image": "old", "annotations": []})
    resp = client.get("/api/review-files")
    assert resp.status_code == 200
    assert resp.json()["files"][0]["image_path"] == ""
    assert resp.json()["files"][0]["count"] == 0


def test_review_files_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REVIEW_DIR 不存在时返回空列表而非报错。"""
    monkeypatch.setattr(server, "REVIEW_DIR", tmp_path / "nonexistent")
    resp = client.get("/api/review-files")
    assert resp.status_code == 200
    assert resp.json()["files"] == []


# ── 队列文件读取 ──────────────────────────────────────────────

def test_review_file_get(review_dir: Path) -> None:
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    resp = client.get("/api/review-file", params={"name": "img_01_review.json"})
    assert resp.status_code == 200
    assert resp.json()["image"] == "img_01"
    assert len(resp.json()["annotations"]) == 2


def test_review_file_traversal_and_suffix_rejected(review_dir: Path) -> None:
    """路径遍历与非法后缀拒绝（不读任意 JSON 文件）。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    # 路径遍历
    resp = client.get("/api/review-file", params={"name": "../../etc/passwd_review.json"})
    assert resp.status_code == 400
    # 后缀不符
    resp = client.get("/api/review-file", params={"name": "img_01.json"})
    assert resp.status_code == 400
    # 不存在
    resp = client.get("/api/review-file", params={"name": "missing_review.json"})
    assert resp.status_code == 404


# ── 原图读取 ──────────────────────────────────────────────────

def test_review_image_whitelist(review_dir: Path, tmp_path: Path) -> None:
    """仅允许图片后缀，拒绝任意文件读取。"""
    from PIL import Image

    img_path = tmp_path / "real.png"
    Image.new("RGB", (4, 4)).save(img_path)

    resp = client.get("/api/review-image", params={"path": str(img_path)})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")

    # 非图片后缀拒绝
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    resp = client.get("/api/review-image", params={"path": str(secret)})
    assert resp.status_code == 404
    # 无后缀
    resp = client.get("/api/review-image", params={"path": "/etc/passwd"})
    assert resp.status_code == 404
    # 不存在
    resp = client.get("/api/review-image", params={"path": str(tmp_path / "missing.png")})
    assert resp.status_code == 404


# ── 保存修正 ──────────────────────────────────────────────────

def test_review_save_filter_and_rename(review_dir: Path) -> None:
    """保存修正：过滤删除框 → 写 COCO → 源队列文件标记 .reviewed。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())

    resp = client.post("/api/review-save", json={
        "queue_file": "img_01_review.json",
        "deleted_indices": [1],  # 删除 person
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["kept"] == 1 and body["deleted"] == 1

    # 源文件已重命名标记，不再出现在队列
    assert (review_dir / "img_01_review.json.reviewed").is_file()
    assert not (review_dir / "img_01_review.json").exists()
    assert client.get("/api/review-files").json()["files"] == []

    # 修正结果写为 COCO（含 image_size 与过滤后的框）
    coco = json.loads((review_dir / "img_01_reviewed.json").read_text(encoding="utf-8"))
    assert coco["images"] == [{"id": 1, "file_name": "img_01.png", "width": 640, "height": 480}]
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [10, 20, 100, 200]
    assert [c["name"] for c in coco["categories"]] == ["car"]


def test_review_save_rejects_bad_names(review_dir: Path) -> None:
    """非法/不存在的队列文件被拒绝。"""
    resp = client.post("/api/review-save", json={
        "queue_file": "../evil_review.json", "deleted_indices": [],
    })
    assert resp.status_code == 400
    resp = client.post("/api/review-save", json={
        "queue_file": "missing_review.json", "deleted_indices": [],
    })
    assert resp.status_code == 404


# ── 已复核列表与 issues（v0.5 Web 增强） ──────────────────────

def test_review_files_reviewed_list(review_dir: Path) -> None:
    """保存修正后 review-files 响应含 reviewed 列表（供重开复查）。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    resp = client.post("/api/review-save", json={
        "queue_file": "img_01_review.json", "deleted_indices": [],
    })
    assert resp.status_code == 200

    data = client.get("/api/review-files").json()
    assert data["files"] == []  # 原队列已出队
    assert [r["name"] for r in data["reviewed"]] == ["img_01_reviewed.json"]
    assert data["reviewed"][0]["count"] == 2
    assert data["reviewed"][0]["image_path"] == "/data/img_01.png"
    assert data["reviewed"][0]["image_stem"] == "img_01"


def test_review_files_reviewed_corrupt_tolerated(review_dir: Path) -> None:
    """损坏的 *_reviewed.json 在 reviewed 扫描中跳过不炸。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    (review_dir / "bad_reviewed.json").write_text("{broken", encoding="utf-8")
    data = client.get("/api/review-files").json()
    assert data["reviewed"] == []
    assert len(data["files"]) == 1  # 正常待复核文件不受影响


def test_review_save_issues_persisted(review_dir: Path) -> None:
    """payload 带 issues → *_reviewed.json 顶层 issues + image_path 落盘。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    issues = [{"id": 1, "x": 1, "y": 1, "width": 5, "height": 5, "text": "重标", "status": "open"}]
    resp = client.post("/api/review-save", json={
        "queue_file": "img_01_review.json", "deleted_indices": [], "issues": issues,
    })
    assert resp.status_code == 200
    out = json.loads((review_dir / "img_01_reviewed.json").read_text(encoding="utf-8"))
    assert out["issues"] == issues
    assert out["image_path"] == "/data/img_01.png"  # 顶层 image_path 供重开显示原图


def test_review_save_edited_with_issues_both_applied(review_dir: Path) -> None:
    """edited 全量重建 + issues 同时生效（互不干扰）。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    resp = client.post("/api/review-save", json={
        "queue_file": "img_01_review.json",
        "edited": [{"x": 1, "y": 2, "width": 3, "height": 4, "label": "car", "confidence": 0.5}],
        "issues": [{"id": 1, "x": 0, "y": 0, "width": 1, "height": 1,
                    "text": "漏检", "status": "open"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["kept"] == 1 and body["deleted"] == 1
    out = json.loads((review_dir / "img_01_reviewed.json").read_text(encoding="utf-8"))
    assert len(out["annotations"]) == 1
    assert len(out["issues"]) == 1


def test_review_save_resave_reviewed(review_dir: Path) -> None:
    """重开已复核文件修正后保存：COCO 源归一化（category_id→label）+ 原地覆写。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    client.post("/api/review-save", json={
        "queue_file": "img_01_review.json", "deleted_indices": [],
    })

    # 重开 reviewed 文件：deleted_indices 模式（COCO 源需归一化）
    resp = client.post("/api/review-save", json={
        "queue_file": "img_01_reviewed.json", "deleted_indices": [1],
    })
    assert resp.status_code == 200
    assert resp.json()["kept"] == 1

    out = json.loads((review_dir / "img_01_reviewed.json").read_text(encoding="utf-8"))
    assert len(out["annotations"]) == 1
    assert out["annotations"][0]["bbox"] == [10, 20, 100, 200]  # car 保留，归一化正确
    assert out["images"][0]["width"] == 640  # 图像尺寸经 images[0] 回退不丢
    # 无 .reviewed.reviewed 重复标记
    assert not (review_dir / "img_01_reviewed.json.reviewed").exists()
    # 仍出现在 reviewed 列表
    data = client.get("/api/review-files").json()
    assert [r["name"] for r in data["reviewed"]] == ["img_01_reviewed.json"]


def test_review_file_get_reviewed_format(review_dir: Path) -> None:
    """GET /api/review-file 读 *_reviewed.json（COCO dict + issues 顶层键）。"""
    _write_review_file(review_dir, "img_01_review.json", _sample_queue())
    client.post("/api/review-save", json={
        "queue_file": "img_01_review.json", "deleted_indices": [],
        "issues": [{"id": 1, "x": 0, "y": 0, "width": 1, "height": 1,
                    "text": "t", "status": "open"}],
    })
    resp = client.get("/api/review-file", params={"name": "img_01_reviewed.json"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["annotations"]) == 2
    assert data["annotations"][0]["bbox"] == [10, 20, 100, 200]  # COCO bbox 数组
    assert data["issues"][0]["text"] == "t"
