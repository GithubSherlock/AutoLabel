"""Web 标注端点测试：task_type 三路由（detection/obb/classification）+
_bbox_from_dict 字段保真 + review-save edited 全量重建 + export-coco。

零真实权重：monkeypatch 模型工厂注入 Fake（duck typing Protocol）。
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from auto2dlabel.models.detection import DetectionResult
from auto2dlabel.models.obb import OBBResult
from auto2dlabel.schema.annotation import ImageLabel
from auto2dlabel.tests import Path
from auto2dlabel.web import server

client = TestClient(server.app)


# ---------- Fake 模型工厂（零真实权重） ----------

class _FakeDetModel:
    """返回固定 DetectionResult 列表（duck typing detect）。"""

    def detect(self, image_path: str, prompts: list[str],
               confidence_threshold: float = 0.3) -> list[DetectionResult]:
        return [DetectionResult(x=10, y=20, width=30, height=40, label="car", confidence=0.85)]


def _fake_det_factory(name: str | None = None, **kwargs: Any) -> _FakeDetModel:
    return _FakeDetModel()


class _FakeObbModel:
    """返回固定 OBBResult（cx, cy, w, h, angle）——测试 x=cx-w/2 转换与 angle 保留。"""

    def detect_obb(self, image_path: str, prompts: list[str],
                   confidence_threshold: float = 0.3) -> list[OBBResult]:
        return [OBBResult(
            cx=50, cy=60, width=40, height=20, angle=0.7, label="car", confidence=0.9,
        )]


def _fake_obb_factory(model_name: str = "yolo11n-obb.pt", **kwargs: Any) -> _FakeObbModel:
    return _FakeObbModel()


class _FakeClsModel:
    """返回固定 top-K ImageLabel（duck typing classify）。"""

    def classify(self, image_path: str, candidates: list[str],
                 top_k: int = 5) -> list[ImageLabel]:
        return [
            ImageLabel(label=c, score=round(0.9 - 0.3 * i, 6))
            for i, c in enumerate(candidates[:2])
        ]


def _fake_cls_factory(model_name: str = "openai/clip-vit-base-patch32") -> _FakeClsModel:
    return _FakeClsModel()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


# ---------- _bbox_from_dict 字段保真 ----------

def test_bbox_from_dict_preserves_angle_and_track_id() -> None:
    """angle/track_id 重建（OBB/跟踪框经 review-save·export-coco 不丢字段）。"""
    b = server._bbox_from_dict({
        "x": 1, "y": 2, "width": 3, "height": 4, "label": "car",
        "confidence": 0.9, "angle": 0.5, "track_id": 7,
    })
    assert b.angle == 0.5
    assert b.track_id == 7
    assert b.confidence == 0.9


def test_bbox_from_dict_defaults() -> None:
    """旧前端 payload 缺 angle/track_id → 默认 0.0/None，向后兼容。"""
    b = server._bbox_from_dict({"x": 1, "y": 2, "width": 3, "height": 4})
    assert b.angle == 0.0
    assert b.track_id is None
    assert b.confidence == 1.0
    assert not b.edited_by_human  # 缺键 → False（AI 初稿）


def test_bbox_from_dict_edited_by_human() -> None:
    """edited_by_human 标记经 _bbox_from_dict 重建保真。"""
    b = server._bbox_from_dict({
        "x": 1, "y": 2, "width": 3, "height": 4, "label": "car", "edited_by_human": True,
    })
    assert b.edited_by_human is True


def test_bbox_to_dict_edited_by_human_conditional() -> None:
    """Bbox.to_dict 仅 edited_by_human=True 时输出该键（防 JSON 膨胀）。"""
    from auto2dlabel.schema.annotation import Bbox

    plain = Bbox(x=0, y=0, width=1, height=1, label="c")
    assert "edited_by_human" not in plain.to_dict()
    assert "edited_by_human" not in Bbox(
        x=0, y=0, width=1, height=1, label="c", edited_by_human=False,
    ).to_dict()
    assert Bbox(
        x=0, y=0, width=1, height=1, label="c", edited_by_human=True,
    ).to_dict()["edited_by_human"] is True


# ---------- /api/annotate 三路由 ----------

def test_annotate_detection_no_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """detection 路由：bboxes 有值、labels 空数组。"""
    monkeypatch.setattr(server, "create_detection_model", _fake_det_factory)
    resp = client.post("/api/annotate", files={"image": ("a.png", _png_bytes(), "image/png")},
                       data={"instruction": "car", "task_type": "detection", "model": "fake.pt"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["bboxes"]) == 1
    assert data["bboxes"][0]["label"] == "car"
    assert data["labels"] == []
    assert data["summary"]["task_type"] == "detection"


def test_annotate_obb_angle_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """obb 路由：x=cx-w/2、y=cy-h/2、angle 保留。"""
    monkeypatch.setattr("auto2dlabel.models.obb.create_obb_model", _fake_obb_factory)
    resp = client.post("/api/annotate", files={"image": ("a.png", _png_bytes(), "image/png")},
                       data={"instruction": "car", "task_type": "obb", "model": "yolo11n-obb.pt"})
    assert resp.status_code == 200
    data = resp.json()
    b = data["bboxes"][0]
    assert b["x"] == pytest.approx(30)   # 50 - 40/2
    assert b["y"] == pytest.approx(50)   # 60 - 20/2
    assert b["angle"] == pytest.approx(0.7)
    assert data["labels"] == []
    assert data["summary"]["task_type"] == "obb"


def test_annotate_classification_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """classification 路由：labels 进 ann.labels + 响应两个字段，bboxes 空。"""
    monkeypatch.setattr(
        "auto2dlabel.models.classification.create_classification_model", _fake_cls_factory,
    )
    resp = client.post("/api/annotate", files={"image": ("a.png", _png_bytes(), "image/png")},
                       data={"instruction": "cat, dog", "task_type": "classification",
                             "model": "openai/clip-vit-base-patch32"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["bboxes"] == []
    assert data["labels"] == [{"label": "cat", "score": 0.9}, {"label": "dog", "score": 0.6}]
    assert data["summary"]["labels"] == data["labels"]
    assert data["summary"]["task_type"] == "classification"


# ---------- /api/export-coco：angle 保真穿透 ----------

def test_export_coco_bbox_with_angle(monkeypatch: pytest.MonkeyPatch) -> None:
    """前端过滤后重建 COCO：带 angle 的 bbox 正常导出（不丢框不报错）。"""
    resp = client.post("/api/export-coco", json={
        "image_path": "a.png", "image_width": 100, "image_height": 100,
        "bboxes": [{"x": 10, "y": 20, "width": 30, "height": 40, "label": "car",
                    "confidence": 0.9, "angle": 0.5}],
        "masks": [],
    })
    assert resp.status_code == 200
    coco = resp.json()
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_export_coco_ignores_edited_by_human() -> None:
    """/api/export-coco 输出纯净 COCO（edited_by_human 不泄漏到导出）。"""
    resp = client.post("/api/export-coco", json={
        "image_path": "a.png", "image_width": 100, "image_height": 100,
        "bboxes": [{"x": 10, "y": 20, "width": 30, "height": 40, "label": "car",
                    "confidence": 0.9, "edited_by_human": True}],
        "masks": [],
    })
    assert resp.status_code == 200
    assert "edited_by_human" not in resp.json()["annotations"][0]


# ---------- /api/review-save：edited 全量重建 ----------

@pytest.fixture()
def review_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(server, "REVIEW_DIR", tmp_path)
    return tmp_path


def test_review_save_edited_rebuild(review_dir: Path) -> None:
    """edited 优先：修正框全量重建（含 angle/track_id/改标签），deleted 计数=原-新。"""
    queue = review_dir / "demo_review.json"
    queue.write_text(json.dumps({
        "image_path": "demo.png",
        "image_size": [100, 100],
        "annotations": [
            {"x": 1, "y": 2, "width": 3, "height": 4, "label": "car", "confidence": 0.8},
            {"x": 5, "y": 6, "width": 7, "height": 8, "label": "person", "confidence": 0.7},
        ],
    }), encoding="utf-8")

    resp = client.post("/api/review-save", json={
        "queue_file": "demo_review.json",
        "edited": [{"x": 11, "y": 12, "width": 30, "height": 40, "label": "truck",
                    "confidence": 0.9, "angle": 0.5, "track_id": 7}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["kept"] == 1
    assert data["deleted"] == 1

    # 源队列标记 .reviewed，输出含修正后的标签/几何
    assert not queue.exists()
    out = json.loads((review_dir / "demo_reviewed.json").read_text(encoding="utf-8"))
    assert len(out["annotations"]) == 1
    assert out["annotations"][0]["bbox"] == [11.0, 12.0, 30.0, 40.0]


def test_review_save_deleted_indices_backward_compat(review_dir: Path) -> None:
    """无 edited 字段 → 旧 deleted_indices 模式不变（向后兼容回归）。"""
    queue = review_dir / "demo_review.json"
    queue.write_text(json.dumps({
        "image_path": "demo.png",
        "image_size": [100, 100],
        "annotations": [
            {"x": 1, "y": 2, "width": 3, "height": 4, "label": "car", "confidence": 0.8},
            {"x": 5, "y": 6, "width": 7, "height": 8, "label": "person", "confidence": 0.7},
        ],
    }), encoding="utf-8")

    resp = client.post("/api/review-save", json={
        "queue_file": "demo_review.json", "deleted_indices": [1],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["kept"] == 1
    assert data["deleted"] == 1
    out = json.loads((review_dir / "demo_reviewed.json").read_text(encoding="utf-8"))
    assert len(out["annotations"]) == 1
    # 无 issues → 不写 issues 键（向后兼容，防静默改结构）
    assert "issues" not in out


def test_review_save_edited_persists_flag(review_dir: Path) -> None:
    """edited 全量重建时 edited_by_human 标记落盘 *_reviewed.json。"""
    queue = review_dir / "demo_review.json"
    queue.write_text(json.dumps({
        "image_path": "demo.png",
        "image_size": [100, 100],
        "annotations": [
            {"x": 1, "y": 2, "width": 3, "height": 4, "label": "car", "confidence": 0.8},
            {"x": 5, "y": 6, "width": 7, "height": 8, "label": "person", "confidence": 0.7},
        ],
    }), encoding="utf-8")

    resp = client.post("/api/review-save", json={
        "queue_file": "demo_review.json",
        "edited": [
            {"x": 11, "y": 12, "width": 30, "height": 40, "label": "truck",
             "confidence": 0.9, "edited_by_human": True},
            {"x": 5, "y": 6, "width": 7, "height": 8, "label": "person", "confidence": 0.7},
        ],
    })
    assert resp.status_code == 200
    out = json.loads((review_dir / "demo_reviewed.json").read_text(encoding="utf-8"))
    assert out["annotations"][0]["edited_by_human"] is True
    assert "edited_by_human" not in out["annotations"][1]  # 未人工改过的框无该键
