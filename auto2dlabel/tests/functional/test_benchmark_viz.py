"""Benchmark 可视化（benchmarks/viz.py）测试。

全 mock 数据（numpy 造图 + 构造预测 dict），不依赖真实数据集/模型：
- VIS_ROOT 定位（项目同级 Visualization/）
- visualize_predictions 分派（bbox / quad / mask / 分类文本条）
- visualize_dataset 镜像路径 + 断点续跑（已存在跳过）
- _mask_polygons bool mask → 轮廓
"""

from __future__ import annotations

import cv2
import pytest

from auto2dlabel.benchmarks import viz
from auto2dlabel.benchmarks.viz import (
    VIS_ROOT,
    _mask_polygons,
    visualize_dataset,
    visualize_predictions,
)
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import Path, np


def _write_dummy_image(path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    """写一张纯黑测试图，返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((size[1], size[0], 3), dtype=np.uint8))
    return path


def _count_colored(out_path: Path) -> int:
    """产物中非零像素数（渲染前后像素 diff 的证据）。"""
    img = cv2.imread(str(out_path))
    assert img is not None
    return int(np.count_nonzero(img))


def test_vis_root_at_project_sibling(tmp_path: Path) -> None:
    """VIS_ROOT = 项目根目录同级的 Visualization/。"""
    assert VIS_ROOT.name == "Visualization"
    assert VIS_ROOT.parent == viz.PROJECT_ROOT.parent


def test_visualize_predictions_bbox(tmp_path: Path) -> None:
    """bbox 预测 → 水平框渲染（有彩色像素）。"""
    img_path = _write_dummy_image(tmp_path / "in" / "a.png")
    out_path = tmp_path / "out" / "a.png"
    preds = [{"bbox": [10, 20, 60, 80], "name": "car", "conf": 0.9}]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_bbox_object(tmp_path: Path) -> None:
    """bbox 预测为 Bbox 对象（benchmark 检测脚本结构）也走同一分派。"""
    img_path = _write_dummy_image(tmp_path / "in" / "b.png")
    out_path = tmp_path / "out" / "b.png"
    preds = [{
        "bbox": Bbox(x=5, y=5, width=50, height=40, label="person", confidence=0.8),
        "name": "person", "conf": 0.8,
    }]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_quad(tmp_path: Path) -> None:
    """quad 预测（OBB 旋转框 8 点）→ 折线渲染。"""
    img_path = _write_dummy_image(tmp_path / "in" / "c.png")
    out_path = tmp_path / "out" / "c.png"
    preds = [{"quad": [10, 10, 80, 10, 80, 60, 10, 60], "name": "ship", "conf": 0.7}]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_mask(tmp_path: Path) -> None:
    """mask 预测（bool ndarray）→ 轮廓叠加渲染。"""
    img_path = _write_dummy_image(tmp_path / "in" / "d.png")
    out_path = tmp_path / "out" / "d.png"
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True
    preds = [{"mask": mask, "name": "car", "conf": 0.85}]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_polygon(tmp_path: Path) -> None:
    """polygon 预测（压缩存储，如 d2sa box-prompted SAM2 输出）→ 免解码填充渲染。"""
    img_path = _write_dummy_image(tmp_path / "in" / "f.png")
    out_path = tmp_path / "out" / "f.png"
    preds = [{
        "polygon": [[30.0, 20.0, 70.0, 20.0, 70.0, 80.0, 30.0, 80.0]],
        "bbox": Bbox(x=30, y=20, width=40, height=60, label="bottle", confidence=0.9),
        "name": "bottle", "conf": 0.9,
    }]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_polygon_xywh_bbox(tmp_path: Path) -> None:
    """polygon + [x,y,w,h] bbox（GT bbox 形式）→ 同样渲染。"""
    img_path = _write_dummy_image(tmp_path / "in" / "g.png")
    out_path = tmp_path / "out" / "g.png"
    preds = [{
        "polygon": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
        "bbox": [10, 10, 40, 40],
        "name": "can", "conf": 0.8,
    }]

    visualize_predictions(img_path, preds, out_path)

    assert out_path.exists()
    assert _count_colored(out_path) > 0


def test_visualize_predictions_classification_strip(tmp_path: Path) -> None:
    """gt_label 非空 → 分类文本条（顶部黑带半透明叠加，像素有变化）。"""
    img_path = _write_dummy_image(tmp_path / "in" / "e.png")
    out_path = tmp_path / "out" / "e.png"
    preds = [
        {"label": "cat", "score": 0.95},
        {"label": "dog", "score": 0.03},
    ]

    visualize_predictions(img_path, preds, out_path, gt_label="cat")

    assert out_path.exists()
    # 顶部黑带 + 文字必然改变原图像素
    assert _count_colored(out_path) > 0


def test_visualize_predictions_missing_image(tmp_path: Path) -> None:
    """原图缺失 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        visualize_predictions(
            tmp_path / "nope.png", [], tmp_path / "out" / "x.png",
        )


def test_mask_polygons_square() -> None:
    """方形 bool mask → 1 个闭合轮廓（≥4 点，浮点坐标）。"""
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True

    polys = _mask_polygons(mask)

    assert len(polys) == 1
    assert len(polys[0]) >= 8  # 至少 4 个点 × 2 坐标
    assert all(isinstance(c, float) for c in polys[0])


def test_mask_polygons_empty() -> None:
    """全 False mask → 无轮廓。"""
    assert _mask_polygons(np.zeros((20, 20), dtype=bool)) == []


def test_visualize_dataset_mirror_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """输出镜像原相对路径：<VIS_ROOT>/<数据集名>/<相对名>。"""
    monkeypatch.setattr(viz, "VIS_ROOT", tmp_path / "Visualization")
    img_path = _write_dummy_image(tmp_path / "images" / "sub" / "1.png")
    gt = {"1": {"file_name": "sub/1.png"}}
    predictions: dict[str, list[dict[str, object]]] = {
        "1": [{"bbox": [5, 5, 40, 40], "name": "car", "conf": 0.9}],
    }

    rendered, skipped = visualize_dataset(
        "demo", gt, predictions,
        image_path_fn=lambda img_id, info: img_path,
    )

    assert (rendered, skipped) == (1, 0)
    assert (tmp_path / "Visualization" / "demo" / "sub" / "1.png").exists()


def test_visualize_dataset_skip_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """产物已存在 → 跳过（断点续跑）。"""
    monkeypatch.setattr(viz, "VIS_ROOT", tmp_path / "Visualization")
    img_path = _write_dummy_image(tmp_path / "images" / "1.png")
    out_path = tmp_path / "Visualization" / "demo" / "1.png"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"placeholder")

    gt = {"1": {"file_name": "1.png"}}
    rendered, skipped = visualize_dataset(
        "demo", gt, {},
        image_path_fn=lambda img_id, info: img_path,
    )

    assert (rendered, skipped) == (0, 1)
    assert out_path.read_bytes() == b"placeholder"  # 不被覆盖


def test_visualize_dataset_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原图缺失 → 跳过不报错。"""
    monkeypatch.setattr(viz, "VIS_ROOT", tmp_path / "Visualization")
    gt = {"9": {"file_name": "9.png"}}

    rendered, skipped = visualize_dataset(
        "demo", gt, {},
        image_path_fn=lambda img_id, info: tmp_path / "images" / "9.png",
    )

    assert (rendered, skipped) == (0, 1)
