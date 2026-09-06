"""create_detection_model 未知模型名兜底回归测试（2026-09-02 实测发现）。

case-040「用不存在的模型 not_a_model.pt」曾直通 ultralytics 下载失败
（FileNotFoundError traceback）——修复后本地不存在的 .pt 名回退 DEFAULT_MODEL。
零真实权重铁律：UltralyticsModel 构造零加载（_load 懒加载）。
"""

from __future__ import annotations

from typing import cast

from auto2dlabel.models.detection import UltralyticsModel, create_detection_model
from auto2dlabel.schema.task_plan import DEFAULT_MODEL


def _as_ultralytics(model: object) -> UltralyticsModel:
    """工厂返回 DetectionModel Protocol，测试需访问实现类私有字段 → cast。"""
    return cast(UltralyticsModel, model)


def test_unknown_pt_falls_back_to_default_model() -> None:
    """不存在的 .pt 名 → 回退 DEFAULT_MODEL（不抛 FileNotFoundError）。"""
    model = _as_ultralytics(create_detection_model("not_a_model.pt"))
    assert model._model_name == DEFAULT_MODEL
    assert model._model is None  # 零加载


def test_known_pt_name_unchanged() -> None:
    """已预置权重名原样通过（不误回退）。"""
    model = _as_ultralytics(create_detection_model("yolo26x.pt"))
    assert model._model_name == "yolo26x.pt"


def test_missing_absolute_path_falls_back() -> None:
    """完整路径 .pt 不存在 → 同样回退（原行为是 ultralytics 下载失败）。"""
    model = _as_ultralytics(create_detection_model("/nonexistent/dir/foo.pt"))
    assert model._model_name == DEFAULT_MODEL
