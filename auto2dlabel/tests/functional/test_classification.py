"""分类任务（CLIP/SigLIP）测试 —— 零模型加载、不 import transformers。

覆盖：模型名分发、_to_labels 排序、ImageLabel 序列化 roundtrip、
export_cls JSON 内容、ExportTool format="cls" 标签透传。
"""

from __future__ import annotations

import json
from pathlib import Path

from auto2dlabel.agent.state import AgentState
from auto2dlabel.export.cls import export_cls
from auto2dlabel.models.classification import (
    ClassificationModel,
    ClipModel,
    SigLIPModel,
    _to_labels,
    create_classification_model,
)
from auto2dlabel.schema.annotation import Annotation, ImageLabel
from auto2dlabel.tools.export import ExportTool


class _FakeProbs:
    """模拟 torch tensor 的 .detach().cpu().tolist() 链。"""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def detach(self) -> _FakeProbs:
        return self

    def cpu(self) -> _FakeProbs:
        return self

    def tolist(self) -> list[float]:
        return self._values


class FakeClassificationModel:
    """实现 ClassificationModel Protocol 的假模型（固定分数）。"""

    model = "fake-cls"

    def classify(
        self,
        image_path: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[ImageLabel]:
        return [ImageLabel(label=c, score=1.0 / (i + 2)) for i, c in enumerate(candidates)][:top_k]


def test_create_classification_model_dispatch() -> None:
    """按名称前缀分发：clip / siglip / 未知名报错。"""
    assert isinstance(create_classification_model("openai/clip-vit-base-patch32"), ClipModel)
    assert isinstance(create_classification_model("google/siglip-base-patch16-224"), SigLIPModel)
    # 大小写不敏感
    assert isinstance(create_classification_model("SIGLIP-x"), SigLIPModel)

    try:
        create_classification_model("resnet50")
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "无法识别的分类模型" in str(e)


def test_to_labels_sorts_desc_and_caps_top_k() -> None:
    """按概率降序取 top-K，分数为 float。"""
    labels = _to_labels(["cat", "dog", "bird"], _FakeProbs([0.1, 0.7, 0.2]), top_k=2)
    assert [(lab.label, lab.score) for lab in labels] == [("dog", 0.7), ("bird", 0.2)]


def test_annotation_labels_roundtrip_via_state() -> None:
    """labels 写入 Annotation.to_dict，经 AgentState.from_dict 完整还原。"""
    ann = Annotation(image_path="/data/a.jpg", image_size=(100, 50))
    ann.labels = [ImageLabel(label="cat", score=0.8), ImageLabel(label="dog", score=0.15)]
    ann.metadata["model"] = "openai/clip-vit-base-patch32"

    restored = AgentState.from_dict(
        {"annotations": [ann.to_dict()], "image_path": "/data/a.jpg"}
    ).annotations[0]
    assert [(lab.label, lab.score) for lab in restored.labels] == [("cat", 0.8), ("dog", 0.15)]
    assert restored.has_annotations  # labels 也计入


def test_export_cls_writes_json(tmp_path: Path) -> None:
    """每图一个 JSON：image_path/width/height/model/labels。"""
    ann = Annotation(image_path="/data/img.png", image_size=(64, 32))
    ann.labels = [ImageLabel(label="cat", score=0.9)]
    ann.metadata["model"] = "google/siglip-base-patch16-224"

    export_cls([ann], tmp_path)

    data = json.loads((tmp_path / "img.json").read_text(encoding="utf-8"))
    assert data["image_path"] == "/data/img.png"
    assert data["width"] == 64 and data["height"] == 32
    assert data["model"] == "google/siglip-base-patch16-224"
    assert data["labels"] == [{"label": "cat", "score": 0.9}]


def test_export_tool_cls_preserves_labels(tmp_path: Path) -> None:
    """ExportTool.forward(format="cls") 从 dict 重建 labels 并导出（目录型格式）。"""
    tool = ExportTool()
    tool.forward(
        annotations=[{
            "image_path": "a.jpg",
            "image_size": (10, 10),
            "labels": [{"label": "cat", "score": 0.7}],
            "metadata": {"model": "openai/clip-vit-base-patch32"},
        }],
        output_path=str(tmp_path),
        format="cls",
    )
    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert data["labels"] == [{"label": "cat", "score": 0.7}]


def test_fake_model_full_flow(tmp_path: Path) -> None:
    """FakeClassificationModel 全流程：classify → Annotation → export_cls。"""
    model: ClassificationModel = FakeClassificationModel()
    ann = Annotation(image_path="cat.jpg", image_size=(8, 8))
    ann.labels = model.classify("cat.jpg", ["cat", "dog"], top_k=1)
    assert [lab.label for lab in ann.labels] == ["cat"]

    export_cls([ann], tmp_path)
    data = json.loads((tmp_path / "cat.json").read_text(encoding="utf-8"))
    assert data["labels"][0]["score"] == 0.5
    assert isinstance(data["labels"][0]["label"], str)


def test_clip_model_constructor_is_lazy() -> None:
    """构造 ClipModel 不触发加载（懒加载红线），权重仅 _load 时下载。"""
    model = ClipModel(model_name="openai/clip-vit-base-patch32")
    assert model._model_name == "openai/clip-vit-base-patch32"
    assert model._model is None
