"""图像分类模型抽象层。

封装 CLIP / SigLIP 零样本分类（HuggingFace transformers 懒加载），
提供统一接口：classify(image_path, candidates, top_k) -> list[ImageLabel]。
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from PIL import Image

from auto2dlabel.models.model_catalog import WEIGHTS_DIR
from auto2dlabel.schema.annotation import ImageLabel


class ClassificationModel(Protocol):
    """分类模型接口（Protocol，允许 duck typing）。"""

    def classify(
        self,
        image_path: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[ImageLabel]:
        ...


def _to_labels(candidates: list[str], probs: Any, top_k: int) -> list[ImageLabel]:
    """按概率降序取 top-K，返回 ImageLabel 列表（纯函数，便于单测）。"""
    from auto2dlabel.schema.annotation import ImageLabel

    scores = probs.detach().cpu().tolist()
    ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
    return [ImageLabel(label=c, score=float(s)) for c, s in ranked[:top_k]]


class ClipModel:
    """CLIP 零样本分类（openai/clip-vit-base-patch32）。

    图像-文本余弦相似度 → 候选标签 softmax 概率。
    权重下载到 auto2dlabel/weights/hf/（HF_HOME 统一管理）。
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None  # transformers 为可选依赖，类型按 Any 处理
        self._processor: Any = None

    def _load(self) -> Any:
        """加载模型与处理器（幂等）。transformers 为可选依赖，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        try:
            from transformers import CLIPModel, CLIPProcessor  # pyright: ignore[reportMissingImports]  # isort: skip
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        self._model = CLIPModel.from_pretrained(self._model_name)
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model.to(self._device)
        return self._model

    def classify(
        self,
        image_path: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[ImageLabel]:
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor
        image = Image.open(image_path).convert("RGB")

        import torch

        inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        return _to_labels(candidates, probs, top_k)


class SigLIPModel:
    """SigLIP 零样本分类（google/siglip-base-patch16-224）。

    原生 sigmoid logits，多候选时按 softmax 归一化排序（与 CLIP 语义一致）。
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None  # transformers 为可选依赖，类型按 Any 处理
        self._processor: Any = None

    def _load(self) -> Any:
        """加载模型与处理器（幂等）。transformers 为可选依赖，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / "hf"))
        try:
            from transformers import AutoProcessor, SiglipModel  # pyright: ignore[reportMissingImports]  # isort: skip
        except ImportError:
            raise ImportError("transformers 未安装，请运行: pip install transformers")

        self._model = SiglipModel.from_pretrained(self._model_name)
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model.to(self._device)
        return self._model

    def classify(
        self,
        image_path: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[ImageLabel]:
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor
        image = Image.open(image_path).convert("RGB")

        import torch

        inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        return _to_labels(candidates, probs, top_k)


def create_classification_model(
    model_name: str = "openai/clip-vit-base-patch32",
) -> ClassificationModel:
    """工厂函数：按模型名前缀创建分类模型（clip / siglip）。"""
    lower = model_name.lower()
    if "siglip" in lower:
        return SigLIPModel(model_name=model_name)
    if "clip" in lower:
        return ClipModel(model_name=model_name)
    from auto2dlabel.models.model_catalog import CLASSIFICATION_MODELS

    raise ValueError(
        f"无法识别的分类模型: '{model_name}'。\n"
        f"可用: {', '.join(CLASSIFICATION_MODELS)}"
    )


from auto2dlabel.tools.device import get_device  # noqa: E402
