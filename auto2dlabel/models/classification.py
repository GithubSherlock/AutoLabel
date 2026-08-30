"""图像分类模型抽象层。

封装 CLIP / SigLIP 零样本分类（HuggingFace transformers 懒加载）与
torchvision ImageNet1K 监督分类，提供统一接口：
classify(image_path, candidates, top_k) -> list[ImageLabel]。
"""

from __future__ import annotations

from typing import Any, Protocol, cast, get_args

from auto2dlabel.configs.model_catalog import TORCHVISION_CLS_MODELS, WEIGHTS_DIR
from auto2dlabel.models import Image, os
from auto2dlabel.models.detection import _match_prompt
from auto2dlabel.schema.annotation import ImageLabel
from auto2dlabel.tools.device import get_device


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


def _topk_labels(
    scores: list[float],
    class_names: list[str],
    top_k: int,
    candidates: list[str] | None = None,
) -> list[ImageLabel]:
    """ImageNet 全类分数 → top-K ImageLabel（纯函数，免 torch 可单测）。

    - candidates 非空：按 detection._match_prompt 子串过滤候选类，再取 top-K
    - candidates 为空/None：纯 top-K ImageNet 类名
    - 命中不足 top_k 时返回全部命中（不填零分占位）
    """
    if top_k <= 0:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    picked: list[tuple[str, float]] = []
    for i in ranked:
        label = class_names[i] if i < len(class_names) else str(i)
        if candidates and not _match_prompt(label, candidates):
            continue
        picked.append((label, float(scores[i])))
        if len(picked) >= top_k:
            break
    return [ImageLabel(label=lab, score=sc) for lab, sc in picked]


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

    def classify_batch(
        self,
        image_paths: list[str],
        candidates: list[str],
        top_k: int = 5,
    ) -> list[list[ImageLabel]]:
        """多图批量分类：processor 一次编码全部图像，logits_per_image 逐行 top-K。"""
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor
        images = [Image.open(p).convert("RGB") for p in image_paths]

        import torch

        inputs = processor(text=candidates, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)
        return [_to_labels(candidates, probs[i], top_k) for i in range(len(image_paths))]


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

    def classify_batch(
        self,
        image_paths: list[str],
        candidates: list[str],
        top_k: int = 5,
    ) -> list[list[ImageLabel]]:
        """多图批量分类：processor 一次编码全部图像，logits_per_image 逐行 top-K。"""
        model = self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor
        images = [Image.open(p).convert("RGB") for p in image_paths]

        import torch

        inputs = processor(text=candidates, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)
        return [_to_labels(candidates, probs[i], top_k) for i in range(len(image_paths))]


class TorchVisionClassifier:
    """torchvision 分类模型（ImageNet1K 监督，14 款：高性能档 + ResNet/ResNeXt）。

    与 CLIP/SigLIP 零样本语义不同：输出 ImageNet1K 1000 类分布，
    candidates 非空时按 _match_prompt 子串过滤候选类，空候选则纯 top-K。
    权重经 TORCH_HOME 自动下载到 auto2dlabel/weights/（TORCH_HOME 已统一）。
    """

    def __init__(
        self,
        model_name: str = "convnext_large",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None            # 懒加载：_load() 才实例化 torchvision 模型
        self._transform: Any = None        # weights.transforms() 预处理管道（_load 时构建）
        # weights.meta["categories"]：1000 类名，import 即得不下载
        self._class_names: list[str] = []

    def _load(self) -> Any:
        """加载模型与预处理管道（幂等）。torchvision 懒加载。"""
        if self._model is not None:
            return self._model
        os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))
        try:
            import torchvision.models as tv_cls
        except ImportError:
            raise ImportError("torchvision 未安装，请运行: pip install torchvision")

        builder: Any = getattr(tv_cls, self._model_name, None)
        if builder is None:
            raise ValueError(
                f"无法识别的 torchvision 分类模型: '{self._model_name}'。\n"
                f"可用: {', '.join(TORCHVISION_CLS_MODELS)}"
            )
        # 0.19+ builder 统一 `__annotations__["weights"]: Optional[Xxx_Weights]`（0.28 实测，
        # builder 函数本身无 .weights 属性，枚举只能从注解泛型解析）
        weights_type = builder.__annotations__.get("weights")
        if weights_type is None:
            raise ValueError(
                f"无法解析 {self._model_name} 的权重枚举（需 torchvision>=0.19）"
            )
        enum_cls = get_args(weights_type)[0]
        w = enum_cls.DEFAULT
        self._transform = w.transforms()          # 类级属性，不触发下载
        self._class_names = list(w.meta["categories"])
        self._model = builder(weights="DEFAULT")  # 此处才可能下载权重
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def classify(
        self,
        image_path: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[ImageLabel]:
        model = self._load()
        transform = self._transform
        assert transform is not None  # _load 已初始化

        import torch

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = model(input_tensor)[0]
        scores = logits.softmax(dim=-1).detach().cpu().tolist()
        return _topk_labels(scores, self._class_names, top_k, candidates)

    def classify_batch(
        self,
        image_paths: list[str],
        candidates: list[str],
        top_k: int = 5,
    ) -> list[list[ImageLabel]]:
        """多图批量分类：torchvision 原生张量批（transform → stack → 一次前向）。"""
        model = self._load()
        transform = self._transform
        assert transform is not None  # _load 已初始化

        import torch

        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in image_paths
        ]).to(self._device)
        with torch.no_grad():
            logits = model(batch)
        scores = logits.softmax(dim=-1).detach().cpu().tolist()
        return [_topk_labels(s, self._class_names, top_k, candidates) for s in scores]


class ClipCropScorer:
    """CLIP 裁剪图打分器 —— 指代约束属性过滤的真实现（v0.4 3a）。

    输入 PIL 裁剪图列表（tools/constraints 逐框裁剪产物，不落盘）+
    文本候选，返回概率矩阵 scores[i][j]（候选原序，softmax 归一化）。
    与 ClipModel 共享 HF_HOME 缓存与懒加载模式；单图协议（classify/
    classify_batch 签名冻结）不受影响——约束层走独立接口。
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

    def score_crops(
        self,
        images: list[Any],
        candidates: list[str],
    ) -> list[list[float]]:
        """裁剪图 × 候选 → 概率矩阵（候选原序）。空输入返回空列表。"""
        if not images or not candidates:
            return []
        self._load()
        processor = self._processor
        assert processor is not None  # _load 已初始化 processor

        import torch

        rgb = [im.convert("RGB") for im in images]
        inputs = processor(text=candidates, images=rgb, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        return cast(
            list[list[float]],
            outputs.logits_per_image.softmax(dim=-1).detach().cpu().tolist(),
        )


def create_classification_model(
    model_name: str = "openai/clip-vit-base-patch32",
) -> ClassificationModel:
    """工厂函数：按模型名前缀创建分类模型（siglip / clip / torchvision）。"""
    lower = model_name.lower()
    if "siglip" in lower:
        return SigLIPModel(model_name=model_name)
    if "clip" in lower:
        return ClipModel(model_name=model_name)
    if lower in TORCHVISION_CLS_MODELS or lower.startswith(
        ("convnext_", "maxvit_", "swin_", "efficientnet_", "vit_", "resnet", "resnext")
    ):
        return TorchVisionClassifier(model_name=model_name)
    from auto2dlabel.configs.model_catalog import CLASSIFICATION_MODELS

    all_models = CLASSIFICATION_MODELS + TORCHVISION_CLS_MODELS
    raise ValueError(
        f"无法识别的分类模型: '{model_name}'。\n"
        f"可用: {', '.join(all_models)}"
    )
