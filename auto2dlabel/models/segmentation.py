"""分割模型抽象层。

封装 SAM2 / FastSAM / Mask R-CNN / SAM3 / torchvision 语义分割等模型，
提供统一接口：generate(image, bboxes) -> List[Mask]。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from auto2dlabel.models import Image, Path, os
from auto2dlabel.schema.annotation import Bbox, Mask

if TYPE_CHECKING:
    # 仅用于类型标注与 cast（字符串前向引用），运行时保持懒加载
    import torch
    from ultralytics import SAM, FastSAM  # type: ignore  # mypy: 顶层动态 __all__ 未显式导出
    from ultralytics.engine.results import Results
    from ultralytics.models.sam import SAM3SemanticPredictor


class SegmentationModel(Protocol):
    """分割模型接口。"""

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        ...


class SAM2Model:
    """SAM2 模型封装（Ultralytics 集成）。

    使用 Ultralytics 的 SAM2 API，支持 box prompt 分割。
    权重自动下载至 auto2dlabel/weights/。
    可用模型: sam2_t.pt / sam2_s.pt / sam2_b.pt / sam2_l.pt
    """

    def __init__(self, device: str | None = None, model_name: str = "sam_b.pt"):
        self._device = device
        self._model_name = model_name
        self._model = None

    def _load(self) -> SAM:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import SAM, settings
        except ImportError:
            raise ImportError("SAM 需要 ultralytics。请运行: pip install -U ultralytics")

        from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
        settings.update({"weights_dir": str(WEIGHTS_DIR)})
        self._model = SAM(self._model_name)
        return self._model

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """对 bbox 区域生成分割 mask。

        每个 bbox 作为 box prompt 发给 SAM2。
        """
        model = self._load()

        if not bboxes:
            return []

        # 将 bbox 转换为 [x1, y1, x2, y2] 格式
        box_prompts = [[b.x, b.y, b.x + b.width, b.y + b.height] for b in bboxes]

        # SAM.__call__ stub 为 Results | Tensor 联合，运行时恒为 list[Results]
        results = cast(
            "list[Results]", model(image_path, bboxes=box_prompts, verbose=False, rect=False)
        )
        r = results[0]

        masks_out = []
        if r.masks is not None:
            for i, bbox in enumerate(bboxes):
                if i < len(r.masks.data):
                    # stub 将 data 标为 ndarray，运行时实为 torch.Tensor
                    mask_np = cast("torch.Tensor", r.masks.data[i]).cpu().numpy()
                    mask_bool = mask_np > 0.5
                    polygon = _mask_to_polygon(mask_bool)
                    masks_out.append(Mask(
                        bbox=bbox, segmentation=polygon,
                        area=float(mask_bool.sum()),
                    ))

        return masks_out


class FastSAMModel:
    """FastSAM 分割模型（Ultralytics，开箱即用）。

    将检测 bbox 与 FastSAM 全图 mask 做 IoU 匹配，
    每个 bbox 取最佳匹配的 mask 作为该实例的分割结果。
    权重自动下载至 auto2dlabel/weights/。

    可用模型: FastSAM-s.pt | FastSAM-x.pt
    """

    def __init__(self, device: str | None = None, model_name: str = "FastSAM-s.pt"):
        self._device = device
        self._model_name = model_name
        self._model = None

    def _load(self) -> FastSAM:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import FastSAM, settings
        except ImportError:
            raise ImportError("ultralytics 未安装。请运行: pip install ultralytics")

        from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
        settings.update({"weights_dir": str(WEIGHTS_DIR)})

        # 优先本地权重
        local_path = WEIGHTS_DIR / self._model_name
        if local_path.exists():
            self._model = FastSAM(str(local_path))
        else:
            self._model = FastSAM(self._model_name)
        return self._model

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """对每个 bbox 匹配最佳 FastSAM mask。

        Args:
            image_path: 图像路径。
            bboxes: 来自检测模型的边界框列表。
            mode: 未使用（保留以匹配接口）。

        Returns:
            Mask 列表，每个对应一个 bbox。
        """
        model = self._load()

        if not bboxes:
            return []

        # FastSAM.__call__ stub 为 Results | Tensor 联合，运行时恒为 list[Results]
        results = cast("list[Results]", model(
            image_path, device=self._device, verbose=False, conf=0.1, iou=0.7,
            rect=False,
        ))
        r = results[0]

        if r.masks is None or len(r.masks.data) == 0:
            return []

        # 获取 FastSAM 的 bbox（原图坐标系）和 polygon（原图坐标系）
        assert r.boxes is not None  # 有 mask 时必带 boxes（stub 标为 Boxes | None）
        # stub 将 xyxy 标为 Tensor | ndarray，运行时实为 torch.Tensor
        sam_boxes = cast("torch.Tensor", r.boxes.xyxy).cpu().numpy()   # (N, 4) — 原图坐标
        sam_polygons = r.masks.xy                 # list of (M, 2) arrays — 原图坐标

        masks_out: list[Mask] = []
        for det_bbox in bboxes:
            x1, y1, x2, y2 = det_bbox.xyxy
            best_iou = 0.0
            best_idx = -1

            for j, sam_box in enumerate(sam_boxes):
                sx1, sy1, sx2, sy2 = sam_box
                ix1, iy1 = max(x1, sx1), max(y1, sy1)
                ix2, iy2 = min(x2, sx2), min(y2, sy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                det_area = (x2 - x1) * (y2 - y1)
                sam_area = (sx2 - sx1) * (sy2 - sy1)
                iou = inter / (det_area + sam_area - inter)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx >= 0 and best_iou > 0.3 and best_idx < len(sam_polygons):
                # 用原图坐标系的 polygon（避免推理分辨率缩放问题）
                pts = sam_polygons[best_idx]  # (M, 2) — 已在原图坐标系
                polygon = [pts.flatten().astype(float).tolist()]
                # 用 bbox 面积近似 mask 面积
                masks_out.append(Mask(
                    bbox=det_bbox,
                    segmentation=polygon,
                    area=det_bbox.area(),
                ))

        return masks_out


# mmdet Cityscapes 8 类（mask_rcnn_r50_fpn_1x_cityscapes 权重下标权威顺序）
CITYSCAPES_THING_NAMES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def build_maskrcnn_cityscapes(device: str | torch.device = "cpu") -> torch.nn.Module:
    """构建与 mmdet Cityscapes 权重对齐的 torchvision Mask R-CNN（v1 结构）。

    - maskrcnn_resnet50_fpn(weights=None, num_classes=9, weights_backbone=None)：
      v1 结构 + 4 层 mask head（torchvision 默认即 4 层，与 mmdet FCNMaskHead 同构）；
      weights_backbone=None 时用 BatchNorm2d（键集含 num_batches_tracked，与 mmdet 一致，
      且免下载 ImageNet 初始化——权重由转换产物整体覆盖）
    - 推理尺度对齐 mmdet cityscapes val 管线（原图 2048×1024 保持 1:1，pad 到 32 整除）
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=9, weights_backbone=None)
    # mmdet test_pipeline: img_scale=(2048, 1024) keep_ratio + Pad(size_divisor=32)
    model.transform = GeneralizedRCNNTransform(
        min_size=1024,
        max_size=2048,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model.to(device)


class MaskRCNNModel:
    """PyTorch Mask R-CNN 分割模型（检测+分割一步完成）。

    - 默认 maskrcnn_resnet50_fpn_v2：COCO 预训练（权重自动下载到 auto2dlabel/weights/）
    - maskrcnn_r50_cityscapes：mmdet cityscapes 域内权重（8 类，解决 30~50px 小目标
      检测域边界），需先运行一次转换脚本生成 weights/maskrcnn_r50_cityscapes.pth
    """

    def __init__(
        self,
        model_name: str = "maskrcnn_resnet50_fpn_v2",
        device: str | None = None,
    ):
        from auto2dlabel.tools.device import get_device
        self._model_name = model_name
        self._device = device or get_device()
        self._model = None

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
        os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))

        if self._model_name == "maskrcnn_r50_cityscapes":
            # cityscapes 域内权重：v1 结构 + 转换产物 strict 加载（校验转换正确性）
            self._model = build_maskrcnn_cityscapes(self._device)
            ckpt_path = WEIGHTS_DIR / "maskrcnn_r50_cityscapes.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"缺少 cityscapes 权重 {ckpt_path}。请先运行: "
                    "python3 -m auto2dlabel.tools.convert_mmdet_cityscapes_maskrcnn"
                )
            import torch
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
        else:
            try:
                from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            except ImportError:
                raise ImportError("torchvision 未安装。请运行: pip install torchvision")

            self._model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """检测 + 分割一步完成（忽略输入的 bboxes，模型自己检测）。

        Args:
            image_path: 图像路径。
            bboxes: 保留参数以匹配接口（模型自己检测，不使用外部 bbox）。
            mode: 未使用。

        Returns:
            Mask 列表。
        """
        model = self._load()

        import torch
        from torchvision.transforms import functional as F

        from auto2dlabel.configs.model_catalog import COCO_CLASSES

        # cityscapes 域内权重输出 8 类（mmdet 下标权威顺序）；COCO 路径维持现状
        class_names = (
            CITYSCAPES_THING_NAMES
            if self._model_name == "maskrcnn_r50_cityscapes"
            else COCO_CLASSES
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(self._device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        return self._parse_output(outputs, class_names)

    def generate_batch(
        self,
        image_paths: list[str],
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[list[Mask]]:
        """批量检测 + 分割（torchvision 原生 list-of-tensors 推理）。

        与 generate 一样忽略输入 bboxes（模型自己检测）；每图解析与 generate
        完全一致（同一 _parse_output）。
        """
        model = self._load()

        import torch
        from torchvision.transforms import functional as F

        from auto2dlabel.configs.model_catalog import COCO_CLASSES

        class_names = (
            CITYSCAPES_THING_NAMES
            if self._model_name == "maskrcnn_r50_cityscapes"
            else COCO_CLASSES
        )

        tensors = [
            F.to_tensor(Image.open(p).convert("RGB")).to(self._device)
            for p in image_paths
        ]
        with torch.no_grad():
            outputs = model(tensors)

        return [self._parse_output(o, class_names) for o in outputs]

    def _parse_output(
        self,
        outputs: dict[str, torch.Tensor],
        class_names: list[str],
    ) -> list[Mask]:
        """torchvision maskrcnn 输出 dict → Mask 列表（单图/批量共用）。"""
        masks_out = []
        for box, label_idx, score, mask_tensor in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"], outputs["masks"]
        ):
            conf = float(score)
            if conf < 0.5:
                continue

            if class_names is CITYSCAPES_THING_NAMES:
                # cityscapes 权重：1-based thing 类索引（全量 500 图 mAP 0.5149
                # 实测口径，行为零变）
                cls_id = int(label_idx) - 1
                if cls_id < 0 or cls_id >= len(class_names):
                    continue
            else:
                # torchvision COCO_V1 权重输出 detectron 91 类 1-based 索引
                # （0=background，10 个占位类）——经映射转 80 类索引。曾直接
                # label-1 索引：前 11 类一致掩盖错位（cat 预测标成 dog），
                # 见 model_catalog.COCO_91_TO_80 注释
                from auto2dlabel.configs.model_catalog import COCO_91_TO_80

                mapped = COCO_91_TO_80.get(int(label_idx))
                if mapped is None:
                    continue
                cls_id = mapped
            label = class_names[cls_id]

            x1, y1, x2, y2 = box.tolist()
            det_bbox = Bbox.from_xyxy(x1, y1, x2, y2, label=label, confidence=conf)

            # mask_tensor: (1, H, W) → binary → polygon
            mask_bool = mask_tensor[0].cpu().numpy() > 0.5
            polygon = _mask_to_polygon(mask_bool)

            masks_out.append(Mask(
                bbox=det_bbox,
                segmentation=polygon,
                area=float(mask_bool.sum()),
            ))

        return masks_out


# VOC↔COCO 别名（torchvision 分割权重为 VOC 21 类，输出统一 COCO 命名）
_VOC_TO_COCO = {
    "aeroplane": "airplane", "motorbike": "motorcycle", "tvmonitor": "tv",
    "pottedplant": "potted plant", "diningtable": "dining table",
}
_COCO_TO_VOC = {v: k for k, v in _VOC_TO_COCO.items()}


class TorchVisionSegModel:
    """torchvision 语义分割模型（FCN / DeepLabV3 / LRASPP，VOC 21 类预训练）。

    全图语义分割（非实例）：out["out"] logits → argmax → 每类一个 mask。
    忽略传入 bbox 坐标（模型全图自检），仅用 bbox.label 作类别过滤；
    输出 label 统一 COCO 命名（aeroplane→airplane 等），与检测/导出管线一致。
    权重自动下载到 auto2dlabel/weights/（通过 TORCH_HOME）。
    """

    def __init__(self, model_name: str = "fcn_resnet50", device: str | None = None):
        from auto2dlabel.tools.device import get_device
        self._model_name = model_name
        self._device = device or get_device()
        self._model: Any = None  # 懒加载：_load() 才实例化 torchvision 模型

    def _load(self) -> Any:
        """加载模型（幂等）。torchvision 构建函数 stub 返回 Any，类型按 Any 处理。"""
        if self._model is not None:
            return self._model
        from auto2dlabel.configs.model_catalog import TORCHVISION_SEG_MODELS, WEIGHTS_DIR
        os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))

        try:
            import torchvision.models.segmentation as tv_seg
        except ImportError:
            raise ImportError("torchvision 未安装。请运行: pip install torchvision")

        if self._model_name not in TORCHVISION_SEG_MODELS:
            raise ValueError(
                f"无法识别的 torchvision 分割模型: '{self._model_name}'。\n"
                f"可用: {', '.join(TORCHVISION_SEG_MODELS)}"
            )
        self._model = getattr(tv_seg, self._model_name)(weights="DEFAULT")
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """全图语义分割（VOC 21 类），每类一个 mask。

        Args:
            image_path: 图像路径。
            bboxes: 忽略坐标，仅用 label 作类别过滤（空 → 全类）。
            mode: 未使用。

        Returns:
            Mask 列表（bbox 由 mask 外接矩形派生）。
        """
        model = self._load()

        import torch
        from torchvision.transforms import functional

        image = Image.open(image_path).convert("RGB")
        image_tensor = functional.to_tensor(image).to(self._device)

        with torch.no_grad():
            output = model([image_tensor])[0]["out"]

        class_mask = _logits_to_class_mask(output[0].cpu().numpy())
        prompts = [b.label for b in bboxes] if bboxes else []
        return _class_mask_to_masks(class_mask, prompts)


def _logits_to_class_mask(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """(C, H, W) logits → (H, W) 类别索引（argmax）。"""
    class_mask: np.ndarray[Any, Any] = logits.argmax(axis=0)
    out: np.ndarray[Any, Any] = class_mask.astype(np.int32)
    return out


def _class_mask_to_masks(class_mask: np.ndarray[Any, Any], prompts: list[str]) -> list[Mask]:
    """argmax 类别图 → Mask 列表。

    - 跳过背景类 0 与空 mask 类
    - VOC 名 → COCO 名作为输出 label（_VOC_TO_COCO）
    - bbox 由 mask 外接矩形派生（cv2.boundingRect），polygon 走 _mask_to_polygon
    - prompts 非空时按 _voc_prompt_match 过滤
    """
    import cv2

    from auto2dlabel.configs.model_catalog import VOC_CLASSES

    masks_out: list[Mask] = []
    for cls_idx in range(1, len(VOC_CLASSES)):
        voc_label = VOC_CLASSES[cls_idx]
        label = _VOC_TO_COCO.get(voc_label, voc_label)
        if prompts and not _voc_prompt_match(label, prompts):
            continue

        mask_bool = class_mask == cls_idx
        if not mask_bool.any():
            continue

        x, y, w, h = cv2.boundingRect(mask_bool.astype(np.uint8))
        det_bbox = Bbox(x=float(x), y=float(y), width=float(w), height=float(h),
                        label=label, confidence=1.0)
        polygon = _mask_to_polygon(mask_bool)
        masks_out.append(Mask(
            bbox=det_bbox, segmentation=polygon,
            area=float(mask_bool.sum()),
        ))
    return masks_out


def _voc_prompt_match(label: str, prompts: list[str]) -> bool:
    """prompt 匹配（含 VOC/COCO 别名等价）。

    label 为 COCO 命名；prompt 可为 COCO 名、VOC 名或子串。
    如 ("airplane", ["aeroplane"])、("tv", ["tvmonitor"]) 均视为命中。
    """
    label_norm = label.strip().lower()
    aliases = {label_norm, _COCO_TO_VOC.get(label, label_norm).lower()}
    for p in prompts:
        p_norm = p.strip().lower()
        if not p_norm:
            continue
        if p_norm in aliases or _VOC_TO_COCO.get(p_norm) in aliases:
            return True
        # 子串双向包含兜底
        if p_norm in label_norm or label_norm in p_norm:
            return True
    return False


def _patch_clip_tokenizer() -> None:
    """兼容 ultralytics SAM3 与 PyPI openai-clip 的 tokenizer 接口差异。

    ultralytics build_sam3 直接实例化 clip.simple_tokenizer.SimpleTokenizer
    并当作可调用对象（tokenizer(texts, context_length=...)）使用，但 PyPI
    openai-clip 的 SimpleTokenizer 未定义 __call__（hasattr 会沿 MRO 误报
    type.__call__，须查 __dict__）。此处补 __call__ 并委托官方 clip.tokenize
    （含 SOS/EOS 包装、list 输入、截断），行为与官方一致。
    """
    try:
        import clip
        from clip.simple_tokenizer import SimpleTokenizer
    except ImportError:
        return  # clip 未安装（SAM3 加载时会自行报错）

    if "__call__" in SimpleTokenizer.__dict__:
        return  # 该 clip 版本已兼容，无需 patch

    def _call(self: Any, texts: str | list[str], context_length: int = 77):
        # self 用 Any：SimpleTokenizer 定义时才 import，无法作注解
        return clip.tokenize(texts, context_length=context_length)

    SimpleTokenizer.__call__ = _call  # type: ignore[method-assign]


class SAM3Model:
    """Meta SAM3 模型封装（Ultralytics 集成）。

    文本 prompt → 检测 + 分割一步完成，开放词汇。
    权重约 3.45 GB，需从 HuggingFace 下载后放入 auto2dlabel/weights/sam3/。

    下载方式:
      1. 自动: python -m auto2dlabel.models.download_sam3
      2. 手动: 访问 https://huggingface.co/facebook/sam3 下载 sam3.pt
    """

    # 下载指引
    DOWNLOAD_HELP = (
        "SAM3 权重下载:\n"
        "  1. 访问 https://huggingface.co/facebook/sam3 申请模型访问权限\n"
        "  2. 批准后下载 sam3.pt（约 3.45 GB）\n"
        "  3. 将 sam3.pt 放入 weights/sam3/ 目录\n\n"
        "或运行自动下载脚本:\n"
        "  python -m auto2dlabel.models.download_sam3"
    )

    MIN_SIZE_BYTES = 2 * 1024 ** 3  # SAM3 checkpoint 应 >= 2 GB

    def __init__(self, device: str | None = None):
        self._device = device
        self._predictor = None

    def _load(self) -> SAM3SemanticPredictor:
        if self._predictor is not None:
            return self._predictor
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            raise ImportError(
                "SAM3 需要 ultralytics>=8.3.237。请运行: pip install -U ultralytics"
            )

        from auto2dlabel.configs.model_catalog import WEIGHTS_DIR
        os.environ.setdefault("ULTRALYTICS_CACHE", str(WEIGHTS_DIR))

        # 查找权重文件
        weight_file = self._find_checkpoint(WEIGHTS_DIR)

        # 验证文件完整性
        self._validate_checkpoint(weight_file)

        # 加载前先修 clip tokenizer 接口差异（openai-clip 旧版不可调用）
        _patch_clip_tokenizer()

        # 用 .pt 文件初始化（config.json 等自动从同目录读取）
        self._predictor = SAM3SemanticPredictor(overrides={
            "model": str(weight_file), "save": False, "project": "runs",
        })
        return self._predictor

    def _find_checkpoint(self, weights_dir) -> Path:
        """查找 SAM3 checkpoint 文件。

        按优先级搜索:
          1. weights/sam3/sam3.pt
          2. weights/sam3.pt

        Returns:
            找到的 checkpoint 路径。

        Raises:
            FileNotFoundError: 未找到权重文件。
        """
        model_dir = weights_dir / "sam3"
        candidates = [
            model_dir / "sam3.pt",
            weights_dir / "sam3.pt",
        ]
        for path in candidates:
            if path.is_file():
                return path

        raise FileNotFoundError(
            "SAM3 权重文件未找到。尝试过:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + f"\n\n{self.DOWNLOAD_HELP}"
        )

    def _validate_checkpoint(self, path: Path) -> None:
        """验证 checkpoint 文件完整性，提前给出明确错误信息。

        检查:
          1. 文件大小是否合理（>= 2 GB）
          2. 若是 ZIP 格式（PyTorch .pt），检查是否缺少 EOCD（下载中断）

        Raises:
            ValueError: checkpoint 异常小或损坏。
        """
        path = Path(path)
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)

        if size_bytes < self.MIN_SIZE_BYTES:
            raise ValueError(
                f"SAM3 权重文件过小 ({size_gb:.2f} GB)，完整模型约 3.45 GB。\n"
                f"文件可能下载不完整。请删除后重新下载:\n"
                f"  rm {path}\n"
                f"\n{self.DOWNLOAD_HELP}"
            )

        # 检查 ZIP 完整性：PyTorch .pt 文件本质是 ZIP，EOCD 缺失 = 下载中断
        with open(path, "rb") as f:
            header = f.read(4)

        if header[:2] == b"PK":  # ZIP/PyTorch checkpoint 格式
            with open(path, "rb") as f:
                if size_bytes > 65536:
                    f.seek(-65536, 2)  # 最后 64KB
                    tail = f.read()
                else:
                    f.seek(0)
                    tail = f.read()

            has_eocd = (
                b"PK\x05\x06" in tail  # 标准 EOCD
                or b"PK\x06\x06" in tail  # ZIP64 EOCD locator
                or b"PK\x06\x07" in tail  # ZIP64 EOCD
            )
            if not has_eocd:
                raise ValueError(
                    f"SAM3 权重文件已损坏（ZIP 缺少中央目录记录）。\n"
                    f"文件大小: {size_gb:.2f} GB — 下载可能在中途中断。\n"
                    f"\n请删除此文件后重新下载:\n"
                    f"  rm {path}\n"
                    f"\n{self.DOWNLOAD_HELP}"
                )

    def generate(
        self,
        image_path: str,
        bboxes: list[Bbox],
        mode: str = "box",
    ) -> list[Mask]:
        """文本 prompt → 检测 + 分割一步完成。

        使用 bboxes 的 label 作为 SAM3 的文本 prompt。
        模型自己检测并分割，不依赖外部 bbox 坐标。
        """
        predictor = self._load()


        # 从 bboxes 提取类别作为文本 prompt（空列表时只能全类检测）
        prompts = list(set(b.label for b in bboxes)) if bboxes else None
        if prompts is None:
            return []  # 无 prompt 时不运行，避免全类检测

        # SAM3SemanticPredictor.__call__ stub 无返回注解，运行时恒为 list[Results]
        results = cast("list[Results]", predictor(image_path, text=prompts))

        masks_out = []
        if results:
            r = results[0]
            # 有 mask 时必带 boxes（原代码 boxes 为 None 会 AttributeError）
            if r.masks is not None and r.boxes is not None:
                for mask_data, box_data, cls_id in zip(
                    r.masks.data, r.boxes.xyxy, r.boxes.cls
                ):
                    # stub 将 data 标为 ndarray，运行时实为 torch.Tensor
                    mask_np = cast("torch.Tensor", mask_data).cpu().numpy()
                    mask_bool = mask_np > 0.5
                    polygon = _mask_to_polygon(mask_bool)

                    x1, y1, x2, y2 = box_data.tolist()
                    label = prompts[int(cls_id)] if prompts and int(cls_id) < len(prompts) else str(int(cls_id))
                    det_bbox = Bbox.from_xyxy(x1, y1, x2, y2, label=label, confidence=1.0)

                    masks_out.append(Mask(
                        bbox=det_bbox, segmentation=polygon,
                        area=float(mask_bool.sum()),
                    ))

        return masks_out


def create_segmentation_model(name: str = "sam2_l.pt", **kwargs):
    """工厂函数：根据名称创建分割模型。

    Args:
        name: 模型名或类别:
            - "fastsam" / "FastSAM-s.pt" / "FastSAM-x.pt" → FastSAMModel
            - "sam2" / "sam_b.pt" / "sam2_t.pt" / ... → SAM2Model
            - "maskrcnn" / "maskrcnn_resnet50_fpn_v2" → MaskRCNNModel
            - "mask2former" / "mask2former_r50_8xb2-lsj-50e_coco" → MMDetMask2FormerModel
            - "sam3" / "sam3.pt" → SAM3Model
            - "fcn_resnet50" / "deeplabv3_*" / "lraspp_*" → TorchVisionSegModel
        **kwargs: 透传参数（如 device, model_name）。
    """
    from auto2dlabel.configs.model_catalog import TORCHVISION_SEG_MODELS

    name_lower = name.lower()

    if name_lower in TORCHVISION_SEG_MODELS or name_lower.startswith(
        ("fcn_", "deeplabv3_", "lraspp_")
    ):
        kwargs.setdefault("model_name", name)
        return TorchVisionSegModel(**kwargs)
    elif "maskrcnn" in name_lower:
        kwargs.setdefault("model_name", name)
        return MaskRCNNModel(**kwargs)
    elif "mask2former" in name_lower:
        # mmdet Mask2Former 实例分割质量档（v0.6 Phase 3a）：自带检测，
        # generate 忽略 bboxes（与 MaskRCNNModel 同款）；零加载 + ImportError 守卫
        from auto2dlabel.models.mmdet_engines import MMDetMask2FormerModel

        kwargs.setdefault(
            "model_name",
            MMDetMask2FormerModel.DEFAULT_NAME if name_lower == "mask2former" else name,
        )
        return MMDetMask2FormerModel(**kwargs)
    elif "sam3" in name_lower:
        return SAM3Model(**kwargs)
    elif name_lower.startswith("sam2") or any(
        name.startswith(p) for p in ["sam_t", "sam_s", "sam_b", "sam_l", "sam2_t", "sam2_s", "sam2_b", "sam2_l", "sam2.1"]
    ):
        kwargs.setdefault("model_name", name if name.endswith(".pt") else "sam_b.pt")
        return SAM2Model(**kwargs)
    elif name.endswith(".pt") and name.lower().startswith("fastsam"):
        kwargs.setdefault("model_name", name)
        return FastSAMModel(**kwargs)
    else:
        # 默认 FastSAM
        return FastSAMModel(**kwargs)


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """将二进制 mask 转换为 COCO polygon 格式。

    使用 OpenCV 轮廓检测，提取外轮廓。

    Args:
        mask: 二进制 mask 数组 (H, W)，dtype=bool。

    Returns:
        COCO 格式 polygon: [[x1,y1, x2,y2, ...]]。
    """
    import cv2

    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # 展平为 [x1, y1, x2, y2, ...]
        flat = contour.flatten().astype(float).tolist()
        polygons.append(flat)

    return polygons
