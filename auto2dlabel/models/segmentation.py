"""分割模型抽象层。

封装 SAM2 / FastSAM 等分割模型，
提供统一接口：generate(image, bboxes) -> List[Mask]。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from PIL import Image

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

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR
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
            "list[Results]", model(image_path, bboxes=box_prompts, verbose=False)
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

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR
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


class MaskRCNNModel:
    """PyTorch Mask R-CNN 分割模型。

    torchvision 内置，COCO 预训练，检测+分割一步完成。
    权重自动下载到 auto2dlabel/weights/（通过 TORCH_HOME）。
    """

    def __init__(self, device: str | None = None):
        from auto2dlabel.tools.device import get_device
        self._device = device or get_device()
        self._model = None

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        import os
        from auto2dlabel.models.model_catalog import WEIGHTS_DIR
        os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))

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

        from auto2dlabel.models.model_catalog import COCO_CLASSES

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(self._device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        masks_out = []
        for box, label_idx, score, mask_tensor in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"], outputs["masks"]
        ):
            conf = float(score)
            if conf < 0.5:
                continue

            cls_id = int(label_idx) - 1  # 1-based → 0-based
            if cls_id < 0 or cls_id >= len(COCO_CLASSES):
                continue
            label = COCO_CLASSES[cls_id]

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

        from auto2dlabel.models.model_catalog import WEIGHTS_DIR
        import os
        os.environ.setdefault("ULTRALYTICS_CACHE", str(WEIGHTS_DIR))

        # 查找权重文件
        weight_file = self._find_checkpoint(WEIGHTS_DIR)

        # 验证文件完整性
        self._validate_checkpoint(weight_file)

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
            f"SAM3 权重文件未找到。尝试过:\n"
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

        from auto2dlabel.models.model_catalog import COCO_CLASSES

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


def create_segmentation_model(name: str = "fastsam", **kwargs):
    """工厂函数：根据名称创建分割模型。

    Args:
        name: 模型名或类别:
            - "fastsam" / "FastSAM-s.pt" / "FastSAM-x.pt" → FastSAMModel
            - "sam2" / "sam_b.pt" / "sam2_t.pt" / ... → SAM2Model
            - "maskrcnn" / "maskrcnn_resnet50_fpn_v2" → MaskRCNNModel
            - "sam3" / "sam3.pt" → SAM3Model
        **kwargs: 透传参数（如 device, model_name）。
    """
    from auto2dlabel.models.model_catalog import SEGMENTATION_MODELS

    name_lower = name.lower()

    if "maskrcnn" in name_lower:
        return MaskRCNNModel(**kwargs)
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
