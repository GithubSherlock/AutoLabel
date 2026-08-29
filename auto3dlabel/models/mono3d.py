"""单目 3D 检测器封装（v0.3 P3）——mmdet3d PGD KITTI 单目引擎。

定位：LiDAR 不可用时的降级方案 + 交叉验证基准（moderate 预期 ≤30% 单目带，
CLAUDE.md 调研结论）。母版 = models/detection3d.py 四段式：零加载 __init__ +
幂等 _load + ImportError 守卫 + create_xxx 工厂（名字路由）。
红线：mmdet3d/mmcv 全部 import 在 _load 守卫内——mmdet3d 未装状态下
import 本模块、走工厂、单测均全绿。

推理链（手写最小链，不跑 DataLoader，绕开 kitti_infos pkl）：
    Config.fromfile → MODELS.build → load_checkpoint →
    data_ dict（images.CAM2 {img_path, cam2img}，官方 inference_mono_3d_detector
    同款骨架）→ Compose(test_pipeline) → pseudo_collate → model.test_step。

输出语义（mmdet3d KITTI 单目模型）：相机系 7 值 [x,y,z,l,h,w,ry]，x/y/z =
底面中心（y 向下）——与 label_2 语义同构，Det3DResult.to_box3d 直接复用
（LiDAR 引擎还需 convert_to(CAM) 一步，单目省去）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.kitti import (
    DEFAULT_CONF,
    KITTI_IMG_H,
    KITTI_IMG_W,
    MMDET3D_CONFIG_DIR,
    MONO3D_NAMES,
    WEIGHTS_DIR,
)
from auto3dlabel.models.detection3d import (
    MMDET3D_KITTI_CLASSES,
    Det3DResult,
    _extract_class_names,
    _init_model_trusted,
)
from auto3dlabel.schema.box3d import Box3D, KittiFrame


def build_mono_data(
    image_path: str | Path,
    cam2img: np.ndarray,
    box_type_3d: Any = None,
    box_mode_3d: Any = None,
) -> dict[str, Any]:
    """单目推理 data_ dict（官方 inference_mono_3d_detector 骨架，绕开 ann_file pkl）。

    LoadImageFromFileMono3D 契约：读 images['CAM2']['img_path'] 与
    ['cam2img']（KITTI P2 3x4），img 加载进 results['img']。
    box_type_3d 必须传 get_box_type('Camera') 返回的**类**（CameraInstance3DBoxes）
    ——LoadImageFromFileMono3D/Pack3DDetInputs 均原样透传，pgd head 以
    img_meta['box_type_3d'](...) 构造框，传字符串 'Camera' 会
    TypeError: 'str' object is not callable（实测）。
    两者模块级零 mmdet3d 依赖，由调用方 _load 提供；纯函数测试可传 None。
    """
    return {
        "images": {
            "CAM2": {
                "img_path": str(image_path),
                "cam2img": np.asarray(cam2img, dtype=np.float32),
            }
        },
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    }


def _mono_output_to_det3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
) -> list[Det3DResult]:
    """单目原始输出（相机系 7 值 [x,y,z,l,h,w,ry] 底面中心）→ Det3DResult。

    L-W 顺序锁死：mmdet3d KITTI 单目 bbox_coder 输出 idx3=l、idx4=h、idx5=w
    （与 LiDAR 引擎同构）；2D 外接框透传（pred_bbox2d=True 头部直出）。
    """
    results: list[Det3DResult] = []
    for i in range(len(boxes)):
        label = class_names[int(labels[i])]
        results.append(
            Det3DResult(
                label=label,
                confidence=float(scores[i]),
                bbox=[float(v) for v in boxes[i]],
            )
        )
    return results


class Mono3dDetector:
    """mmdet3d 单目检测器（懒加载：构造零加载，detect 时 _load 幂等 + ImportError 守卫）。"""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        conf_threshold: float = DEFAULT_CONF,
        device: str | None = None,
    ) -> None:
        self._config_path = str(config_path)
        self._checkpoint_path = str(checkpoint_path)
        self.conf_threshold = conf_threshold
        self._device = device  # None = cuda 可用则 cuda，否则 cpu
        self._model: Any = None
        self._class_names: tuple[str, ...] = MMDET3D_KITTI_CLASSES
        self._pipeline: Any = None
        self._box_type_3d: Any = None
        self._box_mode_3d: Any = None

    def _load(self) -> Any:
        """懒加载 mmdet3d 单目模型（幂等；mmdet3d 未装抛 ImportError，守卫路径）。"""
        if self._model is not None:
            return self._model
        try:
            import mmdet3d  # noqa: F401  # 守卫探测：未装时抛 ImportError
            import torch
            from mmdet3d.structures import (  # pyright: ignore[reportMissingImports]
                get_box_type,
            )
            from mmengine.config import Config
        except ImportError as e:
            raise ImportError(
                "mmdet3d/mmcv 未安装，无法使用单目 3D 检测器"
                "（安装步骤见 milestone/v0.2.md M1）"
            ) from e
        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # 确定性红线（同 Mmdet3dDetector，批量 parity 铁律）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = _init_model_trusted(self._config_path, self._checkpoint_path, device)
        cfg = Config.fromfile(self._config_path)
        self._class_names = _extract_class_names(cfg, model)
        # 测试管线与 box 语义直接取自 config（单一事实源）
        from mmcv.transforms import Compose  # pyright: ignore[reportMissingImports]

        self._pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        # box_type_3d 必须是**类**（见 build_mono_data docstring 的实测 bug）
        self._box_type_3d, self._box_mode_3d = get_box_type(
            cfg.test_dataloader.dataset.box_type_3d
        )
        self._model = model
        return self._model

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（触发懒加载——加载后为 config/dataset_meta 真实类序）。"""
        self._load()
        return self._class_names

    def detect(
        self, frame: KittiFrame, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        """KITTI 单帧图像 → 检测列表（≥阈值）；含 8 角点投影 2D 外接框。

        协议与 Detector3D.detect 同构（frame 输入），调用方零感知差异。
        """
        import torch
        from mmengine.dataset import pseudo_collate  # pyright: ignore[reportMissingImports]

        model = self._load()
        calib = frame.load_calib()
        data_ = build_mono_data(
            frame.image_path,
            calib.P2,
            box_type_3d=self._box_type_3d,
            box_mode_3d=self._box_mode_3d,
        )
        data_ = self._pipeline(data_)
        collated = pseudo_collate([data_])
        with torch.no_grad():
            results = model.test_step(collated)
        # 单目 test_step 返回 results list（每图一个 Det3DDataSample）——
        # 与 LiDAR inference_detector 的 (results, data) 元组不同（实码确认）
        pred: Any = results[0].pred_instances_3d
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        mask = scores >= threshold
        dets = _mono_output_to_det3d(
            boxes[mask], scores[mask], labels[mask], self._class_names
        )
        for det in dets:
            x, y, z, l, h, w, ry = (float(v) for v in det.bbox)
            box = Box3D.from_gt_row(det.label, h, w, l, x, y, z, ry)
            u, v, valid = calib.project_cam_to_image(
                box.corners_cam(), KITTI_IMG_W, KITTI_IMG_H
            )
            if valid.any():
                det.x1, det.y1 = float(u[valid].min()), float(v[valid].min())
                det.x2, det.y2 = float(u[valid].max()), float(v[valid].max())
            else:
                det.x1 = det.y1 = det.x2 = det.y2 = 0.0  # 视野外哨兵（同 LiDAR 引擎）
        return dets


def create_mono3d_detector(model_name: str | None) -> Mono3dDetector | None:
    """工厂：MONO3D_NAMES 路由 → Mono3dDetector；None/未知名 → None。

    构造零加载——config/权重文件存在性延迟到 _load（同 create_detector3d）。
    """
    if not model_name or model_name not in MONO3D_NAMES:
        return None
    entry = MONO3D_NAMES[model_name]
    config_path = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint_path = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    return Mono3dDetector(config_path=config_path, checkpoint_path=checkpoint_path)
