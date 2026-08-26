"""LiDAR 3D 检测器封装（v0.2 P1）——mmdet3d CenterPoint/PointPillars 直检引擎。

母版 = auto2dlabel/models/lane.py 四段式：Protocol（测试注入 Fake）→ 具体类
（零加载 __init__ + 幂等 _load + ImportError 守卫）→ create_xxx 工厂（名字路由）。
红线：mmdet3d/mmcv 全部 import 在 _load 守卫内——mmdet3d 未装状态下
import 本模块、走工厂、单测均全绿。

推理链（手写最小链，不跑 DataLoader）：
    Config.fromfile → MODELS.build → load_checkpoint → Points(velodyne) →
    model.data_preprocessor（Det3DDataPreprocessor，内含 voxelize CUDA op）→
    model(mode='predict') → pred_instances_3d（bboxes_3d/scores_3d/labels_3d）。

输出语义（mmdet3d KITTI 模型）：相机系 [x,y,z,l,h,w,ry]，x/y/z = 底面中心
（y 向下）——与 label_2 语义同构，直接复用 Box3D.from_gt_row 转换
（cy = y - h/2；yaw_bev = rotation_y_to_yaw(ry)，唯一转换点纪律不变）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from auto3dlabel.configs.kitti import (
    DEFAULT_CONF,
    DETECTOR3D_NAMES,
    KITTI_IMG_H,
    KITTI_IMG_W,
    MMDET3D_CONFIG_DIR,
    WEIGHTS_DIR,
)
from auto3dlabel.schema.box3d import Box3D, KittiFrame

# mmdet3d KITTI 3-class config 默认类序（config 多源提取失败时的 fallback）
MMDET3D_KITTI_CLASSES = ("Pedestrian", "Cyclist", "Car")


@dataclass
class Det3DResult:
    """LiDAR 直检单目标输出（相机系 7 值 = mmdet3d KITTI 输出语义）。"""

    label: str  # KITTI 类名（Car/Pedestrian/Cyclist）
    confidence: float
    bbox: list[float]  # [x, y, z, l, h, w, ry] 底面中心（相机系，y 向下）
    x1: float = 0.0  # 2D 投影外接框（导出字段 4-7，8 角点投影极值）
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    def to_box3d(self) -> Box3D:
        """→ Box3D（cy = y - h/2 底面→体积中心；yaw_bev = rotation_y_to_yaw(ry)）。"""
        x, y, z, l, h, w, ry = (float(v) for v in self.bbox)
        box = Box3D.from_gt_row(self.label, h, w, l, x, y, z, ry)
        box.confidence = self.confidence
        box.x1, box.y1, box.x2, box.y2 = self.x1, self.y1, self.x2, self.y2
        return box


class Detector3D(Protocol):
    """3D 检测器协议 —— 测试注入 FakeDetector3D（零真实权重铁律）。"""

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（懒加载后为 config/dataset_meta 真实类序）。"""
        ...

    def detect(
        self, frame: KittiFrame, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        """KITTI 单帧点云 → 3D 检测列表（conf_threshold 覆盖构造默认）。"""
        ...


class Mmdet3dDetector:
    """mmdet3d 检测器（懒加载：构造零加载，detect 时 _load 幂等 + ImportError 守卫）。"""

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

    def _load(self) -> Any:
        """懒加载 mmdet3d 模型（幂等；mmdet3d 未装抛 ImportError，守卫路径）。

        官方 init_model（含 register_all_modules / SyncBN 转换 / dataset_meta）。
        """
        if self._model is not None:
            return self._model
        try:
            import torch
            from mmdet3d.apis import init_model  # pyright: ignore[reportMissingImports]
            from mmengine.config import Config
        except ImportError as e:
            raise ImportError(
                "mmdet3d/mmcv 未安装，无法使用 LiDAR 3D 检测器"
                "（安装步骤见 milestone/v0.2.md M1）"
            ) from e
        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = init_model(
            config=self._config_path, checkpoint=self._checkpoint_path, device=device
        )
        self._class_names = _extract_class_names(Config.fromfile(self._config_path), model)
        self._model = model
        return self._model

    @property
    def class_names(self) -> tuple[str, ...]:
        """类名元组（触发懒加载——加载后为 config/dataset_meta 真实类序）。

        构造后默认 KITTI 3 类；只有 _load 后才反映模型真实类序，
        故读取即触发加载（未装 mmdet3d 时抛 ImportError，同 detect）。
        """
        self._load()
        return self._class_names

    def detect_points(
        self, pts: np.ndarray, conf_threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LiDAR 点云 → (boxes, scores, labels)——模型原始输出系（不做坐标系转换）。

        KITTI 模型：LiDAR 系（x 前/y 左/z 上，z=底面中心，origin=(0.5,0.5,0)）；
        nuScenes 模型：自车系（同向，z=体积中心，origin=(0.5,0.5,0.5)）。
        调用方按模型语义负责 origin 与坐标系转换（KITTI detect / nuScenes 冒烟各持一端）。
        """
        from mmdet3d.apis import inference_detector  # pyright: ignore[reportMissingImports]

        model = self._load()
        # 官方推理链（LoadPointsFromDict pipeline）；1.4.0 实际返回 (results[0], data[0])
        # 类型标注误写为 Det3DDataSample（不可迭代）→ cast 对齐真实返回
        out: Any = cast(Any, inference_detector(model, np.asarray(pts, dtype=np.float32)))[0]
        pred: Any = out.pred_instances_3d
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        mask = scores >= threshold
        return boxes[mask], scores[mask], labels[mask]

    def detect(
        self, frame: KittiFrame, conf_threshold: float | None = None
    ) -> list[Det3DResult]:
        """单帧点云 → 检测列表（≥阈值）；含 8 角点投影 2D 外接框。"""
        boxes, scores, labels = self.detect_points(frame.load_points(), conf_threshold)
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # mmdet3d KITTI 模型输出 LiDAR 系（x 前/y 左/z 上，底面中心）→ 相机系
        # 唯一转换点：官方 convert_to(CAM) + calib 4x4（R0_rect @ Tr_velo_to_cam）
        from mmdet3d.structures import (  # pyright: ignore[reportMissingImports]
            Box3DMode,
            LiDARInstance3DBoxes,
        )

        calib = frame.load_calib()
        # KITTI LiDAR 系 z = 底面中心（mmdet3d 默认 origin=(0.5,0.5,0)）——传 (0.5,0.5,0.5)
        # 会被 convert_to 误当「中心」再做一次底面换算，y_cam 凭空 +h/2（3D AP 全灭）
        lidar_boxes = LiDARInstance3DBoxes(boxes, box_dim=7, origin=(0.5, 0.5, 0.0))
        cam_boxes = lidar_boxes.convert_to(
            Box3DMode.CAM, rt_mat=calib.velo_to_cam_matrix(), correct_yaw=True
        )
        boxes = cam_boxes.tensor.detach().cpu().numpy()

        calib = frame.load_calib()
        results: list[Det3DResult] = []
        for i in range(len(boxes)):
            if float(scores[i]) < threshold:
                continue
            label = self._class_names[int(labels[i])]
            x, y, z, l, h, w, ry = (float(v) for v in boxes[i])
            box = Box3D.from_gt_row(label, h, w, l, x, y, z, ry)
            u, v, valid = calib.project_cam_to_image(
                box.corners_cam(), KITTI_IMG_W, KITTI_IMG_H
            )
            if valid.any():
                x1, y1 = float(u[valid].min()), float(v[valid].min())
                x2, y2 = float(u[valid].max()), float(v[valid].max())
            else:
                x1 = y1 = x2 = y2 = 0.0  # 目标在视野外：2D 框哨兵（导出字段不参与评测）
            results.append(
                Det3DResult(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=[x, y, z, l, h, w, ry],
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return results


def _extract_class_names(cfg: Any, model: Any | None = None) -> tuple[str, ...]:
    """→ 类名元组（多源提取：model.dataset_meta > config metainfo > class_names）。

    全失败默认 3 类序（KITTI：Pedestrian/Cyclist/Car）。
    """
    meta = getattr(model, "dataset_meta", None)
    classes: Any = None
    if isinstance(meta, dict):
        classes = meta.get("classes")
    if not classes:
        metainfo = cfg.get("metainfo")
        if isinstance(metainfo, dict):
            classes = metainfo.get("classes")
    if not classes:
        classes = cfg.get("class_names")
    if classes:
        return tuple(str(c) for c in classes)
    return MMDET3D_KITTI_CLASSES


def create_detector3d(model_name: str | None) -> Detector3D | None:
    """工厂：DETECTOR3D_NAMES 路由 → Mmdet3dDetector；None/未知名 → None（2D 引擎）。

    判定单一事实源：调用方以返回值非 None 判「3D 引擎」（cli/tools3d 共用）。
    构造零加载——config/权重文件存在性延迟到 _load（未装/缺权重时抛错给调用方兜底）。
    """
    if not model_name or model_name not in DETECTOR3D_NAMES:
        return None
    entry = DETECTOR3D_NAMES[model_name]
    config_path = MMDET3D_CONFIG_DIR / entry["config"]
    checkpoint_path = WEIGHTS_DIR / entry["weights_dir"] / entry["checkpoint"]
    return Mmdet3dDetector(config_path=config_path, checkpoint_path=checkpoint_path)
