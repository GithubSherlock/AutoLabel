"""nuScenes 数据访问层（v0.2 P3）：dataroot 校验 + devkit 懒加载守卫 + val 枚举 + GT → NusBox
+ 传感器系→全局系补偿链（v0.3 P3 从 smoke_nuscenes 抽公共，smoke_bevfusion 复用）
+ P2 Web 复核支撑（v0.4）：点云加载 / ego 位姿 / 6 相机 / NusBox↔渲染 dict 往返。

数据源：/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz 解压后的
标准 dataroot（见 configs/nuscenes.DEFAULT_NUSCENES_ROOT）。
GT 走 devkit 原始表（sample_annotation 数值 dict），不构造 devkit Box（免 pyquaternion 路径开销）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.configs.nuscenes import (
    DEFAULT_NUSCENES_ROOT,
    NUSCENES_CAMERAS,
    NUSCENES_CATEGORY_MAP,
)
from auto3dlabel.schema.box3d import Box3D
from auto3dlabel.schema.nuscenes_box import NusBox
from auto3dlabel.tools.geometry import (
    cam_like_to_global,
    global_to_cam_like,
    quat_to_yaw,
    yaw_to_quat,
)


def dataroot_exists(root: Path | None = None) -> bool:
    """标准 dataroot 校验（v1.0-mini 表 + LIDAR_TOP samples 齐备）。"""
    root = root or DEFAULT_NUSCENES_ROOT
    return (root / "v1.0-mini").is_dir() and (root / "samples" / "LIDAR_TOP").is_dir()


def load_nuscenes(root: Path | None = None, version: str = "v1.0-mini") -> Any:
    """devkit NuScenes 实例（懒加载；devkit 未装 → ImportError 带安装指引）。

    红线：补装必须 --no-deps（防 numpy 升 2.2.6 破坏 ABI，同 mmcv 纪律）。
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as e:
        raise ImportError(
            "nuscenes-devkit 未安装（P3 依赖）: "
            "pip install nuscenes-devkit --no-deps + pyquaternion"
        ) from e
    root = root or DEFAULT_NUSCENES_ROOT
    if not dataroot_exists(root):
        raise FileNotFoundError(
            f"nuScenes dataroot 不存在: {root}（先解压 "
            "/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz 到此路径）"
        )
    return NuScenes(version=version, dataroot=str(root), verbose=False)


def val_scene_names(version: str = "v1.0-mini") -> list[str]:
    """devkit 官方 split 的 val 场景名（不硬编码场景列表）。

    devkit 版本键：v1.0-trainval → "val"；v1.0-mini → "mini_val"（键名映射在此登记）。
    """
    split_key = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}.get(version, "val")
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as e:
        raise ImportError("nuscenes-devkit 未安装: pip install nuscenes-devkit --no-deps") from e
    splits: Any = create_splits_scenes()  # devkit 无类型注解，索引推断为 slice
    if split_key not in splits:
        raise KeyError(f"split 无键 {split_key}（可用: {sorted(splits)}）")
    return list(splits[split_key])


def samples_of_scene(nusc: Any, scene_name: str) -> list[dict]:
    """场景 → sample 记录列表（按 first_sample_token 链序）。

    nuScenes 的 sample 表即 LIDAR_TOP keyframe 全集（每 sample 恰一个 LIDAR_TOP）。
    """
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise KeyError(f"场景不存在: {scene_name}")
    samples: list[dict] = []
    token: str = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def gt_boxes_of_sample(nusc: Any, sample_token: str) -> list[NusBox]:
    """sample → GT NusBox 列表（原始表数值直映射，类名走官方 category 映射表）。"""
    boxes: list[NusBox] = []
    for ann in nusc.sample_annotation:
        if ann["sample_token"] != sample_token:
            continue
        if "num_lidar_pts" in ann and ann["num_lidar_pts"] <= 0:
            continue  # 无 LiDAR 观测的标注不参与（简化口径，如实记录）
        name = NUSCENES_CATEGORY_MAP.get(ann["category_name"])
        if name is None:
            continue  # 官方忽略类（animal/debris/emergency 等）不参与评测
        boxes.append(
            NusBox(
                label=name,
                confidence=1.0,
                translation=tuple(ann["translation"]),
                size=tuple(ann["size"]),
                quaternion=tuple(ann["rotation"]),
                track_id=ann["instance_token"],
            )
        )
    return boxes


def sensor_sample_data(
    sample: dict, nusc: Any, sensor: str = "LIDAR_TOP"
) -> dict:
    """sample → 指定 sensor 的 sample_data 记录（默认 LIDAR_TOP）。

    ego_pose 与 sensor 位姿都挂在各自 sample_data 记录上（sample 表无此字段）——
    补偿链与 BEVFusion data_ 构建都从这里取。P6b 单目走 CAM_FRONT 记录
    （calibrated_sensor_token → camera_intrinsic 即 FCOS3D cam2img 内参）。
    """
    record = nusc.get("sample_data", sample["data"][sensor])
    assert isinstance(record, dict)
    return record


def lidar_sample_data(sample: dict, nusc: Any) -> dict:
    """sample → LIDAR_TOP 的 sample_data 记录（sensor_sample_data 薄壳，向后兼容）。"""
    return sensor_sample_data(sample, nusc, "LIDAR_TOP")


def load_lidar_file(path: Path) -> np.ndarray:
    """nuScenes pcd 文件 → (N,5) float32（x,y,z,intensity,elongation），剔除 NaN/Inf。

    v0.4 P2 从 smoke_nuscenes._load_lidar 抽公共：load_lidar_points（devkit 表映射）与
    Web payloads（队列文件直存 pcd_path，无 devkit 实例）共用同一解析。
    """
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return np.asarray(raw[np.isfinite(raw).all(axis=1)])


def load_lidar_points(sample: dict, nusc: Any, dataroot: Path) -> np.ndarray:
    """sample → LIDAR_TOP 主点云 (N,5) float32（文件名经 devkit sample_data 表映射；
    sweeps 不合并 = 简化口径）。"""
    filename = lidar_sample_data(sample, nusc)["filename"]
    return load_lidar_file(dataroot / filename)


def ego_translation_of_sample(sample: dict, nusc: Any) -> tuple[float, float, float]:
    """sample → ego 全局位置 (x,y,z)（LIDAR_TOP 记录的 ego_pose 平移分量）。

    P2 cam_like 帧原点 t_ego（global_to_cam_like 消费）。
    """
    lidar_data = lidar_sample_data(sample, nusc)
    ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    t = ego["translation"]
    return (float(t[0]), float(t[1]), float(t[2]))


def cameras_of_sample(sample: dict, nusc: Any) -> list[dict]:
    """sample → 6 相机视图记录列表（标准序 NUSCENES_CAMERAS）。

    每条 {name, token, filename}：filename 相对 dataroot（Web 复核 6 图渲染消费）。
    """
    cams: list[dict] = []
    for name in NUSCENES_CAMERAS:
        rec = sensor_sample_data(sample, nusc, name)
        cams.append({"name": name, "token": rec["token"], "filename": rec["filename"]})
    return cams


def nusbox_to_box3d_dict(
    box: NusBox,
    ego_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fit_points: int = 0,
) -> dict:
    """NusBox（全局系）→ Box3D 渲染 dict（ego 局部相机式帧，P2 Web/前端零改动消费）。

    转换（单测锚点锁定）：center = M @ (t − t_ego)（geometry.global_to_cam_like；
    ego 默认原点，队列侧传 ego_pose 平移）；(h,w,l) = (size[2],size[0],size[1])；
    yaw_bev = −yaw_g ⇒ rotation_y = −yaw_g − π/2（geometry 唯一转换点）。
    宽轴镜像性：8 角点集合完全相同（Box3D 右向 = 全局左），仅角点序镜像，
    边集/headLine/编辑数学不受影响——勿按「角点一一对应」直觉改转换。
    velocity/track_id 以顶层键透传（buildSaveBody {...a} 全量拷贝自动存活）。
    """
    center = global_to_cam_like(np.asarray([box.translation]), ego_translation)[0]
    w_, l_, h_ = box.size
    d = Box3D(
        label=box.label,
        confidence=box.confidence,
        cx=float(center[0]), cy=float(center[1]), cz=float(center[2]),
        h=h_, w=w_, l=l_,
        yaw_bev=-quat_to_yaw(box.quaternion),
        fit_points=fit_points,
    ).to_dict()
    if box.velocity is not None:
        d["velocity"] = list(box.velocity)
    if box.track_id is not None:
        d["track_id"] = box.track_id
    return d


def box3d_dict_to_nusbox(
    d: dict, ego_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> NusBox:
    """渲染 dict（相机式帧）→ NusBox（全局系）——nusbox_to_box3d_dict 逆变换（Web 保存回写）。

    translation = Mᵀ @ center_camlike + t_ego（geometry.cam_like_to_global）；
    size = (w,l,h) 直映射；四元数 = yaw_to_quat(−yaw_bev)（yaw_g = −yaw_bev）；
    velocity/track_id 从顶层键读回（缺省 None）。ego_translation 必须与渲染侧一致
    （队列 JSON 的 ego_translation 键为单一事实源）。
    """
    box = Box3D.from_dict(d)
    center_glob = cam_like_to_global(
        np.asarray([[box.cx, box.cy, box.cz]]), ego_translation
    )[0]
    velocity_raw = d.get("velocity")
    return NusBox(
        label=box.label,
        confidence=box.confidence,
        translation=(float(center_glob[0]), float(center_glob[1]), float(center_glob[2])),
        size=(box.w, box.l, box.h),
        quaternion=yaw_to_quat(-box.yaw_bev),
        velocity=(
            (float(velocity_raw[0]), float(velocity_raw[1])) if velocity_raw else None
        ),
        track_id=d.get("track_id"),
    )


def rot_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵（Hamilton 约定，nus devkit 同）。

    v0.3 P3 起公共：boxes_sensor_to_global 补偿链 + bevfusion3d 标定位姿共用。
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton 积 q1 ⊗ q2（nus devkit Quaternion 同约定）。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _label_mask(
    labels: np.ndarray, class_names: list[str] | tuple[str, ...]
) -> np.ndarray:
    """标签越界丢弃 + 每调用一次向 stderr 诊断（类表与 head 类数不符时）。

    boxes_sensor_to_global / boxes_cam_to_global 共用（复用不复制）。
    """
    mask = (labels >= 0) & (labels < len(class_names))
    if not bool(mask.all()):
        print(
            f"[诊断] 标签越界跳过 {int((~mask).sum())} 个框 "
            f"len(class_names)={len(class_names)}",
            file=sys.stderr,
        )
    return mask


def boxes_sensor_to_global(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    ego: dict,
    calib: dict,
) -> list[NusBox]:
    """传感器系 9 值输出 → 全局系 NusBox（两级补偿链，v0.3 P3 抽公共）。

    nus 模型 9 值输出 [x,y,z,l,w,h,yaw,vx,vy]（传感器系中心 + 速度）——
    实测校准：2021-08 旧权重 L-W 顺序（与 1.4.0 DeltaXYZWLHRBBoxCoder 的
    W-L 定义相反，跨场景 81.5% car 匹配实证）；yaw 为标准语义（0=+x 前）。
    补偿：传感器系 → calib 位姿 → ego 地面系 → ego_pose → 全局系。
    纯函数：ego/calib 为 devkit ego_pose/calibrated_sensor 记录 dict
    （{'rotation': [w,x,y,z], 'translation': [x,y,z]}），由调用方查表传入；
    标签越界丢弃并每调用一次向 stderr 诊断（类表与 head 类数不符时）。
    """
    r_ego = rot_matrix(tuple(ego["rotation"]))
    t_ego = np.asarray(ego["translation"], dtype=np.float64)
    q_ego = tuple(ego["rotation"])
    r_calib = rot_matrix(tuple(calib["rotation"]))
    t_calib = np.asarray(calib["translation"], dtype=np.float64)
    q_calib = tuple(calib["rotation"])
    out: list[NusBox] = []
    keep = _label_mask(labels, class_names)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        idx = int(labels[i])
        x, y, z, l, w, h, yaw = (float(v) for v in boxes[i, :7])
        center = r_ego @ (r_calib @ np.array([x, y, z]) + t_calib) + t_ego
        quat = quat_mul(q_ego, quat_mul(q_calib, yaw_to_quat(yaw)))
        velocity: tuple[float, float] | None = None
        if boxes.shape[1] >= 9:
            v_glob = r_ego @ (r_calib @ np.array([float(boxes[i, 7]), float(boxes[i, 8]), 0.0]))
            velocity = (float(v_glob[0]), float(v_glob[1]))
        out.append(
            NusBox(
                label=class_names[idx],
                confidence=float(scores[i]),
                translation=(float(center[0]), float(center[1]), float(center[2])),
                size=(w, l, h),
                quaternion=quat,
                velocity=velocity,
            )
        )
    return out


# 相机系→ego 的固定四元数：绕相机 x 轴转 π/2（v0.15 output_to_nusc_box 的 q2）
# = (cos(π/4), sin(π/4), 0, 0)。与 yaw_to_quat 组合时位于 yaw 旋转**之后**（q2 ⊗ q1）。
QX_HALF_PI: tuple[float, float, float, float] = (
    0.7071067811865476,
    0.7071067811865476,
    0.0,
    0.0,
)


def boxes_cam_to_global(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    ego: dict,
    calib: dict,
) -> list[NusBox]:
    """相机系 9 值单目输出 → 全局系 NusBox（复刻 v0.15 output_to_nusc_box 定式，P6b）。

    FCOS3D nuScenes 权重 = 2021-07 训练（v0.15 旧 converter 语义）：
    输出 [x,y,z,w,l,h,yaw,vx,vz] = **几何中心** + dims 直传 (w,l,h) +
    yaw 无负号（相机系 nuScenes yaw）+ 相机 x-z 平面速度。
    v1.4.0 官方 eval 的 output_to_nusc_box 假定**新** converter 语义
    （dims [2,0,1] 重排 + yaw 取负），与旧权重不匹配——本函数按 v0.15 版
    （已下载源码逐行核对）实现：dims 直传、yaw 原样、q_local = q2 ⊗ q1
    （q1 = 绕 z 转 yaw，q2 = 绕 x 转 π/2）、center = 几何中心，
    然后 cam2ego（calib）→ ego2global（ego_pose）两级 rotate+translate。
    纯函数：ego/calib 同 boxes_sensor_to_global 的 devkit 记录契约。
    """
    r_ego = rot_matrix(tuple(ego["rotation"]))
    t_ego = np.asarray(ego["translation"], dtype=np.float64)
    q_ego = tuple(ego["rotation"])
    r_calib = rot_matrix(tuple(calib["rotation"]))
    t_calib = np.asarray(calib["translation"], dtype=np.float64)
    q_calib = tuple(calib["rotation"])
    out: list[NusBox] = []
    keep = _label_mask(labels, class_names)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        idx = int(labels[i])
        x, y, z, w, l, h, yaw = (float(v) for v in boxes[i, :7])
        center = r_ego @ (r_calib @ np.array([x, y, z]) + t_calib) + t_ego
        # q_local = q2 ⊗ q1（Hamilton 积）：先绕相机 z 转 yaw，再绕 x 转 π/2
        q_local = quat_mul(QX_HALF_PI, yaw_to_quat(yaw))
        quat = quat_mul(q_ego, quat_mul(q_calib, q_local))
        velocity: tuple[float, float] | None = None
        if boxes.shape[1] >= 9:
            # 相机 x-z 平面速度 3 向量 [vx, 0, vz] 随两级位姿旋转 → 全局 x-y
            v_glob = r_ego @ (r_calib @ np.array([float(boxes[i, 7]), 0.0, float(boxes[i, 8])]))
            velocity = (float(v_glob[0]), float(v_glob[1]))
        out.append(
            NusBox(
                label=class_names[idx],
                confidence=float(scores[i]),
                translation=(float(center[0]), float(center[1]), float(center[2])),
                size=(w, l, h),
                quaternion=quat,
                velocity=velocity,
            )
        )
    return out
