"""Web 3D 复核 frame-data payload 纯函数（v0.3 P4：四视图真 3D 复核）。

契约：点云经 velo_to_cam 转**相机系**（x 右 y 下 z 前）交付，逐框角点
corners_cam 8x3 同系——yaw/尺寸/标定数学全部留在 Python 侧，前端零 calib
依赖，仅换轴渲染 (x, z, -y)（three 装配见 static/viewer3d.js）。
下采样 seed 确定性：同帧同 seed → 同一批点（审核一致性，不随请求变化）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from auto3dlabel.data.nuscenes import load_lidar_file, load_nuscenes, rot_matrix
from auto3dlabel.schema.box3d import Box3D, load_points_bin
from auto3dlabel.schema.calib import KittiCalib
from auto3dlabel.tools.geometry import GLOBAL_TO_CAM_LIKE, global_to_cam_like

# 单帧点云交付上限（KITTI 帧 ~12 万点；渲染帧率与 JSON 体积折中）
MAX_POINTS = 100_000
_DOWNSAMPLE_SEED = 0


def downsample_points(
    pts: np.ndarray, max_points: int = MAX_POINTS, seed: int = _DOWNSAMPLE_SEED
) -> np.ndarray:
    """(N,C) 点随机均匀下采样到 max_points（seed 确定性；不超限原样返回同一对象）。"""
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pts), size=max_points, replace=False))
    return pts[idx]


def validate_queue_name(name: str) -> bool:
    """队列文件名白名单（从 server.py 抽公共：后缀白名单 + 路径遍历拒绝，行为零变）。"""
    return (
        name.endswith(("_review.json", "_reviewed.json"))
        and "/" not in name
        and "\\" not in name
    )


def _objects_from_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """annotations（Box3D dict 列表）→ objects（坏框跳过；KITTI/nus 共用）。

    index = 原 annotations 下标（前端表格/保存回写对齐，坏框跳过不错位）。
    """
    objects: list[dict[str, Any]] = []
    for i, b in enumerate(annotations):
        if not isinstance(b, dict):
            continue
        try:
            box = Box3D.from_dict(b)
        except (TypeError, ValueError):
            continue  # 坏框跳过（宁缺勿假）
        objects.append({
            "index": i,
            "label": box.label,
            "confidence": box.confidence,
            "fit_points": box.fit_points,
            "corners": box.corners_cam().tolist(),
        })
    return objects


def _nuscenes_points_cam_like(
    data: dict[str, Any], nusc: Any, pcd_path: str, ego: tuple[float, float, float]
) -> np.ndarray:
    """nuScenes 点云 → cam_like 帧（两级位姿正确链；查表失败 → 旧链降级零标定行为）。

    正确链（v0.4 修复既有错位 bug：LIDAR 传感器系 .bin 曾被直接当全局系喂
    global_to_cam_like，四视图错位 ~1800m）：p_glob = R_e·(R_l·p_lidar + t_l) + t_e
    → p_cl = M @ (p_glob − t_ego)。LIDAR 位姿经队列 sample_token 查 devkit 表；
    pcd_path 文件名与 LIDAR 记录不一致（数据被移动/混用）→ 降级旧链（宁缺勿假）。
    """
    pts = load_lidar_file(Path(pcd_path))
    if nusc is not None:
        try:
            sample = nusc.get("sample", str(data.get("sample_token") or ""))
            lidar_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            if Path(pcd_path).name != Path(lidar_rec["filename"]).name:
                raise ValueError("pcd_path 与 LIDAR sample_data 文件名不一致")
            calib = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
            ego_pose = nusc.get("ego_pose", lidar_rec["ego_pose_token"])
            r_l = rot_matrix(tuple(calib["rotation"]))
            t_l = np.asarray(calib["translation"], dtype=np.float64)
            r_e = rot_matrix(tuple(ego_pose["rotation"]))
            t_e = np.asarray(ego_pose["translation"], dtype=np.float64)
            pts_glob = (r_e @ (r_l @ pts[:, :3].T + t_l[:, None])).T + t_e
            return (pts_glob - t_e) @ GLOBAL_TO_CAM_LIKE.T
        except (KeyError, ValueError):
            pass  # 查表失败 → 降级（旧链，无标定行为不变）
    return global_to_cam_like(pts[:, :3], ego)


def _nuscenes_cam_proj(
    cam: dict[str, Any], nusc: Any, image_path: str
) -> dict[str, Any] | None:
    """队列相机 {name,token,filename} → 相机图叠加投影参数（devkit 查表；失败 → None 纯图）。

    P2 = K @ [R_cᵀ·R_egoᵀ·Mᵀ | −R_cᵀ·t_c]（直接吃 cam_like 点，前端 projectP2 零改动）；
    k_inv = R2ᵀ·K⁻¹（像素→cam_like 方向，吸收旋转，前端 p2ToGround 零改动；
    与 KITTI inv(p2[:,:3]) 隐含 R0_rectᵀ·K⁻¹ 的模式对称）；
    cam_center = −k_inv @ P2[:,3]（cam_like 系相机中心，通用精确反投影锚点）。
    img_size 读真实图像尺寸（nuScenes 主点 816.27 不在图中心，2·cu 不可靠）。
    """
    if nusc is None:
        return None
    try:
        sd = nusc.get("sample_data", str(cam.get("token") or ""))
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        k = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
        r_c = rot_matrix(tuple(calib["rotation"]))
        t_c = np.asarray(calib["translation"], dtype=np.float64)
        r_ego = rot_matrix(tuple(ego_pose["rotation"]))
    except (KeyError, ValueError):
        return None
    r2 = r_c.T @ r_ego.T @ GLOBAL_TO_CAM_LIKE.T
    p2 = k @ np.hstack([r2, (-r_c.T @ t_c)[:, None]])
    k_inv = r2.T @ np.linalg.inv(k)
    img_size: tuple[int, int] | None = None
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            img_size = (int(im.width), int(im.height))
    except (OSError, ValueError, ImportError):
        img_size = None
    return {
        "p2": p2.tolist(),
        "k_inv": k_inv.tolist(),
        "cam_center": (-k_inv @ p2[:, 3]).tolist(),
        **({"img_size": list(img_size)} if img_size is not None else {}),
    }


def _nuscenes_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """nuScenes 队列（dataset=="nuscenes"）→ 渲染 payload：cam_like 点云 + 6 相机叠加投影。

    devkit 查表交付（v0.4 增量）：点云两级位姿正确链（_nuscenes_points_cam_like）+
    每相机 p2/k_inv/cam_center/img_size（_nuscenes_cam_proj，前端 projectP2/p2ToGround
    零改动复用）；devkit 不可用/查表失败 → 相机降级纯图（无 p2），点云降级旧链。
    """
    pcd_path = str(data.get("pcd_path") or "")
    try:
        load_lidar_file(Path(pcd_path))  # 先探测可读（错误路径 → 整帧 None）
    except (OSError, ValueError):
        return None
    ego_raw = data.get("ego_translation", (0.0, 0.0, 0.0))
    ego = (float(ego_raw[0]), float(ego_raw[1]), float(ego_raw[2]))
    try:
        nusc = load_nuscenes(Path(str(data.get("dataroot") or "")), str(data.get("version") or "v1.0-mini"))
    except (ImportError, FileNotFoundError, OSError):
        nusc = None  # devkit 不可用 → 相机纯图 + 点云旧链（行为不变）
    pts_cam = _nuscenes_points_cam_like(data, nusc, pcd_path, ego)
    dataroot = str(data.get("dataroot") or "")
    cameras: list[dict[str, Any]] = []
    for c in data.get("cameras", []):
        if not isinstance(c, dict) or not c.get("filename"):
            continue
        image_path = str(Path(dataroot) / c["filename"])
        cam_out: dict[str, Any] = {
            "name": c.get("name", ""),
            "image_path": image_path,
        }
        proj = _nuscenes_cam_proj(c, nusc, image_path)
        if proj is not None:
            cam_out.update(proj)
        cameras.append(cam_out)
    return {
        "image": data.get("image", ""),
        "image_path": "",
        "bev_path": "",
        "points": downsample_points(pts_cam).tolist(),
        "objects": _objects_from_annotations(data.get("annotations", [])),
        "cameras": cameras,
        "dataset": "nuscenes",
    }


def frame_payload(name: str, review_dir: Path) -> dict[str, Any] | None:
    """队列文件名 → 四视图渲染 payload；损坏/缺 pcd/缺 calib → None（端点转 400）。

    KITTI 输出键：image/image_path/bev_path（透传）、points（相机系 (N,3) list）、
    objects（label/confidence/fit_points/corners 8x3 list；坏框跳过）、
    p2/k_inv/img_size（相机图投影叠加：前端 projectP2 投影 + p2ToGround 地面反投影拖动）。
    nuScenes（dataset=="nuscenes"）：cam_like 点云（两级位姿正确链）+ cameras 6 相机，
    每相机带 p2/k_inv/cam_center/img_size（devkit 查表；失败降级纯图，见 _nuscenes_payload）。
    """
    src = review_dir / name
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("dataset") == "nuscenes":
        return _nuscenes_payload(data)
    pcd_path = str(data.get("pcd_path") or "")
    calib_path = str(data.get("calib_path") or "")
    try:
        pts_velo = load_points_bin(Path(pcd_path))
        calib = KittiCalib.from_file(Path(calib_path))
        pts_cam = calib.velo_to_cam(pts_velo)
    except (OSError, ValueError):
        return None
    p2 = calib.P2.astype(np.float64)
    k_inv = np.linalg.inv(p2[:, :3])
    cam_center = -k_inv @ p2[:, 3]  # 相机中心（rect cam0 系；P2 非零平移列的精确反投影锚点）
    img_w, img_h = (data.get("image_size") or [1242, 375])[:2]
    return {
        "image": data.get("image", ""),
        "image_path": data.get("image_path", ""),
        "bev_path": data.get("bev_path", ""),
        "points": downsample_points(pts_cam).tolist(),
        "objects": _objects_from_annotations(data.get("annotations", [])),
        # 相机图叠加投影（KITTI rect：P2 3x4；K⁻¹ + 相机中心供像素→地面平面 y=cy 反投影）
        "p2": p2.tolist(),
        "k_inv": k_inv.tolist(),
        "cam_center": cam_center.tolist(),
        "img_size": [int(img_w), int(img_h)],
    }
