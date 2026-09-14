"""地图矢量投影链(AutoLabel 侧):ego 系折线 → 相机像素 + BEV 面板(纯值)。

**复制自** `AutoDriveData/autodrivedata/mapviz.py` + `geometry.py` 的投影部分,
版本锁定;AutoLabel 不 import AutoDriveData(依赖单向红线),纯值逻辑本地复制。

坐标链(与产出方 mapviz 同口径):
  ego 局部系(x 前向 / y 左向 / z=0)
  → CARLA 世界(绕 z 转 yaw + 平移 ego2global)
  → `world_to_cam`(复制 CARLA_TO_CAM + carla_rotation_matrix)→ 像素

**旋转单位是弧度** —— `cam_pose` 出口口径与产出方一致(度数值直接传会把
侧/后相机画错位;历史上 6 相机里只有 yaw≈0 的 CAM_FRONT 恰好接近正确)。
第一版**只投 CAM_FRONT**(yaw=0,天然避该坑);多相机走 cam_pose + 锚点单测。

图像数据源(CLI 参数传入,读文件契约):`calib.json`(每相机 sensor2ego
= [x,y,z,yaw,pitch,roll] 度 + 3×3 intrinsic)、`ego_pose.json`(同构 6 值)。
`ego` 高度用 `ego2global[2]`,不读其他来源(防 ego 高度 vs 相机挂高混淆)。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# 相机式 CAM_FRONT 专用投影(第一版只投 CAM_FRONT,无需 CARLA_TO_CAM 手性翻转——
# CARLA 相机系与 KITTI cam0 同源,R = CARLA_TO_CAM @ R_wcᵀ,CARLA_TO_CAM 见下)
CARLA_TO_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

PRED_COLOR = (255, 0, 255)  # 品红:路面场景罕见(与产出方同口径)
GT_COLOR = (0, 255, 255)  # 青绿:同罕见(植被绿与其可区分)
BEV_X = (-15.0, 15.0)  # BEV 窗口 x(前向,米)——与模型输出系同口径
BEV_Y = (-30.0, 30.0)  # BEV 窗口 y(左向,米)

# (位置, (pitch, yaw, roll) **弧度**)——世界系的相机位姿
CamPose = tuple[tuple[float, float, float], tuple[float, float, float]]


def carla_rotation_matrix(rotation: tuple[float, float, float]) -> np.ndarray:
    """CARLA Rotation (pitch, yaw, roll)[弧度] → 3×3 旋转阵(列 = 局部系轴在世界系的分量)。

    组合顺序 Rz(yaw)·Ry(pitch)·Rx(roll),与 pycarla `Transform.get_matrix()`
    旋转块逐元素一致(复制自 autodrivedata/geometry.py:39)。
    """
    pitch, yaw, roll = rotation
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def world_to_cam(
    points: np.ndarray,
    cam_location: tuple[float, float, float],
    cam_rotation: tuple[float, float, float],
) -> np.ndarray:
    """世界系点 (N,3) → KITTI 相机系:p_k = R_camK_world @ (p_g − t_cam)。

    R_camK_world = CARLA_TO_CAM @ R_world_camCᵀ(复制自 autodrivedata/geometry.py:68)。
    """
    t = np.asarray(cam_location, dtype=np.float64)
    r_wc = carla_rotation_matrix(cam_rotation)
    r_cam_world = CARLA_TO_CAM @ r_wc.T
    return (np.asarray(points, dtype=np.float64)[:, :3] - t) @ r_cam_world.T


def ego_to_world(pts: np.ndarray, ego: list[float]) -> list[tuple[float, float, float]]:
    """ego 局部系折线点 (N, 2|3) → 世界系 (z = ego 高度)。

    ego = [x, y, z, yaw, pitch, roll] 度(ego_pose.json 与 CARLA transform 同序)。
    """
    a = math.radians(ego[3])
    c, s = math.cos(a), math.sin(a)
    return [
        (
            c * float(p[0]) - s * float(p[1]) + ego[0],
            s * float(p[0]) + c * float(p[1]) + ego[1],
            ego[2],
        )
        for p in pts
    ]


def cam_pose(eg: list[float], se: list[float]) -> CamPose:
    """ego2global + sensor2ego → 相机世界位姿(**位置米 / 姿态弧度**)。

    挂点按 `Rz(yaw_ego)` 旋转后平移;pitch/roll 直接相加——挂点 pitch/roll 恒 0、
    ego 俯仰量级 ~0.1°(30m 处像素偏差 <1px),与产出方同近似。
    """
    a = math.radians(eg[3])
    c, s = math.cos(a), math.sin(a)
    loc = (
        eg[0] + c * se[0] - s * se[1],
        eg[1] + s * se[0] + c * se[1],
        eg[2] + se[2],
    )
    return loc, (
        math.radians(eg[4] + se[4]),
        math.radians(eg[3] + se[3]),
        math.radians(eg[5] + se[5]),
    )


def intrinsics_from_k(k: list[list[float]]) -> tuple[float, float, float, float]:
    """infos 内参 3×3 → (fx, fy, cx, cy)。AutoLabel 侧无 CameraIntrinsics,直接读矩阵。"""
    return float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])


def project_points(
    points: np.ndarray,
    ego: list[float],
    pose: CamPose,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> list[tuple[float, float] | None]:
    """点序列 (N, 2|3) ego 系 → 像素/None(相机后或出图外)。

    深度判定与产出方 world_to_img 同口径:深度 ≤ 0.5m 视为相机后,出图外返回 None。
    """
    loc, rot = pose
    pts_world = np.asarray(ego_to_world(np.asarray(points), ego), dtype=np.float64)
    c = world_to_cam(pts_world, loc, rot)
    out: list[tuple[float, float] | None] = []
    for p in c:
        if float(p[2]) <= 0.5:  # 相机系 z = 深度(CARLA_TO_CAM 第三行取世界 x 前)
            out.append(None)
            continue
        u = fx * (p[0] / p[2]) + cx
        v = fy * (p[1] / p[2]) + cy
        if not (0 <= u < width and 0 <= v < height):
            out.append(None)
            continue
        out.append((float(u), float(v)))
    return out


def project_lines(
    lines: list[np.ndarray],
    ego: list[float],
    pose: CamPose,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> list[list[tuple[float, float]]]:
    """折线列表(ego 系)→ [(u, v) 段列表](段 = 连续可见点的相邻对)。

    相机后(深度 ≤ 0.5m)或出图外的点断开成段,不跨遮挡连线。
    """
    segs: list[list[tuple[float, float]]] = []
    for line in lines:
        run: list[tuple[float, float]] = []
        for q in project_points(line, ego, pose, fx, fy, cx, cy, width, height):
            if q is None:
                if len(run) >= 2:
                    segs.append(run)
                run = []
            else:
                run.append(q)
        if len(run) >= 2:
            segs.append(run)
    return segs


def draw_projected_lines(
    draw: Any,
    lines: list[np.ndarray],
    ego: list[float],
    pose: CamPose,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    color: tuple[int, int, int] = PRED_COLOR,
    line_width: int = 3,
) -> int:
    """投影并画到图上,返回**画出的段数**(0 = 没画上:数值自证,不靠目检)。"""
    segs = project_lines(lines, ego, pose, fx, fy, cx, cy, width, height)
    for seg in segs:
        draw.line([q for p in seg for q in p], fill=color, width=line_width)
    return len(segs)


def bev_panel(
    preds: list[list[np.ndarray]],
    gts: list[list[np.ndarray]] | None = None,
    title: str = "",
    size: tuple[int, int] = (420, 420),
    out_of_window_pts: int = 0,
) -> Any:
    """BEV 面板:pred 品红 / GT 青绿;窗口 BEV_X × BEV_Y(与模型输出系同口径)。

    越窗点(out_of_window_pts>0)在标题追加标记(诊断;越窗不拒收)。
    """
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * w, (BEV_Y[1] - y) / (
            BEV_Y[1] - BEV_Y[0]
        ) * h

    for gc in gts or []:
        for g in gc:
            draw.line([q for p in g for q in px(p[0], p[1])], fill=GT_COLOR, width=1)
    for pc in preds:
        for p in pc:
            draw.line([q for pt in p for q in px(pt[0], pt[1])], fill=PRED_COLOR, width=1)
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(120, 120, 120))
    if title:
        draw.text((4, 2), title, fill=(255, 255, 255))
    if out_of_window_pts:
        draw.text((4, 14), f"越窗 {out_of_window_pts} pts", fill=(255, 255, 0))
    return img
