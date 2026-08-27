"""合成测试数据（零真实权重铁律：不读 KITTI 数据盘，全部硬编码/合成）。

calib 数值 = 真实 calib_000000.txt 提取（已数值验证投影链：
velodyne (8.752,-1.800,-1.546) → 像素 (758,299)，相机系 (1.7834,1.4169,8.4302)）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from auto3dlabel.schema.box3d import KittiFrame

# 真实 calib_000000.txt 的 7 行（P0-P3/R0_rect/Tr_velo_to_cam/Tr_imu_to_velo）
CALIB_LINES = """P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03
R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01
Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
"""

# 投影链数值锚点（真实 calib + 已验证）
VELO_ANCHOR = (8.752, -1.800, -1.546)
ANCHOR_PIXEL = (758, 299)
ANCHOR_CAM = (1.78336479, 1.41691564, 8.43024821)

IMG_W, IMG_H = 1242, 375


def write_calib(path: str | Path) -> None:
    Path(path).write_text(CALIB_LINES, encoding="utf-8")


def rect_points(
    cx: float,
    cz: float,
    length: float,
    width: float,
    yaw: float,
    n: int = 400,
    y_lo: float = -1.5,
    y_hi: float = -0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """合成 BEV 矩形点云（均匀填充），y 均匀——供聚类/拟合测试。

    局部系 u 沿车头（sin yaw, cos yaw），v 沿右（cos yaw, -sin yaw）。
    """
    rng = rng or np.random.default_rng(42)
    u = rng.uniform(-length / 2, length / 2, n)
    v = rng.uniform(-width / 2, width / 2, n)
    x = cx + u * np.sin(yaw) + v * np.cos(yaw)
    z = cz + u * np.cos(yaw) - v * np.sin(yaw)
    y = rng.uniform(y_lo, y_hi, n)
    return np.stack([x, y, z], axis=1)


def write_frame(
    tmp_path: Path,
    frame_id: str = "000000",
    points: np.ndarray | None = None,
    label_lines: list[str] | None = None,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> KittiFrame:
    """落盘合成 KITTI 帧（calib + velodyne + image_2 + 可选 label_2）→ KittiFrame。"""
    root = Path(tmp_path)
    (root / "training" / "image_2").mkdir(parents=True, exist_ok=True)
    (root / "training" / "velodyne").mkdir(parents=True, exist_ok=True)
    (root / "training" / "calib").mkdir(parents=True, exist_ok=True)
    (root / "training" / "label_2").mkdir(parents=True, exist_ok=True)

    write_calib(root / "training" / "calib" / f"{frame_id}.txt")
    if points is None:
        points = np.zeros((1, 4), dtype=np.float32)
    np.asarray(points, dtype=np.float32).tofile(root / "training" / "velodyne" / f"{frame_id}.bin")
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "training" / "image_2" / f"{frame_id}.png"), img)
    if label_lines:
        (root / "training" / "label_2" / f"{frame_id}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )
    return KittiFrame(frame_id=frame_id, root=root)


def gt_line(
    name: str = "Car",
    truncated: float = 0.0,
    occluded: int = 0,
    x1: float = 500.0,
    y1: float = 150.0,
    x2: float = 700.0,
    y2: float = 300.0,
    h: float = 1.5,
    w: float = 1.6,
    l: float = 3.9,
    x: float = 8.0,
    y: float = -0.9,
    z: float = 18.0,
    ry: float = 0.0,
) -> str:
    """合成 label_2 15 字段行（y 为底部中心，地面附近）。"""
    return (
        f"{name} {truncated:.2f} {occluded} 0.00 "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
    )
