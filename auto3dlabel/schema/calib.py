"""KITTI 标定解析与投影链（红线：标定链任何一环错 → 投影全错且难察觉）。

投影链：velodyne → (Tr_velo_to_cam) cam0 → (R0_rect) rectified cam → (P2) 图像像素。
相机系坐标（3D 拟合/导出坐标系）= rectified cam0：x 右、y 下、z 前。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _parse_3x4(line: str) -> np.ndarray:
    return np.array([float(v) for v in line.split()[1:13]], dtype=np.float64).reshape(3, 4)


def _parse_3x3(line: str) -> np.ndarray:
    return np.array([float(v) for v in line.split()[1:10]], dtype=np.float64).reshape(3, 3)


@dataclass
class KittiCalib:
    """calib_*.txt 解析结果。

    P0-P3 3x4 投影矩阵 / R0_rect 3x3 整流旋转 / Tr_velo_to_cam 3x4 / Tr_imu_to_velo 3x4。
    """

    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray
    Tr_imu_to_velo: np.ndarray
    _velo_to_cam4: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        r = np.eye(4, dtype=np.float64)
        r[:3, :3] = self.R0_rect
        t = np.eye(4, dtype=np.float64)
        t[:3, :] = self.Tr_velo_to_cam
        self._velo_to_cam4 = r @ t  # 4x4：velodyne → rectified cam0

    @classmethod
    def from_file(cls, path: str | Path) -> KittiCalib:
        """逐行解析；P2/R0_rect/Tr_velo_to_cam 缺失抛 ValueError（标定链不完整不可静默）。"""
        lines: dict[str, str] = {}
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            if ":" not in line or not line.strip():
                continue
            key = line.split(":", 1)[0].strip()
            lines[key] = line
        missing = [k for k in ("P2", "R0_rect", "Tr_velo_to_cam") if k not in lines]
        if missing:
            raise ValueError(f"标定文件 {path} 缺少必需行: {missing}")

        def get3x4(key: str) -> np.ndarray:
            return _parse_3x4(lines.get(key, f"{key}: " + "0 " * 11))

        def get3x3(key: str) -> np.ndarray:
            return _parse_3x3(lines.get(key, f"{key}: " + "0 " * 8))

        return cls(
            P0=get3x4("P0"),
            P1=get3x4("P1"),
            P2=get3x4("P2"),
            P3=get3x4("P3"),
            R0_rect=get3x3("R0_rect"),
            Tr_velo_to_cam=get3x4("Tr_velo_to_cam"),
            Tr_imu_to_velo=get3x4("Tr_imu_to_velo"),
        )

    def velo_to_cam_matrix(self) -> np.ndarray:
        """4x4：velodyne → rectified cam0（缓存）。"""
        return self._velo_to_cam4

    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """(N,3)（或 N,4 含 intensity，自动取前 3 列）velodyne 点 → (N,3) rectified cam0。"""
        pts = np.asarray(pts, dtype=np.float64)[:, :3]
        hom = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        return np.asarray(hom @ self._velo_to_cam4.T)[:, :3]

    def project_cam_to_image(
        self, pts_cam: np.ndarray, img_width: int, img_height: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,3) rectified cam0 点 → (u, v, valid)：u/v 像素坐标，valid 为 z>0 且图内掩码。

        图像外的点 u/v 置 0（哨兵），调用方必须按 valid 过滤。
        """
        pts_cam = np.asarray(pts_cam, dtype=np.float64)[:, :3]
        hom = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)])
        img = hom @ self.P2.T  # (N,3)
        z = img[:, 2]
        valid = (z > 0) & (pts_cam[:, 2] > 0)
        u = np.where(z > 0, img[:, 0] / np.maximum(z, 1e-12), 0.0)
        v = np.where(z > 0, img[:, 1] / np.maximum(z, 1e-12), 0.0)
        valid &= (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
        u = np.where(valid, u, 0.0)
        v = np.where(valid, v, 0.0)
        return u.astype(np.int32), v.astype(np.int32), valid

    def project_velo_to_image(
        self, pts: np.ndarray, img_width: int, img_height: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,3) velodyne 点 → (u, v, valid)。

        velodyne → cam0 → P2 → 像素，语义同 project_cam_to_image。
        """
        return self.project_cam_to_image(self.velo_to_cam(pts), img_width, img_height)
