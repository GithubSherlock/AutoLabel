"""KITTI 数据 IO：帧解析/路径规范化（懒加载在 KittiFrame 上，此处为批量与目录工具）。"""

from __future__ import annotations

from pathlib import Path

from auto3dlabel.configs.kitti import DEFAULT_KITTI_ROOT
from auto3dlabel.schema.box3d import KittiFrame


def normalize_frame_id(frame_id: str) -> str:
    """'123' → '000123'；已 6 位原样返回。"""
    fid = frame_id.strip()
    if not fid.isdigit():
        raise ValueError(f"非法帧 ID（须为数字）: {frame_id!r}")
    return f"{int(fid):06d}"


def resolve_frame(frame_id: str, root: Path = DEFAULT_KITTI_ROOT) -> KittiFrame:
    """帧 ID（'123'/'000123'）→ KittiFrame（校验文件存在）。"""
    frame = KittiFrame(frame_id=normalize_frame_id(frame_id), root=root)
    if not frame.image_path.is_file():
        raise FileNotFoundError(f"KITTI 帧不存在: {frame.image_path}")
    return frame


def frame_ids_from_dir(
    image_dir: str | Path, root: Path = DEFAULT_KITTI_ROOT, max_frames: int = 0
) -> list[str]:
    """image_2 目录 → 帧 ID 列表（排序；max_frames>0 截断前 N）。"""
    d = Path(image_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {d}")
    ids = sorted(p.stem for p in d.glob("*.png") if p.stem.isdigit())
    if max_frames > 0:
        ids = ids[:max_frames]
    return ids


def frame_ids_by_range(start: int, end: int, max_frames: int = 0) -> list[str]:
    """[start, end) 帧号区间（含 start 不含 end），零填充。"""
    ids = [f"{i:06d}" for i in range(start, end)]
    if max_frames > 0:
        ids = ids[:max_frames]
    return ids
