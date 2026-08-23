"""3D HITL 三档分流 + 复核队列文件（协议同 auto2dlabel web 四端点）。

triage_3d：conf 三档（同 2D）+ fit_points < MIN_FIT_POINTS 或 review_flag 强制 review 档——
3D 质量补充判据（LiDAR 观测性红线：点稀疏 → 拟合退化 → 人工兜底）。
队列文件 = 2D 格式 + pcd_path/calib_path/bev_path + annotations 为 Box3D dict（fit_points 恒输出）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from auto3dlabel.configs.kitti import MIN_FIT_POINTS
from auto3dlabel.schema.box3d import Box3D, KittiFrame


@dataclass
class Triage3D:
    """三档分流（Box3D 版）。"""

    accepted: list[Box3D] = field(default_factory=list)
    review: list[Box3D] = field(default_factory=list)
    hard: list[Box3D] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.review) + len(self.hard)


def triage_3d(
    boxes: list[Box3D], tau_high: float = 0.7, tau_low: float = 0.3
) -> Triage3D:
    """conf 三档 + 拟合质量强制 review（fit_points 低 / review_flag 置位）。"""
    result = Triage3D()
    for b in boxes:
        force_review = b.review_flag or b.fit_points < MIN_FIT_POINTS
        if not force_review and b.confidence >= tau_high:
            result.accepted.append(b)
        elif not force_review and b.confidence >= tau_low:
            result.review.append(b)
        else:
            # 低 conf → hard；拟合质量不足 → review（人工能确认/修正，非不可救）
            if force_review and b.confidence >= tau_low:
                result.review.append(b)
            else:
                result.hard.append(b)
    return result


def write_review_queue(
    frame: KittiFrame,
    triage: Triage3D,
    out_dir: str | Path,
) -> Path:
    """帧 → {frame_id}_review.json（仅 review+hard 两档；无待复核框也写出空队列供 Web 扫描）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    annotations = [b.to_dict() for b in triage.review + triage.hard]
    # BEV 鸟瞰图路径（CLI run 的 viz 产物在 out_dir 的父目录；不存在则置空）
    bev = Path(out_dir).parent / f"{frame.frame_id}_bev.png"
    data = {
        "image": frame.frame_id,
        "image_path": str(frame.image_path),
        "image_size": [1242, 375],  # KITTI 标准尺寸（实际 1224×370 校正后；前端仅展示用）
        "pcd_path": str(frame.velodyne_path),
        "calib_path": str(frame.calib_path),
        "bev_path": str(bev) if bev.is_file() else "",
        "description": "3D 复核队列 — 中置信度/拟合质量不足样本需人工确认",
        "annotations": annotations,
        "summary": {
            "accepted": len(triage.accepted),
            "review_count": len(triage.review),
            "hard_count": len(triage.hard),
            "total_need_review": len(triage.review) + len(triage.hard),
        },
    }
    path = out / f"{frame.frame_id}_review.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
