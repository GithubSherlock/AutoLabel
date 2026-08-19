"""MOT16/20 导出 —— 跟踪结果 txt（frame, id, x, y, w, h, conf, class, visibility）。

MOTChallenge 标准格式，每行一个跟踪框：
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
- frame 从 1 开始（输入 Annotation 顺序即帧序）
- conf 为检测置信度；class 恒 -1（MOT 类目未映射）；visibility 恒 1
- 无 track_id 的 bbox 跳过（MOT 格式无位置容纳未跟踪检测）
"""

from __future__ import annotations

from auto2dlabel.schema.annotation import Annotation
from auto2dlabel.tools import Path


def export_mot(annotations: list[Annotation], output_path: Path) -> None:
    """将带 track_id 的逐帧标注导出为 MOT16/20 格式 txt。

    Args:
        annotations: 逐帧 Annotation（顺序即帧序）。
        output_path: 输出 txt 文件路径。
    """
    lines: list[str] = []
    for frame_idx, ann in enumerate(annotations, start=1):
        for b in ann.bboxes:
            if b.track_id is None:
                continue
            lines.append(
                f"{frame_idx},{b.track_id},{b.x:.1f},{b.y:.1f},"
                f"{b.width:.1f},{b.height:.1f},{b.confidence:.6f},-1,1"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
