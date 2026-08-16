"""分类结果导出。

每张图像对应一个同名的 .json 文件：
{image_path, width, height, model, labels: [{label, score}]}
model 取自 annotation.metadata["model"]（分类分支写入），缺失时为空串。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto2dlabel.schema.annotation import Annotation


def export_cls(annotations: list[Annotation], output_dir: Path) -> None:
    """将分类结果导出为 JSON（每图一个文件，按图像 stem 命名）。

    同名 stem 冲突时后者覆盖前者（与 yolo/voc 语义一致）。
    score 保留原始浮点精度，不取整。

    Args:
        annotations: Annotation 对象列表（labels 字段为分类结果）。
        output_dir: 输出目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        data: dict[str, Any] = {
            "image_path": ann.image_path,
            "width": ann.image_size[0],
            "height": ann.image_size[1],
            "model": ann.metadata.get("model", ""),
            "labels": [{"label": lab.label, "score": lab.score} for lab in ann.labels],
        }
        stem = Path(ann.image_path).stem or "image"
        out_file = output_dir / f"{stem}.json"
        out_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
