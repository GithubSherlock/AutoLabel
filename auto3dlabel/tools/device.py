"""设备工具（3D）——2D 实现零耦合直接复用（红线「复用不复制」）。

刻意排除批量显存实测三件（measure_single_image_memory / auto_tune_batch_size /
resolve_batch_params）：其探测基于 PIL 图像像素面积 + 2D task_type 静态表，
点云单帧推理无对应语义——3D 场景按需再定义，不误用 2D 口径。
"""

from __future__ import annotations

from auto2dlabel.tools.device import (
    disable_tf32,
    get_device,
    get_device_info,
    get_gpu_free_memory_gb,
    print_device,
    recommend_num_workers,
)

__all__ = [
    "disable_tf32",
    "get_device",
    "get_device_info",
    "get_gpu_free_memory_gb",
    "print_device",
    "recommend_num_workers",
]
