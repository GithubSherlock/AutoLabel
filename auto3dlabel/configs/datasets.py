"""3D 数据集知识单一事实源：内置路径表 + 用户自建注册 + LLM 摘要。

与 auto2dlabel/configs/datasets.py 的关系（复用不复制）：
- 3D 只维护 3D 语义数据集（KITTI object / nuScenes mini——帧号标注的数据根）；
  路径常量单一事实源在 configs/kitti.py（DEFAULT_KITTI_ROOT，env KITTI_OBJECT_ROOT
  覆盖）与 configs/nuscenes.py（DEFAULT_NUSCENES_ROOT）——本模块引用而非重定义
- 用户自建数据集读写复用 auto2dlabel.configs.datasets 的 load/save/register
  （file 参数化），3D 写独立 yaml（configs/user_datasets.yaml，不入库）——
  2D/3D 自建语义不同（2D=图像目录；3D=须符 KITTI 目录结构的帧根）不混用
"""

from __future__ import annotations

import os
from pathlib import Path

from auto2dlabel.configs.datasets import (
    DatasetInfo,
)
from auto2dlabel.configs.datasets import (
    load_user_datasets as load_user_datasets,  # re-export（mypy strict 显式导出）
)
from auto2dlabel.configs.datasets import (
    register_user_dataset as _register_user_dataset,
)
from auto2dlabel.configs.datasets import (
    remove_user_dataset as _remove_user_dataset,
)
from auto3dlabel.configs.kitti import DEFAULT_KITTI_ROOT
from auto3dlabel.configs.nuscenes import DEFAULT_NUSCENES_ROOT

# 用户自建 3D 数据集配置（不入库，.gitignore——与权重/数据不入库红线一致）
USER_DATASETS_FILE = Path(__file__).resolve().parent / "user_datasets.yaml"

# 数据集名 → env 覆盖变量（与 configs/{kitti,nuscenes}.py 常量同语义，运行时取当前值）
DATASET_ENV: dict[str, str] = {"kitti": "KITTI_OBJECT_ROOT", "nuscenes_mini": "NUSCENES_ROOT"}

# 内置 3D 数据集（路径引用 configs 常量——单一事实源，改一处不漏另一处）
DATASET_DIRS: dict[str, DatasetInfo] = {
    "kitti": DatasetInfo(
        DEFAULT_KITTI_ROOT,
        ("training", "testing"),
        "KITTI object 3D 检测标注（帧号 → training/{calib,image_2,label_2,velodyne}）",
        "env KITTI_OBJECT_ROOT 可覆盖",
    ),
    "nuscenes_mini": DatasetInfo(
        DEFAULT_NUSCENES_ROOT,
        ("v1.0-mini", "samples", "sweeps"),
        "nuScenes mini 3D 检测冒烟（centerpoint_nus 等 LiDAR 引擎）",
        "env NUSCENES_ROOT 可覆盖",
    ),
}


def resolve_dataset_dir(name: str) -> Path:
    """解析内置 3D 数据集目录：env 覆盖 > configs 常量默认路径。"""
    if name not in DATASET_DIRS:
        raise KeyError(f"未知数据集: {name}（可选: {', '.join(DATASET_DIRS)}）")
    env = DATASET_ENV.get(name)
    if env:
        return Path(os.environ.get(env, str(DATASET_DIRS[name].path)))
    return Path(DATASET_DIRS[name].path)


def register_user_dataset(
    name: str,
    path: str,
    *,
    subdirs: list[str] | None = None,
    task: str = "",
    note: str = "",
) -> DatasetInfo:
    """注册自建 3D 数据集 → configs/user_datasets.yaml（复用 2D 读写，独立文件）。

    3D 自建数据集须符 KITTI 目录结构（training/{calib,image_2,label_2,velodyne}），
    摘要中注明供 LLM/用户知悉。

    Raises:
        ValueError: name 为空或与内置 3D 数据集重名；path 不存在。
    """
    name = name.strip()
    if not name:
        raise ValueError("数据集名不能为空")
    if name in DATASET_DIRS:
        raise ValueError(f"数据集名 {name} 与内置 3D 数据集重名（内置: {', '.join(DATASET_DIRS)}）")
    return _register_user_dataset(
        name, path, subdirs=subdirs, task=task, note=note, file=USER_DATASETS_FILE,
    )


def remove_user_dataset(name: str) -> bool:
    """删除自建 3D 数据集。Returns: 是否存在并已删除。"""
    return _remove_user_dataset(name, file=USER_DATASETS_FILE)


def format_datasets_summary() -> str:
    """3D 数据集速查摘要（内置 + 自建合并，注入 planner3d prompt）。

    3D 的 source（帧根目录）由 CLI 决定而非 LLM 解析路径；摘要供 LLM 识别
    指令中的数据集语义（「KITTI 数据集」→ 帧号标注；自建数据集名 → 可用性）。
    """
    lines: list[str] = []
    user = load_user_datasets(USER_DATASETS_FILE)
    if user:
        lines.append("用户自建 3D 数据集（指令提及这些名字 → 该数据集可用；须符 KITTI 目录结构）：")
        for name, info in user.items():
            extra = f"（子目录: {', '.join(info.subdirs)}）" if info.subdirs else ""
            task_s = f" — {info.task}" if info.task else ""
            note_s = f"（{info.note}）" if info.note else ""
            lines.append(f"- {name}: {info.path}{extra}{task_s}{note_s}")
    lines.append("内置数据集（指令含数据集名时按以下语义理解）：")
    for name, info in DATASET_DIRS.items():
        path = resolve_dataset_dir(name)
        extra = f"（子目录: {', '.join(info.subdirs)}）" if info.subdirs else ""
        env = DATASET_ENV.get(name)
        env_s = f"（env {env} 可覆盖）" if env else ""
        task_s = f" — {info.task}" if info.task else ""
        lines.append(f"- {name}: {path}{extra}{task_s}{env_s}")
    return "\n".join(lines)
