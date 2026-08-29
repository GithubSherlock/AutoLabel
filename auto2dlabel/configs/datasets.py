"""数据集知识单一事实源：内置路径表 + 用户自建数据集注册 + LLM 路径引导摘要。

职责边界（2026-08-29 重组）：
- 本模块 = 数据集「知识」（路径表 / 自建注册 / 摘要），服务 chat/planner 路径引导
- benchmarks/datasets.py = 数据集「操作」（ensure_* 解压 / GT 解析，benchmark 专用），
  常量从本模块 import——两侧路径同源，改一处不漏另一处

自建数据集：`auto2dlabel dataset add <name> <path>` 注册到 USER_DATASETS_FILE
（configs/user_datasets.yaml，不入库——与权重/数据不入库红线一致）；
format_datasets_summary 合并内置 + 自建，LLM 从 chat 指令识别数据集名 → 绝对路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATASETS_ROOT = Path.home() / "autodl-tmp" / "Documents" / "datasets"
ARCHIVE_ROOT = Path("/root/autodl-pub")

# 用户自建数据集配置文件（不入库，.gitignore）
USER_DATASETS_FILE = Path(__file__).resolve().parent / "user_datasets.yaml"


@dataclass(frozen=True)
class DatasetInfo:
    """单个数据集：路径 / 关键子目录 / 任务描述（供 LLM 路径引导摘要）。"""

    path: Path
    subdirs: tuple[str, ...] = ()
    task: str = ""
    note: str = ""


# 内置数据集（key 与 schema/task_plan.BENCHMARK_DATASETS 对齐，planner 校验同源；
# path 与 benchmarks/datasets.py ensure_* 解压目标一致——改动需同步）
DATASET_DIRS: dict[str, DatasetInfo] = {
    "coco": DatasetInfo(
        DATASETS_ROOT / "COCO2017", ("annotations", "val2017"),
        "目标检测 + 实例分割（80 类，金标准）",
        "coco_seg 分割标注同目录 instances_val2017.json",
    ),
    "voc2007": DatasetInfo(
        DATASETS_ROOT / "VOCdevkit" / "VOC2007", ("JPEGImages", "Annotations"),
        "经典目标检测（20 类）",
    ),
    "kitti": DatasetInfo(
        DATASETS_ROOT / "KITTI" / "object", ("training", "testing"),
        "自动驾驶 2D/3D 检测（8 类）",
        "training/{calib,image_2,label_2,velodyne}；KITTI/yolo = 微调产物",
    ),
    "dota": DatasetInfo(
        DATASETS_ROOT / "DOTA", ("images", "labels", "labels_obb"),
        "航拍目标检测（水平框）+ OBB 旋转框（15 类）",
        "labels = 水平框，labels_obb = 旋转框（dota_obb 用）",
    ),
    "mot": DatasetInfo(
        DATASETS_ROOT / "MOT17", ("train", "test"),
        "多目标跟踪（14 序列 × 3 检测器）",
        "train/MOT17-XX-FRCNN；mot20 在 MOT20/train",
    ),
    "cityscapes": DatasetInfo(
        DATASETS_ROOT / "cityscapes", ("gtFine", "leftImg8bit"),
        "城市场景语义/实例分割（域内 Mask R-CNN）",
    ),
    "nuimages": DatasetInfo(
        DATASETS_ROOT / "nuImages", ("samples", "sweeps", "v1.0-mini"),
        "2D 实例分割 mini（nuScenes 图像子集）",
    ),
    "d2sa": DatasetInfo(
        DATASETS_ROOT / "D2SA", ("annotations", "images"),
        "密集零售货架商品检测（SKU 级）",
    ),
    "imagenet100": DatasetInfo(
        DATASETS_ROOT / "imagenet100", (),
        "图像分类（ImageNet 100 类子集）",
        "100 个 wnid 目录即标签",
    ),
    "imagenet1k": DatasetInfo(
        DATASETS_ROOT / "imagenet1k", ("val",),
        "图像分类（ILSVRC2012，1000 类）",
    ),
    "nuscenes_mini": DatasetInfo(
        DATASETS_ROOT / "nuscenes_mini", ("maps", "samples", "sweeps", "v1.0-mini"),
        "3D 检测冒烟（pointpillars_nus，10 类）",
    ),
}

# 数据集名 → env 覆盖变量（与 auto3dlabel/configs 同语义，运行时取当前值）
DATASET_ENV: dict[str, str] = {"kitti": "KITTI_OBJECT_ROOT", "nuscenes_mini": "NUSCENES_ROOT"}


def resolve_dataset_dir(name: str) -> Path:
    """解析内置数据集目录：env 覆盖 > DATASET_DIRS 默认路径。"""
    if name not in DATASET_DIRS:
        raise KeyError(f"未知数据集: {name}（可选: {', '.join(DATASET_DIRS)}）")
    env = DATASET_ENV.get(name)
    if env:
        return Path(os.environ.get(env, str(DATASET_DIRS[name].path)))
    return DATASET_DIRS[name].path


# ================================================================
# 用户自建数据集注册（configs/user_datasets.yaml）
# ================================================================

_USER_DATASETS_KEY = "datasets"


def load_user_datasets(file: Path | None = None) -> dict[str, DatasetInfo]:
    """读取自建数据集配置 → {name: DatasetInfo}（文件缺失/损坏 → {}，不炸 chat）。"""
    import yaml

    file = USER_DATASETS_FILE if file is None else file
    if not file.exists():
        return {}
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    out: dict[str, DatasetInfo] = {}
    for name, entry in (raw.get(_USER_DATASETS_KEY) or {}).items():
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        if not path:
            continue
        subdirs = tuple(s for s in entry.get("subdirs", []) if isinstance(s, str))
        out[str(name)] = DatasetInfo(
            Path(path).expanduser(),
            subdirs,
            str(entry.get("task", "")).strip(),
            str(entry.get("note", "")).strip(),
        )
    return out


def save_user_datasets(datasets: dict[str, DatasetInfo], file: Path | None = None) -> None:
    """写回自建数据集配置（yaml，确定排序，文件缺失自动建目录）。"""
    import yaml

    file = USER_DATASETS_FILE if file is None else file
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _USER_DATASETS_KEY: {
            name: {
                "path": str(info.path),
                "subdirs": list(info.subdirs),
                "task": info.task,
                "note": info.note,
            }
            for name, info in sorted(datasets.items())
        }
    }
    file.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def register_user_dataset(
    name: str,
    path: str,
    *,
    subdirs: list[str] | None = None,
    task: str = "",
    note: str = "",
    file: Path | None = None,
) -> DatasetInfo:
    """注册/覆盖自建数据集（path 必须存在——防 LLM 拿到假路径）。

    Raises:
        ValueError: name 为空或与内置数据集重名；path 不存在。
    """
    name = name.strip()
    if not name:
        raise ValueError("数据集名不能为空")
    if name in DATASET_DIRS:
        raise ValueError(f"数据集名 {name} 与内置数据集重名（内置: {', '.join(DATASET_DIRS)}）")
    p = Path(path).expanduser()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"路径不存在或不是目录: {p}")
    info = DatasetInfo(p, tuple(subdirs or []), task, note)
    datasets = load_user_datasets(file)
    datasets[name] = info
    save_user_datasets(datasets, file)
    return info


def remove_user_dataset(name: str, file: Path | None = None) -> bool:
    """删除自建数据集。Returns: 是否存在并已删除。"""
    datasets = load_user_datasets(file)
    if name not in datasets:
        return False
    del datasets[name]
    save_user_datasets(datasets, file)
    return True


# ================================================================
# LLM 路径引导摘要（planner prompt 注入）
# ================================================================


def format_datasets_summary() -> str:
    """数据集路径速查摘要（内置 + 自建合并，注入 planner prompt，运行时取当前值）。

    用途：LLM 把指令中的数据集名映射为真实 source 目录（如「检测 COCO2017
    验证集」→ /root/autodl-tmp/Documents/datasets/COCO2017/val2017；
    「我的自建数据集 xxx」→ 注册时填写的路径）。
    """
    lines: list[str] = []
    user = load_user_datasets()
    if user:
        lines.append("用户自建数据集（指令提及这些名字 → 用注册路径）：")
        for name, info in user.items():
            extra = f"（子目录: {', '.join(info.subdirs)}）" if info.subdirs else ""
            task_s = f" — {info.task}" if info.task else ""
            note_s = f"（{info.note}）" if info.note else ""
            lines.append(f"- {name}: {info.path}{extra}{task_s}{note_s}")
    lines.append("内置数据集（指令含数据集名时，用下路径构造 source 目录）：")
    for name, info in DATASET_DIRS.items():
        path = resolve_dataset_dir(name)
        extra = f"（子目录: {', '.join(info.subdirs)}）" if info.subdirs else ""
        env = DATASET_ENV.get(name)
        env_s = f"（env {env} 可覆盖）" if env else ""
        task_s = f" — {info.task}" if info.task else ""
        lines.append(f"- {name}: {path}{extra}{task_s}{env_s}")
    return "\n".join(lines)
