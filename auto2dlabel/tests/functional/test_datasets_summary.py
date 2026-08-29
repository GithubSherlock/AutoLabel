"""方案 A：数据集路径引导（DATASET_DIRS / resolve_dataset_dir / format_datasets_summary）。

LLM 路径查询的单一事实源测试：
- 路径表与 planner 的 BENCHMARK_DATASETS key 对齐（新增 benchmark 数据集必须同步）
- env 覆盖语义与 auto3dlabel/configs/{kitti,nuscenes}.py 一致
- 摘要输出可直接注入 planner prompt（含数据集名/路径/任务）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto2dlabel.benchmarks.datasets import (
    DATASET_DIRS,
    DATASET_ENV,
    format_datasets_summary,
    resolve_dataset_dir,
)
from auto2dlabel.schema.task_plan import BENCHMARK_DATASETS


def test_dataset_dirs_covers_benchmark_keys() -> None:
    """planner 校验用的 BENCHMARK_DATASETS 每个 key 都必须在路径表中（或同目录合并）。"""
    # dota_obb 与 dota 同目录（labels_obb 子目录）、coco_seg 与 coco 同目录——已在 note 说明
    merged = {"dota_obb": "dota", "coco_seg": "coco"}
    for key in BENCHMARK_DATASETS:
        assert key in DATASET_DIRS or key in merged, f"路径表缺 {key}"
        target = merged.get(key, key)
        assert target in DATASET_DIRS


def test_resolve_dataset_dir_default() -> None:
    """默认路径 = DATASET_DIRS 表值（绝对路径）。"""
    for name, info in DATASET_DIRS.items():
        assert resolve_dataset_dir(name) == info.path
        assert info.path.is_absolute()


def test_resolve_dataset_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """env 覆盖（KITTI_OBJECT_ROOT / NUSCENES_ROOT）优先于默认路径。"""
    assert DATASET_ENV == {"kitti": "KITTI_OBJECT_ROOT", "nuscenes_mini": "NUSCENES_ROOT"}
    monkeypatch.setenv("KITTI_OBJECT_ROOT", "/tmp/fake_kitti")
    monkeypatch.setenv("NUSCENES_ROOT", "/tmp/fake_nus")
    assert resolve_dataset_dir("kitti") == Path("/tmp/fake_kitti")
    assert resolve_dataset_dir("nuscenes_mini") == Path("/tmp/fake_nus")
    # 无 env 的数据集不受影响
    assert resolve_dataset_dir("coco") == DATASET_DIRS["coco"].path


def test_resolve_dataset_dir_unknown() -> None:
    """未知数据集名 → 明确 KeyError（列可选名）。"""
    with pytest.raises(KeyError, match="未知数据集"):
        resolve_dataset_dir("not_a_dataset")


def test_format_datasets_summary_contains_all() -> None:
    """摘要含全部数据集：名 / 绝对路径 / 任务描述。"""
    summary = format_datasets_summary()
    for name, info in DATASET_DIRS.items():
        assert name in summary
        assert str(info.path) in summary
        assert info.task in summary


def test_format_datasets_summary_env_reflects_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要运行时取 env 当前值——LLM 不会拿到过期路径。"""
    monkeypatch.setenv("KITTI_OBJECT_ROOT", "/tmp/fake_kitti")
    summary = format_datasets_summary()
    assert "/tmp/fake_kitti" in summary
    assert "KITTI_OBJECT_ROOT" in summary  # env 名可见，LLM 可感知覆盖机制


def test_format_datasets_summary_subdirs() -> None:
    """关键子目录渲染（LLM 靠它定位图像目录）。"""
    summary = format_datasets_summary()
    assert "val2017" in summary
    assert "training" in summary  # kitti
    assert "v1.0-mini" in summary  # nuimages / nuscenes_mini
