"""model_catalog3d 收拢回归（v0.3 P6）：路由表单一事实源 + planner3d summary 注入。

照 auto2dlabel/model_catalog 母版：4 张路由表收拢至 configs/model_catalog.py，
消费方（models/* / tests）全部 import 引用；planner3d prompt 经
format_catalog_summary3d() 注入（新增模型零改 prompt 文本）。
护栏：summary 与表同步、旧散副本（kitti.py/nuscenes.py）不复存、prompt 注入不断开。
"""

from __future__ import annotations

import pytest

from auto3dlabel.configs import kitti as kitti_cfg
from auto3dlabel.configs import nuscenes as nuscenes_cfg
from auto3dlabel.configs.model_catalog import (
    _CATALOG_SUMMARY_GROUPS,
    BEVFUSION_NAMES,
    DETECTOR3D_NAMES,
    FCOS3D_NAMES,
    MONO3D_NAMES,
    format_catalog_summary3d,
)

ALL_TABLES: tuple[tuple[str, dict[str, dict]], ...] = (
    ("DETECTOR3D_NAMES", DETECTOR3D_NAMES),
    ("MONO3D_NAMES", MONO3D_NAMES),
    ("BEVFUSION_NAMES", BEVFUSION_NAMES),
    ("FCOS3D_NAMES", FCOS3D_NAMES),
)


@pytest.mark.parametrize("table_name,table", ALL_TABLES)
def test_catalog_entries_fields(table_name: str, table: dict[str, dict]) -> None:
    """每张表每条目字段齐全：config 以 configs/ 或 projects/ 开头、
    checkpoint .pth、weights_dir 非空。"""
    assert table, f"{table_name} 不得为空表"
    for name, entry in table.items():
        assert set(entry) == {"config", "checkpoint", "weights_dir"}, f"{name} 字段异常"
        assert entry["config"].startswith(("configs/", "projects/")), f"{name} config 路径"
        assert entry["checkpoint"].endswith(".pth"), f"{name} checkpoint"
        assert entry["weights_dir"], f"{name} weights_dir 非空"


def test_catalog_keys_disjoint() -> None:
    """4 表 key 两两不相交（协议纪律：同一模型名不得横跨 detect/detect_sample 两张表）。"""
    tables = [table for _, table in ALL_TABLES]
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            overlap = set(tables[i]) & set(tables[j])
            assert not overlap, f"模型名 {overlap} 出现在两张协议不同的表"


def test_summary_contains_all_engine_names() -> None:
    """summary 覆盖全部 9 个引擎名（防回退成硬编码 3 条 LiDAR 清单）。"""
    text = format_catalog_summary3d()
    for table in (DETECTOR3D_NAMES, MONO3D_NAMES, BEVFUSION_NAMES, FCOS3D_NAMES):
        for name in table:
            assert name in text, f"summary 缺失引擎 {name}"


def test_summary_groups_cover_tables() -> None:
    """_CATALOG_SUMMARY_GROUPS 模型名并集 == 4 表 key 并集（新增模型必须进 groups）。"""
    grouped = {name for _, names, _ in _CATALOG_SUMMARY_GROUPS for name in names}
    all_names = {
        name
        for table in (DETECTOR3D_NAMES, MONO3D_NAMES, BEVFUSION_NAMES, FCOS3D_NAMES)
        for name in table
    }
    assert grouped == all_names


def test_planner3d_injects_catalog_summary() -> None:
    """planner3d prompt 含 summary 全文（注入不断开；新增模型随目录自动可见）。"""
    from auto3dlabel.agent.planner3d import _PLANNER3D_SYSTEM_PROMPT

    assert format_catalog_summary3d() in _PLANNER3D_SYSTEM_PROMPT


def test_consumers_share_single_source() -> None:
    """消费方（模型封装）持有 model_catalog 的同一表对象（引用单一事实源）。"""
    from auto3dlabel.models import bevfusion3d, detection3d, mono3d

    assert getattr(detection3d, "DETECTOR3D_NAMES") is DETECTOR3D_NAMES
    assert getattr(mono3d, "MONO3D_NAMES") is MONO3D_NAMES
    assert getattr(mono3d, "FCOS3D_NAMES") is FCOS3D_NAMES
    assert getattr(bevfusion3d, "BEVFUSION_NAMES") is BEVFUSION_NAMES


def test_no_stale_copies_in_old_configs() -> None:
    """旧散副本不复存（kitti.py / nuscenes.py 已删表留指针——防复制回流）。"""
    for module in (kitti_cfg, nuscenes_cfg):
        for attr in ("DETECTOR3D_NAMES", "MONO3D_NAMES", "BEVFUSION_NAMES", "FCOS3D_NAMES"):
            assert not hasattr(module, attr), f"{module.__name__}.{attr} 散副本回流"
