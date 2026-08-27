"""test_tools_reuse：3D tools 复用层身份断言——re-export 直通 2D（is 同一对象），
2D 图像口径三件刻意不泄漏（点云场景无对应语义）。"""

from __future__ import annotations

import auto2dlabel.tools.device as dev2d
import auto2dlabel.tools.evaluate as ev2d
import auto3dlabel.tools.device as dev3d
import auto3dlabel.tools.evaluate as ev3d

DEVICE_NAMES = [
    "disable_tf32",
    "get_device",
    "get_device_info",
    "get_gpu_free_memory_gb",
    "print_device",
    "recommend_num_workers",
]
EXCLUDED_NAMES = ["measure_single_image_memory", "auto_tune_batch_size", "resolve_batch_params"]


def test_device_re_exports_are_identical() -> None:
    """六个通用设备函数与 2D 实现同一对象（复用不复制红线）。"""
    for name in DEVICE_NAMES:
        assert getattr(dev3d, name) is getattr(dev2d, name), name
    assert set(dev3d.__all__) == set(DEVICE_NAMES)


def test_device_excluded_names() -> None:
    """2D 批量显存实测三件不泄漏进 3D（PIL 图像探针口径，点云不适用）。"""
    for name in EXCLUDED_NAMES:
        assert hasattr(dev2d, name)  # 前提：2D 确实存在（防断言空转）
        assert not hasattr(dev3d, name), name


def test_evaluate_tool_identity() -> None:
    """EvaluateTool 本体零 2D 耦合，直通复用（3D 契约在 orchestrator3d 注入）。"""
    assert ev3d.EvaluateTool is ev2d.EvaluateTool
