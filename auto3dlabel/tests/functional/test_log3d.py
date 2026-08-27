"""test_log3d：3D 日志层——log_chat_call 3D 聚合键 + 2D 机制 re-export 身份（零真实数据）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto3dlabel.tools import log as log3d


@pytest.fixture()
def logs_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """日志目录 → tmp_path（save() 读 auto2dlabel 模块命名空间里的 LOGS_DIR）。"""
    monkeypatch.setattr("auto2dlabel.tools.log.LOGS_DIR", tmp_path)
    return tmp_path


def _two_steps() -> list[dict[str, object]]:
    return [
        {
            "step_id": "c1",
            "model": "pointpillars_kitti",
            "boxes3d_count": 3,
            "triage_summary": {"accepted": 2, "review": 1, "hard": 0},
        },
        {
            "step_id": "c2",
            "model": "pointpillars_kitti",
            "boxes3d_count": 1,
            "triage_summary": {"accepted": 0, "review": 1, "hard": 0},
        },
    ]


def test_log_chat_call_aggregates_3d_keys(logs_tmp: Path) -> None:
    """聚合：boxes3d_count 求和 + 三档 triage（含 hard）→ result 四键。"""
    path = log3d.log_chat_call(
        instruction="标注 000123 的汽车",
        plan_summary={"frame_id": "000123"},
        steps_results=_two_steps(),
        elapsed=1.5,
        llm_model="deepseek",
    )
    assert path.parent == logs_tmp
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["result"] == {
        "total_boxes3d": 4,
        "total_accepted": 2,
        "total_review": 2,
        "total_hard": 0,
    }
    assert entry["metadata"]["annotation_type_cn"] == "KITTI 3D 标注"
    assert [s["step"] for s in entry["steps"]] == ["plan_parsed", "step_1_done", "step_2_done"]


def test_log_chat_call_empty_and_missing_keys(logs_tmp: Path) -> None:
    """空步骤 → 全零；缺 triage_summary/boxes3d_count 键容错（不抛）。"""
    path = log3d.log_chat_call(
        instruction="x", plan_summary={}, steps_results=[{"step_id": "c1"}], elapsed=0.1
    )
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["result"] == {
        "total_boxes3d": 0,
        "total_accepted": 0,
        "total_review": 0,
        "total_hard": 0,
    }
    empty = log3d.log_chat_call(instruction="x", plan_summary={}, steps_results=[], elapsed=0.1)
    assert json.loads(empty.read_text(encoding="utf-8"))["result"]["total_boxes3d"] == 0


def test_log_chat_call_error_field(logs_tmp: Path) -> None:
    """error 参数写入日志。"""
    path = log3d.log_chat_call(
        instruction="x", plan_summary={}, steps_results=[], elapsed=0.1, error="plan 失败"
    )
    assert json.loads(path.read_text(encoding="utf-8"))["error"] == "plan 失败"


def test_re_export_identity() -> None:
    """re-export 身份：通用机制直通 2D；本地 log_chat_call 是 3D 适配版。"""
    import auto2dlabel.tools.log as log2d

    assert log3d.AnnotationLogger is log2d.AnnotationLogger
    assert log3d.LOGS_DIR is log2d.LOGS_DIR
    assert log3d.log_llm_call is log2d.log_llm_call
    assert log3d.log_python_api_call is log2d.log_python_api_call
    assert log3d.log_chat_call is not log2d.log_chat_call
    assert log3d.ANNOTATION_TYPES["nuscenes_3d"] == "nuScenes 3D 标注"
