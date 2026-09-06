"""test_cli_nuscenes：nuscenes-queue 子命令薄壳（Fake 注入，零真实权重）。

命令本身 = create_detector3d_any 路由（三引擎统一工厂）+ generate_review_queue
转发；管线逻辑在 test_nuscenes_pipeline.py 已覆盖——此处只验证参数透传与
未知名模型退出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer

from auto3dlabel.agent.planner3d import Plan3D
from auto3dlabel.cli import _run_nuscenes_batch, nuscenes_queue


class _FakeDet:
    class_names = ["car"]


def _patch_torch_cuda(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    """注入假 torch（cuda.is_available 可配置）：_cpu_safe_nuscenes_engine 函数内
    `import torch` 从 sys.modules 取用，测试零真实硬件依赖。"""
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    )


def _patch_queue(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        "auto3dlabel.models.detection3d.create_detector3d_any",
        lambda name: captured.setdefault("name", name) and _FakeDet(),
    )
    monkeypatch.setattr(
        "auto3dlabel.tools.nuscenes_pipeline.generate_review_queue",
        lambda det, out_dir, conf=0.3, dataroot=None, version="v1.0-mini",
        tau_high=0.7, tau_low=0.3, limit=None, seed=42, progress_cb=None:
        captured.update(
            det=det, out_dir=out_dir, conf=conf, dataroot=dataroot, version=version,
            limit=limit,
        ) or {},
    )


def test_nuscenes_queue_forwards_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """det_model 路由 + conf/out_dir/dataroot/version/limit 原样透传 generate_review_queue。"""
    captured: dict[str, Any] = {}
    _patch_queue(monkeypatch, captured)
    out = tmp_path / "reviews"
    nuscenes_queue(det_model="bevfusion", conf=0.5, out_dir=out, dataroot=None,
                   version="v1.0-mini", limit=10)
    assert captured["name"] == "bevfusion"
    assert isinstance(captured["det"], _FakeDet)
    assert captured["conf"] == 0.5
    assert captured["out_dir"] == out
    assert captured["dataroot"] is None and captured["version"] == "v1.0-mini"
    assert captured["limit"] == 10


def test_nuscenes_queue_unknown_model_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知名引擎 → typer.Exit(1)（不裸 traceback）。"""
    monkeypatch.setattr(
        "auto3dlabel.models.detection3d.create_detector3d_any", lambda name: None
    )
    with pytest.raises(typer.Exit):
        nuscenes_queue(det_model="nope")


# ── chat 批量分派（v1.0 P1+）：Plan3D(dataset=nuscenes) → 队列管线 ──

def test_run_nuscenes_batch_forwards_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """chat 分派核心：引擎名 + conf + sample_limit 透传，产物落 out_dir/reviews。"""
    _patch_torch_cuda(monkeypatch, True)  # GPU 语义：不触发降级
    captured: dict[str, Any] = {}
    _patch_queue(monkeypatch, captured)
    plan = Plan3D(
        dataset="nuscenes", sample_limit=100, det_model="bevfusion_nus",
        confidence_threshold=0.4,
    )
    _run_nuscenes_batch(plan, tmp_path / "out")
    assert captured["name"] == "bevfusion_nus"
    assert captured["conf"] == 0.4
    assert captured["limit"] == 100
    assert captured["out_dir"] == tmp_path / "out" / "reviews"


def test_run_nuscenes_batch_cpu_downgrades_cuda_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """纯 CPU + CUDA-only 引擎（bevfusion/spconv op）→ 降级 pointpillars_nus + 提示。"""
    _patch_torch_cuda(monkeypatch, False)
    captured: dict[str, Any] = {}
    _patch_queue(monkeypatch, captured)
    _run_nuscenes_batch(Plan3D(dataset="nuscenes", det_model="bevfusion_nus"),
                        tmp_path / "out")
    assert captured["name"] == "pointpillars_nus", "bevfusion 无 CUDA 必炸 → 降级"
    assert "降级" in capsys.readouterr().out


def test_run_nuscenes_batch_cpu_keeps_cpu_capable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """纯 CPU + CPU 可跑引擎（fcos3d 单目 CNN）→ 原样，无降级提示。"""
    _patch_torch_cuda(monkeypatch, False)
    captured: dict[str, Any] = {}
    _patch_queue(monkeypatch, captured)
    _run_nuscenes_batch(Plan3D(dataset="nuscenes", det_model="fcos3d_nus"),
                        tmp_path / "out")
    assert captured["name"] == "fcos3d_nus"
    assert "降级" not in capsys.readouterr().out


def test_run_nuscenes_batch_unknown_model_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LLM 规划出非 nuscenes 引擎名 → 报可用清单 + typer.Exit(1)。"""
    monkeypatch.setattr(
        "auto3dlabel.models.detection3d.create_detector3d_any", lambda name: None
    )
    with pytest.raises(typer.Exit):
        _run_nuscenes_batch(Plan3D(dataset="nuscenes", det_model="yolo11s.pt"),
                            tmp_path / "out")
