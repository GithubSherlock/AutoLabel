"""test_cli_nuscenes：nuscenes-queue 子命令薄壳（Fake 注入，零真实权重）。

命令本身 = create_detector3d_any 路由（三引擎统一工厂）+ generate_review_queue
转发；管线逻辑在 test_nuscenes_pipeline.py 已覆盖——此处只验证参数透传与
未知名模型退出。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from auto3dlabel.cli import nuscenes_queue


class _FakeDet:
    class_names = ["car"]


def test_nuscenes_queue_forwards_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """det_model 路由 + conf/out_dir/dataroot/version 原样透传 generate_review_queue。"""
    captured: dict = {}

    def fake_create(name: str) -> _FakeDet:
        captured["name"] = name
        return _FakeDet()

    def fake_generate(
        det: object, out_dir: Path, conf: float, dataroot: str | None, version: str
    ) -> dict[str, Path]:
        captured.update(
            det=det, out_dir=out_dir, conf=conf, dataroot=dataroot, version=version
        )
        return {}

    monkeypatch.setattr(
        "auto3dlabel.models.detection3d.create_detector3d_any", fake_create
    )
    monkeypatch.setattr(
        "auto3dlabel.tools.nuscenes_pipeline.generate_review_queue", fake_generate
    )
    out = tmp_path / "reviews"
    nuscenes_queue(det_model="bevfusion", conf=0.5, out_dir=out, dataroot=None,
                   version="v1.0-mini")
    assert captured["name"] == "bevfusion"
    assert isinstance(captured["det"], _FakeDet)
    assert captured["conf"] == 0.5
    assert captured["out_dir"] == out
    assert captured["dataroot"] is None and captured["version"] == "v1.0-mini"


def test_nuscenes_queue_unknown_model_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知名引擎 → typer.Exit(1)（不裸 traceback）。"""
    monkeypatch.setattr(
        "auto3dlabel.models.detection3d.create_detector3d_any", lambda name: None
    )
    with pytest.raises(typer.Exit):
        nuscenes_queue(det_model="nope")
