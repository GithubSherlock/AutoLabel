"""test_cli_install_guard：缺 auto2dlabel 包时 run/chat 入口给安装指引而非裸 ImportError。"""

from __future__ import annotations

import sys

import pytest
import typer

from auto3dlabel import cli


def test_missing_auto2dlabel_exits_with_guidance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """sys.modules 置 None 模拟缺包：typer.Exit(1) + 指引文案（含 pip install -e .）。"""
    monkeypatch.setitem(sys.modules, "auto2dlabel", None)
    with pytest.raises(typer.Exit) as exc:
        cli._require_auto2dlabel()
    assert exc.value.exit_code == 1
    assert "pip install -e ." in capsys.readouterr().out


def test_guard_passes_when_installed() -> None:
    """正常安装（本测试环境）时护栏静默通过。"""
    cli._require_auto2dlabel()
