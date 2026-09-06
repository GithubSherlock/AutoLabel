"""TUI 斜杠命令注册表单元测试（v1.0 P1+：SlashCommand 声明式集中 + dict 分发）。

铁律：零真实执行——ChatApp 构造后不 mount 不 run，append 与各 handler
monkeypatch 成记录器，只验证分发与 /help 文案生成。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from autolabel.tui.app import (
    HELP_TEXT,
    SLASH_COMMANDS,
    ChatApp,
    SlashCommand,
    build_help_text,
)

_EXPECTED_HANDLERS = {
    "/help": "_cmd_help",
    "/model": "_cmd_model",
    "/cost": "_cmd_cost",
    "/new": "_cmd_new",
    "/resume": "_cmd_resume",
    "/cancel": "_cmd_cancel",
    "/review": "_cmd_review",
    "/web": "_cmd_web",
    "/quit": "_cmd_quit",
}


# ---------- 注册表结构 ----------


def test_registry_has_nine_commands() -> None:
    names = [c.name for c in SLASH_COMMANDS]
    assert names == [
        "/help", "/model", "/cost", "/new", "/resume", "/cancel",
        "/review", "/web", "/quit",
    ]
    assert len(set(names)) == len(names)  # 无重复
    assert all(n.startswith("/") for n in names)


def test_registry_entries_typed_and_callable() -> None:
    for c in SLASH_COMMANDS:
        assert isinstance(c, SlashCommand)
        assert c.description
        assert callable(c.handler)


def test_registry_handlers_are_chatapp_methods() -> None:
    """注册时捕获的是类方法本体（分发经 command.handler(self) 调用）。"""
    assert {c.name: getattr(c.handler, "__name__") for c in SLASH_COMMANDS} == _EXPECTED_HANDLERS


# ---------- /help 文案生成 ----------


def test_build_help_text_format() -> None:
    text = build_help_text()
    lines = text.splitlines()
    assert lines[0] == "斜杠命令："
    for i, c in enumerate(SLASH_COMMANDS):
        assert lines[i + 1] == f"  {c.name:<9}{c.description}"
    assert text.endswith("KITTI 帧 000123 中的汽车和行人」）")
    assert text == HELP_TEXT  # 模块级 HELP_TEXT 与生成器同源


# ---------- dict 分发 ----------


@pytest.fixture
def app_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ChatApp, list[tuple[str, str]], list[str]]:
    """ChatApp + handler (命令, args) 调用记录 + append 文本记录（零 mount / 零执行）。

    分发经模块级 _SLASH_BY_NAME：注入记录型 handler（真实 handler 本体已由
    结构测试断言）——SLASH_COMMANDS 在 import 期已捕获原函数对象，
    monkeypatch 类方法不会影响注册表，须整表注入。
    """
    calls: list[tuple[str, str]] = []
    appended: list[str] = []
    monkeypatch.setattr(
        ChatApp,
        "append",
        lambda self, text, markup=False, **kw: appended.append(text),
    )
    from autolabel.tui import app as app_module

    def _recording_handler(name: str) -> Callable[[ChatApp, str], None]:
        def handler(app: ChatApp, args: str) -> None:
            calls.append((name, args))

        return handler

    monkeypatch.setattr(
        app_module,
        "_SLASH_BY_NAME",
        {
            c.name: SlashCommand(c.name, c.description, _recording_handler(c.name))
            for c in SLASH_COMMANDS
        },
    )
    return ChatApp(runner=None), calls, appended


def _run(
    app: ChatApp, calls: list[tuple[str, str]], appended: list[str], text: str
) -> None:
    app._slash_command(text)


def test_dispatch_each_command(
    app_recorder: tuple[ChatApp, list[tuple[str, str]], list[str]],
) -> None:
    app, calls, appended = app_recorder
    for command in SLASH_COMMANDS:
        _run(app, calls, appended, command.name)
        assert calls[-1] == (command.name, "")  # 无参命令 args 为空串
        assert appended[-1] == f"> {command.name}"


def test_dispatch_passes_args(
    app_recorder: tuple[ChatApp, list[tuple[str, str]], list[str]],
) -> None:
    """P2：args 原样传入 handler（/model set openai → ("/model", "set openai")）。"""
    app, calls, appended = app_recorder
    _run(app, calls, appended, "/model set openai")
    assert calls == [("/model", "set openai")]
    _run(app, calls, appended, "/model set-key")
    assert calls == [("/model", "set openai"), ("/model", "set-key")]


def test_dispatch_unknown_command(
    app_recorder: tuple[ChatApp, list[tuple[str, str]], list[str]],
) -> None:
    app, calls, appended = app_recorder
    _run(app, calls, appended, "/foo")
    assert calls == []
    assert appended == ["> /foo", "未知命令 /foo，输入 /help 查看可用命令"]
