"""autolabel TUI 骨架冒烟（v1.0 P1，Textual run_test + Pilot）。

对标 jsdom 冒烟铁律：零真实执行（Fake runner 注入）、零真实 LLM 调用
（凭据横幅按 monkeypatch 判定）。Textual 8.x 坑（docs/Agentic_UI_plan.md
§8.3）：断言统一走 export_screenshot() 且用无空格片段（SVG 空格编码为
&#160;）；长内容滚动出可视区时改查组件树（chat children）。
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.containers import VerticalScroll

from autolabel.tui import app as app_module
from autolabel.tui.app import ChatApp, _subprocess_run


@pytest.fixture(autouse=True)
def _session_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P4 会话日志重定向 tmp_path（防测试写真实 logs/）。"""
    monkeypatch.setattr(app_module, "SESSION_LOG_PATH", tmp_path / "chat_sessions.jsonl")


def _fake_runner(
    collected: list[tuple[str, str, str, list[str]]],
) -> Callable[[str, str, str, Callable[[str], None], Callable[[int, int | None], None]], None]:
    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        lines = [f"路由 {domain} 执行: {instruction}", "检测完成: 7 框"]
        collected.append((instruction, domain, provider, lines))
        on_progress(3, 10)  # P2 契约：runner 经 on_progress 上报（面板 P3 消费）
        for line in lines:
            on_line(line)

    return runner


def test_tui_slash_commands_flow() -> None:
    """P1 验收骨架：启动横幅 + /help /cost /new /quit 全链路（无空格片段断言）。"""
    collected: list[tuple[str, str, str, list[str]]] = []

    async def main() -> None:
        app = ChatApp(runner=_fake_runner(collected))
        async with app.run_test(size=(100, 40)) as pilot:
            svg = app.export_screenshot()
            assert "AutoLabel" in svg, "标题缺失"
            assert "Agentic" in svg, "欢迎语缺失"
            assert "DeepSeek" in svg, "凭据横幅缺失"
            # /help
            await pilot.press(*"/help", "enter")
            await pilot.pause()
            svg = app.export_screenshot()
            assert "/help" in svg, "help 命令列表缺失"
            assert "/cost" in svg, "help 未列 /cost"
            # /model
            await pilot.press(*"/model", "enter")
            await pilot.pause()
            svg = app.export_screenshot()
            assert "provider:" in svg, "model 显示缺失"
            # /cost（空台账路径）
            await pilot.press(*"/cost", "enter")
            await pilot.pause()
            # /new 清空
            await pilot.press(*"/new", "enter")
            await pilot.pause()
            # /quit 退出（run_test 上下文内 exit 即测试结束信号）
            await pilot.press(*"/quit", "enter")

    asyncio.run(main())


def test_tui_instruction_runs_in_worker_and_echoes() -> None:
    """指令 → worker 执行 Fake runner → 行回显 + 完成行（组件树断言）。"""
    collected: list[tuple[str, str, str, list[str]]] = []

    async def main() -> None:
        app = ChatApp(runner=_fake_runner(collected))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press(*"检测汽车", "enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if collected:
                    break
            # runner 收到指令 + 2D 路由 + provider 透传
            assert collected and collected[0][0] == "检测汽车"
            assert collected[0][1] == "2d"
            assert collected[0][2] == "deepseek"
            # 等回显完成（busy 置 False）
            for _ in range(100):
                await pilot.pause(0.05)
                if not app._busy:
                    break
            chat = app.query_one("#chat", VerticalScroll)
            texts = [
                getattr(k, "content", "")
                for k in chat.children
                if hasattr(k, "content")
            ]
            assert "检测完成: 7 框" in texts, f"回显缺失: {texts}"
            assert "✓ 任务 #1 执行完成" in texts, f"完成行缺失: {texts}"

    asyncio.run(main())


def test_tui_instruction_routes_3d_by_frame_id() -> None:
    """3D 强特征指令（KITTI 帧号）→ runner 收到 domain="3d"。"""
    collected: list[tuple[str, str, str, list[str]]] = []

    async def main() -> None:
        app = ChatApp(runner=_fake_runner(collected))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press(*"标注 KITTI 帧 000123 中的汽车和行人", "enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if collected:
                    break
            assert collected and collected[0][1] == "3d", f"路由错误: {collected}"

    asyncio.run(main())


def test_tui_no_credentials_shows_code_fallback_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 key 降级（v0.6 红线在 TUI 保持）：横幅显示「代码直跑」提示。"""
    from auto2dlabel.agent import llm as llm_mod

    class _NoKeyClient:
        model = "deepseek-chat"

        @property
        def has_credentials(self) -> bool:
            return False

    monkeypatch.setattr(llm_mod, "create_client", lambda provider="openai", **kw: _NoKeyClient())

    async def main() -> None:
        app = ChatApp(runner=_fake_runner([]))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            svg = app.export_screenshot()
            # 无空格片段断言（SVG 空格编码为 &#160;，§8.3 坑 4）
            assert "凭据" in svg, "凭据状态缺失"
            assert "代码直跑降级" in svg, "无 key 降级提示缺失"

    asyncio.run(main())


def test_subprocess_run_uses_unbuffered_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """子进程必须 `-u` 无缓冲——stdout 连 pipe 时 Python 默认全缓冲，输出积压到
    进程退出才 flush，TUI 表现为「无响应」（2026-09-02 实测回归，勿删）。"""
    captured: dict[str, list[str]] = {}

    class _FakeProc:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd

        @property
        def stdout(self) -> object:
            return iter([])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    lines: list[str] = []
    progress: list[tuple[int, int | None]] = []
    _subprocess_run(
        "检测汽车", "2d", "deepseek", lines.append, lambda d, t: progress.append((d, t)),
    )
    cmd = captured["cmd"]
    assert cmd[1] == "-u", f"子进程缺 -u 无缓冲标志: {cmd}"
    assert cmd[2] == "-m" and cmd[3] == "auto2dlabel.cli"
    assert "--no-wait" in cmd
    assert "--provider" in cmd and "deepseek" in cmd  # P2：provider 透传子进程
