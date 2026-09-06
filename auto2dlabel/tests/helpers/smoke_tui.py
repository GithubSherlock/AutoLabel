#!/usr/bin/env python3
"""autolabel TUI 独立冒烟（对标 tests/helpers/smoke_web.js jsdom 模式）。

用法：python3 auto2dlabel/tests/helpers/smoke_tui.py
覆盖：组件挂载 / 聚焦 / 斜杠命令（/help /model /cost /new）/ 指令执行
回显（Fake runner 零真实执行）/ 无 key 降级横幅 / SVG 无空格片段断言
（Textual 8.x 坑 §8.3：空格编码 &#160;，断言只用无空格片段）。

零真实权重铁律 TUI 等价物 = run_test + Fake 注入（Pilot 驱动）。
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any

from textual.containers import Container, VerticalScroll
from textual.widgets._input import Input
from textual.widgets._progress_bar import ProgressBar
from textual.widgets._static import Static

from autolabel.tui.app import WELCOME_TEXT, ChatApp

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def fake_runner(
    instruction: str,
    domain: str,
    provider: str,
    on_line: Any,
    on_progress: Any,
) -> None:
    on_progress(3, 10)  # P2 契约：进度经 on_progress 上报（面板 P3 消费）
    on_line(f"路由 {domain} 执行: {instruction}")
    on_line("检测完成: 7 框")


async def smoke() -> None:
    app = ChatApp(runner=fake_runner)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()

        # ① 组件挂载 + 聚焦（query_one 挂载即成立）
        check("chat 容器挂载", bool(app.query_one("#chat", VerticalScroll)))
        prompt = app.query_one("#prompt", Input)
        check("input 挂载", prompt is not None)
        check("input 聚焦", app.focused is prompt)
        check("欢迎语", WELCOME_TEXT.splitlines()[0] in WELCOME_TEXT)

        # ② 凭据横幅（真实环境按 .env 判定：有 key 绿字 / 无 key 红字降级提示）
        svg = app.export_screenshot()
        banner_ok = ("DeepSeek" in svg) and ("凭据" in svg)
        check("凭据横幅", banner_ok)

        # ③ 斜杠命令（SVG 无空格片段断言）
        await pilot.press(*"/help", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/help 命令列表", "/help" in svg and "/quit" in svg)
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/model 列表", "provider:" in svg and "deepseek" in svg)
        await pilot.press(*"/model set openai", "enter")
        await pilot.pause()
        check("/model set 切换", app._provider == "openai")
        svg = app.export_screenshot()
        check("/model set 回显", "已切换" in svg and "openai" in svg)
        await pilot.press(*"/model set-key", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/model set-key 指引", ".env" in svg and "OPENAI_API_KEY" in svg)
        await pilot.press(*"/model set deepseek", "enter")
        await pilot.pause()
        check("/model set 切回", app._provider == "deepseek")
        await pilot.press(*"/cost", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/cost 台账输出", "台账" in svg)
        await pilot.press(*"/new", "enter")
        await pilot.pause()

        # ④ 指令执行回显（组件树断言——长内容滚动出可视区时 SVG 不可靠）
        await pilot.press(*"检测汽车", "enter")
        for _ in range(100):
            await pilot.pause(0.05)
            if not app._busy:
                break
        texts = [
            getattr(k, "content", "") for k in app.query_one("#chat", VerticalScroll).children
        ]
        check("指令回显", "检测完成: 7 框" in texts)
        check("完成行", "✓ 任务 #1 执行完成" in texts)
        check("on_progress 上报", app._progress == (3, 10))
        # P3 任务面板：title + ProgressBar 确定态 + 状态行进度
        panel = app.query_one("#tasks", Container)
        check("任务面板挂载", panel is not None and len(panel.children) == 1)
        bar = app.query_one("#task-1-bar", ProgressBar)
        check("进度条确定态", bar.total == 10 and bar.progress == 3)
        status = app.query_one("#task-1-status", Static)
        check("状态行进度", "3/10" in getattr(status, "content", ""))

        # ⑤ P3 /cancel：阻塞 runner 模拟长任务 → 单任务取消 → 终态面板
        import threading

        gate = threading.Event()

        def blocking_runner(
            instruction: str,
            domain: str,
            provider: str,
            on_line: Any,
            on_progress: Any,
        ) -> None:
            on_progress(5, 120)
            gate.wait(10)  # 阻塞到 /cancel 之后放行
            on_line("长任务输出")

        app3 = ChatApp(runner=blocking_runner)
        async with app3.run_test(size=(100, 40)) as pilot3:
            await pilot3.pause()
            await pilot3.press(*"跟踪 video.mp4 中的行人", "enter")
            for _ in range(100):
                await pilot3.pause(0.05)
                if app3._tasks and app3._tasks[1].total == 120:
                    break
            check("长任务进度上报", app3._tasks[1].total == 120 and app3._tasks[1].done == 5)
            await pilot3.press(*"/cancel 1", "enter")
            await pilot3.pause()
            check("/cancel 单任务状态", app3._tasks[1].status == "cancelled")
            svg3 = app3.export_screenshot()
            check("/cancel 回显", "已取消任务" in svg3)
            await pilot3.press(*"/cancel 1", "enter")
            await pilot3.pause()
            svg3 = app3.export_screenshot()
            check("/cancel 已结束任务", "无需取消" in svg3)
            gate.set()  # 放行 worker 收尾
            for _ in range(100):
                await pilot3.pause(0.05)
                if not app3._busy:
                    break
            await pilot3.press(*"/quit", "enter")

        # ⑥ 无 key 降级横幅（monkeypatch 式替换——冒烟脚本不引 pytest，直接换属性）
        from auto2dlabel.agent import llm as llm_mod

        class _NoKeyClient(SimpleNamespace):
            model = "deepseek-chat"

            @property
            def has_credentials(self) -> bool:
                return False

        original = llm_mod.create_client

        def _fake_create_client(provider: str = "openai", **kw: Any) -> Any:
            return _NoKeyClient()

        llm_mod.create_client = _fake_create_client  # type: ignore[assignment]
        app2 = ChatApp(runner=fake_runner)
        async with app2.run_test(size=(100, 40)) as pilot2:
            await pilot2.pause()
            svg2 = app2.export_screenshot()
            check("无 key 降级提示", "代码直跑降级" in svg2)
            await pilot2.press(*"/quit", "enter")
        llm_mod.create_client = original

        # ⑦ P4 会话：/new 新会话 → 指令落盘 → /resume 列表 + 重放
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        sid = app._session_id
        sid_ok = app._seq == 0 and bool(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", sid)
        )
        check("/new 新会话", sid_ok)
        await pilot.press(*"检测汽车", "enter")
        for _ in range(100):
            await pilot.pause(0.05)
            if not app._busy:
                break
        await pilot.press(*"/resume", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/resume 会话列表", "历史会话" in svg)
        await pilot.press(*f"/resume {sid}", "enter")
        await pilot.pause()
        texts = [
            getattr(k, "content", "") for k in app.query_one("#chat", VerticalScroll).children
        ]
        check("/resume 重放指令行", "> 检测汽车" in texts)
        check("/resume 重放任务终态", any("执行完成" in t for t in texts))

        # ⑧ P5 指挥台：/review 注入统计 + /web 注入 spawn 记录器（不真起服务）
        from autolabel.tui import stats as stats_mod
        from autolabel.tui import webctl as webctl_mod
        from autolabel.tui.stats import ReviewStats

        orig_scan = stats_mod.scan_review_stats
        orig_open = webctl_mod.port_open
        orig_spawn = webctl_mod.spawn_web_server
        orig_wait = webctl_mod.wait_port

        def _fake_scan(d2: Any = None, d3: Any = None) -> list[ReviewStats]:
            return [
                ReviewStats("2d", "outputs", 8, 2, 0, 1, 2),
                ReviewStats("3d", "outputs/kitti3d/reviews", 9, 1, 4, 1, 1),
            ]

        def _fake_open(host: str, port: int, timeout: float = 0.5) -> bool:
            return False

        spawned: list[str] = []

        def _fake_spawn(which: str, popen: Any = None) -> int:
            spawned.append(which)
            return 42

        def _fake_wait(host: str, port: int, timeout: float = 10.0, poll: float = 0.2) -> bool:
            return True

        stats_mod.scan_review_stats = _fake_scan
        webctl_mod.port_open = _fake_open
        webctl_mod.spawn_web_server = _fake_spawn
        webctl_mod.wait_port = _fake_wait
        await pilot.press(*"/review", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/review 三档统计", "HITL" in svg and "待复核" in svg and "已复核" in svg)
        await pilot.press(*"/web", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/web 启动回显", "已启动" in svg and "8765" in svg)
        await pilot.press(*"/web 3d", "enter")
        await pilot.pause()
        svg = app.export_screenshot()
        check("/web 3d 独立服务", "8766" in svg)
        check("/web 注入 spawn 零真服务", spawned == ["2d", "3d"])
        stats_mod.scan_review_stats = orig_scan
        webctl_mod.port_open = orig_open
        webctl_mod.spawn_web_server = orig_spawn
        webctl_mod.wait_port = orig_wait

        await pilot.press(*"/quit", "enter")

    print()
    if FAILURES:
        print(f"SMOKE FAILED: {len(FAILURES)} 项失败 -> {FAILURES}")
        raise SystemExit(1)
    print("smoke_tui ALL PASS")


if __name__ == "__main__":
    asyncio.run(smoke())
