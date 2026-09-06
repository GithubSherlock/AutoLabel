"""P3 任务面板测试 —— Task/ProgressBar/状态行 + /cancel 两级终止 + 续跑引导。

零真实执行铁律：Fake runner 注入（阻塞型模拟长任务）、Fake Popen 记录
terminate/wait/kill、_ACTIVE_PROCS 整表注入——绝不 spawn 真进程。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.containers import Container, VerticalScroll
from textual.widgets._progress_bar import ProgressBar
from textual.widgets._static import Static

from autolabel.tui import app as app_module
from autolabel.tui.app import ChatApp, _cancel_procs, _subprocess_run


@pytest.fixture(autouse=True)
def _session_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """P4 会话日志重定向 tmp_path（防测试写真实 logs/）——app 内绑定与
    sessions 模块符号双 patch；返回 path 供用例直读（顶层 from-import 的
    本地绑定是 patch 前快照，不可用）。"""
    path = tmp_path / "chat_sessions.jsonl"
    monkeypatch.setattr(app_module, "SESSION_LOG_PATH", path)
    monkeypatch.setattr("autolabel.tui.sessions.SESSION_LOG_PATH", path)
    return path


def _chat_texts(app: ChatApp) -> list[str]:
    chat = app.query_one("#chat", VerticalScroll)
    return [getattr(k, "content", "") for k in chat.children if hasattr(k, "content")]


def _status_text(app: ChatApp, task_id: int) -> str:
    return str(getattr(app.query_one(f"#task-{task_id}-status", Static), "content", ""))


def _any_text_contains(app: ChatApp, fragment: str) -> bool:
    """任一对话行含片段（部分行有括注/前缀，不能整行等值）。"""
    return any(fragment in t for t in _chat_texts(app))


async def _wait_for(pilot: Any, pred: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Pilot 轮询直到谓词成立（run_test 内 pause 让主线程消化 call_from_thread）。"""
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if pred():
            return True
    return False


class _FakeProc:
    """记录 terminate/wait/kill 的假 Popen（_ACTIVE_PROCS 注入 /cancel 路径）。"""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.waited: list[float | None] = []

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


class _HangProc(_FakeProc):
    """wait 超时（terminate 后仍活）→ 触发 kill 升级。"""

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout)
        raise subprocess.TimeoutExpired("x", timeout or 0)


def _blocking_runner(
    gate: threading.Event,
    progress: tuple[int, int] = (5, 120),
    lines: tuple[str, ...] = (),
) -> Callable[..., None]:
    """阻塞型 Fake runner：先报进度，再等 gate 放行（模拟长任务供 /cancel）。"""

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        on_progress(*progress)
        for line in lines:
            on_line(line)
        gate.wait(10)

    return runner


# ---------- 面板挂载与进度 ----------


def test_task_panel_mounts_and_progress_updates() -> None:
    """提交指令 → #task-1 挂载（title 截断 + 不确定态 bar）→ on_progress 转确定态。"""
    collected: list[tuple[str, str, str, list[str]]] = []
    gate = threading.Event()

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        collected.append((instruction, domain, provider, []))
        on_progress(3, 10)
        gate.wait(10)

    async def main() -> None:
        app = ChatApp(runner=runner)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测 000860.png 中的汽车")
            ok = await _wait_for(pilot, lambda: bool(collected) and app._tasks[1].total == 10)
            assert ok
            panel = app.query_one("#tasks", Container)
            assert len(panel.children) == 1
            title = app.query_one("#task-1-title", Static)
            assert getattr(title, "content", "") == "#1 检测 000860.png 中的汽车"
            bar = app.query_one("#task-1-bar", ProgressBar)
            assert bar.total == 10 and bar.progress == 3
            assert "3/10" in _status_text(app, 1)
            assert app._tasks[1].status == "running"
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_task_title_truncated_to_40_chars() -> None:
    """长指令 title 截断 40 字（面板行宽约束）。"""
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("跟踪" + "长" * 60 + "中的行人")
            ok = await _wait_for(pilot, lambda: bool(app._tasks))
            assert ok
            title = str(getattr(app.query_one("#task-1-title", Static), "content", ""))
            assert title.startswith("#1 ")
            assert len(title) <= 2 + 41  # "#1 " + 40 字上限（含省略号）
            assert title.endswith("…")
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_task_done_terminal_state() -> None:
    """runner 正常返回 → done 终态 + 完成行（任务编号区分多任务）。"""

    async def main() -> None:
        app = ChatApp(
            runner=lambda ins, dom, prov, on_line, on_prog: on_line("检测完成: 7 框")
        )
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            ok = await _wait_for(pilot, lambda: app._tasks[1].status == "done")
            assert ok
            assert "✓ 任务 #1 执行完成" in _chat_texts(app)
            status = _status_text(app, 1)
            assert "✓ 完成" in status
            bar = app.query_one("#task-1-bar", ProgressBar)
            assert bar.progress == 1 and bar.total == 1  # 无进度上报 → 终态补满

    asyncio.run(main())


def test_task_failed_terminal_state_with_resume_hint() -> None:
    """runner 抛异常 → failed 终态 + 失败行 + --resume 清单续跑引导。"""

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        on_line("重跑失败项: auto2dlabel run --resume outputs/batch_manifest.json")
        raise RuntimeError("boom")

    async def main() -> None:
        app = ChatApp(runner=runner)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测 cars/ 中的汽车")
            ok = await _wait_for(pilot, lambda: app._tasks[1].status == "failed")
            assert ok
            texts = _chat_texts(app)
            assert "✗ 任务 #1 执行失败: boom" in texts
            assert "batch 任务可续跑: auto2dlabel run --resume outputs/batch_manifest.json" in texts
            assert "✗ 失败" in _status_text(app, 1)

    asyncio.run(main())


def test_two_tasks_concurrent_panels() -> None:
    """多任务并发：面板两行独立、进度互不串扰（exclusive=False）。"""
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            app._submit_instruction("检测行人")
            ok = await _wait_for(pilot, lambda: len(app._tasks) == 2)
            assert ok
            panel = app.query_one("#tasks", Container)
            assert len(panel.children) == 2
            assert app._tasks[1].total == 120 and app._tasks[2].total == 120
            assert app._tasks[1].done == 5 and app._tasks[2].done == 5
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


# ---------- /cancel ----------


def test_cancel_fake_runner_marks_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake runner（无登记进程）→ /cancel 仍置 cancelled；worker 收尾不覆盖终态。"""
    monkeypatch.setattr(app_module, "_ACTIVE_PROCS", [])
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("跟踪 video.mp4 中的行人")
            await _wait_for(pilot, lambda: bool(app._tasks))
            app._slash_command("/cancel 1")
            await pilot.pause()
            assert app._tasks[1].status == "cancelled"
            assert "⛔ 已取消" in _status_text(app, 1)
            assert "⛔ 已取消任务 #1" in _chat_texts(app)
            gate.set()  # 放行 worker：cancelled 态不打印完成行、不覆盖终态
            await _wait_for(pilot, lambda: not app._busy)
            assert app._tasks[1].status == "cancelled"
            assert "✓ 任务 #1 执行完成" not in _chat_texts(app)

    asyncio.run(main())


def test_cancel_terminates_registered_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    """登记进程（真实 subprocess 路径）→ /cancel 发 terminate；升级 wait(30s)
    在 daemon 线程异步执行（不冻结 TUI 主线程）——wait 正常返回无需 kill。"""
    proc = _FakeProc()
    monkeypatch.setattr(
        app_module, "_ACTIVE_PROCS", [("跟踪 video.mp4 中的行人", proc)],
    )
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("跟踪 video.mp4 中的行人")
            await _wait_for(pilot, lambda: bool(app._tasks))
            app._slash_command("/cancel 1")
            await pilot.pause()
            assert proc.terminated  # 主线程同步 terminate
            assert _any_text_contains(app, "terminate 已发送")
            # 升级在 daemon 线程：轮询等其完成（Fake wait 立即返回）
            ok = await _wait_for(pilot, lambda: bool(proc.waited))
            assert ok
            assert proc.waited == [30]  # grace 30s：长 C 调用需到字节码边界才响应
            assert not proc.killed  # wait 正常返回，无需 kill 升级
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_cancel_kill_escalation_on_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    """terminate 后 wait 超时（进程仍活）→ kill 升级。"""
    hang = _HangProc()
    ok_proc = _FakeProc()
    monkeypatch.setattr(
        app_module,
        "_ACTIVE_PROCS",
        [("检测汽车", hang), ("检测行人", ok_proc)],
    )
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            await _wait_for(pilot, lambda: bool(app._tasks))
            app._slash_command("/cancel 1")
            await pilot.pause()
            # 升级在 daemon 线程：wait 超时 → kill 兜底（Fake 立即超时）
            ok = await _wait_for(pilot, lambda: hang.killed)
            assert ok
            assert hang.waited == [30]
            assert not ok_proc.terminated  # 只取消目标指令的进程
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_cancel_all_running_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """/cancel 无参 → 全部运行中任务置 cancelled（含已结束任务豁免）。"""
    monkeypatch.setattr(app_module, "_ACTIVE_PROCS", [])
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            app._submit_instruction("检测行人")
            await _wait_for(pilot, lambda: len(app._tasks) == 2)
            app._slash_command("/cancel")
            await pilot.pause()
            assert app._tasks[1].status == "cancelled"
            assert app._tasks[2].status == "cancelled"
            assert "⛔ 已取消 2 个任务" in _chat_texts(app)
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_cancel_invalid_and_done_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """/cancel 参数容错：非法 id / 不存在 / 已结束任务 / 空任务表。"""
    monkeypatch.setattr(app_module, "_ACTIVE_PROCS", [])

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/cancel abc")
            assert _any_text_contains(app, "未知任务 id 'abc'")
            app._slash_command("/cancel 99")
            assert _any_text_contains(app, "无任务 #99")
            app._submit_instruction("检测汽车")
            await _wait_for(pilot, lambda: app._tasks[1].status == "done")
            app._slash_command("/cancel 1")
            assert _any_text_contains(app, "已结束（✓ 完成）")
            app._slash_command("/cancel")
            assert _any_text_contains(app, "无运行中任务")

    asyncio.run(main())


# ---------- _cancel_procs 纯函数 ----------


def test_cancel_procs_instruction_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """指令过滤：只 terminate 匹配指令的进程；None = 全部。

    _cancel_procs 不移除登记（出表靠 _subprocess_run finally）——模拟
    a/c 进程退出出表后，无参取消只剩 b。
    """
    a, b, c = _FakeProc(), _FakeProc(), _FakeProc()
    registry: list[tuple[str, Any]] = [("检测汽车", a), ("检测行人", b), ("检测汽车", c)]
    monkeypatch.setattr(app_module, "_ACTIVE_PROCS", registry)
    n = _cancel_procs("检测汽车")
    assert n == 2 and a.terminated and c.terminated and not b.terminated
    registry[:] = [("检测行人", b)]  # a/c 退出后出表（_unregister_proc 语义）
    n = _cancel_procs()
    assert n == 1 and b.terminated


def test_cancel_procs_empty_registry() -> None:
    """空登记表 → 0 进程，零副作用（Fake runner 场景）。"""
    assert _cancel_procs() == 0
    assert _cancel_procs("检测汽车") == 0


# ---------- _subprocess_run 协议解析与登记 ----------


def test_subprocess_run_parses_progress_and_registers_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """协议行 → on_progress（不回显）；普通行 → on_line；登记/出表成对；
    子进程 env 携带 AUTOLABEL_PROGRESS=1（协议行只在 TUI 通道开启）。"""
    registered: list[str] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(app_module, "_register_proc", lambda ins, p: registered.append(ins))
    monkeypatch.setattr(app_module, "_unregister_proc", lambda p: unregistered.append(p))

    class _FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            self._cmd = cmd
            self._env = kwargs.get("env")

        @property
        def stdout(self) -> Any:
            return iter(["[AL_PROGRESS] 1/2\n", "检测输出行\n", "[AL_PROGRESS] 2/2\n"])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    lines: list[str] = []
    progress: list[tuple[int, int | None]] = []
    _subprocess_run(
        "检测汽车", "2d", "deepseek", lines.append, lambda d, t: progress.append((d, t)),
    )
    assert progress == [(1, 2), (2, 2)]
    assert lines == ["检测输出行\n"]  # 协议行不回显对话区
    assert registered == ["检测汽车"]
    assert len(unregistered) == 1  # finally 出表成对


def test_subprocess_run_env_gates_progress_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """env 门控（#28）：AUTOLABEL_PROGRESS=1 进子进程 env——普通终端 CLI 零污染。"""
    monkeypatch.setattr(app_module, "_register_proc", lambda ins, p: None)
    monkeypatch.setattr(app_module, "_unregister_proc", lambda p: None)
    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured.update(kwargs)

        @property
        def stdout(self) -> Any:
            return iter([])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    _subprocess_run("检测汽车", "2d", "deepseek", lambda line: None, lambda d, t: None)
    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("AUTOLABEL_PROGRESS") == "1"


def test_subprocess_run_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """子进程非零退出 → RuntimeError（worker 转 failed 终态），登记已出表。"""
    registered: list[str] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(app_module, "_register_proc", lambda ins, p: registered.append(ins))
    monkeypatch.setattr(app_module, "_unregister_proc", lambda p: unregistered.append(p))

    class _FakePopen:
        returncode = 1  # Popen.wait() 返回并赋值（app.py 读 returncode 报退出码）

        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            self._cmd = cmd

        @property
        def stdout(self) -> Any:
            return iter(["报错行\n"])

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    with pytest.raises(RuntimeError, match="退出码 1"):
        _subprocess_run(
            "检测汽车", "2d", "deepseek", lambda line: None, lambda d, t: None,
        )
    assert len(registered) == 1 and len(unregistered) == 1


# ---------- 审查修复回归（2026-09-07） ----------


def test_cancel_sibling_tasks_same_instruction() -> None:
    """#15 连带取消：同指令并发任务（进程一并被 terminate）→ sibling 同步
    置 cancelled，不停留在 running 等 worker finally 记 failed。"""
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            app._submit_instruction("检测汽车")  # 同指令并发
            await _wait_for(pilot, lambda: len(app._tasks) == 2)
            app._slash_command("/cancel 1")
            await pilot.pause()
            assert app._tasks[1].status == "cancelled"
            assert app._tasks[2].status == "cancelled"  # sibling 连带
            assert _any_text_contains(app, "连带任务一并取消")
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_progress_monotonic_across_step_switch() -> None:
    """#26 步切换回退单调：多步计划上一步 100/100 → 本步 0/1200，进度条
    视觉不倒退（done 只增、total 只增不减）。"""
    progress: list[tuple[int, int]] = [(100, 100), (0, 1200)]
    gate = threading.Event()

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        for d, t in progress:
            on_progress(d, t)
        gate.wait(10)

    async def main() -> None:
        app = ChatApp(runner=runner)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            await _wait_for(pilot, lambda: app._tasks[1].done == 100)
            # 步切换回退 (0, 1200)：done 单调 100、total 只增到 1200
            assert app._tasks[1].done == 100
            assert app._tasks[1].total == 1200
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_task_result_lands_in_originating_session(
    monkeypatch: pytest.MonkeyPatch, _session_log: Path,
) -> None:
    """#18 会话归属：任务发起后 /new，其终态行必须落回发起时会话——
    不能跟随当前 self._session_id 串到新会话。"""
    gate = threading.Event()

    async def main() -> None:
        app = ChatApp(runner=_blocking_runner(gate))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            await _wait_for(pilot, lambda: bool(app._tasks))
            first_session = app._tasks[1].session_id
            await asyncio.sleep(1.1)  # sid 秒级粒度：跨秒 /new 才产生新 id
            app._slash_command("/new")
            await pilot.pause()
            assert app._session_id != first_session
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)
        # 任务终态行在发起会话、不在新会话
        entries = [
            json.loads(line)
            for line in _session_log.read_text(encoding="utf-8").splitlines()
        ]
        results = [
            e for e in entries
            if e.get("kind") == "task_result" and e.get("task", {}).get("id") == 1
        ]
        assert len(results) == 1
        assert results[0]["session_id"] == first_session

    asyncio.run(main())
