"""P5 HITL 指挥台测试 —— stats.py 三档统计 + webctl.py 端口/spawn + /review /web。

铁律：零真实服务（Fake Popen 记录 cmd，绝不起 Web 服务）、零真实 LLM；
review JSON 用 tmp_path 造（损坏/空目录容错）；端口探测 monkeypatch socket。
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from autolabel.tui import app as app_module
from autolabel.tui.app import ChatApp
from autolabel.tui.stats import ReviewStats, scan_review_stats
from autolabel.tui.webctl import (
    AUTOLABEL3D_PORT_ENV,
    AUTOLABEL_PORT_ENV,
    port_of,
    server_script,
    spawn_web_server,
)


@pytest.fixture(autouse=True)
def _session_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "SESSION_LOG_PATH", tmp_path / "chat_sessions.jsonl")


def _chat_texts(app: ChatApp) -> list[str]:
    return [
        getattr(k, "content", "")
        for k in app.query_one("#chat").children
        if hasattr(k, "content")
    ]


def _write_review(path: Path, summary: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"image": path.stem, "annotations": [], "summary": summary}),
        encoding="utf-8",
    )


# ---------- stats.py 纯函数 ----------


def test_scan_review_stats_counts_two_domains(tmp_path: Path) -> None:
    """2D（无 accepted 两档）+ 3D（accepted 三档）计数 + 已复核文件数。"""
    d2, d3 = tmp_path / "outputs", tmp_path / "outputs" / "kitti3d" / "reviews"
    _write_review(d2 / "a_1_review.json", {"review_count": 5, "hard_count": 2})
    _write_review(d2 / "a_2_review.json", {"review_count": 3, "hard_count": 0})
    (d2 / "a_1_reviewed.json").write_text("{}", encoding="utf-8")
    (d2 / "a_2_review.json.reviewed").write_text("{}", encoding="utf-8")  # 侧车排除
    _write_review(d3 / "b_review.json", {"accepted": 4, "review_count": 9, "hard_count": 1})
    (d3 / "b_reviewed.json").write_text("{}", encoding="utf-8")

    s2d, s3d = scan_review_stats(d2, d3)
    assert isinstance(s2d, ReviewStats) and s2d.domain == "2d"
    assert s2d.pending_review == 8 and s2d.hard == 2
    assert s2d.accepted == 0  # 2D summary 无 accepted 档
    assert s2d.reviewed_files == 1 and s2d.total_files == 2
    assert s3d.domain == "3d"
    assert s3d.pending_review == 9 and s3d.hard == 1 and s3d.accepted == 4
    assert s3d.reviewed_files == 1 and s3d.total_files == 1


def test_scan_review_stats_tolerates_missing_dir_and_corruption(tmp_path: Path) -> None:
    """目录缺失 → 全零；损坏 JSON / 缺 summary → 跳过（collect_review_pool 同款）。"""
    d2 = tmp_path / "outputs"
    _write_review(d2 / "bad_review.json", {"review_count": 1})
    (d2 / "bad_review.json").write_text("{损坏", encoding="utf-8")  # 覆盖为损坏
    (d2 / "no_summary_review.json").write_text('{"image": "x"}', encoding="utf-8")
    _write_review(d2 / "ok_review.json", {"review_count": 2, "hard_count": 1})
    s2d, s3d = scan_review_stats(d2, tmp_path / "nonexistent")
    assert s2d.pending_review == 2 and s2d.hard == 1 and s2d.total_files == 1
    assert s3d.pending_review == 0 and s3d.reviewed_files == 0


def test_review_stats_default_dirs_exist() -> None:
    """默认口径：2D outputs/、3D outputs/kitti3d/reviews/（真实目录可缺）。"""
    s2d, s3d = scan_review_stats()
    assert s2d.dir == "outputs" and s3d.dir == "outputs/kitti3d/reviews"


# ---------- webctl.py 纯函数 ----------


def test_port_of_env_defaults_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert port_of("2d") == (AUTOLABEL_PORT_ENV, 8765)
    assert port_of("3d") == (AUTOLABEL3D_PORT_ENV, 8766)
    monkeypatch.setenv(AUTOLABEL_PORT_ENV, "9000")
    monkeypatch.setenv(AUTOLABEL3D_PORT_ENV, "9001")
    assert port_of("2d") == (AUTOLABEL_PORT_ENV, 9000)
    assert port_of("3d") == (AUTOLABEL3D_PORT_ENV, 9001)


def test_port_open_uses_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    from autolabel.tui import webctl

    calls: list[tuple[str, int, float]] = []

    class _FakeSock:
        def __init__(self, addr: tuple[str, int], timeout: float) -> None:
            calls.append((addr[0], addr[1], timeout))
            raise OSError("拒绝连接")

    monkeypatch.setattr(socket, "create_connection", _FakeSock)
    assert webctl.port_open("127.0.0.1", 8765) is False
    assert calls == [("127.0.0.1", 8765, 0.5)]


def test_server_script_paths() -> None:
    """脚本路径落在两侧包内 web/server.py（web/ 无 __init__.py 走脚本 spawn）。"""
    assert server_script("2d").name == "server.py"
    assert "auto2dlabel" in str(server_script("2d"))
    assert server_script("3d").name == "server.py"
    assert "auto3dlabel" in str(server_script("3d"))


def test_spawn_web_server_cmd_and_detach(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakePopen:
        pid = 4242

        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    spawn_web_server("2d", popen=_FakePopen)
    cmd = captured["cmd"]
    assert cmd[1] == "-u" and cmd[2].endswith("auto2dlabel/web/server.py")
    kwargs = captured["kwargs"]
    assert kwargs["stdout"] is subprocess.DEVNULL  # detach：不占 TUI 输出


def test_wait_port_poll_until_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from autolabel.tui import webctl

    state = {"n": 0}

    def _fake_open(host: str, port: int, timeout: float = 0.5) -> bool:
        state["n"] += 1
        return state["n"] >= 3  # 第 3 次探测才就绪

    monkeypatch.setattr(webctl, "port_open", _fake_open)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert webctl.wait_port("127.0.0.1", 8765, timeout=10.0, poll=0.2) is True
    assert state["n"] == 3
    state["n"] = 0
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def _never(host: str, port: int, timeout: float = 0.5) -> bool:
        state["n"] += 1
        return False

    monkeypatch.setattr(webctl, "port_open", _never)
    # 时钟首次 0.0（deadline=10.0），之后 11.0 → while 条件即超时，循环体一次不进
    clock = iter([0.0, 11.0])
    monkeypatch.setattr(time, "time", lambda: next(clock, 11.0))
    assert webctl.wait_port("127.0.0.1", 8765, timeout=10.0) is False
    assert state["n"] == 0


# ---------- /review /web 命令 ----------


def test_review_command_renders_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """/review 走注入统计（不真扫 outputs/）：三档渲染 + 命令注册。"""
    from autolabel.tui import stats as stats_mod

    monkeypatch.setattr(
        stats_mod,
        "scan_review_stats",
        lambda: [
            ReviewStats("2d", "outputs", 8, 2, 0, 1, 2),
            ReviewStats("3d", "outputs/kitti3d/reviews", 9, 1, 4, 1, 1),
        ],
    )

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/review")
            joined = "\n".join(_chat_texts(app))
            assert "待复核 8" in joined and "困难 2" in joined and "已复核 1" in joined  # 2D
            # 3D：已复核口径与 2D 统一（reviewed_files），accepted 单列不混入（#16）
            assert "待复核 9" in joined and "已复核 1" in joined and "接受标注 4" in joined

    asyncio.run(main())


def test_web_command_reports_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """端口已占用 → 「已在运行」不 spawn（FakePopen 记录：零 spawn）。"""
    from autolabel.tui import webctl

    monkeypatch.setattr(webctl, "port_open", lambda host, port, timeout=0.5: True)
    spawned: list[str] = []

    def _spawn(which: str) -> int:
        spawned.append(which)
        return 1

    monkeypatch.setattr(webctl, "spawn_web_server", _spawn)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/web")
            assert any("已在运行" in t and "8765" in t for t in _chat_texts(app))
            assert spawned == []  # 已占用不 spawn

    asyncio.run(main())


def test_web_command_spawns_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """未占用 → spawn（FakePopen 记录 cmd 不真起服务）→ wait_port → URL。"""
    from autolabel.tui import webctl

    monkeypatch.setattr(webctl, "port_open", lambda host, port, timeout=0.5: False)
    spawned: list[str] = []

    def _spawn(which: str) -> int:
        spawned.append(which)
        return 7

    monkeypatch.setattr(webctl, "spawn_web_server", _spawn)
    monkeypatch.setattr(webctl, "wait_port", lambda host, port, timeout=10.0, poll=0.2: True)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/web 3d")
            assert any("已启动" in t and "8766" in t and "pid 7" in t for t in _chat_texts(app))
            assert spawned == ["3d"]

    asyncio.run(main())


def test_web_command_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """端口 10s 未就绪 → 提示 + kill 指引（spawn 进程存活，用户手动处置）。"""
    from autolabel.tui import webctl

    monkeypatch.setattr(webctl, "port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(webctl, "spawn_web_server", lambda which: 9)
    monkeypatch.setattr(webctl, "wait_port", lambda host, port, timeout=10.0, poll=0.2: False)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/web")
            assert any("未就绪" in t and "kill 9" in t for t in _chat_texts(app))

    asyncio.run(main())


def test_dispatch_passes_args_to_review_and_web() -> None:
    """P5 命令注册在 SLASH_COMMANDS（分发机制由 registry 测试覆盖）。"""
    from autolabel.tui.app import SLASH_COMMANDS

    names = {c.name: c for c in SLASH_COMMANDS}
    assert names["/review"].handler.__name__ == "_cmd_review"
    assert names["/web"].handler.__name__ == "_cmd_web"


# ---------- 审查修复回归（2026-09-07） ----------


def test_web_repeat_within_10s_reports_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """#13 防重复 spawn：首次 /web spawn 后端口未就绪（wait 中），10s 内
    重复 /web 报「启动中」不二次 spawn。"""
    from autolabel.tui import webctl

    monkeypatch.setattr(webctl, "port_open", lambda host, port, timeout=0.5: False)
    spawned: list[str] = []

    def _spawn(which: str) -> int:
        spawned.append(which)
        return 7

    monkeypatch.setattr(webctl, "spawn_web_server", _spawn)
    # 首次 /web 后服务未就绪（wait_port False）→ 登记保留 → 重复 /web 短路
    monkeypatch.setattr(webctl, "wait_port", lambda host, port, timeout=10.0, poll=0.2: False)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/web")
            assert spawned == ["2d"]
            app._slash_command("/web")
            assert any("启动中" in t and "pid 7" in t for t in _chat_texts(app))
            assert spawned == ["2d"]  # 未二次 spawn

    asyncio.run(main())


def test_web_retry_after_spawned_flag_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次 /web 就绪 → 登记清除 → 再次 /web 走端口检测路径（已在运行）。"""
    from autolabel.tui import webctl

    state = {"n": 0}

    def _open(host: str, port: int, timeout: float = 0.5) -> bool:
        state["n"] += 1
        return state["n"] >= 2  # spawn 前检测 False；就绪后再次检测 True

    monkeypatch.setattr(webctl, "port_open", _open)
    spawned: list[str] = []

    def _spawn(which: str) -> int:
        spawned.append(which)
        return 7

    monkeypatch.setattr(webctl, "spawn_web_server", _spawn)
    monkeypatch.setattr(webctl, "wait_port", lambda host, port, timeout=10.0, poll=0.2: True)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/web")
            assert spawned == ["2d"]
            assert not app._web_spawned  # 就绪后登记清除
            app._slash_command("/web")
            assert any("已在运行" in t for t in _chat_texts(app))
            assert spawned == ["2d"]

    asyncio.run(main())


def test_port_of_non_numeric_env_falls_back() -> None:
    """#14 port_of 容错：env 非数字（误配）→ 回退默认端口，/web 不崩。"""
    from autolabel.tui import webctl

    monkeypatch.setenv("AUTOLABEL_PORT", "http://x")
    name, port = webctl.port_of("2d")
    assert name == "AUTOLABEL_PORT" and port == 8765
    monkeypatch.setenv("AUTOLABEL3D_PORT", "abc")
    name, port = webctl.port_of("3d")
    assert name == "AUTOLABEL3D_PORT" and port == 8766
