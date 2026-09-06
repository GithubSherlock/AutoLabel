"""P4 会话管理测试 —— sessions.py 纯函数 + ChatApp /new /resume 集成。

铁律：零真实执行（Fake runner）、零真实 LLM；会话日志统一重定向
tmp_path（SESSION_LOG_PATH monkeypatch，防测试写真实 logs/）。
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.containers import VerticalScroll

from autolabel.tui import app as app_module
from autolabel.tui.app import ChatApp
from autolabel.tui.sessions import (
    SESSION_LOG_PATH,
    append_session_line,
    list_sessions,
    load_sessions,
    make_session_id,
    session_messages,
)


@pytest.fixture(autouse=True)
def _session_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """全部用例会话日志重定向 tmp_path（含 ChatApp 内部引用）。"""
    path = tmp_path / "chat_sessions.jsonl"
    monkeypatch.setattr(app_module, "SESSION_LOG_PATH", path)
    monkeypatch.setattr("autolabel.tui.sessions.SESSION_LOG_PATH", path)
    return path


def _chat_texts(app: ChatApp) -> list[str]:
    chat = app.query_one("#chat", VerticalScroll)
    return [getattr(k, "content", "") for k in chat.children if hasattr(k, "content")]


async def _wait_for(pilot: Any, pred: Callable[[], bool], timeout: float = 5.0) -> bool:
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if pred():
            return True
    return False


# ---------- 纯函数 ----------


def test_append_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    append_session_line(path, "s1", 1, "user", "检测汽车", domain="2d", provider="deepseek")
    append_session_line(
        path, "s1", 2, "task_result", "✓ 任务 #1 执行完成",
        task={"id": 1, "status": "done", "manifest": None},
    )
    entries = load_sessions(path)
    assert len(entries) == 2
    assert entries[0]["text"] == "检测汽车"
    assert entries[0]["domain"] == "2d" and entries[0]["provider"] == "deepseek"
    assert entries[0]["markup"] is False
    assert entries[1]["kind"] == "task_result"
    assert entries[1]["task"] == {"id": 1, "status": "done", "manifest": None}
    assert isinstance(entries[0]["ts"], float)


def test_load_missing_file_empty(tmp_path: Path) -> None:
    assert load_sessions(tmp_path / "nope.jsonl") == []


def test_load_skips_corrupted_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    append_session_line(path, "s1", 1, "user", "好行")
    with open(path, "a", encoding="utf-8") as f:
        f.write("{损坏行\n")
        f.write("\n")  # 空行
        f.write('{"no_session": 1}\n')  # 缺 session_id 的 dict
    entries = load_sessions(path)
    assert len(entries) == 1 and entries[0]["text"] == "好行"


def test_list_sessions_sort_limit_preview(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    long_text = "检测" + "很" * 60 + "中的汽车"
    append_session_line(path, "s-old", 1, "user", long_text, ts=100.0)
    append_session_line(path, "s-old", 2, "system", "路由: 2D", ts=101.0)
    append_session_line(path, "s-new", 1, "user", "跟踪行人", ts=200.0)
    append_session_line(path, "s-mid", 1, "system", "无 user 行", ts=150.0)
    summaries = list_sessions(load_sessions(path), limit=3)
    assert [s["session_id"] for s in summaries] == ["s-new", "s-mid", "s-old"]  # last_ts 倒序
    assert summaries[0]["messages"] == 1
    # 首条 user 截断 40 字（s-old）；无 user 行 → 空预览（s-mid）
    assert summaries[1]["preview"] == ""
    assert len(summaries[2]["preview"]) == 41 and summaries[2]["preview"].endswith("…")


def test_session_messages_filter_and_seq_order(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    for seq, text in [(2, "b"), (1, "a"), (3, "c")]:
        append_session_line(path, "s1", seq, "system", text, ts=float(seq))
    append_session_line(path, "s2", 1, "system", "别家")
    rows = session_messages(load_sessions(path), "s1")
    assert [r["text"] for r in rows] == ["a", "b", "c"]


def test_old_line_missing_fields_tolerance(tmp_path: Path) -> None:
    """旧行缺 markup/task/domain/provider → 默认值（重放防注入）。"""
    path = tmp_path / "s.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"session_id": "s1", "seq": 1, "kind": "system", "text": "旧行"}) + "\n")
    entries = load_sessions(path)
    assert entries[0].get("markup", False) is False
    assert entries[0].get("task") is None
    assert list_sessions(entries)[0]["preview"] == ""  # 无 user 行 → 空预览


def test_make_session_id_format() -> None:
    sid = make_session_id()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", sid)


# ---------- ChatApp 集成 ----------


def test_instruction_and_done_write_session_lines(_session_log: Path) -> None:
    """指令 → user 行（含 domain/provider）；完成 → task_result done；
    逐行子进程回显不进会话。"""
    lines: list[str] = []

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        on_progress(1, 2)
        on_line("检测输出行")
        lines.append(instruction)

    async def main() -> None:
        app = ChatApp(runner=runner)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("检测汽车")
            await _wait_for(pilot, lambda: app._tasks[1].status == "done")
            entries = load_sessions(_session_log)
            user = [e for e in entries if e["kind"] == "user"]
            assert user and user[0]["text"] == "> 检测汽车"
            assert user[0]["domain"] == "2d" and user[0]["provider"] == "deepseek"
            results = [e for e in entries if e["kind"] == "task_result"]
            assert results and results[-1]["task"] == {"id": 1, "status": "done", "manifest": None}
            # worker 行回显 + 进度不进会话（record=False 防膨胀）
            assert all(e["text"] != "检测输出行" for e in entries)
            assert all(e["text"] != "> 检测输出行" for e in entries)

    asyncio.run(main())


def test_failed_task_result_records_manifest(_session_log: Path) -> None:
    """失败 → task_result failed + manifest 提取（--resume 行扫描）。"""

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
            await _wait_for(pilot, lambda: app._tasks[1].status == "failed")
            results = [
                e for e in load_sessions(_session_log) if e["kind"] == "task_result"
            ]
            assert results[-1]["task"] == {
                "id": 1, "status": "failed", "manifest": "outputs/batch_manifest.json",
            }

    asyncio.run(main())


def test_cancel_writes_cancelled_task_result(
    monkeypatch: pytest.MonkeyPatch, _session_log: Path
) -> None:
    """取消 → task_result cancelled（重放可见终态）。"""
    monkeypatch.setattr(app_module, "_ACTIVE_PROCS", [])
    gate = threading.Event()

    def runner(
        instruction: str,
        domain: str,
        provider: str,
        on_line: Callable[[str], None],
        on_progress: Callable[[int, int | None], None],
    ) -> None:
        on_progress(5, 120)
        gate.wait(10)

    async def main() -> None:
        app = ChatApp(runner=runner)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._submit_instruction("跟踪 video.mp4 中的行人")
            await _wait_for(pilot, lambda: bool(app._tasks))
            app._slash_command("/cancel 1")
            await pilot.pause()
            results = [
                e for e in load_sessions(_session_log) if e["kind"] == "task_result"
            ]
            assert results[-1]["task"]["status"] == "cancelled"
            gate.set()
            await _wait_for(pilot, lambda: not app._busy)

    asyncio.run(main())


def test_new_clears_and_resets_session(_session_log: Path) -> None:
    """新会话：清屏 + seq 归零 + session_id 格式合法；旧会话行保留在日志。"""

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.append("旧会话消息", kind="system")
            old_sid = app._session_id
            assert app._seq == 1
            app._slash_command("/new")
            await pilot.pause()  # remove_children 延迟到下一次 pump（AwaitRemove）
            chat = app.query_one("#chat", VerticalScroll)
            texts = [getattr(k, "content", "") for k in chat.children if hasattr(k, "content")]
            assert "旧会话消息" not in texts  # 清屏
            assert app._seq == 0
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", app._session_id)
            # 旧会话已逐行落盘，/new 不动历史
            entries = load_sessions(_session_log)
            assert any(e["session_id"] == old_sid and e["text"] == "旧会话消息" for e in entries)

    asyncio.run(main())


def test_resume_replays_and_continues_session(_session_log: Path) -> None:
    """重放：文本一致 + task_result 附 manifest 引导 + 不重复落盘 +
    会话上下文切换（后续消息续写同 session_id，seq 接续）。"""
    path = _session_log
    sid = "2026-09-06T15-30-00"
    append_session_line(path, sid, 1, "user", "> 检测汽车", domain="2d", provider="deepseek")
    append_session_line(path, sid, 2, "system", "路由: 2D")
    append_session_line(
        path, sid, 3, "task_result", "✗ 任务 #1 执行失败: boom",
        task={"id": 1, "status": "failed", "manifest": "outputs/m.json"},
    )

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command(f"/resume {sid}")
            texts = _chat_texts(app)
            assert "> 检测汽车" in texts
            assert "路由: 2D" in texts
            assert "✗ 任务 #1 执行失败: boom" in texts
            assert "batch 任务可续跑: auto2dlabel run --resume outputs/m.json" in texts
            # 重放零落盘（/resume 命令回显行在会话切换前已写入新会话，属预期）
            assert len(session_messages(load_sessions(path), sid)) == 3
            assert app._session_id == sid and app._seq == 3
            app.append("续写行", kind="system")
            entries = load_sessions(path)
            assert entries[-1]["session_id"] == sid and entries[-1]["seq"] == 4
            # done 任务重放不附引导（完成无需续跑）
            append_session_line(
                path, sid, 5, "task_result", "✓ 任务 #2 执行完成",
                task={"id": 2, "status": "done", "manifest": "outputs/m2.json"},
            )
            app._slash_command(f"/resume {sid}")
            texts = _chat_texts(app)
            assert "✓ 任务 #2 执行完成" in texts
            assert not any("outputs/m2.json" in t for t in texts)  # done 无引导

    asyncio.run(main())


def test_resume_list_and_unknown_session(_session_log: Path) -> None:
    """无参列表（倒序 + 条数）；未知 id 提示。"""
    path = _session_log
    append_session_line(path, "s1", 1, "user", "> 检测汽车", ts=100.0)
    append_session_line(path, "s2", 1, "user", "> 跟踪行人", ts=200.0)

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/resume")
            texts = _chat_texts(app)
            assert "历史会话" in "".join(texts)
            assert "s2" in "".join(texts) and "s1" in "".join(texts)
            assert "1 条" in "".join(texts)
            app._slash_command("/resume 不存在")
            assert any("无会话 '不存在'" in t for t in _chat_texts(app))

    asyncio.run(main())


def test_slash_user_line_recorded(_session_log: Path) -> None:
    """斜杠命令输入行也进会话（kind=user，重放可复现操作序列）。"""

    async def main() -> None:
        app = ChatApp(runner=lambda ins, dom, prov, on_line, on_prog: None)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app._slash_command("/cost")
            entries = load_sessions(_session_log)
            user = [e for e in entries if e["kind"] == "user"]
            assert user and user[0]["text"] == "> /cost"

    asyncio.run(main())


def test_sessions_module_path_is_logs_default() -> None:
    """会话日志默认落 logs/（已 gitignore，不入库）。"""
    assert str(SESSION_LOG_PATH) == "logs/chat_sessions.jsonl"
