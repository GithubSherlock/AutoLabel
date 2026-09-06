"""Textual TUI 外壳（v1.0 P1+P2+P3）。

结构：Header / 对话区 VerticalScroll / 任务面板 Container / Input / Footer。
斜杠命令：/help /model /cost /new /resume /cancel /quit（声明式注册表
SLASH_COMMANDS，v1.0 P1+ 集中替换 if/elif 链，/help 文案由 build_help_text
生成；P2 起 handler 统一签名 (app, args)，args 可为空串）。
P2：/model 三态（无参列注册表 / set <name> 切换 / set-key 密钥指引）+ runner
契约一次升级 5 参（instruction, domain, provider, on_line, on_progress）。
P3：后台任务面板（Task + ProgressBar + 状态行）+ /cancel（terminate→wait→kill
两级）+ [AL_PROGRESS] 行协议解析（子进程 stdout → on_progress，不回显对话区）
+ 取消/失败后 manifest 续跑引导（--resume 行扫描）。
P4：会话持久化（logs/chat_sessions.jsonl 逐行落盘，sessions.py 纯函数）——
append(record=) 行级过滤（子进程回显不进会话）、task_result 任务终态落盘、
/new 开新会话、/resume 无参列表/<id> 重放（不重跑任务，续跑走 manifest 引导）。
P5：HITL 指挥台——/review 三档分流统计（stats.py 纯函数重扫无缓存）、
/web 一键启动 Web 复核（webctl.py 端口检测 + detach spawn + 轮询就绪）。
指令执行：worker 线程内 subprocess 跑旧命令 chat（进程隔离 + 行级回显；
多任务并发不限——@work exclusive=False，每任务独立 Popen/独立 stdout 迭代）。

无 key 降级（v0.6 红线在 TUI 内保持）：启动横幅按 has_credentials 判定，
无凭据显示「代码直跑」提示；执行链自身的降级（_plan_without_llm 等）
由子进程内 chat_command 承担，本壳不重复实现。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets._footer import Footer
from textual.widgets._header import Header
from textual.widgets._input import Input
from textual.widgets._progress_bar import ProgressBar
from textual.widgets._static import Static

from autolabel.route import route_domain
from autolabel.tui.sessions import (
    SESSION_LOG_PATH,
    append_session_line,
    list_sessions,
    load_sessions,
    make_session_id,
    safe_int,
    session_messages,
)

WELCOME_TEXT = (
    "AutoLabel —— Agentic 数据标注（AI 为主、人为辅）。\n"
    "输入标注指令开始，或输入 /help 查看命令。"
)

# 指令执行器（可注入替身供测试；签名 = (指令, 域, provider, 行回调, 进度回调) → 无返回值。
# 默认实现 = subprocess 跑旧命令 chat，每行输出经 on_line 回显；[AL_PROGRESS] done/total
# 协议行经 on_progress(done, total) 上报（P3 任务面板消费））
InstructionRunner = Callable[
    [str, str, str, Callable[[str], None], Callable[[int, int | None], None]], None
]

# 横幅展示名（其余 provider 用注册名原文）
_PROVIDER_DISPLAY = {"deepseek": "DeepSeek"}

# P3：运行中子进程登记表（(指令, Popen)——/cancel 的 terminate 目标；
# _subprocess_run 注册/移除，完成与异常统一出表，Fake runner 零触及）
_ACTIVE_PROCS: list[tuple[str, subprocess.Popen[Any]]] = []

# 取消升级 grace（秒）：见 _cancel_procs——长 C 调用下 SIGTERM 延迟生效，
# 过短会误 kill 打穿 finally 链（成片视频损坏/manifest 未写）
_CANCEL_GRACE = 30.0


def _register_proc(instruction: str, proc: subprocess.Popen[Any]) -> None:
    """登记运行中子进程（/cancel 按指令精确匹配）。"""
    _ACTIVE_PROCS.append((instruction, proc))


def _unregister_proc(proc: subprocess.Popen[Any]) -> None:
    """移除登记（进程对象身份比较；完成/异常统一走 finally）。"""
    _ACTIVE_PROCS[:] = [(i, p) for i, p in _ACTIVE_PROCS if p is not proc]


def _cancel_procs(instruction: str | None = None) -> int:
    """两级取消：terminate → 后台 wait(grace) → 仍活 kill；返回发送 terminate 的进程数。

    instruction=None 取消全部；否则按指令精确匹配（同指令并发时全部命中，
    取消语义宁过勿漏——同一指令的产物相同，重跑等价）。
    wait 在 daemon 线程执行（不冻结 TUI 主线程）；grace=30s——子进程卡在
    长 C 调用（SAHI 切片/CPU 推理/L3 首帧）时 SIGTERM 要等到字节码边界才
    生效，5s 会误 kill 打穿 finally 链（成片视频损坏/manifest 未写），
    多等远好于损坏。
    """
    victims = [
        (i, p) for i, p in _ACTIVE_PROCS if instruction is None or i == instruction
    ]
    for _, proc in victims:
        proc.terminate()
    for _, proc in victims:
        threading.Thread(
            target=_escalate_kill, args=(proc, _CANCEL_GRACE), daemon=True
        ).start()
    return len(victims)


def _escalate_kill(proc: subprocess.Popen[Any], grace: float) -> None:
    """取消升级线程：grace 内未退出则 kill（SIGKILL 兜底，见 _cancel_procs）。"""
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()


def _extract_resume_manifest(buf: list[str]) -> str | None:
    """从 stdout 缓存行提取 `--resume <manifest>` 路径（P3 取消/失败续跑引导）。

    cli_run 失败汇总打印「auto2dlabel run --resume outputs/batch_manifest.json」；
    倒序扫描取最近一条（机制 P4 会话任务摘要复用）。
    """
    import re

    for line in reversed(buf):
        m = re.search(r"--resume\s+(\S+)", line)
        if m:
            return m.group(1).rstrip("。，,.")
    return None


def _fmt_elapsed(seconds: float) -> str:
    """m:ss 计时（任务面板状态行「运行中 0:12 · 3/120」）。"""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


@dataclass
class Task:
    """后台任务条目（P3 任务面板单一事实源；widgets 按 task-<id>-* 挂载）。"""

    id: int
    instruction: str
    domain: str
    # 归属会话（P4+ 修复）：/new 后运行中任务的终态行必须落回发起时的会话，
    # 不能跟着当前 self._session_id 串到新会话
    session_id: str = ""
    status: str = "running"  # running | done | failed | cancelled
    started: float = field(default_factory=time.time)
    done: int = 0
    total: int | None = None  # None = 未知总量（ProgressBar 不确定态）
    elapsed: float = 0.0


_STATUS_LABELS = {"done": "✓ 完成", "failed": "✗ 失败", "cancelled": "⛔ 已取消"}


def run_tui(runner: InstructionRunner | None = None) -> None:
    """启动 TUI（P1 唯一入口；runner 注入供冒烟测试零真实执行）。"""
    app = ChatApp(runner=runner)
    app.run()


# textual 8.x py.typed 为 partial（App/work/on 解析为 Any）——行级 ignore 精确抑制
class ChatApp(App):  # type: ignore[misc]
    """对话式标注 TUI（P1 骨架 + P2 provider 切换 + P3 任务面板/取消）。"""

    TITLE = "AutoLabel"

    def __init__(
        self, runner: InstructionRunner | None = None, provider: str = "deepseek"
    ) -> None:
        super().__init__()
        self._runner: InstructionRunner = runner or _subprocess_run
        self._busy = False
        self._provider = provider
        # P3：任务面板状态（widgets 挂载在 #tasks 容器，经 call_from_thread 主线程更新）
        self._tasks: dict[int, Task] = {}
        self._next_task_id = 1
        # P2 冒烟断言兼容：最近一次进度上报
        self._progress: tuple[int, int | None] | None = None
        # P4：会话上下文（JSONL 逐行落盘；/resume 恢复后 seq 续接防覆盖）
        self._session_id = make_session_id()
        self._seq = 0
        self._new_count = 0  # /new 计数：同秒内多次开新会话时 sid 加 -N 后缀防撞车
        # P5：/web spawn 记录（which -> (pid, 启动时刻)）——10s 内重复 /web
        # 报「启动中」而非二次 spawn 重复起服务
        self._web_spawned: dict[str, tuple[int, float]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(WELCOME_TEXT), id="chat")
        yield Container(id="tasks")  # P3 任务面板：每任务 title + ProgressBar + 状态行
        yield Input(placeholder="输入标注指令（/help 查看命令）", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        banner = self._credential_banner()
        self.append(banner, markup=True, record=False)  # 横幅每次启动都有，不进会话
        self.query_one("#prompt", Input).focus()

    def on_unmount(self) -> None:
        """TUI 退出兜底清理（/quit 与 Ctrl+C 同路径）：terminate 全部运行中子进程。

        实证（2026-09-07）：TUI 退出后子进程写 pipe 得 BrokenPipeError 中途
        崩溃（批量任务 manifest 未写、白跑），输出少时继续占 GPU 数小时无人
        可取消——退出必须同步终止（daemon 线程随进程死，只能此处短 wait）。
        """
        if not _ACTIVE_PROCS:
            return
        for _, proc in list(_ACTIVE_PROCS):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass  # 进程已退出（未回收）
        for _, proc in list(_ACTIVE_PROCS):
            try:
                proc.wait(timeout=2)  # 短 grace：退出场景不再等待长 C 调用
            except subprocess.TimeoutExpired:
                proc.kill()

    def _credential_banner(self) -> str:
        """凭据状态横幅（has_credentials 与 create_client 同源，v0.6 红线）。"""
        from auto2dlabel.agent.llm import create_client

        client = create_client(provider=self._provider)
        display = _PROVIDER_DISPLAY.get(self._provider, self._provider)
        if client.has_credentials:
            return f"[green]✓ {display} 凭据已配置[/green]（模型 {client.model}）"
        return "[red]无 API 凭据 → 指令将走代码直跑降级（LLM 规划/评估不可用）[/red]"

    # ---------- 渲染 ----------

    def append(
        self,
        text: str,
        markup: bool = False,
        record: bool = True,
        kind: str = "system",
        domain: str = "",
        provider: str = "",
        task: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """对话区追加一行并滚动到底（markup=False = 纯文本防误解析）。

        record=True 时落会话 JSONL（P4）；逐行子进程回显与重放行显式
        record=False 防会话膨胀/重复写。session_id 覆盖归属会话（任务
        终态行跨 /new 落回发起时会话，见 _append_task_result）。
        """
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(Static(text, markup=markup))
        chat.scroll_end(animate=False)
        if record:
            self._seq += 1
            append_session_line(
                SESSION_LOG_PATH,
                session_id or self._session_id,
                self._seq,
                kind,
                text,
                domain=domain,
                provider=provider,
                markup=markup,
                task=task,
            )

    def _on_thread_append(self, text: str) -> None:
        """worker 线程 → 主线程回显（call_from_thread 唯一通道；不进会话）。"""
        self.append(text.rstrip("\n"), record=False)

    def _append_task_result(
        self, text: str, task: dict[str, Any], session_id: str | None = None
    ) -> None:
        """任务终态行（task_result 进会话，task 字段含 status/manifest）。

        session_id 指向任务发起时会话——/new 后运行中任务才结束的终态行
        必须落回原会话，否则跨会话串线（#18）。
        """
        self.append(text, kind="task_result", task=task, session_id=session_id)

    # ---------- 任务面板（P3） ----------

    def _task_status_line(self, task: Task) -> str:
        """状态行：运行中带计时与进度（0:12 · 3/120）；终态带档位标签。"""
        if task.status == "running":
            line = f"运行中 {_fmt_elapsed(task.elapsed)}"
            if task.total is not None:
                line += f" · {task.done}/{task.total}"
            return line
        line = _STATUS_LABELS[task.status]
        if task.total is not None:
            line += f" · {task.done}/{task.total}"
        return line

    def _create_task(self, instruction: str, domain: str) -> Task:
        """建任务条目 + 挂载面板行（title 截断 40 字；ProgressBar 初始不确定态）。"""
        task = Task(
            id=self._next_task_id,
            instruction=instruction,
            domain=domain,
            session_id=self._session_id,
        )
        self._next_task_id += 1
        self._tasks[task.id] = task
        title = instruction if len(instruction) <= 40 else instruction[:39] + "…"
        self.query_one("#tasks", Container).mount(
            Vertical(
                Static(f"#{task.id} {title}", id=f"task-{task.id}-title"),
                ProgressBar(total=None, id=f"task-{task.id}-bar"),
                Static(self._task_status_line(task), id=f"task-{task.id}-status"),
                id=f"task-{task.id}",
            )
        )
        return task

    def _on_task_progress(self, task_id: int, done: int, total: int | None) -> None:
        """主线程进度更新：Task 字段 + ProgressBar + 状态行（call_from_thread 通道）。

        多步计划步切换时 done/total 会回退（上一步 100/100 → 本步 0/1200），
        此处单调处理（done 只增、total 只增大）——进度条视觉不倒退；最后
        一步收尾时 done 追平其 total 后 bar 自然满格。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return
        if done < task.done:
            done = task.done  # 步切换回退 → 单调
        if total is not None and (task.total is None or total > task.total):
            task.total = total
        task.done = done
        task.elapsed = time.time() - task.started
        self._progress = (done, total)  # P2 冒烟断言兼容
        bar = self.query_one(f"#task-{task_id}-bar", ProgressBar)
        if total is not None:
            bar.total = total  # 不确定态 → 确定态
        bar.update(progress=done, total=task.total if task.total is not None else total)
        self.query_one(f"#task-{task_id}-status", Static).update(
            self._task_status_line(task)
        )

    def _on_task_terminal(self, task_id: int, status: str) -> None:
        """终态落定（done/failed；已取消不覆盖——/cancel 先置 cancelled）。"""
        task = self._tasks.get(task_id)
        if task is None or task.status != "running":
            return
        task.status = status
        task.elapsed = time.time() - task.started
        bar = self.query_one(f"#task-{task_id}-bar", ProgressBar)
        if task.total is None:
            bar.update(progress=1, total=1)  # 无进度上报的任务：终态补满
        else:
            bar.update(progress=task.done, total=task.total)
        self.query_one(f"#task-{task_id}-status", Static).update(
            self._task_status_line(task)
        )

    def _refresh_task_widgets(self, task: Task) -> None:
        """取消路径共用：状态行 + 进度条同步（total=None 终态补满）。"""
        bar = self.query_one(f"#task-{task.id}-bar", ProgressBar)
        if task.total is None:
            bar.update(progress=1, total=1)
        else:
            bar.update(progress=task.done, total=task.total)
        self.query_one(f"#task-{task.id}-status", Static).update(
            self._task_status_line(task)
        )

    # ---------- 输入 ----------

    @on(Input.Submitted, "#prompt")  # type: ignore[untyped-decorator]
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.startswith("/"):
            self._slash_command(text)
            return
        self._submit_instruction(text)

    def _slash_command(self, text: str) -> None:
        cmd, _, args = text.partition(" ")
        self.append(f"> {text}", kind="user")
        command = _SLASH_BY_NAME.get(cmd)
        if command is None:
            self.append(f"未知命令 {cmd}，输入 /help 查看可用命令")
            return
        command.handler(self, args)

    # ---------- 斜杠命令 handler（注册表见模块尾部） ----------

    def _cmd_help(self, args: str) -> None:
        self.append(HELP_TEXT)

    def _cmd_model(self, args: str) -> None:
        """P2 三态：无参列注册表 / set <name> 切换 / set-key 密钥指引。"""
        from auto2dlabel.agent.llm import PROVIDERS_YAML_PATH, create_client, list_providers

        sub, _, name = args.partition(" ")
        if sub == "set":
            spec = next((s for s in list_providers() if s.name == name), None)
            if spec is None:
                self.append(f"未知 provider '{name}'。可用: {[s.name for s in list_providers()]}")
                return
            self._provider = spec.name
            client = create_client(provider=spec.name)
            state = "凭据已配置" if client.has_credentials else "无凭据（代码直跑）"
            self.append(f"已切换 provider: {spec.name} | model: {client.model} | {state}")
            return
        if sub == "set-key":
            spec = next((s for s in list_providers() if s.name == self._provider), None)
            if spec is None or spec.api_key_env is None:
                self.append(f"provider {self._provider} 为本地免 key 端点，无需配置密钥")
                return
            env_path = PROVIDERS_YAML_PATH.parent / ".env"
            self.append(
                f"密钥不入库：编辑 {env_path}，写入 {spec.api_key_env}=<你的 key>，"
                "重启 TUI 生效（模板见同目录 .env.example）"
            )
            return
        if args:
            self.append(f"未知用法 '{args}'，可用: /model [set <name> | set-key]")
            return
        lines = ["provider 注册表（/model set <name> 切换，/model set-key 配置密钥）:"]
        for spec in list_providers():
            client = create_client(provider=spec.name)
            state = "凭据已配置" if client.has_credentials else "无凭据（代码直跑）"
            mark = "*" if spec.name == self._provider else " "
            lines.append(f" {mark} provider: {spec.name} | model: {client.model} | {state}")
        self.append("\n".join(lines))

    def _cmd_cost(self, args: str) -> None:
        from auto2dlabel.agent.llm import USAGE_LOG_PATH, aggregate_usage, load_usage_logs

        entries = load_usage_logs(USAGE_LOG_PATH)
        if not entries:
            self.append(f"台账为空或不存在: {USAGE_LOG_PATH}")
            return
        rows = aggregate_usage(entries)
        total = sum(float(r["cost_rmb"]) for r in rows)
        lines = [f"LLM usage 台账（{len(entries)} 条记录，{len(rows)} 个调用点 × 模型）:"]
        for r in rows:
            lines.append(
                f"  {r['call_site']} [{r['provider']}/{r['model']}] {r['calls']} 次 · "
                f"prompt {r['prompt_tokens']:,} / completion {r['completion_tokens']:,} · "
                f"¥{r['cost_rmb']:.6f}"
            )
        lines.append(f"合计费用: ¥{total:.6f}")
        self.append("\n".join(lines))

    def _cmd_new(self, args: str) -> None:
        """P4：新会话（旧会话已逐行落盘，无需 flush）——清屏 + 新 session_id + seq 归零。

        同秒内多次 /new：sid 粒度到秒，撞车时加 -N 后缀（#17）。
        """
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        new_id = make_session_id()
        if self._new_count > 0:
            new_id = f"{new_id}-{self._new_count + 1}"
        self._new_count += 1
        self._session_id = new_id
        self._seq = 0
        self.append(f"对话已清空，新会话 {self._session_id} 已开启", record=False)
        running = [t for t in self._tasks.values() if t.status == "running"]
        if running:
            self.append(
                f"[yellow]仍有 {len(running)} 个运行中任务（其终态仍记入原会话）[/yellow]",
                markup=True,
                record=False,
            )

    def _cmd_resume(self, args: str) -> None:
        """P4 两态：无参列最近 5 会话；<session_id> 逐行重放（不重跑任务——
        恢复只展示历史，续跑走 manifest 引导，防重复副作用）。"""
        entries = load_sessions(SESSION_LOG_PATH)
        arg = args.strip()
        if arg:
            rows = session_messages(entries, arg)
            if not rows:
                self.append(f"无会话 '{arg}'（输入 /resume 查看列表）")
                return
            # 恢复会话上下文：后续新消息续写该会话（seq 接续防覆盖）
            self._session_id = arg
            self._seq = max(safe_int(r.get("seq", 0)) for r in rows)
            for r in rows:
                kind = str(r.get("kind", "system"))
                # 严格布尔：损坏行 markup="false" 若走 bool() 会成 True，
                # 重放启用 markup → 历史文本注入 markup 解析
                markup = r.get("markup") is True
                text = str(r.get("text", ""))
                task = r.get("task")
                if kind == "task_result":
                    self.append(text, markup=markup, record=False, kind=kind, task=task)
                    if isinstance(task, dict):
                        manifest = task.get("manifest")
                        if manifest and task.get("status") != "done":
                            self.append(
                                f"batch 任务可续跑: auto2dlabel run --resume {manifest}",
                                record=False,
                            )
                else:
                    self.append(text, markup=markup, record=False, kind=kind)
            self.append(
                f"已恢复会话 {arg}（{len(rows)} 条消息；任务历史仅展示，"
                "续跑请按上方 manifest 引导）",
                record=False,
            )
            return
        summaries = list_sessions(entries)
        if not summaries:
            self.append(f"无历史会话: {SESSION_LOG_PATH}")
            return
        lines = ["历史会话（/resume <session_id> 恢复，倒序最近 5 个）:"]
        for s in summaries:
            lines.append(
                f"  {s['session_id']} | {s['messages']} 条 | {s['preview']}"
            )
        self.append("\n".join(lines))

    def _cmd_cancel(self, args: str) -> None:
        """P3 /cancel：无参取消全部运行中任务；<id> 单任务（terminate→wait→kill）。"""
        arg = args.strip()
        if arg:
            try:
                tid = int(arg)
            except ValueError:
                self.append(
                    f"未知任务 id '{arg}'（用法: /cancel [任务 id]，无参取消全部）"
                )
                return
            task = self._tasks.get(tid)
            if task is None:
                self.append(f"无任务 #{tid}")
                return
            if task.status != "running":
                self.append(f"任务 #{tid} 已结束（{_STATUS_LABELS[task.status]}），无需取消")
                return
            n = _cancel_procs(task.instruction)
            # 连带取消：_cancel_procs 按指令匹配，同指令并发任务的进程一并被
            # terminate——sibling 状态同步置 cancelled，否则停在其 worker
            # finally 里被记成 failed（误导：实为取消）。
            siblings = [
                t
                for t in self._tasks.values()
                if t.id != tid
                and t.status == "running"
                and t.instruction == task.instruction
            ]
            for t in [task, *siblings]:
                t.status = "cancelled"
                t.elapsed = time.time() - t.started
                self._refresh_task_widgets(t)
                self._append_task_result(
                    f"⛔ 已取消任务 #{t.id}",
                    {"id": t.id, "status": "cancelled", "manifest": None},
                    t.session_id,
                )
            verb = "（terminate 已发送，GPU 显存由进程退出回收）" if n else ""
            if siblings:
                verb += f"；同指令 {len(siblings)} 个连带任务一并取消"
            self.append(f"⛔ 已取消任务 #{tid}{verb}")
            return
        running = [t for t in self._tasks.values() if t.status == "running"]
        if not running:
            self.append("无运行中任务")
            return
        n = _cancel_procs()
        for t in running:
            t.status = "cancelled"
            t.elapsed = time.time() - t.started
            self._refresh_task_widgets(t)
            self._append_task_result(
                f"⛔ 已取消任务 #{t.id}",
                {"id": t.id, "status": "cancelled", "manifest": None},
                t.session_id,
            )
        verb = f"（{n} 个进程 terminate 已发送）" if n else ""
        self.append(f"⛔ 已取消 {len(running)} 个任务{verb}")

    def _cmd_review(self, args: str) -> None:
        """P5 HITL 三档分流统计（2D+3D；每次重扫无缓存——标注回流后
        重跑 /review 刷新天然成立）。已复核口径两域统一 = *_reviewed.json
        文件数；3D 另有 accepted（直接采纳）档单列，不混入已复核。"""
        from autolabel.tui.stats import scan_review_stats

        lines = ["HITL 三档分流统计（/web 启动复核，回流后重跑 /review 刷新）:"]
        for s in scan_review_stats():
            row = (
                f"  {s.domain} {s.dir}: 待复核 {s.pending_review} · "
                f"困难 {s.hard} · 已复核 {s.reviewed_files} · 队列文件 {s.total_files}"
            )
            if s.domain == "3d":
                row += f" · 接受标注 {s.accepted}"
            lines.append(row)
        self.append("\n".join(lines))

    def _cmd_web(self, args: str) -> None:
        """P5 /web [2d|3d]：启动 Web 复核服务（端口检测 → spawn detach →
        轮询就绪 → URL；已占用提示「已在运行」）。spawn 后 10s 内重复
        /web 报「启动中」不二次 spawn（服务就绪前端口检测不到，防重复起服务）。"""
        from autolabel.tui import webctl

        which = "3d" if args.strip() == "3d" else "2d"
        env_name, port = webctl.port_of(which)
        url = f"http://localhost:{port}"
        if webctl.port_open("127.0.0.1", port):
            self.append(f"{which} Web 复核已在运行: {url}")
            return
        if which in self._web_spawned:
            pid, spawned_at = self._web_spawned[which]
            if time.time() - spawned_at < 10.0:
                self.append(f"{which} Web 服务启动中（pid {pid}），稍候重试 /web")
                return
            del self._web_spawned[which]  # 超 10s 仍未就绪 → 允许重试
        pid = webctl.spawn_web_server(which)
        self._web_spawned[which] = (pid, time.time())
        if webctl.wait_port("127.0.0.1", port, timeout=10.0):
            del self._web_spawned[which]  # 已就绪：端口检测即可，登记完成使命
            self.append(f"{which} Web 复核已启动: {url}（pid {pid}；关闭: kill {pid}）")
        else:
            self.append(
                f"{which} Web 服务端口 {port} 10s 内未就绪（pid {pid}，"
                f"查看日志或 kill {pid} 后重试）"
            )

    def _cmd_quit(self, args: str) -> None:
        self.exit()

    # ---------- 指令执行 ----------

    def _submit_instruction(self, instruction: str) -> None:
        domain = route_domain(instruction)
        label = "3D" if domain == "3d" else "2D"
        self.append(f"> {instruction}", kind="user", domain=domain, provider=self._provider)
        self.append(f"[dim]路由: {label}[/dim]", markup=True)
        self._busy = True
        task = self._create_task(instruction, domain)
        self._run_worker(task, instruction, domain)

    @work(thread=True, exclusive=False)  # type: ignore[untyped-decorator]
    def _run_worker(self, task: Task, instruction: str, domain: str) -> None:
        """worker 线程执行指令：行回显 + 进度上报（任务面板）+ 终态/续跑引导。

        多任务并发不限（exclusive=False）；stdout 尾部缓存 100 行供 --resume
        引导行扫描（任务上下文在 worker 闭包内，零共享状态竞争）。
        """
        buf: list[str] = []  # stdout 尾部缓存（P3 --resume 引导行扫描，P4 复用）

        def on_line(line: str) -> None:
            buf.append(line)
            if len(buf) > 100:
                buf.pop(0)
            self.call_from_thread(self._on_thread_append, line)

        def on_progress(done: int, total: int | None) -> None:
            self.call_from_thread(self._on_task_progress, task.id, done, total)

        try:
            self._runner(instruction, domain, self._provider, on_line, on_progress)
            if task.status != "cancelled":
                self.call_from_thread(self._on_task_terminal, task.id, "done")
                self.call_from_thread(
                    self._append_task_result,
                    f"✓ 任务 #{task.id} 执行完成",
                    {"id": task.id, "status": "done", "manifest": None},
                    task.session_id,
                )
        except Exception as e:  # 执行器异常绝不带崩 TUI
            if task.status != "cancelled":
                self.call_from_thread(self._on_task_terminal, task.id, "failed")
                hint = _extract_resume_manifest(buf)
                self.call_from_thread(
                    self._append_task_result,
                    f"✗ 任务 #{task.id} 执行失败: {e}",
                    {"id": task.id, "status": "failed", "manifest": hint},
                    task.session_id,
                )
                if hint:
                    self.call_from_thread(
                        self._on_thread_append,
                        f"batch 任务可续跑: auto2dlabel run --resume {hint}",
                    )
        finally:
            self.call_from_thread(self._set_busy, False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy


# ---------- 斜杠命令注册表（v1.0 P1+，借鉴 claude-code 命令声明式集中模式） ----------


@dataclass(frozen=True)
class SlashCommand:
    """斜杠命令条目：新增命令 = 加一行 + handler 方法，/help 文案自动跟上。"""

    name: str  # 含 "/" 前缀（如 "/help"）
    description: str  # 一行说明（进 /help 文案）
    handler: Callable[[ChatApp, str], None]  # 绑定方法 (app, args) → None；args 可为 ""


SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("/help", "显示本帮助", ChatApp._cmd_help),
    SlashCommand("/model", "列出/切换 LLM provider（/model set <name>）", ChatApp._cmd_model),
    SlashCommand("/cost", "LLM usage 台账聚合（logs/llm_usage.jsonl）", ChatApp._cmd_cost),
    SlashCommand("/new", "清空对话并开启新会话", ChatApp._cmd_new),
    SlashCommand("/resume", "恢复会话（无参列表 / <id> 重放）", ChatApp._cmd_resume),
    SlashCommand("/cancel", "取消任务（无参全部 / <id> 单个）", ChatApp._cmd_cancel),
    SlashCommand("/review", "HITL 三档分流统计（2D+3D）", ChatApp._cmd_review),
    SlashCommand("/web", "启动 Web 复核（2D 默认 / 3d 独立）", ChatApp._cmd_web),
    SlashCommand("/quit", "退出", ChatApp._cmd_quit),
]

_SLASH_BY_NAME: dict[str, SlashCommand] = {c.name: c for c in SLASH_COMMANDS}


def build_help_text() -> str:
    """由注册表生成 /help 文案（行格式与旧硬编码逐字一致：2 空格 + 命令名对齐 9 列）。"""
    lines = ["斜杠命令："]
    lines += [f"  {c.name:<9}{c.description}" for c in SLASH_COMMANDS]
    lines.append(
        "直接输入自然语言指令即执行标注（自动路由 2D/3D，如「检测 000860.png 中的汽车」"
        " /「标注 KITTI 帧 000123 中的汽车和行人」）"
    )
    return "\n".join(lines)


HELP_TEXT = build_help_text()


def _subprocess_run(
    instruction: str,
    domain: str,
    provider: str,
    on_line: Callable[[str], None],
    on_progress: Callable[[int, int | None], None],
) -> None:
    """默认执行器：subprocess 跑旧命令 chat（进程隔离 + 行级回显）。

    子进程 stdout 是 pipe → rich 自动去色/markup，行级读即伪流式回显；
    真实 token 级流式在子进程内由 chat 的 LLM 流式改造承担（v1.0 P1）。
    `[AL_PROGRESS] done/total` 协议行（P3 子进程进度通道）→ on_progress，
    不回显对话区（防刷屏）。Popen 登记 _ACTIVE_PROCS 供 /cancel 两级终止
    （子进程 SIGTERM → KeyboardInterrupt → finally 链清理，见 cli 侧
    install_sigterm_interrupt），完成/异常统一出表。
    """
    module = "auto3dlabel.cli" if domain == "3d" else "auto2dlabel.cli"
    # -u unbuffered：子进程 stdout 连 pipe 时 Python 默认全缓冲，
    # 输出会积压到进程退出才 flush（TUI 表现为「无响应」）——必须无缓冲
    cmd = [
        sys.executable, "-u", "-m", module, "chat", instruction,
        "--no-wait", "--provider", provider,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # 协议行开关（cli_common.print_al_progress 门控）：只在 TUI 子进程
        # 开启，普通终端 CLI 输出零污染
        env={**os.environ, "AUTOLABEL_PROGRESS": "1"},
    )
    _register_proc(instruction, proc)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            if stripped.startswith("[AL_PROGRESS]"):
                done_s, _, total_s = stripped[len("[AL_PROGRESS]") :].strip().partition("/")
                try:
                    on_progress(int(done_s), int(total_s))
                except ValueError:
                    pass  # 协议行解析失败不阻断执行
                continue
            on_line(line)
        if proc.wait() != 0:
            # 错误详情已在子进程 stdout 回显（typer 报错等），此处只报退出码，
            # 不再重复整条命令（指令文本本身已显示在对话区）
            raise RuntimeError(f"子进程退出码 {proc.returncode}（详见上方输出）")
    finally:
        _unregister_proc(proc)
