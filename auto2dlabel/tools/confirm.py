"""超时确认工具 —— Plan 模式风格的交互确认。

用户确认 / 超时自动继续 / 修改参数后重新确认。
"""

from __future__ import annotations

import threading


class ConfirmResult:
    """确认结果。"""

    def __init__(self, confirmed: bool, user_input: str | None = None):
        self.confirmed = confirmed  # True = 继续执行, False = 用户取消
        self.user_input = user_input  # 用户的修改文本（可能为 None）


def ask_with_timeout(
    prompt: str,
    timeout: int = 30,
    default_confirm: bool = True,
) -> ConfirmResult:
    """向用户展示确认信息，等待回复或超时自动继续。

    Args:
        prompt: 展示给用户的信息。
        timeout: 等待秒数。0 表示不等待直接通过。
        default_confirm: 超时后是否自动确认。

    Returns:
        ConfirmResult。
    """
    if timeout <= 0:
        print(prompt)
        print("[auto] --no-wait 模式，跳过确认，自动执行。")
        return ConfirmResult(confirmed=True)

    print(prompt)
    print(
        f"\n[yellow]等待 {timeout} 秒后自动执行。"
        "回复 'cancel' 取消，或直接输入修改指令...[/yellow]"
    )

    user_input: list[str | None] = [None]
    lock = threading.Lock()

    def _read_input() -> None:
        try:
            inp = input("> ").strip()
            with lock:
                user_input[0] = inp if inp else None
        except (EOFError, KeyboardInterrupt):
            with lock:
                user_input[0] = None

    reader = threading.Thread(target=_read_input, daemon=True)
    reader.start()
    reader.join(timeout=timeout)

    with lock:
        reply = user_input[0]

    if reader.is_alive():
        # 超时
        print(f"\n[dim]超时 {timeout}s，自动继续...[/dim]")
        return ConfirmResult(confirmed=True)
    elif reply is None:
        # Ctrl+C / EOF
        print("\n[red]已取消[/red]")
        return ConfirmResult(confirmed=False)
    elif reply.lower() in ("cancel", "c", "no", "n", "取消"):
        print("[red]用户取消[/red]")
        return ConfirmResult(confirmed=False)
    elif reply.lower() in ("ok", "y", "yes", "确认", "好", "继续"):
        print("[green]用户确认，立即执行[/green]")
        return ConfirmResult(confirmed=True)
    else:
        # 用户输入了修改指令
        print(f"[dim]收到修改: {reply}[/dim]")
        return ConfirmResult(confirmed=True, user_input=reply)


def ask_text(prompt: str, timeout: int = 30) -> str | None:
    """收集用户自由文本回答（对话式规划用，v0.6）。

    与 ask_with_timeout 的差异：**不解释确认/取消词典**——"ok"/"好"/"no" 等
    一律按普通回答文本返回（对话阶段用户回答"好"不应被吞成确认）。
    取消语义（cancel/取消/no/n / 超时 / 空回车 / EOF）→ None = 放弃对话，
    由调用方走代码兜底（_fill_missing_params 等），与 --no-wait 语义一致。

    Returns:
        用户输入文本（strip 后），或 None 表示放弃对话。
    """
    if timeout <= 0:
        print(prompt)
        print("[auto] --no-wait 模式，跳过对话，使用计划默认值。")
        return None

    print(prompt)
    print(
        f"\n[yellow]等待 {timeout} 秒后跳过对话。"
        "回复 'cancel' 放弃补充，或直接输入回答...[/yellow]"
    )

    user_input: list[str | None] = [None]
    lock = threading.Lock()

    def _read_input() -> None:
        try:
            inp = input("> ").strip()
            with lock:
                user_input[0] = inp if inp else None
        except (EOFError, KeyboardInterrupt):
            with lock:
                user_input[0] = None

    reader = threading.Thread(target=_read_input, daemon=True)
    reader.start()
    reader.join(timeout=timeout)

    with lock:
        reply = user_input[0]

    if reader.is_alive():
        # 超时
        print(f"\n[dim]超时 {timeout}s，跳过对话，使用计划默认值...[/dim]")
        return None
    elif reply is None or reply.lower() in ("cancel", "c", "no", "n", "取消"):
        print("[dim]跳过对话，使用计划默认值继续。[/dim]")
        return None
    return reply


def ask_missing_params(
    missing: dict[int, list[str]],
    timeout: int = 30,
) -> str | None:
    """追问缺失参数。

    Args:
        missing: {step_id: [missing_param_names]}。
        timeout: 等待秒数。

    Returns:
        用户输入的补充信息，或 None 表示超时/取消。
    """
    lines = ["检测到以下参数缺失，请补充："]
    for step_id, params in missing.items():
        lines.append(f"  Step {step_id}: {', '.join(params)}")

    lines.append("")
    lines.append("例如回复: \"数据在 /data/images/，检测汽车和行人\"")
    lines.append("直接回车或超时将使用默认值继续。")

    result = ask_with_timeout("\n".join(lines), timeout=timeout, default_confirm=True)
    return result.user_input
