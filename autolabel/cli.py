"""autolabel 统一入口 CLI（docs/Agentic_UI_plan.md §9.1 命令矩阵）。

- 无参 → Textual TUI（对话界面）
- `autolabel "指令" [选项]` → 一次性对话：route_domain 路由 → 直接调用现有
  对话函数（2D `cli_commands.chat_command` / 3D `auto3dlabel.cli.chat`），零复制
- 旧 `auto2dlabel` / `auto3dlabel` 命令原样保留，零破坏
"""

from __future__ import annotations

from pathlib import Path

import typer

from autolabel.route import route_domain

app = typer.Typer(
    name="autolabel",
    help="Agentic 标注统一入口：无参进入 TUI 对话；带指令直接执行（自动路由 2D/3D）。",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    instruction: str = typer.Argument(None, help="自然语言标注指令（省略则进入 TUI 对话界面）"),
    det_model: str = typer.Option(
        None, "--det-model", "-d", help="检测模型（3D 引擎名强制路由 3D）"
    ),
    no_wait: bool = typer.Option(False, "--no-wait", help="跳过所有确认与追问，直接执行"),
    provider: str = typer.Option("deepseek", "--provider", "-p", help="LLM provider"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志输出"),
    sahi: bool = typer.Option(False, "--sahi", help="启用 SAHI 切片推理（2D 大分辨率图像）"),
    out_dir: str = typer.Option(
        None, "--out-dir", "-o", help="输出目录（3D；默认 outputs/kitti3d）"
    ),
    max_iterations: int = typer.Option(3, "--max-iterations", help="3D Agent Loop 最大迭代"),
) -> None:
    """统一入口：无参 → TUI；指令 → 一次性对话（route_domain 路由执行）。"""
    if ctx.invoked_subcommand is not None:
        return
    if not instruction:
        from autolabel.tui.app import run_tui

        run_tui()
        return

    domain = route_domain(instruction, det_model)
    if domain == "3d":
        _run_3d_chat(instruction, det_model, provider, out_dir, max_iterations)
    else:
        _run_2d_chat(instruction, det_model, no_wait, provider, verbose, sahi)


def _run_2d_chat(
    instruction: str,
    det_model: str | None,
    no_wait: bool,
    provider: str,
    verbose: bool,
    sahi: bool,
) -> None:
    """2D 一次性对话：直接调用 chat_command（零复制；confirm_timeout=0 跳过确认）。"""
    from auto2dlabel.cli_commands import chat_command

    chat_command(
        instruction=instruction,
        det_model=det_model,
        confirm_timeout=0,
        no_wait=no_wait,
        provider=provider,
        verbose=verbose,
        sahi=sahi,
    )


def _run_3d_chat(
    instruction: str,
    det_model: str | None,
    provider: str,
    out_dir: str | None,
    max_iterations: int,
) -> None:
    """3D 一次性对话：直接调用 3D chat 命令体（typer 装饰器返回原函数，可编程调用）。"""
    from auto3dlabel.cli import chat as chat3d

    chat3d(
        instruction=instruction,
        det_model=det_model,
        out_dir=Path(out_dir) if out_dir else Path("outputs/kitti3d"),
        provider=provider,
        max_iterations=max_iterations,
        no_wait=True,
    )


if __name__ == "__main__":
    app()
