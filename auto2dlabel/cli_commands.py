"""轻量命令实现 —— chat（计划交互）/ sample（主动学习采样）/ tools（工具列表）。"""

from __future__ import annotations

import typer
from rich.table import Table

from auto2dlabel.agent.llm import create_client
from auto2dlabel.agent.planner import TaskPlanner
from auto2dlabel.cli_common import collect_images, console, setup_logging
from auto2dlabel.cli_execute import execute_plan
from auto2dlabel.schema.task_plan import TaskPlan


def sample_command(top_k: int, review_dir: str, output: str) -> None:
    """`sample` 命令实现：主动学习采样。"""
    from auto2dlabel.tools.sampling import (
        collect_review_pool,
        sample_priority,
        write_priority_manifest,
    )

    pool = collect_review_pool(review_dir)
    if not pool:
        console.print(f"[yellow]复核队列为空（{review_dir}）[/yellow]")
        return
    selected = sample_priority(pool, top_k=top_k)
    out = write_priority_manifest(selected, output)
    console.print(f"[green]✓ 采样清单: {out}[/green]（{len(selected)}/{len(pool)} 张）")
    for i, item in enumerate(selected, 1):
        console.print(
            f"  {i}. {item.file}  score={item.score:.3f}  "
            f"unc={item.mean_uncertainty:.3f}  classes={item.class_count}"
        )


def tools_command() -> None:
    """`tools` 命令实现：列出所有已注册 Tool。"""
    from auto2dlabel.tools.detection import register as reg_detect
    from auto2dlabel.tools.export import register as reg_export

    reg_detect()
    reg_export()

    # Lazy import to avoid requiring SAM2 for CLI help
    try:
        from auto2dlabel.tools.segmentation import register as reg_seg

        reg_seg()
    except ImportError:
        pass

    from auto2dlabel.tools.registry import registry

    table = Table(title="Registered Annotation Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    for name in registry.list_tools():
        tool = registry.get(name)
        desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
        table.add_row(name, desc)

    console.print(table)


def chat_command(
    instruction: str | None,
    det_model: str | None,
    confirm_timeout: int,
    no_wait: bool,
    provider: str,
    verbose: bool,
    sahi: bool,
    batch_size: int | None = None,
    num_workers: int | None = None,
    viz: bool = True,
    batch_strategy: bool = False,
    refer_l2: bool = False,
    refer_l3: bool = False,
) -> None:
    """`chat` 命令实现：自然语言解析 → 缺失参数追问 → 确认 → 执行 TaskPlan。"""
    setup_logging(verbose)

    from auto2dlabel.tools.confirm import ask_with_timeout

    llm = create_client(provider=provider)
    planner = TaskPlanner(llm_client=llm)
    timeout = 0 if no_wait else confirm_timeout

    # ---- Step 0: 设备信息 ----
    from auto2dlabel.tools.device import print_device
    print_device()

    # ---- Step 1: 获取指令 ----
    if instruction:
        user_text = instruction
    else:
        console.print("[bold]AutoLabel Chat 模式[/bold]")
        console.print("[dim]输入自然语言指令，例如: 检测 /data/images/ 中的汽车和行人[/dim]")
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出[/dim]")
            return
        if not user_text:
            console.print("[red]指令为空，退出[/red]")
            return

    console.print("\n[dim]正在解析指令...[/dim]")

    # ---- Step 2: LLM 解析 → TaskPlan ----
    try:
        plan = planner.parse(user_text, confirm_timeout=timeout)
    except Exception as e:
        console.print(f"[red]解析失败: {e}[/red]")
        raise typer.Exit(code=1)

    # 如果 CLI 指定了 -d，覆盖所有步骤的模型
    if det_model:
        for step in plan.steps:
            step.model_name = det_model

    # ---- Step 3: 缺失参数追问 ----
    _fill_missing_params(plan, planner, timeout)

    # ---- Step 3.5: 类别推荐 ----
    _maybe_recommend_classes(plan, timeout, no_wait)

    # ---- Step 3.8: 批量推理超参数（CLI 显式 > 交互询问 > GPU 显存推荐兜底）----
    _fill_batch_params(plan, timeout, no_wait, batch_size=batch_size, num_workers=num_workers)

    # ---- Step 4: 展示计划 + 确认 ----
    if not no_wait:
        confirm_msg = f"[bold]执行计划如下：[/bold]\n{plan.summary}"
        result = ask_with_timeout(confirm_msg, timeout=timeout)
        if not result.confirmed:
            console.print("[yellow]用户取消[/yellow]")
            return
        if result.user_input:
            try:
                plan = planner.parse(
                    plan.raw_instruction + " " + result.user_input,
                    confirm_timeout=timeout,
                )
                console.print(f"[dim]参数已更新: {plan.summary}[/dim]")
            except Exception:
                console.print("[yellow]无法解析修改，使用原计划[/yellow]")

    # ---- Step 4.5: 跟踪器选择（代码级扫描指令，LLM 不参与）----
    from auto2dlabel.tools.tracking import detect_tracker_kind

    use_bot_sort = detect_tracker_kind(plan.raw_instruction) == "bot_sort"

    # ---- Step 4.6: 指代约束解析（v0.4 3a，代码级；非跟踪步骤时无副作用）----
    from auto2dlabel.tools.constraints import has_relation, parse_referential

    constraint = None
    if any(s.task_type == "tracking" for s in plan.steps):
        try:
            constraint = parse_referential(plan.raw_instruction)
            if not constraint.is_plain:
                console.print(
                    f"[dim]指代约束: 属性 {constraint.attributes}  方位 {constraint.position}[/dim]"
                )
        except ValueError:
            pass  # 无类别关键词时 planner 已兜底，约束保持 None

    # ---- Step 4.7: 指代 L2/L3 触发（v0.5；显式 flag 或关系词自动）----
    use_l2 = refer_l2 or has_relation(plan.raw_instruction)
    use_l3 = refer_l3  # L3 仅显式直用；默认指代路径为阶梯升级（L2 失败自动升级 L3）
    if use_l3:
        console.print("[dim]指代 L3 (Qwen2-VL-7B): 跟踪首帧解析锁定目标[/dim]")
    elif use_l2:
        console.print("[dim]指代 L2 (Florence-2, 失败升级 L3): 跟踪首帧解析锁定目标[/dim]")

    # ---- Step 5: 执行 TaskPlan ----
    console.print(f"\n[bold]开始执行 {len(plan.steps)} 步任务...[/bold]\n")
    execute_plan(
        plan,
        sahi=sahi,
        explicit_batch_size=batch_size,
        explicit_num_workers=num_workers,
        use_bot_sort=use_bot_sort,
        viz=viz,
        constraint=constraint,
        batch_strategy=batch_strategy,
        llm=llm,
        refer_l2=use_l2,
        refer_l3=use_l3,
    )

    console.print("\n[bold green]✓ 全部任务完成[/bold green]")


def _maybe_recommend_classes(plan: TaskPlan, timeout: int, no_wait: bool) -> None:
    """如果 TaskPlan 中有步骤缺少 prompts，运行类别推荐。"""
    from pathlib import Path

    from auto2dlabel.tools.confirm import ask_with_timeout
    from auto2dlabel.tools.recommend import format_recommendation, recommend_classes, scan_classes

    for step in plan.steps:
        if step.task_type == "tracking":
            continue  # 固定类别跟踪不走类别推荐（视频 source 也无法 scan_classes）
        if step.prompts:
            continue  # 用户已指定类别，不需要推荐

        source = Path(step.source)
        if not source.exists():
            continue

        # 如果是目录，取第一张图做扫描
        images = collect_images(source, batch=True) if source.is_dir() else [source]
        if not images:
            continue

        console.print("\n[bold]类别推荐模式[/bold] — 扫描图像以检测可标注的类别...")
        stats = scan_classes(str(images[0]))

        recommended, skipped, _ = recommend_classes(stats)

        if not recommended:
            console.print("[yellow]未检出足够置信度的类别。建议手动指定。[/yellow]")
            continue

        msg = format_recommendation(recommended, skipped, stats)

        if no_wait:
            step.prompts = recommended
            console.print(f"[dim]--no-wait: 自动选择推荐类别 {recommended}[/dim]")
            continue

        result = ask_with_timeout(msg, timeout=timeout)
        if result.user_input and result.user_input.lower() != "ok":
            # 用户自定义类别
            user_classes = [
                c.strip() for c in result.user_input.replace("，", ",").split(",") if c.strip()
            ]
            step.prompts = user_classes
            console.print(f"[dim]已设置类别: {step.prompts}[/dim]")
        else:
            step.prompts = recommended
            console.print(f"[dim]已自动选择: {recommended}[/dim]")


def _parse_batch_input(text: str) -> tuple[int, int] | None:
    """解析 'batch_size=8 num_workers=4' 或 '8 4' → (batch_size, num_workers)。

    要求两个值成对给出；非法返回 None。
    """
    import re

    bs_m = re.search(r"batch_size\s*[=:：]\s*(\d+)", text)
    nw_m = re.search(r"num_workers\s*[=:：]\s*(\d+)", text)
    if bs_m and nw_m:
        return int(bs_m.group(1)), int(nw_m.group(1))
    if bs_m or nw_m:
        return None  # 只给了半个 → 要求成对
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if len(nums) >= 2 and nums[0] >= 1:
        return nums[0], nums[1]
    return None


def _fill_batch_params(
    plan: TaskPlan,
    timeout: int,
    no_wait: bool,
    batch_size: int | None,
    num_workers: int | None,
) -> None:
    """批量推理超参数来源：CLI 显式 > chat 交互询问 > 执行阶段动态实测。

    - CLI 指定 → 覆盖全部步骤（经 execute_plan 的 explicit_* 透传恒优先）
    - 未指定且非 --no-wait：交互询问一次（非法输入重问 ≤3 次，超时/空输入跳过）
    - 仍未定 → 保持 None，执行阶段模型加载后动态实测（测单图峰值显存算最大
      batch；实测不可用时回退静态档位表）
    """
    from auto2dlabel.schema.task_plan import detect_gpu_memory_gb, recommend_batch_params
    from auto2dlabel.tools.confirm import ask_with_timeout

    # 1) CLI 显式指定
    if batch_size is not None or num_workers is not None:
        for step in plan.steps:
            if batch_size is not None:
                step.batch_size = batch_size
            if num_workers is not None:
                step.num_workers = num_workers
        return

    # 2) 交互询问（仅当有步骤缺参数且非 --no-wait）
    needs_ask = any(s.batch_size is None or s.num_workers is None for s in plan.steps)
    if needs_ask and not no_wait and plan.steps:
        gpu_mem = detect_gpu_memory_gb()
        first = plan.steps[0]
        rec_bs, rec_nw = recommend_batch_params(first.task_type, gpu_mem)
        msg = (
            "[bold]批量推理超参数[/bold]（留空回车 = 自动实测最大 batch，"
            f"静态参考 batch_size={rec_bs} num_workers={rec_nw}）\n"
            "  格式: batch_size=N num_workers=M"
        )
        for _ in range(3):
            result = ask_with_timeout(msg, timeout=timeout)
            if not result.confirmed or not result.user_input:
                break
            parsed = _parse_batch_input(result.user_input)
            if parsed is None:
                console.print("[yellow]格式无效，请输入如: batch_size=8 num_workers=4[/yellow]")
                continue
            bs, nw = parsed
            for step in plan.steps:
                step.batch_size = bs
                step.num_workers = nw
            console.print(f"[dim]已设置 batch_size={bs}, num_workers={nw}[/dim]")
            return
        console.print("[dim]未指定 → 执行阶段自动实测（模型加载后测最大 batch）[/dim]")

    # 3) 未定值保持 None → 执行阶段动态实测（模型加载后测单图峰值显存算最大 batch；
    #    实测不可用时由 tools/device.resolve_batch_params 回退静态表兜底）


def _fill_missing_params(plan: TaskPlan, planner: TaskPlanner, timeout: int) -> None:
    """如果 TaskPlan 有缺失参数，追问用户。"""
    from auto2dlabel.tools.confirm import ask_missing_params

    missing = plan.all_missing_params
    if not missing:
        return

    reply = ask_missing_params(missing, timeout=timeout)
    if reply:
        try:
            updated = planner.parse(
                plan.raw_instruction + " " + reply,
                confirm_timeout=timeout,
            )
            plan.steps = updated.steps
        except Exception:
            pass
