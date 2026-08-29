"""轻量命令实现 —— chat（计划交互）/ sample（主动学习采样）/ tools（工具列表）。"""

from __future__ import annotations

import logging
import re
from functools import partial

import typer
from rich.table import Table

from auto2dlabel.agent.dialog import ask_questions
from auto2dlabel.agent.llm import create_client
from auto2dlabel.agent.planner import TaskPlanner
from auto2dlabel.cli_common import collect_images, console, setup_logging
from auto2dlabel.cli_execute import execute_plan
from auto2dlabel.schema.task_plan import TaskPlan

logger = logging.getLogger(__name__)


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

    # ---- Step 2: LLM 解析 → TaskPlan（v0.6 对话式：缺参多轮追问，降级链零回退）----
    try:
        plan = _parse_plan_dialog(planner, user_text, timeout)
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
                plan = _reparse_confirm_edit(planner, plan, result.user_input, timeout)
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
    """如果 TaskPlan 中有步骤缺少 prompts，运行类别推荐。

    v0.6 P2 落地形态：类别推荐保持代码级（模型类别表知识，代码级更准，
    对话轮 questions 不涉及类别推荐）；用户表达「不知道标什么类别」→
    LLM 留空 prompts → 本函数扫描首图统计 + 用户确认（回喂确认闭环）。
    """
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


def _parse_plan_dialog(planner: TaskPlanner, user_text: str, timeout: int) -> TaskPlan:
    """对话式解析（v0.6 P1）：parse_dialog 多轮收集 → 异常降级单轮 parse。

    无 key 零影响（v0.6 验收）：无凭据时零 LLM 调用，代码级提取类别 +
    全默认参数构建计划（缺参交 _fill_missing_params 代码级追问）。
    降级链零回退：对话层异常（LLM 非法响应 / 网络 / 骨架 bug）→ 日志 +
    单轮 parse 兜底（v0.5 行为）；单轮仍失败由调用方红字 + typer.Exit(1)。
    """
    if not planner.llm.has_credentials:
        logger.info("无 API key → 代码兜底构建默认计划（零 LLM 调用）")
        return _plan_without_llm(user_text)
    try:
        return planner.parse_dialog(
            user_text,
            ask_fn=partial(ask_questions, timeout=timeout),
            confirm_timeout=timeout,
        )
    except Exception as e:
        logger.warning("对话解析降级为单轮 parse: %s", e)
        return planner.parse(user_text, confirm_timeout=timeout)


def _plan_without_llm(user_text: str) -> TaskPlan:
    """无 key 代码兜底计划（v0.6 验收「无 key 零影响」）：全默认参数。

    prompts 用 CN_EN_MAP 代码级提取（中英映射兜底红线，LLM 不参与）；
    无关键词 → prompts 留空（不硬造类别，误标比少标更糟，交 Step 3.5
    类别推荐 / _fill_missing_params 接管）。task_type 恒 object_detection：
    任务型语义（跟踪/分割等）在 chat 内由 LLM 判定，无 key 场景保守取
    默认检测（复杂任务请配 key 或走 run --track）。
    """
    from auto2dlabel.schema.task_plan import TaskStep
    from auto2dlabel.tools.prompts import extract_prompts

    prompts: list[str] = []
    try:
        prompts = extract_prompts(user_text)
    except ValueError:
        pass
    plan = TaskPlan(steps=[TaskStep(step_id=1, source="", prompts=prompts)])
    plan.raw_instruction = user_text
    return plan


def _reparse_confirm_edit(
    planner: TaskPlanner, plan: TaskPlan, user_input: str, timeout: int
) -> TaskPlan:
    """Step 4 确认修改重解析（v0.6 G3 修复）。

    对话收集的参数只存 dialog_context、不在 raw_instruction——重解析必须带上，
    否则用户确认时改一句参数，对话成果全丢、字段回归缺失。
    重解析后的 dialog_context 写回累积文本（连续多次修改时链条完整，
    否则第二次修改 base 退化为 raw_instruction、对话字段再次回归缺失）。
    raw_instruction 还原为「原指令 + 用户修改」（v0.5 语义：Step 4.5+
    代码级扫描能看到修改文本，但看不到对话回喂噪声）。
    """
    original = plan.raw_instruction
    base = plan.dialog_context or original
    updated = planner.parse(f"{base} {user_input}", confirm_timeout=timeout)
    updated.dialog_context = f"{base} {user_input}"
    updated.raw_instruction = f"{original} {user_input}"
    return updated


def _fill_missing_params(plan: TaskPlan, planner: TaskPlanner, timeout: int) -> None:
    """如果 TaskPlan 有缺失参数，追问用户。"""
    from auto2dlabel.tools.confirm import ask_missing_params

    missing = plan.all_missing_params
    if not missing:
        return

    reply = ask_missing_params(missing, timeout=timeout)
    if reply:
        if not planner.llm.has_credentials:
            _apply_reply_without_llm(plan, reply)  # v0.6：无 key 代码级字段映射
            return
        try:
            updated = planner.parse(
                plan.raw_instruction + " " + reply,
                confirm_timeout=timeout,
            )
            plan.steps = updated.steps
        except Exception:
            pass


def _apply_reply_without_llm(plan: TaskPlan, reply: str) -> None:
    """无 key 代码级补参（v0.6 零影响链）：自由文本 → 路径/类别字段。

    LLM 缺席时的最小映射（v0.5 语义等价）：首段含 "/" 的路径 token →
    source；extract_prompts → prompts。只填缺失字段，提取不到保持缺参
    （Step 3.5 类别推荐 / execute 校验兜底，宁缺勿错）。
    """
    from auto2dlabel.tools.prompts import extract_prompts

    path_m = re.search(r"[^\s，,]*/[^\s，,]*", reply)
    prompts: list[str] = []
    try:
        prompts = extract_prompts(reply)
    except ValueError:
        pass
    for step in plan.steps:
        if not step.source and path_m:
            step.source = path_m.group(0)
        if not step.prompts and prompts:
            step.prompts = prompts
