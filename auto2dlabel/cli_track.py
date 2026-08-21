"""`run --track` 薄壳 —— 指令解析（LLM/代码级）后委托 TrackingTool 执行。

管线实现在 tools/tracking.py（TrackingTool，与 chat 的 execute_plan track 分支
共用）；本模块只保留 CLI 侧能力：序列级一次性 LLM 指令解析（--llm，
extract_prompts 处理不了的复杂指令如「跟踪穿红衣服的人」，失败回退代码级）
与错误收敛（ValueError → console 红字 + typer.Exit）。
"""

from __future__ import annotations

import typer

from auto2dlabel.cli_common import console
from auto2dlabel.tools.tracking import DEFAULT_REID_MODEL, TrackingTool


def resolve_plan(
    instruction: str,
    use_llm: bool,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    threshold: float,
) -> tuple[list[str], float]:
    """跟踪规划入口：--llm 时序列级一次性 LLM 解析（失败回退），否则代码级解析。

    LLM 覆盖 extract_prompts 处理不了的复杂指令（如「跟踪穿红衣服的人」）；
    每序列仅 1 次调用，绝不逐帧。extract_prompts 的 ValueError 上抛（调用方转 Exit）。
    """
    if use_llm:
        try:
            prompts, llm_threshold = _llm_plan_once(
                instruction, provider, model, api_key, base_url)
            console.print(f"[dim]LLM 规划: 类别 {', '.join(prompts)}[/dim]")
            return prompts, llm_threshold if llm_threshold is not None else threshold
        except Exception as e:
            console.print(f"[yellow]LLM 规划失败，回退 extract_prompts: {e}[/yellow]")
    from auto2dlabel.tools.prompts import extract_prompts

    return extract_prompts(instruction), threshold


def _llm_plan_once(
    instruction: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> tuple[list[str], float | None]:
    """一次性 LLM 指令解析 → (prompts, threshold 建议；None=沿用 CLI 阈值)。

    单轮 tool-less 调用，JSON 输出。prompts 字段对齐 TaskPlan schema（为其子集，
    未来升级完整 planner 编排时无损）。输出无效时抛 ValueError/JSONDecodeError。
    """
    import json

    from auto2dlabel.agent.llm import create_client

    client = create_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是自动标注系统的跟踪指令解析器。把用户指令解析为 JSON：\n"
                    '{"prompts": ["COCO 英文类别名"], "threshold": 0到1的浮点数或 null}\n'
                    "threshold=null 表示沿用 CLI 阈值。只输出 JSON，不要其他文字。"
                ),
            },
            {"role": "user", "content": instruction},
        ],
        temperature=0.0,
    )
    content = (response.content or "").strip()
    # 剥离可能的 markdown 代码块
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)

    raw_prompts = data.get("prompts") or []
    if not raw_prompts or not all(isinstance(p, str) for p in raw_prompts):
        raise ValueError(f"LLM 规划输出无效 prompts: {data}")
    prompts = [str(p) for p in raw_prompts]

    raw_threshold = data.get("threshold")
    llm_threshold: float | None
    if raw_threshold is None:
        llm_threshold = None
    elif (
        isinstance(raw_threshold, (int, float))
        and not isinstance(raw_threshold, bool)
        and 0 <= raw_threshold <= 1
    ):
        llm_threshold = float(raw_threshold)
    else:
        raise ValueError(f"LLM 规划输出无效 threshold: {data}")
    return prompts, llm_threshold


def run_tracking(
    source: str,
    instruction: str,
    threshold: float,
    iou: float,
    export: str,
    output: str,
    det_model: str | None,
    sahi: bool,
    verbose: bool,
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    use_llm: bool = False,
    use_bot_sort: bool = False,
    reid_model_name: str = DEFAULT_REID_MODEL,
    viz: bool = True,
) -> None:
    """`run --track` 实现：指令解析 → TrackingTool 执行（帧序列 → 检测 → 跟踪 → MOT）。

    --llm 时启用序列级一次性 LLM 指令解析（resolve_plan），失败回退代码级
    extract_prompts；无论是否用 LLM，全程无逐帧 LLM 调用。
    use_bot_sort 启用精度档：ReID 特征（CLIP/SigLIP，懒加载）注入融合关联
    + ECC 相机运动补偿；PIL 打开失败自动降级纯 ByteTrack 语义。
    与 chat 共用同一管线（tools/tracking.TrackingTool），错误在此收敛为 Exit。
    """
    from auto2dlabel.cli_common import setup_logging

    setup_logging(verbose)

    try:
        prompts, threshold = resolve_plan(
            instruction, use_llm, provider, model, api_key, base_url, threshold)
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1)

    tool = TrackingTool(
        model_name=det_model,
        iou_threshold=iou,
        use_sahi=sahi,
        use_bot_sort=use_bot_sort,
        reid_model_name=reid_model_name,
        output_dir=output,
        export_format=export,
        viz=viz,
    )
    try:
        tool.forward(source, prompts, confidence_threshold=threshold)
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1)
