"""`run --track` 薄壳 —— 指令解析（LLM/代码级）后委托 TrackingTool 执行。

管线实现在 tools/tracking.py（TrackingTool，与 chat 的 execute_plan track 分支
共用）；本模块只保留 CLI 侧能力：序列级一次性 LLM 指令解析（--llm，
extract_prompts 处理不了的复杂指令如「跟踪穿红衣服的人」，失败回退代码级）
与错误收敛（ValueError → console 红字 + typer.Exit）。
"""

from __future__ import annotations

from pathlib import Path

import typer

from auto2dlabel.cli_common import console
from auto2dlabel.tools.constraints import (
    ReferentialConstraint,
    has_relation,
    parse_referential,
    parse_roi,
)
from auto2dlabel.tools.tracking import DEFAULT_REID_MODEL, TrackingTool, collect_frames


def _resolve_auto_roi(source: str, output: str) -> list[tuple[float, float]] | None:
    """--roi auto：首帧 UFLD 车道线 → 自车车道闭合多边形（v0.5 C2）。

    抽帧幂等（TrackingTool 复用同一帧目录零重复抽取）；车道不足 2 条、
    模型失败、源无帧均黄字降级 None（无 ROI 全图保留，宁多勿漏）。
    """
    from auto2dlabel.models.lane import create_lane_model, resolve_auto_roi

    frames = collect_frames(Path(source), Path(output))
    if not frames:
        console.print("[yellow]--roi auto: 源无帧可解析，跳过自动 ROI[/yellow]")
        return None
    try:
        poly = resolve_auto_roi(create_lane_model(), str(frames[0]))
    except Exception as e:
        console.print(f"[yellow]--roi auto: 车道解析失败（全图保留）: {e}[/yellow]")
        return None
    if poly is None:
        console.print("[yellow]--roi auto: 车道线不足 2 条，跳过自动 ROI（全图保留）[/yellow]")
        return None
    console.print(f"[dim]--roi auto: 自车车道 ROI {len(poly)} 边形（首帧 UFLD）[/dim]")
    return poly


def resolve_plan(
    instruction: str,
    use_llm: bool,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    threshold: float,
) -> ReferentialConstraint:
    """跟踪规划入口：--llm 时序列级一次性 LLM 解析（失败回退），否则代码级解析。

    LLM 覆盖 extract_prompts 处理不了的复杂指令（如「跟踪穿红衣服的人」）；
    每序列仅 1 次调用，绝不逐帧。返回结构化约束（类别 + 可选属性/方位；
    threshold 恒有值——LLM 给 null 时沿用 CLI）。extract_prompts 的
    ValueError 上抛（调用方转 Exit）。
    """
    if use_llm:
        try:
            constraint = _llm_plan_once(
                instruction, provider, model, api_key, base_url)
            console.print(
                f"[dim]LLM 规划: 类别 {', '.join(constraint.prompts)}"
                f"{'  属性: ' + ', '.join(constraint.attributes) if constraint.attributes else ''}"
                f"{'  方位: ' + constraint.position if constraint.position else ''}[/dim]"
            )
            if constraint.threshold is None:
                constraint.threshold = threshold
            return constraint
        except Exception as e:
            console.print(f"[yellow]LLM 规划失败，回退代码级解析: {e}[/yellow]")
    constraint = parse_referential(instruction)
    constraint.threshold = threshold
    return constraint


def _llm_plan_once(
    instruction: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> ReferentialConstraint:
    """一次性 LLM 指令解析 → 结构化约束（threshold 为 None 时沿用 CLI 阈值）。

    单轮 tool-less 调用，JSON 输出。prompts 字段对齐 TaskPlan schema（为其子集，
    未来升级完整 planner 编排时无损）；attributes/position 为 v0.4 3a 约束字段
    （可选，缺省 = 无约束）。输出无效时抛 ValueError/JSONDecodeError。
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
                    '{"prompts": ["COCO 英文类别名"], "threshold": 0到1的浮点数或 null,'
                    ' "attributes": ["颜色等英文形容词，无则空数组"],'
                    ' "position": "left/right/center/top/bottom 或 null"}\n'
                    "threshold=null 表示沿用 CLI 阈值。只输出 JSON，不要其他文字。"
                ),
            },
            {"role": "user", "content": instruction},
        ],
        temperature=0.0,
        max_tokens=512,
        json_mode=True,
        call_site="track.plan",
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

    raw_attrs = data.get("attributes") or []
    if not all(isinstance(a, str) for a in raw_attrs):
        raise ValueError(f"LLM 规划输出无效 attributes: {data}")
    attributes = [str(a) for a in raw_attrs]

    raw_pos = data.get("position")
    position: str | None
    if raw_pos is None:
        position = None
    elif raw_pos in {"left", "right", "center", "top", "bottom"}:
        position = str(raw_pos)
    else:
        raise ValueError(f"LLM 规划输出无效 position: {data}")

    return ReferentialConstraint(
        prompts=prompts, attributes=attributes, position=position,
        threshold=llm_threshold,
    )


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
    roi: str | None = None,
    refer_l2: bool = False,
    refer_l3: bool = False,
) -> None:
    """`run --track` 实现：指令解析 → TrackingTool 执行（帧序列 → 检测 → 跟踪 → MOT）。

    --llm 时启用序列级一次性 LLM 指令解析（resolve_plan），失败回退代码级
    parse_referential；无论是否用 LLM，全程无逐帧 LLM 调用。
    use_bot_sort 启用精度档：ReID 特征（CLIP/SigLIP，懒加载）注入融合关联
    + ECC 相机运动补偿；PIL 打开失败自动降级纯 ByteTrack 语义。
    roi 为手动 ROI（v0.4 3a 空间约束）：矩形 "x1,y1,x2,y2" 或分号多边形，
    注入结构化约束后由 TrackingTool 过滤；"auto" 时走 v0.5 C2 自动车道
    ROI（首帧 UFLD 车道线 → 自车车道多边形，失败黄字无 ROI 全图保留）。
    refer_l2/refer_l3 启用 v0.5 指代 L2/L3：指令含关系词（旁边/附近/之间
    等）也自动触发，序列级一次解析（首帧锁定目标，后续帧轨迹匹配维持）。
    refer_l3 显式直用 L3（Qwen2-VL-7B，GPU）；默认指代路径为阶梯升级
    （L2 失败自动升级 L3，L3 权重未就位/无 GPU 时黄字降级宁多勿漏）。
    与 chat 共用同一管线（tools/tracking.TrackingTool），错误在此收敛为 Exit。
    """
    from auto2dlabel.cli_common import setup_logging

    setup_logging(verbose)

    try:
        constraint = resolve_plan(
            instruction, use_llm, provider, model, api_key, base_url, threshold)
        if roi:
            if roi.strip().lower() == "auto":
                constraint.roi = _resolve_auto_roi(source, output)
            else:
                constraint.roi = parse_roi(roi)
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1)

    # 指代 L2/L3（v0.5）：--refer-l2/--refer-l3 显式启用，或指令含关系词
    # （旁边/附近/之间等）自动触发；序列级一次解析（首帧锁定，不做逐帧）。
    # refer_l3 直用 L3（GPU）；否则阶梯升级（L2 失败自动升级 L3，L3 权重
    # 未就位/无 GPU 时快速失败黄字降级，宁多勿漏）
    referential = None
    referential_phrase: str | None = None
    if refer_l3 or refer_l2 or has_relation(instruction):
        from auto2dlabel.tools.constraints import build_referential_phrase

        referential_phrase = build_referential_phrase(instruction, constraint)
        if refer_l3:
            from auto2dlabel.models.referential_l3 import (
                create_referential_l3_resolver,
            )

            referential = create_referential_l3_resolver()
            console.print(
                f"[dim]指代 L3 (Qwen2-VL-7B): 首帧解析短语 「{referential_phrase}」[/dim]"
            )
        else:
            from auto2dlabel.models.referential_l3 import (
                create_cascade_referential_resolver,
            )

            referential = create_cascade_referential_resolver()
            console.print(
                f"[dim]指代 L2 (Florence-2, 失败升级 L3): 首帧解析短语 "
                f"「{referential_phrase}」[/dim]"
            )

    tool = TrackingTool(
        model_name=det_model,
        iou_threshold=iou,
        use_sahi=sahi,
        use_bot_sort=use_bot_sort,
        reid_model_name=reid_model_name,
        output_dir=output,
        export_format=export,
        viz=viz,
        constraint=constraint,
        referential=referential,
        referential_phrase=referential_phrase,
    )
    try:
        tool.forward(
            source, constraint.prompts,
            confidence_threshold=constraint.threshold or threshold,
        )
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1)
