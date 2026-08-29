"""对话式规划通用骨架（v0.6 P1 / v0.3 P1 共用）。

`parse_with_dialog` 与 schema 无关：渲染问题 / 收集回答 / 回喂 / 轮次控制
全部通用，2D/3D planner 各自注入 system prompt 与 parse_fn（复用不复制）。

轮次语义：**≤max_rounds 次 LLM 解析调用、≤max_rounds-1 次用户问答**
（对齐 Agent Loop max_iterations=3 红线——都是 LLM 调用次数上限）。
末轮先返回不再追问；用户放弃（空回答）→ 返回缺参 plan，由调用方
代码兜底（_fill_missing_params / ask_missing_params，v0.5 行为零回退）。

污染防护：骨架结束处重置 ``plan.raw_instruction = instruction``（任何调用方
安全——对话回喂文本绝不影响后续代码级扫描 parse_referential / has_relation /
detect_tracker_kind 与 batch-strategy 调参 LLM）；对话累积文本只存
``plan.dialog_context``（Step 4 确认修改重解析用）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from auto2dlabel.agent import logging
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.schema.task_plan import PlanQuestion
from auto2dlabel.tools.confirm import ask_text

logger = logging.getLogger(__name__)

# 回喂格式头（prompt 规则描述与此一致；改格式须同步 planner/planner3d prompt）
DIALOG_ANSWER_HEADER = "补充信息（用户在以下问题的回答，请据此更新 JSON）："

PlanT = TypeVar("PlanT")


def parse_with_dialog(
    llm: LLMClient,
    system_prompt: str,
    parse_fn: Callable[[str], PlanT],
    ask_fn: Callable[[list[PlanQuestion]], str],
    instruction: str,
    max_rounds: int = 3,
) -> PlanT:
    """多轮对话收集缺参 → 返回已补全/缺参的 plan。

    Args:
        llm: LLM 客户端（每轮 1 次调用，temperature=0）。
        system_prompt: 调用方注入的规划 prompt（含模型目录/GPU 上下文等资产）。
        parse_fn: LLM 响应 content → plan（schema 绑定，如 _parse_json_response）。
        ask_fn: 渲染问题并收集回答 → 回答文本（""/空白 = 用户放弃对话）。
        instruction: 原始用户指令。
        max_rounds: LLM 解析轮次上限（默认 3）。

    Returns:
        plan：questions 空 / 用户放弃 / 轮次耗尽时返回；抛 ValueError
        （空响应/parse 失败）由调用方降级链处理。
    """
    max_rounds = max(1, max_rounds)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    accumulated: list[str] = []

    for i in range(max_rounds):
        response = llm.chat(messages, tools=None, temperature=0.0)
        if not response.content:
            raise ValueError("LLM 返回空响应，无法解析任务")
        plan = parse_fn(response.content)
        _apply_plan_state(plan, instruction, accumulated)

        questions = _as_questions(getattr(plan, "questions", None))
        if not questions or i == max_rounds - 1:
            return plan

        answers = ask_fn(questions)
        if not answers or not answers.strip():
            logger.info("用户放弃对话补充（第 %d 轮），返回缺参 plan", i + 1)
            return plan

        section = _answer_section(questions, answers)
        accumulated.append(section)
        messages.append({"role": "user", "content": section})

    raise ValueError("parse_with_dialog 轮次循环异常退出")  # 防御（max_rounds>=1 不可达）


def _apply_plan_state(plan: Any, instruction: str, accumulated: list[str]) -> None:
    """骨架兜底 plan 状态：raw_instruction 恒为原始指令（防污染），dialog_context 累积。"""
    if hasattr(plan, "raw_instruction"):
        plan.raw_instruction = instruction
    if hasattr(plan, "dialog_context"):
        plan.dialog_context = "\n\n".join(accumulated)


def _as_questions(raw: Any) -> list[PlanQuestion]:
    """取 plan.questions（非 list / 缺失 → []）；复用 PlanQuestion 过滤语义。"""
    if isinstance(raw, list) and all(isinstance(q, PlanQuestion) for q in raw):
        return raw
    return PlanQuestion.from_list(raw)


def _answer_section(questions: list[PlanQuestion], answers: str) -> str:
    """问题 + 回答 → 回喂文本（id 帮助 LLM 归位；格式与 prompt 规则一致）。"""
    lines = [DIALOG_ANSWER_HEADER]
    for q in questions:
        lines.append(f"- [{q.id}] {q.question}")
    lines.append(f"用户回答：{answers}")
    return "\n".join(lines)


def ask_questions(questions: list[PlanQuestion], timeout: int = 30) -> str:
    """默认 ask_fn：一次性渲染全部问题 → 自由文本回答（一行）。

    timeout<=0（--no-wait）天然返回 ""——跳过对话走代码兜底。
    无确认词典（ask_text）："ok"/"好" 等按普通回答原样返回。
    """
    lines = ["需要补充以下信息（直接回车跳过对话，使用默认值继续）："]
    for q in questions:
        lines.append(f"  - {q.question}")
    result = ask_text("\n".join(lines), timeout=timeout)
    return result or ""
