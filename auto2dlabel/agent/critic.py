"""质检 Agent（Critic）—— 跨模型交叉校验质量评估（v1.1 P2）。

与现有 LLM Evaluate 的区别：EvaluateTool 是 agent loop 里的动作选择
工具（LLM 在 loop 里选 action，forward 只执行代码动作，**不调 LLM**）；
Critic 是**独立的一次 LLM 调用**（独立 prompt + 独立 client/provider +
独立 call_site evaluate_critic），仅在 quality.ok == False 时条件触发
（与 LLM Evaluate 同一判据），用更强模型交叉校验，降低同模型自我确认偏差。

只出意见不出动作：`judgment`/`action` 是建议，管道不执行检测重试
（Critic 是增强不是必需）。无 key / 失败 / 非法 JSON → 静默跳过，绝不
阻断标注关键路径。
"""

from __future__ import annotations

import json
from typing import Any

from auto2dlabel.agent import logging
from auto2dlabel.agent.llm import LLMClient
from auto2dlabel.agent.prompt_loader import load_prompt, render
from auto2dlabel.configs.model_catalog import format_catalog_summary

logger = logging.getLogger(__name__)

# 独立 prompt（与 planner prompt 不同源——跨模型交叉校验要求独立判据）
_CRITIC_SPEC = load_prompt("critic.md")
CRITIC_SYSTEM_PROMPT = render(_CRITIC_SPEC, catalog_summary=format_catalog_summary())

# 台账调用点（cost-critic 聚合键；与 planner.dialog 同表对比）
CRITIC_CALL_SITE = "evaluate_critic"
# provider env 覆盖（默认同 planner 的 deepseek；「更强模型」由用户 env 决定）
CRITIC_PROVIDER_ENV = "AUTOLABEL_CRITIC_PROVIDER"


def critic_provider() -> str:
    """Critic 用 provider（env 覆盖，默认 deepseek——纯本地红线不硬编码）。"""
    return os_getenv(CRITIC_PROVIDER_ENV) or "deepseek"


def os_getenv(name: str) -> str | None:
    import os

    return os.environ.get(name)


def build_critic_client() -> LLMClient:
    """Critic 专用客户端（跨模型交叉校验 = 独立 provider 建第二客户端）。

    经 create_client 走 provider 注册表（台账按 provider+model 记账）；
    has_credentials False 由调用方判定后静默跳过。
    """
    from auto2dlabel.agent.llm import create_client

    return create_client(provider=critic_provider())


def _parse_critic_json(content: str) -> dict[str, Any]:
    """Critic 响应 → {judgment, reason, action}；容错 json_mode + 代码块。"""
    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                raise ValueError(f"Critic 响应 JSON 解析失败: {text[:120]}...")
        else:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    raise ValueError(f"Critic 响应 JSON 解析失败: {text[:120]}...")
            else:
                raise ValueError(f"Critic 响应无 JSON: {text[:120]}...")
    if not isinstance(data, dict):
        raise ValueError(f"Critic 响应非对象: {text[:120]}...")
    judgment = str(data.get("judgment", ""))
    if judgment not in ("pass", "fail"):
        raise ValueError(f"Critic 响应 judgment 非法: {judgment}")
    action = str(data.get("action", ""))
    if action not in ("accept", "flag_for_review", "retry_lower_threshold", "retry_swap_model"):
        raise ValueError(f"Critic 响应 action 非法: {action}")
    return {
        "judgment": judgment,
        "reason": str(data.get("reason", "")),
        "action": action,
    }


def criticize(report: Any, client: LLMClient, call_site: str = CRITIC_CALL_SITE) -> dict[str, Any]:
    """一次 Critic 质检调用（独立 call_site 台账；独立 prompt）。

    Args:
        report: QualityReport（quality.to_dict() 注入 prompt）。
        client: Critic 专用客户端（create_client(provider=critic_provider())）。
        call_site: 台账调用点（默认 evaluate_critic；测试注入区分）。

    Returns:
        {judgment, reason, action}。

    Raises:
        ValueError: LLM 空响应 / JSON 解析失败 / 非法字段（调用方静默跳过）。
    """
    report_json = json.dumps(
        report.to_dict() if hasattr(report, "to_dict") else report, ensure_ascii=False
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Quality report:\n{report_json}"},
    ]
    response = client.chat(
        messages,
        tools=None,
        temperature=0.0,
        max_tokens=1024,
        json_mode=True,
        call_site=call_site,
    )
    if not response.content:
        raise ValueError("Critic 返回空响应")
    return _parse_critic_json(response.content)


def should_criticize(report: Any) -> bool:
    """触发判据（与现有 LLM Evaluate 同源）：quality.ok == False。"""
    return report is not None and not bool(getattr(report, "ok", True))
