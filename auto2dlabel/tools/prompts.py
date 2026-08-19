"""中→英关键词映射与指令 prompt 提取（单一事实源）。

Agent Loop 的 hint 注入（orchestrator）、no-LLM baseline、`run --track`
三处共用同一份映射——曾各持一份副本，此处收敛（复用不复制红线）。
"""

from __future__ import annotations

# 中→英关键词映射（类别翻译不依赖 LLM，代码级兜底）
CN_EN_MAP: dict[str, str] = {
    "行人": "person", "人": "person", "汽车": "car", "车": "car",
    "自行车": "bicycle", "单车": "bicycle", "摩托车": "motorcycle",
    "公交车": "bus", "卡车": "truck", "狗": "dog", "猫": "cat",
    "红绿灯": "traffic light", "交通灯": "traffic light",
}


def extract_prompts(instruction: str) -> list[str]:
    """从用户指令中提取英文 prompts（按 CN_EN_MAP 子串匹配，去重保序）。

    Raises:
        ValueError: 指令中不含任何映射关键词。
    """
    prompts: list[str] = []
    for cn, en in CN_EN_MAP.items():
        if cn in instruction and en not in prompts:
            prompts.append(en)
    if not prompts:
        raise ValueError("无法从指令中提取关键词。请直接使用英文 prompt（如 'person'）。")
    return prompts
