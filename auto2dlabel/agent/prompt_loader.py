"""LLM Prompt 外置加载器（v1.0 P1+，借鉴 claude-code agents/*.md frontmatter 模式）。

prompt 正文进 `agent/prompts/*.md`：frontmatter（YAML：name/description/profile）+
正文（`$占位符` 运行时注入）。**渲染绝不用 str.format/f-string**——正文含字面
JSON 花括号（`{}`），只有 `$name` 占位符参与替换（单遍 regex，防前缀冲突）。

调用档位：frontmatter `profile` → `PROFILES` 参数表（temperature/max_tokens/
json_mode 单一事实源，调用点不再散写）。auto3dlabel 复用本模块（依赖方向
3D → 2D 合法）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import yaml


class ChatProfile(TypedDict):
    """chat() 调用参数档位（TypedDict 使 **PROFILES[profile] 解包过 mypy strict）。"""

    temperature: float
    max_tokens: int
    json_mode: bool


PROFILES: dict[str, ChatProfile] = {
    # 规划档：低温确定性 JSON（推理模型下 llm.py 钳制会抬温，见 llm.py _is_reasoning_model）
    "planning": {"temperature": 0.0, "max_tokens": 1024, "json_mode": True},
}


class PromptLoadError(ValueError):
    """prompt 文件缺失 / frontmatter 非法 / 正文空 / 未知 profile / 占位符未注入。"""


@dataclass(frozen=True)
class PromptSpec:
    """prompt 文件解析产物（frontmatter 元数据 + 原文正文）。"""

    name: str
    description: str
    profile: str
    body: str  # 原文（含 $占位符，尚未注入）
    source: Path


def load_prompt(filename: str, prompts_dir: Path | None = None) -> PromptSpec:
    """加载 `prompts_dir/<filename>`：`---` 分隔 frontmatter + 正文。

    Args:
        filename: prompts 目录下的文件名（如 "planner.md"）。
        prompts_dir: 缺省 = 本模块同目录 `prompts/`（2D 自用）；
            跨包调用方（auto3dlabel）传自己的 prompts 目录。

    Raises:
        PromptLoadError: 文件不存在 / 缺 frontmatter / YAML 非法 / 必填字段缺失
            / profile 未知 / 正文空（文案均带文件路径）。
    """
    path = (prompts_dir or Path(__file__).parent / "prompts") / filename
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptLoadError(f"prompt 文件不可读: {path}（{exc}）") from exc
    if not text.startswith("---\n"):
        raise PromptLoadError(f"prompt 文件缺 frontmatter（须以 `---` 行开头）: {path}")
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        raise PromptLoadError(f"prompt 文件缺 frontmatter 结束行 `---`: {path}")
    try:
        meta = yaml.safe_load(parts[0][4:])
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"prompt frontmatter YAML 非法: {path}（{exc}）") from exc
    if not isinstance(meta, dict):
        raise PromptLoadError(f"prompt frontmatter 必须是 YAML 映射: {path}")
    name = meta.get("name")
    description = meta.get("description")
    profile = meta.get("profile")
    if not isinstance(name, str) or not name:
        raise PromptLoadError(f"prompt frontmatter 缺 name: {path}")
    if not isinstance(description, str) or not description:
        raise PromptLoadError(f"prompt frontmatter 缺 description: {path}")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise PromptLoadError(
            f"prompt frontmatter profile 未知: {path}（可用: {', '.join(PROFILES)}）"
        )
    body = parts[1].strip()
    if not body:
        raise PromptLoadError(f"prompt 正文为空: {path}")
    return PromptSpec(
        name=name, description=description, profile=profile, body=body, source=path,
    )


_PLACEHOLDER_RE = re.compile(r"\$(\w+)")


def render(spec: PromptSpec, **vars: Any) -> str:
    """`$name` 单遍替换注入（值强制 str()，与 f-string 输出一致）。

    正文含字面 JSON 花括号 → 禁用 str.format；未提供的占位符**汇总报错**
    （显式优于静默——漏注入会让 LLM 看到字面 `$name`）。
    """
    missing: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in vars:
            missing.append(name)
            return m.group(0)
        return str(vars[name])

    out = _PLACEHOLDER_RE.sub(_sub, spec.body)
    if missing:
        raise PromptLoadError(
            f"prompt {spec.source.name} 存在未注入的占位符: "
            f"{', '.join(dict.fromkeys(missing))}"
        )
    return out
