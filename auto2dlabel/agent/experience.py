"""标注经验库（v1.1 P1）—— RAG 的最小闭环，零 LLM 调用。

检索单元 = 一次标注会话摘要 {指令, 域/数据集, 任务类型, 选型与阈值,
质量判定, 人工修正动作}。入库从既有结构化数据纯代码构造（TaskPlan /
QualityReport / TriageResult / *_reviewed.json 的 edited_by_human），
**绝无 LLM 生成**（LLM 调用点不扩容红线）；检索 metadata 精确过滤 →
向量 top-k 排序（embedding 复用已集成 CLIP 文本编码，纯本地）。

向量检索用 numpy 暴力余弦（种子量小线性扫足够；faiss 是后续规模化
才引入）。embedding 函数通过 `embed_fn` 注入——单测 Fake 注入零权重，
生产默认 CLIP 文本编码器。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from auto2dlabel.agent import logging

logger = logging.getLogger(__name__)

EXPERIENCE_LOG_PATH = Path("logs/experience.jsonl")
SEEDS_PATH = Path(__file__).resolve().parent.parent / "configs" / "experience_seeds.jsonl"

# top-k 注入上限（few-shot 过长会挤占 planner 预算）
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class ExperienceEntry:
    """一次标注会话的经验摘要（检索单元；种子/运行入库同构）。"""

    instruction: str = ""        # 用户指令原文（检索文本）
    domain: str = ""             # "2d" | "3d"（""=未知）
    dataset: str = ""            # "coco"|"kitti"|"dota"|"cityscapes"|...（""=无）
    task_type: str = ""          # object_detection|instance_segmentation|...（""=无）
    model: str = ""              # 选型（""=无）
    confidence_threshold: float | None = None
    quality: str = ""            # quality.ok "True"/"False"（""=无判定）
    human_action: str = ""       # 人工修正动作（edited_by_human/issues 提取）
    notes: str = ""              # 结论一句话（种子手工写；运行入库缺省空）

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "domain": self.domain,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "model": self.model,
            "confidence_threshold": self.confidence_threshold,
            "quality": self.quality,
            "human_action": self.human_action,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperienceEntry:
        """读 JSONL 行（缺字段 .get 兜底，损坏值容错——绝不 raise）。"""
        return cls(
            instruction=str(d.get("instruction", "")),
            domain=str(d.get("domain", "")),
            dataset=str(d.get("dataset", "")),
            task_type=str(d.get("task_type", "")),
            model=str(d.get("model", "")),
            confidence_threshold=_safe_float(d.get("confidence_threshold")),
            quality=str(d.get("quality", "")),
            human_action=str(d.get("human_action", "")),
            notes=str(d.get("notes", "")),
        )

    def to_history(self) -> str:
        """few-shot 注入的案例文本（planner prompt `<history>` 段一行一条）。"""
        parts = [f"- 指令: {self.instruction}"]
        if self.domain:
            parts.append(f"域: {self.domain}")
        if self.dataset:
            parts.append(f"数据集: {self.dataset}")
        if self.task_type:
            parts.append(f"任务: {self.task_type}")
        if self.model:
            parts.append(f"模型: {self.model}")
        if self.confidence_threshold is not None:
            parts.append(f"阈值: {self.confidence_threshold}")
        if self.quality:
            parts.append(f"质量判定: {self.quality}")
        if self.human_action:
            parts.append(f"人工修正: {self.human_action}")
        if self.notes:
            parts.append(f"经验: {self.notes}")
        return "，".join(parts)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------- 入库（零 LLM，纯代码构造） ----------


def build_entry(
    instruction: str,
    domain: str = "2d",
    task_type: str = "",
    model: str = "",
    confidence_threshold: float | None = None,
    quality: Any = None,
    reviewed_annotations: list[dict[str, Any]] | None = None,
    dataset: str = "",
    notes: str = "",
) -> ExperienceEntry:
    """从既有结构化数据拼装经验条目（纯函数，零 LLM / 零 I/O）。

    quality: QualityReport 实例或 None——quality.ok 判定写 quality 字段。
    reviewed_annotations: *_reviewed.json 的 annotations（人工修正动作提取
        edited_by_human 与 issues；web/server.py 已落盘该结构）。
    """
    q = ""
    if quality is not None:
        ok = getattr(quality, "ok", None)
        if ok is not None:
            q = "True" if ok else "False"

    human = _extract_human_action(reviewed_annotations or [])
    return ExperienceEntry(
        instruction=instruction,
        domain=domain,
        dataset=dataset,
        task_type=task_type,
        model=model,
        confidence_threshold=confidence_threshold,
        quality=q,
        human_action=human,
        notes=notes,
    )


def _extract_human_action(annotations: list[dict[str, Any]]) -> str:
    """从已复核标注提取人工修正动作（无修正 → ""）。"""
    edited = sum(1 for a in annotations if a.get("edited_by_human"))
    issues = sum(1 for a in annotations if a.get("issues"))
    if edited and issues:
        return f"人工修正 {edited} 框 + {issues} 处区域问题"
    if edited:
        return f"人工修正 {edited} 框"
    if issues:
        return f"{issues} 处区域问题"
    return ""


def append_entry(path: Path, entry: ExperienceEntry, *, dedupe: bool = True) -> bool:
    """追加一条经验到 JSONL（写即 flush）；返回是否写入（重复时跳过 False）。

    dedupe：instruction+model 双键去重（重复会话重跑不膨胀）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (entry.instruction, entry.model)
    if dedupe and key in _existing_keys(path):
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return True


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            keys.add((str(d.get("instruction", "")), str(d.get("model", ""))))
    return keys


def load_entries(path: Path | None = None) -> list[ExperienceEntry]:
    """读经验 JSONL；缺文件 → []；损坏行跳过（load_sessions 先例）。"""
    target = path or EXPERIENCE_LOG_PATH
    entries: list[ExperienceEntry] = []
    for d in _load_lines(target):
        entries.append(ExperienceEntry.from_dict(d))
    return entries


def load_seeds() -> list[ExperienceEntry]:
    """读种子条目（configs/experience_seeds.jsonl，包内入库）。"""
    return load_entries(SEEDS_PATH)


def _load_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            lines.append(d)
    return lines


# ---------- 检索（metadata 过滤 + 向量 top-k，纯本地零 LLM） ----------

# embedding 函数注入点：`(text: str) -> list[float]`；生产默认 CLIP 文本编码器，
# 单测 Fake 注入零权重。None = 跳过向量排序（仅 metadata 过滤）。
EmbedFn = Callable[[str], list[float]]

_EMBED_STATE: dict[str, Any] = {}


def _embed_fn() -> EmbedFn | None:
    """惰性构造 CLIP 文本编码器（复用已集成 ClipModel，weights/hf 已缓存）。

    加载失败（transformers 缺失/权重缺失/设备错误）→ None——检索退化为
    metadata 过滤（指令相似度不参与排序），绝不 raise（纯本地红线，
    RAG 是增强不是必需）。
    """
    if _EMBED_STATE.get("_init") is True:
        return _EMBED_STATE.get("_fn")
    _EMBED_STATE["_init"] = True
    try:
        from auto2dlabel.models.classification import ClipModel

        model = ClipModel()

        def _embed(text: str) -> list[float]:
            return _clip_text_embed(model, text)

        _EMBED_STATE["_fn"] = _embed
        return _embed
    except Exception as e:
        logger.warning("经验库 embedding 不可用，检索退化为 metadata 过滤: %s", e)
        _EMBED_STATE["_fn"] = None
        return None


def _clip_text_embed(model: Any, text: str) -> list[float]:
    """CLIP 文本编码 → 归一化向量（processor 单文本；中文指令 CLIP 可编）。"""
    import numpy as np
    import torch

    proc = getattr(model, "_processor", None)
    if proc is None:
        model._load()
        proc = getattr(model, "_processor", None)
        if proc is None:
            return []
    device = getattr(model, "_device", "cpu")
    inputs = proc(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # transformers 5.x：get_text_features 返回 BaseModelOutputWithPooling，
        # 取 pooler_output（get_image_features 同款——见 memory 记录）
        out = model._model.get_text_features(**inputs)
        v = out.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return []
    return cast(list[float], (v / norm).tolist())


def _cosine(a: list[float], b: list[float]) -> float:
    """向量余弦相似度（已归一化 → 点积即余弦）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def retrieve(
    instruction: str,
    domain: str = "",
    dataset: str = "",
    task_type: str = "",
    k: int = DEFAULT_TOP_K,
    entries: list[ExperienceEntry] | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[ExperienceEntry]:
    """metadata 精确过滤 → 向量 top-k 排序（纯本地，零 LLM）。

    metadata 字段（domain/dataset/task_type）非空即参与过滤——精确匹配
    优先于向量（结构化信息不浪费）；过滤后候选集空 → []。

    entries: 注入经验池（默认 = 运行日志 ∪ 种子，运行日志优先）；
    embed_fn: 注入 embedding（默认 CLIP 文本编码；None 且无法加载 →
        metadata 过滤后按原序截断 top-k，不排序）。
    """
    pool = entries if entries is not None else load_entries() + load_seeds()
    candidates = [
        e for e in pool
        if _meta_match(e, domain, dataset, task_type)
    ]
    if not candidates:
        return []

    fn = embed_fn if embed_fn is not None else _embed_fn()
    if fn is None:
        return candidates[:k]  # embedding 不可用 → 仅 metadata 过滤

    try:
        query = fn(instruction)
    except Exception as e:
        logger.warning("经验库检索向量化失败，回退 metadata 过滤: %s", e)
        return candidates[:k]
    if not query:
        return candidates[:k]

    scored = sorted(
        candidates,
        key=lambda e: _cosine(query, fn(e.instruction)),
        reverse=True,
    )
    return scored[:k]


def _meta_match(e: ExperienceEntry, domain: str, dataset: str, task_type: str) -> bool:
    """metadata 精确匹配：调用方非空字段全命中才进候选集。"""
    if domain and e.domain and e.domain != domain:
        return False
    if dataset and e.dataset and e.dataset != dataset:
        return False
    if task_type and e.task_type and e.task_type != task_type:
        return False
    return True


# ---------- few-shot 文本装配（planner 注入） ----------


def format_fewshot(entries: list[ExperienceEntry]) -> str:
    """top-k 经验 → planner prompt 注入段（<history> 一行一条；空 → ""）。"""
    if not entries:
        return ""
    lines = ["<history>", "以下是相似指令的历史标注经验（优先参考其选型与处置）:"]
    for e in entries:
        lines.append(e.to_history())
    lines.append("</history>")
    return "\n".join(lines)
