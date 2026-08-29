"""任务参数规格表：TaskStep 字段的单一事实源（2026-08-29 重构）。

「任务参数定义写死集中」——一张表驱动三处消费：
- missing 判定（schema/task_plan.py TaskStep.missing_params）：required 字段值
  falsy 即缺失（source "" / prompts []）→ CLI 代码级追问（ask_missing_params）
- 询问文案（缺参清单 label 直接展示给用户）
- planner prompt 注入（format_task_params_summary：字段清单 + 默认值，
  与 datasets/catalog 摘要同模式，LLM 拿到规范字段全集）

语义提取规则（如何从自然语言提取 conf/iou/model）是 prompt 质量敏感内容，
保留在 planner Rules 手写——本表只做字段契约，不做语义解析。

schema/task_plan.py 的 DEFAULT_* / REQUIRED_PARAMS 由本表派生（default 字段
为根——改一处不漏另一处）；3D 侧规格表见 auto3dlabel/configs/task_params.py。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    """任务参数字段规格。

    key: TaskStep 字段名（= LLM JSON key，同一事实源）。
    label: 中文展示名（缺参清单 / 询问文案）。
    question: 缺参追问问题（供交互文案；非必填参数留空）。
    required: 必填——值 falsy 即缺失（"" / [] / None）→ 必问。
    default: 规划阶段默认值（prompt 注入参考；TaskStep 实际默认由它派生）。
    ask: 缺失时进入追问链（可选参数，2026-08-29 追问扩展）——值为默认值
        即视为「未指定」→ 询问（文案带默认提醒，回车/超时按默认继续）。
    """

    key: str
    label: str
    question: str
    required: bool
    default: Any
    ask: bool = False


# 单一事实源：TaskStep 全部可提取参数字段。model_hint（LLM 选型理由，仅展示）
# 与 step_id/task_type（结构字段）不在此表——不是用户可指定的任务参数。
TASK_PARAM_SPECS: dict[str, ParamSpec] = {
    spec.key: spec
    for spec in (
        ParamSpec("source", "source（数据路径）", "要标注的数据路径是什么？(目录/图片)", True, ""),
        ParamSpec("prompts", "prompts（检测类别）", "要标注哪些类别？(如: 汽车、行人)", True, []),
        ParamSpec(
            "confidence_threshold",
            "confidence_threshold（置信度）",
            "置信度阈值？(如 0.5)",
            False,
            0.1,
            ask=True,  # 缺超参数 → 追问（默认提醒，回车跳过）
        ),
        ParamSpec("iou_threshold", "iou_threshold（IoU）", "IoU 阈值？(如 0.3)", False, 0.3,
                  ask=True),

        ParamSpec(
            "model_name",
            "model_name（模型）",
            "用哪个模型？(如 yolo26x.pt)",
            False,
            "yolo26x.pt",
            ask=True,  # 缺模型选择 → 追问（默认提醒，回车跳过）
        ),
        # export_format/sahi/num_workers/batch_size 不追问：
        # 格式/切片默认合理；批参数已有独立交互（_fill_batch_params）
        ParamSpec(
            "export_format", "export_format（导出格式）", "导出格式？(如 coco)", False, "coco"
        ),
        ParamSpec("sahi", "sahi（切片推理）", "是否启用切片推理？", False, False),
        ParamSpec("num_workers", "num_workers", "", False, None),
        ParamSpec("batch_size", "batch_size", "", False, None),
    )
}


def format_task_params_summary() -> str:
    """任务参数速查摘要（注入 planner prompt：字段清单 + REQUIRED 标记 + 默认值）。

    REQUIRED 字段缺失时 CLI 会追问——LLM 据此把缺参指令解析为缺字段 JSON
    （questions 分支），而非臆造路径/类别。
    """
    lines = ["Task parameters (field → default; REQUIRED = CLI will ask if missing):"]
    for spec in TASK_PARAM_SPECS.values():
        if spec.required:
            lines.append(f"- {spec.key}: REQUIRED")
        else:
            lines.append(f"- {spec.key}: {spec.default!r}")
    return "\n".join(lines)


# ================================================================
# 代码级守卫（2026-08-29 防御纵深）：LLM 输出不可全信
#
# 漏洞类（实测）：路径/文件名中的数字被 LLM 误提为超参数——
# 「COCO2017」→ batch_size=2017 / num_workers=2017（进程 fork 风险）、
# conf=2017（全零静默失败）。批/进程数以指令文本代码级提取为准
# （LLM 值不采信，同 detect_tracker_kind 红线）；数值类值域违规回默认。
# ================================================================


def coerce_float(value: Any, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """安全转 float：bool/非数值/超出 (lo, hi] → default。

    防两类洞：① "conf": "abc" 直接 float() 抛异常中断解析；
    ② 路径数字误提（conf=2017 → 全零静默失败）。
    """
    if isinstance(value, bool) or value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not lo < v <= hi:
        return default
    return v


def coerce_int(value: Any) -> int | None:
    """安全转 int：bool/非整浮点/字符串垃圾 → None（超参 None = 交执行阶段
    自动实测，绝不把垃圾数字传进 DataLoader）。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def coerce_bool(value: Any) -> bool:
    """安全转 bool：只有真 bool / true 系词为 True——"false"/"no"/"0" 字符串
    不是 True（Python bool("false") 为 True 的经典坑，会静默误开 SAHI）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是", "开")
    if isinstance(value, (int, float)):
        return value != 0
    return False


def coerce_str(value: Any, default: str = "") -> str:
    """安全转 str：None/容器垃圾 → default；str 去首尾空白。"""
    if isinstance(value, str):
        return value.strip()
    return default


def coerce_strs(value: Any) -> list[str]:
    """安全转字符串列表：list → 逐项 str 去空白过滤空项；单 str → 单元素列表；
    其余 → []（[] 保持缺参语义，交追问链，不臆造）。"""
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


# 显式超参表述（代码级提取为准——LLM 提取值不采信，指令文本才是单一事实源）
_BATCH_VALUE_RE = re.compile(
    r"(?:batch[_-]?size|\bbatch\b|批量|批大小)\s*[=:：]?\s*(\d+)", re.IGNORECASE
)
_WORKERS_VALUE_RE = re.compile(
    r"(?:num[_-]?workers?|\bworkers?\b|进程(?:数)?)\s*[=:：]?\s*(\d+)", re.IGNORECASE
)

_MAX_BATCH_SIZE = 512  # 消费级 GPU 上限；更大值视为垃圾 → None（自动实测）


def guard_batch_size(instruction: str) -> int | None:
    """batch_size 守卫：只在指令含显式批量表述时从指令取数，否则 None。

    None 语义 = 执行阶段自动实测/交互（与 planner Rules「仅显式指定才提取」
    一致）；「批量处理时跑最大」等无数字表述同样 None（自动实测，不是 1）。
    实测洞：COCO2017 路径 → LLM 误提 batch_size=2017 → 本守卫无表述 → None。
    """
    m = _BATCH_VALUE_RE.search(instruction)
    if not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= _MAX_BATCH_SIZE:
        return None
    return n


def guard_num_workers(instruction: str) -> int | None:
    """num_workers 守卫：同上；上限钳到 CPU 核数（防「2017 进程」fork 炸弹）。"""
    m = _WORKERS_VALUE_RE.search(instruction)
    if not m:
        return None
    n = int(m.group(1))
    if n < 0:
        return None
    return min(n, os.cpu_count() or 4)


# ── 显式提及判定（2026-08-29 追问链防误追）──────────────────────────
# 值==默认值只能说明「LLM 没提取到非默认值」，不代表用户没指定——
# 「模型用 yolo26x.pt」提取后值恰等于默认，追问「缺模型」是误追。
# 与 guard_batch_size 同一哲学：指令文本代码级提取为准。
# model_name 宽匹配（「用 X」即视为提及）——误判只损失一次询问机会
# （不追问 → 默认执行，与回车跳过同语义），正确场景避免烦人误追。
_EXPLICIT_PARAM_PATTERNS: dict[str, re.Pattern[str]] = {
    "confidence_threshold": re.compile(
        r"(?:置信度|confidence|conf)\s*(?:设为|设置为|改为?|改成?|换成?|调为|调成|为|=|:：)?"
        r"\s*0?\.?\d+"
    ),
    "iou_threshold": re.compile(
        r"iou\s*(?:设为|设置为|改为?|改成?|换成?|调为|调成|为|=|:：)?\s*0?\.?\d+",
        re.IGNORECASE,
    ),
    "model_name": re.compile(
        r"(?:模型|model|采用|使用|用|换成?|改为?|改成?)\s*[=:：]?\s*[A-Za-z][\w./-]*"
    ),
}


def guard_explicit_params(instruction: str) -> set[str]:
    """指令文本中显式提及的参数 key 集合（代码级正则，零 LLM）。

    追问链用它把「显式指定」的可选参数排除出 optional_missing——
    即使值==默认（如 yolo26x.pt）也算已指定，不再误追问。
    """
    return {
        key
        for key, pat in _EXPLICIT_PARAM_PATTERNS.items()
        if pat.search(instruction or "")
    }
