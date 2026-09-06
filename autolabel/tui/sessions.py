"""会话持久化纯函数（v1.0 P4）。

JSONL append-only 单文件（logs/chat_sessions.jsonl，llm_usage.jsonl 先例），
每行一条消息。设计要点：
- 逐行子进程回显不进会话（ChatApp 层 record=False，本模块不感知）；
- 损坏行跳过（load_usage_logs 先例）；旧行缺字段按默认值容错
  （markup/task/domain/provider 均 .get 兜底，防重放注入）；
- 零 textual 依赖——读写产物不是 agent 逻辑，依赖方向最简。

行 schema：
{"session_id": "2026-09-06T15-30-00", "seq": 12, "ts": 1754...,
 "kind": "user|assistant|system|task_result", "text": "检测汽车", "domain": "2d",
 "provider": "deepseek", "markup": false,
 "task": {"id": 3, "status": "done", "manifest": "outputs/batch_manifest.json"}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SESSION_LOG_PATH = Path("logs/chat_sessions.jsonl")

# 显式导出（mypy no_implicit_reexport：app.py 的 from-import 链下测试可直接引用）
__all__ = [
    "SESSION_LOG_PATH",
    "append_session_line",
    "list_sessions",
    "load_sessions",
    "make_session_id",
    "safe_float",
    "safe_int",
    "session_messages",
]


def make_session_id(ts: float | None = None) -> str:
    """时间戳会话 id（YYYY-MM-DDTHH-MM-SS；同秒内 /new 连续两次会撞 id，
    撞车由 ChatApp._cmd_new 加 -N 后缀防覆盖）。"""
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime(ts or time.time()))


def safe_int(value: Any, default: int = 0) -> int:
    """容错 int 转换（旧行/损坏行字段可能是字符串或非数字——重放绝不崩）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """容错 float 转换（ts 字段同上）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def append_session_line(
    path: Path,
    session_id: str,
    seq: int,
    kind: str,
    text: str,
    *,
    ts: float | None = None,
    domain: str = "",
    provider: str = "",
    markup: bool = False,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """追加一行并落盘（写即 flush，进程退出不丢）；返回写入的记录。"""
    record: dict[str, Any] = {
        "session_id": session_id,
        "seq": seq,
        "ts": ts if ts is not None else time.time(),
        "kind": kind,
        "text": text,
        "domain": domain,
        "provider": provider,
        "markup": markup,
    }
    if task is not None:
        record["task"] = task
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_sessions(path: Path) -> list[dict[str, Any]]:
    """读全部行；文件缺失 → []；损坏行跳过（容错先例 llm.py load_usage_logs）。"""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("session_id"):
            entries.append(entry)
    return entries


def list_sessions(entries: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """分组摘要（最近优先）：session_id + 首条 user 截断 40 字 + 消息数 + 最后时间。"""
    by_id: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_id.setdefault(str(e.get("session_id")), []).append(e)
    summaries: list[dict[str, Any]] = []
    for sid, rows in by_id.items():
        first_user = next(
            (str(r.get("text", "")) for r in rows if r.get("kind") == "user"), ""
        )
        summaries.append(
            {
                "session_id": sid,
                "preview": first_user[:40] + ("…" if len(first_user) > 40 else ""),
                "messages": len(rows),
                "last_ts": max(safe_float(r.get("ts", 0)) for r in rows),
            }
        )
    summaries.sort(key=lambda s: s["last_ts"], reverse=True)
    return summaries[:limit]


def session_messages(
    entries: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    """过滤某会话全部行（按 seq 升序，重放顺序；seq 损坏行安全排末尾）。"""
    return sorted(
        (e for e in entries if e.get("session_id") == session_id),
        key=lambda e: safe_int(e.get("seq", 0)),
    )
