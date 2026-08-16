"""批量标注清单 —— --batch 失败隔离与 --resume 续跑的持久化层。

纯函数模块：BatchManifest dataclass + JSON 读写（tmp+rename 原子写）。
CLI 负责编排与状态更新，本模块不做 I/O 之外的决策。

清单 schema: {version, created_at, instruction, config, images: [...]}
单图条目: {path, status(pending/ok/failed), error, elapsed, bbox_count, state_file}

--resume 语义：跳过 status=ok 的图，重跑 failed/pending；不提供 --retry-failed。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1

STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass
class BatchImageEntry:
    """清单中的单图条目。"""

    path: str = ""
    status: str = STATUS_PENDING
    error: str = ""
    elapsed: float = 0.0
    bbox_count: int = 0
    state_file: str = ""  # 成功时写入的 AgentState 快照路径

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "error": self.error,
            "elapsed": self.elapsed,
            "bbox_count": self.bbox_count,
            "state_file": self.state_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchImageEntry:
        return cls(
            path=str(d.get("path", "")),
            status=str(d.get("status", STATUS_PENDING)),
            error=str(d.get("error", "")),
            elapsed=float(d.get("elapsed", 0.0)),
            bbox_count=int(d.get("bbox_count", 0)),
            state_file=str(d.get("state_file", "")),
        )


@dataclass
class BatchManifest:
    """一次 --batch 运行的清单。"""

    version: int = MANIFEST_VERSION
    created_at: str = ""
    instruction: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    images: list[BatchImageEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "instruction": self.instruction,
            "config": self.config,
            "images": [e.to_dict() for e in self.images],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchManifest:
        return cls(
            version=int(d.get("version", MANIFEST_VERSION)),
            created_at=str(d.get("created_at", "")),
            instruction=str(d.get("instruction", "")),
            config=dict(d.get("config", {})),
            images=[BatchImageEntry.from_dict(e) for e in d.get("images", [])],
        )


def new_manifest(
    instruction: str,
    config: dict[str, Any],
    image_paths: list[str],
    created_at: str = "",
) -> BatchManifest:
    """创建新清单（images 全部 pending）。"""
    return BatchManifest(
        version=MANIFEST_VERSION,
        created_at=created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        instruction=instruction,
        config=dict(config),
        images=[BatchImageEntry(path=p) for p in image_paths],
    )


def save_manifest(manifest: BatchManifest, path: str | Path) -> Path:
    """原子写清单（tmp + rename，中断不损坏已有清单）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    return out


def load_manifest(path: str | Path) -> BatchManifest:
    """加载清单；字段缺失容错（from_dict 全部带默认值）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"清单文件不存在: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return BatchManifest.from_dict(data)


def resume_targets(manifest: BatchManifest) -> list[BatchImageEntry]:
    """--resume 续跑目标：跳过 ok，重跑 failed/pending。"""
    return [e for e in manifest.images if e.status != STATUS_OK]


def update_entry(
    manifest: BatchManifest,
    image_path: str,
    *,
    status: str,
    error: str = "",
    elapsed: float = 0.0,
    bbox_count: int = 0,
    state_file: str = "",
) -> bool:
    """更新单图条目状态（找不到返回 False，不抛错）。"""
    for e in manifest.images:
        if e.path == image_path:
            e.status = status
            e.error = error
            e.elapsed = elapsed
            e.bbox_count = bbox_count
            e.state_file = state_file
            return True
    return False
