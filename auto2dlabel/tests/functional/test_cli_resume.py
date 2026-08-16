"""CLI --batch 失败隔离与 --resume 续跑测试（CliRunner + monkeypatch，零模型加载）。

覆盖：单图失败不中断整批、清单状态与快照写入、--resume 跳过 ok 重跑 failed。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from auto2dlabel import cli
from auto2dlabel.agent.state import AgentState
from auto2dlabel.schema.annotation import Annotation, Bbox
from auto2dlabel.tools.batch import STATUS_FAILED, STATUS_OK, load_manifest

runner = CliRunner()


class _FakeOrchestrator:
    """记录调用并模拟：img_bad 抛错，其余返回 1 框结果。"""

    def __init__(self, calls: list[str], raised: set[str], llm_client: Any = None,
                 detection_model: Any = None, iou_threshold: float = 0.5,
                 use_sahi: bool = False) -> None:
        self.calls = calls
        self.raised = raised  # 跨 CLI 调用共享（每次 invoke 新建实例）

    def run(self, image_path: str, instruction: str, confidence_threshold: float) -> AgentState:
        self.calls.append(image_path)
        # img_bad 首次运行抛错（模拟模型崩溃），续跑时恢复（模拟已修复）
        if "img_bad" in image_path and "img_bad" not in self.raised:
            self.raised.add("img_bad")
            raise RuntimeError("模拟模型崩溃")
        state = AgentState(image_path=image_path)
        ann = Annotation(image_path=image_path)
        ann.add_bbox(Bbox(x=1, y=2, width=3, height=4, label="car", confidence=0.9))
        state.annotations = [ann]
        return state


class _FakeLLM:
    model = "fake-llm"


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str], Path]:
    """准备两张真实小图 + 打桩所有外部依赖。"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("img_bad.jpg", "img_good.jpg"):
        Image.new("RGB", (8, 8)).save(img_dir / name)

    calls: list[str] = []
    raised: set[str] = set()
    monkeypatch.setattr(cli, "AgentOrchestrator",
                        lambda **kw: _FakeOrchestrator(calls, raised, **kw))
    monkeypatch.setattr(cli, "create_client", lambda **kw: _FakeLLM())
    monkeypatch.setattr("auto2dlabel.tools.log.log_llm_call", lambda **kw: None)
    monkeypatch.setattr(cli, "_visualize_results", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_triage_and_export", lambda *a, **kw: None)
    return img_dir, calls, tmp_path


def _statuses(out_dir: Path) -> dict[str, str]:
    m = load_manifest(out_dir / "batch_manifest.json")
    return {Path(e.path).name: e.status for e in m.images}


def test_batch_failure_isolated(cli_env: tuple[Path, list[str], Path]) -> None:
    """第 1 图抛错 → 记录失败继续第 2 图，退出码 0，清单与快照正确。"""
    img_dir, calls, tmp_path = cli_env
    out_dir = tmp_path / "out"

    result = runner.invoke(cli.app, [
        "run", str(img_dir), "检测汽车", "--batch", "--output", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    assert [Path(c).name for c in calls] == ["img_bad.jpg", "img_good.jpg"]

    statuses = _statuses(out_dir)
    assert statuses["img_bad.jpg"] == STATUS_FAILED
    assert statuses["img_good.jpg"] == STATUS_OK

    m = load_manifest(out_dir / "batch_manifest.json")
    good = next(e for e in m.images if Path(e.path).name == "img_good.jpg")
    assert good.bbox_count == 1
    assert good.state_file and Path(good.state_file).is_file()  # AgentState 快照已写
    bad = next(e for e in m.images if Path(e.path).name == "img_bad.jpg")
    assert "模拟模型崩溃" in bad.error


def test_resume_skips_ok_reruns_failed(cli_env: tuple[Path, list[str], Path]) -> None:
    """--resume：跳过 ok，只重跑 failed（调用记录不含 img_good）。"""
    img_dir, calls, tmp_path = cli_env
    out_dir = tmp_path / "out"
    runner.invoke(cli.app, ["run", str(img_dir), "检测汽车", "--batch", "--output", str(out_dir)])

    # 把 img_bad 置回 failed（模拟上次运行遗留），img_good 保持 ok
    m = load_manifest(out_dir / "batch_manifest.json")
    for e in m.images:
        if Path(e.path).name == "img_bad":
            e.status = STATUS_FAILED
    from auto2dlabel.tools.batch import save_manifest

    save_manifest(m, out_dir / "batch_manifest.json")

    calls.clear()
    result = runner.invoke(cli.app, ["run", "--resume", str(out_dir / "batch_manifest.json")])
    assert result.exit_code == 0, result.output
    assert [Path(c).name for c in calls] == ["img_bad.jpg"]

    statuses = _statuses(out_dir)
    assert statuses["img_good.jpg"] == STATUS_OK  # 未被重跑
    assert statuses["img_bad.jpg"] == STATUS_OK  # 重跑成功后更新


def test_resume_all_done_no_work(cli_env: tuple[Path, list[str], Path]) -> None:
    """全部 ok 时 --resume 无待重跑图像，正常退出。"""
    img_dir, calls, tmp_path = cli_env
    out_dir = tmp_path / "out"
    runner.invoke(cli.app, ["run", str(img_dir), "检测汽车", "--batch", "--output", str(out_dir)])

    # 手动把 img_bad 也改成 ok
    m = load_manifest(out_dir / "batch_manifest.json")
    for e in m.images:
        e.status = STATUS_OK
    from auto2dlabel.tools.batch import save_manifest

    save_manifest(m, out_dir / "batch_manifest.json")

    calls.clear()
    result = runner.invoke(cli.app, ["run", "--resume", str(out_dir / "batch_manifest.json")])
    assert result.exit_code == 0
    assert calls == []
