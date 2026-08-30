"""3b 批次级策略测试（sample_stats 聚合 / LLM 调参解析与非法回退 /
apply_strategy / execute_plan 挂钩）。

零真实权重：sample_stats 注入 FakeModel（duck typing detect），LLM 用脚本化 Fake；
execute_plan 挂钩测试 monkeypatch _apply_batch_strategy/_execute_chunk/模型工厂。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from auto2dlabel.agent.batch_strategy import (
    BatchStats,
    apply_strategy,
    llm_tune_strategy,
    sample_stats,
)
from auto2dlabel.agent.llm import LLMResponse
from auto2dlabel.cli_execute import _apply_batch_strategy, execute_plan
from auto2dlabel.schema.annotation import Bbox
from auto2dlabel.tests import Path

# ---------- sample_stats ----------

class _FakeModel:
    """按队列返回每图框数（label 恒为 prompts[0]），记录调用。"""

    def __init__(self, boxes_per_call: list[int]) -> None:
        self.boxes_per_call = list(boxes_per_call)
        self.calls: list[tuple[str, list[str], float]] = []

    def detect(
        self, image_path: str, prompts: list[str], confidence_threshold: float = 0.3,
    ) -> list[Bbox]:
        self.calls.append((image_path, list(prompts), confidence_threshold))
        n = self.boxes_per_call.pop(0)
        return [
            Bbox(x=float(i), y=0, width=1, height=1, label=prompts[0], confidence=0.9)
            for i in range(n)
        ]


def test_sample_stats_aggregation() -> None:
    """聚合：总框数 / 0 框图数 / 平均框数 / 缺类（有框图缺 person，0 框图全缺）。"""
    paths = [f"/d/{i}.jpg" for i in range(10)]
    model = _FakeModel([2, 0, 1, 0, 0, 2, 1, 0, 3, 0])
    stats = sample_stats(
        paths, ["car", "person"], 0.3, model=model, max_samples=10,
    )
    assert len(model.calls) == 10  # n ≤ max_samples → 全量抽样
    assert stats.images_sampled == 10
    assert stats.total_boxes == 9
    assert stats.zero_box_images == 5
    assert stats.avg_boxes == pytest.approx(0.9)
    # 有框的 5 张全部只标 car（缺 person）；0 框的 5 张两者皆缺 → 并集
    assert stats.missing_prompts == ["car", "person"]


def test_sample_stats_stride_sampling() -> None:
    """超限抽样：步长 ceil(n/max)，确定性且 ≤ max_samples。"""
    paths = [f"/d/{i}.jpg" for i in range(20)]
    model = _FakeModel([1] * 20)
    stats = sample_stats(paths, ["car"], 0.3, model=model, max_samples=8)
    assert stats.images_sampled == 7  # ceil(20/8)=3 → 20//3 余 7 张
    assert [c[0] for c in model.calls] == paths[::3]


def test_sample_stats_error_counted_as_zero_box() -> None:
    """抽样异常按 0 框计入（宁多勿漏：让 LLM 看到问题信号）。"""

    class _BoomModel:
        def detect(self, image_path: str, prompts: list[str],
                   confidence_threshold: float = 0.3) -> list[Bbox]:
            raise RuntimeError("boom")

    stats = sample_stats(
        ["/d/1.jpg", "/d/2.jpg"], ["car"], 0.3, model=_BoomModel(),
    )
    assert stats.zero_box_images == 2
    assert stats.total_boxes == 0


# ---------- llm_tune_strategy ----------

class _ScriptedLLM:
    model = "fake-llm"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.1, **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content)


def _stats() -> BatchStats:
    return BatchStats(
        images_sampled=5, prompts=["car", "person"], total_boxes=2,
        zero_box_images=3, missing_prompts=["person"], threshold=0.3,
    )


def test_llm_tune_strategy_parses_markdown_wrapped_json() -> None:
    """解析围栏包裹的 JSON（与 _llm_plan_once 同款剥离）。"""
    llm = _ScriptedLLM(
        "```json\n"
        '{"confidence_threshold": 0.25, "suggest_model": "yolo26x.pt",'
        ' "note": "缺类多，降阈值"}\n```'
    )
    out = llm_tune_strategy(llm, "检测汽车", _stats())
    assert out["confidence_threshold"] == 0.25
    assert out["suggest_model"] == "yolo26x.pt"
    assert out["note"] == "缺类多，降阈值"
    assert len(llm.calls) == 1  # 每批 1 次调用


@pytest.mark.parametrize(
    "content",
    [
        '{"confidence_threshold": 2.0}',           # 阈值越界
        '{"confidence_threshold": true}',          # 布尔拒绝
        '{"confidence_threshold": 0.3, "suggest_model": 123}',  # 模型名非法
        "[1, 2]",                                 # 非对象
    ],
)
def test_llm_tune_strategy_invalid_output_raises(content: str) -> None:
    """非法输出 → ValueError（调用方降级代码级默认参数）。"""
    with pytest.raises(ValueError):
        llm_tune_strategy(_ScriptedLLM(content), "检测汽车", _stats())


def test_llm_tune_strategy_null_suggest_model_ok() -> None:
    """suggest_model=null 合法（无模型建议）。"""
    out = llm_tune_strategy(
        _ScriptedLLM('{"confidence_threshold": 0.5, "suggest_model": null}'),
        "检测汽车", _stats(),
    )
    assert out["suggest_model"] is None


# ---------- apply_strategy ----------

def _step(conf: float = 0.3) -> SimpleNamespace:
    return SimpleNamespace(
        confidence_threshold=conf, model_hint="", model_name="yolo11n.pt",
        prompts=["car"],
    )


def test_apply_strategy_overrides_conf_and_hint() -> None:
    """conf 立即覆写；suggest_model 只进 model_hint（不中途换权重）。"""
    step = _step()
    applied = apply_strategy(step, {
        "confidence_threshold": 0.25, "suggest_model": "fasterrcnn_resnet50_fpn_v2",
        "note": "漏检",
    })
    assert step.confidence_threshold == 0.25
    assert applied["old_confidence_threshold"] == 0.3
    assert "fasterrcnn_resnet50_fpn_v2" in step.model_hint
    assert step.model_name == "yolo11n.pt"  # 模型名不动


def test_apply_strategy_clamps_conf() -> None:
    """越界 conf 钳位到 [0.05, 0.95]。"""
    step = _step()
    apply_strategy(step, {"confidence_threshold": 2.0})
    assert step.confidence_threshold == 0.95
    apply_strategy(step, {"confidence_threshold": -1})
    assert step.confidence_threshold == 0.05


def test_apply_strategy_without_suggestion_keeps_hint() -> None:
    """suggest_model 缺省 → model_hint 不动。"""
    step = _step()
    apply_strategy(step, {"confidence_threshold": 0.4})
    assert step.model_hint == ""
    assert step.confidence_threshold == 0.4


# ---------- _apply_batch_strategy（execute_plan 挂钩） ----------

def test_apply_batch_strategy_tunes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    """三步管线：抽样 → LLM 恰 1 次 → conf 覆写；console 输出建议摘要。"""
    monkeypatch.chdir(tmp_path)
    llm = _ScriptedLLM('{"confidence_threshold": 0.25, "suggest_model": null, "note": "ok"}')
    model = _FakeModel([1, 0, 2])
    step = _step()

    applied = _apply_batch_strategy(
        step, [Path(f"/d/{i}.jpg") for i in range(3)], model, llm, "检测汽车",
    )

    assert applied is not None
    assert applied["confidence_threshold"] == 0.25
    assert step.confidence_threshold == 0.25
    assert len(llm.calls) == 1  # 每批 1 次 LLM 调用
    out = capsys.readouterr().out
    assert "批次策略" in out and "0.3→0.25" in out


def test_apply_batch_strategy_no_llm_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    """无 LLM 客户端（无 key）→ 跳过，步骤参数不动，零影响。"""
    monkeypatch.chdir(tmp_path)
    step = _step()
    out = _apply_batch_strategy(step, [Path("/d/1.jpg")], None, None, "检测汽车")
    assert out is None
    assert step.confidence_threshold == 0.3
    assert "无 LLM" in capsys.readouterr().out


def test_apply_batch_strategy_llm_failure_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    """LLM 输出非法 → 黄字降级，步骤参数不动。"""
    monkeypatch.chdir(tmp_path)
    step = _step()
    model = _FakeModel([1, 0])
    out = _apply_batch_strategy(
        step, [Path("/d/1.jpg"), Path("/d/2.jpg")], model,
        _ScriptedLLM("not-json"), "检测汽车",
    )
    assert out is None
    assert step.confidence_threshold == 0.3
    assert "批次策略失败" in capsys.readouterr().out


# ---------- execute_plan 集成挂钩 ----------

def _make_plan(img_dir: Path) -> Any:
    from auto2dlabel.agent.planner import _dict_to_plan

    return _dict_to_plan({
        "steps": [{
            "step_id": 1,
            "task_type": "object_detection",
            "source": str(img_dir),
            "prompts": ["car"],
            "model_name": "yolo11n.pt",
        }],
        "confirm_timeout": 30,
    })


@pytest.fixture()
def plan_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict[str, Any]]]:
    """两张真实小图 + 打桩 execute_plan 全部重路径（零权重零 I/O 污染）。"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(img_dir / name)

    hook_calls: list[dict[str, Any]] = []

    def _fake_hook(step: Any, images: Any, model: Any, llm: Any, instruction: str) -> None:
        hook_calls.append({
            "step": step, "images": images, "model": model,
            "llm": llm, "instruction": instruction,
        })

    monkeypatch.setattr("auto2dlabel.cli_execute._apply_batch_strategy", _fake_hook)
    monkeypatch.setattr("auto2dlabel.cli_execute._execute_chunk", lambda *a, **kw: None)
    monkeypatch.setattr("auto2dlabel.cli_execute._resolve_step_batch", lambda *a, **kw: (1, 0))
    monkeypatch.setattr(
        "auto2dlabel.models.detection.create_detection_model",
        lambda name=None, **kw: _FakeModel([1] * 10),
    )
    monkeypatch.setattr("auto2dlabel.tools.log.log_chat_call", lambda **kw: None)
    monkeypatch.chdir(tmp_path)
    return img_dir, hook_calls


def test_execute_plan_invokes_batch_strategy_once_per_step(
    plan_env: tuple[Path, list[dict[str, Any]]],
) -> None:
    """--batch-strategy：多图检测步骤恰好挂钩一次（LLM 每批 1 次），单图不挂钩。"""
    img_dir, hook_calls = plan_env

    class _LLM:
        model = "fake-llm"

    execute_plan(_make_plan(img_dir), batch_strategy=True, llm=_LLM())
    assert len(hook_calls) == 1
    call = hook_calls[0]
    assert call["instruction"] == ""
    assert len(call["images"]) == 2
    assert call["step"].prompts == ["car"]

    # 单图步骤不挂钩（无批量意义）
    hook_calls.clear()
    (img_dir / "b.jpg").unlink()
    execute_plan(_make_plan(img_dir), batch_strategy=True, llm=_LLM())
    assert hook_calls == []


def test_execute_plan_without_batch_strategy_flag_no_hook(
    plan_env: tuple[Path, list[dict[str, Any]]],
) -> None:
    """未开 --batch-strategy → 挂钩不调用（零影响回归）。"""
    img_dir, hook_calls = plan_env
    execute_plan(_make_plan(img_dir), batch_strategy=False, llm=None)
    assert hook_calls == []
