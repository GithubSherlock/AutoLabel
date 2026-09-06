"""provider 注册表测试（v1.0 P2，零真实 LLM）。

覆盖：providers.yaml 加载/损坏回退、ProviderSpec 字段、create_client 模型层级
（显式 > model_env > spec.model > 内置默认）、api_key_env 判定（无 key → False /
api_key_env: null → 占位 key "EMPTY"）、台账 provider 维度（落盘字段 / 旧行 "-"
容错 / 聚合分行）、yaml price 扩展单价表。

2026-09-07 审查修复回归：#1 跨 provider key 不串扰（client property env 兜底
按 provider_name 门控）、#9 yaml 同名条目字段级回退内置、#10 env 字段类型容错、
#19 provider 空串归一、#20 price: {} → None、#21 重复 name 告警。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from auto2dlabel.agent import llm as llm_mod
from auto2dlabel.agent.llm import (
    OpenAIClient,
    ProviderSpec,
    UsageStats,
    aggregate_usage,
    create_client,
    estimate_cost_rmb,
    list_providers,
    load_provider_config,
    load_usage_logs,
    log_llm_usage,
)

# 测试注入 yaml（绝不含密钥，api_key_env 只引用变量名）
_SAMPLE_YAML = """\
providers:
  - name: ollama
    provider: openai
    model: qwen2.5:7b
    base_url: http://localhost:11434/v1
    api_key_env: null
  - name: myprovider
    provider: openai
    model: my-model
    api_key_env: MYPROVIDER_API_KEY
    price:
      prompt: 3
"""


def _install_yaml(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """把默认加载路径替换为 tmp 文件（create_client/list_providers 全走此路）。"""
    monkeypatch.setattr(
        llm_mod, "_load_default_providers", lambda: llm_mod._load_provider_config(path)
    )


class _CaptureSDK:
    """记录 SDK 构造 kwargs 的桩（client property 调用时 import 后替换构造）。

    零真实客户端/零网络——只断言我们传给 SDK 的参数（key 串扰防护验证）。
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


# ============ yaml 加载 ============


def test_load_provider_config_fields(tmp_path: Path) -> None:
    """yaml 条目 → ProviderSpec 字段齐全；price 缺失键补 0。"""
    path = tmp_path / "providers.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    specs = load_provider_config(path)
    assert len(specs) == 2
    ollama = specs[0]
    assert ollama == ProviderSpec(
        name="ollama",
        provider="openai",
        model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
    )
    mine = specs[1]
    assert mine.name == "myprovider"
    assert mine.api_key_env == "MYPROVIDER_API_KEY"
    assert mine.price == {"prompt": 3.0, "cached": 0.0, "completion": 0.0}


def test_load_missing_or_corrupted_returns_empty(tmp_path: Path) -> None:
    """缺失文件 / 损坏 yaml / 顶层非 dict → []（内置表兜底，绝不 raise）。"""
    assert load_provider_config(tmp_path / "nope.yaml") == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [\n  unclosed", encoding="utf-8")
    assert load_provider_config(bad) == []
    weird = tmp_path / "weird.yaml"
    weird.write_text("- just\n- a list\n", encoding="utf-8")
    assert load_provider_config(weird) == []
    bad_price = tmp_path / "bad_price.yaml"
    bad_price.write_text("providers:\n  - name: x\n    price: not-a-dict\n", encoding="utf-8")
    assert load_provider_config(bad_price)[0].price is None


def test_load_cached_idempotent() -> None:
    """默认路径加载 lru_cache 幂等（同一次进程内返回同一列表）。"""
    assert llm_mod._load_default_providers() is llm_mod._load_default_providers()


def test_yaml_env_fields_non_str_tolerated(tmp_path: Path) -> None:
    """#10：model_env/api_key_env/base_url_env/base_url 误写列表/数字 → None 不 raise。"""
    path = tmp_path / "providers.yaml"
    path.write_text(
        "providers:\n"
        "  - name: x\n"
        "    provider: openai\n"
        "    model: m\n"
        "    model_env: 123\n"
        "    base_url: 456\n"
        "    base_url_env: [DEEPSEEK_BASE_URL]\n"
        "    api_key_env: [DEEPSEEK_API_KEY]\n",
        encoding="utf-8",
    )
    spec = load_provider_config(path)[0]
    assert spec.model_env is None
    assert spec.base_url is None
    assert spec.base_url_env is None
    assert spec.api_key_env is None
    assert spec.model == "m"  # 正常字段不受牵连


def test_load_duplicate_name_warns_not_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """#21：yaml 重复 name → 告警不 raise（静默后者胜消除，两条均保留）。"""
    import logging

    path = tmp_path / "providers.yaml"
    path.write_text(
        "providers:\n"
        "  - name: dup\n    provider: openai\n    model: m1\n"
        "  - name: dup\n    provider: openai\n    model: m2\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="auto2dlabel.agent.llm"):
        specs = load_provider_config(path)
    assert len(specs) == 2
    assert any("重复 name" in r.message for r in caplog.records)


def test_normalize_price_empty_dict_none() -> None:
    """#20：price: {}（空 dict）→ None——0 元与未知不混同（台账记 null）。"""
    assert llm_mod._normalize_price({}) is None
    assert llm_mod._normalize_price(None) is None
    # 部分键非空 dict 语义不变：缺失键补 0
    assert llm_mod._normalize_price({"prompt": 1.5}) == {
        "prompt": 1.5,
        "cached": 0.0,
        "completion": 0.0,
    }


def test_yaml_price_empty_dict_spec_none(tmp_path: Path) -> None:
    """#20：yaml 条目 price: {} → spec.price is None（不进单价表）。"""
    path = tmp_path / "providers.yaml"
    path.write_text(
        "providers:\n  - name: x\n    provider: openai\n    model: m\n    price: {}\n",
        encoding="utf-8",
    )
    assert load_provider_config(path)[0].price is None


# ============ list_providers 并集 ============


def test_list_providers_union_with_builtins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yaml 条目 ∪ 内置三条目（按 name 去重，yaml 优先）。"""
    path = tmp_path / "providers.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    _install_yaml(monkeypatch, path)
    names = [s.name for s in list_providers()]
    assert names == ["deepseek", "openai", "anthropic", "ollama", "myprovider"]
    # yaml 优先：重写 deepseek 条目验证覆盖
    override = tmp_path / "override.yaml"
    override.write_text(
        "providers:\n  - name: deepseek\n    provider: openai\n    model: custom-ds\n",
        encoding="utf-8",
    )
    _install_yaml(monkeypatch, override)
    ds = next(s for s in list_providers() if s.name == "deepseek")
    assert ds.model == "custom-ds"


def test_same_name_partial_yaml_falls_back_to_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#9：与内置同名的 yaml 只写部分字段 → None/空串字段回退内置（字段级合并）。

    整条覆盖会把 api_key_env/base_url 等未声明字段清空——局部覆盖 yaml
    只声明想改的字段，其余继承内置。
    """
    path = tmp_path / "providers.yaml"
    path.write_text(
        "providers:\n  - name: deepseek\n    model: custom-ds\n",
        encoding="utf-8",
    )
    _install_yaml(monkeypatch, path)
    ds = next(s for s in list_providers() if s.name == "deepseek")
    builtin = next(s for s in llm_mod._BUILTIN_PROVIDERS if s.name == "deepseek")
    # yaml 声明字段保留
    assert ds.model == "custom-ds"
    # 未声明字段回退内置对应字段
    assert ds.provider == builtin.provider
    assert ds.model_env == builtin.model_env
    assert ds.base_url == builtin.base_url
    assert ds.base_url_env == builtin.base_url_env
    assert ds.api_key_env == builtin.api_key_env
    assert ds.price == builtin.price


# ============ create_client 模型层级 ============


def test_model_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 model > model_env 的 env > spec.model > 内置默认，四档各一断言。"""
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    # ① 显式 model 最高优先
    assert create_client("deepseek", model="deepseek-reasoner").model == "deepseek-reasoner"
    # ② model_env 的 env 次之
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    assert create_client("deepseek").model == "deepseek-v4-flash"
    # ③ spec.model（yaml 内置 deepseek 条目默认值）
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert create_client("deepseek").model == "deepseek-chat"
    # ④ 内置兜底表默认（yaml 只含 ollama，无 deepseek 条目）
    path = tmp_path / "providers.yaml"
    path.write_text("providers: []\n", encoding="utf-8")
    _install_yaml(monkeypatch, path)
    assert create_client("deepseek").model == "deepseek-chat"


def test_create_client_has_credentials_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 env 无显式 key → False；显式 key / env key → True（v0.6 契约复验）。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not create_client("deepseek").has_credentials
    assert create_client("deepseek", api_key="k").has_credentials
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert create_client("openai").has_credentials
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert create_client("anthropic").has_credentials


def test_create_client_local_endpoint_empty_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """api_key_env: null 的本地端点 → 占位 key "EMPTY"、has_credentials 恒 True。"""
    path = tmp_path / "providers.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    _install_yaml(monkeypatch, path)
    client = create_client("ollama")
    assert client.api_key == "EMPTY"
    assert client.has_credentials
    assert client.provider_name == "ollama"  # 台账 provider 维度
    assert client.base_url == "http://localhost:11434/v1"


def test_registry_client_no_cross_provider_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1：registry 创建的 DeepSeek 客户端不串入 OPENAI_API_KEY。

    DEEPSEEK_API_KEY 缺失而 OPENAI_API_KEY 存在时，若 client property 再兜底
    env 会把 OpenAI key 发给 DeepSeek——修复后 provider_name 非空禁用 env 兜底。
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    import openai  # property 内惰性 import；仅替换构造为桩，零真实 SDK/网络

    monkeypatch.setattr(openai, "OpenAI", _CaptureSDK)
    client = create_client("deepseek")
    assert client.api_key is None  # spec.api_key_env 的 env 缺失 → 无凭据走代码级兜底
    assert client.provider_name == "deepseek"
    sdk = client.client  # 触发 property 构造（键解析在 create_client，此处不得再兜底）
    assert sdk.kwargs["api_key"] is None  # 关键断言：不串入 OPENAI_API_KEY


def test_hand_built_client_keeps_env_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1：手构造 OpenAIClient（provider_name==""）保留 OPENAI_API_KEY env 兜底（v0.6 兼容）。"""
    import openai

    monkeypatch.setattr(openai, "OpenAI", _CaptureSDK)
    # 无显式 key → env 兜底（手构造客户端旧行为）
    client = OpenAIClient(model="gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert client.client.kwargs["api_key"] == "env-key"
    # 显式 key 优先于 env（self.api_key or env 原语义）
    client2 = OpenAIClient(model="gpt-4o", api_key="explicit-key")
    assert client2.client.kwargs["api_key"] == "explicit-key"


def test_create_client_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url_env 的 env 覆盖 spec.base_url（DEEPSEEK_BASE_URL 旧行为保持）。"""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://proxy:9000")
    assert create_client("deepseek").base_url == "http://proxy:9000"
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    assert create_client("deepseek").base_url == "https://api.deepseek.com"


def test_create_client_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知 provider → ValueError 文案含可用列表。"""
    with pytest.raises(ValueError, match="Unknown provider 'nope'"):
        create_client("nope")
    with pytest.raises(ValueError, match=r"Available: \['deepseek'"):
        create_client("nope")


def test_create_client_bad_impl_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """yaml 条目 provider 实现名不受支持 → 清晰 ValueError（非 KeyError）。"""
    path = tmp_path / "providers.yaml"
    path.write_text("providers:\n  - name: x\n    provider: grpc\n    model: m\n", encoding="utf-8")
    _install_yaml(monkeypatch, path)
    with pytest.raises(ValueError, match="不受支持"):
        create_client("x")


# ============ 台账 provider 维度 ============


def test_log_llm_usage_provider_field(tmp_path: Path) -> None:
    """log_llm_usage(provider=...) 落盘 provider 字段；旧行无该字段 → 聚合归 "-"。"""
    path = tmp_path / "usage.jsonl"
    log_llm_usage(
        call_site="planner.parse",
        model="deepseek-chat",
        usage=UsageStats(prompt_tokens=100, completion_tokens=10),
        provider="deepseek",
        path=path,
    )
    assert load_usage_logs(path)[0]["provider"] == "deepseek"

    # 旧行（无 provider 键）→ "-" 容错
    path.write_text(
        '{"call_site": "old.site", "model": "deepseek-chat",'
        ' "prompt_tokens": 5, "completion_tokens": 1}\n',
        encoding="utf-8",
    )
    rows = aggregate_usage(load_usage_logs(path))
    assert rows[0]["provider"] == "-"


def test_aggregate_usage_splits_by_provider(tmp_path: Path) -> None:
    """同 (call_site, model) 不同 provider → 分行。"""
    path = tmp_path / "usage.jsonl"
    for provider in ("deepseek", "ollama"):
        log_llm_usage(
            call_site="planner.parse",
            model="deepseek-chat",
            usage=UsageStats(prompt_tokens=10, completion_tokens=10),
            provider=provider,
            path=path,
        )
    rows = aggregate_usage(load_usage_logs(path))
    assert len(rows) == 2
    assert {r["provider"] for r in rows} == {"deepseek", "ollama"}
    assert all(r["calls"] == 1 for r in rows)


def test_aggregate_provider_empty_string_joins_missing(tmp_path: Path) -> None:
    """#19：provider 空串行与缺键行归一 "-"（同 (call_site, model) 不分裂两行）。"""
    path = tmp_path / "usage.jsonl"
    path.write_text(
        '{"call_site": "a", "model": "m", "provider": "", "prompt_tokens": 1}\n'
        '{"call_site": "a", "model": "m", "prompt_tokens": 2}\n',
        encoding="utf-8",
    )
    rows = aggregate_usage(load_usage_logs(path))
    assert len(rows) == 1
    assert rows[0]["provider"] == "-"
    assert rows[0]["calls"] == 2
    assert rows[0]["prompt_tokens"] == 3


# ============ yaml price 扩展 ============


def test_price_from_yaml_extends_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """代码表未命中的模型 → yaml price 计费（缺失 cached 键补 0）。"""
    path = tmp_path / "providers.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    _install_yaml(monkeypatch, path)
    usage = UsageStats(prompt_tokens=1000, completion_tokens=100, cached_tokens=0)
    # (1000*3 + 100*0) / 1e6 = 0.003
    assert estimate_cost_rmb("my-model", usage) == pytest.approx(0.003)
    # 代码精确表优先于 yaml（deepseek-chat 仍按 2/0.5/8）
    assert estimate_cost_rmb("deepseek-chat", usage) == pytest.approx(0.0028)
