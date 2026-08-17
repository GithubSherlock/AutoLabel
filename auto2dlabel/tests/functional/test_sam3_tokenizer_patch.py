"""_patch_clip_tokenizer 回归测试。

背景：ultralytics SAM3 的 build_sam3 直接实例化
clip.simple_tokenizer.SimpleTokenizer 并当作可调用对象使用，而 PyPI
openai-clip 的 SimpleTokenizer 未定义 __call__（hasattr 沿 MRO 误报
type.__call__）→ SAM3 文本 prompt 路径 TypeError: 'SimpleTokenizer'
object is not callable。补丁在 _load 时懒生效，仅当 clip 缺失版本时
patch，已兼容版本零改动。
"""

from __future__ import annotations

import pytest

from auto2dlabel.models.segmentation import _patch_clip_tokenizer

clip = pytest.importorskip("clip", reason="clip 未安装（SAM3 可选依赖）")


def test_patch_makes_tokenizer_callable_with_list() -> None:
    """patch 后 SimpleTokenizer 实例可被 list[str] 调用，返回 [b, 77] tensor。"""
    _patch_clip_tokenizer()

    from clip.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    # 运行时 patch 后 __call__ 才存在，静态检查器无法推断
    tokenized = tokenizer(["small vehicle", "car"], context_length=77)  # pyright: ignore[reportCallIssue]

    assert tokenized.shape == (2, 77)
    # 官方 tokenize 约定：非零 token 以 EOT 结束，首 token 为 SOS
    assert tokenized[0][0].item() != 0


def test_patch_handles_single_string() -> None:
    """单字符串输入同样可用（[1, 77]）。"""
    _patch_clip_tokenizer()

    from clip.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    tokenized = tokenizer("car", context_length=77)  # pyright: ignore[reportCallIssue]

    assert tokenized.shape == (1, 77)


def test_patch_is_idempotent() -> None:
    """重复调用不叠加、不报错（加载路径可安全多次触发）。"""
    _patch_clip_tokenizer()
    _patch_clip_tokenizer()

    from clip.simple_tokenizer import SimpleTokenizer

    assert "__call__" in SimpleTokenizer.__dict__
    tokenized = SimpleTokenizer()(["car"], context_length=77)  # pyright: ignore[reportCallIssue]
    assert tokenized.shape == (1, 77)
