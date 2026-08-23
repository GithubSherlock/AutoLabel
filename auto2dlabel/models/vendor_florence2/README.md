# vendor_florence2 —— Vendored 官方 Florence-2 代码

## 来源

- 仓库：[microsoft/Florence-2](https://huggingface.co/microsoft/Florence-2-base)（根目录代码文件）
- 文件：`modeling_florence2.py` / `processing_florence2.py` / `configuration_florence2.py`（原样保留 Apache-2.0 许可头）
- 权重：`microsoft/Florence-2-base`，进 `auto2dlabel/weights/hf/`（HF_HOME 统一管理，不入库）

## Vendoring 动机

transformers 5.15.0 内置的 florence2 集成有根本性缺陷，无法直接使用官方 checkpoint：

1. **缺 tokenizer 类**：`tokenization_florence2.py` 不存在，`AutoTokenizer` 把
   florence2 映射到 BartTokenizer（5.15 中已是 RobertaTokenizer 别名），没有
   `image_token`/`additional_special_tokens`。
2. **视觉骨干网络被重写**：5.15 内置建模把官方 DaViT 换成自研结构
   （Florence2VisionMLP=Llama4VisionMLP 等），与官方 checkpoint 权重不兼容
   （UNEXPECTED/MISSING 大量权重 → 输出垃圾文本 "allow allow..."）。

官方自定义代码 + 少量 5.15 兼容补丁（见下）可正常加载官方 checkpoint
（实测 CPU 端到端生成有效 `<loc_x>` 序列）。vendored 到仓库内以保持
**纯本地部署红线**（不 trust_remote_code，不依赖 HF Hub 动态下载代码）。

## 5.15 兼容补丁清单（与原文件的全部差异）

1. `configuration_florence2.py`：`forced_bos_token_id` 读取改 `getattr(..., None)`
   （5.15 PretrainedConfig 无该字段，官方代码直接点访问会 AttributeError）。
2. `modeling_florence2.py` `_supports_sdpa` / `_supports_flash_attn_2`：改为防御性
   getter——5.15 移除了这两个类属性（统一 attention 分派），且
   `super().__init__()` 期间 `language_model` 尚未初始化（`getattr` 兜底）。
3. `processing_florence2.py`：`Florence2Processor.__init__` 扩展特殊 token 时
   `tokenizer.additional_special_tokens` 改 `getattr(..., [])`（5.15 BartTokenizer
   = RobertaTokenizer 别名，无该属性；官方语义是「保留已有 special tokens
   再追加 1024 个」，空列表等价）。
4. `modeling_florence2.py` 模块级新增 `_past_to_legacy` / `_past_length` helper +
   4 处消费点替换：5.15 起 past_key_values 是 Cache 对象
   （EncoderDecoderCache，无 `__getitem__`，`[0][0].shape[2]` 直接 TypeError），
   decoder 循环入口把 Cache 按层还原为官方建模的旧式
   `(self_k, self_v, cross_k, cross_v)` tuple（此后模型自行返回/消费旧式
   tuple，自洽）；序列长度统一走 `get_seq_length()`。首轮 prefill 时 Cache
   预分配未初始化层（keys=None）→ 返回 None（官方语义：无 past）。
5. `modeling_florence2.py` 三处 `_tied_weights_keys` 由官方 list 格式改为
   5.15 dict 格式 `{target: source}`：5.15 只解析 dict（老 list 被忽略）。
6. `configuration_florence2.py`：`Florence2LanguageConfig` 增补
   `tie_word_embeddings: bool = True`——5.15 tie 机制要求
   `config.tie_word_embeddings` 为真才执行（getattr 默认 False 直接跳过），
   官方配置类缺该属性（checkpoint config.json 也无该键）。
7. `modeling_florence2.py`：删除官方两个 `_tie_weights` 方法（5.15 meta-init
   阶段执行会污染 meta 张量）+ wrapper 类加 `from_pretrained` 钩子——加载后
   手动把 encoder/decoder `embed_tokens.weight` 与 `lm_head.weight` 绑到
   `shared.weight` 的同一 Parameter。实测 5.15 下 checkpoint 仅存
   `language_model.model.shared.weight`（51289×768）一份，tie 目标在
   MISSING 记账下保持 meta 张量不 materialize（缺 5-7 时 embed_tokens
   随机初始化，**生成必为垃圾文本**：重复 token 塌陷）。

## dtype 红线（2026-08-23 实测）

官方 `config.json` 声明 `"torch_dtype": "float16"`（checkpoint 以 fp16 存储）。
CPU 推理必须 `dtype=torch.float32` 加载（fp16 权重 + fp32 图像输入 dtype
冲突直接 RuntimeError）；GPU 推理保持 fp16（显存减半、速度翻倍）。
`referential.py` 按 `torch.cuda.is_available()` 自动选择，勿在调用方散副本。

## 方形输入红线（2026-08-23 实测）

官方 `_encode_image` 断言方形特征图（`h * w == num_tokens`，训练尺寸
768×768 → DaViT stride 32 → 24×24=576+1=577 tokens），非方输入 assert 直接炸。
推理前必须 squash 到 768×768（`referential.py` 的 `FLORENCE_INFER_SIZE`），
不保宽高比——Florence 输出坐标按 x/y 分别反变换回原图系（线性可逆）。

## 质量门豁免

本目录为第三方官方代码，参与运行但不参与本项目质量门：pyright/ruff 走
`exclude`（pyproject `[tool.pyright]` / `[tool.ruff]`）；mypy 双保险——
`[tool.mypy] exclude`（仅递归发现 `mypy .` 生效）+ `[[tool.mypy.overrides]]`
`ignore_errors`（显式目录 `mypy auto2dlabel/` 也生效；跨模块调用点
`referential.py` 对无标注 `from_pretrained` 定点 `type: ignore`）。
官方代码风格（`\` 续行等）不按本项目规范改——保持与上游最小 diff，
升级官方版本时按补丁清单重放三处补丁即可。
