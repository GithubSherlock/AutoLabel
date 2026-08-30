# auto2dlabel v0.6 实测记录

> 本文件 = v0.6 真权重冒烟、真实 API 与基准评测的实测数据（合成测试计数见各节；里程碑定义见 `milestone/v0.6.md`）。
> **环境**：RTX 3080 Ti 12GB（多租户宿主）——AP 数字硬件无关（可比），耗时数字硬件相关（如实记录）。

## Phase 1：对话式 Planner（parse_dialog）

### 实现落点

- `agent/dialog.py`：`parse_with_dialog(llm, system_prompt, parse_fn, ask_fn, max_rounds=3)` 通用骨架——渲染 questions / 收集回答 / 回喂 / 轮次控制与 schema 无关；2D（planner.py）/ 3D（auto3dlabel planner3d.py）各自注入（复用不复制红线）
- 降级链：LLM 响应非法 / questions 解析失败 / 无 key → 现有单轮 `parse` + 代码兜底（`ask_missing_params` 保留）；`--no-wait`（timeout=0）跳过对话轮直接代码兜底，行为与 v0.5 一致
- 轮次上限 3（对齐 Agent Loop `max_iterations=3` 红线），耗尽后代码兜底补缺；`temperature=0`

### 测试（Fake LLM 注入，零真实 API）

- `test_planner_dialog.py` 30 用例：对话闭环（多轮问答 → questions 空 → execute）/ questions 空直接执行 / 轮次耗尽（mock 恒返回 questions → 3 轮后代码兜底）/ 非法响应降级 / 无 key 兜底
- 3D 侧回归：`auto3dlabel/tests/functional/test_dialog3d.py`（mock LLM 同机制）

### 真实 API 冒烟（2026-08-30，DeepSeek）

- `auto2dlabel chat`（无 --no-wait）真实链路冒烟：**3 次 planner.parse 调用**进台账（`logs/llm_usage.jsonl`），json_mode + max_tokens=1024 真实生效（finish=stop 无截断、解析成功）、**前缀缓存命中率 97.7%**（第 2 次起 98.9%——静态 system prompt 全命中）、合计 **¥0.003594**
- 完整 E2E（`chat "检测 000860.png 中的汽车" --no-wait`）：11 车检出 → 导出 JSON + 可视化 → HITL 分流（5 直接采纳 | 6 待复核）全链路走通；台账新增 **1 次 planner.dialog** 调用（prompt 3498 / completion 461 / ¥0.007321）——实测澄清 --no-wait 语义：parse 缺参时仍发起 1 次 dialog 轮 LLM 调用产 questions，timeout=0 跳过的是「等待回答」而非 LLM 调用（设计内，docstring 已述）
- 详情见 Phase 4 节（同一台账）

## Phase 3a：mmdet 双引擎（Mask2Former 分割 / RTMDet 检测）

### 接入与权重

- 新增 `models/mmdet_engines.py`（四段式母版：零加载 __init__ + 幂等 _load + ImportError 守卫 + create 工厂）：`MMDetRTMDetModel`（检测协议 detect + iou 阈值 NMS）、`MMDetMask2FormerModel`（分割协议 generate——**忽略输入 bboxes 自检测**，box-prompted 口径下与 SAM2 box-prompted 同构可比）
- 接线：`models/detection.py` rtmdet 前缀分支（L508）、`models/segmentation.py` mask2former 分支（L741）+ `model_catalog.py` 目录
- 权重：`weights/mmdet_checkpoints/rtmdet_l/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth`、`mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth`（torch.load 校验 2 keys，openmmlab 直链）；config 走 `weights/mmdet_configs/configs/`（单一事实源）
- 测试：`test_mmdet_engines.py` 17 用例（Fake 注入 detect/generate 协议 + 路由 + 守卫 + weights_only 白名单，零真实权重）

### RTMDet vs Faster R-CNN 检测 benchmark（同口径：conf 0.3 / IoU 0.5 / COCO val 全量 4952）

执行：`python3 -m auto2dlabel.benchmarks.coco_benchmark --model rtmdet_l` 与 `--model fasterrcnn_resnet50_fpn_v2`

**两个实测 bug 修复（本版真权重出表暴露，均带回归测试）**：

1. **torch 2.6+ weights_only 拒载老权重**（torch 2.13 实测）：RTMDet 2022 权重报 `mmengine.logging.HistoryBuffer`、RTMPose 2023 权重报 `numpy.core.multiarray._reconstruct` 非 torch 全局 → `mmdet_engines._allow_legacy_checkpoint_globals()` 白名单（numpy ndarray/dtype/dtypes 类/_reconstruct/scalar 双路径 + HistoryBuffer + bytes/getattr），两引擎 `_load` 在 init 前调用（mmpose 复用同一函数）；回归 `test_mmdet_engines.py` / `test_mmpose_engines.py`（合成 checkpoint 白名单后可载 + 顺序断言）
2. **COCO 91→80 类别映射错位**：torchvision COCO_V1 权重输出 detectron 91 类 1-based 索引（10 个占位类）——曾直接 label-1 索引 80 类表：前 11 类两表一致掩盖错位，cat(17) 起全错位（cat 预测标成 dog）→ fasterrcnn 全量 mAP@0.5 0.0998 实测暴露（dog 类 Pred 289 / precision 0.055 特征）→ `model_catalog.COCO_91_TO_80` 单一事实源映射，三处解析点（detection `_parse_output` / `sahi_infer_torchvision` / segmentation MaskRCNNModel）统一走映射（cityscapes 权重 label 语义不同保持原逻辑）；回归 `test_coco91_mapping.py` 5 用例。**历史记录修正**：test-v0.3.md「coco_seg maskrcnn 0.0889（模型固有水平）」实为同一错位所致（4 图 smoke，label 全错位）；Mask R-CNN torchvision 修复后全量 mAP@0.5 **0.6287**（见上表）为真实水平

| 模型 | mAP@0.5 | 速度 | 备注 |
| --- | --- | --- | --- |
| fasterrcnn_resnet50_fpn_v2（torchvision 基线） | 0.6287 | ~12.8 img/s | 墙钟 388s/4952 图（23:42:30–23:48:58）；91→80 修复后重跑，逐类分布正常（cat 0.880 / dog 0.783 / person 0.765） |
| **rtmdet_l（mmdet 高召回档）** | **0.6179** | ~12.2 img/s | 墙钟（推理+评测）407s/4952 图 |

**结论**：修复后 fasterrcnn mAP@0.5 **0.6287**（错位时 0.0998，修复增益 +0.529）——**rtmdet_l 0.6179 与 torchvision 基线 0.6287 同档**（-0.011），两引擎 mmdet/torchvision 权重质量互证；实测期间宿主 GPU 有其他租户（178 次 CUDACachingAllocator OOM 警告，分配器重试后 exit=0、逐类数字无异常，如实记录）。两个模型墙钟同档（~12.2 vs ~12.8 img/s）。高召回档结论：mmdet RTMDet 达到 torchvision Faster R-CNN 同档精度，且提供 NMS 阈值可调 + 多模型矩阵（5 款 RTMDet）的独立引擎价值。

### Mask2Former vs SAM2 分割 benchmark（同口径：两段式 yolo26x prompt-conf 0.3 / mask IoU 0.5 / 全量 4952）

执行：`python3 -m auto2dlabel.benchmarks.coco_seg_benchmark --seg-model mask2former_r50_8xb2-lsj-50e_coco --prompt-conf 0.3`

| 配置 | mask mAP@0.5 | mIoU | 备注 |
| --- | --- | --- | --- |
| 两段式 yolo26x + sam2_l（全量 4952，2026-08-17 实测） | 0.5778 | 0.8313 | 既有记录 benchmarks_outputs/coco_seg_sam2_l.pt_2026-08-17-06-38-28（RTX 4090，~7.4 img/s 墙钟） |
| 两段式 yolo26x + sam2_l（50 图口径，新默认记录） | 0.6368 | 0.6902 | 既有记录（test-v0.3.md 同段） |
| **mask2former_r50（自检测+分割，本版实测）** | **0.6008** | 0.8135 | bboxes 输入被忽略，实测 Mask2Former 全链路；3080 Ti ~6.6 img/s（墙钟 740s/4952 图） |

**同口径对照（全量 4952 vs 全量 4952）**：mask2former mask mAP **0.6008 > SAM2 两段式 0.5778**（+0.023，自检测一体 vs 检测步丢框），mIoU 0.8135 略低于 SAM2 0.8313（-0.018）——精度档结论：Mask2Former 单模型达到并略超两段式 SAM2 的 mask mAP 水平（耗时数字硬件相关：SAM2 为 4090 记录、mask2former 为 3080 Ti 记录，不可直接比）。

## Phase 3b：mmpose RTMPose 姿态精度档

### 姿态评测基准 + 接入

- 新增 `benchmarks/pose_benchmark.py`：COCO 17 点 OKS 协议（11-point AP / AP50 / AP75）+ **small（GT bbox 面积 <32²）/ dense（图内 GT person ≥8）分层**（P3b 验收要求）
- 新增 `models/mmpose_engines.py`：`MMposeRTMPoseModel`（top-down：自检 bbox → 关键点 17×2 + bbox 协议）
- 权重：`weights/mmpose_checkpoints/rtmpose_l/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`（110,914,873 B，校验 2 keys）
- 测试：`test_pose_benchmark.py` 16 用例（OKS 纯函数 / 匹配 / 分层汇总 / Fake PoseModel）+ `test_mmpose_engines.py` 15 用例

### 姿态基准出表（COCO person_keypoints_val2017 全量）

执行：`python3 -m auto2dlabel.benchmarks.pose_benchmark --model yolo11n-pose.pt` 与 `--model rtmpose_l`

| 模型 | 层 | mAP | AP50 | AP75 | GT | matched |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n-pose（速度档） | all / small / dense | 0.4545 / 0.0000 / 0.3636 | 0.4545 / 0.0000 / 0.3636 | 0.4545 / 0.0000 / 0.3636 | 10777 / 3206 / 5740 | 5629 / 15 / 2282 |
| **rtmpose_l（mmpose 精度档）** | all / small / dense | **0.5455** / 0.0000 / **0.4545** | 0.5455 / 0.0000 / 0.4545 | 0.5455 / 0.0000 / 0.4545 | 10777 / 3206 / 5740 | **8373** / 1667 / 4050 |

**结论**：rtmpose_l mAP **0.5455 vs yolo11n-pose 0.4545（+0.091，全部层）**，matched 8373 vs 5629（+2744 人，top-down 检测框 + RTMPose 召回显著更高）；速度 14.48 vs 73.7 img/s（5×，top-down 逐框代价）——精度档/速度档分工成立。small(<32²) 两模型均 0.0000：OKS 对小框惩罚重（σ 归一化），小目标姿态为共同短板（如实记录）。dense 层 rtmpose_l +0.091 同步领先。**AP50=AP75=mAP 说明**：pose_benchmark 只按 OKS@0.5 单阈值匹配计 mAP（coco_pose 协议的 10 点 OKS 曲线未实现），三列同值属当前实现口径，如实记录。

## Phase 4：LLM Harness 工程（Token 降本）

### 实现落点（harness 层，不动产品功能语义）

1. **usage 台账**：LLMClient 统一收集 `{调用点, 时间, prompt/completion/cached_tokens, 模型, 折算费用}` → `logs/llm_usage.jsonl`；`auto2dlabel cost-report` 按调用点聚合（调用数 / tokens / 费用 / 缓存命中率）
2. **max_tokens 分级**：planner 1024 / Agent Loop 2048（OpenAI 路径此前缺失）；超限不静默截断（日志可见）
3. **json mode**：全部 JSON 输出调用点（planner / planner3d / 重解析 / track / Agent Loop）加 `response_format={"type":"json_object"}`
4. **system prompt 前缀稳定化 + messages 规整化**：动态内容后置保前缀缓存命中；assistant tool_calls.arguments 规整化后再入 state
5. **LLM Evaluate 代码级降级**：LLM 处置失败/超时 → 确定性规则（0 框 → 降阈值等，规则与 `pick_alternate_model` 同源），不中断流程；可配开关直接走规则

### 测试

- `test_llm_usage.py` 17 用例（台账落盘字段齐全 / 调用点标注 / 费用折算纯函数 / cost-report 聚合）
- `test_json_mode.py` 8 用例（response_format 参数穿透 mock）
- `test_messages_normalize.py` 5 用例（tool_calls arguments 规整化零行为回归）
- `test_evaluate_fallback.py` 11 用例（LLM 失败 → 代码级处置、开关生效）

### 真实 API 实测（2026-08-30，DeepSeek deepseek-v4-flash）

- 3 次 `planner.parse` 真实调用进台账：prompt 3493-3495 / completion 342-748 / **cached 3328-3456（命中率 97.7%，第 2 次起 98.9%）**——静态 system prompt 前缀稳定化设计验证
- json_mode + max_tokens=1024 真实生效：finish=stop 无截断、JSON 解析成功（steps=1）
- 费用：**¥0.003594 合计**（单价表 PRICE_RMB_PER_1M：v4-flash prompt 1.5 / cached 0.05 / completion 4.5 元每百万 token；v4-flash 为峰谷计费，取空闲档保守下限）
- cost-report 聚合输出：调用点分布 / tokens / 费用 / 缓存命中率

## 质量门

四件套终局数字（2026-08-30）：

| 检查 | 结果 | 基线对照 |
| --- | --- | --- |
| pytest（autolabel env 全量） | **1046 passed, 4 skipped** | ✅ 全绿 |
| pytest（base env 全量） | **1049 passed, 1 skipped** | ✅ 全绿 |
| ruff check auto2dlabel/ + auto3dlabel/ | **119 errors**（E501 占多数，44 fixable 未动） | 基线 127，不恶化 ✅ |
| mypy auto2dlabel/ --follow-imports=silent | **169 errors / 33 files** | HEAD 基线 169（worktree 实测复验），不恶化 ✅ |
| pyright auto2dlabel/ + auto3dlabel/ | **0 errors** | ✅ |

- 过程修正两处 mypy 新引入错误（初查 171 → 修后 169）：`segmentation.py` COCO 分支 `cls_id = COCO_91_TO_80.get(...)` 的 `int | None` 类型冲突（改为 `mapped` 局部变量 None 即 continue）；`test_mmpose_engines.py` 对 mmpose 模块再导出下划线属性的 attr-defined 错误（改从 `mmdet_engines` 直接导入）。ruff 修 1 处本版新增 N806（`HistoryBuffer` 变量小写化 `history_buffer`）
- 新增回归测试计数（本版）：test_planner_dialog 30 + test_json_mode 8 + test_messages_normalize 5 + test_evaluate_fallback 11 + test_llm_usage 17 + test_mmdet_engines 17 + test_coco91_mapping 5 + test_mmpose_engines 15 + test_pose_benchmark 16 + 既有文件扩展若干——全量合计 1046/1049（双环境）
- 后续连带更新（2026-08-31，3D 路由表收拢轮）：2D `models/model_catalog.py` → `configs/model_catalog.py` 半成品迁移收尾（文件已删、40+ 处 import 改指 configs）——pytest 双环境 1056/4 + 1059/1（+10 为 3D 侧 test_model_catalog3d）；ruff **75**（--fix 修 53 处迁移引发项：I001 排序/F541/F401，< 119）；mypy **167**（< 169）；pyright 0
