# Auto2dLabel

2D 自动标注主体（AutoLabel 的 2D 分支）：自然语言指令 → LLM Agent 规划 → 多模型引擎执行 → 代码级质量评估（不达标条件触发 LLM Evaluate / Critic）→ HITL 三档分流 → 多格式导出。「AI 为主、人为辅」。

**当前状态**：v0.1–v0.6 ✅ 完成（**105 模型** / 12 数据集 Benchmark）；**v1.0 ✅ 2026-09-06**（Agentic 交互化 P1–P5：`autolabel` 统一入口 + TUI + provider 注册表 + 后台任务面板 + 会话管理 + HITL 指挥台）；**v1.1 P1+P2 ✅ 2026-09-17**（RAG 标注经验库 + 质检 Agent Critic）。里程碑与实测数据见 [milestone/](milestone/) 与 [tests/test-v0.X.md](tests/)。

> 统一入口 `autolabel`（无参 → TUI，带指令 → 自动路由 2D/3D）见仓库根 [README.md](../README.md)；`auto2dlabel` CLI 命令原样保留。

## 能力一览

| 任务 | 模型 | 说明 |
| --- | --- | --- |
| 检测 | G-DINO（开放词汇）+ Ultralytics YOLO/RT-DETR + torchvision + mmdet RTMDet | 33 模型 |
| 分割 | SAM/SAM2/SAM2.1/SAM3 + FastSAM + Mask R-CNN（含 cityscapes 域内）+ Mask2Former + 语义分割 | 25 模型，默认 `sam2_l.pt` |
| 分类 | CLIP/SigLIP + torchvision ImageNet1K 14 款 | 16 模型 |
| OBB | YOLO-OBB 11/12/26 n/s/m/l/x | 15 模型，`Bbox.angle` 弧度约定 |
| Tracking | ByteTrack（默认）/ BoT-SORT（精度档）+ 指代 L1/L2/L3 + ROI | 视频/帧目录 → MOT 导出 + 标注成片 |
| Pose | YOLO-pose 15 款 + RTMPose | COCO keypoints 导出 |

## 常用命令

```bash
# 统一入口（推荐）：无参进 TUI 对话；带指令自动路由
autolabel
autolabel "检测 /data/images 中的汽车"

# Agent Loop（LLM 在环处置，每图 ≤3 迭代）
auto2dlabel run img.jpg "检测汽车和行人" -d yolo26x.pt -t 0.3
auto2dlabel run dir/ "检测汽车" --batch --resume outputs/batch_manifest.json

# chat：Planner 解析（缺参多轮对话确定）→ 代码级执行
auto2dlabel chat "检测 /data/images 中的汽车" --no-wait
auto2dlabel chat "分类为猫和狗，用 clip" --no-wait
auto2dlabel chat "检测 dir/ 中的汽车" --batch-strategy     # 批次级 LLM 调参

# 跟踪与指代
auto2dlabel run video.mp4 "检测行人" --track                       # ByteTrack
auto2dlabel run video.mp4 "检测行人" --track --bot-sort            # BoT-SORT 精度档
auto2dlabel run video.mp4 "跟踪红车旁边的行人" --track --refer-l2   # 指代 L2（失败升级 L3）
auto2dlabel run video.mp4 "检测行人" --track --roi auto            # 自动车道 ROI

# 成本台账与数据回路
auto2dlabel cost-report                                            # LLM usage 聚合
auto2dlabel cost-critic                                            # 质检 vs 规划成本对比
auto2dlabel sample --top-k 5                                       # 主动学习采样
auto2dlabel dataset add mydata /path/to/data --task 实例分割        # 自建数据集注册

# Web 复核
python3 -m auto2dlabel.web.server                                  # → http://localhost:8765

# 测试与质量门
pytest auto2dlabel/tests/                                          # 全项目 1354 passed, 4 skipped
cd auto2dlabel/tests/helpers && npm i jsdom && node smoke_web.js   # 前端冒烟（零浏览器，76 断言）
```

## 架构要点

- **两条执行路径**：`run` = Agent Loop（LLM tool-use 循环，质量不足时条件暴露 EvaluateTool 动态处置）；`chat` = Planner 单轮/多轮 JSON 解析 → `execute_plan` 纯代码执行（执行期零 LLM）
- **LLM 调用点仅 3 处**：① 对话式规划（`planner.parse_dialog` → `agent/dialog.py` 通用骨架）② 质量处置（`quality.ok == False` 条件触发）③ 批次调参（`--batch-strategy`，每批恰 1 次）；全部经 `agent/llm.py` 统一出口（usage 台账 / json mode / 前缀缓存 / 流式）
- **质量防线在代码层**：F4 判据（`agent/evaluate.py`，0 框降阈值重试 / 类别覆盖 / >200 框警告）三条检测路径共用；HITL 三档分流兜底
- **依赖方向**：`auto2dlabel` 为最底层被依赖方，跨包共享常量（如 `model_catalog`）只在此定义，不得反向 import

## 文档导航

- [CLAUDE.md](CLAUDE.md) — 模型选型速查 + 设计红线（每次会话必读）
- [milestone/](milestone/) — 版本里程碑定义与完成记录（v0.1–v0.6 ✅ / [v1.1](milestone/v1.1.md) ✅ RAG + Critic）
- [tests/test-v0.X.md](tests/) — 实测数据（Benchmark / GPU 复测 / 域内权重 / v1.0–v1.1 验收）
- [../docs/AutoLabel_plan.md](../docs/AutoLabel_plan.md) — 计划书（含 §Agent 工作流全景图）
- [../docs/Benchmark_plan.md](../docs/Benchmark_plan.md) — Benchmark 程序规范与质量门标准
- [../docs/datasets_plan.md](../docs/datasets_plan.md) — 数据集计划与就绪状态
- 权重与 `.env` 不入库（模板见 `configs/.env.example`）
