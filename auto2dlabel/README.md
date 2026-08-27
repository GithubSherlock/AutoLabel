# Auto2dLabel

2D 自动标注主体（AutoLabel 的 2D 分支）：自然语言指令 → LLM Agent 规划 → 多模型引擎执行 → 代码级质量评估（不达标条件触发 LLM Evaluate）→ HITL 三档分流 → 多格式导出。「AI 为主、人为辅」。

**当前状态**：v0.1–v0.5 ✅ 完成（84 模型 / 12 数据集 Benchmark）；**v0.6 📋**（对话式 Agent 统一入口，2026-08-28 立项）。里程碑与实测数据见 [milestone/](milestone/) 与 [tests/test-v0.X.md](tests/)。

## 能力一览

| 任务 | 模型 | 说明 |
| --- | --- | --- |
| 检测 | G-DINO（开放词汇）+ Ultralytics YOLO/RT-DETR + torchvision | 29 模型 |
| 分割 | SAM/SAM2/SAM2.1/SAM3 + FastSAM + Mask R-CNN（含 cityscapes 域内）+ 语义分割 | 24 模型，默认 `sam2_l.pt` |
| 分类 | CLIP/SigLIP + torchvision ImageNet1K | 16 模型 |
| OBB | YOLO-OBB 11/12/26 n/s/m/l/x | 15 模型，`Bbox.angle` 弧度约定 |
| Tracking | ByteTrack（默认）/ BoT-SORT（精度档）+ 指代 L1/L2/L3 | 视频/帧目录 → MOT 导出 |
| Pose | YOLO-pose | COCO keypoints 导出 |

## 常用命令

```bash
auto2dlabel run img.jpg "检测汽车和行人" -d yolo26x.pt -t 0.3   # Agent Loop（LLM 在环处置）
auto2dlabel chat "检测 /data/images 中的汽车" --no-wait           # Planner 解析 → 代码级执行
auto2dlabel run video.mp4 "检测行人" --track                      # 跟踪模式（ByteTrack）
auto2dlabel chat "分类为猫和狗，用 clip" --no-wait                # 分类
auto2dlabel sample --top-k 5                                      # 主动学习采样
python3 -m auto2dlabel.web.server                                 # Web 审核 → http://localhost:8765
pytest auto2dlabel/tests/                                         # 测试（质量门基线 595 passed）
```

## 文档导航

- [CLAUDE.md](CLAUDE.md) — 模型选型速查 + 设计红线（每次会话必读）
- [milestone/](milestone/) — 版本里程碑定义与完成记录（v0.1–v0.5 ✅ / v0.6 📋）
- [tests/test-v0.X.md](tests/) — 实测数据（Benchmark / GPU 复测 / 域内权重）
- 计划书 / Benchmark 规范 / 数据集计划 — `../docs/`
- 权重与 `.env` 不入库（GitHub 私有仓库 `GithubSherlock/AutoLabel`）
