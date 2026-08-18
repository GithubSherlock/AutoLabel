# AutoLabel

Agentic 数据标注工具：自然语言指令 → LLM Agent 规划 → 多模型引擎执行 → 代码级质量评估（不达标时条件触发 LLM Evaluate）→ HITL 三档分流 → 多格式导出。范式「AI 为主、人为辅」。

## 当前进度（2026-08）

- **Auto2dLabel**：v0.1–v0.3 ✅ 完成——检测（3 引擎 × 29 模型）+ 实例分割 + 图像分类（CLIP/SigLIP + torchvision 14 款）+ OBB（YOLO-OBB 15 款）+ 语义分割（torchvision 6 款）+ cityscapes 域内 Mask R-CNN（全量 500 图 mAP 0.5149）+ Web 审核闭环 + Agentic 闭环，**共 84 个模型**；11 数据集 Benchmark（检测 5 + OBB 1 + 分割 4 + 分类 ImageNet100）
- v0.4 📋 **3D 基石版**（Tracking + KITTI 域 2D + Agentic 收尾，里程碑见 `auto2dlabel/milestone/v0.4.md`）→ v0.5 📋 Pose 等非 3D 内容；**战略：Auto2dLabel 为 Auto3dLabel 做基石，优先交付支撑 3D 的内容**
- **Auto3dLabel**：调研 ✅，MVP 路线定案（2D → 3D 提升）；**主推进**——v0.1 单帧 MVP 已立项（见 `auto3dlabel/milestone/v0.1.md`，选型速查见 `auto3dlabel/CLAUDE.md`），KITTI object 数据已就位（`Documents/datasets/KITTI/object`，从 `/autodl-pub/data/KITTI/object` 补齐），与 Auto2dLabel v0.4 双线并行
- 里程碑定义与完成记录见 `auto2dlabel/milestone/`；实测数据见 `auto2dlabel/tests/test-v0.X.md`

## 项目结构

- `auto2dlabel/` — 2D 自动标注主体（agent/ models/ tools/ export/ web/ benchmarks/ tests/）；子目录会话必读 `auto2dlabel/CLAUDE.md`（模型选型速查 + 设计红线）
- `auto3dlabel/` — 3D 标注（预留，仅 README + requirements）
- `docs/` — 项目文档（分工见下）
- `auto2dlabel/weights/` — 权重与 `.env` **不入库**（GitHub 私有仓库 `GithubSherlock/AutoLabel`）

## 文档分工

- `docs/AutoLabel_plan.md` — 计划书大纲：项目范围、版本路线、里程碑状态、当前状态、已知缺口、Auto3dLabel 调研
- `docs/Benchmark_plan.md` — Benchmark 程序规范（新增/修改 benchmark 脚本的依据；实测数据只写 `auto2dlabel/tests/test-v0.X.md`）
- `docs/datasets_plan.md` / `docs/DATASETS.md` — 测试数据集计划与说明
- `auto2dlabel/CLAUDE.md` — 模型选型速查 + 设计原则与红线
- `auto2dlabel/milestone/v0.X.md` — 里程碑定义与完成记录；`auto2dlabel/tests/test-v0.X.md` — 实测数据

## 常用命令

```bash
pytest auto2dlabel/tests/                     # 测试在包内（functional/ + helpers/），当前 212 passed
auto2dlabel chat "检测 000860.png 中的汽车" --no-wait   # 自然语言标注
python3 -m auto2dlabel.web.server             # Web 审核界面 → http://localhost:8765
bash auto2dlabel/benchmarks/run_benchmarks.sh # 一键 Benchmark（all/detection/segmentation/classification/obb）
```

## 红线

- 质量门：pyright 0 / mypy / ruff / pytest 全绿（命令与标准见 `docs/Benchmark_plan.md` §10.6，数字基线以 `docs/AutoLabel_plan.md` 当前状态为准）
- 纯本地部署：所有模型可用开源权重；DeepSeek API 为唯一外部依赖（可换本地 Qwen，`configs/.env`）
- 测试代码在 `auto2dlabel/tests/`，不进 cli.py；`auto2dlabel/benchmarks/` 只做整体软件性能测试
- 提交：Conventional Commits；分支 feature/xxx、bugfix/xxx
