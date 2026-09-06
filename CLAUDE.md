# AutoLabel

Agentic 数据标注工具：自然语言指令 → LLM Agent 规划 → 多模型引擎执行 → 代码级质量评估（不达标时条件触发 LLM Evaluate）→ HITL 三档分流 → 多格式导出。范式「AI 为主、人为辅」。

## 当前进度（2026-08）

- **Auto2dLabel**：v0.1–v0.3 ✅ 完成——检测（3 引擎 × 29 模型）+ 实例分割 + 图像分类（CLIP/SigLIP + torchvision 14 款）+ OBB（YOLO-OBB 15 款）+ 语义分割（torchvision 6 款）+ cityscapes 域内 Mask R-CNN（全量 500 图 mAP 0.5149）+ Web 审核闭环 + Agentic 闭环，**共 84 个模型**；11 数据集 Benchmark（检测 5 + OBB 1 + 分割 4 + 分类 ImageNet100）
- v0.4 ✅ **3D 基石版**（Tracking + KITTI 域 2D + Agentic 收尾，里程碑见 `auto2dlabel/milestone/v0.4.md`）→ v0.5 ✅ Pose 等非 3D 内容（Pose + 指代 L2/L3 + 自动车道 ROI + ILSVRC2012 + KITTI 微调等 GPU 项 2026-08-23 收尾）；**战略：Auto2dLabel 为 Auto3dLabel 做基石，优先交付支撑 3D 的内容**
- **Auto3dLabel**：调研 ✅，MVP 路线定案（2D → 3D 提升）；**v0.4 ✅ 完成（2026-09-02）**——v0.1 五步管线单帧 MVP → v0.2 mmdet3d PointPillars/nuScenes → v0.3 模型矩阵 + 四视图复核 + LabelAny3D 决策 → **v0.4 HITL 编辑闭环（P1 cuboid 手柄编辑 + P2 nuScenes 端到端标注闭环 + P3 KITTI 微调闭环，实测 `auto3dlabel/tests/test-v0.4.md`）**，KITTI object 数据已就位（`Documents/datasets/KITTI/object`）
- 里程碑定义与完成记录见 `auto2dlabel/milestone/`；实测数据见 `auto2dlabel/tests/test-v0.X.md`

## 项目结构

- `auto2dlabel/` — 2D 自动标注主体（agent/ models/ tools/ export/ web/ benchmarks/ tests/）；子目录会话必读 `auto2dlabel/CLAUDE.md`（模型选型速查 + 设计红线）
- `auto3dlabel/` — 3D 标注主体（agent/ models→tools/ export/ benchmarks/ web/ tests/，v0.1 已完成；子目录会话必读 `auto3dlabel/CLAUDE.md`）
- `autolabel/` — v1.0 统一入口薄壳包（route_domain 路由 + CLI 分发 + Textual TUI；agent/models 逻辑不入本包）
- `docs/` — 项目文档（分工见下）
- `auto2dlabel/weights/` — 权重与 `.env` **不入库**（GitHub 私有仓库 `GithubSherlock/AutoLabel`）

**包布局与依赖方向**（三个顶层包 = 标准 monorepo，根 pyproject 一个发行版全装）：

- 依赖链单向：`autolabel → auto3dlabel → auto2dlabel`（复用骨架）。**禁止反向 import**（auto3dlabel 不得 import autolabel，会成环）
- 跨包共享常量放**被依赖方**（先例：路由引擎名单 `ENGINE3D_NAMES` 在 `auto3dlabel/configs/model_catalog.py`，autolabel/route.py 只引用不复制）
- 三个 `cli.py` 互不冲突（`autolabel.cli`/`auto2dlabel.cli`/`auto3dlabel.cli` 全限定名隔离）；入口定位：2D 聚合在 cli.py → `cli_commands.py`/`cli_run.py`/`cli_track.py`/`cli_execute.py`/`cli_common.py`，3D 单文件 cli.py，autolabel 薄壳 cli.py

## 文档分工

- `docs/AutoLabel_plan.md` — 计划书大纲：项目范围、版本路线、里程碑状态、当前状态、已知缺口、Auto3dLabel 调研
- `docs/Benchmark_plan.md` — Benchmark 程序规范（新增/修改 benchmark 脚本的依据；实测数据只写 `auto2dlabel/tests/test-v0.X.md`）
- `docs/datasets_plan.md` / `docs/DATASETS.md` — 测试数据集计划与说明
- `auto2dlabel/CLAUDE.md` — 模型选型速查 + 设计原则与红线
- `auto2dlabel/milestone/v0.X.md` — 里程碑定义与完成记录；`auto2dlabel/tests/test-v0.X.md` — 实测数据

## 常用命令

```bash
python -m pytest auto2dlabel/tests auto3dlabel/tests -q  # 全项目测试（当前 1143 passed）
mypy autolabel auto2dlabel auto3dlabel        # 全项目类型检查（当前 169，三个包都要写全——漏 autolabel 会漏检）
auto2dlabel chat "检测 000860.png 中的汽车" --no-wait   # 自然语言标注
autolabel                                     # v1.0 统一入口：无参 → TUI 对话；带指令 → 自动路由 2D/3D
python3 -m auto2dlabel.web.server             # Web 审核界面 → http://localhost:8765
bash auto2dlabel/benchmarks/run_benchmarks.sh # 一键 Benchmark（all/detection/segmentation/classification/obb）
```

## 红线

- 质量门：pyright 0 / mypy / ruff / pytest 全绿（mypy 命令必须写全三包 `mypy autolabel auto2dlabel auto3dlabel`；命令与标准见 `docs/Benchmark_plan.md` §10.6，数字基线以 `docs/AutoLabel_plan.md` 当前状态为准）
- 纯本地部署：所有模型可用开源权重；DeepSeek API 为唯一外部依赖（可换本地 Qwen，`configs/.env`）
- 测试代码在 `auto2dlabel/tests/`，不进 cli.py；`auto2dlabel/benchmarks/` 只做整体软件性能测试
- 提交：Conventional Commits；分支 feature/xxx、bugfix/xxx
