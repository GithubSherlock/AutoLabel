# Auto3dLabel

3D 自动标注主体（AutoLabel 的 3D 分支）：自然语言指令 → 三系引擎（LiDAR 直检 / 单目 / 相机-LiDAR 融合）+ 2D 反投影拟合回退 → 3D bbox 初稿 → 多帧跟踪 ID/运动属性 → HITL 复核（含 cuboid 手柄编辑）→ KITTI/nuScenes 导出。「AI 为主、人为辅」，面向物理 AI / 自动驾驶。

**当前状态**：v0.1 ✅（2026-08-24，反投影拟合 MVP）、v0.2 ✅（2026-08-27，mmdet3d PointPillars + Tracker3D + nuScenes Mini）、**v0.3 ✅（2026-08-31，对话式 Planner + 精度/融合 + Web 真 3D + LabelAny3D 验证）**、**v0.4 ✅（2026-09-02，HITL 编辑闭环 + nuScenes 端到端 + KITTI 微调闭环 + 地图矢量对账）**。里程碑与实测数据见 [milestone/](milestone/) 与 [tests/test-v0.X.md](tests/)。

> 统一入口 `autolabel`（带指令自动路由，3D 引擎名/点云关键词/6 位帧号 → 3D）见仓库根 [README.md](../README.md)。

## 引擎矩阵（三系 9 引擎 + 反投影回退）

| 系 | 引擎 | 数据集 | 说明 |
| --- | --- | --- | --- |
| **LiDAR 直检** | `pointpillars_kitti` / `pvrcnn_kitti` | KITTI | 快速扫 / 精度档；全 val Car moderate **82.0**（超官方 zoo 77.6） |
| **LiDAR 直检** | `pointpillars_nus` / `centerpoint_nus` / `free_anchor_nus` | nuScenes | 快速 / 精度（mAP 23.4）/ 速度档（全 Mini 最快） |
| **单目** | `pgd_kitti` / `fcos3d_nus` | KITTI / nuScenes | 无 LiDAR 降级交叉验证基准 |
| **相机-LiDAR 融合** | `bevfusion_nus` / `bevfusion_lidar_nus` | nuScenes | 融合档（mAP 27.0）；`_lidar` 变体为降级备选 |
| **反投影拟合**（回退） | 复用 auto2dlabel：G-DINO/SAM2 → mask 反投影 → DBSCAN → bbox 拟合 | KITTI / nuScenes | 开放词汇 / 无 3D 权重时；精度天花板较低（v0.1 实证） |

## 常用命令

```bash
# 安装（mmcv 2.1.0 CUDA 13 编译 + mmdet3d 1.4.0，--no-deps 红线）
bash install_libs.sh 3d
bash auto3dlabel/weights/download_detector3d.sh              # 3D 检测权重（openmmlab 直链）

# 单帧 / 批量 / 跟踪（代码级直跑，零 LLM）
auto3dlabel run 000123 "检测汽车和行人" -d yolo11s_kitti              # 2D 反投影引擎
auto3dlabel run 000000-000399 "检测汽车" -d pointpillars_kitti --batch  # LiDAR 直检批量（一次 forward 整批）
auto3dlabel run 003712-003731 "检测汽车" -d pointpillars_kitti --track3d # 多帧跟踪 ID + 速度

# LLM 闭环（v0.3 起缺参对话确定；v1.0 P1+ 支持 nuScenes 批量直跑）
auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"

# nuScenes 端到端（队列 → Web 复核 → devkit 导出 → 回灌评测）
auto3dlabel nuscenes-queue -d bevfusion_nus --limit 20

# 地图矢量对账（MapTR 契约 mapvec_pred/1 → Chamfer 比对 → 报告 + 复核队列）
auto3dlabel mapvec-report --pred-dir <契约目录> --img-root <图像根>

# 训练微调闭环（v0.4 P3：五步管线 + 同口径对比）
python3 -m auto3dlabel.tools.train3d --dry-run               # 数据准备 + config 生成（CPU）
python3 -m auto3dlabel.tools.train3d                         # + 微调训练（需 GPU）
python3 -m auto3dlabel.benchmarks.smoke_kitti 003712-003731 pointpillars_kitti 0.3 [config] [checkpoint]

# Web 复核（three.js 四视图 + cuboid 六面手柄编辑）
REVIEW3D_DIR=outputs/kitti3d/reviews python3 -m auto3dlabel.web.server   # → :8766

# 测试与质量门
pytest auto3dlabel/tests/                                    # 全项目 1354 passed, 4 skipped
ruff check auto3dlabel/ && mypy auto3dlabel/ --follow-imports=silent && pyright auto3dlabel/  # 归零
cd auto3dlabel/tests/helpers && node smoke_web3d.js           # 前端冒烟（94 断言）
```

## 端到端闭环（v0.4）

```txt
引擎推理 → 复核队列（triage_3d 三档）→ Web 复核（点云 + 6 相机 + 四视图 + 手柄编辑）
  → devkit / KITTI label 导出 → 回灌评测（mAP）→ 微调训练 → 域内权重回灌引擎
```

实测（v0.4 P2）：bevfusion 队列 81 样本 / 2696 框 → Web 批量复核 2185 保留 / 464 删除 → 回灌出表 mAP 24.6。KITTI 微调（P3）：40-point 主口径微调全面优于或持平官方（Car hard 76.4→80.5）。

## 文档导航

- [CLAUDE.md](CLAUDE.md) — 技术选型速查 + 设计红线（标定链命门 / yaw 唯一转换点 / 复用不复制）
- [milestone/](milestone/) — v0.1–v0.4 定义、验收与决策记录（v0.4 含 P4 地图矢量对账回填段）
- [tests/test-v0.X.md](tests/) — 实测数据（双口径评测 / 坐标系 bug / 口径偏差对表 / 微调对比）
- [../docs/AutoLabel_plan.md](../docs/AutoLabel_plan.md) — 计划书（含 3D 技术路线调研与 §Agent 工作流全景图）
- [../docs/Benchmark_plan.md](../docs/Benchmark_plan.md) — Benchmark 程序规范
- 数据（KITTI/nuScenes）与权重不入库（**CC BY-NC-SA 非商用许可**）；密钥模板见 `configs/.env.example`
