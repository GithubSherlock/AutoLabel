# Auto3dLabel

3D 自动标注主体（AutoLabel 的 3D 分支）：自然语言指令 → 双引擎（LiDAR 直检 / 2D 反投影拟合）→ 3D bbox 初稿 → 多帧跟踪 ID/运动属性 → HITL 复核 → KITTI/nuScenes 导出。「AI 为主、人为辅」，面向物理 AI / 自动驾驶。

**当前状态**：v0.1 ✅（2026-08-24，反投影拟合 MVP）、v0.2 ✅（2026-08-27，mmdet3d PointPillars + Tracker3D + nuScenes Mini）；**v0.3 📋**（2026-08-28 立项：对话式 Planner 优先 + PV-RCNN/CenterPoint 精度 + BEVFusion 融合 + Web 真 3D 复核 + LabelAny3D 验证）。里程碑与实测数据见 [milestone/](milestone/) 与 [tests/test-v0.X.md](tests/)。

## 双引擎路由

| 引擎 | 模型 | 适用 |
| --- | --- | --- |
| **LiDAR 直检**（主） | mmdet3d PointPillars（`-d pointpillars_kitti` / `pointpillars_nus`） | 有 LiDAR 数据；KITTI 全 val Car moderate **82.0**（超官方 zoo 77.6） |
| **反投影拟合**（回退） | 复用 auto2dlabel：G-DINO/SAM2 → mask 反投影 → DBSCAN → bbox 拟合 | 开放词汇 / 无 LiDAR 检测模型时；精度天花板较低（v0.1 实证） |

## 常用命令

```bash
bash install_libs.sh 3d                                  # 安装（mmcv 2.1.0 CUDA 13 编译 + mmdet3d 1.4.0，--no-deps 红线）
bash auto3dlabel/weights/download_detector3d.sh          # 3D 检测权重（openmmlab 直链）
auto3dlabel run 000123 "检测汽车和行人" -d yolo11s_kitti   # 代码级直跑（2D 反投影引擎）
auto3dlabel run 000000-000399 "检测汽车" -d pointpillars_kitti --batch   # LiDAR 直检批量
auto3dlabel run 003712-003731 "检测汽车" -d pointpillars_kitti --track3d # 多帧跟踪 ID + 速度
auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"     # LLM 闭环（v0.3 起对话确定参数）
REVIEW3D_DIR=outputs/<dir>/reviews python3 -m auto3dlabel.web.server  # Web 3D 复核
pytest auto3dlabel/tests/                                # 测试（全项目基线 723 含 3D）
```

## 文档导航

- [CLAUDE.md](CLAUDE.md) — 技术选型速查 + 设计红线（标定链命门 / yaw 约定 / 复用不复制）
- [milestone/](milestone/) — v0.1 ✅ / v0.2 ✅ / v0.3 📋（定义、验收、决策记录）
- [tests/test-v0.X.md](tests/) — 实测数据（双口径评测 / 坐标系 bug / 口径偏差对表）
- 计划书 / Benchmark 规范 — `../docs/`
- 数据（KITTI/nuScenes）与权重不入库（CC BY-NC-SA 非商用许可）
