# auto3dlabel 开发指南

> Auto3dLabel = 3D 标注（Agentic 预标注层的 3D 分支）。**当前 v0.1 ✅ 已完成（2026-08-24）**——本文件 = 实现落盘后的选型速查（✅=已按此实现 / 📋=演进方向）+ 设计红线。实测数据见 `tests/test-v0.1.md`；里程碑定义与验收见 `milestone/v0.1.md`；调研与路线论证见 `docs/AutoLabel_plan.md` Auto3dLabel 部分。

## v0.1 实现速查（2026-08-24 落盘）

```
auto3dlabel/
├── cli.py                     # typer：run（代码级直跑）/ chat（LLM agent 闭环）；--batch --resume 失败隔离
├── configs/kitti.py           # 路径常量、COCO→KITTI 类映射、ADAPTIVE_EPS/MIN_FIT_POINTS 单一事实源
├── schema/calib.py            # KittiCalib：P2/R0_rect/Tr_velo_to_cam 投影链 + velo_to_cam
├── schema/box3d.py            # Box3D（cx/cy/cz/h/w/l/yaw_bev/fit_points + rotation_y property）+ KittiFrame/FrameResult
├── data/kitti.py              # load_velodyne/calib/label3d + resolve_frame + frame_ids 工具
├── tools/backproject.py       # polygon rasterize → label 图 → 投影点一次查表（115k 点毫秒级）
├── tools/cluster.py           # sklearn DBSCAN（BEV xz）+ 类别 eps + 0.1m 体素降采样 + 粘连标记
├── tools/fit.py               # minAreaRect 角点法 yaw + z 分位高度 + Box3D 组装 + fit_points 门槛
├── tools/geometry.py          # shapely BEV/3D IoU + yaw_bev↔rotation_y **唯一转换点**（rotation_y = yaw−π/2）
├── tools/visualize.py         # 投影自检散点图 / 语义点云 BEV 彩图 / 3D bbox vs GT 对比图（cv2）
├── tools/log.py|device.py|evaluate.py|export.py  # 2D re-export 薄层（复用不复制）+ 3D ExportTool（CLI 直调，不进 LLM registry）
├── tools/pipeline.py          # annotate_frame：detect→SAM2→反投影→聚类→拟合（入口 disable_tf32）
├── export/kitti_label.py      # line_from_box3d 纯函数（15 字段，y 底部中心回写）+ build 全量
├── export/review_queue.py     # triage_3d（conf 三档 + fit_points<20/review_flag 强制 review）+ *_review.json
├── benchmarks/load_gt3d.py    # label_2 parts[8:15] → 7 值 [h,w,l,x,y,z,ry]
├── benchmarks/kitti3d_benchmark.py  # 双层：BEV "quad" 8 值 / 3D "bbox" 7 值，iou_fn 注入 evaluate_per_class
├── agent/tools3d.py           # Detect3DTool(name="detect_objects")/Visualize3DTool + build_3d_registry 独立实例
├── agent/orchestrator3d.py    # 薄壳复用 AgentOrchestrator 循环 + fit_points 判据 + _wrap_evaluate_retry
├── agent/planner3d.py         # 3D planner（task_type="kitti_3d"，JSON 三级解析兜底）
└── web/server.py + static/    # 四端点协议照抄 + 3D save（不走 _bbox_from_dict，保 3D 字段 + 导出 label）
```

**常用命令**：

```bash
auto3dlabel run 000123 "检测汽车和行人" -d yolo11s_kitti            # 代码级直跑（零下载）
auto3dlabel run 000000-000399 "检测汽车和行人" -d pointpillars_kitti --batch  # LiDAR 批量：一次 forward 整批，自动实测批大小跑满 GPU（4090 实测 21764/24564 MiB）
auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"               # LLM 闭环（DeepSeek）
REVIEW3D_DIR=outputs/<dir>/reviews python3 -m auto3dlabel.web.server # Web 复核（:8765 同 2D）
```

**质量门（v0.1 红线，四件套对 auto3dlabel 归零；auto2dlabel 基线 595/0/167/127 不动）**：

```bash
python -m pytest -q                                        # 全项目（723 = 595 基线 + 128 新）
ruff check auto3dlabel/                                    # 0（配置统一在根 pyproject.toml：E741 豁免 l 在 per-file-ignores）
mypy auto3dlabel/ --follow-imports=silent                  # 0（strict；--follow-imports=silent 必须，否则基线依赖噪声）
pyright auto3dlabel/                                       # 0
```

**关键 bug 记录（有回归测试）**：① `evaluate_per_class` 有 quad 优先取 quad → 3D 层必须剥 quad 键喂 7 值（`_without_quad_*`）；② evaluate 重试 detect_fn 必须返回 2D Bbox 视图（`_sync_annotations` 只吃 Bbox）；③ Pedestrian→person 需 `attach_coco_names`（KITTI 名不含 COCO 子串误报缺类）。

## 技术选型速查

| 环节 | 选型 | 说明 |
| --- | --- | --- |
| 2D 基础模型 | **复用 auto2dlabel**（G-DINO + SAM2/SAM3 + YOLO）✅ | 零新增模型；直接 `from auto2dlabel.models import ...` |
| 点云加载 | numpy 直接解析（KITTI `.bin`）✅；nuScenes 延后 v0.2 | **不引入** mmdet3d / OpenPCDet（MVP 阶段） |
| 聚类 | DBSCAN（**sklearn 1.7.2**，唯一新增依赖）✅ | eps 按类别（Car 1.0/Ped 0.5/Cyc 0.7，configs 单一事实源）；>2 万点先体素降采样 |
| bbox 拟合 | 鸟瞰 `cv2.minAreaRect` **角点法 yaw**（boxPoints→最长边方向，不用 angle 字段）+ z 分位高度 ✅ | yaw 唯一转换点 `geometry.py`：`rotation_y = yaw_bev − π/2` |
| 可视化 | **cv2 渲染** ✅（投影自检散点图 / BEV 彩图 / 3D vs GT 对比图） | 不装 open3d |
| 评测 | 3D IoU / BEV IoU 自写 ✅ | 双层 benchmark（BEV quad 8 值 / 3D bbox 7 值）+ difficulty 分层；按 `docs/Benchmark_plan.md` 程序规范 |

## CVAT 参考与演进路径（2026-08-23 调研落盘）

- **CVAT 3D 现状**：仅 PCD 点云 + cuboid 手动标注；3D 格式只有 KITTI raw（tracklet_labels.xml）/ Sly Point Cloud（**无 nuScenes**）；**无任何 3D AI 自动标注**——「3D 自动预标注」正是 Auto3dLabel 的差异化空间
- **可借鉴**：四视图交互（透视 + Top/Side/Front 正交，正交投影只显选中对象 + 自动聚焦）、raycast 模板吸附双击落定、**关联图像**（点云帧绑定相机图同屏，RelatedFile 机制）、backend `SourceType.AUTO` + `score` 预标注数据模型、track 角度最短弧插值
- **不借鉴**：16 值冗余数组（pos+rot+scale+7×0）、Z-up 三欧拉角内部存储（与 KITTI `rotation_y` / nuScenes 四元数均不兼容）、大点云无下采样全量渲染（CVAT 已知短板）
- **内部数据模型定案**：结构化 `{position, size, yaw}`（BEV 角度，红线同）；导出时显式转 KITTI 相机系 `rotation_y` / nuScenes 全局系四元数
- **3D 检测模型演进路径**（预标注不达标时引入，MVP 不动）：
  - LiDAR 系是唯一可达标注业务精度的路线：KITTI car moderate AP3D **74–81**（SECOND 78.2 / PointPillars 77.6 / PV-RCNN 81.4，开源权重齐全），nuScenes VoxelNeXt 64.5 mAP；12GB 可跑、秒级单帧
  - **第一候选 CenterPoint / PointPillars**——KITTI+nuScenes 双支持、OpenPCDet+MMDet3D 双框架权重；**用 MMDet3D 完整训练权重（val 77.6–78.2），避开 OpenPCDet 的 underfit demo 权重**；升级项 VoxelNeXt（全稀疏 64ms/帧）
  - 融合系 BEVFusion（MIT，nuScenes NDS 76.1）留 v2+（需相机-LiDAR 标定对齐）；单目系（moderate ≤30%）只作交叉验证兜底
  - **无 LiDAR 域标注（LabelAny3D 调研，2026-08-24）**：NeurIPS 2025 的 LabelAny3D（arXiv:2601.01676）走分析合成路线——相对深度（比 metric depth 稳）+ Objaverse 形状先验 / TRELLIS 式重建（amodal 补全遮挡，恰是 v0.1 反投影拟合 3D AP 近零的短板）+ SAM/G-DINO；面向 COCO 日常物体（无 GT 可对的域），价值主张是「伪标签训练下游单目检测器优于旧伪标签法」（**间接指标，无直接标注误差数字**）。定位：自动驾驶域（有 LiDAR）不需要它——已有真值域用 LiDAR 系 74–81，引入单目合成是负优化；作为 **v2+「行车记录仪/网络视频等无 LiDAR 数据源」3D 预标注候选**（CVAT 也无此能力，差异化空间）。验证门槛：KITTI 20 帧 val 小规模复现 vs GT 3D IoU（预期显著好于我们的 0%，moderate 大概率仍 <30%，约 1 天成本）
  - 无 ultralytics 式 3D 全家桶，事实标准仍是 OpenPCDet + MMDetection3D（spconv 编译是主要坑，Pillar 系可免）
- **合规**：KITTI（CC BY-NC-SA 3.0）/ nuScenes（CC BY-NC-SA 4.0）数据**非商用**，商业化需自采/商用许可数据；单目模型 repo 许可证多未标注，BEVFusion/SparseDrive（MIT）最干净

## auto2dlabel 可复用清单（2026-08-23 探索落盘）

### 直接 import（零/弱 2D 耦合）

- **Agentic 骨架**：`agent/llm.py`（LLMClient）、`agent/state.py`（AgentState 快照续跑）、`agent/evaluate.py`（evaluate_detections 只取 `.label`，零耦合）、`agent/batch_strategy.py`（model 参数 duck-typed detect）、`tools/registry.py`（Tool/registry，3D 新 Tool 注册即可被调用）、`tools/confirm.py`、`tools/device.py`（disable_tf32 对 3D 同样必须）、`tools/log.py`、`tools/prompts.py`
- **2D 基础模型**：`create_detection_model`（G-DINO 开放词汇，反投影锚点）、`create_segmentation_model`（SAM2 mask——红线「反投影用 mask 不用 bbox」）、CLIP `ClipCropScorer`（属性打分）
- **HITL/过滤**：`tools/hitl.py`（triage_annotations 三档 + `*_review.json` 队列文件格式**原样复用**）、`tools/sampling.py`（主动学习采样）、`tools/constraints.py`（parse_roi/_point_in_polygon——**BEV 平面 ROI 直接复用**）
- **Benchmark 基建**：`benchmarks/common.py` `evaluate_per_class(..., iou_fn=)` iou_fn 可注入——3D 自写 bev_iou/3D IoU 纯函数注入；`benchmarks/kitti_benchmark.py` GT 解析器（已含 3D 字段 h w l x y z rot_y）与 `kitti_difficulty` 判定直接参考

### 仿照模板

- **模型封装母版 = `models/lane.py`**（v0.5 最新模板）：XxxResult dataclass → Protocol（测试注入 Fake）→ 具体类（零加载 `__init__` + 幂等 `_load` + ImportError 守卫）→ create_xxx_model 工厂（名字路由 + ValueError 报清单）→ 批量接口 hasattr 分派
- **Bbox 母版 = `schema/annotation.py`**：可选字段带默认值（angle/keypoints 后加不破坏兼容）、to_dict 非空才输出、**穿透点纪律**（yaw/拟合点数从第一天按穿透点清单管理）
- **Web 母版 = `web/server.py` + `web/static/index.html`**：单文件无构建链；**复核队列四端点（review-files/file/image/save）协议原样复用**——队列文件加 `pcd_path`/`calib`/`3d_annotations` 字段即可；拖拽/双击改标签/删除交互状态机照抄；安全模式（文件名后缀白名单 + 路径遍历拒绝 + 损坏 try/except 跳过）必须继承
- **catalog 单一事实源**：`model_catalog.py` 分组常量 + `format_catalog_summary()` 注入 planner prompt
- **导出模式**：`export/coco.py` build-dict 纯函数（Web 与文件共用单一事实源）+ `tools/export.py` exporters 注册表（3D 新增 kitti/nuscenes 模块）
- **测试体系**：Protocol + Fake 注入零真实权重 + tests/functional/ + test-v0.X.md 实测记录 + 批量 parity 双接口断言（Fake detect/detect_batch 可区分记录）

### 不 import（2D 耦合深，仿骨架重写）

- `cli_execute.py`（46KB 最大单文件）、`agent/planner.py` prompt 内容（JSON 三级解析兜底照抄）、`agent/orchestrator.py` 硬编码工具名（小改参数化或 3D 工具同名）、tracking 系（v0.1 先单帧不碰）、2D 专属导出格式（yolo/voc/labelme/mot）

### 开工顺序（v0.1 已完成，供 v0.2 参照）

✅ 已落地：`from auto2dlabel import` 搭 agent 骨架 + Web 复核队列协议 → 3D 管线（calib/backproject/cluster/fit/geometry/pipeline）→ KITTI 3D benchmark（照 dota_obb_benchmark 结构）→ Agentic/Web 闭环（见上方实现速查）。v0.2 候选：nuScenes 支持、SAM2 视频传播多帧增强（首帧 prompt → 跨帧 mask 传播，权重已在 84 模型目录）、CenterPoint/PointPillars 引入（3D 检测模型演进路线，见下）、LabelAny3D 无 LiDAR 域预标注复现验证（见下）。

## 设计红线

- **标定链是命门**：KITTI P2/R0_rect/Tr_velo_to_cam、nuScenes ego_pose/calibrated_sensor 任何一环错 → 投影全错且难察觉；必须做「点云投影回图像」自检图
- **反投影用 mask 不用 bbox**：bbox 边缘把背景点投进语义点云；SAM mask 才是像素级边界
- **LiDAR 观测性**：只反投影 LiDAR 实际打到的点；遮挡/远距目标点稀疏 → 拟合退化，输出「拟合点数」置信度供 HITL 分流（宁缺勿假）
- **yaw 约定**：内部统一 BEV 平面（KITTI 地面 xz）角度 `yaw_bev`（车头相对 +z 向 +x 为正）；导出时经唯一转换点（`tools/geometry.py`）显式转 KITTI 相机系 `rotation_y`（`= yaw_bev − π/2`），nuScenes 全局系 yaw 到 v0.2 再定，不得混用
- **复用不复制**：Agentic 编排（planner/orchestrator）、HITL 三档、质量评估模式复用 auto2dlabel 骨架；3D 新增代码只放 `auto3dlabel/`
- **依赖护栏（2026-08-28）**：auto2dlabel 为 auto3dlabel 硬依赖（复用 agent/模型/工具骨架）；`bash install_libs.sh 3d` 连带安装 2D 依赖；包本体装法 = 仓库根 `pip install -e .`（根 pyproject 同一发行包装 auto2dlabel + auto3dlabel）；cli run/chat 入口 `_require_auto2dlabel()` 缺包时给安装指引而非裸 ImportError
- **纯本地**：延续 2D 红线——无外部 API；数据（KITTI / nuScenes mini）与权重不入库
- **先单帧后时序**：v0.1 只做单帧；跟踪 ID + 运动属性等 auto2dlabel v1.0 Tracking 交付后共用（ByteTrack/BoT-SORT）
