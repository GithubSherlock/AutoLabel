# v0.1 实测数据

> v0.1 内容（单帧 KITTI 3D 标注 MVP：标定 → 2D mask 反投影 → 聚类拟合 → 导出/评测 → Agentic + Web 闭环）实测记录。功能定义与完成标记见 `../milestone/v0.1.md`。测试环境：RTX 3080 Ti 12GB，torch 2.13.0。

## 标定链投影自检（Phase 1，2026-08-23）

目标：验证「velodyne → R0_rect @ Tr_velo_to_cam → P2 → 像素」投影链与 GT 3D 框几何一致。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 全量点云 | 115384 点 → 图内 **20285（17.6%）** | 帧 000000；自检散点图 /tmp/step1_check.png |
| 投影锚点 | velodyne (8.752,-1.800,-1.546) → 像素 **(758,299)** | 真实 calib_000000.txt；已硬编码入 synth.py 作数值回归（`ANCHOR_PIXEL`） |
| GT 框双重验证 | **26/27 通过**（IoU>0.5） | 框内 LiDAR 点均在 GT 3D 框内 + 3D 角点投影 vs GT 2D 框；唯一异常 000134 Pedestrian IoU=0.49（0.5 阈值边缘） |

**结论**：标定链正确（回归测试 test_calib 10 用例 + synth 锚点双保险）。

## 反投影语义点云（Phase 2，2026-08-23）

| 项目 | 实测 |
| --- | --- |
| 检测+分割 | G-DINO（开放词汇）1 个 person 框 → SAM2 1 个 mask |
| 语义点云 | **469 点**，实例 BEV 中心 (2.1, 9.6) vs GT 行人中心 (1.84, 8.41) |
| 可视化 | 语义点云 BEV 彩图 /tmp/kitti_semantic_bev_000000.png |

**结论**：mask 反投影全链通；实例中心与 GT 吻合。mask 边缘混入约 20% 远处背景点（z>10）——真实数据上「反投影用 mask 不用 bbox」仍比 bbox 干净，且聚类+拟合门槛兜底。

## 20 帧 benchmark 冒烟（Phase 4，2026-08-24 重跑）

**⚠️ 3D 层 bug 修复与数字作废说明**：首版冒烟（2026-08-23）的 3D 层 AP 数字（如 Car easy 4.1%）作废。根因：`evaluate_per_class` 对带 `quad` 键的对象优先取 quad（8 值），而 3D 层的 `iou3d_list` 期望 7 值 `[h,w,l,x,y,z,ry]`——实际拿 8 值 quad 的前 7 个值解包，IoU 全错。修复：kitti3d_benchmark 3D 层剥 quad 键（`_without_quad_gt/_without_quad_pred`，回归测试 test_benchmark3d 9 用例覆盖注入路径）。BEV 层取 quad 是正确语义，旧数字有效。

重跑：20 帧 003712–003731（官方 val 起始），yolo11s_kitti + sam2_l.pt，prompts ['car','person']，conf 0.3。

| 项目 | 实测 |
| --- | --- |
| 耗时 | **12.3s / 20 帧（0.6s/帧）**，共 75 个 3D 框 |
| 3D AP | Car easy **0.0** (12 GT) / moderate 0.0 (23) / hard 0.0 (16)；Pedestrian easy **2.3** (13) / moderate 0.0 (1) / hard 0.0 (4)；Cyclist 0.0（prompts 未含） |
| BEV AP | Car easy 1.8 (12) / moderate **7.0** (23) / hard 0.0 (16)；Pedestrian easy **15.2** (13) / moderate 0.0 (1) / hard 0.0 (4)；Cyclist 0.0 |

**结论**：修复后 3D 层（IoU 0.5 严标准）几乎全部不达标，BEV 层有微弱信号（Pedestrian easy 15.2）。这是**反投影拟合路线精度天花板的实证**（单目+mask 拟合的框中心/尺寸/yaw 与 GT 的 3D IoU 天然难上 0.5）——与 milestone/v0.1.md 预判一致：预标注不达标时的演进方向是引入 CenterPoint/PointPillars（LiDAR 系，KITTI car moderate AP3D 74–81），而非无限调聚类参数。当前数值仍如实记录，作为 v0.2 引入 3D 检测模型的对比基线。

## chat 冒烟（Agentic 闭环，真实 DeepSeek + G-DINO + SAM2）

| 项目 | 实测 |
| --- | --- |
| 指令 | `auto3dlabel chat "标注 KITTI 帧 000000 中的汽车和行人" --det-model kitti_finetune` |
| planner | frame=000000 prompts=['car','person'] conf=0.3 det=grounding-dino-tiny seg=sam2_l.pt |
| 结果 | 1 个 Pedestrian 框（复核档）；采纳 0 / 复核 1 / 困难 0；迭代 3 次 |
| 质量报告 | total_boxes=1、missing_prompts=['car','person']、retried=False（G-DINO tiny 在该帧检出行人、漏检车——覆盖检查如实上报） |
| 产物 | outputs/kitti3d_smoke3/labels/000000.txt（15 字段：`Pedestrian … 1.57 0.86 1.01 1.77 1.42 8.35 -2.71`）+ reviews/000000_review.json |

**调试历程（两处真 bug，均已回归测试）**：
1. `Tool call failed: retry_lower_threshold 需要 detect_fn` → `_wrap_evaluate_retry` 未挂 detect_fn（回归：test_wrap_evaluate_retry_detect_fn_2d_view）
2. `AttributeError: 'dict' object has no attribute 'id'` → evaluate 重试的 detect_fn 必须返回 2D Bbox 视图（`_sync_annotations` 只吃 Bbox；`_box3d_to_bbox2d`）

## Web 复核冒烟（四端点协议 + 3D save）

| 项目 | 实测 |
| --- | --- |
| 启动 | `REVIEW3D_DIR=outputs/kitti3d_smoke/reviews python3 -m auto3dlabel.web.server`（:8766） |
| 端点 | index 200；/api/review-files 返回 1 文件（000000_review.json，count=1，含 image_path/image_stem/modified） |
| 3D save | 单测覆盖：删除/全删/编辑模式 → 3D 字段直通（fit_points 恒输出）+ KITTI label 15 字段回写 + 源文件 .reviewed 标记（test_web3d 11 用例） |

## 质量门（2026-08-24）

| 门 | 结果 |
| --- | --- |
| pytest | **723 passed**（auto2dlabel 基线 595 + auto3dlabel 新增 128 用例，基线零回归） |
| ruff | `ruff check auto3dlabel/` **0**（synth.py calib 数据行 per-file-ignores，同 planner.py 先例；auto2dlabel 基线 127 不动） |
| mypy | `mypy auto3dlabel/ --follow-imports=silent` **0**（strict） |
| pyright | `pyright auto3dlabel/` **0** |

新增 128 用例分布：test_calib 10 / test_backproject 7 / test_cluster 7 / test_fit 14 / test_geometry 27 / test_export_kitti 7 / test_pipeline_fake 17 / test_benchmark3d 9 / test_agent3d 18 / test_web3d 11（全部零真实权重：Protocol/Fake 注入 + synth 合成帧）。
