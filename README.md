<p align="center">
  <h1 align="center">AutoLabel</h1>
  <p align="center"><strong>Agentic 2D/3D Data Annotation — AI-first, Human-in-the-loop</strong></p>
  <p align="center">自然语言驱动 · LLM Agent 编排 · 多模型引擎 · 代码级质量评估 · 多格式导出</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/models-105-orange" alt="105 Models">
  <img src="https://img.shields.io/badge/version-1.0.0-informational" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/2D-105%20models-blueviolet" alt="2D 105 models">
  <img src="https://img.shields.io/badge/3D-9%20engines-9cf" alt="3D 9 engines">
</p>

---

## 目录

- [什么是 AutoLabel](#什么是-autolabel)
- [特性](#特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [用法指南](#用法指南)
- [模型目录](#模型目录)
- [Benchmark 结果](#benchmark-结果)
- [导出格式](#导出格式)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [已知限制](#已知限制)
- [开发与贡献](#开发与贡献)
- [许可证](#许可证)

---

## 什么是 AutoLabel

**AutoLabel** 是一个 Agentic 图像 / 点云自动标注工具，将传统标注工具的范式从「人为主、AI 为辅」反转为 **「AI 为主、人为辅」**。

传统标注软件（CVAT、Label Studio）以人工画布 + 快捷键为核心，AI 预标注仅作辅助插件存在。AutoLabel 重新定义工作流：**用户下达自然语言指令，由 LLM Agent 自主规划、调用检测/分割/3D 模型、代码级评估结果，并在不确定时主动请求人工介入**。

```txt
自然语言指令 → Planner 任务规划 → 多模型引擎执行（检测/分割/分类/OBB/3D）
  → 代码级质量评估（不达标时条件触发 LLM Evaluate）→ HITL 三档分流（auto/review/hard）
  → 多格式导出（COCO/YOLO/VOC/LabelMe/cls/DOTA/MOT/KITTI/nuScenes）
```

> 完整的 L1–L7 分层工作流图（含 LLM 调用点 ①②③ 与数据反哺回路）见
> [`docs/AutoLabel_plan.md` §Agent 工作流全景图](docs/AutoLabel_plan.md)。

### 目标用户

- **AI 算法工程师 / 数据科学家**：为训练或微调模型快速生成高质量预标注，再在 Label Studio / CVAT 中精修
- **使用场景**：单人或小团队在消费级硬件（Apple Silicon / 单 GPU）上对百到千张级图像、KITTI/nuScenes 级点云进行自动预标注
- **不是面向**：专业标注团队的大规模生产管线（需 Web 协作、权限管理、SaaS），也不面向零 AI 背景的标注员

### 子项目

| 子项目 | 状态 | 说明 |
| --- | --- | --- |
| **auto2dlabel** | ✅ v0.1–v0.6 + v1.0（P1–P5）+ v1.1（P1–P2） | 2D 检测/分割/分类/OBB/Tracking/Pose + VLM 指代 + TUI 对话入口 + RAG 经验库 + 质检 Agent，**105 模型**（v0.3 口径 84 + Pose 16 + mmdet 5） |
| **auto3dlabel** | ✅ v0.1–v0.4 | 3D 标注：LiDAR 直检 / 单目 / 融合三系 **9 引擎** + 反投影拟合回退 + 多帧跟踪 + nuScenes 端到端闭环 + KITTI 微调闭环 |
| **autolabel** | ✅ v1.0 | 统一入口薄壳：`autolabel` 无参 → Textual TUI；带指令 → `route_domain` 自动路由 2D/3D |

---

## 特性

### 交互与编排

- **🗣️ 统一入口 + 对话式参数确定** — `autolabel` 无参进 Textual TUI（斜杠命令 `/help` `/model` `/cost` `/new` `/resume` `/cancel` `/review` `/web` `/quit`）；带指令则 `route_domain` 代码级路由 2D/3D。`chat` 缺参时由 LLM 多轮对话问出（`questions[]` → 渲染 → 回填 ≤3 轮），非法响应/无 key 逐级降级至单轮解析 + 代码兜底
- **🧠 Agentic 编排** — LLM 规划多步任务（检测 → 分割 → 导出），token 高效设计（结果摘要注入）；**F4 代码级质量评估**（0 框重试/类别覆盖/超框警告）在 tool 层完成，仅 `quality.ok == False` 时条件暴露 LLM Evaluate（accept / flag / retry / 换模型），失败降级确定性规则
- **📚 RAG 标注经验库（v1.1）** — 标注会话摘要零 LLM 入库（`logs/experience.jsonl`），planner 解析时 metadata 过滤 + CLIP 文本向量 top-k 检索，注入 few-shot；检索失败静默降级
- **🔍 质检 Agent（Critic，v1.1）** — 质量不达标时由独立 provider 出意见（交叉校验，避免同模型自我确认偏差），只出意见不执行重试；`cost-critic` 对比质检/规划成本
- **💸 LLM Harness** — 统一出口落 usage 台账（`logs/llm_usage.jsonl`：调用点/tokens/缓存命中/折算费用），`cost-report` 按调用点聚合；max_tokens 分级 + json mode + 前缀缓存（实测命中率 97.7%）+ 流式输出

### 2D 能力（auto2dlabel）

- **🔧 4 引擎 × 105 模型** — Grounding DINO（开放词汇）、Ultralytics YOLO/RT-DETR、PyTorch Vision、mmdet（RTMDet）；分割 SAM/SAM2/SAM3/Mask R-CNN（含 cityscapes 域内权重）/FastSAM/Mask2Former/torchvision 语义分割；分类 CLIP/SigLIP + torchvision ImageNet1K 14 款；OBB YOLO-OBB 15 款；Pose YOLO-pose 15 款 + RTMPose
- **🎥 Tracking** — 视频/帧目录逐帧检测 + ByteTrack（默认）/ BoT-SORT（精度档：ReID 外观关联 + ECC 相机运动补偿），MOT 导出 + 轨迹可视化 + 标注成片
- **🎯 指代约束 L1/L2/L3** — L1 属性（CLIP 逐框零样本）+ 方位（坐标分位）；L2 Florence-2；L3 Qwen2-VL-7B 4bit（GPU），L2 失败自动升级阶梯；另有 ROI 空间约束与自动车道检测
- **🔁 Batch 韧性 + 动态调优** — 单图失败隔离 + `--resume` 断点续跑（AgentState 快照）；批量推理动态实测最大 batch 用满 GPU；**parity 双铁律**（`rect=False` + 关闭 TF32）保证批量结果与逐图逐位一致
- **🔍 自动类别推荐** — 扫描全图 COCO 80 类，按检出数量与置信度排序推荐 Top-K
- **⚡ `--no-llm` Baseline** — 绕过 LLM，内置关键词映射直调模型（对照实验 / 离线场景）

### 3D 能力（auto3dlabel）

- **🧊 三系 9 引擎** — LiDAR 直检（PointPillars / PV-RCNN / CenterPoint / FreeAnchor）、单目（PGD / FCOS3D）、相机-LiDAR 融合（BEVFusion）；KITTI 全 val Car moderate **82.0**（PointPillars）/ PV-RCNN 精度档
- **🔄 反投影拟合回退** — 无 LiDAR 检测模型时复用 2D 能力：G-DINO/SAM2 → mask 反投影 → DBSCAN 聚类 → bbox 拟合（开放词汇，精度天花板较低，v0.1 实证）
- **🛰️ nuScenes 端到端闭环** — 队列生成 → Web 复核（点云 + 6 相机）→ devkit 标签导出 → 回灌评测（v0.4 P2，81 样本 / 2696 框实测）
- **🗺️ 地图矢量对账** — 消费 MapTR 逐帧契约 `mapvec_pred/1`（复制自 AutoDriveData，依赖单向红线）：Chamfer 匹配 TP/FP/FN + CD 分布 + 越窗计数 → CAM_FRONT overlay + BEV 面板 + 对账报告 + 复核队列
- **🔧 训练微调闭环** — mmdet3d 五步微调管线（`train3d`）+ 同口径对比评测，40-point 主口径微调全面优于或持平官方（Car hard 76.4→80.5）

### 质量闭环（共性）

- **✅ HITL 三档分流** — 置信度三档自动分流 + Web 复核队列 + 主动学习采样；低置信与质量不合格项强制入复核
- **🖼️ Web 复核界面** — FastAPI + Canvas SPA（CVAT 借鉴：快捷键/undo/手柄/列表/过滤/右键/区域 issue）；2D `:8765`，3D `:8766`（three.js 四视图 + cuboid 六面手柄编辑）
- **📝 结构化日志** — 每次调用的完整 JSON 日志（LLM / Chat / PythonAPI），可复现、可审计

---

## 系统架构

**依赖方向单向**：`autolabel → auto3dlabel → auto2dlabel`（禁止反向 import，跨包共享常量放被依赖方）。

```txt
autolabel/        # 统一入口薄壳：route_domain 路由 + CLI 分发 + Textual TUI
    ↓
auto3dlabel/      # 3D 标注主体（复用 2D 的 agent/模型/工具骨架）
    ↓
auto2dlabel/      # 2D 标注主体（Agentic 编排 + 模型层 + Web 复核 + Benchmark）
```

**LLM 决策点仅 3 处**且条件触发——① 对话式规划 ② 质量处置 ③ 批次调参。质量防线在 F4 代码层 + HITL，**不引入多 agent 编排框架**（论证见 `auto2dlabel/milestone/v0.6.md`）。

---

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA GPU（推荐）或 Apple Silicon MPS 或 CPU（3D LiDAR 直检需 GPU，反投影引擎可 CPU）
- 约 10GB 磁盘空间（模型权重）

### 1. 安装

```bash
# 方式一：交互式安装脚本（2d / 3d）
bash install_libs.sh 2d
bash install_libs.sh 3d      # 含 mmdet3d/mmcv 硬装（CUDA 13 编译，--no-deps 红线）

# 方式二：手动安装（仓库根同一发行包装三个包）
pip install -e .

# 可选：开发依赖 + 全部模型引擎
pip install -e ".[all]"
```

### 2. 配置

```bash
cp auto2dlabel/configs/.env.example auto2dlabel/configs/.env   # 2D
cp auto3dlabel/configs/.env.example  auto3dlabel/configs/.env  # 3D（复用同一 LLM key）
```

编辑 `auto2dlabel/configs/.env`：

```bash
DEEPSEEK_API_KEY = "your-api-key"
DEEPSEEK_BASE_URL = https://api.deepseek.com
DEEPSEEK_MODEL = deepseek-chat        # 标注规划任务用 deepseek-chat（推理模型低温下产空 content，见已知坑）
DETECTION_MODEL = yolo26x.pt          # 默认检测模型
```

多 provider 注册表见 `auto2dlabel/configs/providers.yaml`（deepseek / openai / anthropic / ollama，本地端点免 key）。**密钥只写 `.env`（不入库）**，`providers.yaml` 仅引用环境变量名。

### 3. 下载模型权重

```bash
bash auto2dlabel/weights/download_weights.sh         # 2D 权重（交互式选择）
bash auto2dlabel/weights/download_sam3.sh            # SAM3（~3.4GB）
bash auto2dlabel/weights/download_qwen_l3.sh         # 指代 L3（Qwen2-VL-7B）
bash auto3dlabel/weights/download_detector3d.sh      # 3D 检测权重（openmmlab 直链）
```

### 4. 第一条命令

```bash
# 统一入口：无参进 TUI 对话界面
autolabel

# 统一入口：带指令自动路由（本例 3D 引擎名 → 强制 3D）
autolabel "标注 KITTI 帧 000123 中的汽车和行人" -d pointpillars_kitti

# 2D 单图标注
auto2dlabel run photo.jpg "检测汽车和行人" -d yolo26x.pt -t 0.5

# 输出：
#   outputs/photo_20260920_143052.json        — COCO JSON 标注
#   vis_outputs/vis_photo_20260920_143052.png — 可视化
#   logs/LLM_20260920_143052.log              — 结构化日志
```

---

## 用法指南

### CLI 命令矩阵

| 命令 | 说明 |
| --- | --- |
| `autolabel` | 无参 → TUI；`autolabel "指令"` → 自动路由 2D/3D 一次性对话 |
| `auto2dlabel run` | 单图/批量标注（Agent Loop，LLM 在环处置） |
| `auto2dlabel chat` | 自然语言规划 → 代码级执行（v0.6 起缺参对话确定） |
| `auto2dlabel sample` | 主动学习采样（聚合审核队列按不确定性排序） |
| `auto2dlabel cost-report` / `cost-critic` | LLM usage 台账聚合 / 质检成本对比 |
| `auto2dlabel dataset` | `add` / `list` / `remove` — 用户自建数据集注册（供 LLM 路径引导） |
| `auto3dlabel run` / `chat` | 3D 代码级直跑 / LLM 闭环（`--batch` `--track3d`） |
| `auto3dlabel nuscenes-queue` | nuScenes 复核队列生成（三引擎路由） |
| `auto3dlabel mapvec-report` | MapTR 矢量对账（契约 → 比对 → 报告 + 复核队列） |
| `python3 -m auto2dlabel.web.server` / `auto3dlabel.web.server` | Web 复核（`:8765` / `:8766`） |

### 2D 检测与分割

```bash
# 指定模型与阈值
auto2dlabel run photo.jpg "检测汽车、行人、自行车" -d yolo26x.pt -t 0.5

# 批量处理目录 + 失败隔离续跑
auto2dlabel run ./images/ "检测车辆和行人" -d yolo11n.pt --batch --resume outputs/batch_manifest.json

# 开放词汇（不限类别）
auto2dlabel chat "检测所有红色车辆" -d IDEA-Research/grounding-dino-tiny --no-wait

# 两段式分割（检测 + SAM2 mask）
auto2dlabel chat "检测并分割 000860.png 中的汽车和行人，用 sam2_l.pt" --no-wait

# 大分辨率图像切片推理
auto2dlabel run photo.jpg "检测汽车" --sahi
```

### 分类 / OBB / Pose

```bash
auto2dlabel chat "分类为猫和狗，用 clip" --no-wait                      # CLIP 零样本（中文候选可直接用）
auto2dlabel chat "分类为 cat 和 dog，用 convnext_large" --no-wait      # torchvision 监督（候选须英文）
auto2dlabel chat "检测旋转框，用 yolo11n-obb.pt" --no-wait             # OBB → dota/yolo_obb 导出
auto2dlabel chat "对 photo.jpg 做姿态估计" --no-wait                    # YOLO-pose → COCO keypoints
```

### 视频跟踪与指代

```bash
auto2dlabel run video.mp4 "检测行人" --track                              # ByteTrack + MOT 导出 + 标注成片
auto2dlabel run video.mp4 "检测行人" --track --bot-sort                   # 精度档：ReID 外观关联 + ECC
auto2dlabel run video.mp4 "检测左边红色的汽车" --track                     # 指代 L1：属性 + 方位过滤（零新权重）
auto2dlabel run video.mp4 "跟踪红车旁边的行人" --track --refer-l2          # 指代 L2（Florence-2）
auto2dlabel run video.mp4 "跟踪第二辆车后面的人" --track --refer-l3        # 指代 L3（Qwen2-VL-7B，GPU）
auto2dlabel run video.mp4 "检测行人" --track --roi auto                    # 自动车道 ROI（UFLD）
```

### 批量推理超参

四档来源：**CLI 显式** > chat 交互询问（非法重问 ≤3 次）> 执行阶段动态实测（模型加载后测单图峰值显存 → 空闲显存 × 0.85 ÷ 峰值 = 最大 batch）> 规划阶段静态表。

```bash
auto2dlabel chat "检测 /data/images 中的汽车" --batch-size 8 --num-workers 4
auto2dlabel chat "检测 dir/ 中的汽车" --batch-strategy   # 抽样 ≤8 张 + LLM 每批 1 次调参
```

### 3D 标注

```bash
auto3dlabel run 000123 "检测汽车和行人" -d yolo11s_kitti                  # 代码级直跑（2D 反投影引擎）
auto3dlabel run 000000-000399 "检测汽车" -d pointpillars_kitti --batch     # LiDAR 直检批量（一次 forward 整批）
auto3dlabel run 003712-003731 "检测汽车" -d pointpillars_kitti --track3d   # 多帧跟踪 ID + 速度
auto3dlabel chat "标注 KITTI 帧 000123 中的汽车和行人"                      # LLM 闭环（对话确定参数）
auto3dlabel nuscenes-queue -d bevfusion_nus --limit 20                     # nuScenes 复核队列
auto3dlabel mapvec-report --pred-dir <契约目录> --img-root <图像根>          # MapTR 矢量对账
```

### Web 复核

```bash
python3 -m auto2dlabel.web.server                                        # 2D 复核 → :8765
REVIEW3D_DIR=outputs/kitti3d/reviews python3 -m auto3dlabel.web.server   # 3D 复核 → :8766
```

功能：上传/加载 → 自动标注 → Canvas 可视化（mask 叠加 / OBB 旋转框 / 3D 四视图）→ 快捷键 / undo / 拖拽手柄编辑（3D 含 cuboid 六面手柄）→ 复核队列回流（`edited_by_human` 数据回路）→ 下载 COCO JSON / KITTI label。

### Python API

```python
from auto2dlabel.models.detection import create_detection_model
from auto2dlabel.tools.visualize import detect_and_visualize

detect_and_visualize(
    image_path="photo.jpg",
    prompts=["car", "person"],
    model_name="yolo26x.pt",
    confidence_threshold=0.3,
)
```

---

## 模型目录

### 检测（33）

| 引擎 | 数量 | 示例 | 特点 |
| --- | --- | --- | --- |
| **Grounding DINO** (HF) | 3 | `grounding-dino-tiny/base/large` | 开放词汇，文本 prompt 直出 bbox |
| **Ultralytics** | 17 | YOLO11/12/26 n/s/m/l/x + RT-DETR l/x | 统一 API，权重自动下载 |
| **PyTorch Vision** | 9 | Faster R-CNN / RetinaNet / SSD / SSDLite / FCOS | COCO 预训练，高召回 |
| **mmdet** | 4 | `rtmdet_s/m/l/x` | 高召回档（v0.6，与 Faster R-CNN 同档更快） |

### 分割（25）

| 模型 | 数量 | 特点 |
| --- | --- | --- |
| **SAM3** | 1 | 文本 prompt → 检测+分割一步，开放词汇，最强但慢（~20s） |
| **SAM / SAM2 / SAM2.1** | 13 | bbox prompt → mask；`sam2_l.pt` 为**默认分割模型**（coco_seg 0.6368） |
| **Mask R-CNN** | 3 | COCO 两款 + `maskrcnn_r50_cityscapes` **域内权重**（cityscapes 全量 0.5149 vs 基线 0.0082） |
| **FastSAM** | 2 | bbox IoU 匹配，最轻量（丢框多，快速预览） |
| **Mask2Former** | 1 | 实例分割质量档（v0.6，mask mAP 0.6008 超两段式 SAM2 0.5778） |
| **torchvision 语义分割** | 6 | FCN / DeepLabV3 / LRASPP，VOC 21 类全图逐类 mask |

分割自动路由：`sam3`/`maskrcnn`/`fcn*`/`deeplabv3*`/`lraspp*` 自带检测；`sam`/`fastsam` 先检测后分割。两段式 prompt 过滤阈值默认 0.3。

### 分类（16） / OBB（15） / Pose（16）

| 任务 | 模型 | 说明 |
| --- | --- | --- |
| 图像分类 | CLIP `clip-vit-base-patch32` / SigLIP `siglip-base-patch16-224` | 零样本，多候选 softmax top-K，中文候选可直接用 |
| 图像分类 | torchvision 14 款（convnext/maxvit/swin/efficientnet/vit/resnet/resnext） | ImageNet1K 监督 top-K，**候选须英文** |
| OBB | `yolo11/12/26 n/s/m/l/x-obb.pt` | 旋转框，`Bbox.angle` 弧度约定，DOTA / YOLO-OBB 导出 |
| Pose | `yolo11/12/26 n/s/m/l/x-pose.pt` + `rtmpose_l` | COCO 17 点，keypoints 随 bbox 输出 |

### 3D 引擎（9）

| 系 | 引擎 | 数据集 | 说明 |
| --- | --- | --- | --- |
| LiDAR 直检 | `pointpillars_kitti` / `pvrcnn_kitti` | KITTI | 快速扫 / 精度档 |
| LiDAR 直检 | `pointpillars_nus` / `centerpoint_nus` / `free_anchor_nus` | nuScenes | 快速 / 精度 / 速度档 |
| 单目 | `pgd_kitti` / `fcos3d_nus` | KITTI / nuScenes | 无 LiDAR 降级交叉验证 |
| 融合 | `bevfusion_nus` / `bevfusion_lidar_nus` | nuScenes | 相机-LiDAR 融合（lidar 变体为降级备选） |
| 反投影 | 复用 2D G-DINO + SAM2 | KITTI / nuScenes | 开放词汇回退，无需 3D 权重 |

---

## Benchmark 结果

覆盖 **12 数据集**（检测 5 + 分割 4 + 分类 2 + OBB 1），一键 `bash auto2dlabel/benchmarks/run_benchmarks.sh`。
完整实测数据见 [`auto2dlabel/tests/test-v0.X.md`](auto2dlabel/tests/) 与 [`auto3dlabel/tests/test-v0.X.md`](auto3dlabel/tests/)。

| 数据集 | 模型 | 指标 | 值 |
| --- | --- | --- | --- |
| COCO 2017 val | yolo26x（GPU） | mAP@0.5 | 0.4668 |
| VOC 2007 | yolo26x（GPU） | mAP@0.5 | 0.5909 |
| KITTI 2D | yolo26x（GPU） | mAP@0.5 | 0.4081 |
| DOTA Task1 OBB | yolo11n-obb（50 图 / 全量） | mAP@0.5 | 0.7047 / 0.6161 |
| COCO seg（50 图） | yolo26x + **sam2_l** | mask mAP | 0.6368 |
| Cityscapes seg（全量 500 图） | **maskrcnn_r50_cityscapes**（域内权重） | mask mAP | **0.5149**（基线 0.0082，62.8×） |
| D2SA seg | box-prompted（上界） | mask mAP | 0.8873 |
| ImageNet100 | resnet18 ImageNet1K（CPU / GPU） | top-1 / top-5 | 0.7800 / 0.9460（CPU）；0.7400 / 0.9400（GPU） |
| ILSVRC2012 val（5 万图） | resnet18 ImageNet1K | top-1 / top-5 | 0.6968 / 0.8899 |
| KITTI 3D | `pointpillars_kitti` | Car easy/moderate/hard | **89.8 / 82.0 / 77.2**（moderate 超官方 zoo 77.6） |
| KITTI 3D | `pvrcnn_kitti` | Car moderate | 86.9（精度档） |
| nuScenes Mini | `bevfusion_nus` | mAP | 27.0（融合档最高） |

**域边界如实记录**：DOTA 航拍（0.066）、MOT 密集行人（0.0909）在 COCO 预训练模型下失效——GPU 复测定性为**数据域边界**（规模/SAHI/架构均无效），出路是域内微调回采闭环（cityscapes 0.0082→0.5149、KITTI 0.2702→0.8867 两例已闭环）。

模块划分与程序规范见 [`docs/Benchmark_plan.md`](docs/Benchmark_plan.md)，数据集就绪状态见 [`docs/datasets_plan.md`](docs/datasets_plan.md)。

---

## 导出格式

| 格式 | 输出 | 适用框架 |
| --- | --- | --- |
| **COCO JSON** | 单文件（images/annotations/categories） | Detectron2, MMDetection, TorchVision |
| **YOLO txt** | 每图 `.txt`（归一化） + `classes.txt` | Ultralytics YOLO |
| **Pascal VOC XML** | 每图 `.xml`（绝对像素） | 旧项目兼容 |
| **LabelMe JSON** | 每图 `.json`（polygon） | LabelMe 生态 |
| **cls JSON** | 每图 `.json`（image_path/width/height/model/labels） | 分类训练 |
| **DOTA txt** | 每图 `.txt`（8 角点） | 旋转框（航拍） |
| **YOLO-OBB txt** | 每图 `.txt`（`cls cx cy w h angle`） | Ultralytics OBB |
| **MOT txt** | 序列级合并（逐帧 + track_id） | 多目标跟踪 |
| **COCO keypoints** | 单文件（keypoints + num_keypoints） | 姿态估计 |
| **KITTI label** | 每帧 `.txt`（15 字段，含 3D） | KITTI 3D 检测 |
| **nuScenes devkit** | 全局四元数 + velocity + track_id | nuScenes 3D 检测 |

COCO/YOLO/VOC 已通过 round-trip 测试：导出 → 回读 → 坐标误差 < 1e-3（YOLO/COCO）或 < 1px（VOC）。

---

## 项目结构

```txt
AutoLabel/
├── autolabel/                    # 统一入口薄壳（v1.0）
│   ├── cli.py                    # 无参 → TUI；指令 → 路由一次性对话
│   ├── route.py                  # route_domain 路由判据单一事实源
│   └── tui/app.py                # Textual ChatApp（9 斜杠命令 + 任务面板 + 会话）
├── auto2dlabel/                  # 2D 标注主体（v0.1–v1.1）
│   ├── cli.py                    # Typer 入口（业务在 cli_commands/cli_run/cli_execute/cli_track）
│   ├── agent/                    # Agentic 编排层
│   │   ├── orchestrator.py       # Agent Loop（run，LLM 在环处置）
│   │   ├── planner.py            # NL 任务规划器（chat）
│   │   ├── dialog.py             # 对话式规划通用骨架（2D/3D 共用）
│   │   ├── llm.py                # LLM 客户端 + usage 台账 + provider 注册表
│   │   ├── evaluate.py           # F4 代码级质量评估 + 处置动作
│   │   ├── critic.py             # 质检 Agent（v1.1）
│   │   ├── experience.py         # RAG 标注经验库（v1.1）
│   │   └── batch_strategy.py     # 批次级 LLM 调参
│   ├── models/                   # 模型层（model_catalog 为单一事实源）
│   ├── tools/                    # 工具层（detection/segmentation/tracking/constraints/refer/
│   │                             #   export/recommend/hitl/sampling/device/visualize/...）
│   ├── schema/ export/ web/      # 数据 Schema / 导出器 / Web 复核
│   ├── benchmarks/               # Benchmark 套件（12 数据集）
│   ├── milestone/ tests/         # 里程碑定义 / 实测数据 + pytest 用例
│   ├── configs/                  # .env（不入库）+ providers.yaml + model_catalog
│   └── weights/                  # 模型权重（不入库）
├── auto3dlabel/                  # 3D 标注主体（v0.1–v0.4）
│   ├── cli.py                    # run / chat / nuscenes-queue / mapvec-report
│   ├── models/detection3d.py     # 三系 9 引擎统一工厂
│   ├── tools/                    # backproject/cluster/fit/geometry/pipeline/track3d/
│   │                             #   nuscenes_pipeline/train3d/mapvec_compare
│   ├── schema/ data/ export/     # Box3D/NusBox/MapVec 契约 / KITTI/nuScenes 数据 / 导出
│   ├── agent/                    # orchestrator3d / planner3d / tools3d
│   ├── web/                      # 3D 复核（three.js 四视图 + 手柄编辑）
│   ├── benchmarks/               # KITTI 双口径（官方 40-point）+ nuScenes + 冒烟套件
│   └── milestone/ tests/ weights/
├── docs/                         # 计划书 / Benchmark 规范 / 数据集计划 / UI 企划
├── install_libs.sh               # 安装脚本（2d / 3d）
├── gitpush.sh                    # 提交推送（含学术加速 + master 校验）
└── pyproject.toml                # 单一发行包（三包同装）
```

---

## 路线图

| 版本 | 状态 | 内容 |
| --- | --- | --- |
| **v0.1–v0.2** | ✅ | 检测（3 引擎）+ 分割 + 类别推荐 + Web 基础审核 + Agentic 闭环 |
| **v0.3** | ✅ | 分类（CLIP/SigLIP + torchvision 14）+ OBB（YOLO-OBB 15）+ F4 质量评估 + LLM Evaluate + batch 动态调优 + cityscapes 域内权重（当时目录口径 **84 模型**） |
| **v0.4** | ✅ 2026-08-23 | Tracking（ByteTrack / BoT-SORT + 指代 L1）+ KITTI 域内微调（0.8867）+ AgentState 续跑 + Web 三件套 |
| **v0.5** | ✅ 2026-08-23 | Pose（YOLO-pose）+ 指代 L2/L3（Florence-2 / Qwen2-VL-7B）+ 自动车道 ROI + ILSVRC2012 |
| **v0.6** | ✅ 2026-08-31 | 对话式 Agent 统一入口（`parse_dialog` 多轮）+ mmdet/mmpose 双引擎 + LLM Harness Token 降本 |
| **v1.0** | ✅ 2026-09-06 | Agentic 交互化：`autolabel` 统一入口 + Textual TUI + provider 注册表 + 后台任务面板 + 会话管理 + HITL 指挥台 |
| **v1.1** | ✅ P1+P2 2026-09-17 | RAG 标注经验库（CLIP 检索 + planner few-shot 注入）+ 质检 Agent 独立化（Critic + cost-critic）；Router Agent 缓行（[milestone/v1.1.md](auto2dlabel/milestone/v1.1.md)） |
| **auto3dlabel v0.1** | ✅ 2026-08-24 | 单帧 KITTI 五步管线（标定→反投影→聚类→拟合）+ Agentic / Web 闭环 |
| **auto3dlabel v0.2** | ✅ 2026-08-27 | mmdet3d PointPillars（KITTI Car moderate 82.0 超官方）+ Tracker3D + nuScenes Mini |
| **auto3dlabel v0.3** | ✅ 2026-08-31 | 对话式 Planner + PV-RCNN/CenterPoint 精度 + BEVFusion 融合 + Web 真 3D 复核 + LabelAny3D 验证（不接入）+ 模型矩阵扩展 |
| **auto3dlabel v0.4** | ✅ 2026-09-02 | HITL 编辑闭环（cuboid 六面手柄 + undo/redo）+ nuScenes 端到端标注闭环 + KITTI 微调闭环 |
| **auto3dlabel v0.4 P4** | ✅ 2026-09-15 | 地图矢量对账（MapTR 契约 `mapvec_pred/1` 消费：Chamfer 比对 + overlay/BEV + 报告 + 复核队列；周期外增量，已回填 [milestone/v0.4.md](auto3dlabel/milestone/v0.4.md)） |

详细里程碑定义与验收见各包 `milestone/`，实测数据见各包 `tests/test-v0.X.md`。

---

## 已知限制

诚实记录当前版本的限制与设计取舍：

- **域边界是硬约束** — COCO 预训练模型在特定域（MOT 密集行人、DOTA 航拍、cityscapes 小目标）精度低，GPU 复测定性为数据域边界（规模 / SAHI / 架构均无效）；出路是**域内微调回采闭环**（两例已闭环）
- **OBB 只支持 YOLO-OBB** — Oriented R-CNN 因 mmrotate 依赖重延后（取舍注记见计划书）
- **3D 反投影拟合路线精度天花板** — v0.1 实证 3D AP 近零；已用 LiDAR 直检引擎（moderate 82.0）为主，反投影仅作开放词汇回退
- **auto3dlabel 依赖 mmdet3d 硬装** — mmcv 2.1.0 需 CUDA 13 源码编译（`install_libs.sh 3d`）；无 GPU 时 LiDAR 直检不可用（反投影引擎与部分 nuScenes 引擎可 CPU 跑，bevfusion/centerpoint 依赖自定义 CUDA op 会自动降级）
- **LLM 在单任务中边际价值有限** — v0.1c 实验表明单一检测任务上 LLM Agent 与 `--no-llm` 结果相同；LLM 价值体现在多任务编排、模糊指令处理、异常处置
- **推理模型不适用于规划任务** — `deepseek-v4-flash` 等推理模型在低温确定性 JSON 请求下推理链吃满 max_tokens 产空 content；`chat()` 统一出口已钳制（温度 1.0 + 预算 ≥4096），但**标注规划建议 `.env` 用 `deepseek-chat`**
- **Benchmark 路径硬编码** — 数据集路径指向 AutoDL 服务器特定目录（`~/autodl-tmp/Documents/datasets`、`/root/autodl-pub`），其他环境需手动修改
- **MapVec 契约为复制而非依赖** — `schema/mapvec*.py` 复制自 AutoDriveData（依赖单向红线：AutoLabel 不反向 import）；产出方改契约需同步，由交叉验证测试锁定一致性
- **数据许可非商用** — KITTI（CC BY-NC-SA 3.0）/ nuScenes（CC BY-NC-SA 4.0）均**非商用**，商业化需自采或商用许可数据
- **密钥管理** — `configs/.env` 含真实 API Key，已 gitignore 且提供 `.env.example` 模板；**公开部署前请确认未误提交**

---

## 开发与贡献

本项目目前为个人工程作品集项目。欢迎 Issue 和 PR。

### 质量门（每版本硬性门槛）

```bash
python -m pytest auto2dlabel/tests auto3dlabel/tests -q   # 当前 1354 passed, 4 skipped
mypy autolabel auto2dlabel auto3dlabel                    # 当前 168（三包必须写全）
ruff check .                                              # 当前 87（存量 benchmark 文件）
pyright autolabel auto2dlabel auto3dlabel                 # 当前 3（mapvec 交叉验证存量）
```

### 前端冒烟（零浏览器）

```bash
cd auto2dlabel/tests/helpers && npm i jsdom && node smoke_web.js    # 2D Web 复核 76 断言
cd auto3dlabel/tests/helpers && node smoke_web3d.js                 # 3D Web 复核 94 断言
```

### 约定

- **提交**：Conventional Commits（`feat:` / `fix:` / `docs:` / `refactor:`）
- **分支**：`feature/xxx`、`bugfix/xxx`
- **测试**：代码在 `auto2dlabel/tests/`、`auto3dlabel/tests/`（不进 `cli.py`）；`benchmarks/` 只做整体软件性能测试
- **红线**：权重与 `.env` **不入库**；纯本地部署（DeepSeek API 为唯一外部依赖，可换本地 Qwen/Ollama）；Protocol + Fake 注入保证单测零真实权重

---

## 许可证

[MIT](LICENSE) © AutoLabel Team

---

<p align="center">
  <sub>Built with ❤️ by an AI-first perception engineer. 用 AI 重新定义标注。</sub>
</p>
