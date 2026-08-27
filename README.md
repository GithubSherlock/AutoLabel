<p align="center">
  <h1 align="center">AutoLabel</h1>
  <p align="center"><strong>Agentic 2D/3D Data Annotation — AI-first, Human-in-the-loop</strong></p>
  <p align="center">自然语言驱动 · LLM Agent 编排 · 多模型引擎 · 代码级质量评估 · 多格式导出</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/models-84%2B-orange" alt="84+ Models">
  <img src="https://img.shields.io/badge/version-0.5.0-informational" alt="Version 0.5.0">
</p>

---

## 目录

- [什么是 AutoLabel](#什么是-autolabel)
- [特性](#特性)
- [快速开始](#快速开始)
- [用法指南](#用法指南)
- [模型目录](#模型目录)
- [导出格式](#导出格式)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [已知限制](#已知限制)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 什么是 AutoLabel

**AutoLabel** 是一个 Agentic 图像自动标注工具，将传统标注工具的范式从「人为主、AI 为辅」反转为 **「AI 为主、人为辅」**。

传统的标注软件（如 CVAT、Label Studio）以人工画布 + 快捷键为核心，AI 预标注能力仅作为辅助插件存在。AutoLabel 重新定义了工作流：**用户只需下达自然语言指令，由 LLM Agent 自主规划、调用检测/分割模型、评估结果，并在不确定时主动请求人工介入**。

```txt
用户指令（自然语言）
        │
        ▼
  LLM Agent 规划
        │
        ▼
  工具调用（检测 / 分割 / …）
        │
        ▼
  HITL 置信度分流
        │
        ▼
  多格式导出（COCO / YOLO / VOC / LabelMe）
```

### 目标用户

- **AI 算法工程师 / 数据科学家**：为训练或微调模型快速生成高质量预标注，再在 Label Studio / CVAT 中精修
- **使用场景**：单人或小团队在消费级硬件（Apple Silicon / 单 GPU）上对百到千张级别图像进行自动预标注
- **不是面向**：专业标注团队的大规模生产管线（需 Web 协作、权限管理、SaaS），也不面向零 AI 背景的标注员

### 子项目

| 子项目 | 状态 | 说明 |
| --- | --- | --- |
| **auto2dlabel** | ✅ v0.1–v0.5 完成，v0.6 📋 | 2D 检测/分割/分类/OBB + Tracking + Pose + VLM 指代 + Web 审核闭环，84 模型 |
| **auto3dlabel** | ✅ v0.1–v0.2 完成，v0.3 📋 | 3D 标注：LiDAR 直检（mmdet3d PointPillars）+ 反投影拟合回退引擎 + 多帧跟踪 + nuScenes |

---

## 特性

- **🗣️ 自然语言交互** — `chat` 命令支持中英文自然语言描述标注任务，LLM 自动提取参数（类别、阈值、模型），缺失参数交互追问（v0.6 升级为 LLM 多轮对话确定参数）
- **🧠 Agentic 编排** — LLM Agent 规划多步任务（检测 → 分割 → 导出），自动调用工具，token 高效设计（结果摘要注入）；F4 代码级质量评估（0 框重试/类别覆盖/超框警告）+ 不合格时条件触发 LLM Evaluate（accept / flag / retry / 换模型）
- **🔧 3 引擎 × 84+ 模型** — Grounding DINO（开放词汇）、Ultralytics YOLO（YOLO11/12/26 + RT-DETR + YOLO-pose）、PyTorch Vision（Faster R-CNN/RetinaNet/SSD/FCOS）；分割支持 SAM/SAM2/SAM3/Mask R-CNN（含 cityscapes 域内权重）/FastSAM/torchvision 语义分割（FCN/DeepLabV3/LRASPP）；分类 CLIP/SigLIP/torchvision ImageNet1K（convnext/maxvit/swin/efficientnet/vit/resnet/resnext）；OBB YOLO-OBB 11/12/26（n/s/m/l/x）
- **✅ HITL 置信度分流** — 三档阈值自动分流 + Web 复核队列（人工修正回流）+ 主动学习采样；低置信与质量不合格项强制入复核
- **📦 多格式导出** — COCO JSON / YOLO txt / Pascal VOC XML / LabelMe JSON / cls JSON / DOTA / YOLO-OBB txt / MOT / COCO keypoints（Pose），已通过 round-trip 测试（坐标误差 < 1e-3）
- **🖼️ 零样本图像分类** — CLIP/SigLIP 多候选 softmax 排序取 top-K，`Annotation.labels` + cls JSON 导出
- **📐 OBB 旋转框** — YOLO-OBB 15 枚（yolo11/12/26 n/s/m/l/x），`Bbox.angle` 弧度约定，DOTA 8 角点 / YOLO-OBB txt 导出，旋转多边形可视化
- **🎥 Tracking** — 视频/帧目录逐帧检测 + ByteTrack（默认）/ BoT-SORT（精度档，ReID + ECC），MOT 导出 + 轨迹可视化；指代约束 L1（属性+方位）/ L2（Florence-2）/ L3（Qwen2-VL-7B）阶梯
- **💀 Pose** — YOLO-pose 关键点检测，COCO keypoints 导出
- **🔁 Batch 韧性 + 动态调优** — 单图失败隔离 + `--resume` 断点续跑（AgentState 快照）；批量推理动态实测最大 batch 用满 GPU（parity 双铁律：rect=False + TF32 关闭）；`sample` 按不确定性排序聚合审核队列
- **⚡ `--no-llm` Baseline** — 绕过 LLM Agent，使用内置关键词映射直调检测模型，用于对比实验和离线场景
- **🔍 自动类别推荐** — 扫描全图 80 个 COCO 类别，按检出数量和置信度排序推荐 Top-K 类别
- **🌐 Web 审核界面** — FastAPI + Canvas SPA（CVAT 借鉴：快捷键/undo/手柄/过滤/区域 issue），mask 叠加 + 复核队列 + edited_by_human 数据回路 + COCO JSON / 可视化下载
- **📊 Benchmark 套件** — 覆盖 12 数据集（COCO/VOC2007/KITTI/Cityscapes/nuImages/DOTA/DOTA-OBB/D2SA/MOT17/MOT20/coco_seg/ImageNet100 + ILSVRC2012），检测/分割/分类/OBB 全任务评测，一键 `run_benchmarks.sh`
- **🧊 3D 标注（auto3dlabel）** — LiDAR 直检（mmdet3d PointPillars，KITTI 全 val Car moderate 82.0 超官方）+ 反投影拟合开放词汇回退引擎 + Tracker3D 多帧 ID/速度 + KITTI/nuScenes 导出 + 3D/BEV IoU 评测
- **📝 结构化日志** — 每次调用的完整 JSON 日志（LLM/Chat/PythonAPI），可复现、可审计

---

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA GPU（推荐）或 Apple Silicon MPS 或 CPU
- 约 10GB 磁盘空间（模型权重）

### 1. 安装

```bash
# 方式一：交互式安装脚本（2d / 3d）
bash install_libs.sh 2d
bash install_libs.sh 3d      # 含 mmdet3d/mmcv 硬装（CUDA 13 编译，--no-deps 红线）

# 方式二：手动安装
pip install -r auto2dlabel/requirements.txt
cd auto2dlabel && pip install -e .

# 可选：安装开发依赖和所有模型引擎
pip install -e ".[all]"
```

### 2. 配置

编辑 `auto2dlabel/configs/.env`：

```bash
DEEPSEEK_API_KEY = "your-api-key"
DEEPSEEK_BASE_URL = https://api.deepseek.com
DEEPSEEK_MODEL = deepseek-v4-pro
DETECTION_MODEL = yolo26x.pt          # 默认检测模型
```

支持 LLM Provider：DeepSeek（默认）、OpenAI、Anthropic、Qwen（OpenAI 兼容接口）。

### 3. 下载模型权重

```bash
bash auto2dlabel/weights/download_weights.sh   # 交互式选择
bash auto2dlabel/weights/download_sam3.sh       # SAM3 权重（~3.4GB）
```

### 4. 第一条命令

```bash
# 用 YOLO12 nano 检测图片中的汽车和行人
auto2dlabel run photo.jpg "检测汽车和行人" -d yolo12n.pt -t 0.5

# 输出：
#   outputs/photo_20260810_143052.json       — COCO JSON 标注
#   vis_outputs/vis_photo_20260810_143052.png — 可视化
#   logs/LLM_20260810_143052.log             — 结构化日志
```

---

## 用法指南

### CLI 命令一览

```bash
auto2dlabel --help           # 查看所有命令
auto2dlabel run --help       # run 命令参数
auto2dlabel chat --help      # chat 命令参数
```

### `run` — 单张 / 批量标注

```bash
# 基础用法：单张图片 + 自然语言指令
auto2dlabel run photo.jpg "检测所有汽车"

# 指定检测模型和置信度阈值
auto2dlabel run photo.jpg "检测汽车、行人、自行车" -d yolo26x.pt -t 0.5

# 批量处理整个目录
auto2dlabel run ./images/ "检测车辆和行人" -d yolo11n.pt --batch

# 导出为 YOLO 格式
auto2dlabel run photo.jpg "检测汽车" -d yolo26x.pt -e yolo

# 不使用 LLM 的 Baseline 模式（关键词映射 + 直调模型）
auto2dlabel run photo.jpg "检测汽车和行人" --no-llm -d yolo26x.pt

# 使用不同 LLM Provider
auto2dlabel run photo.jpg "detect all cars" -p openai -m gpt-4o
```

### `chat` — 自然语言交互式标注

`chat` 命令是 v0.1 的核心功能——你只需要说一段话，系统自动解析并执行：

```bash
# 单步任务
auto2dlabel chat "检测 000860.png 中的汽车和行人，conf=0.5，用 faster-rcnn"

# 多步任务（检测 + 分割）
auto2dlabel chat "检测并分割 000860.png 中的汽车和行人，用 sam3" --no-wait

# 更复杂的多步指令
auto2dlabel chat "
先检测 /data/images/ 下所有图片中的汽车和行人，置信度 0.5，
然后分割检测到的所有汽车，用 SAM，
最后导出为 COCO 格式
" --no-wait

# 交互模式（不传参数，进入对话）
auto2dlabel chat
```

**Chat 能力一览**：

| 能力 | 说明 | 示例 |
| --- | --- | --- |
| NL 参数提取 | 中英文类别名自动映射 | "汽车"→"car"、"行人"→"person" |
| 模型别名 | 模糊模型名自动匹配 | "faster rcnn"→`fasterrcnn_resnet50_fpn_v2` |
| 缺失追问 | 缺少参数时交互确认，30s 超时自动默认 | "检测哪些类别？用默认模型？" |
| 多步编排 | 检测 → 分割 → 导出顺序执行 | "先检测汽车，再分割" |
| 类别推荐 | 未指定类别时自动扫描推荐 | 扫描 80 个 COCO 类，Top-10 建议 |

### Web 审核界面

启动 Web 服务，在浏览器中审核和修正标注结果：

```bash
python -m auto2dlabel.web.server       # 2D 复核
REVIEW3D_DIR=outputs/<dir>/reviews python3 -m auto3dlabel.web.server  # 3D 复核
# 访问 http://localhost:8765
```

功能：上传图片 → 选择检测/分割模型 → 调整阈值 → 自动标注 → Canvas 可视化（mask 叠加）→ 快捷键/undo/拖拽手柄编辑 → 复核队列（人工修正回流）→ 下载 COCO JSON。

### Python API

```python
from auto2dlabel.models.detection import create_detection_model
from auto2dlabel.tools.visualize import detect_and_visualize

# 一键检测 + 可视化 + 导出 + 日志
detect_and_visualize(
    image_path="photo.jpg",
    prompts=["car", "person"],
    model_name="yolo26x.pt",
    confidence_threshold=0.3,
)
```

### Benchmark

```bash
# 运行所有 benchmark（需要数据集）
python -m auto2dlabel.benchmarks.run_all

# 运行单个数据集
python -m auto2dlabel.benchmarks.coco_benchmark --max-images 50
python -m auto2dlabel.benchmarks.voc_benchmark --max-images 100

# 一键包装（环境检查 + 分组透传，CWD 自定位任意目录可跑，见 docs/Benchmark_plan.md §10.8）
bash auto2dlabel/benchmarks/run_benchmarks.sh detection
bash auto2dlabel/benchmarks/run_benchmarks.sh classification   # 分类组（imagenet100）
# 仅 CPU 时检测/分割全组加 --extra "--model yolo11n.pt"（默认 yolo26x 超 run_all 超时）
```

**实际 Benchmark 结果（精选，完整实测见 `auto2dlabel/tests/test-v0.X.md`）**：

| 数据集 | 模型 | 指标 | 值 |
| --- | --- | --- | --- |
| COCO 2017 val | yolo26x（GPU 复测 +0.20） | mAP@0.5 | 0.65+ |
| VOC 2007 (100 imgs) | yolo26x | mAP@0.5 | 0.592 |
| KITTI | yolo26x | mAP@0.5 | 0.441 |
| DOTA Task1 OBB | yolo11n-obb | mAP@0.5 | 0.7047（批量/逐图一致 0.8397，parity 修复后） |
| COCO seg (50 imgs) | yolo26x + **sam2_l**（默认分割模型） | mask mAP | 0.6368 |
| Cityscapes seg | **maskrcnn_r50_cityscapes**（域内权重） | mask mAP | **0.5149**（基线 0.0082，62.8×） |
| ILSVRC2012 val（5 万图） | ImageNet1K 监督 | top-1 / top-5 | 0.6968 / 0.8899 |
| MOT17+MOT20 | yolo26x + ByteTrack | MOTA / IDF1 | 域边界：检测 recall 瓶颈（详见 test-v0.3/0.4） |
| KITTI 3D（auto3dlabel） | pointpillars_kitti（mmdet3d） | Car 官方口径 easy/moderate/hard | **89.8 / 82.0 / 77.2**（moderate 超官方 77.6） |

模块划分、缺口与程序规范见 `docs/Benchmark_plan.md`，数据集就绪状态见 `docs/datasets_plan.md`。

### 测试

```bash
pytest auto2dlabel/tests/ -v
# functional/test_export_roundtrip.py  — COCO/YOLO/VOC 导出→回读一致性
# functional/test_benchmark_metrics.py — IoU/AP 边界条件
# functional/test_no_llm_baseline.py   — no-LLM baseline / benchmark 工具
```

---

## 模型目录

### 检测模型（29 个）

| 引擎 | 数量 | 示例 | 特点 |
| --- | --- | --- | --- |
| **Grounding DINO** (HF) | 3 | `grounding-dino-tiny/base/large` | 开放词汇，文本 prompt 直出 bbox，无需预定义类别 |
| **Ultralytics YOLO** | 17 | YOLO11/12/26 n/s/m/l/x, RT-DETR l/x | 统一 API，自动下载权重至 `auto2dlabel/weights/` |
| **PyTorch Vision** | 9 | Faster R-CNN, RetinaNet, SSD, SSDLite, FCOS | COCO 预训练，torchvision 内置 |

### 推荐场景

| 场景 | 推荐模型 | 理由 |
| --- | --- | --- |
| 自动标注（宁多勿漏） | `fasterrcnn_resnet50_fpn_v2` | 召回率最高，person 检出 16（vs YOLO 仅 6），置信度 85%+ |
| 快速预览 | `yolo11n.pt` / `yolo12n.pt` | 速度最快，适合快速扫图 |
| 均衡选择 | `yolo26x.pt` | 精度与速度折中，VOC mAP@0.5 达 59.2% |
| 开放词汇 | `IDEA-Research/grounding-dino-tiny` | 无需预设类别，中文 prompt 直译后使用 |
| 卫星/航拍 | `yolo11n-obb.pt` | OBB 旋转框（v0.3 已支持） |

### 分割模型（24 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **SAM3** | Ultralytics (Meta) | ~3.4GB | 文本 prompt → 检测+分割一步，开放词汇，**最强** |
| **SAM 2 / 2.1** | Ultralytics (Meta) | 自动下载 | bbox prompt → mask，精度高；`sam2_l.pt` 为**默认分割模型** |
| **SAM** | Ultralytics (Meta) | ~160MB | bbox prompt → mask，轻量快速 |
| **FastSAM** | Ultralytics | ~140MB | bbox IoU 匹配，最轻量 |
| **Mask R-CNN** | torchvision | 自动下载 | COCO 预训练，检测+分割一步 |
| **Mask R-CNN cityscapes** | torchvision（mmdet 权重转换） | 需转换 | **域内权重**：cityscapes 全量 mask mAP 0.5149（基线 0.0082） |
| **torchvision 语义分割** | torchvision | 自动下载 | FCN×2 / DeepLabV3×3 / LRASPP×1，VOC 21 类全图分割（每类一个 mask） |

**实测对比（000860.png，"car + person"，conf=0.1）**：

| 维度 | SAM | SAM3 | Mask R-CNN |
| --- | --- | --- | --- |
| 耗时 | 1.0s | 20.6s | 2.5s |
| 总 mask 数 | 14 | **41** | 36 |
| mask 精度（多边形点数） | 260 | **892** | 860 |

### 分类模型（16 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **CLIP** | transformers | openai/clip-vit-base-patch32 | 零样本分类，多候选 softmax 排序 |
| **SigLIP** | transformers | google/siglip-base-patch16-224 | sigmoid logits，softmax 归一化排序 |
| **torchvision** | torchvision | 自动下载（TORCH_HOME） | convnext_large/base、maxvit_t、swin_b、efficientnet_v2_l、vit_b_16 + resnet18/34/50/101/152、resnext50_32x4d/101_32x8d/101_64x4d；ImageNet1K 监督 top-K，candidates 子串过滤（候选须英文；与 CLIP/SigLIP 零样本语义不同），resnet50 通用之选 |

### OBB 旋转框（15 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **YOLO-OBB** | Ultralytics | yolo11/12/26 n/s/m/l/x-obb.pt | 旋转框检测，`dota` / `yolo_obb` 导出 |

---

## 导出格式

| 格式 | 输出 | 适用框架 |
| --- | --- | --- |
| **COCO JSON** | 单文件，含 images/annotations/categories | Detectron2, MMDetection, TorchVision |
| **YOLO txt** | 每图一个 `.txt`（归一化坐标） + `classes.txt` | Ultralytics YOLO 全家桶 |
| **Pascal VOC XML** | 每图一个 `.xml`（绝对像素坐标） | 旧项目兼容 |
| **LabelMe JSON** | 每图一个 `.json`（polygon 标注） | LabelMe 生态 |
| **cls JSON** | 每图一个 `.json`（image_path/width/height/model/labels） | 分类训练数据 |
| **DOTA txt** | 每图一个 `.txt`（8 角点 `class_id x1 y1 … x4 y4 0`） | 旋转框检测（航拍） |
| **YOLO-OBB txt** | 每图一个 `.txt`（`cls cx cy w h angle`，弧度） | Ultralytics OBB 训练 |

所有格式均已通过 round-trip 测试：导出 → 回读 → 坐标误差 < 1e-3（YOLO/COCO）或 < 1px（VOC）。

---

## 项目结构

```txt
AutoLabel/
├── auto2dlabel/                  # 2D 标注主体（v0.1–v0.5 ✅，v0.6 📋）
│   ├── cli.py                    # CLI 入口（Typer，业务逻辑在 cli_*.py）
│   ├── agent/                    # Agentic 编排层
│   │   ├── orchestrator.py       # Agent Loop 编排器（run 命令）
│   │   ├── planner.py            # NL 任务规划器（chat 命令）
│   │   ├── llm.py                # LLM 客户端（DeepSeek/OpenAI/Anthropic）
│   │   ├── evaluate.py           # F4 代码级质量评估 + 处置动作
│   │   ├── batch_strategy.py     # 批次级 LLM 调参（每批 1 次）
│   │   └── state.py              # Agent 状态管理（快照续跑）
│   ├── tools/                    # 工具层（detection/segmentation/classification/obb/
│   │                             #   tracking/constraints/refer/export/recommend/hitl/
│   │                             #   batch/device/evaluate/sampling/visualize/log/confirm）
│   ├── models/                   # 模型层（model_catalog 84 模型单一事实源）
│   ├── schema/                   # 数据 Schema（Bbox/Mask/Annotation/TaskPlan）
│   ├── export/                   # 导出器（COCO/YOLO/VOC/LabelMe/cls/DOTA/YOLO-OBB/MOT/keypoints）
│   ├── web/                      # Web 审核界面（FastAPI + Canvas SPA，CVAT 借鉴交互）
│   ├── benchmarks/               # Benchmark 套件（12 数据集 + run_benchmarks.sh）
│   ├── milestone/ tests/         # 里程碑定义 + 实测数据
│   ├── configs/.env              # 配置文件
│   └── weights/                  # 模型权重目录（不入库）
├── auto3dlabel/                  # 3D 标注主体（v0.1–v0.2 ✅，v0.3 📋）
│   ├── cli.py                    # CLI：run / chat（代码级直跑 + LLM 闭环）
│   ├── models/detection3d.py     # LiDAR 3D 检测器（mmdet3d：pointpillars 等）
│   ├── tools/                    # backproject/cluster/fit/geometry/pipeline/track3d
│   ├── data/ schema/ export/     # KITTI/nuScenes 数据、Box3D/NusBox、label 导出
│   ├── agent/                    # orchestrator3d / planner3d / tools3d
│   ├── benchmarks/               # KITTI 双口径（官方 40-point）+ nuScenes 评测
│   ├── web/                      # 3D 复核（复用 2D 四端点协议）
│   ├── milestone/ tests/         # 里程碑定义 + 实测数据
│   └── weights/                  # mmdet3d 权重 + configs（不入库）
├── docs/                         # 计划书 / Benchmark 规范 / 数据集计划
├── install_libs.sh               # 安装脚本（2d / 3d）
└── pyproject.toml                # 项目配置
```

---

## 路线图

| 版本 | 状态 | 内容 |
| --- | --- | --- |
| **v0.1–v0.2** | ✅ 完成 | 检测 + 分割 + 分类 + OBB + Web 审核 + Agentic 闭环 |
| **v0.3** | ✅ 完成 | 分类（CLIP/SigLIP/torchvision）+ OBB（YOLO-OBB）+ F4 质量评估 + LLM Evaluate + batch 动态调优 + cityscapes 域内权重 + 12 数据集 Benchmark（84 模型） |
| **v0.4** | ✅ 完成 | Tracking（ByteTrack/BoT-SORT + ReID + 指代 L1）+ KITTI 域内微调 + AgentState 续跑 + Web 三件套 |
| **v0.5** | ✅ 完成 | Pose（YOLO-pose）+ 指代 L2/L3（Florence-2/Qwen2-VL-7B）+ 自动车道 ROI + ILSVRC2012（top-1 0.6968）+ KITTI 微调闭环（0.8867） |
| **v0.6** | 📋 立项 2026-08-28 | 对话式 Agent 统一入口（chat 唯一入口 + LLM 多轮对话确定参数） |
| **auto3dlabel v0.1** | ✅ 2026-08-24 | 单帧 KITTI 五步管线（标定→反投影→聚类→拟合）+ Agentic/Web 闭环 |
| **auto3dlabel v0.2** | ✅ 2026-08-27 | mmdet3d PointPillars（KITTI Car moderate 82.0 超官方）+ Tracker3D + nuScenes Mini |
| **auto3dlabel v0.3** | 📋 立项 2026-08-28 | 对话式 Planner → PV-RCNN/CenterPoint 精度 → BEVFusion 融合 → Web 真 3D 复核 → LabelAny3D 验证 |

详细里程碑与实测数据见各包 `milestone/` 与 `tests/test-v0.X.md`。

---

## 已知限制

我们诚实地记录当前版本的限制和设计取舍：

- **域边界是硬约束**：COCO 预训练模型在特定域（MOT 密集行人、DOTA 航拍、cityscapes 小目标）精度低，GPU 复测定性为数据域边界（规模/SAHI/架构均无效）——出路是**域内微调回采闭环**（cityscapes 0.0082→0.5149、KITTI 0.8867 两例已闭环）
- **OBB 只支持 YOLO-OBB**：Oriented R-CNN 因 mmrotate 依赖重延后（取舍注记见计划书）
- **Benchmark 路径硬编码**：数据集路径指向 AutoDL 服务器特定目录（`~/autodl-tmp/Documents/datasets`、`/root/autodl-pub`），在其他环境需手动修改
- **LLM 在单任务中边际价值有限**：v0.1c 实验表明，在单一检测任务上 LLM Agent 与 `--no-llm` 结果相同，LLM 的价值体现在多任务编排、模糊指令处理、异常处理等场景
- **auto3dlabel 依赖 mmdet3d 硬装**：mmcv 2.1.0 需 CUDA 13 源码编译（`install_libs.sh 3d`），无 GPU 环境无法启用 LiDAR 直检引擎（反投影拟合引擎仍可 CPU 跑）
- **3D 反投影拟合路线精度天花板**：v0.1 实证 3D AP 近零——已用 LiDAR 直检引擎（moderate 82.0）为主，反投影仅作开放词汇回退
- **nuScenes 数据非商用**：KITTI/nuScenes 均为 CC BY-NC-SA 许可，商业化需自采/商用许可数据
- **API Key 安全**：`auto2dlabel/configs/.env` 中硬编码了默认 API Key，公开使用时请注意替换为环境变量

---

## 贡献

本项目目前为个人工程作品集项目。欢迎 Issue 和 PR。

开发环境设置：

```bash
pip install -e ".[dev]"
ruff check .          # Lint
mypy auto2dlabel/     # Type check
pytest auto2dlabel/tests/ -v  # Test
```

---

## 许可证

[MIT](LICENSE) © AutoLabel Team

---

<p align="center">
  <sub>Built with ❤️ by an AI-first perception engineer. 用 AI 重新定义标注。</sub>
</p>
