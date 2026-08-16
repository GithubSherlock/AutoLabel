<p align="center">
  <h1 align="center">AutoLabel</h1>
  <p align="center"><strong>Agentic 2D Image Annotation — AI-first, Human-in-the-loop</strong></p>
  <p align="center">自然语言驱动 · LLM Agent 编排 · 多模型引擎 · 多格式导出</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/models-64%2B-orange" alt="64+ Models">
  <img src="https://img.shields.io/badge/version-0.3.0-informational" alt="Version 0.3.0">
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

```
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
| **auto2dlabel** | ✅ v0.3 完成 | 2D 检测 + 分割 + 分类（CLIP/SigLIP）+ OBB + NL 交互 + Web 审核 |
| **auto3dlabel** | 📋 待实现 | 3D 点云标注（占位） |

---

## 特性

- **🗣️ 自然语言交互** — `chat` 命令支持中英文自然语言描述标注任务，LLM 自动提取参数（类别、阈值、模型），缺失参数交互追问
- **🧠 Agentic 编排** — LLM Agent 规划多步任务（检测 → 分割 → 导出），自动调用工具，token 高效设计（结果摘要注入，避免全量 bbox 回传）
- **🔧 3 引擎 × 64+ 模型** — Grounding DINO（开放词汇）、Ultralytics YOLO（YOLOv5–v12/v26/World/RT-DETR）、PyTorch Vision（Faster R-CNN/RetinaNet/SSD/FCOS）；分割支持 SAM/SAM2/SAM3/Mask R-CNN/FastSAM；分类 CLIP/SigLIP；OBB YOLO-OBB（n/s/m/l/x）
- **✅ HITL 置信度分流** — 三档阈值：高置信（≥0.7）直接接受 / 中置信（0.3–0.7）待人工审核 / 低置信（<0.3）难例队列
- **📦 多格式导出** — COCO JSON / YOLO txt / Pascal VOC XML / LabelMe JSON / cls JSON / DOTA / YOLO-OBB txt，已通过 round-trip 测试（坐标误差 < 1e-3）
- **🖼️ 零样本图像分类** — CLIP/SigLIP 多候选 softmax 排序取 top-K，`Annotation.labels` + cls JSON 导出
- **📐 OBB 旋转框** — YOLO-OBB 5 枚，`Bbox.angle` 弧度约定，DOTA 8 角点 / YOLO-OBB txt 导出，旋转多边形可视化
- **🔁 Batch 韧性 + 主动学习** — 单图失败隔离不中断整批 + `--resume` 断点续跑；`sample` 按不确定性排序聚合审核队列
- **⚡ `--no-llm` Baseline** — 绕过 LLM Agent，使用内置关键词映射直调检测模型，用于对比实验和离线场景
- **🔍 自动类别推荐** — 扫描全图 80 个 COCO 类别，按检出数量和置信度排序推荐 Top-K 类别
- **🌐 Web 审核界面** — FastAPI + Canvas SPA，支持模型选择、标注可视化、mask Canvas 叠加、结果筛选删除、复核队列（人工修正回流）、COCO JSON / 可视化下载
- **📊 Benchmark 套件** — 覆盖 COCO 2017 / VOC 2007 / KITTI / Cityscapes / nuImages / DOTA / D2SA / MOT17 / MOT20 共 9 个数据集，支持检测、分割与旋转框评测
- **📝 结构化日志** — 每次调用的完整 JSON 日志（LLM/Chat/PythonAPI），可复现、可审计

---

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA GPU（推荐）或 Apple Silicon MPS 或 CPU
- 约 10GB 磁盘空间（模型权重）

### 1. 安装

```bash
# 方式一：交互式安装脚本
bash install_libs.sh 2d

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
# 用 YOLOv8 nano 检测图片中的汽车和行人
auto2dlabel run photo.jpg "检测汽车和行人" -d yolov8n.pt -t 0.5

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
auto2dlabel run photo.jpg "检测汽车" -d yolov8x.pt -e yolo

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
python -m auto2dlabel.web.server
# 访问 http://localhost:8765
```

功能：上传图片 → 选择检测/分割模型 → 调整阈值 → 自动标注 → Canvas 可视化 → 点击删除误检 → 下载 COCO JSON。

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
```

**实际 Benchmark 结果（2026-08-06）**：

| 数据集 | 模型 | 指标 | 值 |
| --- | --- | --- | --- |
| COCO 2017 val (50 imgs) | yolov8x | mAP@0.5 | 0.453 |
| VOC 2007 (100 imgs) | yolo26x | mAP@0.5 | 0.592 |
| KITTI | yolo26x | mAP@0.5 | 0.441 |
| COCO seg (50 imgs) | yolov8x + FastSAM-s | mask mAP | 0.500 |
| Cityscapes seg | FastSAM-s | mask mAP | 0.007* |
| nuImages | yolo26x + FastSAM | mAP | 0.383 |

> \* Cityscapes 分数极低是因为 FastSAM 对该数据域过度分割，需换用 SAM3。

### 测试

```bash
pytest auto2dlabel/tests/ -v
# functional/test_export_roundtrip.py  — COCO/YOLO/VOC 导出→回读一致性
# functional/test_benchmark_metrics.py — IoU/AP 边界条件
# functional/test_no_llm_baseline.py   — no-LLM baseline / benchmark 工具
```

---

## 模型目录

### 检测模型（47 个）

| 引擎 | 数量 | 示例 | 特点 |
| --- | --- | --- | --- |
| **Grounding DINO** (HF) | 3 | `grounding-dino-tiny/base/large` | 开放词汇，文本 prompt 直出 bbox，无需预定义类别 |
| **Ultralytics YOLO** | 35 | YOLOv5–v12 n/s/m/l/x, YOLO-World, v26, RT-DETR | 统一 API，自动下载权重至 `auto2dlabel/weights/` |
| **PyTorch Vision** | 9 | Faster R-CNN, RetinaNet, SSD, SSDLite, FCOS | COCO 预训练，torchvision 内置 |

### 推荐场景

| 场景 | 推荐模型 | 理由 |
| --- | --- | --- |
| 自动标注（宁多勿漏） | `fasterrcnn_resnet50_fpn_v2` | 召回率最高，person 检出 16（vs YOLO 仅 6），置信度 85%+ |
| 快速预览 | `yolov8n.pt` / `yolo11n.pt` | 速度最快，适合快速扫图 |
| 均衡选择 | `yolov8x.pt` / `yolo26x.pt` | 精度与速度折中，VOC mAP@0.5 达 59.2% |
| 开放词汇 | `IDEA-Research/grounding-dino-tiny` | 无需预设类别，中文 prompt 直译后使用 |
| 卫星/航拍 | `yolo11n-obb.pt` | OBB 旋转框（v0.3 已支持） |

### 分割模型（8+ 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **SAM3** | Ultralytics (Meta) | ~3.4GB | 文本 prompt → 检测+分割一步，开放词汇，**最强** |
| **SAM 2.1** | Ultralytics (Meta) | 自动下载 | bbox prompt → mask，精度高 |
| **SAM** | Ultralytics (Meta) | ~160MB | bbox prompt → mask，轻量快速 |
| **FastSAM** | Ultralytics | ~140MB | bbox IoU 匹配，最轻量 |
| **Mask R-CNN** | torchvision | 自动下载 | COCO 预训练，检测+分割一步 |

**实测对比（000860.png，"car + person"，conf=0.1）**：

| 维度 | SAM | SAM3 | Mask R-CNN |
| --- | --- | --- | --- |
| 耗时 | 1.0s | 20.6s | 2.5s |
| 总 mask 数 | 14 | **41** | 36 |
| mask 精度（多边形点数） | 260 | **892** | 860 |

### 分类模型（2 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **CLIP** | transformers | openai/clip-vit-base-patch32 | 零样本分类，多候选 softmax 排序 |
| **SigLIP** | transformers | google/siglip-base-patch16-224 | sigmoid logits，softmax 归一化排序 |

### OBB 旋转框（5 个）

| 模型 | 引擎 | 权重 | 特点 |
| --- | --- | --- | --- |
| **YOLO-OBB** | Ultralytics | yolo11n/s/m/l/x-obb.pt | 旋转框检测，`dota` / `yolo_obb` 导出 |

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
├── auto2dlabel/                  # 主包
│   ├── cli.py                    # CLI 入口（Typer）
│   ├── agent/                    # Agentic 编排层
│   │   ├── orchestrator.py       # Agent Loop 编排器
│   │   ├── planner.py            # NL 任务规划器（chat 命令）
│   │   ├── llm.py                # LLM 客户端（DeepSeek/OpenAI/Anthropic）
│   │   └── state.py              # Agent 状态管理
│   ├── tools/                    # 工具层
│   │   ├── detection.py          # 检测工具
│   │   ├── segmentation.py       # 分割工具
│   │   ├── export.py             # 导出工具
│   │   ├── recommend.py          # 类别推荐
│   │   ├── hitl.py               # HITL 置信度分流
│   │   ├── batch.py              # batch 清单 / --resume
│   │   ├── evaluate.py           # LLM Evaluate 工具（不进 registry）
│   │   ├── sampling.py           # 主动学习采样
│   │   ├── visualize.py          # 可视化
│   │   ├── log.py                # 结构化日志
│   │   └── confirm.py            # 交互确认
│   ├── models/                   # 模型层
│   │   ├── model_catalog.py      # 模型目录（49+ 检测 + 8 分割 + 2 分类 + 5 OBB）
│   │   ├── detection.py          # 检测模型工厂
│   │   ├── segmentation.py       # 分割模型工厂
│   │   ├── classification.py     # 分类模型（CLIP/SigLIP）
│   │   └── obb.py                # OBB 模型（YOLO-OBB）
│   ├── schema/                   # 数据 Schema
│   │   ├── annotation.py         # Bbox / Mask / ImageLabel / Annotation
│   │   └── task_plan.py          # 任务计划
│   ├── export/                   # 导出器（COCO / YOLO / VOC / LabelMe / cls / DOTA / YOLO-OBB）
│   ├── web/                      # Web 审核界面（FastAPI + Canvas，mask 叠加 + 复核队列）
│   ├── benchmarks/               # Benchmark 套件（9 数据集）
│   ├── milestone/                # 里程碑定义
│   ├── configs/.env              # 配置文件
│   └── weights/                  # 模型权重目录
├── auto3dlabel/                  # 3D 标注（待实现）
├── auto2dlabel/tests/            # 测试（functional/ 用例 + helpers/ 工具）
├── AutoLabel_plan.md             # 项目设计文档
├── install_libs.sh               # 安装脚本
└── pyproject.toml                # 项目配置
```

---

## 路线图

| 版本 | 状态 | 内容 |
| --- | --- | --- |
| **v0.1a** | ✅ 完成 | Object Detection + 3 引擎 × 47 模型 → COCO/YOLO/VOC 导出 + Agent Loop |
| **v0.1b** | ✅ 完成 | Instance Segmentation（SAM/SAM2/Mask R-CNN/FastSAM）+ `chat` 命令 |
| **v0.1c** | ✅ 完成 | HITL 置信度分流 + `--no-llm` Baseline + Round-trip 测试 + LLM 价值量化实验 |
| **v0.2** | ✅ 完成 | SAM3 + 类别推荐 + Web 审核界面（FastAPI + Canvas） |
| **v0.3** | ✅ 完成 | 分类（CLIP/SigLIP）+ OBB（YOLO-OBB）+ M2 mask 叠加/复核队列闭环 + M3 Evaluate 节点/batch 续跑/主动学习采样 |
| **v0.4** | 📋 计划中 | Pose Estimation（ViTPose/RTMPose） |
| **v1.0** | 📋 计划中 | Object Tracking（视频时序）+ 3D 点云（auto3dlabel） |

---

## 已知限制

我们诚实地记录当前版本的限制和设计取舍：

- **Batch 仍为串行循环**：单图失败已隔离（不中断整批）、支持 `--resume` 断点续跑，但无并发处理；checkpoint 仅写 `to_dict` 快照，`from_dict` 恢复未实现
- **OBB 只支持 YOLO-OBB**：Oriented R-CNN 因 mmrotate 依赖重延后；DOTA Task1 旋转 GT 评测挂起（`rotate_iou` 指标已就绪，GT 归档未验证到）
- **Web 审核无拖拽编辑**：bbox 拖拽/标签编辑、分类与 OBB 结果的 Web 展示列入后续版本
- **Benchmark 路径硬编码**：数据集路径指向 AutoDL 服务器特定目录（`~/autodl-tmp/Documents/datasets`、`/root/autodl-pub`），在其他环境需手动修改
- **LLM 在单任务中边际价值有限**：v0.1c 实验表明，在单一检测任务上 LLM Agent 与 `--no-llm` 结果相同，LLM 的价值体现在多任务编排、模糊指令处理、异常处理等场景
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
