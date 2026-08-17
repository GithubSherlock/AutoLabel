# Benchmark 计划书

AutoLabel 的 Benchmark 套件是 **自动标注产物 vs 官方 GT** 的量化评测体系，用 mAP@0.5 / mask mAP / mIoU / mDice 等指标证明「AI 为主、人为辅」自动标注的质量主张，为模型选型与回归对比提供数据依据。

> 本文档是「计划 + 程序规范」双重文档：前半部分镜像 `AutoLabel_plan.md` 的章节结构，描述 Benchmark 计划；末尾 **「Benchmark 程序规范」** 是智能体创建/修改 Python benchmark 脚本的可执行依据。实测数据一律写入 `auto2dlabel/tests/test-v0.X.md`，本文档只引用不收录。

## 项目背景

AutoLabel 的核心主张是「AI 为主、人为辅」（见 `AutoLabel_plan.md` 项目背景）：用户下达自然语言指令，Agent 自主规划并调用模型生成预标注。这个主张要成立，自动标注质量必须经得起量化检验——**与官方 GT 对比评测**（而非口头宣称或目测）是唯一可信的证明。Benchmark 套件从 v0.1 落地（9 数据集，见 `AutoLabel_plan.md` 版本路线），并随版本持续扩展。

评测三原则：

- **公平**：统一基线 conf=0.3 / IoU=0.5（`auto2dlabel/schema/task_plan.py:32-36` 的 `BENCHMARK_DEFAULT_*` 常量）；GT 类名直接作检测 prompts，不手工挑图、不调参作弊
- **可复现**：`--max-images N` 冒烟 → 全量两级跑法，任何一次运行可重放
- **可追溯**：每次运行落 JSON + Markdown 双产物于 `benchmarks_outputs/`（时间戳命名、独立快照），批量跑再落 `summary_{ts}.md` 汇总

## 总纲

**Benchmark 套件 = 10 数据集自动标注 vs GT 评测**，全流程管线：

```text
数据准备：ensure_*（幂等解压，DATASETS_ROOT=~/autodl-tmp/Documents/datasets，归档 ARCHIVE_ROOT=/root/autodl-pub）
    │
GT 解析：load_*_ground_truth → {image_id: {file_name, objects: [{name, bbox[x1,y1,x2,y2] | mask}]}}
    │
模型推理：检测 = create_detection_model(model).detect(img, GT 类名作 prompts, conf)
    │        分割 = 检测 conf≥0.5 → seg_model.generate(box-prompt)（D2SA：GT bbox 直接作 prompt，无检测步）
    │        大图 = run_detection_sahi（切片 640px + 跨 tile NMS，kitti/dota/mot 支持）
    │        分类 = create_classification_model(model).classify(img, candidates, top_k)
    │
逐类评测：evaluate_per_class / evaluate_mask_per_class / evaluate_classification
    │        （每类 IoU 贪心匹配 + 11 点插值 AP → precision/recall/f1/ap/gt_count/pred_count，mask 另计 mIoU/mDice）
    │
报告：console 表（format_result_table / format_mask_result_table / format_classification_table，
    │   按 AP 降序 + 尾部 mAP 汇总行）
    │   → save_results 写 {dataset}_{model}_{ts}.json（MD 由脚本自行 write_text）
    └──→ run_all 聚合 summary_{ts}.md
```

- 检测与分割共用同一套「GT → prompt → 推理 → 逐类评测」骨架，仅推理步与指标函数不同
- 评测粒度是**逐类**而非逐图：`evaluate_per_class`（`common.py:80`）的 `gt_count/pred_count` 反映该类样本量，小样本类虚高可被察觉
- 每次运行是独立快照（时间戳命名），不做增量覆盖——支持跨版本回归对比
- `rotate_iou`（`common.py:44`，shapely，DOTA 8 点四边形 IoU）已接入 OBB benchmark（`evaluate_per_class(iou_fn=rotate_iou)`，见 dota_obb_benchmark.py）

## Benchmark 模块划分

总体按**三大任务模块**组织，各模块共享同一套「GT → 推理 → 逐类评测 → 报告」管线（总纲图），仅数据集、推理步与指标函数不同。数据集明细与磁盘就绪状态见 `datasets_plan.md` 就绪状态总表。

| 模块 | 模型（评测对象） | 数据集 | 指标 | 脚本 | 状态 |
| --- | --- | --- | --- | --- | --- |
| **分类** | CLIP / SigLIP 零样本 + torchvision 14 款监督（ImageNet1K） | ImageNet100（主力，已就绪）、ILSVRC2012 val（重载）、cifar-10/100（冒烟）、CUB200-2011（细粒度） | top-1 / top-5 准确率 | classification_benchmark.py | ✅ ImageNet100 就绪，其余 📋 规划 |
| **目标检测** | 3 引擎 × 29 模型（默认 yolo26x.pt） | COCO2017 / VOC2007 / KITTI / DOTA(HBB) / MOT | mAP@0.5 + 逐类 P/R/F1/AP | coco/voc/kitti/dota/mot 5 脚本 | ✅ 就绪 |
| **旋转框检测（OBB）** | yolo11/12/26 n/s/m/l/x-obb.pt（默认 yolo11n-obb.pt） | DOTA Task1（15 类全评，rotate IoU） | mAP@0.5 + 逐类 P/R/F1/AP（`evaluate_per_class(iou_fn=rotate_iou)`） | dota_obb_benchmark.py | ✅ 就绪（CPU 实测见 test-v0.3.md） |
| **实例分割** | FastSAM（box-prompt，默认）/ SAM2 / SAM3 / Mask R-CNN | COCO-seg / Cityscapes / nuImages / D2SA | mask mAP@0.5 + mIoU + mDice | coco_seg/cityscapes/nuimages/d2sa 4 脚本 | ✅ 就绪 |

### 分类模块（分体说明）

- **双路评测**：零样本路（CLIP/SigLIP，中文候选可直接用）与监督路（torchvision 14 款，候选须英文、子串过滤）在相同 GT 上分别评测，天然可比
- **GT 形式**：目录（wnid）或标注文件即标签，无 bbox；ImageNet100 为 ImageNet1K 子集，与 torchvision 权重同源，是监督分类的直接评测集；wnid → 英文名经 devkit `ILSVRC2012_devkit_t12.tar.gz` 的 `data/meta.mat`（**非 map_clsloc.txt，两 devkit 均无此文件**）转出
- **脚本要点**：沿用 §10.1 骨架，但无检测步——`load_xxx_ground_truth` 返回 `{image_id: {file_name, label}}`；指标 `evaluate_classification` / `format_classification_table` 已落地 `common.py`；cifar 仅作指标链路冒烟（32×32 需 resize，不作质量结论）
- **评测协议差异（报告与实测数据须注明）**：监督路径在完整 ImageNet1K 1000 类空间取 top-K（官方协议），零样本路径在 100 类候选空间取 top-K——两者 top-1/top-5 不可直接横向比较

### 目标检测模块（分体说明）

- 5 数据集覆盖：通用场景（COCO/VOC）、驾驶（KITTI）、航拍（DOTA HBB，仅 4/15 类可评）、密集行人（MOT 单类 person）
- 大图场景（kitti/dota/mot）支持 `--sahi` 切片推理；DOTA 旋转框另走 OBB 模块（Task1 15 类全评 + rotate_iou，已落地）

### 实例分割模块（分体说明）

- 统一「检测 conf≥0.5 → box-prompt 分割」两段式，D2SA 例外（box-prompted 无检测步）
- 域难度阶梯：COCO（通用）→ nuImages（驾驶）→ Cityscapes（实测 mAP 0.007*，FastSAM 域差距最大）→ D2SA（密集零售）

## Target User（目标用户）

- **第一读者 = 智能体（Agent）**：Claude Code Agent 按本文档末尾「Benchmark 程序规范」创建/修改 benchmark 脚本、维护 bash 一键运行——文档措辞是可执行规范，而非愿景描述
- **第二读者 = 算法工程师**：读 `auto2dlabel/tests/test-v0.X.md` 与 `benchmarks_outputs/` 的实测数据，判断模型域差距（如 VOC 的 yolo26x 59.2% vs FRCNN 24.5%、Cityscapes 0.007* 的域问题），据此做模型选型（`auto2dlabel/CLAUDE.md` 模型选型速查表直接引用这些实测）
- **不是面向**：最终标注产品的使用者（benchmark 是工程验证工具，不是产品功能）

## 项目范围

评测对象 = **公开数据集 × 自动标注 pipeline**，按三大任务模块组织：目标检测 5 + 实例分割 4 + 分类 ImageNet100（已就绪，另 3 个分类数据集规划中，见「Benchmark 模块划分」）。模型侧默认值由 `schema/task_plan.py:34-35` 统一（检测 `yolo26x.pt`、分割 `FastSAM-s.pt`），单脚本可 `--model`/`--seg-model` 覆盖。

**分类模块（ImageNet100 已就绪，其余规划）**：

| 数据集 | key | 任务 | 类别 → 评测类 | 数据来源（archive） | 脚本 | 特殊点 |
| --- | --- | --- | --- | --- | --- | --- |
| ImageNet100 | imagenet100 | 分类 | 100 → 100 | `ImageNet100/imagenet100.zip`（14GB，每类均匀抽样解压） | `classification_benchmark.py` | ✅ 就绪；ImageNet1K 子集，与 torchvision 权重同源；wnid 目录即标签；`--per-class` 抽样解压 + `--top-k` |
| ILSVRC2012 val | imagenet_val | 分类 | 1000 → 1000 | `ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar`（6.7GB）+ devkit GT | 待建 | 完整重载；CLIP 零样本 + torchvision 监督双路 |
| cifar-10 / cifar-100 | cifar10 / cifar100 | 分类 | 10 / 100 | `cifar-10/`、`cifar-100/`（各 ~170MB） | 待建 | 32×32 仅作指标链路冒烟（resize 后不作质量结论） |
| CUB200-2011 | cub200 | 分类 | 200 | `CUB200-2011/CUB_200_2011.tgz`（1.1GB） | 待建 | 细粒度，检验零样本 CLIP 上限 |

**目标检测模块（已就绪）**：

| 数据集 | key（run_all / schema） | 任务 | 类别 → 评测类 | 数据来源（archive） | 脚本 | 特殊点 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO 2017 val（5,000 图） | coco2017 / coco | 检测 | 80 → 80 | `COCO2017/val2017.zip` + 标注 | `coco_benchmark.py` | ⚠ 路径硬编码不调 `ensure_*`（缺口 5） |
| PASCAL VOC 2007 test（4,952 图） | voc2007 | 检测 | 20 → 20 | `VOCdevkit/VOC2007.tar.gz` | `voc_benchmark.py` | ⚠ 同上；`VOC_TO_COCO`（voc_benchmark.py:43）定义未使用（缺口 7） |
| KITTI object（截取 300 图） | kitti | 检测 | 8 → 7（Car/Van→car 等映射，Misc/DontCare 跳过） | `KITTI/object/data_object_image_2.zip` + `label_2.zip` | `kitti_benchmark.py` | `ensure_kitti`（datasets.py:135）按标注筛选只解压前 N 图；支持 `--sahi` |
| DOTA v1.0 val（458 图） | dota | 检测（HBB） | 15 → **4**（plane→airplane 等映射） | `DOTA/val/images/part1.zip` + `Val_Task2_gt.zip` | `dota_benchmark.py` | `SKIPPED_CLASSES`（dota_benchmark.py:46）跳过 11 类无 COCO 对应；支持 `--sahi`；旋转 GT（Task1）未接入（缺口 1） |
| MOT17-FRCNN + MOT20 | mot | 检测 | 单类 person | 已解压，仅路径解析 | `mot_benchmark.py` | `MOT_CLASS_MAP={1:"person",7:"person"}`（mot_benchmark.py:37）；`--max-images` 跨序列均匀采样帧；支持 `--sahi` |

**实例分割模块（已就绪）**：

| 数据集 | key（run_all / schema） | 任务 | 类别 → 评测类 | 数据来源（archive） | 脚本 | 特殊点 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO 2017 val 实例分割 | coco_seg | 分割 | 80 → 80 | 同 COCO | `coco_seg_benchmark.py` | 检测 conf≥0.5 → FastSAM box-prompt；RLE/polygon 双格式 GT |
| Cityscapes val（504 图） | cityscapes | 分割 | 8 个 thing 类 | `gtFine_trainvaltest.zip` + `leftImg8bit_trainvaltest.zip`（仅 val 分片） | `cityscapes_benchmark.py` | instanceIds.png 解析（`THING_CLASSES`，cityscapes_benchmark.py:33）；实测 mAP 0.007* 域差距（见 test-v0.2.md 相关讨论） |
| nuImages Mini（50 图） | nuimages | 分割 | 23 → 6（层级名折叠） | `nuScenes/nuImages/Mini/nuimages-v1.0-mini.tgz` | `nuimages_benchmark.py` | `NUIMAGES_TO_COCO`（nuimages_benchmark.py:33）层级映射（vehicle.car→car 等）；RLE mask；nuscenes-devkit 依赖 |
| D2SA val（60 SKU） | d2sa | 分割 | 60 → **27 超类** | `D2SA/annotations.tar.gz` + `images.tar.gz` | `d2sa_benchmark.py` | **box-prompted 无检测步骤**（GT bbox 直接作 FastSAM prompt，d2sa_benchmark.py:102）；按 supercategory 聚合 |

指标族：

- **分类指标**（已落地）：top-1 / top-K 准确率（`evaluate_classification` + `format_classification_table`，common.py）+ 每类 correct / total / accuracy
- **bbox 指标**：mAP@0.5（逐类 11 点插值 AP 均值）+ 每类 precision / recall / f1 / ap / gt_count / pred_count
- **mask 指标**：mask mAP@0.5 + mIoU + mDice + matched_count（在 bbox 指标族基础上扩展）

**Out of Scope（明确不做）**：

- OBB 旋转框评测：`rotate_iou` 就绪、DOTA Task1 旋转 GT 已定位在归档（labelTxt.zip），脚本未写——待排期（见缺口表）
- Pose / Tracking 基准：随主项目 v0.4 / v1.0 推进（见版本路线）
- 模型训练 / 微调：benchmark 只评估现有权重，不做调优实验
- 自定义数据集评测：须先按「程序规范 §10.3」走数据集接入流程，非默认范围
- no-LLM 对照实验：属 `tests/helpers/no_llm_baseline.py` 的 LLM vs no-LLM 对比工具（M1c），**不算**数据集基准

## 版本路线

Benchmark 版本随主项目版本对齐，不单独立版本号。

| 主项目版本 | Benchmark 里程碑 | 状态 |
| --- | --- | --- |
| **v0.1** | 9 数据集基准落地（检测 5 + 分割 4）：mAP@0.5 / mask mAP 基线，VOC/BDD100K 域差距实测 | ✅ 完成（实测见 `tests/test-v0.1.md`） |
| **v0.2** | 无独立增量（分割模型实测对比随主版本，见 `tests/test-v0.2.md`） | ✅ 完成 |
| **v0.3** | 分割基准扩展至 9 数据集 + `rotate_iou`（shapely）就绪 + SAHI 切片推理（kitti/dota/mot） | ✅ 完成 |
| **v0.3 补课** | 分类基准落地：ImageNet100 / ILSVRC2012 val（CLIP/SigLIP 零样本 + torchvision 14 款监督，top-1/top-5） | 🔄 ImageNet100 ✅（CPU 可行性验证见 `tests/test-v0.3.md`），ILSVRC2012 val 📋 |
| **v0.4** | Pose 基准：COCO keypoints（AP / OKS 类指标） | 📋 规划 |
| **v1.0** | Tracking 基准：MOT17 / MOT20 时序（MOTA / IDF1 / MT） | 📋 规划 |

## 里程碑状态

| 里程碑（主项目编号） | Benchmark 内容 | 状态 |
| --- | --- | --- |
| **M1 单流程跑通** (v0.1a) | F9 Benchmark 落地：`common.py` + `datasets.py` + `run_all.py` + 9 脚本，JSON/MD 双产物输出至 `benchmarks_outputs/` | ✅ 完成 |
| **M1c 价值证明** (v0.1c) | no-LLM 对照实验（`tests/helpers/no_llm_baseline.py`，LLM vs no-LLM 对比，非数据集基准） | ✅ 完成 |
| **M3 Agentic 闭环** | benchmark 可经自然语言触发：`is_benchmark_intent`（benchmark_runner.py:27）意图识别 → `execute_benchmark`（benchmark_runner.py:68）subprocess 执行 | ✅ 完成 |
| **v0.3 补课** | 分类基准：ImageNet100 / ILSVRC2012 val + `evaluate_classification` 入 common.py | 🔄 ImageNet100 ✅，ILSVRC2012 val 📋 |
| **v0.4 / v1.0** | Pose / Tracking 基准 | 📋 规划 |

## 当前状态（2026-08）

| 项 | 状态 |
| --- | --- |
| **Benchmark 套件** | 检测 5 + OBB 1 + 分割 4 + 分类 ImageNet100 共 11 数据集就绪（`auto2dlabel/benchmarks/` 15 文件）；另 3 个分类数据集归档已就绪待接入（见 `datasets_plan.md`）；双入口：`python -m auto2dlabel.benchmarks.run_all` 一键 + 单脚本直跑；`bash auto2dlabel/benchmarks/run_benchmarks.sh` 一键包装（all/detection/segmentation/classification/obb 分组，见程序规范 §10.8）；纯 CPU 全量可行性已验证（见 `tests/test-v0.3.md`） |
| **实测数据** | 在 `auto2dlabel/tests/test-v0.1.md`、`test-v0.2.md`（VOC 100 图 yolo26x mAP 0.592 等）；完整产物在 `benchmarks_outputs/`（不入库） |
| **质量门** | pyright 0 / mypy strict ≤ 基线 / ruff ≤ 基线 / pytest 全绿（每版本硬性门槛，数字基线以 `AutoLabel_plan.md` 当前状态为准） |
| **已知不一致** | 7 项缺口见下节（分类基准缺失、docstring 过期、双注册键名不一致等） |

## 已知缺口 / 待办

| 缺口 | 位置 | 归入 |
| --- | --- | --- |
| **分类数据集接入不全**：ImageNet100 已闭环（脚本 + `ensure_imagenet100` 抽样解压 + 双注册 + 实测）；ILSVRC2012 val / cifar / CUB200 归档已就绪但未接入（`classification_benchmark.py --dataset` choices 可扩） | `datasets_plan.md` 就绪总表 | v0.3 补课 |
| ~~**OBB 旋转框评测未实现**~~ ✅ 已闭合：`dota_obb_benchmark.py`（Task1 旋转 GT 解析 + yolo-obb 推理 + `rotate_iou` 评估 + 三注册）；CPU 实测 50 图 mAP 0.7047（见 `tests/test-v0.3.md`）；obb 类名映射 bug 一并修复（DOTA_CLASSES + 别名兼容） | `dota_obb_benchmark.py` | 已闭合 |
| ~~**docstring 过期**~~ ✅ 已修复：`run_all.py:5` 与 `datasets.py:558` `ensure_all` 均已更新为 10 数据集 | `run_all.py:5`、`datasets.py:558` | 已闭合 |
| **双注册键名不一致**：`schema/task_plan.py:39` 用 `"coco"`，`run_all.py:21` 用 `"coco2017"` | `task_plan.py:39`、`run_all.py:21` | 低优先级一致性 |
| **数据集路径硬编码**：`DATASETS_ROOT`/`ARCHIVE_ROOT` 指向 AutoDL 服务器特定目录，README:407 已注明 | `common.py:22`、`datasets.py:12-13` | 低优先级一致性 |
| **coco/voc 不调 `ensure_*`**：`coco_benchmark.py:28`、`voc_benchmark.py` 路径硬编码，与其余 7 脚本不一致 | `coco_benchmark.py:28` | 低优先级一致性 |
| **`VOC_TO_COCO` 未使用**：定义了别名映射但推理直接用 `VOC_CLASSES` | `voc_benchmark.py:43` | 低优先级一致性 |
| **无 Pose / Tracking 基准**：v0.4 / v1.0 规划（见版本路线表） | `task_plan.py:39` | 后续版本 |

## 下一步

| 优先级 | 事项 | 内容 |
| --- | --- | --- |
| 1 | 分类数据集扩展 | ✅ ImageNet100 已闭环（`evaluate_classification` 入 common.py + `ensure_imagenet100` 抽样解压 + wnid 映射 + 双路脚本 + 双注册，CPU 全量实测见 `tests/test-v0.3.md`）；下一步：ILSVRC2012 val 接入（`--dataset imagenet_val`）→ cifar 冒烟 |
| 2 | 一致性缺口消化 | 修复 2 处 docstring 过期、统一 schema/run_all 键名（coco2017 ↔ coco）、coco/voc 接入 `ensure_*`、清理 `VOC_TO_COCO` |
| 3 | DOTA Task1 旋转评测 | ✅ 已闭环：`dota_obb_benchmark.py` + obb 组注册（`run_benchmarks.sh obb`）+ CPU 实测（test-v0.3.md）；GPU 全量命令见 test-v0.3.md「GPU 复测清单」 |
| 4 | v0.4 Pose / v1.0 Tracking 基准 | 随主项目版本推进（见版本路线表）；Tracking 可复用已就绪的 MOT17/MOT20 数据 |
| 5 | 实测数据沉淀 | 新实验按程序规范 §10.7 写入 `tests/test-v0.X.md`；必要时 `run_all` 全量重跑刷新 `benchmarks_outputs/` |

---

## Benchmark 程序规范

> 本节是智能体创建/修改 benchmark 脚本的**直接依据**。规范以现有实现为准绳（现网代码即文档的镜像）；修改公共层（`common.py` / `datasets.py` / `run_all.py`）时必须同步本节。

### 10.1 新增 benchmark 脚本的标准流程与骨架

新脚本按五段式骨架组织（对齐现有脚本 `# ==== 1. GT 加载 / 2. 推理 / 3. 主流程 ====` 注释结构，参考 `dota_benchmark.py`）：

1. **文件头 docstring**：数据集 + 任务类型 + GT 格式 + 特殊点（参考 `d2sa_benchmark.py:1-10`）
2. **配置段**：`CONFIG` dict / 类映射 dict（如 `KITTI_TO_COCO`、`DOTA_TO_COCO`、`NUIMAGES_TO_COCO`）/ `SKIPPED_CLASSES`（参考 `dota_benchmark.py:35-50`）
3. **`load_xxx_ground_truth(max_images)`**：返回统一 `{image_id: {file_name, objects: [{name, bbox[x1,y1,x2,y2] | mask}]}}`；COCO 系 `[x,y,w,h]` 须转 `[x1,y1,x2,y2]`（`coco_benchmark.py:45`）
4. **推理函数**：检测 = `create_detection_model(model).detect(...)`（GT 类名作 prompts，`coco_benchmark.py:93`）；分割 = 检测 conf≥0.5 → `seg_model.generate(img, bbox 列表)`（`coco_seg_benchmark.py:83`）；box-prompted 模式直接 GT bbox（`d2sa_benchmark.py:102`）
5. **`main()`**：parser → `ensure_*` → 打印设备 → GT 加载 → 推理 → 逐类 `evaluate_per_class`/`evaluate_mask_per_class` → `format_result_table`/`format_mask_result_table` → `save_results` + `md_path.write_text`

**骨架硬性要点**（智能体必须遵守）：

- **bootstrap 双模式**：文件顶部固定 `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` + `sys.path.insert(0, str(PROJECT_ROOT))`，先于一切包导入（参考 `coco_benchmark.py:14-17` 模式）——同一脚本同时支持 `python -m auto2dlabel.benchmarks.xxx`（包内）与 `python auto2dlabel/benchmarks/xxx.py`（直接运行；`run_all.py:76-81` 与 `benchmark_runner.py` 均用后者）
- **库导入约定**：`from auto2dlabel.benchmarks import datetime, json, np, time`（`benchmarks/__init__.py` 集中再导出，满足 mypy strict no-implicit-reexport）
- **parser**：优先复用 `build_parser`（common.py:414，标准 6 参 `--max-images/--conf/--model/--iou/--top-classes/--sahi`）；分割脚本追加 `--seg-model`（默认 `FastSAM-s.pt`）。⚠ coco/voc 手写 argparse 属历史遗留（缺口 5），新代码一律用 `build_parser`
- **评测**：逐类循环，IoU 阈值统一用 `IOU_MATCH_THRESHOLD`（common.py:23 = 0.5），不得脚本内重复定义（`coco_benchmark.py:38` 的重复定义为反面教材）
- **MD 报告**：H1 数据集名 + 摘要行（`- **模型**: ... | **conf**: ... | **mAP@0.5**: ...`）+ summary 代码块；分割版指标行换 `mask mAP@0.5 | mIoU | mDice`
- **冒烟先行**：新脚本先 `--max-images 2` 验证跑通，再全量
- **分类脚本特殊点**（v0.3 补课，ImageNet100 已落地）：GT 为 `{image_id: {file_name, label}}`（目录/标注文件即标签，无 bbox）；指标 `evaluate_classification` + `format_classification_table`（common.py，summary 键 `{"top-1", "top-K"}`）；CLIP/SigLIP 零样本与 torchvision 监督两路共用同一 GT 加载，模型差异只在推理函数；`--per-class` 每类均匀抽样解压（`extract_zip_members_uniform`，datasets.py）；**协议差异**：监督 `candidates=[]`（1000 类空间 top-K）vs 零样本 `candidates=类名`（100 类候选空间）——报告须注明；产物扩展键 `seconds_per_image` / `images_per_second`（§10.4 允许）

### 10.2 必须复用的 common.py 函数清单

| 函数 / 常量 | 位置 | 用途 |
| --- | --- | --- |
| `PROJECT_ROOT` / `OUTPUT_DIR` / `DATASETS_ROOT` / `IOU_MATCH_THRESHOLD` | `common.py:20-23` | 路径常量与统一 IoU 阈值 |
| `compute_iou` | `common.py:30` | bbox IoU（`evaluate_per_class` 内部使用） |
| `rotate_iou` | `common.py:44` | OBB 四边形 IoU（shapely；dota_obb benchmark 经 `evaluate_per_class(iou_fn=rotate_iou)` 接入） |
| `evaluate_per_class` | `common.py:80` | 全部检测基准的逐类指标（贪心匹配 + 11 点 AP） |
| `polygon_to_mask` / `coco_seg_to_mask` / `rle_to_mask` / `nuscenes_rle_to_mask` | `common.py:161/177/192/208` | 分割 GT 光栅化（多边形 / COCO RLE / nuScenes RLE） |
| `compute_mask_iou` / `compute_dice` | `common.py:217/226` | mask 匹配与 mIoU/mDice |
| `evaluate_mask_per_class` | `common.py:235` | 全部分割基准的逐类指标 |
| `format_result_table` / `format_mask_result_table` | `common.py:324/356` | console 报告（AP 降序 + 尾部 mAP 汇总行） |
| `save_results` | `common.py:392` | JSON 落盘（MD 由脚本自行 write_text） |
| `build_parser` | `common.py:414` | 标准 CLI |
| `run_detection_sahi` | `common.py:585` | 大图切片推理 + 跨 tile NMS（dota/kitti/mot；复用 models/detection.py 的 SAHI helper） |
| `detect_image_sahi` / `sahi_infer_yolo` / `sahi_infer_torchvision` / `nms_per_class` | `models/detection.py:564/450/476/519` | SAHI 单图推理已迁出 common.py（供标注 pipeline 复用；common.py 顶层引用） |

**红线**：新增通用逻辑优先进 `common.py`；新脚本**禁止**复制以上任一函数实现。

### 10.3 数据集接入规范（ensure_* 幂等）

- 常量：`DATASETS_ROOT = ~/autodl-tmp/Documents/datasets`（`datasets.py:12`，与 `common.py:22` 同源）、`ARCHIVE_ROOT = /root/autodl-pub`（`datasets.py:13`）
- **幂等原则**：`extract_zip_members`（datasets.py:16）/ `extract_tar_members`（datasets.py:52）按缺失文件检查跳过已解压项；每个 `ensure_xxx()` 开头做快速幂等检查
- **增量原则**：大 archive 只提所需——`member_prefix`（cityscapes 仅 val 分片）、`max_files`（KITTI 仅前 300 图）、按标注文件名筛选解压
- **两类 ensure_**：真解压型（coco/voc/kitti/cityscapes/nuimages/dota/d2sa）与**路径解析型**（MOT17/MOT20 已解压，仅存在性检查 + FileNotFoundError，`datasets.py:356/375`）
- **新增数据集三步走**：① 写 `ensure_xxx()` 幂等解压 → ② 写 `load_xxx_ground_truth()` 格式解析 → ③ 双注册（见 §10.5）
- **一致性要求**：新脚本**必须**调 `ensure_*`（coco/voc 历史硬编码为已知缺口，禁止复刻）；`ensure_all`（datasets.py:398）保持与注册表同步

### 10.4 输出约定与 JSON schema

- 命名与位置：`{dataset}_{model}_{YYYY-MM-DD-HH-MM-SS}.json` 与同名 `.md` → 仓库根 `benchmarks_outputs/`（`save_results`，common.py:392）
- **JSON 统一键**：

| 键 | 类型 | 说明 |
| --- | --- | --- |
| `timestamp` | str | 运行时间戳 |
| `dataset` | str | 数据集显示名 |
| `image_count` | int | 实际评测图数 |
| `model` / `seg_model` / `det_model` | str | 三选一：检测脚本用 `model`；纯 box-prompt 分割用 `seg_model`；「检测→分割」两段式用 `det_model` + `seg_model` |
| `confidence_threshold` | float | 检测置信度 |
| `iou_match_threshold` / `mask_iou_threshold` | float | bbox / mask 匹配阈值 |
| `summary` | dict | 检测 `{"mAP@0.5"}`；分割 `{"mask mAP@0.5", "mIoU", "mDice"}` |
| `per_class` | dict | `{cls: {precision, recall, f1, ap, gt_count, pred_count[, miou, mdice, matched_count]}}`（mask 版多 3 键） |

- **允许扩展键**：`class_mapping`（dota）、`person_count`（mot）、`instance_count`/`supercategory_count`/`mode`（d2sa）——扩展键不得与统一键重名
- **不可变约定**：时间戳命名保证每次运行独立快照，禁止覆盖写同名文件；JSON 用 `ensure_ascii=False` 序列化

### 10.5 双注册要求（run_all + schema/task_plan）

新数据集必须**两处同时注册**，缺一处即链路断裂：

| 注册表 | 位置 | 内容 | 消费方 |
| --- | --- | --- | --- |
| `BENCHMARKS` | `run_all.py:19-69` | `{key: {script, model, default_args}}`，subprocess 直跑 + 3600s 超时 | `run_all`（含 `--only/--extra` CLI） |
| `BENCHMARK_DATASETS` | `task_plan.py:39` | `{key: {script, task_type}}` | Agent 链路 `tests/helpers/benchmark_runner.py:68`（超时 7200s，只查此表） |
| `DATASET_CN_MAP` | `task_plan.py:54` | 中文别名 → key（如 航拍→dota、密集行人→mot） | 自然语言意图识别 |

- **键名一致性红线**：两处 key 必须完全一致（现状 `"coco"`（task_plan.py:39）vs `"coco2017"`（run_all.py:21）不一致 = 缺口 3，新注册不得再犯）
- 触发链路（供理解）：自然语言 → `is_benchmark_intent`（benchmark_runner.py:27）→ `BenchmarkRequest`（task_plan.py:203）→ `to_cli_args()`（task_plan.py:247）→ subprocess 脚本

### 10.6 质量门

| 门 | 命令 | 标准 |
| --- | --- | --- |
| 静态检查 | `pyright` / `mypy --strict` / `ruff check` | 0 errors（`pyproject.toml` 已配） |
| 单元测试 | `pytest auto2dlabel/tests/` | 全绿（`test_benchmark_metrics.py` 覆盖 IoU/AP 边界） |
| 冒烟 | `python auto2dlabel/benchmarks/xxx.py --max-images 2` | 跑通且 JSON 可 `json.loads`、MD 含摘要行 |
| 产物检查 | `ls benchmarks_outputs/` | `{dataset}_{model}_{ts}.json/.md` 成对出现 |

- 全量跑不设时间门槛，由 `run_all` 3600s 与 `execute_benchmark` 7200s 超时兜底
- 数字基线以 `AutoLabel_plan.md` 当前状态为准（pyright 0 / mypy 172 / ruff 163 / pytest 全绿），每版本硬性门槛

### 10.7 实测数据写入 tests/test-v0.X.md 的格式规范

- **标题**：`# v0.X 实测数据（自 ` + "`milestone/v0.X.md`" + ` 转移）` + blockquote 指向 milestone（`test-v0.1.md:1`）
- **层级**：H2 功能节（如 `## F9 Benchmark 实测数据`）/ H3 实验节（`### VOC 2007 100 张图对比实验（conf=0.3）`，`test-v0.1.md:7` 模式——实验参数写进 H3 标题）
- **表格规范**：管道表格；模型名反引号；同列最佳值**加粗**；百分比 1 位小数、耗时 `Ns`
- **结论段**：每表后必跟 `**结论**：…` 段，给出可决策的解读（对照 `test-v0.1.md:14` 的 yolo26x vs FRCNN 写法）
- **历史值标注**：已移除模型的结果必须标注「历史值，模型已移除」（README 快照对 yolov8x 的处理方式）
- **存放红线**：实测数据只写 `test-v0.X.md`，**不**回写本文档 / README（文档分工见 `docs/CLAUDE.md`）

### 10.8 bash 一键运行（run_benchmarks.sh）

`run_benchmarks.sh`（`auto2dlabel/benchmarks/` 下）是 `run_all.py` 的薄包装，**不复刻 subprocess 调度**，三职责：环境检查、分组、透传。脚本开头 `cd "$(dirname "$0")/../.."` **CWD 自定位**到仓库根，任意工作目录可直接运行。

```bash
bash auto2dlabel/benchmarks/run_benchmarks.sh              # 全部 10 数据集
bash auto2dlabel/benchmarks/run_benchmarks.sh detection    # 仅检测组（coco2017,voc2007,kitti,dota,mot）
bash auto2dlabel/benchmarks/run_benchmarks.sh segmentation # 仅分割组（coco_seg,cityscapes,nuimages,d2sa）
bash auto2dlabel/benchmarks/run_benchmarks.sh classification # 仅分类组（imagenet100）
bash auto2dlabel/benchmarks/run_benchmarks.sh --only voc2007,kitti            # 指定数据集
bash auto2dlabel/benchmarks/run_benchmarks.sh detection --extra "--conf 0.1"  # 透传额外参数
bash auto2dlabel/benchmarks/run_benchmarks.sh --check-only # 只做环境/数据集检查，不实跑
```

| 组名 | 数据集（→ `run_all --only`） | 默认模型 |
| --- | --- | --- |
| `detection` | coco2017, voc2007, kitti, dota, mot | yolo26x.pt |
| `segmentation` | coco_seg, cityscapes, nuimages, d2sa | FastSAM-s.pt |
| `classification` | imagenet100 | resnet18 |
| `all` | 全部 10 个 | 按 run_all 注册表 |

⚠ **仅 CPU 服务器**：检测/分割默认模型 yolo26x.pt 在 CPU 上 50 图必超 `run_all` 3600s 超时——全组运行必须加 `--extra "--model yolo11n.pt"` 覆盖（CPU 实测见 `tests/test-v0.3.md`）。

- **环境检查**：python3 存在 + `import auto2dlabel` 可用；nvidia-smi 缺失仅 warn（CPU 可跑，不阻断）
- **调度边界**：bash 只做检查 + 分组 + 透传（`--only`/`--extra` 原样转发 `python -m auto2dlabel.benchmarks.run_all`），超时、汇总、退出码统计全部归 `run_all.py`；脚本退出码透传

---

## 文档维护

- 本文档是活文档：`common.py` / `datasets.py` / `run_all.py` / `schema/task_plan.py` 的公共接口变更时同步更新 §10 程序规范
- 数据集就绪状态（磁盘实况）以 `datasets_plan.md` 就绪状态总表为准，数据集增删时两文档同步更新
- 实测数据一律外置 `auto2dlabel/tests/test-v0.X.md`（呼应 `docs/CLAUDE.md` 文档分工：plan 只保留大纲）
- 版本路线与里程碑状态随主项目 `AutoLabel_plan.md` 同步更新
