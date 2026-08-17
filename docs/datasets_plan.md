# 测试数据集计划

> 2026-08-16 磁盘实况核查：已解压数据集位于 `/root/autodl-tmp/Documents/datasets/`（`DATASETS_ROOT`，与 `auto2dlabel/benchmarks/datasets.py` 一致），归档源位于 `/root/autodl-pub/`（`ARCHIVE_ROOT`）。所有推荐数据集均含 GT 标注，可直接用于 benchmark 对比。本文档按 Benchmark 三大模块（**分类 / 目标检测 / 实例分割**，见 `Benchmark_plan.md` 模块划分）组织。

---

## 就绪状态总表

| 数据集 | 模块 | 规模（评测子集） | 已解压（datasets/） | 归档源（/root/autodl-pub） | 状态 |
| --- | --- | --- | --- | --- | --- |
| ImageNet100 | 分类 | 100 类（wnid 目录） | `imagenet100/`（每类 50 张抽样解压 = 5000 图 / 566MB） | `ImageNet100/imagenet100.zip`（14GB） | ✅ 抽样解压就绪（`--per-class` 可调） |
| ILSVRC2012 val | 分类 | 50,000 图 / 1,000 类 | — | `ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar`（6.7GB）+ devkit GT | 📦 归档可用，未解压 |
| cifar-10 / cifar-100 | 分类 | 10 / 100 类，32×32 | — | `cifar-10/`、`cifar-100/`（各 ~170MB） | 📦 归档可用（冒烟用） |
| CUB200-2011 | 分类 | 200 类细粒度 | — | `CUB200-2011/CUB_200_2011.tgz`（1.1GB） | 📦 归档可用，未解压 |
| PASCAL VOC 2007 | 检测 | 4,952 图（test）/ 20 类 | `VOCdevkit/`（904MB） | `VOCdevkit/` | ✅ 就绪 |
| COCO 2017 val | 检测 | 5,000 图 / 80 类 | `COCO2017/`（826MB） | `COCO2017/` | ✅ 就绪 |
| KITTI object | 检测 | 300 图（截取）/ 8 类 | `KITTI/`（269MB） | `KITTI/` | ✅ 就绪 |
| DOTA v1.0 val | 检测 | 458 图 / 15 类 | `DOTA/`（3.3GB，HBB 已解压） | `DOTA/val/`（含旋转 GT，见下） | ✅ HBB 就绪；🔄 旋转 GT 归档可用 |
| MOT17 / MOT20 | 检测 | FRCNN 7 seq / 4 seq，单类 person | `MOT17/`（5.7GB）、`MOT20/`（4.9GB） | `mot17/`、`MOT20/` | ✅ 就绪 |
| BDD100K | 检测 | 10,000 图（val）/ 10 类 | — | — | ❌ 未就绪（需注册下载） |
| COCO 2017 val 实例分割 | 分割 | 5,000 图 / 80 类 | 同 COCO2017 | 同上 | ✅ 就绪 |
| Cityscapes val | 分割 | 500 图 / 8 thing 类 | `cityscapes/`（1.3GB） | `cityscapes/` | ✅ 就绪 |
| nuImages Mini | 分割 | 50 图 | `nuImages/`（117MB） | `nuScenes/nuImages/` | ✅ 就绪 |
| D2SA val | 分割 | 3,600 图 / 15,654 实例 / 60 SKU | `D2SA/`（4.5GB） | `D2SA/` | ✅ 就绪 |
| ADE20K 2016 | 分割（语义） | 20,210 图 / 150 类 | — | `ADEChallengeData2016/` | 📦 归档可用（远期） |

---

## 一、分类（Benchmark 模块：分类）

分类模型 v0.3 已上线（CLIP/SigLIP 零样本 + torchvision 14 款监督），以下数据集支撑分类基准落地：

### ImageNet100（主力）

| 信息 | 值 |
| --- | --- |
| 图像数 | 100 类（ImageNet1K 子集，wnid 目录组织） |
| GT 格式 | 目录名即标签（`imagenet100/{wnid}/*.JPEG`） |
| 归档 | `/root/autodl-pub/ImageNet100/imagenet100.zip`（14GB，未解压） |

- 与 torchvision 14 款分类模型权重**同源**（ImageNet1K 监督），是监督分类的直接评测集
- wnid → 英文类名映射经 devkit `ILSVRC2012_devkit_t12.tar.gz` 的 `data/meta.mat`（`load_imagenet100_meta`，datasets.py；**非 map_clsloc.txt——两 devkit 均无此文件**）转出，CLIP 零样本候选用英文名

### ILSVRC2012 val（完整 1000 类重载）

| 信息 | 值 |
| --- | --- |
| 图像数 | 50,000 |
| 类别 | 1,000 |
| GT 格式 | devkit `ILSVRC2012_validation_ground_truth.txt`（每行 wnid 对应图像序号） |
| 归档 | `/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar`（6.7GB）+ `ILSVRC2012_devkit_t12.tar.gz` |

- 完整 ImageNet 评测；注意分类模型权重下载在 `auto2dlabel/weights/hub/checkpoints/`，不占数据集目录

### cifar-10 / cifar-100（冒烟）

| 信息 | 值 |
| --- | --- |
| 图像数 | 60,000（各） |
| 类别 | 10 / 100 |
| 归档 | `/root/autodl-pub/cifar-10/cifar-10-python.tar.gz`、`/root/autodl-pub/cifar-100/cifar-100-python.tar.gz` |

- 32×32 小图，torchvision 模型需 resize 至 224——**仅作指标链路冒烟**（快速验证 top-K 计算正确性），不作质量结论依据

### CUB200-2011（细粒度）

| 信息 | 值 |
| --- | --- |
| 图像数 | 11,788 / 200 类（含 bbox/关键点标注） |
| 归档 | `/root/autodl-pub/CUB200-2011/CUB_200_2011.tgz`（1.1GB） |

- 细粒度鸟种分类，域难度显著高于 ImageNet——用于检验零样本 CLIP 的细粒度上限

---

## 二、目标检测（Benchmark 模块：目标检测）

### PASCAL VOC 2007

| 信息 | 值 |
| --- | --- |
| 图像数 | 4,952（test） |
| 类别 | 20（person, car, bicycle, dog, cat, bird, ...） |
| GT 格式 | XML（Annotations/*.xml） |
| 位置 | `datasets/VOCdevkit/VOC2007/`（已解压，904MB） |

**测试方式**：`python -m auto2dlabel.benchmarks.voc_benchmark`（GT XML → 逐类 mAP@0.5）。

### COCO val 2017

| 信息 | 值 |
| --- | --- |
| 图像数 | 5,000 |
| 类别 | 80 |
| GT 格式 | COCO JSON（`annotations/instances_val2017.json`） |
| 位置 | `datasets/COCO2017/`（已解压，826MB） |

**测试方式**：`python -m auto2dlabel.benchmarks.coco_benchmark`（同时是 `coco_seg_benchmark` 的图像源）。

### KITTI object

| 信息 | 值 |
| --- | --- |
| 图像数 | 300（按标注筛选解压，`ensure_kitti` 可扩） |
| 类别 | 8（Car/Pedestrian/Cyclist/Van/Truck/Tram/Person_sitting，Misc/DontCare 跳过） |
| GT 格式 | txt（KITTI 15 字段） |
| 位置 | `datasets/KITTI/object/`（已解压，269MB） |

### DOTA v1.0 val（HBB 已就绪 + 旋转 GT 可用）

| 信息 | 值 |
| --- | --- |
| 图像数 | 458 |
| 类别 | 15（仅 4 类有 COCO 对应，HBB 评测类） |
| GT 格式（HBB） | txt 8 值四边形的 min/max 外接框（`DOTA/labels/`，已解压） |
| GT 格式（旋转，Task1） | txt 8 值四边形 `x1 y1 x2 y2 x3 y3 x4 y4 class difficulty` + imagesource/gsd 头行 |
| 位置 | `datasets/DOTA/`（3.3GB）；旋转 GT 在归档 `DOTA/val/labelTxt-v1.0/labelTxt.zip`（458 文件，**未解压**） |

- 旋转 GT 已在归档——`common.rotate_iou`（shapely）就绪，OBB 评测可排期（见 `Benchmark_plan.md` 缺口表）
- 另有 v1.5 标注（`DOTA/val/labelTxt-v1.5/`，16 类 + HBB 变体）可作扩展

### MOT17 / MOT20（检测视角）

| 信息 | 值 |
| --- | --- |
| 规模 | MOT17-FRCNN 7 seq / MOT20 4 seq |
| 类别 | 单类 person（`MOT_CLASS_MAP={1:"person",7:"person"}`） |
| GT 格式 | `gt/gt.txt`（frame,id,x,y,w,h,conf,class,visibility） |
| 位置 | `datasets/MOT17/`、`datasets/MOT20/`（已解压，路径解析型） |

- 当前仅逐帧检测评测；v1.0 Tracking 基准将升级为 MOTA/IDF1/MT 时序指标

### BDD100K（未就绪）

- 磁盘无归档，需注册 https://bdd-data.berkeley.edu/ 后下载 val 图像 + detection GT JSON
- 历史价值：test-v0.1.md 的 BDD100K 实验揭示了 COCO 训练模型的 traffic sign 域差距——数据集可得后可固化为基准脚本

---

## 三、实例分割（Benchmark 模块：实例分割）

### COCO 2017 val 实例分割

- 图像同检测；GT 用 `instances_val2017.json` 的 `segmentation` 字段（polygon / RLE 双格式）
- **测试方式**：`python -m auto2dlabel.benchmarks.coco_seg_benchmark`（检测 conf≥0.5 → FastSAM box-prompt）

### Cityscapes val

| 信息 | 值 |
| --- | --- |
| 图像数 | 500（val） |
| 类别 | 8 个 thing 类（person/car/truck/bus/train/motorcycle/bicycle/rider） |
| GT 格式 | `gtFine/val/*_gtFine_instanceIds.png`（`instance_id*1000 + class_id` 编码） |
| 位置 | `datasets/cityscapes/`（1.3GB，仅解压 val 分片） |

### nuImages Mini

| 信息 | 值 |
| --- | --- |
| 图像数 | 50 |
| GT 格式 | nuScenes 自定义 RLE mask + 层级类别名（`vehicle.car` 等） |
| 位置 | `datasets/nuImages/`（117MB，v1.0-mini） |

### D2SA val（box-prompted 模式）

| 信息 | 值 |
| --- | --- |
| 图像数 | 3,600（val，全量 22,562） |
| 实例数 | 15,654 |
| 类别 | 60 SKU → 27 超类聚合 |
| GT 格式 | COCO JSON（`annotations/D2S_amodal_validation.json`，RLE 反斜杠格式，pycocotools `annToMask` 处理） |
| 位置 | `datasets/D2SA/`（4.5GB） |

- 无检测步骤：GT bbox 直接作 FastSAM prompt，评估给定位置下的原始分割质量

### ADE20K 2016（远期）

- 归档 `/root/autodl-pub/ADEChallengeData2016/`（20,210 图 / 150 类语义分割）
- 与现有 torchvision 语义分割模型（FCN/DeepLabV3/LRASPP，VOC 21 类）类表不同，接入需重映射——列为远期

---

## 四、Benchmark 实验流程（更新版）

数据集对比实验已由 `auto2dlabel/benchmarks/` 脚本接管（规范见 `Benchmark_plan.md` §10），不再手写流程：

```bash
# 单数据集（冒烟 → 全量）
python -m auto2dlabel.benchmarks.voc_benchmark --max-images 2      # 冒烟
python -m auto2dlabel.benchmarks.voc_benchmark                     # 全量（默认 50 图）

# 多模型对比：同数据集不同 --model 各跑一次，产物时间戳命名天然可对比
python -m auto2dlabel.benchmarks.coco_benchmark --model yolo26x.pt
python -m auto2dlabel.benchmarks.coco_benchmark --model fasterrcnn_resnet50_fpn_v2

# 批量：一键跑全组 / 分组
python -m auto2dlabel.benchmarks.run_all
bash auto2dlabel/benchmarks/run_benchmarks.sh segmentation --extra "--conf 0.1"

# 分类（ImageNet100，每类 50 张抽样解压，监督 1000 类空间 top-K）
python -m auto2dlabel.benchmarks.classification_benchmark --dataset imagenet100 --model resnet18 --max-images 50
bash auto2dlabel/benchmarks/run_benchmarks.sh classification

# LLM vs no-LLM 对照（非数据集基准，M1c 框架）
# → auto2dlabel/tests/helpers/no_llm_baseline.py
```

- 结果产物：`benchmarks_outputs/{dataset}_{model}_{ts}.json/.md`，批量汇总 `summary_{ts}.md`
- 实测结论沉淀：`auto2dlabel/tests/test-v0.X.md`（格式规范见 `Benchmark_plan.md` §10.7）

---

## 五、GT 对比工具

评测指标已自研实现（`auto2dlabel/benchmarks/common.py`），取代早期 pycocotools 流程：

| 用途 | 工具 | 说明 |
| --- | --- | --- |
| 检测逐类评测 | `evaluate_per_class` | 贪心 IoU 匹配 + 11 点插值 AP（common.py:80） |
| 分割逐类评测 | `evaluate_mask_per_class` | mask IoU 匹配 + mIoU/mDice（common.py:235） |
| mask 光栅化 | `polygon_to_mask` / `coco_seg_to_mask` / `rle_to_mask` / `nuscenes_rle_to_mask` | 多边形 / COCO RLE（pycocotools）/ nuScenes RLE（common.py:161-208） |
| 旋转框 IoU | `rotate_iou` | shapely 8 值四边形（common.py:44，OBB 评测待接入） |
| 分类（已落地） | `evaluate_classification` / `format_classification_table` | top-1 / top-K 准确率 + 逐类 correct/total/accuracy（common.py，ImageNet100 实测见 `tests/test-v0.3.md`） |

pycocotools 现仅保留 mask RLE 解码用途（`frPyObjects` + `decode`），不再承担 mAP 计算。
