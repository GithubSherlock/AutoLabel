# AutoDL 服务器数据集总览

> 路径：`~/autodl-pub` | 更新时间：2026-08-04

---

## 一、目标检测 (Object Detection)

| 数据集 | 简介 | 路径 |
| --- | --- | --- |
| **COCO2017** | 大规模通用目标检测/实例分割/关键点检测，80 类，~118k 训练图像 | `COCO2017/` |
| **COCO14** | COCO 早期版本（2014），与 2017 版划分不同 | `COCO14/` |
| **VOCdevkit** | PASCAL VOC 2007 + 2012，20 类经典目标检测/分割基准 | `VOCdevkit/` |
| **DOTA** | 航拍图像旋转框目标检测，15 类（飞机、船舶、车辆等），含 v1.0 和 v1.5 标注 | `DOTA/` |
| **TT100K** | 中国交通标志检测，100k+ 图像，221 类交通标志 | `TT100K/` |
| **KITTI (object)** | 自动驾驶场景 2D/3D/BEV 目标检测，8 类（Car, Pedestrian, Cyclist 等） | `KITTI/object/` |
| **nuScenes** | 大规模自动驾驶 3D 目标检测，23 类，1000 场景，6 摄像头 + 激光雷达 + 雷达 | `nuScenes/` |
| **nuImages** | nuScenes 的 2D 图像子集，~93k 标注图像，用于 2D 检测/分割 | `nuScenes/nuImages/` |
| **cityscapes** | 城市场景语义分割与检测，8 大类 30 细类，也可用于 2D 目标检测 | `cityscapes/` |
| **D2SA** | 密集零售货架商品检测，SKU 级别标注 | `D2SA/` |
| **CelebA** | 大规模人脸属性与 landmark 检测，~200k 名人图像 | `CelebA/` |
| **CUB200-2011** | 细粒度鸟类分类/检测，200 类，含部位标注和分割 mask | `CUB200-2011/` |

---

## 二、实例分割 (Instance Segmentation)

| 数据集 | 简介 | 路径 |
| --- | --- | --- |
| **COCO2017** | 实例分割金标准，80 类逐像素 mask 标注 | `COCO2017/` |
| **COCO14** | COCO 早期版本，含实例分割 mask | `COCO14/` |
| **VOCdevkit** | PASCAL VOC 分割标注（class-level），可用于语义/实例分割 | `VOCdevkit/` |
| **cityscapes** | 城市场景实例级分割（8 类 thing 有实例 mask），自动驾驶场景 | `cityscapes/` |
| **nuImages** | nuScenes 2D 子集，含实例分割 mask（搭配 nuScenes-lidarseg） | `nuScenes/nuImages/` |
| **S3DIS** | 室内 3D 场景语义/实例分割，6 区域 271 房间，13 类 | `S3DIS/` |
| **KITTI (object)** | 含 2D/3D 实例级框标注，可用于 BEV 实例分割 | `KITTI/object/` |
| **D2SA** | 密集货架商品实例级检测（可转换为实例分割任务） | `D2SA/` |

---

## 三、3D 任务 / 自动驾驶

### 3.1 自动驾驶全栈

| 数据集 | 简介 | 传感器 | 路径 |
| --- | --- | --- | --- |
| **nuScenes** | 完整自动驾驶数据集，1000 场景 20s，3D 检测/跟踪/预测/BEV 建图 | 6×Camera + 1×LiDAR + 5×Radar + IMU/GPS | `nuScenes/Fulldatasetv1.0/` |
| **nuScenes-lidarseg** | nuScenes 激光雷达点云语义分割，32 类 | LiDAR | `nuScenes/nuScenes-lidarseg/` |
| **nuImages** | nuScenes 2D 图像子集，检测/分割/跟踪 | 6×Camera | `nuScenes/nuImages/` |
| **KITTI** | 自动驾驶经典基准：2D/3D 检测、里程计、场景流、深度估计 | 2×Camera + Velodyne LiDAR + GPS/IMU | `KITTI/` |
| ┣ **KITTI object** | 目标检测（2D/3D/BEV），7481 训练 + 7518 测试 | Camera + LiDAR | `KITTI/object/` |
| ┗ **KITTI sceneflow** | 场景流/光流/视差估计，Stereo 200 + Flow 200 场景 | Stereo Camera | `KITTI/sceneflow/` |
| **SemanticKITTI** | KITTI Odometry 序列的激光雷达点云语义分割，28 类 | Velodyne LiDAR | `SemanticKITTI/` |
| **cityscapes** | 城市场景语义理解，像素级标注（semantic + instance），~25k 图像 | Stereo Camera | `cityscapes/` |

### 3.2 室内 3D 场景理解

| 数据集 | 简介 | 路径 |
| --- | --- | --- |
| **S3DIS** | 斯坦福室内 3D 语义/实例分割，6 区域 271 房间，13 类，含 XYZ+RGB 点云 | `S3DIS/` |
| **ADEChallengeData2016** | MIT 场景解析（ADE20K），150 类语义分割，~20k 图像，含室内+室外场景 | `ADEChallengeData2016/` |

### 3.3 点云相关

| 数据集 | 相关任务 | 路径 |
| --- | --- | --- |
| **SemanticKITTI** | 点云语义分割、全景分割 | `SemanticKITTI/` |
| **nuScenes-lidarseg** | 激光雷达点云分割 | `nuScenes/nuScenes-lidarseg/` |
| **KITTI object (velodyne)** | 点云 3D 目标检测 | `KITTI/object/` |
| **S3DIS** | 室内点云语义/实例分割 | `S3DIS/` |

---

## 四、其他任务数据集

| 数据集 | 任务 | 简介 | 路径 |
| --- | --- | --- | --- |
| **ImageNet** | 图像分类 | ILSVRC2012，1000 类，~1.2M 训练图像 | `ImageNet/` |
| **ImageNet100** | 图像分类 | ImageNet 100 类子集 | `ImageNet100/` |
| **cifar-10** | 图像分类 | 10 类 32×32 小图，经典基准 | `cifar-10/` |
| **cifar-100** | 图像分类 | 100 类 32×32 小图 | `cifar-100/` |
| **CelebA** | 人脸属性/识别 | 200k 名人图像，40 属性 + 5 landmark | `CelebA/` |
| **MOT20** | 多目标跟踪 | 拥挤场景行人跟踪，8 个序列 | `MOT20/` |
| **mot17** | 多目标跟踪 | 多目标跟踪基准，14 个序列（7×2 检测器） | `mot17/` |
| **GOT10k** | 单目标跟踪 | 1 万视频段，563 类，运动/外观变化 | `GOT10k/` |
| **LaSOT** | 单目标跟踪 | 大规模长时跟踪，70 类 1400 段（含扩展子集） | `LaSOT/` |
| **Vimeo-90k** | 视频超分/插帧 | 90k 高清视频七连帧 | `Vimeo-90k/` |
| **DIV2K** | 图像超分辨率 | 2K 分辨率高清图像，含低分辨率版本 | `DIV2K/` |
| **BERT-Pretrain-Model** | NLP 预训练模型 | BERT Base/Large, Chinese, Cased/Uncased 权重 | `BERT-Pretrain-Model/` |

---

## 五、任务-数据集速查矩阵

| 数据集 | 目标检测 | 实例分割 | 3D检测 | 点云分割 | 语义分割 | 跟踪 | 自动驾驶 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COCO2017 | ✅ | ✅ | — | — | — | — | — |
| COCO14 | ✅ | ✅ | — | — | — | — | — |
| VOCdevkit | ✅ | ✅ | — | — | ✅ | — | — |
| DOTA | ✅ | — | — | — | — | — | — |
| TT100K | ✅ | — | — | — | — | — | — |
| D2SA | ✅ | ✅ | — | — | — | — | — |
| KITTI | ✅ | — | ✅ | — | — | — | ✅ |
| nuScenes | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| SemanticKITTI | — | — | — | ✅ | ✅ | — | ✅ |
| cityscapes | ✅ | ✅ | — | — | ✅ | — | ✅ |
| S3DIS | — | ✅ | — | ✅ | ✅ | — | — |
| ADE20K | — | — | — | — | ✅ | — | — |
| ImageNet | — | — | — | — | — | — | — |
| CUB200-2011 | ✅ | — | — | — | — | — | — |
| CelebA | ✅ | — | — | — | — | — | — |
| MOT20/17 | — | — | — | — | — | ✅ | — |
| GOT10k/LaSOT | — | — | — | — | — | ✅ | — |

---

## 六、说明

- **目标检测**：含边界框标注的数据集均可用于 2D 目标检测；KITTI/nuScenes 同时支持 3D/BEV 检测。
- **实例分割**：需要逐像素 mask 标注（区分同一类的不同实例），COCO 是最完善的选择；cityscapes 仅对 "thing"（可数物体）类提供实例级 mask；S3DIS 用于 3D 点云实例分割。
- **3D/自动驾驶**：nuScenes 是最全面的自动驾驶数据集（多模态、多任务）；KITTI 是经典基准但规模较小；SemanticKITTI 专注点云分割；S3DIS 专注室内 3D 场景。
- `BERT-Pretrain-Model` 是 NLP 预训练模型权重，不属于视觉任务。
- 分类数据集（ImageNet, CIFAR）无检测/分割标注，但可作为检测网络 backbone 预训练使用。

---

## 七、数据集 ↔ 项目适用性映射

### 7.1 auto2dlabel（2D 图像标注：目标检测 + 实例分割）

| 数据集 | 可用性 | 适用任务 |
| --- | --- | --- |
| **COCO2017** | ✅ 直接可用 | 目标检测、实例分割（80 类，金标准） |
| **COCO14** | ✅ 直接可用 | 目标检测、实例分割 |
| **VOCdevkit** | ✅ 直接可用 | 目标检测、语义分割（20 类，class-level mask） |
| **DOTA** | ✅ 直接可用 | 目标检测（航拍旋转框，15 类） |
| **TT100K** | ✅ 直接可用 | 目标检测（交通标志，221 类） |
| **KITTI (object)** | ✅ 2D 部分可用 | 目标检测（含 2D bounding box，8 类） |
| **nuImages** | ✅ 直接可用 | 目标检测、实例分割（~93k 标注图像） |
| **cityscapes** | ✅ 直接可用 | 目标检测、实例分割（thing 类）、语义分割 |
| **D2SA** | ✅ 直接可用 | 目标检测（密集货架商品，SKU 级） |
| **CelebA** | ✅ 直接可用 | 目标检测（人脸 bbox + landmark） |
| **CUB200-2011** | ✅ 直接可用 | 目标检测（细粒度鸟类，200 类 + 部位标注） |
| **ADEChallengeData2016** | ⚠️ 部分可用 | 语义分割（150 类），无实例级 mask |
| **ImageNet / ImageNet100** | ❌ 不适用 | 仅分类标签，无检测/分割标注 |
| **cifar-10 / cifar-100** | ❌ 不适用 | 仅分类标签，32×32 小图不适合标注 |
| **MOT20 / mot17** | ❌ 不适用 | 多目标跟踪，auto2dlabel 不支持跟踪 |
| **GOT10k / LaSOT** | ❌ 不适用 | 单目标跟踪，auto2dlabel 不支持跟踪 |
| **Vimeo-90k** | ❌ 不适用 | 视频超分/插帧，非标注任务 |
| **DIV2K** | ❌ 不适用 | 图像超分辨率，非标注任务 |
| **BERT-Pretrain-Model** | ❌ 不适用 | NLP 模型权重，非视觉任务 |
| **S3DIS** | ❌ 不适用 | 3D 室内点云，auto2dlabel 仅处理 2D 图像 |
| **nuScenes (3D)** | ❌ 不适用 | 3D 点云检测，auto2dlabel 不处理点云 |
| **SemanticKITTI** | ❌ 不适用 | 3D 点云分割，auto2dlabel 不处理点云 |
| **KITTI sceneflow** | ❌ 不适用 | 场景流/视差估计，非标注任务 |

### 7.2 auto3dlabel（3D 标注：点云检测 + 点云分割，待实现）

| 数据集 | 可用性 | 适用任务 |
| --- | --- | --- |
| **nuScenes** | ✅ 直接可用 | 3D 目标检测（LiDAR + 6 摄像头，23 类，1000 场景） |
| **nuScenes-lidarseg** | ✅ 直接可用 | 点云语义分割（32 类，LiDAR） |
| **KITTI (object)** | ✅ 直接可用 | 3D 目标检测（Velodyne LiDAR，8 类）、BEV 检测 |
| **SemanticKITTI** | ✅ 直接可用 | 点云语义分割（28 类）、全景分割 |
| **S3DIS** | ✅ 直接可用 | 室内 3D 点云语义/实例分割（13 类，271 房间） |
| **KITTI sceneflow** | ⚠️ 潜在可用 | 场景流/光流/视差估计（Stereo），非标注但可扩展 |
| **cityscapes** | ⚠️ 潜在可用 | Stereo Camera 数据，可用于深度估计（非核心任务） |
| **COCO2017 / COCO14** | ❌ 不适用 | 纯 2D 图像，无 3D/点云数据 |
| **VOCdevkit** | ❌ 不适用 | 纯 2D 图像 |
| **DOTA / TT100K** | ❌ 不适用 | 纯 2D 图像 |
| **D2SA / CelebA / CUB200** | ❌ 不适用 | 纯 2D 图像 |
| **ADEChallengeData2016** | ❌ 不适用 | 纯 2D 语义分割，无 3D 数据 |
| **ImageNet / CIFAR** | ❌ 不适用 | 纯 2D 分类 |
| **MOT / GOT / LaSOT** | ❌ 不适用 | 2D 跟踪任务 |
| **Vimeo-90k / DIV2K** | ❌ 不适用 | 超分任务 |
| **BERT-Pretrain-Model** | ❌ 不适用 | NLP 模型权重 |

### 7.3 双项目共用数据集

| 数据集 | auto2dlabel | auto3dlabel | 说明 |
| --- | --- | --- | --- |
| **KITTI (object)** | ✅ 2D 检测 | ✅ 3D 检测 | 同时含 2D/3D/BEV 标注，分别使用图像和点云 |
| **nuScenes** | ❌ | ✅ 3D 检测 | 主数据集为 3D 点云；其 2D 子集 nuImages 归 auto2dlabel |
| **nuImages** | ✅ 检测+分割 | ❌ | nuScenes 的 2D 图像子集，仅适用于 auto2dlabel |
| **cityscapes** | ✅ 检测+分割 | ⚠️ 深度估计 | 2D 标注用于 auto2dlabel；Stereo 可用于 auto3dlabel 深度估计扩展 |

### 7.4 总结

| 项目 | 可直接使用 | 部分可用 | 不适用 |
| --- | --- | --- | --- |
| **auto2dlabel** | COCO2017, COCO14, VOCdevkit, DOTA, TT100K, KITTI (2D), nuImages, cityscapes, D2SA, CelebA, CUB200-2011（11 个） | ADE20K（1 个） | 其余 9 个 |
| **auto3dlabel** | nuScenes, nuScenes-lidarseg, KITTI (3D), SemanticKITTI, S3DIS（5 个） | KITTI sceneflow, cityscapes（2 个） | 其余 14 个 |
