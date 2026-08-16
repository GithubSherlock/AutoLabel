# 测试数据集计划

> 所有推荐数据集均包含 Ground Truth 标注，可直接用于 benchmark 对比。

---

## 一、PASCAL VOC 2007 — 快速验证

| 信息 | 值 |
| --- | --- |
| 图像数 | 4,952（test） |
| 类别 | 20（person, car, bicycle, dog, cat, bird, ...） |
| GT 格式 | XML（与 AutoLabel VOC 导出格式兼容） |
| 下载大小 | ~430MB |

```bash
# 图像 + 标注
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

# GT 位置: VOCdevkit/VOC2007/Annotations/*.xml
# 图像位置: VOCdevkit/VOC2007/JPEGImages/*.jpg
```

**测试方式**：用 AutoLabel 生成 COCO JSON → 与原始 VOC XML GT 对比。

---

## 二、COCO val 2017 — 标准 benchmark

| 信息 | 值 |
| --- | --- |
| 图像数 | 5,000 |
| 类别 | 80 |
| GT 格式 | COCO JSON（与 AutoLabel COCO 导出格式兼容） |
| 下载大小 | 图像 ~1GB + 标注 ~200MB |

```bash
# 图像
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Ground Truth 标注（必需）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# GT 位置: annotations/instances_val2017.json
# 图像位置: val2017/*.jpg
```

**测试方式**：AutoLabel 导出的 COCO JSON 可与 `instances_val2017.json` 用 `pycocotools` 直接计算 mAP / Recall。

---

## 三、BDD100K — 驾驶场景（与感知融合岗最对口）

| 信息 | 值 |
| --- | --- |
| 图像数 | 10,000（val） |
| 类别 | 10（car, person, rider, bus, truck, bike, ...） |
| GT 格式 | COCO JSON |
| 下载大小 | 图像 ~1.8GB + 标注 ~5MB |

```bash
# 需要注册: https://bdd-data.berkeley.edu/
# 下载 val 图像 + detection GT JSON
# GT 格式与 AutoLabel COCO 导出完全兼容
```

---

## 四、建议的 Benchmark 实验流程

### 单模型测试（验证功能）

```bash
# 1. 下载 VOC 2007 test
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

# 2. 用 AutoLabel 自动标注
auto2dlabel chat "检测 VOCdevkit/VOC2007/JPEGImages/ 中的所有 car 和 person，conf=0.5，用 faster rcnn" --no-wait

# 3. 导出结果为 COCO JSON（与 GT 同格式，可直接 diff）
```

### 多模型对比（量化实验）

```bash
# 同一数据集分别跑两个模型
auto2dlabel chat "检测 val2017/ 中的 car person bicycle，conf=0.5，用 faster rcnn v2" --no-wait
auto2dlabel chat "检测 val2017/ 中的 car person bicycle，conf=0.5，用 yolov8x" --no-wait

# 对比两份 COCO JSON → 计算每类的 Recall / Precision
```

### LLM vs no-LLM 对照

```bash
# 复用 v0.1c 的对照实验框架
auto2dlabel run val2017/ "detect car person bicycle" -d fasterrcnn_resnet50_fpn_v2 -t 0.5 --no-llm
auto2dlabel run val2017/ "detect car person bicycle" -d fasterrcnn_resnet50_fpn_v2 -t 0.5
```

---

## 五、GT 对比工具

```bash
# 用 pycocotools 计算 COCO 指标（Python 脚本）
python3 -c "
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO('annotations/instances_val2017.json')
pred = gt.loadRes('outputs/val2017_*.json')
evaluator = COCOeval(gt, pred, 'bbox')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
"
```
