# v0.1 实测数据（自 `milestone/v0.1.md` 转移）

> 实测细节与验收命令，供复盘与回归对比。功能定义与完成标记见 `../milestone/v0.1.md`。

## F9 Benchmark 实测数据

### VOC 2007 100 张图对比实验（conf=0.3）

| 模型 | mAP@0.5 | Avg Recall | 零检出类别 | 耗时 |
| --- | --- | --- | --- | --- |
| `fasterrcnn_resnet50_fpn_v2` | 24.5% | 27.2% | **11/20** | 61.6s |
| **`yolo26x.pt`** | **59.2%** | **61.9%** | 6/20 | 101.5s |

**结论**：yolo26x 在 VOC 20 类上全面碾压 FRCNN_V2（mAP 2.4×）。FRCNN_V2 仅在 car/person 等交通类别上有微弱优势，对 bird/cat/dog/chair/sheep 等室内和动物类别全部挂零。

### BDD100K 1000 张图对比实验（yolo26x.pt）

| conf | mAP@0.5 | car recall | person recall | traffic sign recall |
| --- | --- | --- | --- | --- |
| 0.5 | 16.8% | 41.9% | 33.5% | **0%** |
| **0.1** | **23.5%** | **63.7%** | **59.7%** | **0%** |

**结论**：降阈值后 car/person 召回提升至可用水平（60%+），但 **traffic sign（3,475 GT/1000 张）完全无法检出**——这是 COCO 训练模型在 BDD100K 上的域差距。rider/bike/motor 同样零检出。**car + person + truck + bus + traffic_light 五类合并 mAP 约 46.9%**，对驾驶场景预标注可接受；traffic sign 必须 fine-tune 或换专用模型。这是面试中讨论模型选型和 domain gap 的绝佳素材。

## 验收测试命令

```bash
# Test 1: 自然语言一次性命令
auto2dlabel chat "检测 000860.png 中的汽车和行人，conf=0.5，用 faster-rcnn"

# Test 2: 交互模式追问
auto2dlabel chat
> 帮我标注数据集
# 预期：Agent 追问类别、阈值等缺失信息

# Test 3: 多任务编排
auto2dlabel chat "先检测 /data/A 的汽车，conf=0.5；再检测 /data/B 的行人，conf=0.3"

# Test 4: 超时确认（--no-wait 跳过）
auto2dlabel chat "检测 000860.png 中的汽车" --no-wait

# Test 5: 回归测试
pytest auto2dlabel/tests/
```
