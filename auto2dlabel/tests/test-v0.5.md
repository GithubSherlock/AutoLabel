# v0.5 实测数据

> v0.5 内容（Pose / 指代 L2 / 自动车道 ROI / ILSVRC2012 val / Web 复核交互增强）实测记录。功能定义与完成标记见 `../milestone/v0.5.md`。测试环境：32 核 CPU，torch 2.13.0。Pose 冒烟时无 GPU（纯 CPU）；指代 L2 冒烟时会话中期 GPU 已可用（fp16 加载），CPU 可行性由 Pose 冒烟与 vendor 早期 CPU 端到端生成验证佐证。

## Pose 姿态估计冒烟（YOLO-pose，2026-08-23）

目标：验证「yolo11n-pose.pt → 17 关键点解析 → Bbox.keypoints 穿透 → COCO keypoints 导出 → 骨架可视化」全链路在 CPU 上可行。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 权重 | yolo11n-pose.pt（6.0MB，自动下载） | github 源下载 6:24（~16KB/s 慢于推理，已归位 `auto2dlabel/weights/` 后懒加载瞬时） |
| 推理 | **389.68s / 单图**（CPU） | COCO val2017/000000000139.jpg（640×426）；YOLO11n-pose CPU 无批量化单图极慢——纯本地冒烟可用，实用需 GPU（v0.5 Pose 为 GPU 分级能力） |
| 检测 | 1 框 person conf=0.816 bbox=(411.1,156.1,53.1,141.2) | 与 person 类检测一致 |
| 关键点 | **num_kpts=17**（全可见） | `_parse_pred` 逐点对齐 COCO-17，v=conf>0 取整 |
| COCO 导出 | `num_keypoints=17`、`keypoints` 展平 **51 值** | 内嵌 COCO JSON（无独立 coco_keypoints 格式），字段与官方语义一致 |
| 可视化 | /tmp/pose_smoke.png（444KB，272640 非零像素） | `draw_keypoints` 骨架线 + 圆点像素级验证命中（回归测试 `test_draw_keypoints_pixels` 同款判据） |

**结论**：Pose 全链路 CPU 通——检测→关键点→序列化四穿透点→COCO 内嵌导出→骨架可视化。质量门四件套：pytest 492 passed（新增 test_pose.py 15 用例）/ pyright 0 / mypy --strict 169（基线零新增）/ ruff 127/45（≤基线）。CLI 与 Web 分支由单测覆盖（execute_plan pose 步骤 FakePoseModel e2e + /api/annotate pose 分支），真实权重冒烟即上表。

## 指代 L2 冒烟（Florence-2，2026-08-23）

目标：验证「关系指代指令 → 检测框候选 → Florence-2 `<OPEN_VOCABULARY_DETECTION>` 单次解析 → 中心匹配保原框」全链路。冒烟环境：GPU 已可用（fp16 加载，CUDA 推理）。

### Vendored 官方代码（5.15 内置集成不可用的根因修复）

transformers 5.15.0 内置 florence2 集成有根本缺陷（缺 tokenizer 类；视觉骨干被重写与官方 checkpoint 不兼容 → UNEXPECTED/MISSING 大量权重），直接加载输出垃圾文本（"allow allow..." 重复 token 塌陷）。方案：vendor 官方代码（`models/vendor_florence2/`，不 trust_remote_code）+ 7 处 5.15 兼容补丁（Cache 抽象 / tie 机制 / meta-init / tokenizer 属性，清单见 vendor README）。权重绑定修复链实测：`tied lm_head: True | tied dec: True` 后生成 `'</s><s> person next to car<loc_667><loc_759><loc_998><loc_998></s>'` 有效定位序列。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 权重 | microsoft/Florence-2-base（~1G，HF 镜像下载） | checkpoint 仅存 `shared.weight`（51289×768）一份，fp16 存储；CPU 须 fp32 加载、GPU fp16（`referential.py` 按 `cuda.is_available()` 自动选） |
| 输入 | squash 768×768 | 官方 `_encode_image` 断言方形特征图（DaViT stride 32 → 577 tokens），坐标按 x/y 独立反变换回原图系 |
| 生成 | 2.5s（GPU 单次 `<OD>` generate，num_beams=3） | 序列级一次调用（首帧锁定 + 后续帧轨迹-检测匹配维持），不做逐帧 |
| 集成 | yolo26x 15 框（13 person + 2 car）→ `the person next to the red car` → 保留 (person, 0, 268, 107, 156) | Florence 输出 (0,267,110,426) 经中心距匹配回正确人框；resolve 全程 12.6s（GPU） |
| 兜底 | 解析失败/无输出 → 宁多勿漏保留全部框 | 单测覆盖（空输出 / 异常 / 空候选三例） |

### 短语措辞实验（9 变体 → 定冠词定稿）

Florence-2 对关系短语的**主体/地标歧义**实测：`"person next to car"` 出**车**框、`"person near car"` 出车框——无冠词时模型把关系短语理解为「定位地标」；主体加定冠词 `"the person next to the car"` 稳定出行人框。定稿 `build_referential_phrase`：`"the {主体} {关系词} the {属性+语境}"`（`红车旁边的行人` → `the person next to the red car`）；"near" 偏向地标已从常用映射中规避（中文「靠近/附近」仍映射 near，由定冠词兜底）。

**结论**：指代 L2 全链路通（GPU）。质量门四件套：pytest 518 passed（test_referential_l2.py 26 用例含 FakeResolver 注入断言 L1 不调用）/ pyright 0 / mypy --strict 169（基线零新增；vendor 目录 overrides 豁免 + 调用点定点 ignore）/ ruff 127/45（≤基线）。CPU 实用路径按 milestone 规划 5-30s/图（低频序列级一次），GPU 到位后 12.6s。

## 指代 L3 冒烟（Qwen2-VL-7B 4bit，2026-08-23）

目标：验证「Qwen2-VL-7B-Instruct NF4 4bit 加载 → grounding JSON prompt → 坐标解析（1000 网格/[0,1] 判别）→ 中心匹配保原框」全链路（GPU 3080 Ti 12GB）。合成场景：1000×600 白底左红车（100,200,350,400）右蓝车（650,200,900,400）。

### 三处环境/API 坑修复（实测根因）

| # | 报错 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `bitsandbytes 4-bit quantization requires accelerate` | 环境缺 accelerate | `pip install accelerate>=1.1.0` |
| 2 | `Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be the same` | 设备移动条件写反——float 张量（pixel_values）留在 CPU | 全部带 `.to` 的张量上设备 |
| 3 | `ValueError: Image features and image tokens do not match, tokens: 0, features: 756` | transformers 5.x Qwen2-VL 必须经 chat template 注入 `<\|image_pad\|>` 占位 token | `apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`（官方 demo 同款） |

坑 1 恰好实测了兜底路径：`_load` 异常 → resolve 黄字「解析失败（保留全部框）」+ `last_failed=True` 返回全部（宁多勿漏不崩管线）。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 权重 | 官方 fp16 16GB 经 HF 镜像下载（`weights/download_qwen_l3.sh`）→ bitsandbytes NF4 4bit 加载 **~4.5G 显存** | 730 shard 加载 ~15s；无 GPU/权重未就位均快速失败（守卫单测覆盖） |
| 生成输出 | `` ```json [{"bbox_2d": [96, 333, 350, 666]}] ``` `` | 1000 网格（官方坐标约定）→ 像素 (96, 200, 350, 400) |
| 匹配 | 红车短语 `the red car` → 保留 **红车框 (100,200)** | 模型输出与红车 GT 位置一致；中心距匹配保原框 |
| 耗时 | **33.2s 端到端**（权重加载 ~15s + 生成 ~7s） | 序列级一次调用（首帧锁定）；L2 12.6s / L3 33.2s 成本阶梯成立 |

**结论**：指代 L3 全链路通（GPU）。质量门四件套：pytest 581 passed（test_referential_l3.py 21 用例：解析纯函数 / Cascade 阶梯升级 / GPU·权重守卫 / CLI 路由）/ pyright 0 / mypy --strict 167（基线 169 减 2——detection.py `_model: Any` 注解）/ ruff 127/45（≤基线）。

## 自动车道 ROI 冒烟（UFLD，--roi auto，2026-08-23）

目标：验证「首帧 UFLD 车道线 → 自车车道闭合多边形 → filter_by_spatial 车道内过滤 → 跟踪」全链路。

### 权重获取（官方 pth 不可达 → ailia ONNX 预案）

官方 tusimple_res18.pth 仅 Google Drive / 百度网盘（AutoDL 均不可直连，gdown 连接超时实测）；HF 镜像/ModelScope 无 UFLD 镜像。按 milestone 预案降级 **ailia ONNX**：同一官方权重的官方转换，`storage.googleapis.com/ailia-models/ultra-fast-lane-detection/tusimple_18.onnx` 可达（下载脚本 `auto2dlabel/weights/download_lane_weights.sh`，sha256 校验 `fd0af1a9...65b6`）。实测 245MB（griding_num=100 + use_aux 头，比预案估的 ~40MB 大）。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 推理 | **1.33s / 单图**（CPU，onnxruntime） | 288×800 ResNet18；首帧一次解析（`--roi auto`），不逐帧 |
| 检测（KITTI 000001，1242×375） | 2 车道：左线 37 点 / 右线 36 点，顶部 x 611/675 汇聚、底部 411/880 展开，y 182→370 上→下 | 行序/左右方向经双场景数值验证（ailia 样例图 + KITTI）；KITTI 000000 无输出（弯道域外，回退全图保留宁多勿漏） |
| 多边形 | 73 边形（左线正序 + 右线反序闭合） | `select_ego_lane_pair` 夹中线对（4 线取 lane1/lane2，全偏一侧退化为最靠中线两线） |
| CLI 冒烟 | `/tmp/kitti3` 3 帧 `run ... --track --roi auto --no-viz` → `--roi auto: 自车车道 ROI 73 边形（首帧 UFLD）` → 3 帧 2 条轨迹 | 只有车道内汽车进跟踪 |
| 失败降级 | 源无帧 / 模型异常 / <2 车道 → 黄字 + 无 ROI 全图保留 | 单测覆盖三路径 |

**结论**：自动车道 ROI 全链路 CPU 通（UFLD ONNX 1.33s 首帧解析 + 3a 空间过滤零改动复用）。质量门四件套：pytest 534 passed（test_lane_roi.py 16 用例）/ pyright 0 / mypy --strict 169（基线零新增）/ ruff 127/45（≤基线）。解码与 ailia 官方后处理逐行等价（软 argmax rel 定位 + raw 行 r ↔ anchor[r] 行序），依赖 onnxruntime（pyproject 新增，纯 pip CPU 版）。

## ILSVRC2012 val 分类扩展冒烟（2026-08-23）

目标：验证「6.7GB val tar **不整解压**——devkit GT 每类分层抽样 → 1000 类 meta → resnet18 top-K」全链路。冒烟时 GPU 已可用（RTX 3080 Ti，CUDA）；CPU 可行性同 ImageNet100 先例（resnet18 单图轻量推理）。

### 归档格式实测（与官方文档不符处兼容双格式）

本归档 devkit 的 `ILSVRC2012_validation_ground_truth.txt` 为**单列类 ID**（5 万行，行序即图序，第 i 行 ↔ `ILSVRC2012_val_{i:08d}.JPEG`，实测行 1 → 00000001.JPEG → 490）；官方两列 "文件名 类ID" 格式一并兼容（`parse_imagenet1k_validation_gt` 双分支）。`meta.mat` 1860 synset 中 ILSVRC2012_ID 1-1000 为 1k 类（1001+ 附加集剔除），ID→英文名首词与 torchvision 100/100 匹配模式一致（样例 490 → sea snake，GT 类 ID 全覆盖 meta 1000 键）。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 数据准备 | 6.7GB tar 成员扫描 + 选择性解压 **2000/2000 张**（每类 2）→ `datasets/imagenet1k/val/` | 不整解压；manifest（`_selected_pc2.txt`）幂等，二调免扫 tar 索引；per_class=1→1000 冒烟档 / 0 或 50→全量 5 万 |
| GT | 200 张（`--max-images 200` 跨类均匀 → **200 个类各 1 张**） | 1000 类 ID 全覆盖 meta |
| 推理 | resnet18 top-5，**1.8s / 200 图（111.5 img/s，GPU）** | 批量参数 resolve_batch_params 自动实测；监督路径 candidates=[] 纯 1000 类 top-K（官方协议） |
| 精度 | **top-1 0.7100 / top-5 0.8800**（142/176） | vs 随机基线 1/1000 = 0.1%，远超；GT 英文名与预测精确字符串比较 |
| 产物 | `benchmarks_outputs/imagenet1k_resnet18_*.json/.md` | run_all 注册 `--dataset imagenet1k --per-class 2 --max-images 200` |

**结论**：ILSVRC2012 val 接入闭环——分层抽样 + 幂等解压 + 双格式 GT 兼容 + 1000 类 meta（12 数据集 Benchmark）。质量门四件套：pytest 547 passed（test_imagenet1k_gt.py 13 用例）/ pyright 0 / mypy --strict 169（基线零新增）/ ruff 127/45（≤基线）。

### 全量 5 万图基准（GPU 复测，2026-08-23）

GPU 到位后按计划复测全量（`--per-class 50 --max-images 0` → 50000 张）：

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| 数据 | 解压 **50002 文件**（幂等续传，manifest `_selected_pc50.txt`） | 6.7GB tar 选择性全量抽取 |
| 推理 | **440.8s / 50000 图（113.4 img/s，GPU）** | 与冒烟吞吐一致（111.5 img/s）；批量参数自动实测 |
| 精度 | **top-1 0.6968 / top-5 0.8899**（44494/50000，998 类出现） | 与官方 resnet18 ImageNet1k top-1 **0.6976** 一致 → 管线正确性独立验证；2 类 val 无样本不参与 |
| 产物 | `benchmarks_outputs/imagenet1k_resnet18_2026-08-23-11-04-47.json/.md` | 12 数据集全量档闭环 |

**结论**：ILSVRC2012 val 全量基准完成——resnet18 与官方精度一致，分类扩展 v0.5 项完整闭环（12 数据集 Benchmark）。

## Web 交互增强冒烟（jsdom + 端到端 API，2026-08-23）

目标：验证 Web 复核界面 CVAT 式增强（P0 未保存确认/快捷键/undo + P1 手柄/列表/过滤/右键 + P2 区域 issue + edited_by_human 数据回路 + 已复核重开）全链路可用。无真实浏览器环境（AutoDL 无头容器），采用三层替代验证：

| 层 | 方式 | 规模 | 说明 |
| --- | --- | --- | --- |
| 单测 | pytest `test_web_annotate.py` + `test_web_review.py` | **26 passed** | edited_by_human 保真/条件输出、issues 落盘/双生效、reviewed 列表/损坏容错、COCO 重开原地覆盖（TestClient + monkeypatch REVIEW_DIR，零模型） |
| 前端 | jsdom 冒烟 `smoke_web.js`（Node + jsdom 加载真实 app.js） | **76/76 passed** | 全键盘/鼠标事件驱动断言：渲染、undo 6 类、OBB 缩放/旋转数学、列表委托、过滤、隐藏锁定、右键、issues 状态机、saveReview body、空态/分类回归 |
| 服务 | 隔离服务器端到端（`AUTOLABEL_PORT=8799`，相对 REVIEW_DIR） | **5/5** | index 引用 app.js → 队列扫描 → edited+issues 保存落盘（`edited_by_human: True` 注入 annotation + 顶层 `issues`/`image_path`）→ reviewed 列表 → 重开原地覆盖无重复 `.reviewed` 标记 |

冒烟暴露并修复 3 个前端 bug（均被断言捕获）：

| bug | 根因 | 修复 |
| --- | --- | --- |
| undo 后再次 undo 崩溃 | `commitDrag` 的 undo 闭包捕获模块级 `drag` 变量（置 null 后解引用报错） | 闭包改捕获 mousedown 快照的局部解构 `const {i, before, type} = drag` |
| issueMode 下点击已有 issue 无效 | mousedown 分派未先测 issue 命中（点击开新草稿而非翻转状态） | issueMode 分支加 `findIssueAt` 优先命中检查 |
| resize 参考点错误 | 旧公式 `w1 = 2·\|p−C0\|` 假设中心不动，实测对角点漂移 | 旋转系内「对角固定 F + 手柄吸附指针」解：`w1 = sx·(plx−fx)`、`C_new = (p+F)/2`（edge 手柄 sx=0 轴不变），θ=0 与 θ=π/4 数值断言双验证 |

关键数值断言（`smoke_web.js` ⑤节）：θ=0 拖左上角手柄 (10,10)→(0,0)，对角 (40,40) 固定 → 框 (0,0,40,40)；θ=π/4 框 (w0=141.4, h0=70.7) 拖右下角手柄到局部 (100,40) → w1=170.7、h1=75.35、对角点不动、keypoints 仿射随动（1e-6 容差）。

**结论**：Web 增强全链路闭环——AI 初稿 → 人工修正（`edited_by_human` 落盘形成数据回路）→ issues 区域反馈 → 已复核重开。回归运行：`cd auto2dlabel/tests/helpers && node smoke_web.js`（需 `npm i jsdom`，脚本不入 pytest 体系）。
