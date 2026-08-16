# 项目计划书

AutoLabel 是一个软件开发计划，其中暂定包括 **Auto2dLabel** 和 **Auto3dLabel** 两个子项目。**Auto2dLabel** 这个名字意味着通过借助当下先进的 CV 以及 LLM 模型自动生成2维标签，同理，**Auto3dLabel** 这个名字意味着借助这些模型自动生成3维标签。该计划秉持着先易后难的原则，先实现核心功能，再通过数据接口拓展其他功能。

## 项目背景

由于当前 AI 大爆发的背景，之前的大部分软件都可能需要被 Agentic AI 重制，其中包括自动化生成数据标签软件。传统的数据标注软件（如 CVAT、Label Studio）核心是「人工画布 + 快捷键」，自动化能力（预标注）作为辅助插件存在，其工作流仍然是「人为主、AI 为辅」。一个真正 Agentic 的标注软件应该将范式反转为 **「AI 为主、人为辅」**：用户只需下达自然语言指令，由 LLM Agent 自主规划、调用模型、评估结果、并在不确定时主动请求人工介入——这正是 AutoLabel 要做的。

## Target User（目标用户）

- **主要用户**：AI 算法工程师 / 数据科学家，需要为训练/微调模型快速生成高质量预标注，再在外部审核工具（Label Studio / CVAT）中精修
- **使用场景**：单人或小团队在消费级硬件（Apple Silicon / 单 GPU）上对百到千张级别图像进行自动预标注，容忍适度的人工审核
- **不是面向**：专业标注团队的大规模生产管线（需 Web 协作、权限管理、SaaS），也不面向零 AI 背景的标注员

## 项目范围

明确包含哪些子项目（Auto2dLabel / Auto3dLabel），以及**不包含**什么（Out of Scope），避免范围蔓延

### Auto2dLabel

2维数据标签在数学上通常是由点、线、面组成的节点、线段、矩形以及多边形。通过这些几何形状对一张 x*y 维度的图像上的若干个区域进行覆盖，从而生成针对若干个现实实体的标注。根据 **Ultralytics** 平台上的信息，2维标注可以分为：

**目标检测（Object Detection）**：一种涉及识别图像或视频流中物体的位置和类别的任务。检测器的输出是一组包围图像中物体的边界框，以及每个框的类别标签和置信度分数。

**实体分割（Instance Segmentation）**：比目标检测更进一步，涉及识别图像中的各个物体并将其从图像的其余部分中分割出来。模型的输出是一组勾勒出图像中每个物体轮廓的掩码或轮廓，以及每个物体的类别标签和置信度分数。

**图像分类（Image Classification）**：三种任务中最简单的一种，涉及将整张图像分类为一组预定义类别中的某一个。分类器的输出是单个类别标签和置信度分数。

**姿态估计（Pose Estimation）**：一种涉及识别图像中特定点（通常称为关键点）位置的任务。关键点可以表示物体的各个部位，如关节、地标或其他显著特征。模型的输出是一组表示图像中物体上关键点的点，通常附带每个点的置信度分数。

**OBB 模板检测（Oriented Bounding Box Object Detection）**：在传统目标检测的基础上增加了方向角度，使用旋转的边界框以更好地贴合物体的朝向，从而提高定位精度。这在航空或卫星图像等物体不与图像轴对齐的应用中特别有用。

**目标跟踪（Object Tracking）**：一种不仅识别帧内物体的位置和类别，还在视频推进过程中为每个检测到的物体维护唯一 ID 的任务。

在当前的计算机视觉工程中，大部分用到的还是2维数据标签

**Out of Scope（明确不做）**：3D 点云标注、医疗影像专业标注、需要专家先验的细粒度领域。

## 版本路线（Auto2dLabel）

| 版本 | 覆盖任务 | 状态 |
| --- | --- | --- |
| **v0.1** | 目标检测（3 引擎 × 49+ 模型）+ 实例分割（SAM2/SAM3/FastSAM/Mask R-CNN）+ 类别自动推荐 + Web 基础审核 + 9 数据集 Benchmark | ✅ 完成 |
| **v0.2** | 实例分割（SAM3）+ 类别推荐 + Web 审核（记录见 `auto2dlabel/milestone/v0.2.md`） | ✅ 完成 |
| **v0.3** | Image Classification（CLIP/SigLIP）+ OBB（YOLO-OBB）+ M2/M3 闭环补全 | ✅ 完成 |
| **v0.4** | + Pose Estimation（ViTPose/RTMPose） | 📋 规划 |
| **v1.0** | + Object Tracking（ByteTrack/BoT-SORT，视频时序） | 📋 规划 |

> 编号注记：早期路线表曾把「分类+OBB」标为 v0.2，实现时后移为 v0.3（v0.2 编号已被分割里程碑文件占用，不重命名现有文件）。

## 里程碑状态（Auto2dLabel）

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| **M0 调研期** | 模型选型、竞品（CVAT / Label Studio / Roboflow）分析 | ✅ 完成 |
| **M1 单流程跑通** (v0.1a) | Agent Loop + DetectionTool + 3 引擎 × 47 模型 + CLI + COCO/YOLO/VOC 导出 + 日志 + 可视化 | ✅ 完成 |
| **M1b 分割** (v0.2) | SAM2/SAM3/FastSAM/Mask R-CNN 集成 + 类别推荐 + Web 基础审核 | ✅ 完成 |
| **M1c 价值证明** (v0.1c) | no-LLM 对照实验 + HITL 置信度三档分流（对照能力现于 `tests/helpers/no_llm_baseline.py`） | ✅ 完成 |
| **M2 服务化 + 前端审核** | FastAPI + 审核 UI + mask Canvas 叠加 + 复核队列闭环（消费方 + 保存回流 + 后端 COCO 导出） | ✅ 完成 |
| **M3 Agentic 闭环** | LLM Evaluate 节点（条件暴露）+ batch 失败隔离/--resume + 主动学习采样 | ✅ 完成 |

## 已知缺口 / 待办

（2026-08 核查遗留的三项已全部闭合：LLM Evaluate 节点、mask Canvas 叠加 + 复核队列、batch 失败重试均于 v0.3 周期完成，见 `auto2dlabel/milestone/v0.3.md` Phase 4）

| 缺口 | 归入 |
| --- | --- |
| Oriented R-CNN 未实现（mmrotate 依赖重，取舍注记） | 暂无计划 |
| DOTA Task1 旋转 GT 评测（rotate_iou 已就绪，GT 归档未验证到） | P2 挂起 |
| AgentState from_dict 恢复未实现（to_dict 快照已写） | checkpoint 完整版 |
| Web 审核：bbox 拖拽/标签编辑、分类与 OBB 结果展示 | 后续版本 |

## 差异化定位

- **不是**：又一个标注画布工具（CVAT/Label Studio 已经做得很好）。
- **是**：一个 **Agentic 预标 + 多模型引擎 + 结构化日志 + 多格式导出**的自动标注层。用户说「检测汽车和行人」，系统自主选择 prompt、调模型、评估、导出——全程无需手动框选。

---

## Auto3dLabel

> 本部分为 3D 标注现状调研（2025-08），作为 Auto3dLabel 的规划输入。核心结论：**3D 标注远不止 3D bbox**；自动标注存在四条主流技术路线，建议 MVP 走「2D 基础模型 → 3D 提升」，远期评估「重建驱动（NeRF/3DGS）」。

### 1. 3D 标注对象（标的是什么）

| 任务 | 标注内容 | 代表数据集 |
| --- | --- | --- |
| **3D 目标检测** | 3D bbox（中心 + 尺寸 + 朝向 yaw） | KITTI、nuScenes、Waymo |
| **3D 语义分割** | 点云中每个点打类别标签 | SemanticKITTI |
| **3D 实例分割** | 每个点类别 + 实例 ID | SemanticKITTI、Waymo |
| **目标跟踪** | 跨帧给同一物体维持唯一 ID | nuScenes、Argoverse |
| **3D 占用网格 (Occupancy)** | 空间体素化为 occupied/free/类别 | Occ3D-nuScenes |
| **HD 地图 / 车道线** | 车道线、路沿、停止线的 3D 折线 | Argoverse、nuPlan |
| **可行驶区域** | BEV 下的路面多边形 | 车企内部数据 |
| **运动属性** | 每帧速度、加速度、静止/移动状态 | nuScenes |
| **关键点** | 车辆四角点、人体骨架 3D 关键点 | 内部数据 |

其中 bbox + 跟踪 ID + 运动属性通常作为「一套标注」同时完成。

### 2. 人工标注的实际工作流

现代 3D 标注工具（Scale AI、Segments.ai、车企自研平台）界面均为**多视图协同**：

1. **BEV 俯视图为主视图**——标注员在鸟瞰图上画旋转矩形，天然贴合 bbox 定义
2. **侧视图/正视图辅助微调**——高度、俯仰角在侧视图调整
3. **点云投影到相机图像**——多相机同步显示，用于确认被遮挡物体的类别和边界
4. **时序传播辅助**——标好首帧后用跟踪传播到后续帧，人工只修漂移

关键事实：**纯人工逐点标注点云分割不现实**（单帧几十万点），3D 语义/实例分割实际都是「自动生成候选 + 人工修正」。

### 3. 自动标注技术路线（2025–2026 现状）

| 路线 | 方法 | 代表工作 | 点评 |
| --- | --- | --- | --- |
| **1. 模型伪标签** | 已有 3D 检测/分割模型直接出预测，人工审核 | 行业标准（各车企预标注管线） | 需先有 3D 模型；配合主动学习只审低置信度样本 |
| **2. 2D 基础模型 → 3D 提升** ⭐ | Grounding DINO / SAM2 出 2D 框/mask → 反投影点云 → 聚类成 3D mask / 拟合 bbox → 多视角一致性融合 | AutoLabel-2D (IEEE 2025) | **复用 Auto2dLabel 已有能力**，无需 3D 模型，最符合「先易后难」 |
| **3. 重建驱动（NeRF / 3DGS）** | 先重建场景，在重建上标一次 → 为所有帧生成一致标签，天然解决遮挡 | GS-Occ3D (ICCV 2025)、AutoOcc (ICCV 2025)、VESPA (CVPR 2026)、NeuRAD | 3DGS 已取代 NeRF 成为主流（可扩展性好）；质量最高但计算开销大、对消费级硬件不友好；动态物体需单独建模防拖影 |
| **4. 纯视觉 / 伪 LiDAR** | 单目/双目估深度生成伪点云 → 上面做标注 + VLM 校验闭环 | Pseudo-LiDAR + VLM (TRC 2026) | 有 LiDAR 走路线 2、无 LiDAR 走路线 3 时，路线 4 两头不讨好（详见调研讨论） |

**行业趋势**：从依赖 LiDAR 转向纯视觉管线（LiDAR 采集车成本高）；3DGS 取代 NeRF；VLM 驱动开放词汇标注；显式动态物体建模；光线投射区分 observed/unobserved 体素。

### 4. Auto3dLabel 路线建议

1. **MVP 走路线 2（2D → 3D 提升）**：已有 Grounding DINO + SAM2 + ByteTrack 全套 2D 能力，新增「点云加载 → 2D mask 反投影 → DBSCAN 聚类 → 多视角投票 → bbox 拟合」模块即可产出 3D bbox 初稿
2. **3D 输入**：先支持单帧点云 + 多相机（nuScenes / KITTI 格式），再扩展时序
3. **跟踪 ID + 运动属性**：跨帧 ID 维持与速度估计是 3D 标注最具价值的增量（衔接 Auto2dLabel v1.0）
4. **VLM 校验**：LLM 判断 bbox 是否贴合点云/图像，作为低成本高收益的审核环节（参考 VESPA 思路）
5. **3DGS 路线（路线 3）**：质量最高但需 24GB+ 显存 GPU 与长时间训练，列为远期（详见架构讨论，未写入本文件）
6. **输出格式**：目标支持 nuScenes / KITTI / Waymo Open Dataset 标注格式导出

### 5. 竞品分析：AutoLabel-2D（IEEE 2025）

论文《An Auto-Labeling tool for Occupancy Grid and BEV in Autonomous Driving Dataset》是「2D → 3D 提升」路线的真实实现，与本项目直接相关（连命名都接近）。

**技术管线**（据摘要推断）：

```text
输入: 多相机图像 + LiDAR 点云（含相机-LiDAR 外参标定）
  │
  ├─ [A] 2D 语义分割（多模型集成）
  │     Grounding DINO ──► 开放词汇 bbox ──┐
  │     YOLOv7        ──► 快速检测 bbox ────┼─► SAM2 ──► 2D mask
  │     GAI prompt 精炼 ◄── 失败区域反馈 ───┘  （边界/远距离优化）
  │
  ├─ [B] 投影：2D mask ──外参标定──► 语义 LiDAR 点云
  ├─ [C] 多帧融合 + 动静分离（位姿对齐累积；动态物体跟踪/场景流分离）
  ├─ [D] 占据网格：体素化 + 光线投射 ──► occupied/free/semantic 网格
  │
  输出: 语义点云 + 3D 语义占用网格 / BEV 图
```

**相同点**：基础模型（Grounding DINO + SAM2）完全重合；同走 2D → 3D 提升路线；同为模块化设计；同面向自动驾驶街景。

**关键区别**：

| 维度 | AutoLabel-2D | 本项目 AutoLabel |
| --- | --- | --- |
| 核心范式 | 固定 pipeline | Agentic 编排（LLM 规划/评估/请求人工） |
| LLM 角色 | 仅作 prompt 精炼模块 | 编排器 + HITL 分流 |
| 输出产物 | 语义点云 + 占用网格（HD 地图/场景理解） | 训练标签（COCO/YOLO bbox 等） |
| 人工介入 | 无（全自动） | HITL 三档分流 + 主动学习回流 |
| 模型策略 | 多模型集成（YOLOv7 + G-DINO 互补） | 47 模型 × 3 引擎可选切换 |
| 2D/3D 范围 | 纯 3D | 2D 六类任务 + 3D 预留 |

**对本项目的启示**：

1. 路线 2 可行性被验证：其阶段 A（G-DINO + SAM2）恰是 Auto2dLabel 已有能力
2. 差异即机会：它无 HITL、无训练标签导出、无 Agentic 编排——这三样正是本项目的差异化定位；其 pipeline 可作为 Auto3dLabel 的 no-LLM baseline，叠加 Agent 决策 + 人工审核闭环
3. YOLOv7 与 G-DINO 集成互补召回的做法值得借鉴，Auto3dLabel 阶段可考虑多模型投票融合
