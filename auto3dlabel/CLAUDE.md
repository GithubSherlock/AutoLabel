# auto3dlabel 开发指南

> Auto3dLabel = 3D 标注（Agentic 预标注层的 3D 分支）。当前 v0.1 📋 未动工——本文件是动工前的技术选型速查 + 设计红线。调研与路线论证见 `docs/AutoLabel_plan.md` Auto3dLabel 部分；里程碑定义与验收见 `milestone/v0.1.md`。

## 技术选型速查

| 环节 | 选型 | 说明 |
| --- | --- | --- |
| 2D 基础模型 | **复用 auto2dlabel**（G-DINO + SAM2/SAM3 + YOLO） | 零新增模型；直接 `from auto2dlabel.models import ...` |
| 点云加载 | numpy 直接解析（KITTI `.bin` / nuScenes pcd）；nuscenes-devkit 仅 GT/标定解析 | **不引入** mmdet3d / OpenPCDet（MVP 不需要） |
| 聚类 | DBSCAN（scipy） | eps 按距离自适应；备选 HDBSCAN |
| bbox 拟合 | 鸟瞰 `cv2.minAreaRect` + z 分位高度 | yaw 从矩形角度导出；导出时转 KITTI rotation_y / nuScenes 四元数（两者零位/方向约定不同） |
| 可视化 | open3d（可选）或 matplotlib BEV 散点 | 点云投影自检图必做 |
| 评测 | 3D IoU / BEV IoU 自写（GT 对齐） | benchmark 按 `docs/Benchmark_plan.md` 程序规范 |

## 设计红线

- **标定链是命门**：KITTI P2/R0_rect/Tr_velo_to_cam、nuScenes ego_pose/calibrated_sensor 任何一环错 → 投影全错且难察觉；必须做「点云投影回图像」自检图
- **反投影用 mask 不用 bbox**：bbox 边缘把背景点投进语义点云；SAM mask 才是像素级边界
- **LiDAR 观测性**：只反投影 LiDAR 实际打到的点；遮挡/远距目标点稀疏 → 拟合退化，输出「拟合点数」置信度供 HITL 分流（宁缺勿假）
- **yaw 约定**：内部统一 BEV (x,y) 角度；导出时显式转 KITTI 相机系 `rotation_y` / nuScenes 全局系 yaw，不得混用
- **复用不复制**：Agentic 编排（planner/orchestrator）、HITL 三档、质量评估模式复用 auto2dlabel 骨架；3D 新增代码只放 `auto3dlabel/`
- **纯本地**：延续 2D 红线——无外部 API；数据（KITTI / nuScenes mini）与权重不入库
- **先单帧后时序**：v0.1 只做单帧；跟踪 ID + 运动属性等 auto2dlabel v1.0 Tracking 交付后共用（ByteTrack/BoT-SORT）
