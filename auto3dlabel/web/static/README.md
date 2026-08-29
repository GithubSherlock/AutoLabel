# static/ 第三方资源溯源

## three.min.js（v0.3 P4 四视图渲染）

- **来源**：https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js
- **版本**：three r149（2023-03；r150 起官方不再发布 min 构建版）
- **许可证**：MIT
- **sha256**：`8a5f7249903b54d30f79f708699d2fed2d6a1d0741a4cd41377d1f01bb5a2271`
- **大小**：608,081 字节
- **使用方式**：`<script src="static/three.min.js">`（相对路径——挂载 /3d/ 与独立 :8766 双兼容）
- **不依赖**：四视图 scissor/OrbitControls 未引入——旋转/缩放为手写球坐标交互（viewer3d.js），零额外文件

其余文件（logic3d.js / viewer3d.js / app.js / index.html）为本项目源码，非第三方。
