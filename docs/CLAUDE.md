# AutoLabel

Agentic 数据标注工具：LLM Agent 编排 + 多模型引擎（检测/分割）+ HITL 三档分流 + 多格式导出。

## 项目结构

- `auto2dlabel/` — 2D 自动标注（v0.1 已完成：检测 + 实例分割 + 类别推荐 + Web 基础审核 + Benchmark）
- `auto3dlabel/` — 3D 标注（预留，仅 README + requirements，规划见 `AutoLabel_plan.md` 调研部分）
- `auto2dlabel/tests/` — **测试在包内**：`functional/` pytest 用例、`helpers/` 测试函数工具（no-LLM baseline / benchmark runner）、`data/` 样例图、实测数据文档；运行 `pytest auto2dlabel/tests/`

## 文档分工

- `AutoLabel_plan.md` — 只保留大纲：项目范围、版本路线、里程碑状态、已知缺口、Auto3dLabel 调研
- `auto2dlabel/CLAUDE.md` — 模型选型速查 + 设计红线（每次会话必读）
- `auto2dlabel/milestone/` — 各版本里程碑定义与完成记录；实测数据见 `auto2dlabel/tests/test-v0.1.md`、`test-v0.2.md`

## 常用命令

```bash
pytest auto2dlabel/tests/              # 跑测试（包内 functional/ + helpers/）
bash install_libs.sh                   # 交互式安装依赖
auto2dlabel chat "检测 000860.png 中的汽车" --no-wait   # 自然语言标注
python3 -m auto2dlabel.web.server      # Web 审核界面 → http://localhost:8765
```
