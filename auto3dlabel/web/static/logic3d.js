// Auto3dLabel v0.3 P4 逻辑层——零 three 依赖（node + jsdom 可测，见 tests/helpers/smoke_web3d.js）。
// 职责：队列状态机 + frame-data payload 拉取 + 选中/改标签/删除 + 保存体构建 +
// 列表/表格 HTML 纯函数 + 正交聚焦纯数学 + v0.4 P1 cuboid 手柄编辑（纯函数 + 状态机 + undo）。
// 渲染装配见 viewer3d.js（three）。
// 所有 fetch 用**相对路径**（"api/..."）：独立模式 :8766 与挂载模式 /3d/ 双兼容
// （绝对 /api/... 在挂载模式下会打到 2D 端点——协议同但数据格式不同，禁止）。
(function (root) {
  "use strict";

  const state = {
    name: null,     // 当前队列文件名
    data: null,     // review-file 原始 JSON（表格/保存用）
    payload: null,  // frame-data payload（点云 + 角点，objects 下标 = annotations 下标）
    deleted: new Set(),
    selected: null, // 对象下标（annotations 序；null = 未选中）
    editing: null,  // v0.4 P1 手柄编辑态：{i, view, mode, snapshot, cur} | null
  };
  const listeners = [];
  const on = (fn) => listeners.push(fn);
  const emit = () => listeners.forEach((fn) => fn(state));

  // ── 纯数学 ─────────────────────────────────────────────────
  // 相机系 (x 右, y 下, z 前) → three 渲染系 (x, z, -y)：地面 (x,z) 平面为 X-Y 平面，
  // 高度 y 取负为深度（front 视图向 -z 看即相机向前）。
  const camToThree = (x, y, z) => [x, z, -y];

  // 8 角（corners_cam 序：低 0-3 [右前,右后,左后,左前] 高 4-7 同序）→ 12 边下标对
  const EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const edgesOf = (corners) => EDGES.map(([a, b]) => [corners[a], corners[b]]);

  // 类别颜色（单一事实源：viewer3d 渲染 / app.js 相机图叠加共用）
  const COLOR_MAP = { Car: 0x4da3ff, Pedestrian: 0x58d68d, Cyclist: 0xe8b339 };
  const SEL_COLOR = 0xffd166;
  const colorFor = (label) => COLOR_MAP[label] || 0xb0b6c0;

  // 车头方向线（BEV 平面）：前边（角 0/3）中点 → 车头单位向量 → [前端点, 伸出点]（相机系）
  function headLine(corners) {
    const cx = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
    const cz = (corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2]) / 4;
    const cy = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
    const fx = (corners[0][0] + corners[3][0]) / 2;
    const fz = (corners[0][2] + corners[3][2]) / 2;
    const dx = fx - cx, dz = fz - cz;
    const norm = Math.hypot(dx, dz);
    if (norm < 1e-9) return null; // 退化框（零长度）
    const ux = dx / norm, uz = dz / norm;
    const ext = norm * 1.5; // 前边中点再伸出 1.5×半长
    return [
      [fx, cy, fz],
      [fx + ux * ext, cy, fz + uz * ext],
    ];
  }

  // 正交聚焦（CVAT 借鉴）：换轴后中心 + 包围半径（正交视图对齐 + 相机距离的依据）
  function orthoFocus(corners) {
    const t = corners.map((c) => camToThree(c[0], c[1], c[2]));
    const n = t.length;
    const cx = t.reduce((s, p) => s + p[0], 0) / n;
    const cy = t.reduce((s, p) => s + p[1], 0) / n;
    const cz = t.reduce((s, p) => s + p[2], 0) / n;
    let r = 0;
    for (const p of t) r = Math.max(r, Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz));
    return { center: [cx, cy, cz], radius: Math.max(r, 0.5) };
  }

  // ── v1.0 相机图叠加投影纯函数（KITTI rect：P2 3x4、K⁻¹ 3x3，payload 交付）──
  // 投影：cam 系点 → 像素 [u,v]；w≤0.1（相机后方/焦平面）判无效返回 null。
  function projectP2(p2, xyz) {
    const x = xyz[0], y = xyz[1], z = xyz[2];
    const w = p2[2][0] * x + p2[2][1] * y + p2[2][2] * z + p2[2][3];
    if (w <= 0.1) return null;
    return [
      (p2[0][0] * x + p2[0][1] * y + p2[0][2] * z + p2[0][3]) / w,
      (p2[1][0] * x + p2[1][1] * y + p2[1][2] * z + p2[1][3]) / w,
    ];
  }

  // 反投影：像素 (u,v) 射线（起点相机中心 camCenter、方向 K⁻¹[u,v,1]）∩ 地面平面
  // y=cy → cam 系 [x,z]；射线近平行地面（|dy|<1e-9，点在地平线）无解返回 null。
  // 精确解（含 P2 非零平移列的相机中心项，非雅可比近似）。
  function p2ToGround(kInv, camCenter, cy, u, v) {
    const d0 = kInv[0][0] * u + kInv[0][1] * v + kInv[0][2];
    const d1 = kInv[1][0] * u + kInv[1][1] * v + kInv[1][2];
    const d2 = kInv[2][0] * u + kInv[2][1] * v + kInv[2][2];
    if (Math.abs(d1) < 1e-9) return null;
    const t = (cy - camCenter[1]) / d1;
    return [camCenter[0] + t * d0, camCenter[2] + t * d2];
  }

  // 框 8 角投影：corners_cam 8x3 → 8×[u,v]|null（后方角 null；边绘制跳过含 null 的边）
  function boxCorners2d(p2, corners) {
    return corners.map((c) => projectP2(p2, c));
  }

  // 相机图拖动：ptr = {x,z}（cam 地面坐标，p2ToGround 反投影），drag 起点为锚平移 cx/cz
  function moveGroundEdit(c, drag, ptr) {
    c.cx = drag.cx0 + (ptr.x - drag.x0);
    c.cz = drag.cz0 + (ptr.z - drag.z0);
  }

  // ── v1.0 P2 相机图叠加几何纯函数（app.js 绘制 + smoke 数值断言共用）──
  // 图像范围 [0,w]×[0,h]；投影坐标可为 null（相机后方角）。
  // 投影线两端的 null 处理：任一端 null → 该边不可画（穿越焦平面的边极少且无良定义）。

  // Liang-Barsky 线段裁剪：a/b = [u,v] 投影点（非 null）→ 图像矩形内段 [x1,y1,x2,y2]；
  // 完全在图像外 → null（app.js 跳过该边）。
  function clipSegImage(a, b, w, h) {
    let t0 = 0, t1 = 1;
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const p = [-dx, dx, -dy, dy];
    const q = [a[0], w - a[0], a[1], h - a[1]];
    for (let k = 0; k < 4; k++) {
      if (Math.abs(p[k]) < 1e-9) {
        if (q[k] < 0) return null; // 平行且在边界外
      } else {
        const r = q[k] / p[k];
        if (p[k] < 0) { if (r > t1) return null; t0 = Math.max(t0, r); }
        else { if (r < t0) return null; t1 = Math.min(t1, r); }
      }
    }
    return [a[0] + t0 * dx, a[1] + t0 * dy, a[0] + t1 * dx, a[1] + t1 * dy];
  }

  // 相机图叠加几何：框 corners_cam 8x3 → {segs, handles, indicator}
  // - segs: 12 边裁剪到图像后的段 [[x1,y1,x2,y2]...]（无 null 端点才裁剪）
  // - handles: 底角 4（corner，sx/sy 同 Top 视图 CORNER_SIGN）+ 底边中点 4（edge，
  //   EDGE_SIGN），仅当该手柄真实投影在图像内（视野外手柄拖动反投影无良定义 → 不画）
  // - indicator: 框完全出界时最近图像边缘的指示标记 {u,v,dir}（dir: 'l'|'r'|'t'|'b'）；
  //   部分可见 → null。v/u 用有效角投影均值 clamp 到边缘内。
  function camOverlayGeom(p2, ann, w, h) {
    const corners = boxToCorners(ann);
    const c2 = boxCorners2d(p2, corners);
    const segs = [];
    for (const e of edgesOf(c2)) {
      const [a, b] = e;
      if (!a || !b) continue;
      const s = clipSegImage(a, b, w, h);
      if (s) segs.push(s);
    }
    // 手柄：底角 4（corners 4-7 = 底面）+ 底边中点 4（右/尾/左/头，同 viewer3d EDGE_PAIRS）
    const CORNER_SIGN = [[1, 1], [1, -1], [-1, -1], [-1, 1]]; // 右前/右后/左后/左前
    const EDGE_PAIRS = [[4, 5], [5, 6], [6, 7], [7, 4]];      // 右/尾/左/头（底面）
    const EDGE_SIGN = [[1, 0], [0, -1], [-1, 0], [0, 1]];
    const inImg = (p) => p && p[0] >= 0 && p[0] <= w && p[1] >= 0 && p[1] <= h;
    const handles = [];
    for (let i = 0; i < 4; i++) {
      const p = c2[4 + i];
      if (inImg(p)) handles.push({ u: p[0], v: p[1], mode: "corner", sx: CORNER_SIGN[i][0], sy: CORNER_SIGN[i][1] });
    }
    for (let i = 0; i < 4; i++) {
      const [ai, bi] = EDGE_PAIRS[i];
      const a = c2[ai], b = c2[bi];
      if (!a || !b) continue;
      const p = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      if (inImg(p)) handles.push({ u: p[0], v: p[1], mode: "edge", sx: EDGE_SIGN[i][0], sy: EDGE_SIGN[i][1] });
    }
    // 指示标记：segs 为空但框有效投影存在（至少一角可投影）→ 最近边缘
    let indicator = null;
    if (segs.length === 0) {
      const vis = c2.filter((p) => p !== null);
      if (vis.length) {
        const u = vis.reduce((s, p) => s + p[0], 0) / vis.length;
        const v = vis.reduce((s, p) => s + p[1], 0) / vis.length;
        const du = u < 0 ? -u : (u > w ? u - w : Infinity);
        const dv = v < 0 ? -v : (v > h ? v - h : Infinity);
        if (du <= dv) indicator = { u: u < 0 ? 0 : w, v: Math.max(0, Math.min(h, v)), dir: u < 0 ? "l" : "r" };
        else indicator = { u: Math.max(0, Math.min(w, u)), v: v < 0 ? 0 : h, dir: v < 0 ? "t" : "b" };
      }
    }
    return { segs, handles, indicator };
  }

  // 相机图手柄命中（仅选中框的手柄）：返回手柄对象或 null（角点 10px 优先于边中点 8px）
  function camHitHandle(geom, u, v) {
    let best = null, bestD = Infinity;
    for (const hd of geom.handles) {
      const d = Math.hypot(hd.u - u, hd.v - v);
      const tol = hd.mode === "corner" ? 10 : 8;
      if (d < tol && (d < bestD || (d < bestD + 1e-9 && hd.mode === "corner"))) {
        bestD = d; best = hd;
      }
    }
    return best;
  }

  // 相机图指示标记命中（视野外框选中入口）：{u,v} 12px 内
  function camHitIndicator(geoms, u, v) {
    let best = null, bestD = 12;
    geoms.forEach((g, i) => {
      if (!g.indicator) return;
      const d = Math.hypot(g.indicator.u - u, g.indicator.v - v);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  }

  // ── v0.4 P1 cuboid 编辑纯函数 ───────────────────────────────
  // 相机系与 Box3D 语义对齐（x 右 y 下 z 前）；Top 视图平面 = (x, z)，
  // Side 视图平面 = (z, y)。yaw↔rotation_y 照 tools/geometry.py 唯一转换公式
  // （wrap 到 [-π, π]；产品导出链路不动，此为编辑渲染必要转换）。

  function wrapPi(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a <= -Math.PI) a += 2 * Math.PI;
    return a;
  }

  function rotationYToYaw(ry) {
    return wrapPi(ry + Math.PI / 2);
  }

  function yawToRotationY(yaw) {
    return wrapPi(yaw - Math.PI / 2);
  }

  // annotations dict（cx/cy/cz/h/w/l/rotation_y）→ corners_cam 8x3（相机系）。
  // 公式照 box3d.py corners_bev/corners_cam：车头 (sin yaw, cos yaw)、右向 (dz, −dx)。
  function boxToCorners(b) {
    const yaw = rotationYToYaw(b.rotation_y || 0);
    const dx = Math.sin(yaw), dz = Math.cos(yaw);
    const px = dz, pz = -dx;
    const hl = b.l / 2, hw = b.w / 2;
    const cx = b.cx, cz = b.cz;
    const f = [
      [cx + dx * hl + px * hw, cz + dz * hl + pz * hw], // 右前
      [cx - dx * hl + px * hw, cz - dz * hl + pz * hw], // 右后
      [cx - dx * hl - px * hw, cz - dz * hl - pz * hw], // 左后
      [cx + dx * hl - px * hw, cz + dz * hl - pz * hw], // 左前
    ];
    const yLow = b.cy - b.h / 2, yHigh = b.cy + b.h / 2;
    return [
      [f[0][0], yLow, f[0][1]], [f[1][0], yLow, f[1][1]],
      [f[2][0], yLow, f[2][1]], [f[3][0], yLow, f[3][1]],
      [f[0][0], yHigh, f[0][1]], [f[1][0], yHigh, f[1][1]],
      [f[2][0], yHigh, f[2][1]], [f[3][0], yHigh, f[3][1]],
    ];
  }

  // 2D 编辑母版（auto2dlabel app.js resize/rotate）在 3D 的适配：
  // BEV 编辑域 = Top 平面 (x,z) 上的 2D 框 {中心 (cx,cz), w=width, l=height}，
  // 2D angle t = **−yaw**（yaw=0 车头 +z → t=0：宽轴 (1,0)=右向 ✓、高轴 (0,1)=车头 ✓；
  // yaw=π/2 车头 +x → t=−π/2：宽轴 (0,−1) ✓、高轴 (1,0) ✓）
  const MIN_BOX3D = 0.3; // 米级最小尺寸（同 2D MIN_BOX 语义）

  // Top 视图 resize：对角固定 + 手柄吸附（2D 母版公式原样，w/l 齐变或单轴）
  // ptr = {x, z}（cam 系 Top 平面）；drag = {t0, cx0, cz0, w0, l0, sx, sy}
  function resizeTopEdit(b, drag, ptr) {
    const cos = Math.cos(drag.t0), sin = Math.sin(drag.t0);
    const rx = ptr.x - drag.cx0, ry = ptr.z - drag.cz0;
    const plx = rx * cos + ry * sin;   // 指针沿宽轴（w 方向）
    const ply = -rx * sin + ry * cos;  // 指针沿长轴（l 方向）
    const fx = -drag.sx * drag.w0 / 2, fy = -drag.sy * drag.l0 / 2; // 对角局部
    const w1 = drag.sx !== 0 ? Math.max(drag.sx * (plx - fx), MIN_BOX3D) : drag.w0;
    const l1 = drag.sy !== 0 ? Math.max(drag.sy * (ply - fy), MIN_BOX3D) : drag.l0;
    const clx = drag.sx !== 0 ? (plx + fx) / 2 : 0;
    const cly = drag.sy !== 0 ? (ply + fy) / 2 : 0;
    b.cx = drag.cx0 + clx * cos - cly * sin;
    b.cz = drag.cz0 + clx * sin + cly * cos;
    b.w = w1;
    b.l = l1;
  }

  // Top 视图 yaw 旋转：**增量式**（3D 车头有向，不能用 2D 无向框的绝对角 ±90° 歧义）
  // drag = {yaw0, t0}（t0 = mousedown 指针方位角，cam Top 平面）
  function rotateYawEdit(b, drag, ptr) {
    const t1 = Math.atan2(ptr.z - drag.cz0, ptr.x - drag.cx0);
    b.yaw = wrapPi(drag.yaw0 + (t1 - drag.t0));
  }

  // Side 视图 resize（垂直平面 (z,y)，一维编辑，对面固定）：
  // 顶面手柄 → h += −δy_cam、cy += δy_cam/2；底面手柄 → h += +δy_cam、cy += δy_cam/2
  // （cam y 向下：屏幕 up 拖动 = y 减小 = 框变高）
  // drag = {mode: 'top'|'bottom', h0, cy0}
  function resizeSideEdit(b, drag, ptr) {
    const dy = drag.mode === 'top' ? -(ptr.y - drag.cy0) : (ptr.y - drag.cy0);
    b.h = Math.max(drag.h0 + dy, MIN_BOX3D);
    b.cy = drag.cy0 + dy / 2;
  }

  // Side 视图前后平移（cz 沿 z 挪整框，l 不动——车头轴非 z 轴时沿 z 缩 l 无意义，
  // 长度编辑由 Top 视图车头/车尾边中点手柄覆盖）
  function moveSideEdit(b, drag, ptr) {
    b.cz = drag.cz0 + (ptr.z - drag.cz0);
  }

  // ── HTML 纯函数（jsdom 断言友好）────────────────────────────
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function renderFilesHtml(res) {
    const files = (res.files || []).map((f) => `
      <div class="file-item" data-name="${esc(f.name)}">
        <div class="name">${esc(f.image_stem || f.name)}</div>
        <div class="meta">${f.count} 框</div>
      </div>`).join("") || '<div class="empty">无待复核队列</div>';
    const reviewed = (res.reviewed || []).map((f) => `
      <div class="file-item" data-name="${esc(f.name)}">
        <div class="name">${esc(f.image_stem || f.name)}</div>
        <div class="meta">${f.count} 框 · 已复核</div>
      </div>`).join("") || '<div class="empty">无</div>';
    return { files, reviewed, dir: res.dir || "" };
  }

  function renderTableHtml() {
    const d = state.data;
    if (!d) return "";
    const anns = d.annotations || [];
    const rows = anns.map((a, i) => {
      const del = state.deleted.has(i);
      const sel = state.selected === i ? ' class="sel"' : "";
      const flag = a.review_flag ? ' <span class="flag">⚠ review</span>' : "";
      const dims = `${a.h?.toFixed(2)}×${a.w?.toFixed(2)}×${a.l?.toFixed(2)}`;
      const rot = a.rotation_y == null ? "—" : a.rotation_y.toFixed(2);
      return `<tr${sel} style="${del ? "opacity:.35" : ""}">
        <td>${i}</td>
        <td class="label" data-i="${i}" title="双击改标签">${esc(a.label)}</td>
        <td class="num">${(a.confidence ?? 0).toFixed(2)}</td>
        <td class="num">${esc(dims)}</td>
        <td class="num">${rot}${flag}</td>
        <td class="num">${a.fit_points ?? "—"}</td>
        <td><button class="del" data-i="${i}">${del ? "恢复" : "删除"}</button></td>
      </tr>`;
    }).join("");
    return `
      <table>
        <tr><th>#</th><th>类别</th><th>conf</th><th>h×w×l (m)</th><th>rotation_y</th><th>拟合点数</th><th></th></tr>
        ${rows || '<tr><td colspan="7" class="empty">无框</td></tr>'}
      </table>`;
  }

  // ── 状态机 ─────────────────────────────────────────────────
  async function loadFiles() {
    const r = await fetch("api/review-files");
    const res = await r.json();
    if (!res.error) emit();
    return res;
  }

  async function loadFrame(name) {
    const r = await fetch("api/review-file?name=" + encodeURIComponent(name));
    const data = await r.json();
    if (data.error) {
      // 加载失败清状态（防残留上一帧 payload 与选中态）
      state.name = null;
      state.data = null;
      state.payload = null;
      state.deleted = new Set();
      state.selected = null;
      emit();
      return data;
    }
    const pr = await fetch("api/frame-data?name=" + encodeURIComponent(name));
    const payload = await pr.json();
    state.name = name;
    state.data = data;
    state.payload = payload.error ? null : payload;
    state.deleted = new Set();
    state.selected = null;
    emit();
    return data;
  }

  function selectObject(i) {
    state.selected = i;
    emit();
  }

  function changeLabel(i, label) {
    if (!state.data) return;
    const a = (state.data.annotations || [])[i];
    if (!a) return;
    a.label = label;
    a.edited_by_human = true;
    emit();
  }

  function toggleDelete(i) {
    state.deleted.has(i) ? state.deleted.delete(i) : state.deleted.add(i);
    emit();
  }

  // ── v0.4 P1 编辑状态机 + undo/redo（2D 闭包栈模式，MAX 32 会话级）──
  const undoStack = [];
  const redoStack = [];
  const MAX_UNDO = 32;

  function pushUndo(name, undoFn, redoFn) {
    undoStack.push({ name, undo: undoFn, redo: redoFn });
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack.length = 0; // 新操作清重做栈
  }

  function undo() {
    const op = undoStack.pop();
    if (!op) return;
    op.undo();
    redoStack.push(op);
    emit();
  }

  function redo() {
    const op = redoStack.pop();
    if (!op) return;
    op.redo();
    undoStack.push(op);
    emit();
  }

  // 编辑写回 annotations[i]（几何键 + rotation_y 派生）——编辑体操作队列 JSON 原字段
  // （buildSaveBody 全量直通，后端零改动）。人工标记在 endEdit 确有几何变化时置位
  // （点击手柄零拖动不产生假标记）。
  function _applyEdit(i, cur) {
    const ann = state.data.annotations[i];
    ann.cx = cur.cx; ann.cy = cur.cy; ann.cz = cur.cz;
    ann.h = cur.h; ann.w = cur.w; ann.l = cur.l;
    ann.rotation_y = yawToRotationY(cur.yaw);
  }

  // beginEdit(i, view, mode, drag)：view = 1 Top | 2 Side | 3 相机图；mode = 'corner'|'edge'|
  // 'yaw'|'top'|'bottom'|'move'（view 3 仅 'move'：地面平移）。drag = 视图层 mousedown
  // 几何锚点（各编辑纯函数的 drag 参数）；snapshot = 编辑前全量深拷贝（undo 锚点）。
  function beginEdit(i, view, mode, drag) {
    if (!state.data || !state.data.annotations[i]) return;
    const ann = state.data.annotations[i];
    const cur = {
      cx: ann.cx, cy: ann.cy, cz: ann.cz,
      h: ann.h, w: ann.w, l: ann.l,
      yaw: rotationYToYaw(ann.rotation_y || 0),
    };
    state.editing = {
      i, view, mode,
      snapshot: JSON.parse(JSON.stringify(ann)),
      cur,
      drag: drag || null,
    };
    emit();
  }

  // editTo(view, mode, ptr)：ptr = cam 系平面坐标（Top {x,z} / Side {z,y}）。
  // 按 view/mode 调编辑纯函数 → cur 更新 → 回写 annotations[i] → emit。
  function editTo(view, mode, ptr) {
    const e = state.editing;
    if (!e) return;
    const c = e.cur;
    if (view === 1) {
      if (mode === 'corner' || mode === 'edge') {
        resizeTopEdit(c, e.drag, ptr);
      } else if (mode === 'yaw') {
        rotateYawEdit(c, e.drag, ptr);
      }
    } else if (view === 2) {
      if (mode === 'top' || mode === 'bottom') {
        resizeSideEdit(c, e.drag, ptr);
      } else if (mode === 'move') {
        moveSideEdit(c, e.drag, ptr);
      }
    } else if (view === 3) {
      if (mode === 'move') {
        moveGroundEdit(c, e.drag, ptr);
      } else if (mode === 'corner' || mode === 'edge') {
        resizeTopEdit(c, e.drag, ptr); // 相机图角/边拖动 = 地面平面 resize（公式同 Top 视图）
      }
    }
    c.yaw = wrapPi(c.yaw);
    _applyEdit(e.i, c);
    emit();
  }

  // endEdit()：编辑结束；几何键与 snapshot 有变化（容差 1e-9——yaw↔rotation_y 往返
  // 有浮点噪声，(x+π/2)-π/2 ≠ x 严格不成立）→ 置人工标记 + 入 undo 栈（闭包捕获局部
  // 副本；零变化编辑不置位不入栈）。
  const GEOM_KEYS = ["cx", "cy", "cz", "h", "w", "l", "rotation_y"];
  function endEdit() {
    const e = state.editing;
    if (!e) return;
    state.editing = null;
    const ann = state.data.annotations[e.i];
    if (GEOM_KEYS.some((k) => Math.abs(ann[k] - e.snapshot[k]) > 1e-9)) {
      ann.edited_by_human = true;
      const snap = e.snapshot, fin = JSON.parse(JSON.stringify(ann));
      const i = e.i;
      pushUndo(
        "edit3d",
        () => { state.data.annotations[i] = snap; emit(); },
        () => { state.data.annotations[i] = fin; emit(); },
      );
    } else {
      // 零变化（点击手柄零拖动）：恢复 snapshot 原 dict，清掉 editTo 的浮点往返噪声
      state.data.annotations[e.i] = JSON.parse(JSON.stringify(e.snapshot));
    }
    emit();
  }

  // 保存体：edited 全量重建（保留框原样直通 + 人工复核标记），删除框剔除
  function buildSaveBody() {
    const anns = (state.data && state.data.annotations) || [];
    const edited = anns
      .map((a, i) => ({ ...a, edited_by_human: true }))
      .filter((_, i) => !state.deleted.has(i));
    return { queue_file: state.name, edited };
  }

  const api = {
    state, on, camToThree, EDGES, edgesOf, headLine, orthoFocus, esc,
    renderFilesHtml, renderTableHtml,
    loadFiles, loadFrame, selectObject, changeLabel, toggleDelete, buildSaveBody,
    // v0.4 P1 编辑
    wrapPi, rotationYToYaw, yawToRotationY, boxToCorners,
    resizeTopEdit, rotateYawEdit, resizeSideEdit, moveSideEdit, MIN_BOX3D,
    projectP2, p2ToGround, boxCorners2d, moveGroundEdit,
    clipSegImage, camOverlayGeom, camHitHandle, camHitIndicator,
    COLOR_MAP, SEL_COLOR, colorFor,
    beginEdit, editTo, endEdit, pushUndo, undo, redo,
  };
  root.L3D = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
