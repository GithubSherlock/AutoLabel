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

  // beginEdit(i, view, mode, drag)：view = 1 Top | 2 Side；mode = 'corner'|'edge'|'yaw'|
  // 'top'|'bottom'|'move'。drag = 视图层 mousedown 几何锚点（各编辑纯函数的 drag 参数）；
  // snapshot = 编辑前全量深拷贝（undo 锚点）。
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
    beginEdit, editTo, endEdit, pushUndo, undo, redo,
  };
  root.L3D = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
