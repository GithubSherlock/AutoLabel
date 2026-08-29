// Auto3dLabel v0.3 P4 逻辑层——零 three 依赖（node + jsdom 可测，见 tests/helpers/smoke_web3d.js）。
// 职责：队列状态机 + frame-data payload 拉取 + 选中/改标签/删除 + 保存体构建 +
// 列表/表格 HTML 纯函数 + 正交聚焦纯数学。渲染装配见 viewer3d.js（three）。
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
  };
  root.L3D = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
