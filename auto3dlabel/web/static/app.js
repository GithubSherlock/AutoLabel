// Auto3dLabel v0.3 P4 复核前端薄壳：侧栏/图像/表格 DOM 挂载 + 状态机委托 logic3d.js。
// 状态与纯函数全在 L3D（jsdom 可测）；本文件只做 DOM 装配与事件绑定。
// 所有 fetch 用**相对路径**（"api/..."）：独立模式 :8766 与挂载模式 /3d/ 双兼容。
(function () {
  "use strict";
  const L3D = window.L3D;
  const $ = (sel) => document.querySelector(sel);

  function esc(s) { return L3D.esc(s); }

  async function loadFiles() {
    const res = await L3D.loadFiles();
    if (res.error) return;
    const html = L3D.renderFilesHtml(res);
    $("#dir").textContent = html.dir;
    $("#files").innerHTML = html.files;
    $("#reviewed").innerHTML = html.reviewed;
    document.querySelectorAll("#files .file-item, #reviewed .file-item").forEach((el) => {
      el.onclick = () => loadFrame(el.dataset.name);
    });
  }

  async function loadFrame(name) {
    const data = await L3D.loadFrame(name);
    if (data.error) {
      $("#detail").innerHTML = `<div class="empty">${esc(data.error)}</div>`;
      $("#view3d-wrap").style.display = "none";
      return;
    }
    document.querySelectorAll(".file-item").forEach((el) =>
      el.classList.toggle("active", el.dataset.name === name));
    // index.html 初始 display:none；成功打开才显示四视图（此前从未恢复 → 3D 画布不可见）
    $("#view3d-wrap").style.display = "block";
    renderDetail();
  }

  function renderDetail() {
    const d = L3D.state.data;
    const anns = d.annotations || [];
    // KITTI 有标定（payload.p2）→ 相机图加投影框叠加层；nuScenes 相机图为纯图
    const hasCalib = !!(L3D.state.payload && L3D.state.payload.p2);
    const cam = d.image_path
      ? `<figure>${hasCalib
          ? `<div class="cam-wrap"><img id="cam-img" src="api/review-image?path=${encodeURIComponent(d.image_path)}"><canvas id="cam-overlay"></canvas></div>`
          : `<img src="api/review-image?path=${encodeURIComponent(d.image_path)}">`
        }<figcaption>相机图${hasCalib ? " · 点/拖投影框选中并平移" : ""}</figcaption></figure>` : "";
    const bev = d.bev_path
      ? `<figure><img src="api/review-image?path=${encodeURIComponent(d.bev_path)}"><figcaption>BEV 鸟瞰</figcaption></figure>` : "";
    // nuScenes 6 相机：image_path 只存在于 frame-data payload（_nuscenes_payload 计算
    // dataroot/filename），队列文件 cameras 仅 {name,token,filename}——从 payload 读，否则 path=undefined
    const cams = (((L3D.state.payload || {}).cameras) || []).map((c) =>
      `<figure><img src="api/review-image?path=${encodeURIComponent(c.image_path)}"><figcaption>${esc(c.name)}</figcaption></figure>`).join("");
    $("#detail").innerHTML = `
      <h3 style="margin-top:0">${esc(d.image || L3D.state.name)} <span style="color:var(--dim);font-weight:400">— ${anns.length} 框待复核</span></h3>
      <div class="imgs">${cam}${bev}${cams}</div>
      <div id="table-wrap">${L3D.renderTableHtml()}</div>
      <div class="toolbar">
        <button id="save">保存复核结果</button>
        <span id="status"></span>
      </div>`;
    bindTable();
    $("#save").onclick = save;
    bindCamOverlay();
  }

  // ── v1.0 相机图叠加：投影框显示 + 点击选中 + 拖动平移（view=3 'move' 编辑）──
  function camNativePoint(e) {
    const img = $("#cam-img");
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    return {
      u: (e.clientX - rect.left) * (img.naturalWidth / (rect.width || 1)),
      v: (e.clientY - rect.top) * (img.naturalHeight / (rect.height || 1)),
    };
  }

  function distToSeg(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function camHitTest(u, v) {
    const p2 = L3D.state.payload && L3D.state.payload.p2;
    if (!p2) return null;
    const tol = 8; // 原生像素命中阈值
    let best = null, bestD = tol;
    (L3D.state.data.annotations || []).forEach((ann, i) => {
      if (L3D.state.deleted.has(i) || ann.cx == null) return;
      const c2 = L3D.boxCorners2d(p2, L3D.boxToCorners(ann));
      for (const e of L3D.edgesOf(c2)) {
        const [a, b] = e;
        if (!a || !b) continue;
        const d = distToSeg(u, v, a[0], a[1], b[0], b[1]);
        if (d < bestD) { bestD = d; best = i; }
      }
    });
    return best;
  }

  function drawCamOverlay() {
    const ov = $("#cam-overlay"), img = $("#cam-img");
    const p2 = L3D.state.payload && L3D.state.payload.p2;
    if (!ov || !img || !p2) return;
    const ctx = ov.getContext("2d");
    if (!ctx) return; // jsdom 无 canvas 实现：绘制静默跳过（结构断言靠 DOM）
    const sx = ov.width / (img.naturalWidth || 1); // 原生像素 → 画布像素
    ctx.setTransform(sx, 0, 0, sx, 0, 0);
    ctx.clearRect(0, 0, img.naturalWidth, img.naturalHeight);
    (L3D.state.data.annotations || []).forEach((ann, i) => {
      if (ann.cx == null) return;
      const del = L3D.state.deleted.has(i);
      const sel = L3D.state.selected === i;
      const c2 = L3D.boxCorners2d(p2, L3D.boxToCorners(ann));
      ctx.strokeStyle = del ? "rgba(229,83,75,.55)"
        : (sel ? "#ffd166" : "#" + L3D.colorFor(ann.label).toString(16).padStart(6, "0"));
      ctx.lineWidth = (del ? 1 : 2) / sx;
      ctx.setLineDash(del ? [6, 4] : []);
      ctx.beginPath();
      for (const e of L3D.edgesOf(c2)) {
        const [a, b] = e;
        if (!a || !b) continue;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (!del && c2[0]) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = `${12 / sx}px sans-serif`;
        ctx.fillText(ann.label, c2[0][0] + 3 / sx, c2[0][1] - 3 / sx);
      }
    });
  }

  let camDrag = null; // {cy, x0, y0, moved}：相机图拖动会话（view=3 'move' 编辑）
  let camResizeHandler = null;
  function bindCamOverlay() {
    const ov = $("#cam-overlay"), img = $("#cam-img");
    if (!ov) return;
    const fit = () => {
      const rect = img.getBoundingClientRect();
      ov.width = rect.width || 1;
      ov.height = rect.height || 1;
      drawCamOverlay();
    };
    img.onload = fit;
    if (camResizeHandler) window.removeEventListener("resize", camResizeHandler);
    camResizeHandler = fit;
    window.addEventListener("resize", camResizeHandler);
    fit();

    ov.onmousedown = (e) => {
      const p = camNativePoint(e);
      if (!p) return;
      const i = camHitTest(p.u, p.v);
      if (i == null) return;
      L3D.selectObject(i);
      const ann = L3D.state.data.annotations[i];
      const pl = L3D.state.payload;
      const camCenter = pl.cam_center || [0, 0, 0]; // 旧 payload 兜底（零平移近似）
      const g0 = L3D.p2ToGround(pl.k_inv, camCenter, ann.cy, p.u, p.v);
      if (!g0) return; // 点在地平线附近无地面解
      L3D.beginEdit(i, 3, "move", { cx0: ann.cx, cz0: ann.cz, x0: g0[0], z0: g0[1] });
      camDrag = { cy: ann.cy, x0: e.clientX, y0: e.clientY, moved: false };
    };
    ov.onmousemove = (e) => {
      if (!camDrag) return;
      const p = camNativePoint(e);
      if (!p) return;
      if (!camDrag.moved && Math.hypot(e.clientX - camDrag.x0, e.clientY - camDrag.y0) > 3) {
        camDrag.moved = true; // 3px 阈值区分点击/拖动
      }
      if (!camDrag.moved) return;
      const pl = L3D.state.payload;
      const g = L3D.p2ToGround(pl.k_inv, pl.cam_center || [0, 0, 0], camDrag.cy, p.u, p.v);
      if (!g) return;
      L3D.editTo(3, "move", { x: g[0], z: g[1] });
    };
    const up = () => {
      if (camDrag) { L3D.endEdit(); camDrag = null; }
    };
    ov.onmouseup = up;
    ov.onmouseleave = up; // 拖出画布结束编辑（防 mouseup 落空）
  }

  function bindTable() {
    $("#table-wrap").querySelectorAll("tr").forEach((tr) => {
      tr.onclick = (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "BUTTON") return;
        const i = Number(tr.dataset.i ?? tr.querySelector("[data-i]")?.dataset.i);
        if (!Number.isNaN(i)) L3D.selectObject(i);
      };
    });
    $("#table-wrap").querySelectorAll("button.del").forEach((b) => {
      b.onclick = () => L3D.toggleDelete(Number(b.dataset.i));
    });
    // 双击类别单元格 → inline 改标签（保存时经 buildSaveBody 全量回写）
    $("#table-wrap").querySelectorAll("td.label").forEach((td) => {
      td.ondblclick = () => {
        const i = Number(td.dataset.i);
        const input = document.createElement("input");
        input.value = L3D.state.data.annotations[i].label;
        td.textContent = "";
        td.appendChild(input);
        input.focus();
        const commit = () => {
          L3D.changeLabel(i, input.value.trim() || L3D.state.data.annotations[i].label);
        };
        input.onkeydown = (e) => { if (e.key === "Enter") input.blur(); };
        input.onblur = commit;
      };
    });
  }

  async function save() {
    const res = await fetch("api/review-save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(L3D.buildSaveBody()),
    });
    const out = await res.json();
    $("#status").textContent = out.ok
      ? `已保存：保留 ${out.kept} 框，删除 ${out.deleted} 框 → ${out.saved_path}`
      : `保存失败: ${out.error || res.status}`;
    if (out.ok) { $("#detail").innerHTML = '<div class="empty">已保存 ✓</div>'; loadFiles(); }
  }

  // 订阅状态机：选中/删除/改标签/编辑 → 表格重渲染 + 相机图投影框跟随（四视图 canvas 不动）
  L3D.on((state) => {
    if (!state.data || !$("#table-wrap")) return;
    $("#table-wrap").innerHTML = L3D.renderTableHtml();
    bindTable();
    drawCamOverlay();
  });

  loadFiles();
})();
