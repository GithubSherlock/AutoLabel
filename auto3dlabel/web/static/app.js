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
    // 相机图列表：KITTI 主相机（image_path）+ nuScenes 6 相机（payload.cameras，
    // image_path 只存在于 frame-data payload——队列文件 cameras 仅 {name,token,filename}）
    const camFigs = [];
    if (d.image_path) camFigs.push({ cap: "相机图", src: d.image_path });
    ((L3D.state.payload || {}).cameras || []).forEach((c) =>
      camFigs.push({ cap: c.name, src: c.image_path }));
    const camHtml = camFigs.map((f, i) => {
      const params = L3D.camProjParams(L3D.state.payload, i);
      const img = `<img id="cam-img-${i}" src="api/review-image?path=${encodeURIComponent(f.src)}">`;
      return `<figure>${params
        ? `<div class="cam-wrap">${img}<canvas class="cam-overlay" id="cam-overlay-${i}"></canvas></div>`
        : img}<figcaption>${esc(f.cap)}${params
          ? " · 点框选中/拖动平移 · 拖角点边缩放 · 边缘三角=视野外框" : ""}</figcaption></figure>`;
    }).join("");
    const bev = d.bev_path
      ? `<figure><img src="api/review-image?path=${encodeURIComponent(d.bev_path)}"><figcaption>BEV 鸟瞰</figcaption></figure>` : "";
    $("#detail").innerHTML = `
      <h3 style="margin-top:0">${esc(d.image || L3D.state.name)} <span style="color:var(--dim);font-weight:400">— ${anns.length} 框待复核</span></h3>
      <div class="imgs">${camHtml}${bev}</div>
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
  // 多相机参数化（KITTI 1 相机 / nuScenes 6 相机）：每图 overlay 独立 p2/k_inv/cam_center。
  function camNativePoint(e, img) {
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

  function camHitTest(u, v, p2) {
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

  function drawCamOverlay(idx) {
    const ov = document.getElementById("cam-overlay-" + idx);
    const img = document.getElementById("cam-img-" + idx);
    const params = L3D.camProjParams(L3D.state.payload, idx);
    if (!ov || !img || !params) return;
    const p2 = params.p2;
    const ctx = ov.getContext("2d");
    if (!ctx) return; // jsdom 无 canvas 实现：绘制静默跳过（结构断言靠 DOM）
    const W = img.naturalWidth || 1, H = img.naturalHeight || 1;
    const sx = ov.width / W; // 原生像素 → 画布像素
    ctx.setTransform(sx, 0, 0, sx, 0, 0);
    ctx.clearRect(0, 0, W, H);
    (L3D.state.data.annotations || []).forEach((ann, i) => {
      if (ann.cx == null) return;
      const del = L3D.state.deleted.has(i);
      const sel = L3D.state.selected === i;
      const geom = L3D.camOverlayGeom(p2, ann, W, H);
      ctx.strokeStyle = del ? "rgba(229,83,75,.55)"
        : (sel ? "#ffd166" : "#" + L3D.colorFor(ann.label).toString(16).padStart(6, "0"));
      ctx.lineWidth = (del ? 1 : 2) / sx;
      ctx.setLineDash(del ? [6, 4] : []);
      ctx.beginPath();
      for (const s of geom.segs) { // 边已裁剪到图像内（视野外部分不画，避免出画布）
        ctx.moveTo(s[0], s[1]);
        ctx.lineTo(s[2], s[3]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (del) return;
      ctx.font = `${12 / sx}px sans-serif`;
      if (geom.segs.length) { // 可见框：标签放第一条可见边起点
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(ann.label, geom.segs[0][0] + 3 / sx, geom.segs[0][1] - 3 / sx);
      }
      if (geom.indicator) { // 完全在相机视野外：最近图像边缘画三角标记 + 标签（可点击选中）
        const ind = geom.indicator;
        ctx.fillStyle = sel ? "#ffd166"
          : "#" + L3D.colorFor(ann.label).toString(16).padStart(6, "0");
        const s6 = 6 / sx;
        ctx.beginPath();
        if (ind.dir === "l") { ctx.moveTo(0, ind.v); ctx.lineTo(s6, ind.v - s6); ctx.lineTo(s6, ind.v + s6); }
        else if (ind.dir === "r") { ctx.moveTo(W, ind.v); ctx.lineTo(W - s6, ind.v - s6); ctx.lineTo(W - s6, ind.v + s6); }
        else if (ind.dir === "t") { ctx.moveTo(ind.u, 0); ctx.lineTo(ind.u - s6, s6); ctx.lineTo(ind.u + s6, s6); }
        else { ctx.moveTo(ind.u, H); ctx.lineTo(ind.u - s6, H - s6); ctx.lineTo(ind.u + s6, H - s6); }
        ctx.closePath();
        ctx.fill();
        ctx.fillText(ann.label,
          Math.max(2 / sx, Math.min(W - 70 / sx, ind.u + 3 / sx)), ind.v - 3 / sx);
      }
      if (sel) { // 选中框手柄：底角 4 + 底边中点 4（黄色圆，拖动 = 地面 resize）
        ctx.fillStyle = "#ffd166";
        for (const hd of geom.handles) {
          const r = (hd.mode === "corner" ? 4 : 3) / sx;
          ctx.beginPath();
          ctx.arc(hd.u, hd.v, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
  }

  let camDrag = null; // {idx, mode, yPlane, x0, y0, moved}：相机图拖动会话（view=3 编辑）
  let camResizeFits = [];
  function bindCamOverlay() {
    camResizeFits.forEach((f) => window.removeEventListener("resize", f));
    camResizeFits = [];
    document.querySelectorAll(".cam-overlay").forEach((ov) => {
      const idx = Number(ov.id.replace("cam-overlay-", ""));
      const img = document.getElementById("cam-img-" + idx);
      const params = L3D.camProjParams(L3D.state.payload, idx);
      if (!img || !params) return;
      const fit = () => {
        const rect = img.getBoundingClientRect();
        ov.width = rect.width || 1;
        ov.height = rect.height || 1;
        drawCamOverlay(idx);
      };
      img.onload = fit;
      camResizeFits.push(fit);
      window.addEventListener("resize", fit);
      fit();

      // 指示标记命中表：每框 geom（下标 = annotations 下标；删除框不建）
      const indicatorGeoms = () => {
        return (L3D.state.data.annotations || []).map((a, i) => {
          if (L3D.state.deleted.has(i) || a.cx == null) return null;
          return L3D.camOverlayGeom(params.p2, a, img.naturalWidth || 1, img.naturalHeight || 1);
        });
      };

      ov.onmousedown = (e) => {
        const p = camNativePoint(e, img);
        if (!p) return;
        const camCenter = params.cam_center || [0, 0, 0]; // 旧 payload 兜底（零平移近似）
        const W = img.naturalWidth || 1, H = img.naturalHeight || 1;

        // 1) 选中框手柄命中（角点 > 边中点）→ beginEdit corner/edge（地面 resize）
        const sel = L3D.state.selected;
        if (sel != null) {
          const ann = L3D.state.data.annotations[sel];
          if (ann && ann.cx != null) {
            const hd = L3D.camHitHandle(L3D.camOverlayGeom(params.p2, ann, W, H), p.u, p.v);
            if (hd) {
              const yaw = L3D.rotationYToYaw(ann.rotation_y || 0);
              const yPlane = ann.cy + ann.h / 2; // 底面（cam y 向下：cy+h/2 = 地面）
              const g0 = L3D.p2ToGround(params.k_inv, camCenter, yPlane, p.u, p.v);
              if (!g0) return; // 点在地平线附近无地面解
              L3D.beginEdit(sel, 3, hd.mode, {
                t0: -yaw, cx0: ann.cx, cz0: ann.cz, w0: ann.w, l0: ann.l, sx: hd.sx, sy: hd.sy,
              });
              camDrag = { idx, mode: hd.mode, yPlane, x0: e.clientX, y0: e.clientY, moved: false };
              return;
            }
          }
        }

        // 2) 视野外指示标记命中 → 选中该框（几何编辑走 Top 视图手柄/拖进视野后再调）
        const ii = L3D.camHitIndicator(indicatorGeoms(), p.u, p.v);
        if (ii != null) {
          L3D.selectObject(ii);
          return;
        }

        // 3) 框边命中 → 选中 + 地面平移（move）
        const i = camHitTest(p.u, p.v, params.p2);
        if (i == null) return;
        L3D.selectObject(i);
        const ann = L3D.state.data.annotations[i];
        const g0 = L3D.p2ToGround(params.k_inv, camCenter, ann.cy, p.u, p.v);
        if (!g0) return;
        L3D.beginEdit(i, 3, "move", { cx0: ann.cx, cz0: ann.cz, x0: g0[0], z0: g0[1] });
        camDrag = { idx, mode: "move", yPlane: ann.cy, x0: e.clientX, y0: e.clientY, moved: false };
      };
      ov.onmousemove = (e) => {
        if (!camDrag || camDrag.idx !== idx) return;
        const p = camNativePoint(e, img);
        if (!p) return;
        if (!camDrag.moved && Math.hypot(e.clientX - camDrag.x0, e.clientY - camDrag.y0) > 3) {
          camDrag.moved = true; // 3px 阈值区分点击/拖动
        }
        if (!camDrag.moved) return;
        const g = L3D.p2ToGround(params.k_inv, params.cam_center || [0, 0, 0],
          camDrag.yPlane, p.u, p.v);
        if (!g) return;
        L3D.editTo(3, camDrag.mode, { x: g[0], z: g[1] });
      };
      const up = () => {
        if (camDrag && camDrag.idx === idx) { L3D.endEdit(); camDrag = null; }
      };
      ov.onmouseup = up;
      ov.onmouseleave = up; // 拖出画布结束编辑（防 mouseup 落空）
    });
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
    document.querySelectorAll(".cam-overlay").forEach((ov) =>
      drawCamOverlay(Number(ov.id.replace("cam-overlay-", ""))));
  });

  loadFiles();
})();
