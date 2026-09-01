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
    renderDetail();
  }

  function renderDetail() {
    const d = L3D.state.data;
    const anns = d.annotations || [];
    const cam = d.image_path
      ? `<figure><img src="api/review-image?path=${encodeURIComponent(d.image_path)}"><figcaption>相机图</figcaption></figure>` : "";
    const bev = d.bev_path
      ? `<figure><img src="api/review-image?path=${encodeURIComponent(d.bev_path)}"><figcaption>BEV 鸟瞰</figcaption></figure>` : "";
    // nuScenes 6 相机（payloads _nuscenes_payload 的 cameras 键；.imgs 已 flex-wrap 零 CSS 改动）
    const cams = (d.cameras || []).map((c) =>
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

  // 订阅状态机：选中/删除/改标签 → 表格重渲染（保持四视图 canvas 不动）
  L3D.on((state) => {
    if (!state.data || !$("#table-wrap")) return;
    $("#table-wrap").innerHTML = L3D.renderTableHtml();
    bindTable();
  });

  loadFiles();
})();
