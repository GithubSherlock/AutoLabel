// Auto3dLabel 简版复核前端：队列文件 → 相机图 + BEV 图 + 3D 框列表（删除/保存）。
// 无画布拖拽/无点云上传——v0.1 刻意最小化（红线：复用 2D Web 协议，不加新端点）。

let state = { name: null, data: null, deleted: new Set() };

const $ = (sel) => document.querySelector(sel);

async function jget(url) {
  const r = await fetch(url);
  return r.json();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function loadFiles() {
  const res = await jget("/api/review-files");
  $("#dir").textContent = res.dir || "";
  $("#files").innerHTML = (res.files || []).map((f) => `
    <div class="file-item" data-name="${esc(f.name)}">
      <div class="name">${esc(f.image_stem || f.name)}</div>
      <div class="meta">${f.count} 框</div>
    </div>`).join("") || '<div class="empty">无待复核队列</div>';
  $("#reviewed").innerHTML = (res.reviewed || []).map((f) => `
    <div class="file-item" data-name="${esc(f.name)}">
      <div class="name">${esc(f.image_stem || f.name)}</div>
      <div class="meta">${f.count} 框 · 已复核</div>
    </div>`).join("") || '<div class="empty">无</div>';
  document.querySelectorAll("#files .file-item, #reviewed .file-item").forEach((el) => {
    el.onclick = () => loadFile(el.dataset.name);
  });
}

async function loadFile(name) {
  const data = await jget("/api/review-file?name=" + encodeURIComponent(name));
  if (data.error) { $("#detail").innerHTML = `<div class="empty">${esc(data.error)}</div>`; return; }
  state = { name, data, deleted: new Set() };
  document.querySelectorAll(".file-item").forEach((el) => el.classList.toggle("active", el.dataset.name === name));
  render();
}

function render() {
  const d = state.data;
  if (!d) return;
  const anns = d.annotations || [];
  const rows = anns.map((a, i) => {
    const del = state.deleted.has(i);
    const flag = a.review_flag ? ' <span class="flag">⚠ review</span>' : "";
    const dims = `${a.h?.toFixed(2)}×${a.w?.toFixed(2)}×${a.l?.toFixed(2)}`;
    const rot = a.rotation_y == null ? "—" : a.rotation_y.toFixed(2);
    return `<tr style="${del ? "opacity:.35" : ""}">
      <td>${i}</td><td>${esc(a.label)}</td>
      <td class="num">${(a.confidence ?? 0).toFixed(2)}</td>
      <td class="num">${esc(dims)}</td>
      <td class="num">${rot}${flag}</td>
      <td class="num">${a.fit_points ?? "—"}</td>
      <td><button class="del" data-i="${i}">${del ? "恢复" : "删除"}</button></td>
    </tr>`;
  }).join("");

  const cam = d.image_path
    ? `<figure><img src="/api/review-image?path=${encodeURIComponent(d.image_path)}"><figcaption>相机图</figcaption></figure>` : "";
  const bev = d.bev_path
    ? `<figure><img src="/api/review-image?path=${encodeURIComponent(d.bev_path)}"><figcaption>BEV 鸟瞰</figcaption></figure>` : "";

  $("#detail").innerHTML = `
    <h3 style="margin-top:0">${esc(d.image || state.name)} <span style="color:var(--dim);font-weight:400">— ${anns.length} 框待复核</span></h3>
    <div class="imgs">${cam}${bev}</div>
    <table>
      <tr><th>#</th><th>类别</th><th>conf</th><th>h×w×l (m)</th><th>rotation_y</th><th>拟合点数</th><th></th></tr>
      ${rows || '<tr><td colspan="7" class="empty">无框</td></tr>'}
    </table>
    <div class="toolbar">
      <button id="save">保存复核结果</button>
      <span id="status"></span>
    </div>`;
  document.querySelectorAll("button.del").forEach((b) => {
    b.onclick = () => {
      const i = Number(b.dataset.i);
      state.deleted.has(i) ? state.deleted.delete(i) : state.deleted.add(i);
      render();
    };
  });
  $("#save").onclick = save;
}

async function save() {
  const anns = state.data.annotations || [];
  // edited 全量重建：保留框原样直通（Box3D dict），删除框剔除；每框标人工复核
  const edited = anns
    .map((a, i) => ({ ...a, edited_by_human: true }))
    .filter((_, i) => !state.deleted.has(i));
  const res = await fetch("/api/review-save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ queue_file: state.name, edited }),
  });
  const out = await res.json();
  $("#status").textContent = out.ok
    ? `已保存：保留 ${out.kept} 框，删除 ${out.deleted} 框 → ${out.saved_path}`
    : `保存失败: ${out.error || res.status}`;
  if (out.ok) { state = { name: null, data: null, deleted: new Set() }; loadFiles(); }
}

loadFiles();
