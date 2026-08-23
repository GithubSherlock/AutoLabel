// AutoLabel Web 前端逻辑（v0.5 — CVAT 式交互增强）
//
// 分区：A 状态 / B 工具 / C 模型加载与标注 / D 会话生命周期 / E 渲染 /
//       F 可见性谓词 / G 命中与拖拽 / H 选中与键盘 / I 撤销重做 /
//       J 结果列表 / K 右键菜单 / L 保存导出 / M 问题区域 issues / N 初始化

// ══ A. 全局状态 ═══════════════════════════════════════════════
let currentData = null;       // 当前标注数据 {image_url, bboxes, masks, ...}
let deletedIds = new Set();   // 已删除 bbox 下标（不落盘，保存走 deleted_indices）
let reviewFileName = null;    // 当前复核的队列文件名
let lastReviewedName = null;  // 最近保存的修正文件名
let detModels = [];
let segModels = [];
let modelMeta = {};           // name → {type}

// v0.5 交互增强状态（全按 bbox 下标键；resetSessionState 为唯一清场入口）
let selectedIndex = -1;       // 选中 bbox 下标
let hiddenIds = new Set();    // 隐藏（画布不显示，列表 dimmed 可恢复）
let lockedIds = new Set();    // 锁定（不可拖/缩放/删，仅选中）
let zOrder = [];              // 绘制/命中顺序（下标数组；删除不剔除，索引稳定）
let undoStack = [], redoStack = [];
const MAX_UNDO = 32;          // 会话级撤销上限（仿 CVAT annotations-history）
let confFilter = 0;           // 置信度过滤（0 = 全部）
let hoverIdx = -1;            // 列表 hover 对应 bbox 下标
let currentIssues = [];       // [{id, x, y, width, height, text, status}]
let issueMode = false;        // N 键进入的框选问题模式
let issueDraft = null;        // 拖拽中的问题草稿 {x0, y0, x1, y1}
let nextIssueId = 1;
let listSort = null;          // null | 'asc' | 'desc'（仅列表排序，不动 zOrder）
let drag = null;              // 画布拖拽状态 {type:'move'|'resize'|'rotate', i, ...}

// ══ B. 工具 ═══════════════════════════════════════════════════

// 颜色生成（确定性哈希）
function getColor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  const h = Math.abs(hash) % 360;
  const s = 0.7, l = 0.55;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };
  const r = Math.round(f(0) * 255);
  const g = Math.round(f(8) * 255);
  const b = Math.round(f(4) * 255);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

function canvasPos(e) {
  const c = document.getElementById('canvas');
  const rect = c.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (c.width / rect.width),
    y: (e.clientY - rect.top) * (c.height / rect.height),
  };
}

// bbox 快照/恢复（JSON 深拷贝，undo 用）
function snapBox(i) {
  return JSON.parse(JSON.stringify(currentData.bboxes[i]));
}
function restoreBox(i, snap) {
  Object.assign(currentData.bboxes[i], snap);
}

// 人工修正标记：置 session 级 edited + 该框 edited_by_human（四操作提交时调用）
function markEdited(i) {
  if (!currentData) return;
  currentData.edited = true;
  const b = currentData.bboxes[i];
  if (b) b.edited_by_human = true;
}

// ══ C. 模型加载与标注 ═════════════════════════════════════════

async function loadModels() {
  try {
    const [detResp, segResp, obbResp, poseResp, clsResp] = await Promise.all([
      fetch('/api/models'),
      fetch('/api/seg-models'),
      fetch('/api/obb-models'),
      fetch('/api/pose-models'),
      fetch('/api/cls-models'),
    ]);
    const detData = await detResp.json();
    const segData = await segResp.json();
    const obbData = await obbResp.json();
    const poseData = await poseResp.json();
    const clsData = await clsResp.json();

    // 展平检测模型
    detModels = [];
    modelMeta = {};
    const detOrder = ['grounding_dino', 'ultralytics', 'pytorch_vision', 'custom'];
    const detLabels = {
      grounding_dino: '🌐 Grounding DINO（开放词汇）',
      ultralytics: '🚀 Ultralytics YOLO',
      pytorch_vision: '🔬 PyTorch Vision',
      custom: '📦 自定义权重',
    };
    const detSel = document.getElementById('model');
    detSel.innerHTML = '';
    for (const cat of detOrder) {
      const items = detData.models[cat] || [];
      if (!items.length) continue;
      const optgroup = document.createElement('optgroup');
      optgroup.label = detLabels[cat];
      optgroup.dataset.kind = 'det';
      for (const m of items) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        optgroup.appendChild(opt);
        detModels.push(m.name);
        modelMeta[m.name] = m.type;
      }
      detSel.appendChild(optgroup);
    }

    // OBB 旋转框模型（同一 select，任务类型切换时过滤显示）
    if (obbData.models && obbData.models.length) {
      const obbGroup = document.createElement('optgroup');
      obbGroup.label = '📐 YOLO-OBB（旋转框）';
      obbGroup.dataset.kind = 'obb';
      for (const m of obbData.models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        obbGroup.appendChild(opt);
        modelMeta[m.name] = m.type;
      }
      detSel.appendChild(obbGroup);
    }

    // 姿态估计模型（同一 select，任务类型切换时过滤显示）
    if (poseData.models && poseData.models.length) {
      const poseGroup = document.createElement('optgroup');
      poseGroup.label = '🧍 YOLO-pose（COCO 17 点）';
      poseGroup.dataset.kind = 'pose';
      for (const m of poseData.models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        poseGroup.appendChild(opt);
        modelMeta[m.name] = m.type;
      }
      detSel.appendChild(poseGroup);
    }

    // 展平分类模型
    const clsSel = document.getElementById('clsModel');
    clsSel.innerHTML = '';
    const clsOrder = ['hf_zero_shot', 'torchvision', 'custom'];
    const clsLabels = {
      hf_zero_shot: '🌐 CLIP / SigLIP（零样本）',
      torchvision: '🔬 torchvision ImageNet',
      custom: '📦 自定义权重',
    };
    for (const cat of clsOrder) {
      const items = clsData.models[cat] || [];
      if (!items.length) continue;
      const optgroup = document.createElement('optgroup');
      optgroup.label = clsLabels[cat];
      for (const m of items) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        optgroup.appendChild(opt);
      }
      clsSel.appendChild(optgroup);
    }

    // 展平分割模型
    segModels = [];
    const segOrder = ['fastsam', 'sam', 'sam2', 'sam3', 'maskrcnn', 'torchvision', 'custom'];
    const segLabels = {
      fastsam: '⚡ FastSAM',
      sam: '🎯 SAM',
      sam2: '🎯 SAM2',
      sam3: '🎯 SAM3',
      maskrcnn: '🔬 Mask R-CNN',
      torchvision: '🔬 torchvision 语义分割 (VOC)',
      custom: '📦 自定义权重',
    };
    const segSel = document.getElementById('segModel');
    segSel.innerHTML = '';
    for (const cat of segOrder) {
      const items = segData.models[cat] || [];
      if (!items.length) continue;
      const optgroup = document.createElement('optgroup');
      optgroup.label = segLabels[cat];
      for (const m of items) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        optgroup.appendChild(opt);
        segModels.push(m.name);
      }
      segSel.appendChild(optgroup);
    }

    // 如果分割模型列表为空，添加默认项
    if (segSel.children.length === 0) {
      const opt = document.createElement('option');
      opt.value = 'FastSAM-s.pt';
      opt.textContent = 'FastSAM-s.pt';
      segSel.appendChild(opt);
      segModels.push('FastSAM-s.pt');
    }

    console.log(`加载 ${detModels.length} 个检测模型, ${segModels.length} 个分割模型`);
  } catch (e) {
    console.error('模型列表加载失败:', e);
    // 降级：使用硬编码默认值
    const detSel = document.getElementById('model');
    detSel.innerHTML = `
      <option value="yolo26x.pt">yolo26x.pt</option>
      <option value="yolo12n.pt">yolo12n.pt</option>
      <option value="fasterrcnn_resnet50_fpn_v2">fasterrcnn_resnet50_fpn_v2</option>
      <option value="IDEA-Research/grounding-dino-tiny">grounding-dino-tiny</option>
    `;
    detModels = ['yolo26x.pt', 'yolo12n.pt', 'fasterrcnn_resnet50_fpn_v2', 'IDEA-Research/grounding-dino-tiny'];
    modelMeta = {
      'yolo26x.pt': 'coco_classes',
      'yolo12n.pt': 'coco_classes',
      'fasterrcnn_resnet50_fpn_v2': 'coco_classes',
      'IDEA-Research/grounding-dino-tiny': 'open_vocabulary',
    };
  }
}

// 模型选中更新
document.getElementById('model').addEventListener('change', function() {
  const name = this.value;
  const type = modelMeta[name] || 'unknown';
  const tag = document.getElementById('modelTag');
  const info = document.getElementById('modelInfo');
  if (type === 'open_vocabulary') {
    tag.innerHTML = '<span class="info-tag open">开放词汇</span>';
    info.textContent = '✅ 支持任意类别名，无需 COCO 预定义';
  } else if (type === 'coco_classes') {
    tag.innerHTML = '<span class="info-tag coco">COCO 80类</span>';
    info.textContent = '⚠️ 仅检测 COCO 80 类中的目标，按 prompt 过滤';
  } else {
    tag.innerHTML = '<span class="info-tag custom">自定义</span>';
    info.textContent = '📦 本地权重文件';
  }
});

// 任务类型切换
function onTaskTypeChange() {
  const type = document.getElementById('taskType').value;
  const modelSel = document.getElementById('model');
  const modelLabel = modelSel.previousElementSibling;
  const clsSel = document.getElementById('clsModel');
  const clsInfo = document.getElementById('clsModelInfo');
  const detOnly = document.getElementById('detOnly');
  const segSection = document.getElementById('segSection');
  const instr = document.getElementById('instruction');

  // 模型 select 按任务过滤 optgroup（det 组 / obb 组 / pose 组）
  const kindOf = t => (t === 'obb' ? 'obb' : t === 'pose' ? 'pose' : 'det');
  Array.from(modelSel.querySelectorAll('optgroup')).forEach(g => {
    g.style.display = (g.dataset.kind === kindOf(type)) ? '' : 'none';
  });
  if (type === 'obb') modelSel.value = 'yolo11n-obb.pt';
  if (type === 'pose') modelSel.value = 'yolo11n-pose.pt';

  modelLabel.style.display = type === 'classification' ? 'none' : '';
  modelSel.style.display = type === 'classification' ? 'none' : '';
  clsSel.style.display = type === 'classification' ? '' : 'none';
  clsInfo.style.display = type === 'classification' ? '' : 'none';
  detOnly.style.display = type === 'classification' ? 'none' : '';
  segSection.style.display = (type === 'detection' && document.getElementById('withSeg').checked) ? 'block' : 'none';

  // 指令提示
  if (type === 'classification') {
    if (instr.value === 'car, person' || instr.value === '检测 car 和 person') {
      instr.value = 'cat, dog, car, person';
    }
    instr.placeholder = '候选标签，如: cat, dog, car (逗号分隔，零样本任意语言)';
  } else {
    instr.placeholder = '如: car, person, bicycle (逗号分隔)';
  }
}

// 分割开关
function toggleSegSection() {
  const seg = document.getElementById('segSection');
  seg.style.display = document.getElementById('withSeg').checked ? 'block' : 'none';
}

// 运行标注
async function runAnnotation() {
  const file = document.getElementById('imageInput').files[0];
  if (!file) { alert('请先选择图像'); return; }

  const runBtn = document.getElementById('runBtn');
  const errorBox = document.getElementById('errorBox');
  errorBox.innerHTML = '';
  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="spinner"></span>正在标注...';

  document.getElementById('results').innerHTML = '';
  document.getElementById('summary').innerHTML = '';
  document.getElementById('resultCount').textContent = '';

  const taskType = document.getElementById('taskType').value;

  const form = new FormData();
  form.append('image', file);
  form.append('instruction', document.getElementById('instruction').value);
  form.append('conf', document.getElementById('conf').value);
  form.append('iou', document.getElementById('iou').value);
  form.append('task_type', taskType);
  form.append('model', taskType === 'classification'
    ? document.getElementById('clsModel').value
    : document.getElementById('model').value);
  if (taskType === 'detection') {
    form.append('with_seg', document.getElementById('withSeg').checked);
    if (document.getElementById('withSeg').checked) {
      form.append('seg_model', document.getElementById('segModel').value);
    }
  }

  try {
    const resp = await fetch('/api/annotate', { method: 'POST', body: form });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      errorBox.innerHTML = `<div class="error-box">❌ ${data.error || '未知错误'}</div>`;
      runBtn.disabled = false;
      runBtn.innerHTML = '🚀 开始标注';
      return;
    }

    // 释放旧复核 blob URL（防内存泄漏）
    if (currentData && currentData.image_url && currentData.image_url.startsWith('blob:')) {
      URL.revokeObjectURL(currentData.image_url);
    }
    currentData = data;
    resetSessionState();

    // 显示图像
    const img = new Image();
    img.onload = () => {
      const c = document.getElementById('canvas');
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      drawMasks(ctx);
      drawBoxes(ctx);
      drawIssues(ctx);
      drawHandles(ctx);
    };
    img.src = data.image_base64;

    // 摘要（分类任务渲染 labels，检测/OBB 渲染框统计）
    const labels = data.labels || [];
    let summaryHtml;
    if (taskType === 'classification') {
      const labelText = labels.map(l =>
        `<span class="badge" style="background:#16578a">${escapeHtml(l.label)} ${(l.score * 100).toFixed(1)}%</span>`
      ).join(' ') || '无结果';
      summaryHtml = `
        <div class="summary">
          🏷️ 分类结果（${labels.length} 候选）: ${labelText}<br>
          <small>模型: ${data.summary.model}</small>
        </div>`;
    } else {
      const segInfo = data.summary.with_seg ? ` | 🖌️ 分割: ${data.summary.seg_model}` : '';
      summaryHtml = `
        <div class="summary">
          <b>${data.summary.total}</b> 个目标 &nbsp;|&nbsp;
          类别: ${data.summary.classes.join(', ') || '无'}<br>
          <small>均置信度: ${(data.summary.avg_conf * 100).toFixed(1)}% &nbsp;|&nbsp; 模型: ${data.summary.model}${segInfo}</small>
        </div>`;
    }
    document.getElementById('summary').innerHTML = summaryHtml;

    // 结果列表
    renderResultList();
    document.getElementById('dlBtn').style.display = 'block';
    document.getElementById('dlVis').style.display = 'block';
    document.getElementById('maskToggle').style.display = (data.masks && data.masks.length) ? 'block' : 'none';

  } catch (e) {
    errorBox.innerHTML = `<div class="error-box">❌ 请求失败: ${e.message}</div>`;
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '🚀 开始标注';
  }
}

// ══ D. 会话生命周期 ═══════════════════════════════════════════

// 清场唯一入口：五集合 + undo 栈 + issues + 筛选 + 选中（索引稳定靠这里统一）
function resetSessionState() {
  deletedIds.clear();
  hiddenIds.clear();
  lockedIds.clear();
  undoStack = [];
  redoStack = [];
  selectedIndex = -1;
  hoverIdx = -1;
  currentIssues = [];
  nextIssueId = 1;
  issueMode = false;
  issueDraft = null;
  listSort = null;
  confFilter = 0;
  const slider = document.getElementById('confFilter');
  if (slider) slider.value = 0;
  const val = document.getElementById('confFilterVal');
  if (val) val.textContent = '全部';
  zOrder = (currentData && currentData.bboxes) ? currentData.bboxes.map((_, i) => i) : [];
  const btn = document.getElementById('issueBtn');
  if (btn) btn.classList.remove('active');
  const hint = document.getElementById('issueHint');
  if (hint) hint.style.display = 'none';
  const menu = document.getElementById('ctxMenu');
  if (menu) menu.style.display = 'none';
  renderIssuesSummary();
}

// dirty 语义：有编辑或删除即未保存
function isDirty() {
  return !!currentData && (currentData.edited || deletedIds.size > 0);
}
function confirmDiscard() {
  if (!isDirty()) return true;
  return confirm('当前有未保存的修改，确定要离开吗？');
}

// 标注 / 复核队列 模式切换
function switchMode(mode) {
  if (!confirmDiscard()) return;
  const annotateOn = mode === 'annotate';
  document.getElementById('annotateSection').style.display = annotateOn ? 'block' : 'none';
  document.getElementById('reviewSection').style.display = annotateOn ? 'none' : 'block';
  document.getElementById('tabAnnotate').classList.toggle('mode-active', annotateOn);
  document.getElementById('tabReview').classList.toggle('mode-active', !annotateOn);

  // 清空当前数据，避免跨模式串扰
  if (currentData && currentData.image_url && currentData.image_url.startsWith('blob:')) {
    URL.revokeObjectURL(currentData.image_url);
  }
  currentData = null;
  reviewFileName = null;
  resetSessionState();
  document.getElementById('results').innerHTML = '';
  document.getElementById('summary').innerHTML = '';
  document.getElementById('resultCount').textContent = '';
  document.getElementById('errorBox').innerHTML = '';
  document.getElementById('dlBtn').style.display = 'none';
  document.getElementById('dlVis').style.display = 'none';
  document.getElementById('maskToggle').style.display = 'none';
  document.getElementById('reviewActions').style.display = 'none';
  if (!annotateOn) loadReviewFiles();
}

// ══ E. 渲染 ═══════════════════════════════════════════════════

function _strokeBox(ctx, b) {
  if (b.angle && Math.abs(b.angle) > 1e-4) {
    // 旋转框：绕中心旋转绘制（Bbox.angle 弧度，width 轴相对 x 轴）
    ctx.save();
    ctx.translate(b.x + b.width / 2, b.y + b.height / 2);
    ctx.rotate(b.angle);
    ctx.strokeRect(-b.width / 2, -b.height / 2, b.width, b.height);
    ctx.restore();
  } else {
    ctx.strokeRect(b.x, b.y, b.width, b.height);
  }
}

function drawBoxes(ctx) {
  if (!currentData) return;
  for (const i of zOrder) {
    const b = currentData.bboxes[i];
    if (!b || !visibleOnCanvas(i)) continue;
    const color = getColor(b.label);
    // hover 高亮（外层白边，锁定框不高亮）
    if (i === hoverIdx && !lockedIds.has(i)) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 5;
      _strokeBox(ctx, b);
    }
    // 主描边：选中框虚线
    ctx.strokeStyle = i === selectedIndex ? '#53a8b6' : color;
    ctx.lineWidth = i === selectedIndex ? 3 : 2.5;
    if (i === selectedIndex) ctx.setLineDash([5, 4]);
    _strokeBox(ctx, b);
    ctx.setLineDash([]);

    // 标签背景
    const text = `${b.label} ${(b.confidence*100).toFixed(0)}%`;
    ctx.font = 'bold 13px -apple-system, sans-serif';
    const tm = ctx.measureText(text);
    const th = 16;
    const tx = b.x, ty = Math.max(b.y - th - 2, 2);
    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, tm.width + 8, th + 4);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, tx + 4, ty + th - 2);

    // 姿态任务：COCO 17 点骨架（无 keypoints 自动跳过）
    drawKeypoints(ctx, b);
  }
}

// COCO 17 点骨架（与 tools/visualize.py 同一线对集，含冗余髋-膝-踝边）
const COCO_SKELETON = [
  [0,1],[0,2],[1,3],[2,4],[5,7],[7,9],[6,8],[8,10],[5,6],[11,12],
  [11,13],[13,15],[12,14],[14,16],[5,11],[6,12],[11,12],[13,14],[15,16],
];

function drawKeypoints(ctx, b) {
  if (!b.keypoints || !b.keypoints.length) return;
  const color = getColor(b.label);
  const kp = b.keypoints;  // [[x, y, v], ...] v=0 未标注
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  COCO_SKELETON.forEach(([a, c]) => {
    const pa = kp[a], pb = kp[c];
    if (!pa || !pb || pa[2] <= 0 || pb[2] <= 0) return;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  });
  ctx.fillStyle = color;
  kp.forEach(k => {
    if (k[2] <= 0) return;
    ctx.beginPath();
    ctx.arc(k[0], k[1], 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

// 分割 mask 叠加（polygon 半透明填充 + 描边）
function drawMasks(ctx) {
  if (!currentData || !currentData.masks || !currentData.masks.length) return;
  if (!document.getElementById('showMasks').checked) return;
  currentData.masks.forEach(m => {
    // 对应 bbox 被删除/隐藏/过滤时跳过该 mask
    if (m.bbox_index !== undefined && m.bbox_index >= 0 && !visibleOnCanvas(m.bbox_index)) return;
    const color = getColor(m.bbox.label);
    (m.segmentation || []).forEach(poly => {
      if (!poly || poly.length < 6) return;
      const path = new Path2D();
      path.moveTo(poly[0], poly[1]);
      for (let i = 2; i < poly.length; i += 2) path.lineTo(poly[i], poly[i + 1]);
      path.closePath();
      ctx.fillStyle = color + '55';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.fill(path);
      ctx.stroke(path);
    });
  });
}

// 选中框手柄：8 点缩放 + 旋转柄（局部坐标经 R(θ) 转世界坐标）
function drawHandles(ctx) {
  if (!currentData || selectedIndex < 0) return;
  const b = currentData.bboxes[selectedIndex];
  if (!b || lockedIds.has(selectedIndex)) return;
  const t = b.angle || 0;
  const cos = Math.cos(t), sin = Math.sin(t);
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2;
  const halfW = b.width / 2, halfH = b.height / 2;
  const locals = [
    [-halfW, -halfH], [halfW, -halfH], [halfW, halfH], [-halfW, halfH],
    [0, -halfH], [0, halfH], [halfW, 0], [-halfW, 0],
  ];
  ctx.save();
  locals.forEach(([lx, ly], idx) => {
    const wx = cx + lx * cos - ly * sin;
    const wy = cy + lx * sin + ly * cos;
    const size = idx < 4 ? 7 : 5;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(wx - size / 2, wy - size / 2, size, size);
    ctx.strokeStyle = '#53a8b6';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(wx - size / 2, wy - size / 2, size, size);
  });
  // 旋转柄：顶部中点上方 24px
  const twx = cx + halfH * sin;
  const twy = cy - halfH * cos;
  const rwx = cx + (halfH + 24) * sin;
  const rwy = cy - (halfH + 24) * cos;
  ctx.strokeStyle = '#53a8b6';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(twx, twy);
  ctx.lineTo(rwx, rwy);
  ctx.stroke();
  ctx.fillStyle = '#53a8b6';
  ctx.beginPath();
  ctx.arc(rwx, rwy, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(rwx, rwy, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// 重绘（图像 + mask + 框 + issues + 手柄）
function redraw() {
  if (!currentData) return;
  const c = document.getElementById('canvas');
  const ctx = c.getContext('2d');
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    drawMasks(ctx);
    drawBoxes(ctx);
    drawIssues(ctx);
    drawHandles(ctx);
  };
  img.src = currentData.image_url || currentData.image_base64;
}

// ══ F. 可见性谓词（单一事实源） ══════════════════════════════

function passesConfFilter(i) {
  return confFilter <= 0 || (currentData.bboxes[i].confidence || 0) >= confFilter;
}
function visibleOnCanvas(i) {
  return !deletedIds.has(i) && !hiddenIds.has(i) && passesConfFilter(i);
}
function visibleInList(i) {
  return !deletedIds.has(i) && passesConfFilter(i);
}
// 置信度过滤滑条（复核区；标注模式恒 0 = 全部）
function onConfFilterChange() {
  confFilter = Number(document.getElementById('confFilter').value);
  document.getElementById('confFilterVal').textContent = confFilter > 0 ? confFilter.toFixed(2) : '全部';
  redraw();
  renderResultList();
}

// 列表顺序：zOrder 序（可加三态置信度排序，仅影响列表不影响画布）
function listOrder() {
  let order = zOrder.filter(visibleInList);
  if (listSort) {
    order = order.slice().sort((a, b) => {
      const d = (currentData.bboxes[a].confidence || 0) - (currentData.bboxes[b].confidence || 0);
      return listSort === 'desc' ? -d : d;
    });
  }
  return order;
}

// ══ G. 命中与拖拽 ═════════════════════════════════════════════

function hitBox(p, b) {
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2;
  if (b.angle && Math.abs(b.angle) > 1e-4) {
    // 旋转框命中近似：中心距离 ≤ 外接半径
    return Math.hypot(p.x - cx, p.y - cy) <= Math.max(b.width, b.height) / 2;
  }
  return p.x >= b.x && p.x <= b.x + b.width && p.y >= b.y && p.y <= b.y + b.height;
}

function findBox(p) {
  if (!currentData) return -1;
  for (let k = zOrder.length - 1; k >= 0; k--) {
    const i = zOrder[k];
    if (!visibleOnCanvas(i)) continue;
    if (hitBox(p, currentData.bboxes[i])) return i;
  }
  return -1;
}

// 手柄命中（仅当前选中框）：{type:'resize'|'rotate', sx, sy}；sx/sy 为固定方向符号
function hitHandle(p) {
  if (!currentData || selectedIndex < 0) return null;
  const b = currentData.bboxes[selectedIndex];
  if (!b || lockedIds.has(selectedIndex)) return null;
  const t = b.angle || 0;
  const cos = Math.cos(t), sin = Math.sin(t);
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2;
  const handles = [
    [-b.width/2, -b.height/2, -1, -1], [b.width/2, -b.height/2, 1, -1],
    [b.width/2, b.height/2, 1, 1], [-b.width/2, b.height/2, -1, 1],
    [0, -b.height/2, 0, -1], [0, b.height/2, 0, 1],
    [b.width/2, 0, 1, 0], [-b.width/2, 0, -1, 0],
  ];
  for (const [lx, ly, sx, sy] of handles) {
    const wx = cx + lx * cos - ly * sin;
    const wy = cy + lx * sin + ly * cos;
    if (Math.hypot(p.x - wx, p.y - wy) <= 10) return { type: 'resize', sx, sy };
  }
  const rwx = cx + (b.height / 2 + 24) * sin;
  const rwy = cy - (b.height / 2 + 24) * cos;
  if (Math.hypot(p.x - rwx, p.y - rwy) <= 12) return { type: 'rotate', sx: 0, sy: 0 };
  return null;
}

// 缩放（旋转坐标系内：对角点固定 + 手柄吸附指针；keypoints 从快照仿射，无累积误差）
//   手柄局部 (sx·w1/2, sy·h1/2) 吸附指针 pl → 对角局部 P_fix = (-sx·w0/2, -sy·h0/2) 固定
//   ⇒ w1 = sx·(plx - P_fix_lx)，C_new 局部 = (pl + P_fix) / 2（edge 手柄 sx=0 时该轴不变）
const MIN_BOX = 4;
function resizeBoxDuringDrag(b, drag, p) {
  const cos = Math.cos(drag.t0), sin = Math.sin(drag.t0);
  const rx = p.x - drag.cx0, ry = p.y - drag.cy0;
  const plx = rx * cos + ry * sin;   // 指针局部 x（沿宽轴）
  const ply = -rx * sin + ry * cos;  // 指针局部 y（沿高轴）
  const fx = -drag.sx * drag.w0 / 2, fy = -drag.sy * drag.h0 / 2;  // 对角点局部坐标
  const w1 = drag.sx !== 0 ? Math.max(drag.sx * (plx - fx), MIN_BOX) : drag.w0;
  const h1 = drag.sy !== 0 ? Math.max(drag.sy * (ply - fy), MIN_BOX) : drag.h0;
  const clx = drag.sx !== 0 ? (plx + fx) / 2 : 0;  // 新中心局部 x（对角固定轴为 0）
  const cly = drag.sy !== 0 ? (ply + fy) / 2 : 0;
  const cx1 = drag.cx0 + clx * cos - cly * sin;
  const cy1 = drag.cy0 + clx * sin + cly * cos;
  b.width = w1;
  b.height = h1;
  b.x = cx1 - w1 / 2;
  b.y = cy1 - h1 / 2;
  // keypoints 从 mousedown 快照做绝对仿射（避免逐帧累积漂移）
  if (drag.before.keypoints && drag.w0 > 0 && drag.h0 > 0) {
    b.keypoints = drag.before.keypoints.map(k => {
      const klx = (k[0] - drag.cx0) * cos + (k[1] - drag.cy0) * sin;
      const kly = -(k[0] - drag.cx0) * sin + (k[1] - drag.cy0) * cos;
      const nx = cx1 + (klx * w1 / drag.w0) * cos - (kly * h1 / drag.h0) * sin;
      const ny = cy1 + (klx * w1 / drag.w0) * sin + (kly * h1 / drag.h0) * cos;
      return [nx, ny, k[2]];
    });
  }
}

// 旋转：θ = atan2 + π/2，归一化 (-π/2, π/2]；keypoints 从快照绕中心旋转 dθ
function rotateBoxDuringDrag(b, drag, p) {
  let t1 = Math.atan2(p.y - drag.cy0, p.x - drag.cx0) + Math.PI / 2;
  while (t1 > Math.PI / 2) t1 -= Math.PI;
  while (t1 <= -Math.PI / 2) t1 += Math.PI;
  const dθ = t1 - drag.t0;
  b.angle = t1;
  if (drag.before.keypoints) {
    const cos = Math.cos(dθ), sin = Math.sin(dθ);
    b.keypoints = drag.before.keypoints.map(k => {
      const dx = k[0] - drag.cx0, dy = k[1] - drag.cy0;
      return [drag.cx0 + dx * cos - dy * sin, drag.cy0 + dx * sin + dy * cos, k[2]];
    });
  }
}

function bindCanvasEditing() {
  const c = document.getElementById('canvas');
  c.addEventListener('mousedown', e => {
    if (!currentData) return;
    if (e.button === 2) return;  // 右键交给 contextmenu
    const p = canvasPos(e);
    if (issueMode) {
      // issue 模式：点击已有问题区域 → 切换解决状态；否则开始框选新问题
      const hitIssue = findIssueAt(p);
      if (hitIssue >= 0) {
        toggleIssueStatus(hitIssue);
        return;
      }
      issueDraft = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      e.preventDefault();
      return;
    }
    // 普通模式点击已有问题区域：切换解决状态
    const issueIdx = findIssueAt(p);
    if (issueIdx >= 0) {
      toggleIssueStatus(issueIdx);
      return;
    }
    // 手柄优先（当前选中框）
    const handle = hitHandle(p);
    if (handle) {
      const i = selectedIndex;
      const b = currentData.bboxes[i];
      drag = {
        type: handle.type, i, sx: handle.sx, sy: handle.sy,
        cx0: b.x + b.width / 2, cy0: b.y + b.height / 2,
        w0: b.width, h0: b.height, t0: b.angle || 0,
        before: snapBox(i),
      };
      e.preventDefault();
      return;
    }
    const i = findBox(p);
    if (i >= 0) {
      selectBox(i);
      const b = currentData.bboxes[i];
      if (lockedIds.has(i)) return;  // 锁定：仅选中不拖
      drag = { type: 'move', i, dx: b.x - p.x, dy: b.y - p.y, before: snapBox(i) };
    } else {
      selectBox(-1);
    }
  });

  c.addEventListener('mousemove', e => {
    const p = canvasPos(e);
    if (issueMode && issueDraft) {
      issueDraft.x1 = p.x;
      issueDraft.y1 = p.y;
      redraw();
      drawIssueDraft();
      return;
    }
    if (!drag || !currentData) return;
    const b = currentData.bboxes[drag.i];
    if (!b) { drag = null; return; }
    if (drag.type === 'move') {
      const nx = Math.max(0, p.x + drag.dx);
      const ny = Math.max(0, p.y + drag.dy);
      const dx = nx - b.x, dy = ny - b.y;
      b.x = nx;
      b.y = ny;
      // 关键点随框平移（绝对像素坐标）
      if (b.keypoints) {
        b.keypoints = b.keypoints.map(k => [k[0] + dx, k[1] + dy, k[2]]);
      }
    } else if (drag.type === 'resize') {
      resizeBoxDuringDrag(b, drag, p);
    } else if (drag.type === 'rotate') {
      rotateBoxDuringDrag(b, drag, p);
    }
    currentData.edited = true;
    redraw();
  });

  const commitDrag = () => {
    if (issueMode && issueDraft) {
      finishIssueDraft();
      return;
    }
    if (!drag || !currentData) { drag = null; return; }
    // 捕获局部值进 undo 闭包（drag 随即置 null，闭包不能再读 drag 变量）
    const { i, before, type } = drag;
    const b = currentData.bboxes[i];
    if (b) {
      const after = snapBox(i);
      if (JSON.stringify(before) !== JSON.stringify(after)) {
        const name = type === 'move' ? '移动' : type === 'resize' ? '缩放' : '旋转';
        pushUndo(name, () => restoreBox(i, before), () => restoreBox(i, after));
        markEdited(i);
      }
    }
    drag = null;
    redraw();
  };
  c.addEventListener('mouseup', commitDrag);
  c.addEventListener('mouseleave', commitDrag);
}

// ══ H. 选中与键盘 ═════════════════════════════════════════════

function selectBox(i, opts) {
  opts = opts || {};
  selectedIndex = i;
  redraw();
  renderResultList();
  if (opts.center && i >= 0 && currentData) centerCanvasOn(currentData.bboxes[i]);
}

function selectNext(dir) {
  if (!currentData) return false;
  const vis = zOrder.filter(visibleOnCanvas);
  if (!vis.length) return false;
  const pos = vis.indexOf(selectedIndex);
  let cur = pos >= 0 ? pos : (dir > 0 ? -1 : vis.length);
  selectBox(vis[(cur + dir + vis.length) % vis.length], { center: true });
  return true;
}

function centerCanvasOn(b) {
  const c = document.getElementById('canvas');
  const main = document.querySelector('.main');
  if (!main) return;
  const rect = c.getBoundingClientRect();
  const mainRect = main.getBoundingClientRect();
  const scale = rect.width / c.width;
  const bx = (b.x + b.width / 2) * scale + rect.left - mainRect.left + main.scrollLeft;
  const by = (b.y + b.height / 2) * scale + rect.top - mainRect.top + main.scrollTop;
  main.scrollTo({ left: bx - main.clientWidth / 2, top: by - main.clientHeight / 2, behavior: 'smooth' });
}

function deleteBox(i) {
  if (!currentData || lockedIds.has(i) || deletedIds.has(i)) return;
  deletedIds.add(i);
  pushUndo('删除', () => { deletedIds.delete(i); }, () => { deletedIds.add(i); });
  if (selectedIndex === i) selectedIndex = -1;
  redraw();
  renderResultList();
}

function isFormTarget(t) {
  return t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA');
}

function bindKeydown() {
  document.addEventListener('keydown', e => {
    // Ctrl+S：任意位置保存（review 模式有文件时；恒拦截防浏览器存页）
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      if (reviewFileName) saveReview();
      return;
    }
    // Escape：issue 模式 → 右键菜单 → 取消选中
    if (e.key === 'Escape') {
      if (issueMode) { toggleIssueMode(); e.preventDefault(); return; }
      const menu = document.getElementById('ctxMenu');
      if (menu && menu.style.display === 'block') { hideCtxMenu(); e.preventDefault(); return; }
      if (selectedIndex >= 0) { selectBox(-1); e.preventDefault(); }
      return;
    }
    // Ctrl+Z / Ctrl+Shift+Z（表单控件内不拦，交给输入框自身撤销）
    if ((e.ctrlKey || e.metaKey) && !isFormTarget(e.target) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
      return;
    }
    if (isFormTarget(e.target)) return;
    // Tab / Shift+Tab：循环选中下一个可见框并居中
    if (e.key === 'Tab' && currentData) {
      if (selectNext(e.shiftKey ? -1 : 1)) e.preventDefault();
      return;
    }
    // Delete / Backspace：删除选中框
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIndex >= 0) {
      e.preventDefault();
      deleteBox(selectedIndex);
      return;
    }
    // N：切换框选问题模式
    if (e.key.toLowerCase() === 'n' && currentData) {
      toggleIssueMode();
      return;
    }
  });
}

// ══ I. 撤销 / 重做（闭包栈，MAX 32 会话级） ══════════════════

function pushUndo(name, undoFn, redoFn) {
  undoStack.push({ name, undo: undoFn, redo: redoFn });
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack.length = 0;  // 新操作清重做栈
}

function undo() {
  if (!currentData) return;
  const op = undoStack.pop();
  if (!op) return;
  op.undo();
  redoStack.push(op);
  redraw();
  renderResultList();
}

function redo() {
  if (!currentData) return;
  const op = redoStack.pop();
  if (!op) return;
  op.redo();
  undoStack.push(op);
  redraw();
  renderResultList();
}

// ══ J. 结果列表（事件委托；行点击 = 选中） ════════════════════

function renderResultList() {
  if (!currentData) return;
  const order = listOrder();
  document.getElementById('resultCount').textContent = `(${order.length}/${currentData.bboxes.length} 项)`;
  const labels = [...new Set(currentData.bboxes.map(b => b.label))].sort();
  const sortLabel = listSort === 'asc' ? '↕ 置信度↑' : listSort === 'desc' ? '↕ 置信度↓' : '↕ 默认序';
  const header = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <button id="sortBtn" title="列表排序（不影响画布）">${sortLabel}</button>
    <span style="font-size:11px;color:#8899aa">👁隐藏 · 🔒锁定 · 🗑删除</span>
  </div>`;
  let rows = '';
  order.forEach(i => {
    const b = currentData.bboxes[i];
    if (!b) return;
    const color = getColor(b.label);
    const hidden = hiddenIds.has(i);
    const locked = lockedIds.has(i);
    const sel = i === selectedIndex;
    const options = labels.map(l =>
      `<option value="${escapeHtml(l)}"${l === b.label ? ' selected' : ''}>${escapeHtml(l)}</option>`
    ).join('') + '<option value="__custom__">✏️ 自定义…</option>';
    rows += `<div class="result-item${sel ? ' selected' : ''}${hidden ? ' dimmed' : ''}" id="item-${i}" data-i="${i}">
      <div class="row-main">
        <span class="badge" style="background:${color};color:#fff">${escapeHtml(b.label)}</span>
        <select class="label-sel" data-i="${i}" title="改标签">${options}</select>
        <span class="conf">${(b.confidence*100).toFixed(1)}%</span>
        ${b.edited_by_human ? '<span class="edited-tag" title="人工修正">🖊</span>' : ''}
        ${hidden ? '<span class="dim-tag">已隐藏</span>' : ''}
      </div>
      <div class="row-actions">
        <button class="icon-btn" data-i="${i}" data-act="hide" title="${hidden ? '显示' : '隐藏'}">${hidden ? '👁' : '🙈'}</button>
        <button class="icon-btn" data-i="${i}" data-act="lock" title="${locked ? '解锁' : '锁定'}">${locked ? '🔓' : '🔒'}</button>
        <button class="icon-btn" data-i="${i}" data-act="del" title="删除">🗑</button>
      </div>
    </div>`;
  });
  document.getElementById('results').innerHTML =
    header + (rows || '<p style="color:#8899aa">未检测到目标</p>');
  const selEl = document.getElementById('item-' + selectedIndex);
  if (selEl) selEl.scrollIntoView({ block: 'nearest' });
}

function changeLabel(i, value) {
  if (!currentData) return;
  const b = currentData.bboxes[i];
  if (!b) return;
  if (value === '__custom__') {
    const label = prompt('输入新标签:', b.label);
    if (label === null || !label.trim()) { renderResultList(); return; }
    value = label.trim();
  }
  if (value === b.label) { renderResultList(); return; }
  const before = snapBox(i);
  b.label = value;
  const after = snapBox(i);
  pushUndo('改标签', () => restoreBox(i, before), () => restoreBox(i, after));
  markEdited(i);
  renderResultList();
  redraw();
}

function toggleHidden(i) {
  if (hiddenIds.has(i)) hiddenIds.delete(i); else hiddenIds.add(i);
  renderResultList();
  redraw();
}

function toggleLock(i) {
  if (lockedIds.has(i)) lockedIds.delete(i); else lockedIds.add(i);
  if (lockedIds.has(i) && selectedIndex === i) redraw();  // 手柄消失
  renderResultList();
  redraw();
}

// 置顶/置底：只动 zOrder 数组，bboxes 数组顺序永不改变（索引稳定）
function bringToFront(i) {
  if (zOrder.indexOf(i) < 0 || zOrder[zOrder.length - 1] === i) return;
  const before = [...zOrder];
  zOrder.splice(zOrder.indexOf(i), 1);
  zOrder.push(i);
  const after = [...zOrder];
  pushUndo('置顶', () => { zOrder = [...before]; }, () => { zOrder = [...after]; });
  redraw();
  renderResultList();
}

function sendToBack(i) {
  if (zOrder.indexOf(i) <= 0) return;
  const before = [...zOrder];
  zOrder.splice(zOrder.indexOf(i), 1);
  zOrder.unshift(i);
  const after = [...zOrder];
  pushUndo('置底', () => { zOrder = [...before]; }, () => { zOrder = [...after]; });
  redraw();
  renderResultList();
}

function bindListEvents() {
  const box = document.getElementById('results');
  box.addEventListener('click', e => {
    if (e.target.id === 'sortBtn') {
      listSort = listSort === null ? 'desc' : listSort === 'desc' ? 'asc' : null;
      renderResultList();
      return;
    }
    const act = e.target.closest('[data-act]');
    if (act) {
      e.stopPropagation();
      const i = Number(act.dataset.i);
      if (act.dataset.act === 'hide') toggleHidden(i);
      else if (act.dataset.act === 'lock') toggleLock(i);
      else if (act.dataset.act === 'del') deleteBox(i);
      return;
    }
    if (e.target.closest('select')) return;  // 改标签由 change 事件处理
    const item = e.target.closest('[data-i]');
    if (item) selectBox(Number(item.dataset.i), { center: true });
  });
  box.addEventListener('change', e => {
    const sel = e.target.closest('.label-sel');
    if (!sel) return;
    changeLabel(Number(sel.dataset.i), sel.value);
  });
  // hover 高亮联动画布
  box.addEventListener('mouseover', e => {
    const item = e.target.closest('[data-i]');
    const ni = item ? Number(item.dataset.i) : -1;
    if (ni !== hoverIdx) {
      hoverIdx = ni;
      redraw();
    }
  });
  box.addEventListener('mouseleave', () => {
    if (hoverIdx !== -1) {
      hoverIdx = -1;
      redraw();
    }
  });
}

// ══ K. 右键菜单 ═══════════════════════════════════════════════

function bindContextMenu() {
  const c = document.getElementById('canvas');
  const menu = document.getElementById('ctxMenu');
  c.addEventListener('contextmenu', e => {
    e.preventDefault();
    if (!currentData) return;
    const i = findBox(canvasPos(e));
    if (i < 0) { hideCtxMenu(); return; }
    selectBox(i);
    showCtxMenu(e.clientX, e.clientY, i);
  });
  menu.addEventListener('click', e => {
    const item = e.target.closest('[data-act]');
    if (!item) return;
    const i = Number(menu.dataset.i);
    const b = currentData && currentData.bboxes[i];
    if (!b) { hideCtxMenu(); return; }
    const act = item.dataset.act;
    if (act === 'label') {
      const label = prompt('修改标签:', b.label);
      if (label !== null && label.trim()) changeLabel(i, label.trim());
    } else if (act === 'lock') toggleLock(i);
    else if (act === 'hide') toggleHidden(i);
    else if (act === 'front') bringToFront(i);
    else if (act === 'back') sendToBack(i);
    else if (act === 'del') deleteBox(i);
    hideCtxMenu();
  });
  // 点击菜单外/滚动关闭
  document.addEventListener('click', e => {
    if (!menu.contains(e.target)) hideCtxMenu();
  });
  document.addEventListener('scroll', hideCtxMenu, true);
}

function showCtxMenu(x, y, i) {
  const menu = document.getElementById('ctxMenu');
  const locked = lockedIds.has(i);
  const hidden = hiddenIds.has(i);
  menu.innerHTML = `
    <div class="menu-item" data-act="label">✏️ 改标签</div>
    <div class="menu-item" data-act="lock">${locked ? '🔓 解锁' : '🔒 锁定'}</div>
    <div class="menu-item" data-act="hide">${hidden ? '👁 显示' : '🙈 隐藏'}</div>
    <div class="menu-item" data-act="front">⬆ 置顶</div>
    <div class="menu-item" data-act="back">⬇ 置底</div>
    ${locked ? '' : '<div class="menu-item danger" data-act="del">🗑 删除</div>'}
  `;
  menu.dataset.i = i;
  menu.style.display = 'block';
  menu.style.left = Math.min(x, window.innerWidth - 170) + 'px';
  menu.style.top = Math.min(y, window.innerHeight - 200) + 'px';
}

function hideCtxMenu() {
  const menu = document.getElementById('ctxMenu');
  if (menu) menu.style.display = 'none';
}

// ══ L. 保存 / 导出 ════════════════════════════════════════════

async function downloadJSON() {
  if (!currentData) return;
  const kept = currentData.bboxes.filter((b, i) => !deletedIds.has(i));
  const keptMasks = (currentData.masks || []).filter(m =>
    m.bbox_index === undefined || m.bbox_index < 0 || !deletedIds.has(m.bbox_index));

  const payload = {
    image_path: currentData.image_path || 'upload.jpg',
    image_width: document.getElementById('canvas').width,
    image_height: document.getElementById('canvas').height,
    bboxes: kept,
    masks: keptMasks,
    model: currentData.summary?.model,
  };

  try {
    const resp = await fetch('/api/export-coco', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert('导出失败: ' + (data.error || resp.status));
      return;
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'annotations.json';
    a.click();
  } catch (e) {
    alert('导出失败: ' + e.message);
  }
}

function downloadVis() {
  if (!currentData?.image_base64) return;
  const a = document.createElement('a');
  a.href = currentData.image_base64;
  a.download = 'autolabel_visualization.png';
  a.click();
}

// ── 复核队列 ──

async function loadReviewFiles() {
  const box = document.getElementById('reviewFiles');
  box.innerHTML = '<p style="color:#8899aa">加载中...</p>';
  try {
    const resp = await fetch('/api/review-files');
    const data = await resp.json();
    const esc = s => String(s).replace(/'/g, "\\'");
    let html = '';
    const files = data.files || [];
    const reviewed = data.reviewed || [];
    if (files.length) {
      html += '<h3 style="margin-top:8px">待复核</h3>';
      files.forEach(f => {
        html += `<div class="result-item" onclick="loadReviewFile('${esc(f.name)}')">
          <span>📄 ${f.image_stem || f.name}</span>
          <span>${f.count} 框</span>
        </div>`;
      });
    }
    if (reviewed.length) {
      html += '<h3 style="margin-top:8px">已复核（点击重开）</h3>';
      reviewed.forEach(f => {
        html += `<div class="result-item" onclick="loadReviewFile('${esc(f.name)}')">
          <span>✅ ${f.image_stem || f.name}</span>
          <span>${f.count} 框${f.issues ? ' · ⚠' + f.issues : ''}</span>
        </div>`;
      });
    }
    box.innerHTML = html || '<p style="color:#8899aa">队列为空（outputs/ 下无复核文件）</p>';
  } catch (e) {
    box.innerHTML = `<div class="error-box">❌ 加载失败: ${e.message}</div>`;
  }
}

async function loadReviewFile(name) {
  if (!confirmDiscard()) return;
  // 释放旧 blob URL（防内存泄漏）
  if (currentData && currentData.image_url && currentData.image_url.startsWith('blob:')) {
    URL.revokeObjectURL(currentData.image_url);
  }
  reviewFileName = name;
  document.getElementById('reviewActions').style.display = 'none';
  const errorBox = document.getElementById('errorBox');
  errorBox.innerHTML = '';
  const isReviewed = name.endsWith('_reviewed.json');

  try {
    const fileResp = await fetch('/api/review-file?name=' + encodeURIComponent(name));
    const qdata = await fileResp.json();
    if (!fileResp.ok || qdata.error) {
      errorBox.innerHTML = `<div class="error-box">❌ ${qdata.error || '队列文件不可读'}</div>`;
      reviewFileName = null;
      return;
    }

    const imgPath = qdata.image_path || '';
    if (!imgPath) {
      errorBox.innerHTML = '<div class="error-box">❌ 该文件缺少 image_path 字段（旧版本文件），无法显示原图</div>';
      reviewFileName = null;
      return;
    }

    const imgResp = await fetch('/api/review-image?path=' + encodeURIComponent(imgPath));
    if (!imgResp.ok) {
      errorBox.innerHTML = '<div class="error-box">❌ 原图不可读: ' + imgPath + '</div>';
      reviewFileName = null;
      return;
    }
    const imgUrl = URL.createObjectURL(await imgResp.blob());

    // 标注数据：已复核文件（COCO dict）转换回前端框结构
    let bboxes, issues;
    if (isReviewed) {
      const catMap = {};
      (qdata.categories || []).forEach(cat => { catMap[cat.id] = cat.name; });
      bboxes = (qdata.annotations || []).map(a => ({
        x: a.bbox ? a.bbox[0] : 0,
        y: a.bbox ? a.bbox[1] : 0,
        width: a.bbox ? a.bbox[2] : 0,
        height: a.bbox ? a.bbox[3] : 0,
        label: catMap[a.category_id] || 'unknown',
        confidence: a.score !== undefined ? a.score : 1.0,
        edited_by_human: !!a.edited_by_human,
      }));
      issues = qdata.issues || [];
    } else {
      bboxes = qdata.annotations || [];
      issues = qdata.issues || [];
    }

    currentData = {
      image_url: imgUrl,
      image_path: imgPath,
      bboxes,
      masks: [],
      summary: { total: bboxes.length, model: '' },
    };
    resetSessionState();
    currentIssues = issues.map(iss => ({ ...iss }));
    nextIssueId = currentIssues.reduce((m, iss) => Math.max(m, iss.id || 0), 0) + 1;
    renderIssuesSummary();

    const img = new Image();
    img.onload = () => {
      const c = document.getElementById('canvas');
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      drawBoxes(ctx);
      drawIssues(ctx);
    };
    img.src = imgUrl;

    renderResultList();
    if (!currentData.bboxes.length) {
      document.getElementById('results').innerHTML = '<p style="color:#8899aa">无待复核标注</p>';
    }
    document.getElementById('summary').innerHTML = `
      <div class="summary">
        <b>${currentData.bboxes.length}</b> 个目标（${isReviewed ? '已复核重开' : '待复核'}）<br>
        <small>文件: ${name} · 拖拽修正 · 列表点击选中 · Tab 切换 · Del 删除 · Ctrl+Z 撤销 · Ctrl+S 保存 · N 框选问题</small>
      </div>`;
    document.getElementById('reviewActions').style.display = 'block';
  } catch (e) {
    errorBox.innerHTML = `<div class="error-box">❌ 加载失败: ${e.message}</div>`;
  }
}

function setSaveState(saving) {
  const btn = document.getElementById('saveReviewBtn');
  if (!btn) return;
  btn.disabled = saving;
  btn.innerHTML = saving ? '<span class="spinner"></span>保存中...' : '💾 保存修正';
}

async function saveReview() {
  if (!reviewFileName || !currentData) return;
  setSaveState(true);
  try {
    const kept = currentData.bboxes.filter((b, i) => !deletedIds.has(i));
    const body = { queue_file: reviewFileName, deleted_indices: [...deletedIds] };
    if (currentData.edited) body.edited = kept;  // 修正框全量重建
    if (currentIssues.length) body.issues = currentIssues;
    const resp = await fetch('/api/review-save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert('保存失败: ' + (data.error || resp.status));
      return;
    }
    lastReviewedName = data.saved_path.split('/').pop();
    alert(`已保存: ${data.saved_path}\n保留 ${data.kept} 框，删除 ${data.deleted} 框` +
      (currentIssues.length ? `\n⚠ ${currentIssues.length} 个问题已随标注落盘` : ''));
    reviewFileName = null;
    // 保存成功：画面收敛为已落盘状态，清 dirty 与撤销栈（修「edited 永不回落」误报坑）
    currentData.bboxes = kept;
    currentData.edited = false;
    resetSessionState();
    redraw();
    renderResultList();
    document.getElementById('summary').innerHTML = `
      <div class="summary">
        ✅ 已保存: ${data.saved_path.split('/').pop()}<br>
        <small>保留 ${data.kept} 框，删除 ${data.deleted} 框 · 可在左侧重新打开继续</small>
      </div>`;
    loadReviewFiles();
  } catch (e) {
    alert('保存失败: ' + e.message);
  } finally {
    setSaveState(false);
  }
}

async function downloadReviewJSON() {
  if (!lastReviewedName) { alert('请先保存修正'); return; }
  try {
    const resp = await fetch('/api/review-file?name=' + encodeURIComponent(lastReviewedName));
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert('下载失败: ' + (data.error || resp.status));
      return;
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = lastReviewedName;
    a.click();
  } catch (e) {
    alert('下载失败: ' + e.message);
  }
}

// ══ M. 问题区域 issues（画框 + 输入 + 半透明浮层 + 解决/重开） ══

function toggleIssueMode() {
  if (!currentData) return;
  issueMode = !issueMode;
  issueDraft = null;
  const btn = document.getElementById('issueBtn');
  const hint = document.getElementById('issueHint');
  const c = document.getElementById('canvas');
  if (btn) btn.classList.toggle('active', issueMode);
  if (hint) hint.style.display = issueMode ? 'block' : 'none';
  if (c) c.style.cursor = issueMode ? 'crosshair' : 'pointer';
  redraw();
}

function drawIssueDraft() {
  if (!issueDraft) return;
  const c = document.getElementById('canvas');
  const ctx = c.getContext('2d');
  const d = issueDraft;
  const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
  const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
  ctx.fillStyle = 'rgba(255, 212, 59, 0.2)';
  ctx.strokeStyle = '#ffd43b';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}

function finishIssueDraft() {
  if (!issueDraft) return;
  const d = issueDraft;
  issueDraft = null;
  const c = document.getElementById('canvas');
  const x = Math.max(0, Math.min(d.x0, d.x1));
  const y = Math.max(0, Math.min(d.y0, d.y1));
  const w = Math.min(Math.abs(d.x1 - d.x0), c.width - x);
  const h = Math.min(Math.abs(d.y1 - d.y0), c.height - y);
  redraw();
  if (w < 3 || h < 3) return;  // 点击而非拖框：忽略
  openIssueModal({ x, y, w, h });
}

function drawIssues(ctx) {
  if (!currentData || !currentIssues.length) return;
  currentIssues.forEach(iss => {
    const resolved = iss.status === 'resolved';
    ctx.fillStyle = resolved ? 'rgba(82, 183, 136, 0.2)' : 'rgba(255, 212, 59, 0.2)';
    ctx.strokeStyle = resolved ? '#52b788' : '#ffd43b';
    ctx.lineWidth = 1.5;
    ctx.fillRect(iss.x, iss.y, iss.width, iss.height);
    if (resolved) {
      ctx.strokeRect(iss.x, iss.y, iss.width, iss.height);
    } else {
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(iss.x, iss.y, iss.width, iss.height);
      ctx.setLineDash([]);
    }
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = resolved ? '#52b788' : '#ffd43b';
    ctx.fillText(`#${iss.id} ${iss.text.slice(0, 12)}`, iss.x + 2, Math.max(iss.y - 4, 10));
  });
}

function findIssueAt(p) {
  for (let k = currentIssues.length - 1; k >= 0; k--) {
    const iss = currentIssues[k];
    if (p.x >= iss.x && p.x <= iss.x + iss.width && p.y >= iss.y && p.y <= iss.y + iss.height) return k;
  }
  return -1;
}

function toggleIssueStatus(k) {
  const iss = currentIssues[k];
  if (!iss) return;
  iss.status = iss.status === 'resolved' ? 'open' : 'resolved';
  renderIssuesSummary();
  redraw();
}

function openIssueModal(rect) {
  const modal = document.getElementById('issueModal');
  if (!modal) return;
  modal.style.display = 'flex';
  modal.dataset.rect = JSON.stringify(rect);
  const ta = document.getElementById('issueText');
  ta.value = '';
  ta.focus();
}

function confirmIssue() {
  const modal = document.getElementById('issueModal');
  const rect = JSON.parse(modal.dataset.rect || 'null');
  const text = document.getElementById('issueText').value.trim();
  modal.style.display = 'none';
  if (rect && text) {
    currentIssues.push({
      id: nextIssueId++,
      x: rect.x, y: rect.y, width: rect.w, height: rect.h,
      text: text, status: 'open',
    });
    renderIssuesSummary();
  }
  redraw();
}

function cancelIssue() {
  document.getElementById('issueModal').style.display = 'none';
  redraw();
}

function renderIssuesSummary() {
  const el = document.getElementById('issuesSummary');
  if (!el) return;
  if (!currentIssues.length) { el.textContent = ''; return; }
  const resolved = currentIssues.filter(i => i.status === 'resolved').length;
  el.textContent = `⚠ ${currentIssues.length} 个问题（${resolved} 已解决）· 点击画布中问题区域可切换解决状态`;
}

// ══ N. 初始化 ═════════════════════════════════════════════════

loadModels();
onTaskTypeChange();
bindCanvasEditing();
bindListEvents();
bindContextMenu();
bindKeydown();

// 未保存修改离开提醒
window.addEventListener('beforeunload', e => {
  if (isDirty()) {
    e.preventDefault();
    e.returnValue = '';
  }
});
