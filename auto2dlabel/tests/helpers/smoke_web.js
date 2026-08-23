// AutoLabel Web v0.5 前端核心逻辑冒烟（jsdom，零浏览器）
// 运行：npm i jsdom && node smoke_web.js（jsdom 不随仓库，装到本目录或 NODE_PATH 指向）
// 覆盖：渲染 / 选中键盘 / undo 六类 / resize·rotate 数学 / 列表委托 / 过滤 / 隐藏锁定 /
//       右键菜单 / issues 状态机 / saveReview body / 空态
const { JSDOM } = require('jsdom');
const fs = require('fs');

const ROOT = require('path').join(__dirname, '..', '..', 'web', 'static');
const html = fs.readFileSync(ROOT + '/index.html', 'utf8');
const appjs = fs.readFileSync(ROOT + '/app.js', 'utf8');

const dom = new JSDOM(html, { url: 'http://localhost/', runScripts: 'outside-only' });
const { window } = dom;
const { document } = window;

// ── mocks ──
const ctxProxy = new Proxy({}, {
  get: (t, k) => {
    if (k === 'measureText') return () => ({ width: 42 });
    return () => undefined;
  },
  set: () => true,
});
window.HTMLCanvasElement.prototype.getContext = function () { return ctxProxy; };
window.HTMLCanvasElement.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, width: this.width || 100, height: this.height || 100 };
};
window.Image = class {
  set src(v) { this._src = v; setTimeout(() => { this.width = 100; this.height = 100; this.onload && this.onload(); }, 0); }
  get src() { return this._src; }
};
window.HTMLElement.prototype.scrollIntoView = function () {};
window.HTMLElement.prototype.scrollTo = function () {};
window.alert = () => {};
window.prompt = () => null;
window.confirm = () => true;
// jsdom 无文件选择器：mock files
Object.defineProperty(document.getElementById('imageInput'), 'files', {
  value: [{ name: 'a.png' }], configurable: true,
});

const modelLists = {
  '/api/models': { models: { grounding_dino: [], ultralytics: [{ name: 'yolo12n.pt', type: 'coco_classes' }], pytorch_vision: [], custom: [] } },
  '/api/seg-models': { models: { fastsam: [{ name: 'FastSAM-s.pt', type: 'fastsam' }], sam: [], sam2: [], sam3: [], maskrcnn: [], torchvision: [], custom: [] } },
  '/api/obb-models': { models: [{ name: 'yolo11n-obb.pt', type: 'yolo_obb' }] },
  '/api/pose-models': { models: [{ name: 'yolo11n-pose.pt', type: 'yolo_pose' }] },
  '/api/cls-models': { models: { hf_zero_shot: [], torchvision: [], custom: [] } },
};
let savedBody = null;
window.fetch = async (url, opts = {}) => {
  const u = String(url);
  if (modelLists[u]) return { json: async () => modelLists[u] };
  if (u === '/api/annotate') return { ok: true, json: async () => ({
    image_base64: 'data:image/png;base64,xxx',
    image_path: 'a.png',
    bboxes: [
      { x: 10, y: 10, width: 30, height: 30, label: 'car', confidence: 0.9 },
      { x: 50, y: 50, width: 20, height: 40, label: 'person', confidence: 0.5, angle: 0.3 },
      { x: 80, y: 5, width: 10, height: 10, label: 'car', confidence: 0.2 },
    ],
    labels: [], masks: [],
    summary: { total: 3, classes: ['car', 'person'], avg_conf: 0.53, model: 'yolo12n.pt', with_seg: false, seg_model: null, task_type: 'detection' },
  }) };
  if (u === '/api/review-save') {
    savedBody = JSON.parse(opts.body);
    return { ok: true, json: async () => ({ ok: true, saved_path: '/x/smoke_reviewed.json', kept: 2, deleted: 1 }) };
  }
  if (u.startsWith('/api/review-files')) return { json: async () => ({ files: [], reviewed: [] }) };
  throw new Error('unexpected fetch: ' + u);
};

// 桥接：app.js 的 let/const 词法绑定（vm.runInContext 每次 eval 词法环境独立，
// 必须在同一 eval 调用内用 getter 导出到 window.__S）
const NAMES = ['currentData','deletedIds','reviewFileName','lastReviewedName','selectedIndex','hiddenIds','lockedIds','zOrder','undoStack','redoStack','confFilter','hoverIdx','currentIssues','issueMode','issueDraft','nextIssueId','listSort','drag'];
const FUNCS = ['getColor','escapeHtml','canvasPos','snapBox','restoreBox','markEdited','loadModels','toggleSegSection','runAnnotation','resetSessionState','isDirty','confirmDiscard','switchMode','drawBoxes','drawKeypoints','drawMasks','drawHandles','redraw','passesConfFilter','visibleOnCanvas','visibleInList','listOrder','onConfFilterChange','hitBox','findBox','hitHandle','resizeBoxDuringDrag','rotateBoxDuringDrag','bindCanvasEditing','selectBox','selectNext','centerCanvasOn','deleteBox','isFormTarget','bindKeydown','pushUndo','undo','redo','renderResultList','changeLabel','toggleHidden','toggleLock','bringToFront','sendToBack','bindListEvents','bindContextMenu','showCtxMenu','hideCtxMenu','downloadJSON','downloadVis','loadReviewFiles','loadReviewFile','setSaveState','saveReview','downloadReviewJSON','toggleIssueMode','drawIssueDraft','finishIssueDraft','drawIssues','findIssueAt','toggleIssueStatus','openIssueModal','confirmIssue','cancelIssue','renderIssuesSummary'];
const exportJs = '\n;window.__S = {' +
  NAMES.map(n => `get ${n}(){ return ${n}; }, set ${n}(v){ ${n} = v; }`).join(',') + ',' +
  FUNCS.map(n => `${n}: ${n}`).join(',') + '};';
window.eval(appjs + exportJs);
const S = window.__S;

// ── 断言辅助 ──
let passed = 0, failed = 0;
function check(name, cond) {
  if (cond) { passed++; console.log('  ✓ ' + name); }
  else { failed++; console.log('  ✗ ' + name); }
}
const tick = (ms = 0) => new Promise(r => setTimeout(r, ms));
const kd = (key, opts = {}) => document.dispatchEvent(new window.KeyboardEvent('keydown', { key, bubbles: true, ...opts }));
const canvas = () => document.getElementById('canvas');
const ev = (type, x, y, opts = {}) => canvas().dispatchEvent(
  new window.MouseEvent(type, { clientX: x, clientY: y, bubbles: true, ...opts }));

(async () => {
  await tick(10);  // loadModels

  console.log('── ① 标注渲染与列表 ──');
  await S.runAnnotation();
  check('currentData 3 框', S.currentData && S.currentData.bboxes.length === 3);
  check('zOrder 恒等 [0,1,2]', JSON.stringify(S.zOrder) === '[0,1,2]');
  check('列表渲染 3 项', document.querySelectorAll('.result-item').length === 3);
  check('计数 (3/3)', document.getElementById('resultCount').textContent.includes('3/3'));
  check('列表含标签下拉+图标', !!document.querySelector('.label-sel') && !!document.querySelector('[data-act="lock"]'));

  console.log('── ② 选中与键盘 ──');
  S.selectBox(1);
  check('selectBox 选中 1', S.selectedIndex === 1);
  check('选中行样式', document.getElementById('item-1').classList.contains('selected'));
  kd('Tab');
  check('Tab → 2', S.selectedIndex === 2);
  kd('Tab');
  check('Tab 环绕 → 0', S.selectedIndex === 0);
  kd('Tab', { shiftKey: true });
  check('Shift+Tab 回绕 → 2', S.selectedIndex === 2);
  kd('Escape');
  check('Esc 取消选中', S.selectedIndex === -1);

  console.log('── ③ 删除 / dirty / undo·redo ──');
  S.selectBox(2);
  kd('Delete');
  check('Del 删除框 2', S.deletedIds.has(2));
  check('删除后 dirty', S.isDirty() === true);
  check('删除后选中清空', S.selectedIndex === -1);
  kd('z', { ctrlKey: true });
  check('Ctrl+Z 恢复删除', !S.deletedIds.has(2) && S.isDirty() === false);
  kd('z', { ctrlKey: true, shiftKey: true });
  check('Ctrl+Shift+Z 重做删除', S.deletedIds.has(2));
  kd('z', { ctrlKey: true });
  check('再次撤销干净', !S.deletedIds.has(2));

  console.log('── ④ 拖拽 move + 提交（undo/标记）──');
  S.selectBox(0);
  ev('mousedown', 25, 25);   // 框 0 中心（避开手柄）
  check('mousedown 起 drag.move', S.drag && S.drag.type === 'move' && S.drag.i === 0);
  ev('mousemove', 35, 35);
  check('move 后位置 (20,20)', S.currentData.bboxes[0].x === 20 && S.currentData.bboxes[0].y === 20);
  ev('mouseup', 35, 35);
  check('提交后 undo 栈 +1', S.undoStack.length === 1);
  check('提交后 edited + edited_by_human', S.currentData.edited === true && S.currentData.bboxes[0].edited_by_human === true);
  kd('z', { ctrlKey: true });
  check('撤销 move 回 (10,10)', S.currentData.bboxes[0].x === 10 && S.currentData.bboxes[0].y === 10);

  console.log('── ⑤ resize 手柄拖拽（mousedown 分派 + 旋转系数学）──');
  // θ=0：拖左上角手柄到 (0,0)，对角 (40,40) 固定
  ev('mousedown', 10, 10);
  check('命中左上角手柄 → resize', S.drag && S.drag.type === 'resize' && S.drag.sx === -1 && S.drag.sy === -1);
  ev('mousemove', 0, 0);
  const b0 = S.currentData.bboxes[0];
  check('θ=0 缩放对角固定 (0,0)-(40,40)', b0.x === 0 && b0.y === 0 && b0.width === 40 && b0.height === 40);
  ev('mouseup', 0, 0);
  check('缩放提交 undo +1', S.undoStack.length === 1);
  // 单元级：θ≠0 旋转系缩放数学（对角固定不变量）
  const b = { x: -70.7, y: -35.4, width: 141.4, height: 70.7, angle: Math.PI / 4, keypoints: [[0, 0, 2]] };
  const bsnap = JSON.parse(JSON.stringify(b));
  const d = { type: 'resize', sx: 1, sy: 1, cx0: 0, cy0: 0, w0: 141.4, h0: 70.7, t0: Math.PI / 4, before: bsnap };
  const c45 = Math.cos(Math.PI / 4), s45 = Math.sin(Math.PI / 4);
  const P_fix = { x: -70.7 * c45 + 35.35 * s45, y: -70.7 * s45 - 35.35 * c45 };  // R(θ)·(-w0/2, -h0/2)
  // 拖右下角手柄到 R(θ)·(100, 40)：局部 (100,40) → w1=100+70.7=170.7, h1=40+35.35=75.35
  const px = 100 * c45 - 40 * s45, py = 100 * s45 + 40 * c45;
  S.resizeBoxDuringDrag(b, d, { x: px, y: py });
  const w1 = b.width, h1 = b.height;
  const Cnew = { x: b.x + w1 / 2, y: b.y + h1 / 2 };
  const Pnew = { x: Cnew.x + (-w1 / 2) * c45 - (-h1 / 2) * s45, y: Cnew.y + (-w1 / 2) * s45 + (-h1 / 2) * c45 };
  check('θ=π/4 缩放 w1=170.7 h1=75.35', Math.abs(w1 - 170.7) < 1e-6 && Math.abs(h1 - 75.35) < 1e-6);
  check('θ=π/4 对角点固定', Math.abs(Pnew.x - P_fix.x) < 1e-6 && Math.abs(Pnew.y - P_fix.y) < 1e-6);
  check('θ=π/4 keypoints 仿射', Math.abs(b.keypoints[0][0] - Cnew.x) < 1e-6 && Math.abs(b.keypoints[0][1] - Cnew.y) < 1e-6);

  console.log('── ⑥ rotate 数学（归一化 + keypoints）──');
  const r = { x: 0, y: 0, width: 10, height: 10, angle: 0, keypoints: [[5, 0, 2]] };
  const rsnap = JSON.parse(JSON.stringify(r));
  const rd = { t0: 0, cx0: 0, cy0: 0, before: rsnap };
  S.rotateBoxDuringDrag(r, rd, { x: 100, y: 0 });  // 水平右 → θ=π/2
  check('rotate θ=π/2（边界保持）', Math.abs(r.angle - Math.PI / 2) < 1e-9);
  check('rotate keypoints 绕中心', Math.abs(r.keypoints[0][0]) < 1e-9 && Math.abs(r.keypoints[0][1] - 5) < 1e-9);
  S.rotateBoxDuringDrag(r, { ...rd, t0: r.angle, before: JSON.parse(JSON.stringify(r)) }, { x: 0, y: -100 });
  check('rotate 归一化 0（π→0）', Math.abs(r.angle) < 1e-9);

  console.log('── ⑦ 列表事件委托 ──');
  await S.runAnnotation();  // 重置
  const items = () => document.querySelectorAll('.result-item');
  // 改标签（自定义路径：__custom__ → prompt）
  window.prompt = () => 'truck';
  const sel = document.querySelector('.label-sel[data-i="0"]');
  sel.value = '__custom__';
  sel.dispatchEvent(new window.Event('change', { bubbles: true }));
  check('改标签生效 + 标记', S.currentData.bboxes[0].label === 'truck' && S.currentData.bboxes[0].edited_by_human === true);
  check('改标签 undo +1', S.undoStack.length === 1);
  check('标签下拉含新标签', document.getElementById('results').textContent.includes('truck'));
  // 隐藏图标
  const hideBtn = () => document.querySelector('[data-act="hide"]');
  hideBtn().click();
  check('隐藏框 0', S.hiddenIds.has(0));
  check('隐藏后列表 dimmed', !!document.querySelector('#item-0.dimmed'));
  hideBtn().click();
  check('再点显示', !S.hiddenIds.has(0));
  // 锁定（锁定后删除应被拒）
  document.querySelector('[data-act="lock"]').click();
  check('锁定框 0', S.lockedIds.has(0));
  document.querySelector('[data-act="del"]').click();
  check('锁定框删除被拒', !S.deletedIds.has(0));
  document.querySelector('[data-act="lock"]').click();  // 解锁
  // 删除图标
  document.querySelector('[data-act="del"]').click();
  check('图标删除框 0', S.deletedIds.has(0));
  check('列表剩 2 项', items().length === 2);
  // 排序三态
  document.getElementById('sortBtn').click();
  check('排序 desc', S.listSort === 'desc');
  const firstConf = document.querySelector('.result-item .conf');
  check('desc 后首项 conf=50.0%（剩框1 0.5 / 框2 0.2）', firstConf.textContent.trim() === '50.0%');
  document.getElementById('sortBtn').click();
  check('排序 asc', S.listSort === 'asc');
  document.getElementById('sortBtn').click();
  check('排序回默认 null', S.listSort === null);
  // 行点击选中
  document.getElementById('item-1').click();
  check('行点击选中 1', S.selectedIndex === 1);
  S.undo();  // 撤销图标删除框 0
  check('undo 恢复图标删除', !S.deletedIds.has(0) && items().length === 3);

  console.log('── ⑧ confFilter 过滤 ──');
  const slider = document.getElementById('confFilter');
  slider.value = '0.5';
  S.onConfFilterChange();
  check('confFilter=0.5 画布过滤低置信框', S.confFilter === 0.5 && !S.visibleOnCanvas(2) && S.visibleOnCanvas(0));
  check('列表过滤 (2/3)', document.getElementById('resultCount').textContent.includes('2/3'));
  check('过滤值标签', document.getElementById('confFilterVal').textContent === '0.50');
  slider.value = '0';
  S.onConfFilterChange();
  check('回 0 全显示', S.visibleOnCanvas(2));

  console.log('── ⑨ 右键菜单 + 置顶置底 ──');
  S.showCtxMenu(0, 0, 0);
  const menu = document.getElementById('ctxMenu');
  check('菜单显示 6 项（未锁含删除）', menu.style.display === 'block' && menu.querySelectorAll('.menu-item').length === 6);
  S.toggleLock(0);
  S.showCtxMenu(0, 0, 0);
  check('锁定后菜单 5 项（无删除）', menu.querySelectorAll('.menu-item').length === 5);
  const beforeOrder = [...S.zOrder];
  S.bringToFront(1);
  check('置顶后 zOrder 末位=1', S.zOrder[S.zOrder.length - 1] === 1);
  S.sendToBack(1);
  check('置底后 zOrder 首=1', S.zOrder[0] === 1);
  check('置顶再置底后 zOrder=[1,0,2]', JSON.stringify(S.zOrder) === '[1,0,2]');
  check('bboxes 数组本身不变（索引语义稳定）', S.currentData.bboxes.length === 3 && S.currentData.bboxes[0].x === 10);
  S.undo(); S.undo();  // 撤销置底 + 置顶
  check('undo 恢复 zOrder 原序', JSON.stringify(S.zOrder) === JSON.stringify(beforeOrder));
  S.hideCtxMenu();
  check('菜单关闭', menu.style.display === 'none');
  S.toggleLock(0);  // 解锁还原

  console.log('── ⑩ issues 状态机全链路 ──');
  S.toggleIssueMode();
  check('issueMode on + 按钮激活', S.issueMode === true && document.getElementById('issueBtn').classList.contains('active'));
  ev('mousedown', 10, 10);
  check('mousedown 起 issueDraft', S.issueDraft && S.issueDraft.x0 === 10);
  ev('mousemove', 30, 20);
  check('draft 跟随', S.issueDraft.x1 === 30 && S.issueDraft.y1 === 20);
  ev('mouseup', 30, 20);
  check('松手弹 modal', document.getElementById('issueModal').style.display === 'flex');
  document.getElementById('issueText').value = '此处漏检';
  S.confirmIssue();
  check('issue 创建', S.currentIssues.length === 1 && S.currentIssues[0].text === '此处漏检');
  check('issues 摘要', document.getElementById('issuesSummary').textContent.includes('1 个问题'));
  check('画布命中 issue', S.findIssueAt({ x: 20, y: 15 }) === 0);
  ev('mousedown', 20, 15);
  check('点击 issue 翻转 resolved', S.currentIssues[0].status === 'resolved');
  ev('mousedown', 20, 15);
  check('再点重开 open', S.currentIssues[0].status === 'open');
  kd('Escape');
  check('Esc 退出 issueMode', S.issueMode === false);
  // saveReview body 含 issues
  S.reviewFileName = 'smoke_review.json';
  S.deletedIds.add(2);
  await S.saveReview();
  check('saveReview body 含 issues', savedBody && savedBody.issues.length === 1 && savedBody.issues[0].text === '此处漏检');
  check('saveReview body 含 deleted_indices', savedBody && JSON.stringify(savedBody.deleted_indices) === '[2]');
  check('保存后 dirty 清零', S.isDirty() === false);
  check('保存后 bboxes 收敛 kept', S.currentData.bboxes.length === 2);

  console.log('── ⑪ 空态 / 分类回归 ──');
  S.currentData = { image_url: 'data:,', image_base64: 'data:image/png;base64,', bboxes: [], masks: [], summary: {} };
  S.resetSessionState();
  S.renderResultList();
  check('空态不炸 + 提示', document.getElementById('results').textContent.includes('未检测到目标'));
  check('空态 Tab 不选中', S.selectNext(1) === false && S.selectedIndex === -1);
  kd('Delete');
  check('空态 Del 安全', S.deletedIds.size === 0);

  console.log('\n结果: ' + passed + ' passed, ' + failed + ' failed');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('脚本异常:', e); process.exit(2); });
