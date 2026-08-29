// Auto3dLabel v0.3 P4 前端逻辑层冒烟（node + jsdom，零浏览器，不加载 three.min.js）
// 运行：cd auto3dlabel/tests/helpers && npm i jsdom && node smoke_web3d.js
// 覆盖 logic3d.js 全部导出：camToThree/EDGES/edgesOf/headLine/orthoFocus 纯数学、
// renderFilesHtml/renderTableHtml、状态机（loadFiles/loadFrame/selectObject/changeLabel/
// toggleDelete/buildSaveBody）、esc 转义、**fetch 相对路径红线**（api/ 开头，挂载 /3d/ 兼容）。
const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..', 'web', 'static');
const html = fs.readFileSync(ROOT + '/index.html', 'utf8');
const logic3d = fs.readFileSync(ROOT + '/logic3d.js', 'utf8');

const dom = new JSDOM(html, { url: 'http://localhost:8766/', runScripts: 'outside-only' });
const { window } = dom;
const { document } = window;
window.eval(logic3d);
const L3D = window.L3D;
if (!L3D) { console.error('FAIL: logic3d.js 未挂载 L3D'); process.exit(1); }

// ── fixtures ──
// 相机系框：cx=0 cy=-1 cz=10 l=4 w=1.6 h=1.5，车头 +z（前边 z=12 后边 z=8）。
// 角序 = corners_cam：低 0-3 [右前,右后,左后,左前] 高 4-7 同序
const CORNERS = [
  [0.8, -1.75, 12], [0.8, -1.75, 8], [-0.8, -1.75, 8], [-0.8, -1.75, 12],
  [0.8, -0.25, 12], [0.8, -0.25, 8], [-0.8, -0.25, 8], [-0.8, -0.25, 12],
];
const QUEUE_DATA = {
  image: '000123', image_path: '/tmp/a.png', bev_path: '/tmp/b.png', pcd_path: '/tmp/p.bin',
  calib_path: '/tmp/c.txt',
  annotations: [
    { label: 'Car', confidence: 0.62, cx: 8, cy: 1.4, cz: 18, h: 1.5, w: 1.6, l: 3.9,
      rotation_y: 0.05, fit_points: 31, truncated: 0, occluded: 0, alpha: 0,
      x1: 580, y1: 170, x2: 700, y2: 230, review_flag: false },
    { label: 'Pedestrian', confidence: 0.4, cx: 3, cy: 0.8, cz: 7, h: 1.7, w: 0.6, l: 0.8,
      rotation_y: -0.1, fit_points: 9, truncated: 0, occluded: 0, alpha: 0,
      x1: 0, y1: 0, x2: 0, y2: 0, review_flag: true },
  ],
};
const PAYLOAD = {
  points: [[1, 2, 3], [4, 5, 6]],
  objects: [
    { index: 0, label: 'Car', confidence: 0.62, fit_points: 31, corners: CORNERS },
    { index: 1, label: 'Pedestrian', confidence: 0.4, fit_points: 9, corners: CORNERS },
  ],
};

// ── fetch mock（记录 URL 供相对路径红线断言）──
const replies = {
  'api/review-files': {
    files: [{ name: '000123_review.json', image_stem: '000123', count: 2 }],
    reviewed: [{ name: '000122_reviewed.json', image_stem: '000122', count: 1 }],
    dir: '/tmp/review3d',
  },
  'api/review-file?name=000123_review.json': QUEUE_DATA,
  'api/frame-data?name=000123_review.json': PAYLOAD,
};
const urls = [];
window.fetch = async (url) => {
  urls.push(url);
  const body = replies[url];
  return { json: async () => body || { error: 'not found' } };
};

// ── 断言计数 ──
let n = 0;
const ok = (cond, msg) => {
  n++;
  if (!cond) { console.error(`FAIL ${n}: ${msg}`); process.exitCode = 1; }
};
const close = (a, b, tol, msg) => ok(Math.abs(a - b) < tol, `${msg}（${a} vs ${b}）`);

(async () => {
  // 1. camToThree：(x右,y下,z前) → [x, z, -y]
  const t = L3D.camToThree(1, 2, 3);
  ok(t[0] === 1 && t[1] === 3 && t[2] === -2, 'camToThree 换轴 (x,z,-y)');

  // 2. EDGES/edgesOf：12 边 24 端点
  ok(L3D.EDGES.length === 12, 'EDGES 12 边');
  const es = L3D.edgesOf(CORNERS);
  ok(es.length === 12 && es.every((e) => e.length === 2 && e[0].length === 3), 'edgesOf 映射');
  ok(es[8][0][0] === CORNERS[0][0], '竖边 (0,4) 起点');

  // 3. headLine：车头 +z → 前边中点 (0,-1.75,12) 伸出 1.5×半长 → (0,-1.75,15)
  const hl = L3D.headLine(CORNERS);
  ok(hl !== null, 'headLine 非退化');
  close(hl[0][0], 0, 1e-9, 'headLine 前端点 x');
  close(hl[0][2], 12, 1e-9, 'headLine 前端点 z');
  close(hl[1][2], 15, 1e-9, 'headLine 伸出点 z=15');

  // 4. orthoFocus：换轴中心 (0,10,1) + 包围半径
  const f = L3D.orthoFocus(CORNERS);
  close(f.center[0], 0, 1e-9, 'focus center x');
  close(f.center[1], 10, 1e-9, 'focus center y（=cam z）');
  close(f.center[2], 1, 1e-9, 'focus center z（=-cam y）');
  close(f.radius, Math.hypot(0.8, 2, 0.75), 1e-9, 'focus radius');

  // 5. renderFilesHtml：列表 HTML + 转义
  const h = L3D.renderFilesHtml(await (await window.fetch('api/review-files')).json());
  ok(h.files.includes('000123_review.json'), 'files HTML 含队列名');
  ok(h.reviewed.includes('已复核'), 'reviewed HTML 标记');
  ok(h.dir === '/tmp/review3d', 'dir 透传');
  const evil = L3D.renderFilesHtml({ files: [{ name: '<script>_review.json', count: 1 }], reviewed: [] });
  ok(evil.files.includes('&lt;script&gt;') && !evil.files.includes('<script>_'), 'esc 转义文件名');

  // 6. 状态机 loadFiles：state 订阅触发
  let emitted = 0;
  L3D.on(() => emitted++);
  const res = await L3D.loadFiles();
  ok(res.files.length === 1 && emitted === 1, 'loadFiles 触发 emit');
  ok(urls[urls.length - 1] === 'api/review-files', 'fetch 相对路径（挂载兼容红线）');

  // 7. loadFrame：data + payload + 状态复位
  const data = await L3D.loadFrame('000123_review.json');
  ok(data === QUEUE_DATA && L3D.state.name === '000123_review.json', 'loadFrame data');
  ok(L3D.state.payload === PAYLOAD, 'loadFrame payload');
  ok(L3D.state.selected === null && L3D.state.deleted.size === 0, 'loadFrame 状态复位');
  ok(urls.includes('api/review-file?name=000123_review.json'), 'review-file 相对路径');
  ok(urls.includes('api/frame-data?name=000123_review.json'), 'frame-data 相对路径');

  // 8. 表格 HTML：行数/选中/删除样式
  const tbl = L3D.renderTableHtml();
  ok((tbl.match(/<tr/g) || []).length === 3, '表格 2 行 + 表头');
  ok(tbl.includes('⚠ review'), 'review_flag 标记');
  L3D.selectObject(0);
  ok(L3D.state.selected === 0, 'selectObject');
  ok(L3D.renderTableHtml().includes('class="sel"'), '选中行样式');
  L3D.toggleDelete(1);
  ok(L3D.state.deleted.has(1) && L3D.renderTableHtml().includes('opacity:.35'), '删除行样式');
  L3D.toggleDelete(1);
  ok(!L3D.state.deleted.has(1), 'toggleDelete 恢复');

  // 9. changeLabel：改标签 + 人工修正标记
  L3D.changeLabel(0, 'Truck');
  ok(L3D.state.data.annotations[0].label === 'Truck', 'changeLabel 回写 data');
  ok(L3D.state.data.annotations[0].edited_by_human === true, 'edited_by_human 置位');
  ok(L3D.renderTableHtml().includes('Truck'), '改标签后表格刷新');

  // 10. buildSaveBody：编辑全量 + 删除剔除 + queue_file
  L3D.toggleDelete(1);
  const body = L3D.buildSaveBody();
  ok(body.queue_file === '000123_review.json', 'saveBody queue_file');
  ok(body.edited.length === 1 && body.edited[0].label === 'Truck', 'saveBody 剔除删除框');
  ok(body.edited[0].edited_by_human === true, 'saveBody 全量人工标记');
  L3D.toggleDelete(1);

  // 11. 空态：无 payload 的 frame-data error → payload null 不崩
  replies['api/frame-data?name=000999_review.json'] = { error: 'bad' };
  replies['api/review-file?name=000999_review.json'] = { error: 'bad' };
  const bad = await L3D.loadFrame('000999_review.json');
  ok(bad.error === 'bad' && L3D.state.payload === null, 'frame-data error → payload null');

  console.log(`smoke_web3d: ${n} 断言，${process.exitCode ? 'FAIL' : 'OK'}`);
})();
