// Auto3dLabel v0.3 P4 前端逻辑层冒烟（node + jsdom，零浏览器，不加载 three.min.js）
// 运行：cd auto3dlabel/tests/helpers && npm i jsdom && node smoke_web3d.js
// 覆盖 logic3d.js 全部导出：camToThree/EDGES/edgesOf/headLine/orthoFocus 纯数学、
// renderFilesHtml/renderTableHtml、状态机（loadFiles/loadFrame/selectObject/changeLabel/
// toggleDelete/buildSaveBody）、esc 转义、**fetch 相对路径红线**（api/ 开头，挂载 /3d/ 兼容）；
// 末尾第 12 节加载 viewer3d.js/app.js 真实代码（THREE mock）做渲染装配回归。
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
// KITTI 相机图投影 fixtures（fx=700 主点 (600,180) 零平移 → 手算往返锚点）
const P2 = [[700, 0, 600, 0], [0, 700, 180, 0], [0, 0, 1, 0]];
const K_INV = [[1 / 700, 0, -600 / 700], [0, 1 / 700, -180 / 700], [0, 0, 1]];
const PAYLOAD = {
  points: [[1, 2, 3], [4, 5, 6]],
  objects: [
    { index: 0, label: 'Car', confidence: 0.62, fit_points: 31, corners: CORNERS },
    { index: 1, label: 'Pedestrian', confidence: 0.4, fit_points: 9, corners: CORNERS },
  ],
  p2: P2, k_inv: K_INV, cam_center: [0, 0, 0], img_size: [1242, 375],
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

  // ── v0.4 P1 编辑层断言（纯函数 + 状态机 + undo 栈）──

  // 12. boxToCorners 锚点：yaw=0（rotation_y=-π/2）与 CORNERS fixture 逐点一致
  const BOX_YAW0 = { cx: 0, cy: -1, cz: 10, h: 1.5, w: 1.6, l: 4, rotation_y: -Math.PI / 2 };
  const yaw0 = L3D.boxToCorners(BOX_YAW0);
  let cOk = true;
  for (let i = 0; i < 8; i++) {
    for (let d = 0; d < 3; d++) if (Math.abs(yaw0[i][d] - CORNERS[i][d]) > 1e-9) cOk = false;
  }
  ok(cOk, 'boxToCorners(yaw=0) 与 CORNERS fixture 逐点一致');

  // 13. boxToCorners 不变量（yaw=0.05 非轴对齐）：中心 (cx,cy,cz) + 边长集合 {h,w,l}
  const cs13 = L3D.boxToCorners(L3D.state.data.annotations[0]);
  const ctr = cs13.reduce((s, c) => [s[0] + c[0], s[1] + c[1], s[2] + c[2]], [0, 0, 0]);
  close(ctr[0] / 8, 8, 1e-9, 'yaw=0.05 角点中心 cx');
  close(ctr[1] / 8, 1.4, 1e-9, 'yaw=0.05 角点中心 cy');
  close(ctr[2] / 8, 18, 1e-9, 'yaw=0.05 角点中心 cz');
  const lens = [...new Set(L3D.EDGES.map(([a, b]) => Math.round(1e2 * Math.hypot(
    cs13[a][0] - cs13[b][0], cs13[a][1] - cs13[b][1], cs13[a][2] - cs13[b][2])) / 1e2))].sort();
  ok(lens.length === 3 && lens[0] === 1.5 && lens[1] === 1.6 && lens[2] === 3.9,
    `yaw=0.05 边长集合 {h,w,l}={1.5,1.6,3.9}（实际 ${lens}）`);
  close(cs13[0][1], 1.4 - 0.75, 1e-9, '低角 y = cy-h/2');
  close(cs13[4][1], 1.4 + 0.75, 1e-9, '高角 y = cy+h/2');
  // yaw=π/2+0.05 → 车头 (sin,cos)=(cos 0.05, -sin 0.05)：z 分量为负 → 车头边 z 小于车尾边
  ok(cs13[0][2] < cs13[1][2] && cs13[3][2] < cs13[2][2], 'yaw=0.05 车头边 z < 车尾边（车头 +x 略偏 -z）');

  // 14. rotationYToYaw / yawToRotationY：往返 + wrap（照 tools/geometry.py 唯一转换点）
  close(L3D.yawToRotationY(L3D.rotationYToYaw(0.05)), 0.05, 1e-9, 'ry→yaw→ry 往返');
  close(L3D.rotationYToYaw(0.05), Math.PI / 2 + 0.05, 1e-9, 'ry=0.05 → yaw=π/2+0.05');
  close(L3D.yawToRotationY(Math.PI / 2 + 0.05), 0.05, 1e-9, 'yaw=π/2+0.05 → ry=0.05');
  const yW = L3D.rotationYToYaw(3.1);
  ok(yW > -Math.PI && yW <= Math.PI, 'ry=3.1 wrap 到 (-π, π]');
  close(L3D.yawToRotationY(yW), 3.1, 1e-9, 'wrap 后往返恢复 ry');
  close(L3D.rotationYToYaw(-3.1), -3.1 + Math.PI / 2, 1e-9, 'ry=-3.1 负向换算');

  // 15. Top corner resize（2D 母版：对角固定 + 手柄吸附；快照式两轮不漂移）
  const b15 = { cx: 0, cz: 10, w: 1.6, l: 4 };
  const drag15 = { t0: 0, cx0: 0, cz0: 10, w0: 1.6, l0: 4, sx: 1, sy: 1 };
  L3D.resizeTopEdit(b15, drag15, { x: 1.6, z: 14 });
  close(b15.w, 2.4, 1e-9, 'corner w 变 2.4');
  close(b15.l, 6, 1e-9, 'corner l 变 6');
  close(b15.cx, 0.4, 1e-9, 'corner cx 联动');
  close(b15.cz, 11, 1e-9, 'corner cz 联动');
  close(b15.cx - b15.w / 2, -0.8, 1e-9, 'corner 对角（左后）固定 x');
  close(b15.cz - b15.l / 2, 8, 1e-9, 'corner 对角（左后）固定 z');
  L3D.resizeTopEdit(b15, drag15, { x: 2.4, z: 16 });
  close(b15.cx - b15.w / 2, -0.8, 1e-9, 'corner 两轮对角固定 x');
  close(b15.cz - b15.l / 2, 8, 1e-9, 'corner 两轮对角固定 z');

  // 16. Top edge resize（车头边 sy=1）：单轴 l 变 w 不动 + 车尾边固定
  const b16 = { cx: 0, cz: 10, w: 1.6, l: 4 };
  L3D.resizeTopEdit(b16, { t0: 0, cx0: 0, cz0: 10, w0: 1.6, l0: 4, sx: 0, sy: 1 }, { x: 0, z: 14 });
  close(b16.w, 1.6, 1e-9, 'edge 单轴 w 不变');
  close(b16.l, 6, 1e-9, 'edge l 变 6');
  close(b16.cx, 0, 1e-9, 'edge cx 不变');
  close(b16.cz, 11, 1e-9, 'edge cz 联动');
  close(b16.cz - b16.l / 2, 8, 1e-9, 'edge 车尾边固定');

  // 17. Top rotate：yaw 增量式（3D 车头有向，非 2D 无向绝对角）+ 中心不动 + wrap
  const b17 = { cx: 0, cz: 10, yaw: 0 };
  L3D.rotateYawEdit(b17, { yaw0: 0, t0: Math.atan2(2, 0.8), cx0: 0, cz0: 10 }, { x: 1, z: 12 });
  close(b17.yaw, Math.atan2(2, 1) - Math.atan2(2, 0.8), 1e-9, 'yaw 增量 = 指针方位角差');
  close(b17.cx, 0, 1e-9, 'rotate 中心 x 不动');
  close(b17.cz, 10, 1e-9, 'rotate 中心 z 不动');
  L3D.rotateYawEdit(b17, { yaw0: 3.1, t0: 0, cx0: 0, cz0: 0 }, { x: 0, z: 1 });
  ok(b17.yaw > -Math.PI && b17.yaw <= Math.PI, 'rotate wrap 到 (-π, π]');

  // 18. Side 顶/底手柄：h + cy 联动、对面固定（cam y 向下语义：屏幕上方 = y 减小）
  const b18 = { cy: 1.4, h: 1.5 };
  L3D.resizeSideEdit(b18, { mode: 'top', cy0: 1.4, h0: 1.5 }, { z: 18, y: 0.9 });
  close(b18.h, 2.0, 1e-9, 'top 手柄上拉 h=2.0');
  close(b18.cy, 1.65, 1e-9, 'top 手柄 cy 联动');
  close(b18.cy - b18.h / 2, 0.65, 1e-9, 'top 顶面固定（=原 cy0-h0/2）');
  L3D.resizeSideEdit(b18, { mode: 'bottom', cy0: 1.4, h0: 1.5 }, { z: 18, y: 1.9 });
  close(b18.h, 2.0, 1e-9, 'bottom 手柄 h=2.0');
  close(b18.cy, 1.65, 1e-9, 'bottom 手柄 cy 联动');
  close(b18.cy + b18.h / 2, 2.65, 1e-9, 'bottom 底面固定（=原 cy0+h0/2）');

  // 19. MIN_BOX3D 钳位（Top w/l + Side h 压穿对面）
  const b19 = { cx: 0, cz: 10, w: 1.6, l: 4 };
  L3D.resizeTopEdit(b19, drag15, { x: -5, z: 3 });
  close(b19.w, L3D.MIN_BOX3D, 1e-9, 'corner 压穿对角 w 钳 MIN_BOX3D');
  close(b19.l, L3D.MIN_BOX3D, 1e-9, 'corner 压穿对角 l 钳 MIN_BOX3D');
  const b19s = { cy: 1.4, h: 1.5 };
  L3D.resizeSideEdit(b19s, { mode: 'top', cy0: 1.4, h0: 1.5 }, { z: 18, y: 5 });
  close(b19s.h, L3D.MIN_BOX3D, 1e-9, 'top 手柄压穿 h 钳 MIN_BOX3D');

  // 20. 编辑状态机完整生命周期：beginEdit → editTo → endEdit → undo → redo
  const pre20 = JSON.parse(JSON.stringify(L3D.state.data.annotations[0]));
  L3D.beginEdit(0, 1, 'edge', { t0: 0, cx0: 8, cz0: 18, w0: 1.6, l0: 3.9, sx: 1, sy: 0 });
  ok(L3D.state.editing && L3D.state.editing.i === 0 && L3D.state.editing.view === 1
    && L3D.state.editing.mode === 'edge', 'beginEdit 进入编辑态');
  close(L3D.state.editing.cur.yaw, Math.PI / 2 + 0.05, 1e-9, 'beginEdit cur.yaw 快照');
  L3D.editTo(1, 'edge', { x: 9, z: 18 });
  const a20 = L3D.state.data.annotations[0];
  close(a20.w, 1.8, 1e-9, 'editTo 回写 w');
  close(a20.cx, 8.1, 1e-9, 'editTo 回写 cx');
  close(a20.l, 3.9, 1e-9, 'editTo l 不变');
  close(a20.rotation_y, 0.05, 1e-9, 'editTo rotation_y 往返');
  ok(L3D.state.editing !== null, '编辑中 editing 保持');
  L3D.endEdit();
  ok(L3D.state.editing === null, 'endEdit 清编辑态');
  ok(a20.edited_by_human === true, 'endEdit 置人工标记');
  L3D.undo();
  const a20u = L3D.state.data.annotations[0];
  close(a20u.w, 1.6, 1e-9, 'undo 恢复 w');
  close(a20u.cx, 8, 1e-9, 'undo 恢复 cx');
  ok(JSON.stringify(a20u) === JSON.stringify(pre20), 'undo 恢复原 dict 全字段');
  L3D.redo();
  close(L3D.state.data.annotations[0].w, 1.8, 1e-9, 'redo 重放编辑');
  L3D.undo(); // 清理回原态（组 22 复用 ann0）

  // 21. 零变化编辑（点击手柄零拖动）：无回写、不入 undo 栈
  const pre21 = JSON.stringify(L3D.state.data.annotations[1]);
  L3D.beginEdit(1, 2, 'top', { mode: 'top', cy0: 0.8, h0: 1.7 });
  L3D.editTo(2, 'top', { z: 7, y: 0.8 }); // dy=0 零变化
  L3D.endEdit();
  ok(L3D.state.editing === null, '零变化 endEdit 清态');
  ok(JSON.stringify(L3D.state.data.annotations[1]) === pre21, '零变化编辑无任何回写');
  L3D.undo();
  ok(JSON.stringify(L3D.state.data.annotations[1]) === pre21, '零变化编辑不入 undo 栈');

  // 22. undo 后新编辑清 redoStack（2D 母版语义）；此刻 redoStack 含组 20 的 redo op
  L3D.beginEdit(0, 1, 'edge', { t0: 0, cx0: 8, cz0: 18, w0: 1.6, l0: 3.9, sx: 1, sy: 0 });
  L3D.editTo(1, 'edge', { x: 10, z: 18 }); // plx=2 → w=2.8
  L3D.endEdit();
  L3D.redo(); // 新编辑已清 redoStack → 无 op 可重放
  close(L3D.state.data.annotations[0].w, 2.8, 1e-9, 'undo 后新编辑清 redo（redo 无效果）');
  L3D.undo();
  close(L3D.state.data.annotations[0].w, 1.6, 1e-9, '清理：undo 回原态');

  // 11. 空态：无 payload 的 frame-data error → payload null 不崩
  replies['api/frame-data?name=000999_review.json'] = { error: 'bad' };
  replies['api/review-file?name=000999_review.json'] = { error: 'bad' };
  const bad = await L3D.loadFrame('000999_review.json');
  ok(bad.error === 'bad' && L3D.state.payload === null, 'frame-data error → payload null');

  // 12. 渲染装配回归：加载 viewer3d.js + app.js 真实代码（THREE mock）。
  // 修复项三连：a) renderDetail 必须从 payload 读 cameras（队列文件 cameras 仅
  // {name,token,filename}，曾 path=undefined 全 404）；b) rebuildObjects null payload
  // 不崩（frame-data 失败曾 TypeError 中断打开流程）；c) hline 块作用域（曾
  // ReferenceError 致 emit 链崩溃、详情区永不渲染 = 无法打开）。
  const V3 = function (x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; };
  V3.prototype = {
    copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; },
    set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; },
    add(v) { return new V3(this.x + v.x, this.y + v.y, this.z + v.z); },
    multiplyScalar(k) { return new V3(this.x * k, this.y * k, this.z * k); },
    dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; },
    project() { return this; },
  };
  function Obj3() { this.position = new V3(); }
  Obj3.prototype.add = function () {};
  Obj3.prototype.remove = function () {};
  class GeoObj extends Obj3 { constructor(geo, mat) { super(); this.geometry = geo; this.material = mat; } }
  class Cam extends Obj3 { constructor() { super(); this.aspect = 1; this.zoom = 1; this.left = -20; this.right = 20; this.top = 20; this.bottom = -20; }
    lookAt() {} updateProjectionMatrix() {} updateMatrixWorld() {} }
  window.THREE = {
    Vector3: V3, Group: Obj3, Scene: Obj3, GridHelper: Obj3, Points: GeoObj, LineSegments: GeoObj, Line: GeoObj,
    Mesh: GeoObj, PerspectiveCamera: Cam, OrthographicCamera: Cam,
    WebGLRenderer: class { constructor() {} setClearColor() {} setSize() {} setViewport() {} setScissor() {}
      setScissorTest() {} render() {} },
    BufferGeometry: class { constructor() { this.attributes = {}; } setAttribute(n, a) { this[n] = a; this.attributes[n] = a; } computeBoundingSphere() {} },
    BufferAttribute: class {},
    PointsMaterial: class { constructor(o) { Object.assign(this, o); } },
    LineBasicMaterial: class { constructor() { this.color = { setHex() {} }; } },
    MeshBasicMaterial: class { constructor(o) { Object.assign(this, o); } },
    SphereGeometry: class {},
    Raycaster: class { constructor() { this.ray = {}; } },
    Plane: class {},
    Sphere: class { constructor(c, r) { this.center = c; this.radius = r; } },
  };
  window.requestAnimationFrame = () => 0;

  const NUS_QUEUE = {
    dataset: 'nuscenes', image: 'tok123', pcd_path: '/tmp/p.bin', ego_translation: [0, 0, 0],
    cameras: [
      { name: 'CAM_FRONT', token: 't1', filename: 'samples/CAM_FRONT/x.jpg' },
      { name: 'CAM_BACK', token: 't2', filename: 'samples/CAM_BACK/x.jpg' },
    ],
    annotations: [{ label: 'Car', confidence: 0.9, cx: 1, cy: 1.4, cz: 18, h: 1.5, w: 1.6, l: 3.9, rotation_y: 0, fit_points: 10 }],
  };
  const NUS_PAYLOAD = {
    points: [[1, 2, 3]],
    objects: [{ index: 0, label: 'Car', confidence: 0.9, fit_points: 10, corners: CORNERS }],
    dataset: 'nuscenes',
    // 后端 _nuscenes_cam_proj 交付契约：有标定相机带 p2/k_inv/cam_center；
    // 查表失败相机仅 {name,image_path}（前端纯图降级）
    cameras: [
      { name: 'CAM_FRONT', image_path: '/root/datasets/nuscenes_mini/samples/CAM_FRONT/x.jpg',
        p2: P2, k_inv: K_INV, cam_center: [0, 0, 0] },
      { name: 'CAM_BACK', image_path: '/root/datasets/nuscenes_mini/samples/CAM_BACK/x.jpg',
        p2: P2, k_inv: K_INV, cam_center: [0, -1.58, -3] }, // 每相机独立相机中心
      { name: 'CAM_FRONT_LEFT', image_path: '/root/datasets/nuscenes_mini/samples/CAM_FRONT_LEFT/x.jpg' },
    ],
  };
  replies['api/review-files'].files.push({ name: 'nus123_review.json', image_stem: 'nus123', count: 1 });
  replies['api/review-file?name=nus123_review.json'] = NUS_QUEUE;
  replies['api/frame-data?name=nus123_review.json'] = NUS_PAYLOAD;

  let evalErr = null;
  try {
    window.eval(fs.readFileSync(ROOT + '/viewer3d.js', 'utf8'));
    window.eval(fs.readFileSync(ROOT + '/app.js', 'utf8'));
  } catch (e) { evalErr = e; }
  ok(evalErr === null, `viewer3d/app.js eval 无异常（${evalErr && evalErr.message}）`);
  await new Promise((r) => setTimeout(r, 10)); // 等 app.js loadFiles 渲染列表
  const nusItem = [...document.querySelectorAll('#files .file-item')]
    .find((el) => el.dataset.name === 'nus123_review.json');
  ok(!!nusItem, 'nuScenes 队列条目已渲染');
  let clickErr = null;
  if (nusItem) { try { await nusItem.onclick(); } catch (e) { clickErr = e; } }
  ok(clickErr === null, `点击打开 nuScenes 无异常（${clickErr && clickErr.message}）`);
  const wrap = document.querySelector('#view3d-wrap');
  ok(wrap && wrap.style.display === 'block', '打开成功后 3D 四视图容器显示（曾恒 display:none）');
  const imgs = [...document.querySelectorAll('#detail .imgs img')];
  ok(imgs.length === 3, `相机图渲染 3 张（${imgs.length}）`);
  ok(imgs.length > 0 && !imgs.some((im) => /undefined/.test(im.src)),
    '相机图 src 无 undefined（读 payload.cameras.image_path）');
  ok(imgs.length > 0 && imgs[0].src.startsWith('http://localhost:8766/api/review-image?path=%2Froot%2F'),
    '相机图 src 为 review-image 绝对路径');
  ok(L3D.state.payload && L3D.state.payload.cameras.length === 3, 'payload.cameras 3 相机');

  // frame-data 失败（payload null）→ rebuildObjects(null) 经 viewer3d 监听不抛异常
  let badErr = null;
  try {
    const bad2 = await L3D.loadFrame('000999_review.json');
    ok(bad2.error === 'bad' && L3D.state.payload === null, 'payload null 下 viewer3d 监听不崩');
  } catch (e) { badErr = e; }
  ok(badErr === null, `payload null 不抛异常（${badErr && badErr.message}）`);

  // 13. 相机图叠加（v1.0）：投影纯函数 + 叠加 canvas 挂载 + view=3 拖动编辑
  const pp = L3D.projectP2(P2, [3.5, 1.75, 7]);
  close(pp[0], 3.5 * 700 / 7 + 600, 1e-9, 'projectP2 u（fx·x/z + cu）');
  close(pp[1], 1.75 * 700 / 7 + 180, 1e-9, 'projectP2 v（fy·y/z + cv）');
  ok(L3D.projectP2(P2, [0, 0, -1]) === null, '相机后方点投影 null');
  const g13 = L3D.p2ToGround(K_INV, [0, 0, 0], 1.75, pp[0], pp[1]);
  close(g13[0], 3.5, 1e-9, 'p2ToGround 投影→反投影往返 x');
  close(g13[1], 7, 1e-9, 'p2ToGround 投影→反投影往返 z');
  ok(L3D.p2ToGround(K_INV, [0, 0, 0], 1.75, 600, 180) === null, '地平线像素无地面解 null');
  const c2 = L3D.boxCorners2d(P2, CORNERS);
  ok(c2.length === 8 && c2.every((p) => p !== null), 'boxCorners2d 8 角全有效');

  const kittiItem = [...document.querySelectorAll('#files .file-item')]
    .find((el) => el.dataset.name === '000123_review.json');
  let kittiClickErr = null;
  if (kittiItem) { try { await kittiItem.onclick(); } catch (e) { kittiClickErr = e; } }
  ok(kittiClickErr === null, `打开 KITTI 帧无异常（${kittiClickErr && kittiClickErr.message}）`);
  ok(!!document.querySelector('#cam-overlay-0'), 'KITTI 帧相机图挂载投影叠加 canvas');
  const kittiOv = document.querySelector('#cam-overlay-0');
  ok(kittiOv && kittiOv.width > 0 && kittiOv.height > 0, '叠加 canvas 已适配尺寸');
  ok(!document.querySelector('#cam-overlay-1'), 'KITTI 帧仅 1 相机叠加层');

  // camProjParams：KITTI 顶层参数 / nuScenes 每相机参数 / 无标定相机 null（纯图降级）
  ok(L3D.camProjParams(PAYLOAD, 0) && L3D.camProjParams(PAYLOAD, 0).p2 === P2,
    'camProjParams KITTI idx=0 走 payload 顶层');
  ok(L3D.camProjParams(PAYLOAD, 1) === null, 'camProjParams KITTI idx>0 null');
  ok(L3D.camProjParams(NUS_PAYLOAD, 0) && L3D.camProjParams(NUS_PAYLOAD, 0).cam_center[1] === 0,
    'camProjParams nus idx=0 走 cameras[0]');
  ok(L3D.camProjParams(NUS_PAYLOAD, 1) && L3D.camProjParams(NUS_PAYLOAD, 1).cam_center[1] === -1.58,
    'camProjParams nus idx=1 走 cameras[1]（每相机独立参数）');
  ok(L3D.camProjParams(NUS_PAYLOAD, 2) === null, 'camProjParams nus 无标定相机 null（纯图）');
  ok(L3D.camProjParams(null, 0) === null, 'camProjParams null payload null');

  const nusItem2 = [...document.querySelectorAll('#files .file-item')]
    .find((el) => el.dataset.name === 'nus123_review.json');
  if (nusItem2) { await nusItem2.onclick(); }
  ok(document.querySelectorAll('.cam-overlay').length === 2,
    'nuScenes 有标定相机 2 张 → 2 个叠加层（曾无标定纯图，本次修复后加投影）');
  ok(!document.querySelector('#cam-overlay-2'), 'nuScenes 无标定相机（CAM_FRONT_LEFT）纯图无叠加层');

  // nuScenes 相机图拖动用每相机参数：spy p2ToGround 的 cam_center 实参
  // （CAM_BACK cc=[0,-1.58,-3] ≠ CAM_FRONT cc=[0,0,0]；u,v 选投影框边线上点）
  const backImg = document.getElementById('cam-img-1');
  const backOv = document.getElementById('cam-overlay-1');
  backImg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1242, height: 375 });
  Object.defineProperty(backImg, 'naturalWidth', { value: 1242 });
  Object.defineProperty(backImg, 'naturalHeight', { value: 375 });
  const origP2G = L3D.p2ToGround;
  const spyCCs = [];
  L3D.p2ToGround = (kInv, cc, cy, u, v) => { spyCCs.push([cc, u, v]); return origP2G(kInv, cc, cy, u, v); };
  L3D.state.selected = null; // 清 KITTI 帧残留选中（手柄分支不干扰边命中）
  backOv.onmousedown({ clientX: 715, clientY: 234 }); // NUS 框投影边线点（yaw=π/2 车头边）
  ok(L3D.state.selected === 0, 'CAM_BACK mousedown 命中投影框边 → 选中');
  backOv.onmousemove({ clientX: 726, clientY: 234 });
  backOv.onmouseup();
  L3D.p2ToGround = origP2G; // spy 复位
  ok(spyCCs.length >= 2, `拖动链路经 p2ToGround（${spyCCs.length} 次）`);
  ok(spyCCs.length >= 1 && spyCCs[0][0][1] === -1.58 && spyCCs[0][0][2] === -3,
    `CAM_BACK 拖动用 cameras[1].cam_center（${JSON.stringify(spyCCs[0] && spyCCs[0][0])}）`);
  const nusAnn = L3D.state.data.annotations[0];
  ok(nusAnn.edited_by_human === true, 'nuScenes 相机图拖动置人工标记');
  // 数值锚点：cc=[0,-1.58,-3] 下 t=(1.4+1.58)/d1，d1=(234-180)/700 → 与 cc=[0,0,0] 显著不同
  close(nusAnn.cx, 1.6069, 1e-2, 'CAM_BACK 拖动 cx 数值（cam_center 反投影锚点）');

  // view=3 'move' 拖动编辑：走 beginEdit/editTo/endEdit + undo（相机图拖动语义）
  await L3D.loadFrame('000123_review.json');
  L3D.selectObject(0);
  const preMove = JSON.stringify(L3D.state.data.annotations[0]);
  L3D.beginEdit(0, 3, 'move', { cx0: 8, cz0: 18, x0: 0, z0: 0 });
  L3D.editTo(3, 'move', { x: 1.5, z: -0.5 });
  close(L3D.state.data.annotations[0].cx, 9.5, 1e-9, '相机图拖动平移 cx+1.5');
  close(L3D.state.data.annotations[0].cz, 17.5, 1e-9, '相机图拖动平移 cz-0.5');
  L3D.endEdit();
  ok(L3D.state.data.annotations[0].edited_by_human === true, '相机图拖动置人工标记');
  L3D.undo();
  ok(JSON.stringify(L3D.state.data.annotations[0]) === preMove, '相机图拖动可撤销');

  // 14. 相机图叠加几何（v1.0 P2）：边缘裁剪 + 视野外指示 + 角点/边手柄拖动
  // clipSegImage（Liang-Barsky，图像 1242×375）
  const sIn = L3D.clipSegImage([100, 100], [200, 200], 1242, 375);
  ok(sIn && sIn[0] === 100 && sIn[2] === 200, 'clip 全内段原样');
  ok(L3D.clipSegImage([-100, 100], [-50, 200], 1242, 375) === null, 'clip 全出界（左）null');
  const sL = L3D.clipSegImage([-100, 100], [300, 100], 1242, 375);
  ok(sL && sL[0] === 0 && sL[1] === 100 && sL[2] === 300 && sL[3] === 100, 'clip 跨左边界裁剪 u=0');
  const sT = L3D.clipSegImage([100, -50], [100, 100], 1242, 375);
  ok(sT && sT[1] === 0 && sT[3] === 100, 'clip 跨上边界裁剪 v=0');
  const sDiag = L3D.clipSegImage([-100, -100], [100, 100], 1242, 375);
  ok(sDiag && sDiag[0] === 0 && sDiag[1] === 0 && sDiag[2] === 100 && sDiag[3] === 100,
    'clip 斜线穿角裁剪到 (0,0)-(100,100)');

  // camOverlayGeom 可见框（ann0：cx=8 cz=18 → u≈911 v≈234 全在图像内）
  const ann14 = { cx: 8, cy: 1.4, cz: 18, h: 1.5, w: 1.6, l: 3.9, rotation_y: 0.05 };
  const gVis = L3D.camOverlayGeom(P2, ann14, 1242, 375);
  ok(gVis.segs.length === 12, `可见框 12 边全在图像内（实际 ${gVis.segs.length}）`);
  ok(gVis.indicator === null, '可见框无指示标记');
  ok(gVis.handles.length === 8, `可见框手柄 8 个（4 角+4 边，实际 ${gVis.handles.length}）`);
  ok(gVis.handles.filter((hd) => hd.mode === 'corner').length === 4, '手柄含 4 角点');
  ok(gVis.handles.filter((hd) => hd.mode === 'edge').length === 4, '手柄含 4 边中点');
  const signs = gVis.handles.filter((hd) => hd.mode === 'corner').map((hd) => `${hd.sx},${hd.sy}`).sort();
  ok(signs.join('|') === '-1,-1|-1,1|1,-1|1,1', `corner 手柄 sx/sy 四象限（${signs.join('|')}）`);
  ok(gVis.handles.every((hd) => hd.u >= 0 && hd.u <= 1242 && hd.v >= 0 && hd.v <= 375),
    '手柄均在图像内（视野外手柄不画）');
  const segsIn = gVis.segs.every((s) => s[0] >= 0 && s[0] <= 1242 && s[1] >= 0 && s[1] <= 375
    && s[2] >= 0 && s[2] <= 1242 && s[3] >= 0 && s[3] <= 375);
  ok(segsIn, 'segs 全部裁剪到图像内');

  // camOverlayGeom 视野外框（cx=-6 cz=4 cy=0 yaw=0：u 全负、v 在图像内）→ indicator 左缘
  const annOut = { cx: -6, cy: 0, cz: 4, h: 1, w: 1, l: 2, rotation_y: -Math.PI / 2 };
  const gOut = L3D.camOverlayGeom(P2, annOut, 1242, 375);
  ok(gOut.segs.length === 0, '视野外框 segs 空（无可见边）');
  ok(gOut.indicator !== null && gOut.indicator.dir === 'l' && gOut.indicator.u === 0,
    `视野外框指示标记在左缘（${gOut.indicator && gOut.indicator.dir}/${gOut.indicator && gOut.indicator.u}）`);
  ok(gOut.indicator.v > 0 && gOut.indicator.v < 375, '指示标记 v 在图像内');
  ok(gOut.handles.length === 0, '视野外框无手柄（反投影无良定义）');

  // camOverlayGeom 部分出界框（跨左边界）→ 裁剪后仍有 segs、无指示
  const annPart = { cx: -5, cy: 0, cz: 10, h: 1, w: 2, l: 6.2, rotation_y: -Math.PI / 2 };
  const gPart = L3D.camOverlayGeom(P2, annPart, 1242, 375);
  ok(gPart.segs.length > 0, `部分出界框裁剪后仍有可见边（${gPart.segs.length}）`);
  ok(gPart.indicator === null, '部分出界框无指示标记');
  ok(gPart.segs.every((s) => s[0] >= 0 && s[2] >= 0), '裁剪段 u 非负');

  // camHitHandle：角点半径命中 + 角优先于边 + 空白处 null
  const cH = gVis.handles.find((hd) => hd.mode === 'corner');
  const hitC = L3D.camHitHandle(gVis, cH.u + 3, cH.v - 3);
  ok(hitC && hitC.mode === 'corner' && hitC.sx === cH.sx && hitC.sy === cH.sy, '手柄命中角点（±10px）');
  ok(L3D.camHitHandle(gVis, 50, 50) === null, '远离手柄处命中 null');
  ok(L3D.camHitHandle({ handles: [] }, cH.u, cH.v) === null, '无手柄 geom 命中 null');

  // camHitIndicator：标记 12px 内命中框下标
  const iH = L3D.camHitIndicator([gVis, gOut], 3, gOut.indicator.v);
  ok(iH === 1, `指示标记命中视野外框下标 1（实际 ${iH}）`);
  ok(L3D.camHitIndicator([gVis], 600, 180) === null, '无标记处命中 null');

  // view=3 corner 拖动 = 地面 resize（对角固定 + 两轮不漂移，公式同 Top 视图）
  const pre14 = JSON.stringify(L3D.state.data.annotations[0]);
  const yaw14 = L3D.rotationYToYaw(0.05);
  const opp = (b) => { // 左后底角（对角固定锚）：d=车头 p=右向
    const dx = Math.sin(yaw14), dz = Math.cos(yaw14), px = Math.cos(yaw14), pz = -Math.sin(yaw14);
    return [b.cx - dx * b.l / 2 - px * b.w / 2, b.cz - dz * b.l / 2 - pz * b.w / 2];
  };
  const oppPre = opp(L3D.state.data.annotations[0]);
  L3D.beginEdit(0, 3, 'corner', { t0: -yaw14, cx0: 8, cz0: 18, w0: 1.6, l0: 3.9, sx: 1, sy: 1 });
  ok(L3D.state.editing && L3D.state.editing.view === 3 && L3D.state.editing.mode === 'corner',
    'beginEdit view=3 corner 进入编辑态');
  L3D.editTo(3, 'corner', { x: 9, z: 18 });
  const oppMid = opp(L3D.state.data.annotations[0]);
  close(oppMid[0], oppPre[0], 1e-9, 'corner 拖动对角（左后）固定 x');
  close(oppMid[1], oppPre[1], 1e-9, 'corner 拖动对角（左后）固定 z');
  close(L3D.state.data.annotations[0].h, 1.5, 1e-9, 'corner 拖动 h 不变');
  L3D.editTo(3, 'corner', { x: 10, z: 18 });
  const opp2 = opp(L3D.state.data.annotations[0]);
  close(opp2[0], oppPre[0], 1e-9, 'corner 两轮对角固定 x（快照式不漂移）');
  close(opp2[1], oppPre[1], 1e-9, 'corner 两轮对角固定 z（快照式不漂移）');
  L3D.endEdit();
  ok(L3D.state.data.annotations[0].edited_by_human === true, 'corner 拖动置人工标记');
  L3D.undo();
  ok(JSON.stringify(L3D.state.data.annotations[0]) === pre14, 'corner 拖动可撤销');

  // view=3 edge 拖动（右边 sx=1：单轴 w 变 l 不动，左边固定）
  const pre14e = JSON.stringify(L3D.state.data.annotations[0]);
  const leftEdge = (b) => { // 左边（x = cx - 右向·w/2）：对边固定锚
    const px = Math.cos(yaw14), pz = -Math.sin(yaw14);
    return [b.cx - px * b.w / 2, b.cz - pz * b.w / 2];
  };
  const lePre = leftEdge(L3D.state.data.annotations[0]);
  L3D.beginEdit(0, 3, 'edge', { t0: -yaw14, cx0: 8, cz0: 18, w0: 1.6, l0: 3.9, sx: 1, sy: 0 });
  L3D.editTo(3, 'edge', { x: 9, z: 18 });
  close(L3D.state.data.annotations[0].l, 3.9, 1e-9, 'edge 单轴 l 不变');
  const leMid = leftEdge(L3D.state.data.annotations[0]);
  close(leMid[0], lePre[0], 1e-9, 'edge 对边（左）固定 x');
  close(leMid[1], lePre[1], 1e-9, 'edge 对边（左）固定 z');
  L3D.endEdit();
  L3D.undo();
  ok(JSON.stringify(L3D.state.data.annotations[0]) === pre14e, 'edge 拖动可撤销');

  console.log(`smoke_web3d: ${n} 断言，${process.exitCode ? 'FAIL' : 'OK'}`);
})();
