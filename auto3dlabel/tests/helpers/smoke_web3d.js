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

  console.log(`smoke_web3d: ${n} 断言，${process.exitCode ? 'FAIL' : 'OK'}`);
})();
