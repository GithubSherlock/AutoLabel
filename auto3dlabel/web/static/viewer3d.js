// Auto3dLabel v0.3 P4 渲染装配层（three r149 UMD 全局 THREE）。
// 依赖 L3D（logic3d.js）：camToThree/edgesOf/headLine/orthoFocus 纯数学 + state 订阅。
// 四视图 2×2 scissor：透视（旋转/缩放/选中）+ Top/Side/Front 正交（只显选中框 + 自动聚焦，
// CVAT 借鉴）。点云与角点均相机系交付、此处换轴 (x, z, -y) 渲染——零 calib 依赖。
(function (root) {
  "use strict";
  const L3D = root.L3D;
  if (!L3D || !root.THREE) return; // jsdom 冒烟不加载本文件（缺依赖静默）

  const canvas = document.getElementById("view3d-canvas");
  if (!canvas) return;

  const THREE = root.THREE;
  const COLOR_MAP = { Car: 0x4da3ff, Pedestrian: 0x58d68d, Cyclist: 0xe8b339 };
  const SEL_COLOR = 0xffd166;
  const colorFor = (label) => COLOR_MAP[label] || 0xb0b6c0;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x14161a);

  const scene = new THREE.Scene();
  const grid = new THREE.GridHelper(160, 32, 0x3a4150, 0x262c38);
  grid.position.y = 0;
  scene.add(grid);

  // 点云（换轴后 Float32Array [x, z, -y]）与包围球
  const pointsGeo = new THREE.BufferGeometry();
  const pointsMat = new THREE.PointsMaterial({
    color: 0x8fa3bf, size: 1.2, sizeAttenuation: true,
  });
  const pointCloud = new THREE.Points(pointsGeo, pointsMat);
  pointCloud.frustumCulled = false;
  scene.add(pointCloud);
  let cloudCenter = new THREE.Vector3(0, 0, 0);
  let cloudRadius = 30;

  // 四视图：左上透视 / 右上 Top / 左下 Side / 右下 Front
  const persp = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
  const makeOrtho = () => new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 1000);
  const cams = [persp, makeOrtho(), makeOrtho(), makeOrtho()];
  const ORTHO_AXIS = [
    null,
    new THREE.Vector3(0, 1, 0),    // Top：俯视 X-Y 平面（地面）
    new THREE.Vector3(1, 0, 0),    // Side：从 +x 看（车右侧）
    new THREE.Vector3(0, 0, 1),    // Front：从 +z 看（车头方向，y 朝上）
  ];
  // 透视球坐标（相对 target）
  let theta = 0.65, phi = 1.05, radius = 60;
  const perspTarget = new THREE.Vector3(0, 0, 0);

  // 场景对象：{index, group, edges, head, center, radius}
  const objects = [];
  let builtForPayload = null;

  function setCloud(points) {
    const n = points.length;
    const buf = new Float32Array(n * 3);
    let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const t = L3D.camToThree(points[i][0], points[i][1], points[i][2]);
      buf[i * 3] = t[0]; buf[i * 3 + 1] = t[1]; buf[i * 3 + 2] = t[2];
      for (let d = 0; d < 3; d++) {
        if (t[d] < min[d]) min[d] = t[d];
        if (t[d] > max[d]) max[d] = t[d];
      }
    }
    pointsGeo.setAttribute("position", new THREE.BufferAttribute(buf, 3));
    pointsGeo.computeBoundingSphere();
    cloudCenter.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    cloudRadius = Math.max(
      Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 10
    );
    if (!L3D.state.selected) perspTarget.copy(cloudCenter);
    radius = cloudRadius * 2.2;
  }

  function rebuildObjects(payload) {
    for (const o of objects) scene.remove(o.group);
    objects.length = 0;
    builtForPayload = payload;
    for (const obj of payload.objects || []) {
      const corners = obj.corners;
      if (!corners || corners.length !== 8) continue;
      const group = new THREE.Group();
      const pts = [];
      for (const e of L3D.edgesOf(corners)) {
        for (const c of e) pts.push(...L3D.camToThree(c[0], c[1], c[2]));
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
      const edges = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
        color: colorFor(obj.label),
      }));
      edges.frustumCulled = false;
      group.add(edges);

      const head = L3D.headLine(corners);
      if (head) {
        const hp = [...L3D.camToThree(head[0][0], head[0][1], head[0][2]),
          ...L3D.camToThree(head[1][0], head[1][1], head[1][2])];
        const hgeo = new THREE.BufferGeometry();
        hgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(hp), 3));
        const hline = new THREE.Line(hgeo, new THREE.LineBasicMaterial({ color: 0xe5534b }));
        hline.frustumCulled = false;
        group.add(hline);
      }
      const focus = L3D.orthoFocus(corners);
      scene.add(group);
      objects.push({
        index: obj.index,
        label: obj.label,
        group,
        edges,
        center: new THREE.Vector3(...focus.center),
        radius: focus.radius,
      });
    }
  }

  function syncObjects() {
    const st = L3D.state;
    if (st.payload !== builtForPayload) {
      setCloud(st.payload && st.payload.points ? st.payload.points : []);
      rebuildObjects(st.payload);
    }
    focusSelected();
  }

  // 选中对象 → 正交三视图自动聚焦（CVAT 借鉴：正交投影只显选中对象 + 自动聚焦）
  function focusSelected() {
    const st = L3D.state;
    const sel = st.selected != null ? objects.find((o) => o.index === st.selected) : null;
    if (!sel) return;
    const c = sel.center, r = sel.radius;
    for (let i = 1; i < 4; i++) {
      const cam = cams[i];
      const dist = r * 4 + 1;
      cam.position.set(
        c.x + ORTHO_AXIS[i].x * dist,
        c.y + ORTHO_AXIS[i].y * dist,
        c.z + ORTHO_AXIS[i].z * dist
      );
      cam.lookAt(c);
      const s = r * 2.4;
      cam.left = -s; cam.right = s; cam.top = s; cam.bottom = -s;
      cam.updateProjectionMatrix();
    }
    perspTarget.copy(c);
    radius = Math.max(r * 3.5, 4);
  }

  function syncMaterials(viewIndex) {
    const ortho = viewIndex > 0;
    pointsMat.sizeAttenuation = !ortho;
    pointsMat.size = ortho ? 1.0 : 1.2;
    const st = L3D.state;
    for (const o of objects) {
      const sel = st.selected === o.index;
      // 正交视图只显选中框；透视视图全部显示
      o.group.visible = ortho ? sel : true;
      // 标签现取（改标签后颜色即时更新）
      const ann = st.data && st.data.annotations && st.data.annotations[o.index];
      const label = ann ? ann.label : o.label;
      o.edges.material.color.setHex(sel ? SEL_COLOR : colorFor(label));
    }
  }

  L3D.on(syncObjects);

  function updatePersp() {
    persp.position.set(
      perspTarget.x + radius * Math.sin(theta) * Math.cos(phi),
      perspTarget.y + radius * Math.sin(phi),
      perspTarget.z + radius * Math.cos(theta) * Math.cos(phi)
    );
    persp.lookAt(perspTarget);
  }

  function layout() {
    const w = canvas.clientWidth || 640, h = canvas.clientHeight || 480;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      renderer.setSize(w, h, false);
    }
    for (const cam of cams) {
      cam.aspect = w / h / 2;
      cam.updateProjectionMatrix();
    }
  }

  function render() {
    requestAnimationFrame(render);
    layout();
    updatePersp();
    const w = canvas.width, h = canvas.height;
    const halfW = Math.floor(w / 2), halfH = Math.floor(h / 2);
    for (let i = 0; i < 4; i++) {
      const x = i % 2 === 0 ? 0 : halfW;
      const y = i < 2 ? 0 : halfH; // 左上 0 / 右上 1 / 左下 2 / 右下 3
      syncMaterials(i);
      renderer.setViewport(x, y, halfW, halfH);
      renderer.setScissor(x, y, halfW, halfH);
      renderer.setScissorTest(true);
      renderer.render(scene, cams[i]);
    }
    renderer.setScissorTest(false);
  }

  // ── 交互：拖拽旋转 / 滚轮缩放 / 点击选中 ──────────────────────
  let drag = null; // {x, y, qi, moved}
  function viewAt(e) {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const w = canvas.clientWidth || 640, h = canvas.clientHeight || 480;
    const halfW = w / 2, halfH = h / 2;
    return {
      qi: (py < halfH ? 0 : 2) + (px < halfW ? 0 : 1),
      nx: ((px % halfW) / halfW) * 2 - 1,
      ny: -(((py % halfH) / halfH) * 2 - 1),
    };
  }

  canvas.addEventListener("mousedown", (e) => {
    drag = { x: e.clientX, y: e.clientY, qi: viewAt(e).qi, moved: false };
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.qi === 0) {
      theta -= dx * 0.006;
      phi = Math.min(1.45, Math.max(0.15, phi - dy * 0.006));
    }
    drag.x = e.clientX; drag.y = e.clientY;
  });
  window.addEventListener("mouseup", () => { drag = null; });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const qi = viewAt(e).qi;
    const k = 1 + e.deltaY * 0.001;
    if (qi === 0) {
      radius = Math.min(400, Math.max(2, radius * k));
    } else {
      const cam = cams[qi];
      cam.zoom = Math.min(8, Math.max(0.2, cam.zoom / k));
      cam.updateProjectionMatrix();
    }
  }, { passive: false });

  canvas.addEventListener("click", (e) => {
    if (drag && drag.moved) return; // 拖拽不算点击
    const v = viewAt(e);
    const ray = new THREE.Raycaster(
      new THREE.Vector2(v.nx, v.ny), cams[v.qi]
    ).ray;
    let best = null, bestD = Infinity;
    for (const o of objects) {
      if (v.qi > 0 && L3D.state.selected !== o.index) continue; // 正交视图只可点选中框
      const sphere = new THREE.Sphere(o.center, o.radius);
      if (ray.intersectsSphere(sphere)) {
        const d = ray.distanceToPoint(o.center) - o.radius;
        if (d < bestD) { bestD = d; best = o; }
      }
    }
    if (best) L3D.selectObject(best.index);
  });

  syncObjects();
  render();
})(typeof window !== "undefined" ? window : globalThis);
