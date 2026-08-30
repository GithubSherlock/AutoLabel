"""自动车道 ROI（v0.5 C2）—— Ultra-Fast-Lane-Detection（UFLD）车道线检测。

官方 tusimple_res18.pth 只挂 Google Drive / 百度网盘（AutoDL 均不可直连），
按 milestone/v0.5.md 预案降级 **ailia ONNX**（同一官方权重官方转换，
storage.googleapis.com 可达，`weights/download_lane_weights.sh` 下载 +
sha256 校验）。onnxruntime CPU 推理 288×800 ResNet18 单图百 ms 级，
满足「序列级一次」车道 ROI 场景（`--roi auto`，首帧解析不逐帧）。

预处理 = 官方训练语义（全图 squash Resize(288,800) + ImageNet 归一化），
坐标按 x/y 独立反变换回原图系（线性可逆，同 Florence-2 方形输入红线）。

接口（Protocol，duck typing 注入 Fake 零真实权重单测铁律）：
    detect_lanes(image_path) -> list[list[(x, y)]]  车道点列（原图坐标，上→下）
    resolve_auto_roi(model, image_path) -> 自车车道闭合多边形 | None

自车车道 = 底端 x 夹住画面中线的相邻两线（左线正序 + 右线反序闭合）；
不足 2 线回退 None（调用方黄字 + 全图保留，宁多勿漏）。
"""

from __future__ import annotations

from typing import Any, Protocol

from auto2dlabel.configs.model_catalog import WEIGHTS_DIR

# 官方 configs/tusimple.py + data/constant.py 常量（UFLD 训练/解码协议）
LANE_INFER_W = 800
LANE_INFER_H = 288
GRIDING_NUM = 100          # 列网格数（+1 为无车道类）
NUM_ROW = 56               # 行锚数
NUM_LANES = 4              # 最多车道数

# tusimple 行锚（288 输入空间 y 坐标，对应全图 y = anchor * orig_h / 288）
TUSIMPLE_ROW_ANCHOR = [
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
    184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236,
    240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284,
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LANE_WEIGHTS_PATH = WEIGHTS_DIR / "tusimple_18.onnx"


class LaneModel(Protocol):
    """车道线检测协议 —— 测试注入 FakeLaneModel（零真实权重铁律，同 ReID 模式）。"""

    def detect_lanes(self, image_path: str) -> list[list[tuple[float, float]]]:
        """检测车道线 → 每条线为 (x, y) 点列（原图坐标，上→下，≥3 点）。"""
        ...


def _softmax_cols(x: Any) -> Any:
    """axis=0 softmax（数值稳定），返回 numpy 数组。"""
    import numpy as np

    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def decode_lane_output(
    out: Any, orig_w: int, orig_h: int
) -> list[list[tuple[float, float]]]:
    """UFLD 输出 (GRIDING_NUM+1, NUM_ROW, NUM_LANES) → 车道点列（原图坐标）。

    仿官方 generate_tusimple_lines rel 定位（软 argmax 期望列）+ ailia ONNX
    行序修正：raw 行 k ↔ row_anchor[NUM_ROW-1-k]（ailia 转换器行序与官方
    pytorch 相反，冒烟实测验证）。无车道行（argmax = 无车道类）丢弃；
    有效点 <3 的车道整体丢弃。x = loc * col_w * orig_w / 800、
    y = anchor * orig_h / 288（squash 反变换）。
    """
    import numpy as np

    a = np.asarray(out, dtype=np.float32)
    a = a[:, ::-1, :]  # ONNX 行序反转为官方 pytorch 行序（k ↔ 55-k）
    prob = _softmax_cols(a[:-1, :, :])
    idx = (np.arange(GRIDING_NUM, dtype=np.float32) + 1).reshape(-1, 1, 1)
    loc = (prob * idx).sum(axis=0)  # (NUM_ROW, NUM_LANES) 软 argmax 期望列
    loc[a.argmax(axis=0) == GRIDING_NUM] = -1.0  # 无车道哨兵
    col_w = (LANE_INFER_W - 1) / (GRIDING_NUM - 1)
    lanes: list[list[tuple[float, float]]] = []
    for j in range(loc.shape[1]):
        pts: list[tuple[float, float]] = []
        for k in range(loc.shape[0]):
            if loc[k, j] < 0:
                continue
            x = loc[k, j] * col_w * orig_w / LANE_INFER_W
            y = TUSIMPLE_ROW_ANCHOR[NUM_ROW - 1 - k] * orig_h / LANE_INFER_H
            pts.append((float(x), float(y)))
        pts.reverse()  # 翻转后 k 升序 = 下→上，反转为上→下（anchor 升序）
        if len(pts) >= 3:
            lanes.append(pts)
    return lanes


def select_ego_lane_pair(
    lanes: list[list[tuple[float, float]]], img_width: int
) -> list[list[tuple[float, float]]]:
    """自车车道两线选择：按底端 x 排序后取夹住画面中线的相邻对。

    4 线场景（左左/左/右/右右）底端夹线对即 lane1/lane2（自车两侧）；
    全部在中线一侧（弯道）时退化为最靠近中线的两线；<2 线返回空列表。
    """
    if len(lanes) < 2:
        return []
    ordered = sorted(lanes, key=lambda lane: lane[-1][0])
    cx = img_width / 2
    for i in range(len(ordered) - 1):
        if ordered[i][-1][0] <= cx <= ordered[i + 1][-1][0]:
            return [ordered[i], ordered[i + 1]]
    return sorted(ordered, key=lambda lane: abs(lane[-1][0] - cx))[:2]


def points_to_polygon(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """左右两线 → 闭合多边形（左线正序上→下 + 右线反序下→上）。"""
    left_pts = [(float(x), float(y)) for x, y in left]
    right_pts = [(float(x), float(y)) for x, y in right]
    return left_pts + list(reversed(right_pts))


def resolve_auto_roi(
    model: LaneModel, image_path: str
) -> list[tuple[float, float]] | None:
    """首帧自动车道 ROI：检测车道 → 自车两线 → 闭合多边形；不足 2 线回退 None。

    图片尺寸只读 PIL 头部（零解码开销）；调用方负责 None 降级
    （黄字提示 + 无 ROI 全图保留，宁多勿漏）。
    """
    lanes = model.detect_lanes(image_path)
    if len(lanes) < 2:
        return None
    from PIL import Image

    w, _ = Image.open(image_path).size
    pair = select_ego_lane_pair(lanes, w)
    if not pair:
        return None
    return points_to_polygon(pair[0], pair[1])


class UltraFastLaneONNXModel:
    """UFLD 车道线检测（ailia ONNX 权重，onnxruntime CPU 推理，懒加载）。

    detect_lanes：cv2 读图 → squash Resize(288,800) → ImageNet 归一化 →
    NCHW → session.run → decode_lane_output。onnxruntime 未装抛 ImportError
    （守卫在 _load，构造零加载）。
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or str(LANE_WEIGHTS_PATH)
        self._session: Any = None

    def _load(self) -> Any:
        """懒加载 onnxruntime 会话（幂等；onnxruntime 未装抛 ImportError）。"""
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime 未安装，请运行: pip install onnxruntime")
        self._session = ort.InferenceSession(
            self._weights_path, providers=["CPUExecutionProvider"]
        )
        return self._session

    def detect_lanes(self, image_path: str) -> list[list[tuple[float, float]]]:
        """检测车道线（原图坐标点列，上→下）。"""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"车道检测无法读取图片: {image_path}")
        orig_h, orig_w = img.shape[:2]
        x = cv2.resize(img, (LANE_INFER_W, LANE_INFER_H))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            x[:, :, c] = (x[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        blob = np.expand_dims(x.transpose(2, 0, 1), 0)
        session = self._load()
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: blob})[0][0]
        return decode_lane_output(out, orig_w, orig_h)


def create_lane_model(weights_path: str | None = None) -> LaneModel:
    """工厂函数：创建车道线检测模型（当前仅 UFLD ONNX 一种实现）。"""
    return UltraFastLaneONNXModel(weights_path=weights_path)
