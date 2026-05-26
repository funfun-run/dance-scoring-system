# 舞蹈评分系统 — 项目骨架重构设计规格 v2.0

> **目标**：基于竞赛方案书（设计文档.doc）和 DK-2500 硬件规格，将项目重组为适配嵌入式边缘计算的分层 Python 包，为 OpenVINO 加速、摄像头实时跟练、ROS2 节点化、无线传输和硬件 GPIO 控制留好扩展点。
>
> **读者**：AI agent 和人类开发者。本文档是后续所有功能开发的权威结构参考。

---

## 一、目标文件树

```
intel/
├── pyproject.toml                      # pip install -e . 可编辑安装
├── requirements.txt                    # 精确 pin 版本
├── README.md
├── .gitignore
│
├── scripts/                            # CLI 入口（无 ROS2、无 GUI 也能跑）
│   ├── score.py                        # 离线评分
│   ├── split.py                        # 视频八拍分割
│   └── run_live.py                     # 摄像头实时跟练
│
├── src/dance_scoring/                  # 主包
│   ├── __init__.py
│   │
│   ├── core/                           # AI推理层
│   │   ├── __init__.py
│   │   ├── config.py                   # 全局配置 & 评分参数（源自 d3e03e9 初版）
│   │   ├── frame.py                    # PoseFrame 数据类
│   │   ├── extractor.py                # MediaPipe Pose 姿态提取
│   │   ├── inference.py                # OpenVINO 推理引擎（占位）
│   │   ├── alignment.py                # fastdtw 时序对齐 & 相似度评分
│   │   ├── scorer.py                   # 评分引擎（总分 + 分段 + 薄弱环节）
│   │   ├── segments.py                 # 分段结果 & 练习片段生成
│   │   └── correction.py              # 部位纠正（预留占位）
│   │
│   ├── video/                          # 数据处理层
│   │   ├── __init__.py
│   │   ├── info.py                     # 视频元信息
│   │   ├── beat_detector.py            # Madmom 节拍检测（音频优先，光流退化）
│   │   ├── splitter.py                 # 八拍切割 & 慢动作提取
│   │   └── merger.py                   # 片段合并
│   │
│   ├── camera/                         # 感知采集层
│   │   ├── __init__.py
│   │   ├── base.py                     # CameraBase ABC
│   │   ├── usb.py                      # USB 摄像头（DK-2500: 3×USB-A + 1×USB-C）
│   │   └── stream.py                   # 帧缓冲 & 流管理
│   │
│   ├── gui/                            # 交互反馈层（HDMI 外接屏）
│   │   ├── __init__.py
│   │   ├── app.py                      # 主窗口（部署时全屏模式）
│   │   ├── components.py               # 可复用组件
│   │   └── worker.py                   # 后台任务线程
│   │
│   ├── platform/                       # DK-2500 硬件适配层
│   │   ├── __init__.py
│   │   ├── npu.py                      # OpenVINO NPU 加速接口（占位）
│   │   └── gpio.py                     # 40-pin GPIO（占位）
│   │
│   ├── transfer/                       # 数据交换
│   │   ├── __init__.py
│   │   ├── base.py                     # TransferBase ABC
│   │   ├── wifi.py                     # M.2 E-Key WiFi（WiFi5/WiFi6）
│   │   └── bluetooth.py               # M.2 E-Key BT
│   │
│   └── ros2/                           # ROS2 节点层（可选，预留）
│       ├── __init__.py
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── camera_node.py          # CameraBase → Image topic
│       │   ├── pose_node.py            # PoseExtractor → PoseLandmarks topic
│       │   ├── beat_node.py            # 节拍检测 → Service
│       │   ├── alignment_node.py       # fastdtw 对齐 → Service
│       │   └── scoring_node.py         # Scorer → Action（异步评分 + 进度）
│       ├── interfaces/
│       │   ├── __init__.py
│       │   ├── PoseLandmarks.msg       # 33关键点 + 置信度
│       │   ├── ScoreResult.msg         # 总分 + 分段 + 薄弱环节
│       │   └── CorrectionHint.msg      # 部位纠正建议（预留）
│       └── launch/
│           ├── scoring_system.launch.py  # 全节点启动
│           └── live_scoring.launch.py    # 实时跟练节点
│
├── docs/
│   └── specs/
│       └── 2026-05-26-project-restructure-design.md
│
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_extractor.py
    └── test_alignment.py
```

注：`output/` 目录和模型缓存（`~/.cache/dance_scoring/`）均为运行时自动创建，不纳入源码骨架。

---

## 二、依赖层次图

```
                  scripts/         gui/           ros2/
                     │              │              │
                     └──────┬───────┘──────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       camera/           video/            core/
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                   ┌────────┴────────┐
                   │                 │
               platform/         transfer/

        互相不依赖，各自独立      互相不依赖，各自独立
```

### Import 白名单

| 从 | 可以 import | 禁止 |
|----|------------|------|
| `core` | core 内部, numpy, scipy, fastdtw, mediapipe | gui, video, camera, platform, transfer, tkinter |
| `video` | core.config, video 内部, cv2, numpy, madmom | gui, camera, platform, tkinter |
| `camera` | cv2, numpy, abc | core, video, gui, platform, transfer |
| `gui` | core, video, camera, tkinter | platform, transfer, ros2 |
| `platform` | core, openvino, GPIO 适配层 | gui, transfer, video, camera |
| `transfer` | stdlib + socket | 所有业务模块 |
| `ros2` | core, video, camera, rclpy | gui, platform, transfer |
| `scripts` | 所有模块 | — |

---

## 三、各模块接口清单

### 3.1 `core/config.py` — 全局配置

```python
# ======== 评分参数源自 d3e03e9843ffce875ab03db868c43dfc387aacdb 初版提交 ========

@dataclass
class Config:
    score_threshold: float = 50.0
    target_fps: int = 30
    keypoint_confidence: float = 0.5
    interp_window: int = 3

# 常量（模块级）
BEATS_PER_SEGMENT: int = 8
DEFAULT_BPM: int = 120
SCORE_THRESHOLD: float = 50.0
PASS_SCORE: float = 60.0
SLOW_SPEED: float = 0.8
TARGET_FPS: int = 30
MIN_SEGMENT_DURATION: float = 0.3

ANGLE_JOINTS: List[Tuple[int, int, int]] = [
    (1, 0, 4), (7, 0, 8), (7, 0, 11), (8, 0, 12),
    (11, 13, 15), (12, 14, 16), (13, 15, 17), (14, 16, 18),
    (13, 15, 19), (14, 16, 20), (15, 17, 19), (16, 18, 20),
    (13, 11, 23), (14, 12, 24), (11, 23, 24), (12, 24, 23),
    (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28),
    (25, 27, 29), (26, 28, 30), (25, 27, 31), (26, 28, 32),
    (27, 29, 31), (28, 30, 32),
]

ANGLE_WEIGHTS: np.ndarray = np.array([
    1.0, 1.0, 0.8, 0.8, 1.3, 1.3, 0.6, 0.6, 0.5, 0.5,
    0.4, 0.4, 1.2, 1.2, 1.4, 1.4, 1.3, 1.3, 1.5, 1.5,
    0.8, 0.8, 0.7, 0.7, 0.6, 0.6
], dtype=np.float32)

SCORE_TOLERANCE: float = 3.0
SCORE_PENALTY_SMALL: float = 1.8
SCORE_PENALTY_LARGE: float = 3.0
SCORE_PENALTY_THRESHOLD: float = 15.0
DTW_WINDOW_RATIO: float = 0.1
Z_AXIS_WEIGHT: float = 0.3

MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_CACHE_DIR: str = "~/.cache/dance_scoring"
```

### 3.2 `core/frame.py` — 姿态数据

```python
@dataclass
class PoseFrame:
    fid: int                    # 帧序号
    kp3d: np.ndarray            # shape (33, 3), world landmarks
    conf: np.ndarray            # shape (33,), 置信度
    angles: np.ndarray = None   # shape (26,), __post_init__ 自动计算
    vec: np.ndarray = None      # shape (66+26,), __post_init__ 自动拼装

    def __post_init__(self) -> None: ...
    def _calc_angles(self) -> np.ndarray: ...
```

### 3.3 `core/extractor.py` — 姿态提取

```python
def download_model() -> None:
    """下载 MediaPipe 模型到 MODEL_CACHE_DIR，首次运行时自动调用"""

class PoseExtractor:
    def __init__(self, cfg: Config) -> None: ...

    def extract(
        self,
        source: str | int,                     # 视频路径 或 摄像头 device_id
        progress_callback: Callable[[int], None] | None = None
    ) -> List[PoseFrame]:
        """
        从视频文件或摄像头实时流提取所有帧的姿态。
        progress_callback: 参数为 0-100 的进度百分比
        """

    def _interpolate(self, poses: List[PoseFrame]) -> List[PoseFrame]:
        """低置信度关键点相邻帧线性插值修复"""
```

### 3.4 `core/inference.py` — OpenVINO 推理引擎（占位）

```python
"""
OpenVINO 推理加速接口层。
TODO: 具体实现需结合 MediaPipe + OpenVINO 集成方式确定。
MediaPipe 本身支持 OpenVINO delegate，后续 agent 需调研：
- MediaPipe 的 OpenVINO delegate 配置方式
- 是否可直接通过 MediaPipe Python API 指定 OpenVINO 后端
- 或需要独立加载 OpenVINO IR 模型并通过 inference engine 调用
参考: Intel Core Ultra 5 225U 集成 NPU，Ubuntu 22.04 + OpenVINO 工具链
"""

@dataclass
class InferenceConfig:
    """占位配置，字段待后续按实际方案调整"""
    device: str = "NPU"          # "NPU" | "CPU" | "GPU"
    precision: str = "INT8"      # "FP32" | "FP16" | "INT8"


class PoseInferenceEngine:
    """
    姿态推理加速引擎（占位）。
    当前直接使用 MediaPipe PoseLandmarker（core/extractor.py），
    无需经过此类。后续引入 OpenVINO 加速时再填充实现。
    """
    def __init__(self, model_path: str, cfg: InferenceConfig) -> None:
        raise NotImplementedError(
            "OpenVINO inference not yet implemented. "
            "See core/inference.py docstring for integration guidance."
        )
```

### 3.5 `core/alignment.py` — 时序对齐 & 相似度

```python
def dtw_constrained(
    mat: np.ndarray,
    window: int
) -> Tuple[List[Tuple[int, int]], float]:
    """
    约束窗口 DTW 算法，O(N) 复杂度。
    mat: (n_ref, n_user) 距离矩阵
    window: 搜索窗口大小
    返回: (对齐路径, 总代价)
    """

def fastdtw_align(
    ref_features: np.ndarray,
    user_features: np.ndarray,
    radius: int = 1
) -> Tuple[List[Tuple[int, int]], float]:
    """
    fastdtw 时序对齐（备选方案）。
    内部调用 fastdtw 库，作为 dtw_constrained 的替代加速选项。
    使用时机：序列长度差超过 2x 时自动切换。
    """

def similarity_score(min_dist: float) -> float:
    """Sigmoid 归一化: Similarity = 1 / (1 + min_dist)，映射到 0~100 分制"""
```

### 3.6 `core/scorer.py` — 评分引擎

```python
class Scorer:
    def __init__(self, cfg: Config, bpm: int = DEFAULT_BPM) -> None: ...

    def score(
        self,
        ref: List[PoseFrame],
        user: List[PoseFrame],
        progress_callback: Callable[[int, str], None] | None = None
    ) -> Tuple[float, List[dict], List[dict], List[Tuple[int, int]]]:
        """
        返回: (总评分数, 分段结果列表, 低分段落列表, DTW 对齐路径)
        progress_callback: (百分比, 阶段描述)
        """

    def _grade_overall(self, fs: List[float], segs: List[dict]) -> float: ...
    def _nonlinear_score(self, avg_diff: float) -> float: ...


def locate_weak_points(
    segs: List[dict],
    path: List[Tuple[int, int]],
    ref_poses: List[PoseFrame],
    user_poses: List[PoseFrame]
) -> List[dict]:
    """
    薄弱环节定位（新增扩展函数，不修改 Scorer 类）。
    返回: [{'segment_id','joint_index','deviation_angle','frame_id'}, ...]
    """
```

### 3.7 `core/segments.py` — 分段与练习片提取

```python
def seg_by_beats(
    ref: List[PoseFrame],
    path: List[Tuple[int, int]],
    fs: List[float],
    target_fps: int,
    bpm: int
) -> List[dict]:
    """按 BPM 八拍分段，返回每段 [{'id','start_time','end_time','score','qualified'}, ...]"""

def extract_clips_from_segments(
    segs: List[dict],
    segments_dir: str = "output/segments",
    out_dir: str = "output/low_score_clips",
    cfg: Config = None
) -> List[str]:
    """提取低于阈值的段落慢动作视频，返回输出文件路径列表"""
```

### 3.8 `core/correction.py` — 部位纠正（预留占位）

```python
"""
舞蹈动作部位纠正建议生成。
TODO: 后续实现关节→中文部位名映射 + 纠正方向判断。
"""

# 关节 → 中文部位名映射（预留）
JOINT_NAMES_CN: Dict[int, str] = {}   # e.g. {15: "右肘", 13: "右膝", ...}

def generate_correction(
    joint_index: int,
    ref_angle: float,
    user_angle: float,
    deviation: float
) -> dict:
    """
    返回:
    {
        'joint_name': '右肘',
        'direction': 'too_bent' | 'too_straight' | 'misaligned',
        'suggestion': '请将右肘稍抬高',
        'severity': 'minor' | 'moderate' | 'major'
    }
    """
    raise NotImplementedError()
```

### 3.9 `video/info.py`

```python
def get_video_info(video_path: str) -> Tuple[float, int, float, int, int]:
    """返回: (fps, frame_count, duration_seconds, width, height)"""
```

### 3.10 `video/beat_detector.py` — 节拍检测

```python
def detect_beats_from_audio(video_path: str) -> Tuple[List[float], float] | None:
    """
    Madmom 音频节拍检测（优先方案）。
    返回: (beat_times_seconds, bpm) 或 None
    """

def detect_beats_from_motion(video_path: str) -> Tuple[List[float], float] | None:
    """
    光流运动节拍检测（退化方案，无音频轨时使用）。
    返回: (beat_times_seconds, estimated_bpm) 或 None
    """

def detect_beats(video_path: str) -> Tuple[List[float], float]:
    """
    自动选择策略（新增扩展函数，不修改现有两个检测函数）：
    音频 → 光流 → 固定BPM兜底，必须返回有效值。
    """
```

### 3.11 `video/splitter.py`

```python
def get_beat_segments(
    beat_times: List[float],
    duration: float,
    beats_per_seg: int = BEATS_PER_SEGMENT
) -> List[dict] | None:
    """根据节拍时间点生成分段列表"""

def calculate_segments_fixed(
    duration_seconds: float,
    bpm: int
) -> List[dict]:
    """固定 BPM 计算分段"""

def extract_slow_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str
) -> None:
    """提取单个 0.8x 慢动作片段"""
```

### 3.12 `video/merger.py`

```python
def merge_videos(video_list: List[str], output_path: str) -> None:
    """将多个视频片段合并为一个文件"""
```

### 3.13 `camera/base.py`

```python
from abc import ABC, abstractmethod

class CameraBase(ABC):
    @abstractmethod
    def open(self, device_id: int = 0, width: int = 640, height: int = 480) -> bool: ...
    @abstractmethod
    def read(self) -> np.ndarray | None: ...
    @abstractmethod
    def close(self) -> None: ...
    @staticmethod
    def list_devices() -> List[Tuple[int, str]]: ...
        """返回 [(device_id, device_name), ...]"""
```

### 3.14 `camera/usb.py`

```python
class USBCamera(CameraBase):
    """USB 摄像头实现（DK-2500: 3×USB-A + 1×USB-C）"""
    def open(self, device_id: int = 0, width: int = 640, height: int = 480) -> bool: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...
```

### 3.15 `camera/stream.py`

```python
class FrameBuffer:
    """线程安全固定容量帧缓冲，新帧覆盖旧帧"""
    def put(self, frame: np.ndarray) -> None: ...
    def get(self) -> np.ndarray | None: ...

class CameraStream:
    """后台线程持续读取摄像头，帧存入 FrameBuffer"""
    def __init__(self, camera: CameraBase, fps: int = TARGET_FPS) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_latest(self) -> np.ndarray | None: ...
```

### 3.16 `gui/app.py`

```python
class MainApp:
    """Tkinter 主窗口，部署时 HDMI 外接屏全屏"""
    def __init__(self) -> None: ...
    def run(self) -> None: ...
    def _do_score(self) -> None: ...
    def _do_split(self) -> None: ...
    def _do_live_practice(self) -> None: ...    # 新增：实时跟练入口
```

### 3.17 `gui/components.py`

```python
class VideoPreview(tk.Frame):
    """视频预览面板"""
    def load(self, path: str) -> None: ...
    def stop(self) -> None: ...

class ProgressDialog(tk.Toplevel):
    """进度条弹窗"""
    def update_progress(self, percent: int, msg: str = "") -> None: ...

class ScoreResultDialog(tk.Toplevel):
    """总分 + 分段成绩表 + 薄弱环节列表"""

class SegmentListDialog(tk.Toplevel):
    """分割结果列表 + 逐段播放"""

class PracticeOverlay(tk.Frame):                # 新增
    """实时跟练叠加层：当前得分 + 关节偏差提示"""
```

### 3.18 `gui/worker.py`

```python
class Worker:
    """后台任务基类"""
    def start(self) -> None: ...
    def cancel(self) -> None: ...
    def is_running(self) -> bool: ...

class SplitWorker(Worker): ...
class ScoreWorker(Worker): ...
class LivePracticeWorker(Worker): ...           # 新增：实时跟练后台线程
```

### 3.19 `platform/npu.py` — DK-2500 NPU 加速（占位）

```python
"""
DK-2500 NPU 加速接口层。
TODO: Intel Core Ultra 5 225U 集成 NPU，通过 OpenVINO 工具链调用。
后续 agent 需调研：
- 如何在 Ubuntu 22.04 上安装 OpenVINO NPU 驱动
- MediaPipe Pose 模型如何导出为 OpenVINO IR 格式
- 推理时如何指定 NPU device
参见 core/inference.py 占位说明。
"""

class NPUManager:
    """NPU 设备管理（占位）"""
    @staticmethod
    def available() -> bool:
        """检测 OpenVINO NPU 运行时是否可用"""
        raise NotImplementedError()
```

### 3.20 `platform/gpio.py` — 40-pin GPIO（占位）

```python
"""
DK-2500 40-pin GPIO 封装（占位）。
后续 agent 参考 deepseek_markdown_20260526_2912f4.md 中的引脚定义表实现。

预留用途（参考 DK-2500 40-pin 定义）：
- GPIO3（引脚7）：  物理按键-开始跟练
- GPIO6（引脚15）： LED-就绪状态
- GPIO19（引脚16）：LED-评分中
"""

class GPIO:
    OUT = 0
    IN = 1

    @staticmethod
    def setup(pin: int, mode: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def output(pin: int, value: bool) -> None:
        raise NotImplementedError()

    @staticmethod
    def input(pin: int) -> bool:
        raise NotImplementedError()

    @staticmethod
    def cleanup() -> None:
        raise NotImplementedError()
```

### 3.21 `transfer/base.py`

```python
from abc import ABC, abstractmethod

class TransferBase(ABC):
    @abstractmethod
    def send(self, file_path: str, target: str) -> bool: ...
    @abstractmethod
    def receive(self, save_dir: str) -> str | None: ...
    @abstractmethod
    def discover(self) -> List[str]: ...
```

### 3.22 `transfer/wifi.py` & `transfer/bluetooth.py`（占位）

```python
class WiFiTransfer(TransferBase):
    """M.2 E-Key WiFi 传输（WiFi5/WiFi6）"""
    def send(self, file_path: str, target: str) -> bool:
        raise NotImplementedError()
    def receive(self, save_dir: str) -> str | None:
        raise NotImplementedError()
    def discover(self) -> List[str]:
        raise NotImplementedError()

class BluetoothTransfer(TransferBase):
    """M.2 E-Key BT 传输"""
    def send(self, file_path: str, target: str) -> bool:
        raise NotImplementedError()
    def receive(self, save_dir: str) -> str | None:
        raise NotImplementedError()
    def discover(self) -> List[str]:
        raise NotImplementedError()
```

### 3.23 `ros2/nodes/` — ROS2 节点（预留，占位）

```python
# camera_node.py
class CameraNode(Node):
    """USB 摄像头 → sensor_msgs/Image topic"""
    def __init__(self) -> None: ...

# pose_node.py
class PoseNode(Node):
    """订阅 Image → PoseExtractor 推理 → 发布 PoseLandmarks topic"""
    def __init__(self) -> None: ...

# beat_node.py
class BeatNode(Node):
    """节拍检测 Service"""
    def __init__(self) -> None: ...

# alignment_node.py
class AlignmentNode(Node):
    """fastdtw 对齐 Service"""
    def __init__(self) -> None: ...

# scoring_node.py
class ScoringNode(Node):
    """异步评分 Action Server"""
    def __init__(self) -> None: ...
```

### 3.24 `ros2/interfaces/` — 自定义消息

```
PoseLandmarks.msg:
    Header header
    uint32 frame_id
    float32[99] landmarks        # 33 关键点 × 3D 坐标，展平
    float32[33] confidence

ScoreResult.msg:
    Header header
    float32 overall_score
    ScoreSegment[] segments
    WeakPoint[] weak_points

ScoreSegment.msg:
    uint32 id
    float32 start_time
    float32 end_time
    float32 score
    bool qualified

WeakPoint.msg:
    uint32 segment_id
    uint8 joint_index
    float32 deviation
    uint32 frame_id

CorrectionHint.msg:              # 预留
    string joint_name
    string suggestion
    float32 deviation
    string severity
```

---

## 四、数据流

### 4.1 离线评分流程（GUI 路径）

```
用户点击"开始评分"
    ↓
ScoreWorker._run()                              ← 后台线程
    ├─[1] PoseExtractor.extract(ref_video)      → List[PoseFrame]
    ├─[2] PoseExtractor.extract(user_video)     → List[PoseFrame]
    ├─[3] Scorer.score(ref, user)
    │    ├── dtw_constrained(mat)               → path, cost
    │    ├── _nonlinear_score()                 → per-frame scores
    │    └── seg_by_beats()                     → segment results
    ├─[4] locate_weak_points(segs, path, ref, user) → weak_points
    ├─[5] extract_clips_from_segments(segs)     → low-score clips
    └─[6] 可选: generate_correction()           → 纠错建议（预留）
    ↓
ScoreResultDialog.show(overall_score, segments, weak_points)
```

### 4.2 视频分割流程

```
用户点击"分割参考视频"
    ↓
SplitWorker._run()
    ├── detect_beats_from_audio()               → 优先 Madmom
    ├── detect_beats_from_motion()              → 退化光流
    └── calculate_segments_fixed()              → 兜底固定 BPM
    ↓
for each segment:
    extract_slow_segment()                      → ref_seg_NN_slow.mp4
    ↓
merge_videos()                                  → all_segments_merged.mp4
```

### 4.3 实时跟练流程（摄像头模式）

```
用户点击"开始跟练"
    ↓
LivePracticeWorker._run()
    ├─[1] CameraStream.start()                  → 后台读取帧
    ├─[2] 循环:
    │    ├── CameraStream.read_latest()         → 当前帧
    │    ├── PoseExtractor.extract(frame)       → PoseFrame
    │    ├── 与当前参考分段逐帧比对              → per-frame diff
    │    └── PracticeOverlay.update(diff)       → GUI 实时叠加偏差
    └─[4] 停止: 调用 Scorer.score()             → 完整评分报告
```

### 4.4 ROS2 节点数据流（预留）

```
CameraNode ──Image──→ PoseNode ──PoseLandmarks──→ AlignmentNode ──alignment──→ ScoringNode
                          │                                                            │
                          └──────────── BeatNode ──beat_times───────────────────────────┘
                                                                                        │
                                                                         ScoreResult + CorrectionHint
                                                                                        ↓
                                                                              GUI / TransferNode
```

---

## 五、扩展指南（给后续 AI Agent）

1. **新模块放对层**：
   - 算法/评分/纠正 → `core/`
   - 视频/音频处理 → `video/`
   - 摄像头采集 → `camera/`
   - UI → `gui/`
   - 硬件适配（NPU/GPIO）→ `platform/`
   - 网络传输 → `transfer/`
   - ROS2 包装 → `ros2/`

2. **依赖规则**：只能向下 import。`core/` 零业务依赖，`platform/` 和 `transfer/` 互不依赖，`ros2/` 只能 import `core/video/camera`。

3. **命名约定**：全部使用包内绝对路径 import
   ```python
   from dance_scoring.core.config import Config
   from dance_scoring.core.frame import PoseFrame
   ```

4. **不修改现有代码**：现有函数/类签名保持不变。新增功能通过新增独立函数/类实现，通过组合调用而非修改内部逻辑。

5. **占位模块**：标记 `NotImplementedError` 的模块（`inference.py`、`platform/`、`transfer/`、`ros2/`、`correction.py`）在填充实现前需先调研 DK-2500 硬件文档和对应 SDK。

6. **ROS2 准备**：`core/` 保持无状态/自封闭，每个数据类（`PoseFrame`、`WeakPoint`）可直接序列化为 ROS2 消息。ROS2 节点只做薄包装，内部 import 业务模块。

7. **评分参数基准**：所有评分相关参数以 `d3e03e9843ffce875ab03db868c43dfc387aacdb` 版本为准，不得随意修改。

8. **技术指标**（来自竞赛方案书 2.5.1，仅作测试验证参考，非代码常量）：

| 指标 | 目标值 |
|------|--------|
| 八拍分段精度 | ≥95%，误差 ≤100ms |
| 单帧推理延迟 | ≤50ms，帧率 ≥20fps |
| 评分与人工评估一致性 | ≥85% |
| 薄弱环节定位准确率 | ≥90% |
| 模型体积压缩 | ≥50%，算力降低 ≥40% |

---

## 六、当前文件 → 新文件 映射表

| 当前文件 | 新位置 | 说明 |
|---------|--------|------|
| `score_dance.py` (Config+常量) | `core/config.py` | |
| `score_dance.py` (PoseFrame) | `core/frame.py` | |
| `score_dance.py` (PoseExtractor+download_model) | `core/extractor.py` | |
| `score_dance.py` (_dtw_constrained) | `core/alignment.py` | 提取为模块级函数 |
| `score_dance.py` (Scorer+_nonlinear_score+_grade_overall) | `core/scorer.py` | |
| `score_dance.py` (_seg_by_beats+extract_clips_from_segments) | `core/segments.py` | |
| `score_dance.py` (main block) | `scripts/score.py` | |
| `split_8beats.py` (get_video_info) | `video/info.py` | |
| `split_8beats.py` (detect_beats_*) | `video/beat_detector.py` | |
| `split_8beats.py` (get_beat_segments+calculate_segments_fixed+extract_slow_segment) | `video/splitter.py` | |
| `split_8beats.py` (merge_videos) | `video/merger.py` | |
| `split_8beats.py` (main block) | `scripts/split.py` | |
| `gui_main.py` | `gui/app.py` | |
| `gui_components.py` | `gui/components.py` | |
| `gui_worker.py` | `gui/worker.py` | |
| `tempCodeRunnerFile.py` | 删除 | 临时文件 |
| `启动GUI.bat` | 保留根目录，修改路径指向新 src | |
| `pose_landmarker_lite.task` | 模型缓存 `~/.cache/dance_scoring/` | 不在源码树中 |
| `videos/` 目录 | 用户自行管理 | 不再作为固定目录 |
| `GUI需求规格说明书.md` | `docs/gui-requirements.md` | |
| `GUI界面及代码改动需求.txt` | 从根目录移入 `docs/` | |
| 嵌套目录 `dance-scoring-system-main/` | 取消嵌套，统一到 `intel/` 根下 | |
