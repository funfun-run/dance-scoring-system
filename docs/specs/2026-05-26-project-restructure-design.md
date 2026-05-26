# 舞蹈评分系统 — 项目骨架重构设计规格

> **目标**：将当前混乱的单文件堆叠结构重组为分层的 Python 包，为后续实时舞蹈识别、无线传输、DK-2500 硬件适配和 ROS2 节点化留好扩展点。
>
> **读者**：AI agent 和人类开发者。本文档是后续所有功能开发的权威结构参考。

---

## 一、目标文件树

```
intel/                              # 项目根
├── pyproject.toml                  # pip install -e . 可编辑安装
├── requirements.txt                # 精确 pin 依赖版本
├── README.md
├── .gitignore
├── .claude/                        # Claude Code 配置（不动）
├── .agents/                        # Agent skills（不动）
├── scripts/                        # CLI 入口
│   ├── score.py                    # CLI 评分，替代 score_dance.py main
│   └── split.py                    # CLI 分割，替代 split_8beats.py main
├── src/dance_scoring/              # 主包
│   ├── __init__.py
│   ├── core/                       # 核心算法层 — 零外部依赖
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── frame.py
│   │   ├── extractor.py
│   │   ├── dtw.py
│   │   ├── scorer.py
│   │   └── segments.py
│   ├── video/                      # 视频处理 — 仅依赖 core.config
│   │   ├── __init__.py
│   │   ├── info.py
│   │   ├── beat_detector.py
│   │   ├── splitter.py
│   │   └── merger.py
│   ├── gui/                        # GUI 层 — 依赖 core + video
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── components.py
│   │   └── worker.py
│   ├── camera/                     # 摄像头抽象（一期占位）
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── usb.py
│   │   └── stream.py
│   ├── transfer/                   # 无线传输（一期占位）
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── wifi.py
│   │   └── bluetooth.py
│   └── hardware/                   # DK-2500 适配（一期占位）
│       ├── __init__.py
│       ├── gpio.py
│       └── display.py
├── assets/                         # 静态资源
│   ├── videos/
│   │   ├── reference.mp4
│   │   └── user.mp4
│   └── model/
│       └── pose_landmarker_lite.task
├── output/                         # 运行时产出 .gitignore
│   ├── segments/
│   └── low_score_clips/
├── docs/
│   ├── gui-requirements.md
│   ├── hardware-spec.md
│   └── specs/
│       └── 2026-05-26-project-restructure-design.md
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_dtw.py
    └── test_extractor.py
```

---

## 二、依赖层次图

```
  scripts/        gui/
     │              │
     └──┬───────────┘
        │
   ┌────┴────┐
   │  core   │  ← 零业务依赖，仅 numpy/scipy/mediapipe
   └─────────┘
        │
   ┌────┴──────────┐
   │    video      │  ← 仅依赖 core（config 常量 + frame 可能）
   └───────────────┘

  camera/   transfer/   hardware/
     ↑          ↑          ↑
     └──────────┸──────────┘
         只能被 scripts 或 gui 调用
         互相不依赖，各自独立
```

### Import 白名单

| 从 | 可以 import | 禁止 |
|----|------------|------|
| `core` | core 内部, numpy, scipy, mediapipe | gui, video, camera, transfer, hardware, tkinter |
| `video` | core.config, video 内部, cv2, numpy, librosa | gui, camera, tkinter |
| `gui` | core, video, camera(未来), tkinter | transfer, hardware |
| `camera` | cv2, numpy, abc | gui, transfer, hardware, core |
| `transfer` | stdlib only | 所有业务模块 |
| `hardware` | stdlib only | 所有业务模块 |
| `scripts` | 所有模块 | — |

---

## 三、各模块接口清单

### 3.1 `core/config.py` — 全局配置

```python
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

ANGLE_JOINTS: List[Tuple[int, int, int]]    # 26 个关节三元组
ANGLE_WEIGHTS: np.ndarray                    # 26 个权重

Z_AXIS_WEIGHT: float = 0.3
DTW_WINDOW_RATIO: float = 0.1
SCORE_TOLERANCE: float = 3.0
SCORE_PENALTY_SMALL: float = 1.8
SCORE_PENALTY_LARGE: float = 3.0
SCORE_PENALTY_THRESHOLD: float = 15.0

MODEL_URL: str
MODEL_PATH: str = "pose_landmarker_lite.task"
```

### 3.2 `core/frame.py` — 单帧姿态数据

```python
@dataclass
class PoseFrame:
    fid: int                    # 帧序号
    kp3d: np.ndarray            # shape (33, 3), world landmarks
    conf: np.ndarray            # shape (33,), 置信度
    angles: np.ndarray = None   # shape (26,), 关节角度（自动计算）
    vec: np.ndarray = None      # shape (66+26,), 特征向量（自动拼装）
```

### 3.3 `core/extractor.py` — 姿态提取器

```python
def download_model() -> None:
    """下载 MediaPipe 模型文件到 MODEL_PATH"""

class PoseExtractor:
    def __init__(self, cfg: Config) -> None: ...
    def extract(self, path: str, progress_callback: Callable[[int], None] | None = None) -> List[PoseFrame]:
        """
        从视频文件提取所有帧的姿态。
        progress_callback: 参数为 0-100 的进度百分比
        """
    def _interpolate(self, poses: List[PoseFrame]) -> List[PoseFrame]:
        """低置信度关键点插值修复"""
```

### 3.4 `core/dtw.py` — 动态时间规整

```python
def dtw_constrained(mat: np.ndarray, window: int) -> Tuple[List[Tuple[int,int]], float]:
    """
    约束窗口 DTW 算法。
    mat: (n_ref, n_user) 距离矩阵
    window: 搜索窗口大小
    返回: (对齐路径, 总代价)
    """
```

### 3.5 `core/scorer.py` — 评分引擎

```python
class Scorer:
    def __init__(self, cfg: Config, bpm: int = DEFAULT_BPM) -> None: ...

    def score(
        self,
        ref: List[PoseFrame],
        user: List[PoseFrame],
        progress_callback: Callable[[int, str], None] | None = None
    ) -> Tuple[float, List[dict], List[dict], List[Tuple[int,int]]]:
        """
        返回: (总评分数, 分段结果列表, 低分段落列表, DTW 对齐路径)
        progress_callback: (百分比, 阶段描述)
        """

    def _grade_overall(self, fs: List[float], segs: List[dict]) -> float: ...
    def _nonlinear_score(self, avg_diff: float) -> float: ...
```

### 3.6 `core/segments.py` — 分段与练习片提取

```python
def seg_by_beats(
    ref: List[PoseFrame],
    path: List[Tuple[int,int]],
    fs: List[float],
    target_fps: int,
    bpm: int
) -> List[dict]:
    """按 BPM 将评分结果分段，返回 [{'id','start_time','end_time','score','qualified'}, ...]"""

def extract_clips_from_segments(
    segs: List[dict],
    segments_dir: str = "output/segments",
    out_dir: str = "output/low_score_clips",
    cfg: Config = None
) -> List[str]:
    """复制/提取低分段的慢动作练习视频，返回输出文件路径列表"""
```

### 3.7 `video/info.py` — 视频信息

```python
def get_video_info(video_path: str) -> Tuple[float, int, float, int, int]:
    """返回: (fps, frame_count, duration_seconds, width, height)"""
```

### 3.8 `video/beat_detector.py` — 节拍检测

```python
def detect_beats_from_audio(video_path: str) -> Tuple[List[float], float] | None:
    """音频节拍检测，返回 (beat_times_seconds, tempo_bpm) 或 None"""

def detect_beats_from_motion(video_path: str) -> Tuple[List[float], float] | None:
    """运动光流节拍检测，返回 (beat_times_seconds, estimated_bpm) 或 None"""
```

### 3.9 `video/splitter.py` — 视频分段

```python
def get_beat_segments(
    beat_times: List[float],
    duration: float,
    beats_per_seg: int = 8
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
    """提取单个慢动作片段（0.8x）"""
```

### 3.10 `video/merger.py` — 合并

```python
def merge_videos(video_list: List[str], output_path: str) -> None:
    """将多个视频片段合并为一个文件"""
```

### 3.11 `gui/app.py` — 主窗口

```python
class MainApp:
    """Tkinter 主窗口。职责：布局组装、事件绑定、协调各组件"""
    def __init__(self) -> None: ...
    def run(self) -> None: ...
```

### 3.12 `gui/components.py` — 可复用组件

```python
class VideoPreview(tk.Frame):
    """视频预览面板：缩略图 + 信息 + 播放/停止"""
    def load(self, path: str) -> None: ...
    def stop(self) -> None: ...

class ProgressDialog(tk.Toplevel):
    """进度条弹窗"""
    def update_progress(self, percent: int, msg: str = "") -> None: ...

class ScoreResultDialog(tk.Toplevel):
    """评分结果表格窗口"""

class SegmentListDialog(tk.Toplevel):
    """分割结果列表 + 逐段播放窗口"""
```

### 3.13 `gui/worker.py` — 后台线程

```python
class Worker:
    """后台任务基类。start() 启动线程，on_progress/on_done 回调"""
    def start(self) -> None: ...
    def cancel(self) -> None: ...
    def is_running(self) -> bool: ...

class SplitWorker(Worker): ...
class ScoreWorker(Worker): ...
```

### 3.14 `camera/base.py` — 摄像头抽象（占位）

```python
from abc import ABC, abstractmethod

class CameraBase(ABC):
    @abstractmethod
    def open(self, device_id: int = 0) -> bool: ...
    @abstractmethod
    def read(self) -> np.ndarray | None: ...
    @abstractmethod
    def close(self) -> None: ...
    @staticmethod
    def list_devices() -> List[str]: ...
```

### 3.15 `transfer/base.py` — 传输抽象（占位）

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

---

## 四、数据流

### 评分流程（GUI 路径）

```
用户点击"开始评分"
    ↓
MainApp._do_score()
    ↓
ScoreWorker._run()                    ← 后台线程
    ├─[step 1] PoseExtractor.extract(ref_video)  → List[PoseFrame]
    ├─[step 2] PoseExtractor.extract(user_video) → List[PoseFrame]
    ├─[step 3] Scorer.score(ref, user)
    │    ├── dtw_constrained(mat)     → path, cost
    │    ├── _nonlinear_score()       → per-frame scores
    │    └── seg_by_beats()           → segment results
    └─[step 4] extract_clips_from_segments(segs) → low-score clips
    ↓
progress.destroy() + ScoreResultDialog.show(result)
```

### 分割流程

```
用户点击"分割参考视频"
    ↓
SplitWorker._run()
    ├── detect_beats_from_audio()     → 优先
    ├── detect_beats_from_motion()    → 退化
    └── calculate_segments_fixed()    → 兜底
    ↓
for each segment:
    extract_slow_segment()            → ref_seg_NN_slow.mp4
    ↓
merge_videos()                        → all_segments_merged.mp4
```

---

## 五、给 AI Agent 的扩展指南

### 添加新功能模块时，遵循以下规则：

1. **新模块放对位置**：
   - 算法类 → `core/`
   - 视频/音频处理 → `video/`
   - UI 相关 → `gui/`
   - 新硬件外设 → `hardware/`
   - 跨设备通信 → `transfer/`
   - 命令行工具 → `scripts/`

2. **遵守依赖层次**：只能向下依赖，不能向上。新模块先写 ABC 基类，再写实现。

3. **遵循命名约定**：所有 import 使用包内路径
   ```python
   from dance_scoring.core.config import Config
   from dance_scoring.core.frame import PoseFrame
   ```
   不使用相对 import。

4. **核心类保持不变**：`PoseFrame`、`Config`、`PoseExtractor`、`Scorer` 的公开签名尽量稳定。如果必须改，同步更新本文档。

5. **ROS2 准备**：每个 `core/` 模块保持无状态（或状态封装在类内），方便包装为 ROS2 node。不要引入全局可变状态。

6. **文件行数**：每个文件不超过 300 行。超过则拆。

---

## 六、当前文件 → 新文件 映射表

| 当前文件 | 新位置 | 说明 |
|---------|--------|------|
| `score_dance.py` (Config+常量) | `core/config.py` | |
| `score_dance.py` (PoseFrame) | `core/frame.py` | |
| `score_dance.py` (PoseExtractor+download_model) | `core/extractor.py` | |
| `score_dance.py` (_dtw_constrained) | `core/dtw.py` | 提取为模块级函数 |
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
| `pose_landmarker_lite.task` | `assets/model/` | |
| `videos/reference.mp4` | `assets/videos/reference.mp4` | |
| `videos/user.mp4` | `assets/videos/user.mp4` | |
| `GUI需求规格说明书.md` | `docs/gui-requirements.md` | |
| `GUI界面及代码改动需求.txt` | `docs/gui-requirements.txt` | 从根目录移入 |
| 根目录 `.venv` | 保留 | 开发使用 |
| `dance-scoring-system-main/.venv` | 删除 | 冗余 |
| 双层嵌套目录 | 取消嵌套 | 统一到 `intel/` 根下 |
