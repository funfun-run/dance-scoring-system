# OpenVINO 加速集成设计文档

> 日期: 2026-06-03 | 状态: 已确认 | 目标平台: Intel DK-2500 (Core Ultra 5 225U) + Ubuntu 22.04

---

## 1. 背景与目标

### 1.1 当前状态

项目离线评分流水线已完整可用（视频分割 → 姿态提取 → DTW 对齐 → 分段打分 → 练习片段导出）。但以下模块待完成：

- `core/inference.py` — OpenVINO 推理引擎 (stub)
- `core/correction.py` — 纠正提示生成 (stub)
- `core/alignment.py` — 与 `dtw.py` 完全重复 (bug)
- 节拍检测方案：原竞赛方案书采用 Madmom，实际实现采用 librosa（更轻量，无需额外系统依赖，功能等价）
- `camera/usb.py` / `camera/stream.py` — 摄像头封装 (stub)
- `scripts/run_live.py` — 实时跟练 (stub)
- `platform/npu.py` — NPU 管理 (stub)

### 1.2 核心目标

**首要目标**：实现实时跟练功能，推理延迟 < 33ms/帧（30fps），通过 OpenVINO NPU 加速。

**次要目标**：后续可能嵌入 Qwen3.5 4B 模型优化纠正输出（架构已做预留，非硬依赖）。

### 1.3 方案选择

选择**方案 C：混合 IR + optimum-intel**。

- Pose 检测走纯 OpenVINO IR 路径以获得最低延迟
- LLM 纠正走 `optimum-intel` 路径（可插拔，架构解耦）
- 技术栈统一在 `ov.Core()` 管理

**方案 C 内置解耦**：Pose 推理管线是独立的，去掉 LLM 完全不影响核心功能。

---

## 2. 硬件规格 (DK-2500)

| 参数 | 规格 |
|------|------|
| CPU | Intel Core Ultra 5 225U (12C/14T, 15W TDP, 4.8GHz) |
| 内存 | 16GB DDR5 SO-DIMM (双通道 2×8GB) |
| 存储 | 128GB SSD (M.2 2280) |
| GPU | Intel Graphics (SoC 集成) |
| NPU | Intel AI Boost (Meteor Lake 集成, ~10 TOPS INT8) |
| 显示 | HDMI 2.1 + DP 1.4a + eDP/LVDS |
| USB | 4× USB 3.2 (1×Type-C + 3×Type-A) |
| GPIO | 40-pin JTAG (GPIO/PWM/UART/I2C/I2S/GSPI) |
| 以太网 | 4× GbE LAN (Intel i210) |
| OS | Ubuntu 22.04 |
| 尺寸 | 200 × 215mm |
| 功耗 | 整板最大 60W, 24V DC-in |

**关键结论**：NPU 共享 16GB DDR5 系统内存，无独立显存限制。Pose 模型 (~5MB) 完全不是问题；Qwen3.5 4B INT4 (~4-5GB) 内存容量可行，但 NPU 的 Transformer 推理效率待实测。

---

## 3. 架构设计

### 3.1 最终目标架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  CLI          │  │  GUI (ttkbootstrap)│ 实时跟练界面     │ │
│  │  score.py    │  │  app.py + live_view│                  │ │
│  │  split.py    │  │  components.py     │                  │ │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘ │
│         │                 │                     │            │
├─────────┼─────────────────┼─────────────────────┼────────────┤
│         │          应用编排层                     │            │
│  ┌──────┴─────────────────┴─────────────────────┴──────────┐ │
│  │               Pipeline / Runner                          │ │
│  │  OfflinePipeline (score.py)   LivePipeline (run_live.py) │ │
│  │  CorrectionProvider (RuleBased / LLM - 可插拔)            │ │
│  └──────────────────────────┬───────────────────────────────┘ │
│                             │                                  │
├─────────────────────────────┼──────────────────────────────────┤
│         │                   │                                  │
│  ┌──────┴──────┐  ┌─────────┴────────┐  ┌──────────────────┐  │
│  │  PoseEngine  │  │  AlignmentEngine │  │  CorrectionEngine│  │
│  │  (抽象接口)   │  │  (DTW / fastdtw) │  │  (Rule / LLM)    │  │
│  └──────┬──────┘  └──────────────────┘  └──────────────────┘  │
│         │                                                      │
│  ┌──────┴──────────────────────────────────────┐              │
│  │          PoseEngine 实现                      │              │
│  │  ┌──────────────────┐  ┌──────────────────┐ │              │
│  │  │ MediaPipeBackend  │  │ OpenVINOBackend   │ │  ← 可切换   │
│  │  │ (现有实现，无依赖) │  │ (IR 推理，NPU加速) │ │              │
│  │  └──────────────────┘  └──────────────────┘ │              │
│  └─────────────────────────────────────────────┘              │
│                                                                │
├──────────────────────────────────────────────────────────────┤
│                      硬件抽象层                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Camera   │  │ NPU      │  │ GPIO     │  │ Transfer      │  │
│  │ (USB/RTSP)│  │ Manager  │  │ Manager  │  │ (WiFi/BLE)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计原则

- **Phase 1 全部用现有 MediaPipe 跑通功能闭环**
- **Phase 2 在不改业务逻辑的前提下替换推理后端**
- **GUI 层完全不感知底层用的是 MediaPipe 还是 OpenVINO**
- **LLM 纠正模块通过抽象接口解耦，可随时插拔**

---

## 4. 分阶段计划

| 阶段 | 内容 | 优先级 | 依赖 |
|------|------|--------|------|
| Phase 1a | 修复 alignment + 实现 correction 规则引擎 | 🔴 高 | 无 |
| Phase 1b | 实现 camera (usb/stream/base) | 🔴 高 | 无 |
| Phase 1c | 实现 run_live.py 实时跟练 | 🔴 高 | 1a, 1b |
| Phase 1d | GUI ttkbootstrap 改造 + 实时界面 | 🟡 中 | 1c |
| Phase 2a | OpenVINO 模型转换流水线 | 🔴 高 | 无 |
| Phase 2b | 实现 inference.py + npu.py | 🔴 高 | 2a |
| Phase 2c | PoseEngine 抽象层 + 后端切换 | 🔴 高 | 1a-1c, 2b |
| Phase 2d | 性能基准测试 | 🟡 中 | 2c |
| Phase 3 | Qwen3.5 / GPIO / Transfer / ROS2 | 🟢 低 | 按需 |

---

## 5. 各阶段详细设计

### 5.1 Phase 1a: alignment 修复 + correction 规则引擎

#### 5.1.1 alignment.py 修复

**问题**：`core/alignment.py` 与 `core/dtw.py` MD5 完全一致，无独特功能。

**方案**：替换为基于 `fastdtw` 库的快速对齐实现。

```
alignment.py (重新设计)
├── fastdtw_alignment(seq1, seq2, radius)
│   - 基于 fastdtw Python 库
│   - 返回 (distance, path, aligned_seq1, aligned_seq2)
│   - radius: 搜索窗口半径，默认值根据序列长度自适应
│
└── 与 dtw.py 的区别：
    - dtw.py: Sakoe-Chiba 约束窗口 + 完整距离矩阵 → O(N*W)，精确
    - alignment.py: fastdtw 近似算法 → O(N)，快速，适合实时场景
```

**scorer.py 调用方式**：
```python
scorer = Scorer(config, alignment_method="dtw")      # 默认，精确（离线）
scorer = Scorer(config, alignment_method="fastdtw")  # 快速（实时模式）
```

#### 5.1.2 correction.py 实现

**目标**：根据每段的打分结果，自动生成中文纠正建议。

**流程**：聚合偏差 → 排序 → 取 Top-3 → 查关节名映射 → 填入模板。

**关节映射表** (33 个 MediaPipe 关键点 → 中文名)：
```python
JOINT_NAMES_CN = {
    0:  "鼻尖", 1: "左眼内角", 2: "左眼", 3: "左眼外角",
    4:  "右眼内角", 5: "右眼", 6: "右眼外角",
    7:  "左耳", 8: "右耳",
    9:  "嘴角左", 10: "嘴角右",
    11: "左肩", 12: "右肩",
    13: "左肘", 14: "右肘",
    15: "左腕", 16: "右腕",
    17: "左小指", 18: "右小指", 19: "左食指", 20: "右食指",
    21: "左拇指", 22: "右拇指",
    23: "左髋", 24: "右髋",
    25: "左膝", 26: "右膝",
    27: "左踝", 28: "右踝",
    29: "左脚跟", 30: "右脚跟",
    31: "左脚尖", 32: "右脚尖",
}
```

**纠正模板**：

| 偏差方向 | 模板 |
|----------|------|
| 肘关节角度偏大 | `{关节名}角度偏差{值}°，请弯曲{关节名}` |
| 肘关节角度偏小 | `{关节名}角度偏差{值}°，请伸直{关节名}` |
| 膝关节角度偏大 | `{关节名}弯曲不足{值}°，请降低重心` |
| 膝关节角度偏小 | `{关节名}过度弯曲{值}°，请站直一些` |

**接口**：
```python
def generate_correction(
    segment_scores: List[SegmentScore],
    top_n: int = 3,
    threshold_deg: float = 10.0,
) -> Dict[int, str]:  # {segment_id: correction_text}
```

---

### 5.2 Phase 1b: 摄像头模块

#### 5.2.1 camera/base.py 完善

`list_devices()` 从硬编码空列表改为实际 OpenCV 设备枚举：
```python
@staticmethod
def list_devices(max_index: int = 8) -> List[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
```

抽象接口：`open() -> bool`, `read() -> Optional[np.ndarray]`, `close() -> None`, `is_opened() -> bool`。

#### 5.2.2 camera/usb.py 实现

```python
class UsbCamera(CameraBase):
    def __init__(self, device_id=0, resolution=(640, 480), fps=30):
        ...
    def open(self) -> bool:
        # cv2.VideoCapture(device_id) + 设置分辨率/帧率
    def read(self) -> Optional[np.ndarray]:
        # 返回 RGB 格式 (与 MediaPipe mp.ImageFormat.SRGB 一致)
    def close(self) / is_opened(self):
        # 标准生命周期管理
```

#### 5.2.3 camera/stream.py 实现

```python
class NetworkStream(CameraBase):
    """支持 RTSP/HTTP 视频流，通过 cv2.VideoCapture(url) 通用能力"""
    def __init__(self, url: str): ...
    # read/close/is_opened 与 UsbCamera 逻辑复用
```

---

### 5.3 Phase 1c: 实时跟练 (run_live.py)

#### 5.3.1 主循环流程

```
① 初始化
   ├── 摄像头打开 (UsbCamera)
   ├── PoseExtractor 初始化 (MediaPipe 路径)
   ├── 加载参考视频 → 预提取参考姿态序列到内存
   └── 创建 DTW 滑动窗口对齐器

② 每帧循环 (~30fps target)
   ├── camera.read() → 当前帧
   ├── extractor 提取 PoseFrame
   ├── 追加到滑动窗口
   ├── 窗口满 → DTW 对齐 → 打分 → 生成纠正建议
   └── 输出: {得分, 错误关节, 建议文本}

③ 退出 → camera.close()
```

#### 5.3.2 滑动窗口对齐策略

```
参考视频: |──── 第1段 ────|──── 第2段 ────|──── 第3段 ────|...
                     ↑ 当前参考段
用户实时:  ───────────[──── 滑动窗口 ────]──→ 时间
                        ↑ 最新帧
每次 DTW: 参考段 vs 用户窗口 → path → 只取窗口后半段结果 (避免边界抖动)
```

#### 5.3.3 模块结构

```python
# run_live.py (单文件，Phase 1 不过度抽象)
class LiveConfig:           # 实时模式配置
class SlidingWindowBuffer:  # 滑动窗口管理
class LiveScorer:           # 核心编排 (camera + extractor + dtw + scorer)
def main():                 # CLI 入口 + 文本输出
```

#### 5.3.4 输出示例

```
📷 摄像头已打开 | 30fps | 640x480
📐 参考姿态已加载: 120帧 (4段)
▶️ 开始跟练...

[段1] 得分:72.3 ✓ | 右肘角度偏差:11.2° — 请抬高右前臂
[段2] 得分:45.2 ✗ | 左膝弯曲不足:15.3° — 请降低重心

🏆 总评: 良好 | 通过率:3/4
📊 薄弱部位: 左膝, 右肘
```

---

### 5.4 Phase 1d: GUI 现代化 + 实时界面

#### 5.4.1 技术选型

选择 **ttkbootstrap** 主题库。理由：改动最小（import 替换），视觉效果接近现代 Web 风格，与现有线程模型兼容。

#### 5.4.2 色板设计 (theme.py)

```python
COLORS = {
    "bg_primary":    "#1a1a2e",   # 深蓝黑背景
    "bg_card":       "#16213e",   # 卡片面板
    "accent":        "#e94560",   # 强调红 — 失败/错误
    "accent_green":  "#0f9b58",   # 通过绿
    "accent_yellow": "#f5a623",   # 警告黄
    "text_primary":  "#eaeaea",   # 主文字
    "text_secondary":"#a0a0b0",   # 次要文字
}
```

#### 5.4.3 新增 gui/live_view.py

```
LivePanel 布局:
┌──────────────────────────────────────────┐
│  🎬 实时跟练                     [_][□][×]│
├────────────────────┬─────────────────────┤
│                    │  📊 当前: 第 2 / 4 段 │
│   摄像头画面        │  🎯 得分:   72.3 分   │
│   + 骨骼叠加        │  ████████████░░░ 75%  │
│   (PoseOverlay)    │                     │
│                    │  ✏️ 纠正建议:         │
│                    │  请抬高右前臂         │
│                    │                     │
│                    │  💪 薄弱部位:        │
│                    │  🔴 右肘  🟡 左膝     │
├────────────────────┴─────────────────────┤
│ [▶开始] [⏸暂停] [⏹停止] [🔄循环] [📁选参考]│
│ 倍速: [0.5x] [0.8x] [1.0x]              │
└──────────────────────────────────────────┘
```

功能说明：
- **倍速调节**: 下拉或按钮组，控制练习片段播放速度（0.5x / 0.8x / 1.0x），对应 segments.py 的慢动作逻辑
- **循环练习**: 开关按钮，开启后当前段练习结束后自动重新开始，方便反复练习薄弱段
- `PoseOverlay`: 在视频帧上绘制 33 关节点 + 骨骼连线，偏离关节红色高亮，正常关节绿色
- 数据流: `LiveScorer(worker线程)` → 回调 → `LivePanel._on_update(data)`
- GUI 完全不感知底层推理后端

#### 5.4.4 现有组件改造

| 组件 | 改造项 |
|------|--------|
| app.py | 导入 ttkbootstrap, 主题设置, 新增"实时跟练"入口按钮 |
| components.py | 控件使用主题色, 得分卡片化显示, 段列表加状态图标 |
| worker.py | 不变 |

---

### 5.5 Phase 2a: 模型转换流水线

#### 5.5.1 转换流程

```
MediaPipe .task (Flatbuffer zip)
    ↓ 解包
  pose_landmarker.tflite (原始大小)
    ↓ ovc 转换 + 量化
  pose_landmarker.xml + pose_landmarker.bin (OpenVINO IR)
    ↓ 体积对比
  记录压缩率 → meta.json
```

支持三种精度等级：

| 精度 | 说明 | NPU 速度 | 体积压缩 | 适用场景 |
|------|------|---------|---------|----------|
| FP16 | 半精度浮点 | **最快** | ~50% | 默认选项，DK-2500 NPU 原生支持，精度损失可忽略 |
| FP32 | 原始精度 | 慢 | ~0% | 精度验证 baseline |
| INT8 | 8-bit 整数量化 | 中等 | ~75% | 追求极致压缩时可选，需校准数据 |

> **选型理由**：Intel Core Ultra 5 225U (Meteor Lake) 的 NPU 对 FP16 有原生硬件加速，推理速度最快。FP16 同时满足竞赛方案的 50% 体积压缩要求。INT8 虽然压缩率更高，但量化/反量化引入额外开销，在 NPU 上实际推理速度通常不及 FP16，且需校准数据，故不作为默认选项。

#### 5.5.2 脚本 scripts/convert_model.py

```bash
python scripts/convert_model.py                    # 默认 FP16（DK-2500 NPU 最快）
python scripts/convert_model.py --precision FP32   # 无压缩（精度验证用）
python scripts/convert_model.py --precision INT8   # 极致压缩（需校准）
```

**职责**：
1. 检查 `~/.cache/dance_scoring/` 下模型, 无则触发下载
2. 解包 .task 文件 (zipfile)
3. 调用 `openvino.convert_model()` 转换，默认 FP16（`compress_to_fp16=True`）
4. 可选 INT8 量化（需 nncf 校准），可选 FP32（无压缩，精度 baseline）
5. 生成 `meta.json` (输入输出规格 + 源模型 hash + 压缩率)
6. 输出原始 TFLite 大小 vs IR 大小的对比

#### 5.5.3 产物

```
src/dance_scoring/models/
├── pose_landmarker.xml      # 模型图
├── pose_landmarker.bin      # 权重（FP16 下体积约为原始的 50%）
└── pose_landmarker_meta.json # {input, outputs, source_hash, compression_ratio, precision}
```

---

### 5.6 Phase 2b: 推理引擎 + NPU 管理

#### 5.6.1 core/inference.py 重写

```python
@dataclass
class PoseInferenceResult:
    kp3d: np.ndarray        # (33, 3) 世界坐标
    kp2d: np.ndarray        # (33, 2) 像素坐标
    visibility: np.ndarray  # (33,) 0~1
    presence: float         # 是否有人

class PoseInferenceEngine:
    def __init__(self, model_dir: Path, device: str = "NPU"):
        # ov.Core().compile_model(ir_xml, device)
    def infer(self, rgb_image: np.ndarray) -> PoseInferenceResult:
        # 预处理 + 推理 + 后处理
    def warmup(self, rounds: int = 3):
        # 用空图预热 NPU，避免首帧尖峰
```

**预处理**：resize → 归一化 [-1,1] → NCHW 布局转换。

#### 5.6.2 platform/npu.py 重写

```python
class NPUManager:
    @staticmethod
    def available() -> bool:
        # ov.Core().available_devices 包含 "NPU"
    @staticmethod
    def best_device() -> str:
        # 优先级: NPU > GPU > CPU
    @staticmethod
    def device_info(device: str = "NPU") -> dict:
        # 返回设备属性
```

---

### 5.7 Phase 2c: PoseEngine 抽象层

#### 5.7.1 core/engine.py

```python
class PoseEngine(Protocol):
    def extract(self, path: str) -> List[PoseFrame]: ...
    def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]: ...
    @property
    def backend_name(self) -> str: ...

def create_pose_engine(backend: str = "auto", cfg: Config = Config()) -> PoseEngine:
    """
    "auto"      → 自动选择 (NPU > GPU > CPU → MediaPipe fallback)
    "mediapipe" → 强制 MediaPipe
    "openvino"  → 强制 OpenVINO IR
    """
```

#### 5.7.2 与现有代码的关系

- `PoseExtractor` (extractor.py) 保持原样，作为 `MediaPipeEngine` 内部实现
- `_interpolate()` 提取为独立函数，两个后端共享
- `scorer.py` 增加 `alignment_method` 参数 (dtw/fastdtw)

---

### 5.8 Phase 2d: 性能验证

#### 5.8.1 scripts/benchmark.py

```
输出:
=================================================
  姿态推理性能对比
=================================================
测试视频: reference.mp4
测试帧数: 100
设备: Intel Core Ultra 5 225U

指标              MediaPipe(CPU)    OpenVINO(NPU)
──────────────────────────────────────────────
平均推理(ms)        18.2            5.1
P99 延迟(ms)        22.4            6.8
首帧延迟(ms)        45.1            12.3
帧处理吞吐(fps)     54.9            196.0
CPU 占用(%)         45%             12%

模型体积对比:
  原始 TFLite:      5.6 MB
  IR (FP16):        2.8 MB (压缩 50%)
=================================================
结论: OpenVINO 延迟降低 72.0%, 吞吐提升 3.6x
      模型体积压缩 50%, 满足竞赛指标(≥50%)
      精度: FP16（DK-2500 NPU 原生加速，速度最快）
```

#### 5.8.2 竞赛指标对照

| 指标 | 竞赛要求 | 目标值 |
|------|---------|--------|
| 单帧推理延迟 | ≤50ms | <33ms (更严格) |
| 实时帧率 | ≥20fps | 30fps (更严格) |
| 八拍分段精度 | ≥95% | ≥95% |
| 节拍检测误差 | ≤100ms | ≤100ms |
| 模型体积压缩 | ≥50% | ≥50% (FP16 预期 ~50%) |
| 算力消耗降低 | ≥40% | ≥40% |
| 评分一致性 | ≥85% | ≥85% |
| 薄弱环节定位 | ≥90% | ≥90% |

#### 5.8.3 正确性验证

两个后端提取同一视频，逐帧对比关键点坐标误差。期望平均误差 < 1mm。

---

### 5.9 项目结构优化

```
src/dance_scoring/
├── core/
│   ├── config.py          (不变)
│   ├── frame.py           (不变)
│   ├── extractor.py       (不变, MediaPipeEngine 内部实现)
│   ├── engine.py          ← 新增: 推理引擎抽象层 + 工厂函数
│   ├── dtw.py             (不变)
│   ├── alignment.py       ← 重写: fastdtw 实现
│   ├── inference.py       ← 重写: OpenVINO 推理引擎
│   ├── scorer.py          ← 小改: 支持 alignment_method 参数
│   ├── segments.py        (不变)
│   ├── correction.py      ← 重写: 规则引擎
│   └── __init__.py
├── video/                 (不变)
├── camera/
│   ├── base.py            ← 重写: list_devices 实际枚举
│   ├── usb.py             ← 重写: cv2.VideoCapture 封装
│   └── stream.py          ← 重写: RTSP/HTTP 流
├── gui/
│   ├── theme.py           ← 新增: ttkbootstrap 主题配置
│   ├── app.py             ← 改造: import ttkbootstrap + 实时入口
│   ├── components.py      ← 改造: 主题色控件
│   ├── live_view.py       ← 新增: 实时跟练界面
│   └── worker.py          (不变)
├── platform/
│   ├── npu.py             ← 重写: 设备探测 + 信息
│   └── gpio.py            (保留 stub, 低优)
├── transfer/              (保留 stub)
├── ros2/                  (保留 stub)
└── models/                ← 新增: IR 模型存放
    ├── pose_landmarker.xml
    └── pose_landmarker.bin

scripts/
├── score.py               ← 小改: 增加 --backend 参数
├── split.py               (不变)
├── run_live.py            ← 重写: 实时跟练 CLI
├── convert_model.py       ← 新增: 模型转换
└── benchmark.py           ← 新增: 性能对比
```

**依赖更新 (requirements.txt)**：
```
fastdtw        # Phase 1a: 快速 DTW 对齐
ttkbootstrap   # Phase 1d: GUI 现代主题
openvino       # Phase 2a: OpenVINO 运行时
```

---

## 6. 数据流总览

### 6.1 离线模式 (Phase 1a 之后)

```
参考视频.mp4 + 用户视频.mp4
    ↓
PoseEngine.extract() × 2
    ↓
DTW / fastdtw 对齐
    ↓
逐帧打分 → 分段聚合
    ↓
correction.py → 中文纠正建议
    ↓
输出: 得分报告 + 练习片段
```

### 6.2 实时模式 (Phase 1c 之后)

```
摄像头 (UsbCamera/NetworkStream)
    ↓ 每帧
PoseEngine.extract_frame()
    ↓ 窗口满
DTW 对齐 → 打分 → 纠正建议
    ↓ 回调
GUI (live_view) / 终端文本
```

### 6.3 加速模式 (Phase 2c 之后)

```
实时模式 100% 复用
    ↓ 仅切换
create_pose_engine("openvino")
    ↓
PoseInferenceEngine → ov.Core(NPU) 推理
```

---

## 7. Phase 3 (远期规划)

| 模块 | 方案 | 触发条件 |
|------|------|----------|
| Qwen3.5 4B | optimum-intel 集成, OVModelForCausalLM | Phase 2 完成 + 有需求 |
| GPIO | platform/gpio.py 实现 LED/按键 | DK-2500 实机部署 |
| Transfer | WiFi Direct / BLE 文件传输 | 手机端协同需求 |
| ROS2 | 全套 nodes/launch/interfaces | 机器人集成需求 |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| MediaPipe TFLite → IR 转换算子不兼容 | 预先验证转换; 保留 MediaPipe 作为 fallback |
| NPU driver 在 Ubuntu 22.04 上不可用 | `best_device()` 自动回退 GPU → CPU |
| fastdtw 精度不如标准 DTW | 离线模式默认用 DTW; 实时模式用 fastdtw + 可配置 |
| ttkbootstrap 与现有控件不兼容 | 逐文件改造, 每步可回退 |
| Qwen3.5 4B NPU 上性能不足 | 架构解耦, 通过 `CorrectionProvider` 接口回退规则引擎 |

---

## 9. 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-03 | 初始版本, 方案确认 |
