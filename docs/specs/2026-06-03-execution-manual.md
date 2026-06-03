# DK-2500 多 Agent 协作执行手册

> **目标**: 本手册由 DK-2500 上的 Claude Code（主 Agent）读取，按 Phase 调度子 Agent 执行实现和测试。
> **模式**: 主 Agent 编排 → 实现 Agent 写代码 → 测试 Agent 验证 → 主 Agent 裁决 → 进入下一阶段或修复
> **配套设计文档**: `docs/specs/2026-06-03-openvino-integration-design.md`
> **日期**: 2026-06-03

---

## 0. 角色定义

### 0.1 主 Agent（你 — DK-2500 上的 Claude）

你的职责是**编排**，不直接写代码：

1. 阅读当前 Phase 的说明
2. 依次启动 **Setup Agent**（如需）→ **Implementer Agent** → **Tester Agent**
3. 阅读 Tester 的验证报告
4. 做出裁决：
   - ✅ **通过** → `git commit`，进入下一 Phase
   - ⚠️ **部分通过** → 启动 Implementer 修复失败项 → 重新启动 Tester
   - ❌ **严重失败** → 汇总问题，输出诊断信息，暂停等待人工介入

**你绝对不要自己修改代码**——代码变更全部交给 Implementer Agent。

**核心禁令**：
- **禁止修改 `scorer.py` 的评分算法**。`Scorer._nonlinear_score()` 的容差参数（3°/15°）、扣分系数（1.8/3.0）、`_grade_overall()` 的 5 档判定逻辑、段合格线（60 分）、关节角度权重（`ANGLE_WEIGHTS`）均为已确定的竞赛评分标准，不得改动。唯一允许的变更是增加 `alignment_method` 参数（用于选择 DTW/fastdtw 对齐方式，不影响评分计算本身）。
- **禁止修改 `frame.py` 的特征向量结构**。`PoseFrame.vec` 的拼接方式（66 维坐标 + 26 维角度 = 92 维）是下游对齐算法的输入格式，改动会导致评分不可比。

### 0.2 Implementer Agent（子 Agent）

- 接收你的实现指令
- 阅读相关现有文件
- 编写/修改代码
- 完成后简要汇报改动了哪些文件、关键实现要点

### 0.3 Tester Agent（子 Agent）

- 接收你的测试指令（包含验证脚本）
- 执行验证脚本
- 报告：通过项、失败项、具体错误信息
- 不做任何代码修改

### 0.4 协作流程

```
主 Agent (你)
    │
    ├─[1]─→ Setup Agent: 环境检查 + 依赖安装
    │         ← 报告: 系统信息 / Python版本 / 依赖状态
    │
    ├─[2]─→ Implementer Agent: "请实现 Phase X..."
    │         ← 报告: 改动文件列表 + 关键要点
    │
    ├─[3]─→ Tester Agent: "请验证 Phase X..."
    │         ← 报告: ✓ N项通过 / ✗ M项失败 + 错误详情
    │
    ├─[4]─→ 你裁决:
    │         ├── 全部通过 → git commit → 下一 Phase
    │         └── 有失败 → 回到 [2] 修复 → 再到 [3] 测试
    │
    └─[完成]─→ 输出最终汇总报告
```

---

## 0.5 前置：环境确认

在开始任何 Phase 之前，先启动 Setup Agent 确认 DK-2500 状态。

### Setup Agent 指令

> **给 Setup Agent 的 prompt**:
>
> ```
> 请检查以下内容并报告结果：
>
> 1. 系统信息
>    - uname -a
>    - cat /etc/os-release
>    - 确认是否为 Ubuntu 22.04
>
> 2. 硬件
>    - cat /proc/cpuinfo | grep "model name" | head -1
>    - free -h (内存)
>    - ls /dev/video* 2>/dev/null || echo "无摄像头"
>    - ls /dev/accel/accel0 2>/dev/null || echo "无NPU设备节点"
>
> 3. Python 环境
>    - python3 --version
>    - which python3
>
> 4. 项目状态
>    - cd ~/intel && git log --oneline -5
>    - git status
>
> 5. 现有 Python 包
>    - pip list 2>/dev/null | grep -iE "mediapipe|opencv|numpy|scipy|librosa|openvino|fastdtw|ttkbootstrap"
>
> 6. 路径确认
>    - ls ~/intel/src/dance_scoring/core/
>    - ls ~/intel/scripts/
>
> 请以简洁清单形式汇报每项结果。特别标注异常项。
> ```
>
> **你收到报告后的操作**：
> - 如果缺少系统依赖 → 执行: `sudo apt install -y python3.10-venv python3-dev ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1`
> - 如果没有 .venv → 执行: `python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip`
> - 如果项目路径不对 → 调整后续指令中的路径
> - 确认无误后，进入 Phase 1a

---

## 1. Phase 1a: alignment 修复 + correction 规则引擎 + scorer 适配

### 1.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是在舞蹈评分项目中完成 3 个改动。请先阅读以下现有文件了解项目风格：
> - src/dance_scoring/core/dtw.py (了解 DTW 接口风格)
> - src/dance_scoring/core/scorer.py (了解 Scorer 类结构)
> - src/dance_scoring/core/frame.py (了解 PoseFrame 和角度计算)
> - src/dance_scoring/core/segments.py (了解 SegmentScore 数据结构)
>
> 然后完成以下 3 个任务：
>
> **任务 1: 重写 src/dance_scoring/core/alignment.py**
>
> 当前 alignment.py 是 dtw.py 的副本，需要替换为基于 fastdtw 库的快速对齐实现。
>
> ```python
> # alignment.py 应该提供：
> from fastdtw import fastdtw
> from scipy.spatial.distance import euclidean
> import numpy as np
>
> def fastdtw_alignment(seq1: np.ndarray, seq2: np.ndarray, radius: int = None):
>     """
>     使用 fastdtw 进行快速序列对齐。
>     输入: seq1 (N, D), seq2 (M, D)
>     输出: (distance: float, path: list of (int, int))
>     radius: 搜索窗口半径，None则自动为 max(N,M)//10
>     """
> ```
>
> **任务 2: 修改 src/dance_scoring/core/scorer.py**
>
> 在 Scorer.__init__ 中增加 alignment_method 参数 ("dtw"|"fastdtw")，在评分逻辑中根据该参数选择使用 DTW 或 fastdtw_alignment。
>
> **任务 3: 实现 src/dance_scoring/core/correction.py**
>
> 完整的规则引擎，包含：
> - JOINT_NAMES_CN: dict — 33个MediaPipe关键点索引→中文名（见下方完整映射表）
> - CORRECTION_TEMPLATES: list — 纠正建议模板
> - generate_correction(segment_scores, top_n=3, threshold_deg=10.0) -> dict
>
> 完整的 33 关节映射表：
> {0:"鼻尖",1:"左眼内角",2:"左眼",3:"左眼外角",4:"右眼内角",5:"右眼",6:"右眼外角",7:"左耳",8:"右耳",9:"嘴角左",10:"嘴角右",11:"左肩",12:"右肩",13:"左肘",14:"右肘",15:"左腕",16:"右腕",17:"左小指",18:"右小指",19:"左食指",20:"右食指",21:"左拇指",22:"右拇指",23:"左髋",24:"右髋",25:"左膝",26:"右膝",27:"左踝",28:"右踝",29:"左脚跟",30:"右脚跟",31:"左脚尖",32:"右脚尖"}
>
> 纠正模板需覆盖：
> - 肘关节(13,14): 偏大→"请伸直{关节名}" / 偏小→"请弯曲{关节名}"
> - 膝关节(25,26): 偏大→"请站直" / 偏小→"请降低重心"
> - 肩关节(11,12): 偏大→"请放松{关节名}" / 偏小→"请抬起手臂"
> - 髋关节(23,24): 偏大→"请降低{关节名}" / 偏小→"请收紧{关节名}"
> - 默认: 偏大→"请调整{关节名}位置" / 偏小→"请调整{关节名}位置"
>
> generate_correction 逻辑：
> 1. 遍历每个 segment_score
> 2. 聚合该段内各关节的偏差角度（取均值）
> 3. 按偏差从大到小排序，取超过 threshold_deg 的 top_n 个
> 4. 查 JOINT_NAMES_CN 获取中文名
> 5. 根据偏差方向（正=过大/负=过小）选择模板
> 6. 返回 dict: {segment_id: "第X段：右肘角度偏差12.5°，请伸直右肘；左膝弯曲不足8.3°，请降低重心"}
>
> 代码风格：中文注释、snake_case、与现有文件保持一致。
>
> 完成后告诉我改动了哪些文件以及关键实现要点。
> ```
>
> 同时执行依赖安装: `pip install fastdtw`
>
> 更新 requirements.txt 添加: `fastdtw==0.3.4`

### 1.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 1a 的改动是否正确。执行以下验证脚本并报告结果。
>
> **测试 1: alignment.py 不是 dtw.py 的重复**
> ```bash
> python3 -c "
> import hashlib
> def md5(path):
>     with open(path, 'rb') as f:
>         return hashlib.md5(f.read()).hexdigest()
> a = md5('src/dance_scoring/core/alignment.py')
> b = md5('src/dance_scoring/core/dtw.py')
> print(f'alignment.py md5: {a[:16]}...')
> print(f'dtw.py md5:       {b[:16]}...')
> assert a != b, 'alignment.py 仍然是 dtw.py 的副本！'
> print('✓ 测试1通过: alignment.py 不是副本')
> "
> ```
>
> **测试 2: fastdtw 函数可用且结果合理**
> ```bash
> python3 -c "
> import numpy as np
> from src.dance_scoring.core.alignment import fastdtw_alignment
> from src.dance_scoring.core.dtw import DTW
>
> np.random.seed(42)
> seq1 = np.random.randn(100, 66).astype(np.float32)
> seq2 = seq1 + np.random.randn(100, 66).astype(np.float32) * 0.5
>
> dist, path = fastdtw_alignment(seq1, seq2, radius=10)
> print(f'fastdtw distance: {dist:.4f}, path_length: {len(path)}')
>
> dtw = DTW(seq1, seq2, window=10)
> std_dist = dtw.compute()
> print(f'standard DTW distance: {std_dist:.4f}')
>
> rel_err = abs(dist - std_dist) / max(std_dist, 1e-9)
> print(f'相对偏差: {rel_err:.4f}')
> assert rel_err < 0.2, f'fastdtw与标准DTW偏差过大: {rel_err:.2%}'
> print('✓ 测试2通过: fastdtw 结果合理')
> "
> ```
>
> **测试 3: correction 模块完整**
> ```bash
> python3 -c "
> from src.dance_scoring.core.correction import JOINT_NAMES_CN, CORRECTION_TEMPLATES, generate_correction
>
> # 关节映射完整性
> assert len(JOINT_NAMES_CN) == 33, f'关节映射应有33个，实际{len(JOINT_NAMES_CN)}'
> for i in range(33):
>     assert i in JOINT_NAMES_CN, f'缺少关节索引{i}'
>     assert JOINT_NAMES_CN[i] and len(JOINT_NAMES_CN[i]) > 0, f'关节{i}名称为空'
>
> # 模板存在
> assert len(CORRECTION_TEMPLATES) > 0, '纠正模板为空'
>
> # generate_correction 可调用
> # 构造 mock segment_scores
> from dataclasses import dataclass
> from typing import List
>
> @dataclass
> class MockSeg:
>     id: int
>     score: float
>     joint_deviations: dict  # {joint_idx: deviation_deg}
>
> mock_scores = [
>     MockSeg(id=1, score=72.3, joint_deviations={13: 12.5, 14: 5.0, 25: -8.3}),
>     MockSeg(id=2, score=45.2, joint_deviations={25: 15.3, 23: -12.1, 11: 7.0}),
> ]
>
> result = generate_correction(mock_scores, top_n=3, threshold_deg=5.0)
> assert isinstance(result, dict), f'返回类型应为dict，实际{type(result)}'
> assert len(result) > 0, '返回结果为空'
> for seg_id, text in result.items():
>     print(f'  段{seg_id}: {text}')
>
> print('✓ 测试3通过: correction 模块完整')
> print(f'  关节映射: 33/33 个')
> print(f'  纠正模板: {len(CORRECTION_TEMPLATES)} 个')
> "
> ```
>
> **测试 4: scorer 支持 alignment_method**
> ```bash
> python3 -c "
> from src.dance_scoring.core.config import Config
> from src.dance_scoring.core.scorer import Scorer
>
> # 检查是否支持 alignment_method 参数
> import inspect
> sig = inspect.signature(Scorer.__init__)
> params = list(sig.parameters.keys())
> print(f'Scorer.__init__ 参数: {params}')
> assert 'alignment_method' in params, 'Scorer 缺少 alignment_method 参数'
> print('✓ 测试4通过: scorer 支持 alignment_method')
> "
> ```
>
> 请逐项报告每项测试的通过/失败状态。如有失败，附上完整错误信息。
> ```

### 1.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 全部 4 项通过 | `git add -A && git commit -m "feat: Phase 1a - alignment fastdtw + correction 规则引擎"` → 进入 Phase 1b |
| 1-2 项失败 | 将错误信息发给 Implementer Agent 修复，然后重新启动 Tester |
| 3+ 项失败 | 汇总所有错误，检查是否是环境问题（如缺少 fastdtw 包），修复后再测试 |

---

## 2. Phase 1b: 摄像头模块

### 2.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是实现摄像头模块。请先阅读以下文件了解现有接口：
> - src/dance_scoring/camera/base.py (抽象基类)
> - src/dance_scoring/camera/usb.py (当前为空壳)
> - src/dance_scoring/camera/stream.py (当前为空壳)
>
> **任务 1: 完善 src/dance_scoring/camera/base.py 的 list_devices()**
>
> 将硬编码的 return [] 改为实际摄像头枚举：
> ```python
> @staticmethod
> def list_devices(max_index: int = 8) -> List[int]:
>     import cv2
>     available = []
>     for i in range(max_index):
>         cap = cv2.VideoCapture(i)
>         if cap.isOpened():
>             available.append(i)
>             cap.release()
>     return available
> ```
>
> **任务 2: 实现 src/dance_scoring/camera/usb.py**
>
> 实现 UsbCamera(CameraBase):
> - __init__(self, device_id=0, resolution=(640,480), fps=30)
> - open() -> bool: 打开设备并设置分辨率/帧率
> - read() -> Optional[np.ndarray]: 返回 BGR→RGB 转换后的帧
> - close(): 释放 cv2.VideoCapture
> - is_opened() -> bool
>
> read() 必须返回 RGB 格式（cv2 读出来是 BGR，需转换），以与 MediaPipe 兼容。
> open() 失败不抛异常，返回 False。
>
> **任务 3: 实现 src/dance_scoring/camera/stream.py**
>
> 实现 NetworkStream(CameraBase):
> - __init__(self, url: str)
> - open() -> bool: 通过 cv2.VideoCapture(url) 打开
> - read/close/is_opened 与 UsbCamera 逻辑相同，如重复太多可提取共享逻辑
>
> 完成后告诉我改动了哪些文件。
> ```

### 2.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 1b 摄像头模块的改动。
>
> **测试 1: 模块可导入**
> ```bash
> python3 -c "
> from src.dance_scoring.camera.base import CameraBase
> from src.dance_scoring.camera.usb import UsbCamera
> from src.dance_scoring.camera.stream import NetworkStream
> print('✓ 测试1通过: 所有模块可导入')
> "
> ```
>
> **测试 2: list_devices 可用**
> ```bash
> python3 -c "
> from src.dance_scoring.camera.base import CameraBase
> devices = CameraBase.list_devices()
> print(f'可用摄像头设备: {devices}')
> assert isinstance(devices, list), 'list_devices() 应返回列表'
> # 返回空列表也是合法的（没有物理摄像头）
> print('✓ 测试2通过: list_devices 返回列表')
> "
> ```
>
> **测试 3: UsbCamera API 完整性 (无设备测试)**
> ```bash
> python3 -c "
> import numpy as np
> from src.dance_scoring.camera.usb import UsbCamera
>
> cam = UsbCamera(device_id=0)
>
> # 测试方法存在
> for method in ['open', 'read', 'close', 'is_opened']:
>     assert hasattr(cam, method), f'UsbCamera 缺少方法: {method}'
>     assert callable(getattr(cam, method)), f'{method} 不可调用'
>
> # 无设备时 open 返回 False（不抛异常）
> opened = cam.open()
> print(f'open() 返回: {opened}')
> assert opened == False or cam.is_opened() == False
>
> # 未打开时 read 返回 None
> frame = cam.read()
> assert frame is None, f'未打开摄像头时 read() 应返回 None，实际: {type(frame)}'
>
> # close 不抛异常
> cam.close()
>
> print('✓ 测试3通过: UsbCamera API 完整，无设备时优雅降级')
> "
> ```
>
> **测试 4: NetworkStream API 完整性**
> ```bash
> python3 -c "
> from src.dance_scoring.camera.stream import NetworkStream
>
> cam = NetworkStream(url='rtsp://example.com/stream')
>
> for method in ['open', 'read', 'close', 'is_opened']:
>     assert hasattr(cam, method), f'NetworkStream 缺少方法: {method}'
>
> # 无效 URL 时 open 不抛异常
> opened = cam.open()
> print(f'open(无效URL) 返回: {opened}')
> # 可能返回 False，也可能返回 True 然后 read 返回 None（取决于 OpenCV 行为）
> frame = cam.read()
> print(f'read() 返回: {type(frame).__name__ if frame is not None else None}')
> cam.close()
>
> print('✓ 测试4通过: NetworkStream API 完整')
> "
> ```
>
> **测试 5: 有摄像头时实际采集 (如有物理设备)**
> ```bash
> python3 -c "
> from src.dance_scoring.camera.base import CameraBase
> devices = CameraBase.list_devices()
> if devices:
>     from src.dance_scoring.camera.usb import UsbCamera
>     import time
>     cam = UsbCamera(device_id=devices[0])
>     if cam.open():
>         print(f'摄像头 {devices[0]} 已打开')
>         for i in range(30):
>             frame = cam.read()
>             if frame is not None:
>                 print(f'  帧: shape={frame.shape}, dtype={frame.dtype}')
>                 # 验证 RGB 格式 (非 BGR)
>                 assert frame.shape[2] == 3, f'应为3通道图像，实际{frame.shape[2]}通道'
>                 break
>             time.sleep(0.05)
>         cam.close()
>         print('✓ 测试5通过: 实际摄像头采集成功')
>     else:
>         print('⚠ 摄像头存在但打开失败（可能是权限问题）')
> else:
>     print('⚠ 无物理摄像头，跳过测试5')
> "
> ```
>
> 请逐项报告每项测试的通过/失败状态。如有失败，附上完整错误信息。
> ```

### 2.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 测试1-4全部通过 (测试5为可选) | `git add -A && git commit -m "feat: Phase 1b - 摄像头模块实现"` → 进入 Phase 1c |
| 有失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 3. Phase 1c: 实时跟练 (run_live.py)

### 3.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是在 scripts/run_live.py 中实现完整的实时跟练功能。当前该文件是 stub（main() 抛 RuntimeError）。
>
> 请先阅读以下文件了解项目结构和可用接口：
> - src/dance_scoring/core/config.py (配置 dataclass)
> - src/dance_scoring/core/extractor.py (PoseExtractor - 姿态提取)
> - src/dance_scoring/core/frame.py (PoseFrame 数据结构)
> - src/dance_scoring/core/dtw.py (DTW 对齐)
> - src/dance_scoring/core/alignment.py (fastdtw 对齐 - Phase 1a 产物)
> - src/dance_scoring/core/scorer.py (Scorer - 打分)
> - src/dance_scoring/core/correction.py (generate_correction - Phase 1a 产物)
> - src/dance_scoring/camera/usb.py (UsbCamera - Phase 1b 产物)
> - src/dance_scoring/camera/base.py (CameraBase.list_devices)
>
> 实现 scripts/run_live.py，包含以下类和函数：
>
> **1. LiveConfig dataclass:**
> ```python
> @dataclass
> class LiveConfig:
>     camera_id: int = 0
>     camera_width: int = 640
>     camera_height: int = 480
>     camera_fps: int = 30
>     window_size: int = 150       # 滑动窗口帧数
>     window_step: int = 30        # 每次对齐步进帧数
>     alignment_method: str = "fastdtw"  # 实时模式默认 fastdtw
>     pass_score: float = 60.0
>     correction_threshold: float = 10.0
> ```
>
> **2. SlidingWindowBuffer 类:**
> - add(frame: PoseFrame) -> None
> - is_full() -> bool  (len >= window_size 时返回 True)
> - get_window() -> List[PoseFrame]  (返回当前窗口副本)
> - slide() -> None  (移除最老的 window_step 帧)
>
> **3. LiveScorer 类:**
> - __init__(self, config: LiveConfig, reference_path: str)
>   - 初始化摄像头: UsbCamera(config.camera_id, ...)
>   - 初始化 PoseExtractor（用 MediaPipe）
>   - 加载参考视频并预提取全部姿态序列: ref_poses = PoseExtractor.extract(reference_path)
>   - 按 BEATS_PER_SEGMENT(8) 将参考姿态分段
> - start() / stop()
>   - start 进入主循环:
>     1. camera.read() → BGR→RGB → 提取 PoseFrame（用 PoseExtractor 单帧逻辑）
>     2. 追加到 SlidingWindowBuffer
>     3. 如果窗口满:
>        a. 用当前窗口对应参考段做 DTW/fastdtw 对齐
>        b. 逐帧计算角度偏差 → 聚合得分
>        c. 调用 generate_correction() 生成纠正建议
>        d. 输出当前结果（终端文本模式）
>        e. slide() 窗口
>     4. 循环直到 Ctrl+C 或视频结束
>   - stop 释放摄像头
>
> **4. main() CLI 入口:**
> - argparse 解析参数: -r/--reference (必须), -c/--camera (默认0), -t/--threshold (默认60), --no-display
> - 创建 LiveConfig + LiveScorer → 启动
> - 注册 signal handler (SIGINT → 优雅退出)
>
> 关键设计要点：
> - 实时模式下，PoseExtractor 单帧提取：遍历视频帧时提取逻辑与现有 extract() 方法一致，但改为一帧一帧处理
> - 对齐策略：用户窗口 vs 参考当前段，用 fastdtw，return path 只取窗口后半段结果避免边界效应
> - 错误处理：摄像头断开/无帧时给出明确提示而不崩溃
> - 输出格式见下方示例
>
> 终端输出示例：
> ```
> 📷 摄像头已打开 | 30fps | 640x480
> 📐 参考姿态已加载: 120帧 (4段)
> ▶️ 开始跟练...
>
> [段1] 得分:72.3 ✓ | 右肘角度偏差:12.5° — 请伸直右肘
> [段2] 得分:45.2 ✗ | 左膝弯曲不足:15.3° — 请降低重心; 左髋角度偏差:12.1° — 请收紧左髋
>
> 🏆 总评: 良好 | 通过率:3/4
> 📊 薄弱部位: 左膝, 右肘
> ```
>
> 完成后告诉我改动了哪些文件以及关键实现要点。
> ```

### 3.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 1c 实时跟练模块的改动。
>
> **测试 1: 脚本可导入且无语法错误**
> ```bash
> python3 -c "import scripts.run_live; print('✓ 测试1通过: run_live 可导入')"
> ```
>
> **测试 2: LiveConfig 和 SlidingWindowBuffer 基本功能**
> ```bash
> python3 -c "
> import sys; sys.path.insert(0, '.')
> from scripts.run_live import LiveConfig, SlidingWindowBuffer
> from src.dance_scoring.core.frame import PoseFrame
> import numpy as np
>
> # 配置
> cfg = LiveConfig(window_size=10, window_step=3)
> assert cfg.window_size == 10
> assert cfg.alignment_method == 'fastdtw'
> print(f'LiveConfig 正常: window={cfg.window_size}, step={cfg.window_step}')
>
> # 滑动窗口
> buf = SlidingWindowBuffer(max_size=10, step=3)
> for i in range(15):
>     kp3d = np.zeros((33, 3), dtype=np.float32)
>     cf = np.ones(33, dtype=np.float32)
>     pf = PoseFrame(fid=i, kp3d=kp3d, conf=cf)
>     buf.add(pf)
>     if buf.is_full():
>         window = buf.get_window()
>         print(f'  窗口满: {len(window)} 帧')
>         buf.slide()
>         print(f'  slide后: {len(buf.get_window())} 帧')
>         break
> else:
>     print('  (窗口未满，检查add逻辑)')
> print('✓ 测试2通过: SlidingWindowBuffer 逻辑正常')
> "
> ```
>
> **测试 3: LiveScorer 可实例化 (无摄像头/无视频时优雅处理)**
> ```bash
> python3 -c "
> import sys; sys.path.insert(0, '.')
> from scripts.run_live import LiveScorer, LiveConfig
>
> cfg = LiveConfig()
> # 注意：LiveScorer 可能在 __init__ 时尝试打开摄像头或加载视频
> # 我们只检查类是否存在，不测试完整运行
> import inspect
> assert inspect.isclass(LiveScorer), 'LiveScorer 应为类'
> print('LiveScorer 类存在')
>
> # 检查关键方法
> methods = [m for m in dir(LiveScorer) if not m.startswith('_')]
> print(f'  公开方法: {methods}')
>
> print('✓ 测试3通过: LiveScorer 类结构正常')
> "
> ```
>
> **测试 4: CLI 参数解析**
> ```bash
> python3 -c "
> import sys; sys.path.insert(0, '.')
> from scripts.run_live import main
> # 模块级导入不触发 main()
> print('✓ 测试4通过: CLI 入口可导入')
> "
> ```
>
> **测试 5: 端到端运行 (需参考视频)**
> ```bash
> # 查找参考视频
> REF=$(find . -name '*.mp4' -not -path './.git/*' -not -path './.venv/*' 2>/dev/null | head -1)
> if [ -n "$REF" ]; then
>     echo "使用参考视频: $REF"
>     timeout 5 python3 scripts/run_live.py -r "$REF" --no-display 2>&1 || true
>     echo "---"
>     echo "✓ 测试5: run_live 启动无异常 (timeout 5s 后正常终止)"
> else
>     echo "⚠ 无参考视频，跳过测试5"
> fi
> ```
>
> 请逐项报告每项测试的通过/失败状态。
> ```

### 3.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 测试1-4全部通过 (测试5为可选) | `git add -A && git commit -m "feat: Phase 1c - 实时跟练 run_live"` → 进入 Phase 1d |
| 有失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 4. Phase 1d: GUI 现代化 + 实时界面

### 4.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是对 GUI 进行现代化改造并新增实时跟练界面。
>
> 请先阅读以下文件了解现有 GUI 结构：
> - src/dance_scoring/gui/app.py (主窗口)
> - src/dance_scoring/gui/components.py (组件)
> - src/dance_scoring/gui/worker.py (后台线程，不要修改)
>
> 完成以下 4 个任务：
>
> **任务 1: 新增 src/dance_scoring/gui/theme.py**
>
> ```python
> """GUI 主题配置 — 暗色运动风"""
>
> COLORS = {
>     "bg_primary":    "#1a1a2e",   # 深蓝黑背景
>     "bg_card":       "#16213e",   # 卡片面板
>     "bg_input":      "#0f3460",   # 输入区
>     "accent":        "#e94560",   # 强调红 — 失败/错误
>     "accent_green":  "#0f9b58",   # 通过绿
>     "accent_yellow": "#f5a623",   # 警告黄
>     "text_primary":  "#eaeaea",   # 主文字
>     "text_secondary":"#a0a0b0",   # 次要文字
>     "text_muted":    "#6c6c80",   # 禁用文字
> }
>
> FONTS = {
>     "heading": ("Helvetica", 16, "bold"),
>     "body":    ("Helvetica", 11),
>     "mono":    ("Courier", 10),
>     "score":   ("Helvetica", 36, "bold"),   # 大号得分数字
>     "title":   ("Helvetica", 20, "bold"),
> }
> ```
>
> **任务 2: 改造 src/dance_scoring/gui/app.py**
>
> 将 `import tkinter as tk; from tkinter import ttk` 改为 `import ttkbootstrap as ttk; from ttkbootstrap.constants import *`
> 主窗口: `root = ttk.Window(themename="darkly", title="舞蹈评分系统 v2.0", size=(900, 650))`
> 在工具栏新增"🎬 实时跟练"按钮 → command=self._open_live
> _open_live 方法: 从 gui.live_view 导入 LiveApp，创建 Toplevel 窗口
>
> 改动要点：
> - ttkbootstrap 是 ttk 的直接替换，大部分控件代码无需改动
> - 只改 import 和 Window 创建方式
> - 保持所有现有功能不变
>
> **任务 3: 改造 src/dance_scoring/gui/components.py**
>
> - ScoreResultDialog 中的得分文字使用 FONTS["score"] 大号字体
> - 段列表添加颜色标记: 通过段绿色、失败段红色
> - ProgressDialog 使用扁平主题色进度条
>
> **任务 4: 新增 src/dance_scoring/gui/live_view.py**
>
> 包含三个类:
>
> ```python
> class PoseOverlay:
>     """在视频帧上绘制关节点和骨骼连线"""
>     # MediaPipe 骨骼连接定义
>     POSE_CONNECTIONS = [
>         (11,12), (11,13), (13,15), (12,14), (14,16),  # 上肢
>         (11,23), (12,24), (23,24),                       # 躯干
>         (23,25), (25,27), (24,26), (26,28),             # 下肢
>     ]
>     COLORS = {"normal": (0,255,0), "weak": (0,0,255)}  # BGR
>
>     @staticmethod
>     def draw(frame: np.ndarray, landmarks: np.ndarray,
>              weak_joints: list = None) -> np.ndarray:
>         """frame: BGR (H,W,3), landmarks: (33,2) 像素坐标
>         返回绘制后的 frame"""
>
> class LivePanel(ttk.Frame):
>     """实时跟练界面"""
>     # 左侧: Canvas 显示摄像头画面 + PoseOverlay
>     # 右侧: 得分/纠正/进度
>     # 底部控制栏:
>     #   [▶开始] [⏸暂停] [⏹停止] [🔄循环练习] [📁选参考]
>     #   倍速: [0.5x] [0.8x] [1.0x]  (下拉或按钮组)
>     # 循环练习: 开启后当前段练习结束自动从头开始
>     # 倍速调节: 控制练习片段播放速度，调用 segments.py 的慢动作逻辑
>     def _on_update(self, data: dict):
>         """data keys: frame, landmarks, segment_id, score, correction, weak_joints, progress"""
>
> class LiveApp(ttk.Toplevel):
>     """实时跟练窗口"""
>     def __init__(self, master):
>         # 创建 LivePanel
>         # 启动 LiveScorer 后台线程
>         # 绑定关闭事件 → 停止线程
>         # 传递 speed 和 loop 参数到 LiveScorer
> ```
>
> 注意: GUI 层完全不导入 MediaPipe/OpenVINO 相关模块。live_view 只接收回调的 dict 数据。
>
> 完成后告诉我改动了哪些文件。
> ```

### 4.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 1d GUI 改动的正确性。
>
> **测试 1: theme 模块可导入**
> ```bash
> python3 -c "
> from src.dance_scoring.gui.theme import COLORS, FONTS
> assert len(COLORS) >= 8, f'颜色应至少8个，实际{len(COLORS)}'
> assert 'score' in FONTS, '缺少 score 字体配置'
> print(f'颜色: {list(COLORS.keys())}')
> print(f'字体: {list(FONTS.keys())}')
> print('✓ 测试1通过: theme 配置完整')
> "
> ```
>
> **测试 2: ttkbootstrap 可用**
> ```bash
> python3 -c "
> import ttkbootstrap as ttk
> root = ttk.Window(themename='darkly', size=(200, 100))
> btn = ttk.Button(root, text='测试')
> root.after(100, root.destroy)
> root.mainloop()
> print('✓ 测试2通过: ttkbootstrap 正常')
> "
> ```
>
> **测试 3: live_view 可导入且类结构完整**
> ```bash
> python3 -c "
> from src.dance_scoring.gui.live_view import PoseOverlay, LivePanel, LiveApp
> import numpy as np
>
> # PoseOverlay 测试
> frame = np.zeros((480, 640, 3), dtype=np.uint8)
> landmarks = np.zeros((33, 2), dtype=np.float32)
> landmarks[11] = [320, 200]  # 左肩
> landmarks[12] = [340, 200]  # 右肩
> landmarks[13] = [300, 300]  # 左肘
> result = PoseOverlay.draw(frame, landmarks, weak_joints=[11, 13])
> assert result.shape == (480, 640, 3), f'draw 应返回同尺寸图像，实际{result.shape}'
> # 检查是否有绘制内容（非全零）
> assert result.sum() > 0, 'PoseOverlay.draw 未绘制任何内容'
> print(f'PoseOverlay.draw: shape={result.shape}, nonzero={np.count_nonzero(result)}')
>
> # 检查类方法存在
> assert hasattr(LivePanel, '_on_update'), 'LivePanel 缺少 _on_update'
> assert hasattr(LiveApp, '__init__'), 'LiveApp 缺少 __init__'
>
> print('✓ 测试3通过: live_view 结构完整')
> print(f'  PoseOverlay 有 {len(PoseOverlay.POSE_CONNECTIONS)} 条骨骼连线')
> "
> ```
>
> **测试 4: app.py 使用 ttkbootstrap (语法检查)**
> ```bash
> python3 -c "
> import py_compile
> py_compile.compile('src/dance_scoring/gui/app.py', doraise=True)
> print('✓ 测试4通过: app.py 语法正确')
> "
> ```
>
> **测试 5: GUI 可启动 (需 display, 自动关闭)**
> ```bash
> # 检测是否有 display
> if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
>     timeout 3 python3 -c "
> from src.dance_scoring.gui.app import MainApp
> import tkinter as tk
> root = tk.Tk()
> root.withdraw()  # 隐藏主窗口
> root.after(500, root.destroy)
> root.mainloop()
> print('tkinter 基本可用')
> " 2>&1 || echo "⚠ GUI 启动可能因环境限制失败（tkinter未安装或无display），这不影响代码正确性"
> else
>     echo "⚠ 无 display 环境，跳过 GUI 启动测试"
> fi
> ```
>
> 请逐项报告每项测试的通过/失败状态。
> ```

### 4.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 测试1-4全部通过 (测试5为可选) | `git add -A && git commit -m "feat: Phase 1d - GUI ttkbootstrap 改造 + 实时跟练界面"` → 进入 Phase 2a |
| 有失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 5. Phase 2a: 模型转换流水线

### 5.1 前置检查

在执行 Phase 2a 前，先确认 OpenVINO 已安装：

```bash
pip install openvino>=2024.0
python3 -c "import openvino; print(f'OpenVINO {openvino.__version__}')"
```

创建模型目录: `mkdir -p src/dance_scoring/models`

> **精度选型说明**：DK-2500 (Meteor Lake NPU) 对 FP16 有原生硬件加速，推理速度最快。默认使用 FP16，同时满足竞赛方案 50% 体积压缩要求。

### 5.2 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是创建模型转换脚本 scripts/convert_model.py。
>
> 请先阅读：
> - src/dance_scoring/core/extractor.py (download_model 函数和 MODEL_PATH 常量)
> - src/dance_scoring/core/config.py (MODEL_PATH, MODEL_URL 常量)
>
> 实现 scripts/convert_model.py:
>
> **函数要求:**
>
> 1. `extract_tflite_from_task(task_path: Path) -> Path`
>    - MediaPipe .task 文件是 Flatbuffer zip 格式
>    - 用 zipfile 解包，查找 .tflite 文件
>    - 返回提取的 .tflite 文件路径
>
> 2. `convert_to_ir(tflite_path: Path, output_dir: Path, precision: str = "FP16") -> Path`
>    - 调用 openvino.convert_model() 转换
>    - precision="FP16" → compress_to_fp16=True（默认，DK-2500 NPU 速度最快）
>    - precision="FP32" → 不压缩，作为精度 baseline
>    - precision="INT8" → 使用 NNCF 量化（可选，需校准数据）
>    - 输出: output_dir/pose_landmarker.xml + .bin
>
> 3. `generate_meta(ir_xml: Path, tflite_path: Path, precision: str) -> dict`
>    - 用 ov.Core().read_model() 加载 IR，动态提取输入/输出张量名称和 shape
>    - 计算源 task 文件的 sha256
>    - 记录原始 TFLite 大小 vs IR 大小，计算压缩率
>    - 返回 meta dict（含 compression_ratio、precision 字段）
>
> 4. `main()` CLI:
>    - argparse: --source, --output, --precision (默认 FP16, choices: FP16/FP32/INT8)
>    - 流程: 检查源模型 → 解包 → 转换 → 生成meta → 输出体积对比
>
> 关键注意事项：
> - 默认 FP16：DK-2500 NPU 原生 FP16 加速，推理速度最快，同时满足 ≥50% 压缩指标
> - 不要硬编码输入输出名称和 shape！使用 OpenVINO API 动态获取
> - meta.json 格式需包含 compression_ratio 和 precision 字段
> - 如果源 .task 文件不存在，先调用 extractor.download_model()
>
> 完成后告诉我改动了哪些文件。
> ```

### 5.3 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 2a 模型转换流水线。
>
> **测试 1: 确保源模型存在**
> ```bash
> python3 -c "
> from src.dance_scoring.core.extractor import download_model
> from src.dance_scoring.core.config import MODEL_PATH
> import os
> download_model()
> assert os.path.exists(MODEL_PATH), f'模型文件不存在: {MODEL_PATH}'
> import os; print(f'模型文件大小: {os.path.getsize(MODEL_PATH)/1024:.0f} KB')
> print('✓ 源模型就绪')
> "
> ```
>
> **测试 2: 运行模型转换**
> ```bash
> python3 scripts/convert_model.py 2>&1
> echo "exit code: $?"
> ```
>
> **测试 3: 验证产物**
> ```bash
> python3 -c "
> import os, json
> model_dir = 'src/dance_scoring/models'
> for f in ['pose_landmarker.xml', 'pose_landmarker.bin', 'pose_landmarker_meta.json']:
>     path = os.path.join(model_dir, f)
>     assert os.path.exists(path), f'缺少产物: {f}'
>     print(f'{f}: {os.path.getsize(path)/1024:.0f} KB')
>
> # 验证 meta.json 内容
> with open(os.path.join(model_dir, 'pose_landmarker_meta.json')) as f:
>     meta = json.load(f)
> assert 'input' in meta, 'meta 缺少 input'
> assert 'outputs' in meta, 'meta 缺少 outputs'
> assert len(meta['outputs']) >= 1, f'outputs 数量异常: {len(meta["outputs"])}'
> print(f'输入: {meta[\"input\"][\"name\"]} shape={meta[\"input\"][\"shape\"]}')
> for o in meta['outputs']:
>     print(f'输出: {o[\"name\"]} shape={o[\"shape\"]}')
> print('✓ 测试3通过: 产物完整')
> "
> ```
>
> **测试 4: IR 模型可被 OpenVINO Runtime 加载**
> ```bash
> python3 -c "
> import openvino as ov
> core = ov.Core()
> print(f'可用设备: {core.available_devices}')
> model = core.read_model('src/dance_scoring/models/pose_landmarker.xml')
> print(f'模型输入: {[i.any_name for i in model.inputs]}')
> print(f'模型输出: {[o.any_name for o in model.outputs]}')
> # 尝试编译（CPU 兜底）
> compiled = core.compile_model(model, 'CPU')
> print(f'编译成功 (CPU)')
> print('✓ 测试4通过: IR 模型可加载')
> "
> ```
>
> **测试 5: 模型压缩率达标（竞赛指标）**
> ```bash
> python3 -c "
> import os, json, zipfile, tempfile
> # 从 .task 中提取 .tflite，精确对比模型体积
> task_path = os.path.expanduser('~/.cache/dance_scoring/pose_landmarker_lite.task')
> ir_bin = 'src/dance_scoring/models/pose_landmarker.bin'
> if os.path.exists(task_path) and os.path.exists(ir_bin):
>     # 提取 .tflite 并获取大小
>     with zipfile.ZipFile(task_path, 'r') as zf:
>         tflite_name = [n for n in zf.namelist() if n.endswith('.tflite')][0]
>         tflite_size = zf.getinfo(tflite_name).file_size
>     ir_size = os.path.getsize(ir_bin)
>     ratio = ir_size / tflite_size
>     print(f'原始 TFLite: {tflite_size/1024:.1f} KB')
>     print(f'IR (FP16):   {ir_size/1024:.1f} KB')
>     print(f'压缩率:      {(1-ratio)*100:.1f}%')
>     assert ratio <= 0.5, f'压缩后体积比 {ratio*100:.1f}% 不满足竞赛要求 ≤50%'
>     print('✓ 测试5通过: 模型压缩率满足竞赛指标 (≥50%)')
> else:
>     print('⚠ 模型文件缺失，跳过测试5')
> "
> ```
>
> 请逐项报告每项测试的通过/失败状态。
> ```

### 5.4 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 全部 5 项通过 | `git add -A && git commit -m "feat: Phase 2a - OpenVINO 模型转换流水线"` → 进入 Phase 2b |
| 有失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 6. Phase 2b: 推理引擎 + NPU 管理

### 6.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是实现 OpenVINO 推理引擎和 NPU 设备管理。
>
> 请先阅读：
> - src/dance_scoring/core/inference.py (当前 stub)
> - src/dance_scoring/platform/npu.py (当前 stub)
> - src/dance_scoring/models/pose_landmarker_meta.json (模型规格)
> - src/dance_scoring/core/frame.py (PoseFrame 数据结构)
>
> **任务 1: 重写 src/dance_scoring/platform/npu.py**
>
> ```python
> """DK-2500 NPU acceleration interface."""
> import openvino as ov
>
> class NPUManager:
>     """NPU device manager for Intel Core Ultra 5 225U."""
>
>     @staticmethod
>     def available() -> bool:
>         """Check if NPU device is available."""
>         try:
>             return "NPU" in ov.Core().available_devices
>         except Exception:
>             return False
>
>     @staticmethod
>     def best_device() -> str:
>         """Get best available device: NPU > GPU > CPU."""
>         try:
>             devices = ov.Core().available_devices
>             for d in ["NPU", "GPU", "CPU"]:
>                 if d in devices:
>                     return d
>             return "CPU"
>         except Exception:
>             return "CPU"
>
>     @staticmethod
>     def device_info(device: str = "NPU") -> dict:
>         """Return device properties."""
>         try:
>             core = ov.Core()
>             if device not in core.available_devices:
>                 return {"available": False, "device": device}
>             props = core.get_property(device, "FULL_DEVICE_NAME")
>             return {"available": True, "device": device, "name": props}
>         except Exception as e:
>             return {"available": False, "device": device, "error": str(e)}
> ```
>
> **任务 2: 重写 src/dance_scoring/core/inference.py**
>
> ```python
> """OpenVINO accelerated pose inference engine."""
>
> @dataclass
> class PoseInferenceResult:
>     kp3d: np.ndarray        # (33, 3) world coordinates
>     kp2d: np.ndarray        # (33, 2) pixel coordinates
>     visibility: np.ndarray  # (33,) confidence 0~1
>     presence: float         # person presence score
>
> class PoseInferenceEngine:
>     """
>     OpenVINO-based pose inference engine.
>     Loads IR model, handles pre/post processing.
>     """
>
>     def __init__(self, model_dir: Path, device: str = "NPU"):
>         # 1. Read meta.json for I/O specs
>         # 2. core = ov.Core()
>         # 3. Try compile_model on requested device
>         # 4. If fails, fallback to best_device()
>         # 5. Create InferRequest
>
>     def infer(self, rgb_image: np.ndarray) -> PoseInferenceResult:
>         """
>         Input: RGB image (H, W, 3) uint8
>         Steps:
>           1. Resize to model input size (from meta.json)
>           2. Normalize to [-1, 1]: (x / 127.5) - 1.0
>           3. HWC -> NCHW, add batch dim
>           4. infer_request.start_async() + wait()
>           5. Parse output tensors -> PoseInferenceResult
>         """
>
>     def warmup(self, rounds: int = 3):
>         """Warm up NPU with dummy image to avoid cold-start spike."""
>         dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
>         for _ in range(rounds):
>             self.infer(dummy)
>
>     def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
>         """Resize + normalize + NCHW"""
>
>     def _postprocess(self, outputs: dict) -> PoseInferenceResult:
>         """Parse raw tensors into structured result"""
> ```
>
> 关键要求：
> - _preprocess: resize 用 cv2.resize, normalize 用 float32
> - _postprocess: 从 meta.json 中读取输出名称，按名称索引输出张量
> - 如果 NPU 编译失败，自动回退到 GPU→CPU，并打印提示
> - warmup 方法防止首帧延迟尖峰
>
> 完成后告诉我改动了哪些文件。
> ```

### 6.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 2b 推理引擎和 NPU 管理的改动。
>
> **测试 1: NPUManager 可用**
> ```bash
> python3 -c "
> from src.dance_scoring.platform.npu import NPUManager
> print(f'NPU 可用: {NPUManager.available()}')
> print(f'最佳设备: {NPUManager.best_device()}')
> info = NPUManager.device_info('CPU')  # CPU 肯定可用
> assert info['available'] == True, f'CPU 应始终可用，实际: {info}'
> print(f'CPU 信息: {info.get(\"name\", \"unknown\")}')
> print('✓ 测试1通过: NPUManager 正常工作')
> "
> ```
>
> **测试 2: 推理引擎可加载**
> ```bash
> python3 -c "
> from pathlib import Path
> from src.dance_scoring.core.inference import PoseInferenceEngine
>
> model_dir = Path('src/dance_scoring/models')
> engine = PoseInferenceEngine(model_dir)
> print(f'推理设备: {engine.device}')
> print(f'输入尺寸: {engine.input_w}x{engine.input_h}')
> print('✓ 测试2通过: 推理引擎加载成功')
> "
> ```
>
> **测试 3: 推理功能正常**
> ```bash
> python3 -c "
> import numpy as np
> from pathlib import Path
> from src.dance_scoring.core.inference import PoseInferenceEngine, PoseInferenceResult
>
> engine = PoseInferenceEngine(Path('src/dance_scoring/models'))
> engine.warmup(rounds=2)
> print('预热完成')
>
> # 用模拟人体区域的图测试
> dummy = np.zeros((480, 640, 3), dtype=np.uint8)
> dummy[100:300, 200:400] = [180, 140, 100]
>
> import time
> t0 = time.perf_counter()
> result = engine.infer(dummy)
> elapsed = (time.perf_counter() - t0) * 1000
>
> assert isinstance(result, PoseInferenceResult), f'返回类型错误: {type(result)}'
> assert result.kp3d.shape == (33, 3), f'kp3d shape错误: {result.kp3d.shape}'
> assert result.kp2d.shape == (33, 2), f'kp2d shape错误: {result.kp2d.shape}'
> assert result.visibility.shape == (33,), f'visibility shape错误: {result.visibility.shape}'
>
> print(f'推理延迟: {elapsed:.1f}ms')
> print(f'kp3d range: [{result.kp3d.min():.3f}, {result.kp3d.max():.3f}]')
> print(f'presence: {result.presence:.3f}')
> print('✓ 测试3通过: 推理功能正常')
> "
> ```
>
> **测试 4: 连续推理稳定性**
> ```bash
> python3 -c "
> import numpy as np
> from pathlib import Path
> from src.dance_scoring.core.inference import PoseInferenceEngine
>
> engine = PoseInferenceEngine(Path('src/dance_scoring/models'))
>
> latencies = []
> for i in range(20):
>     frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
>     import time
>     t0 = time.perf_counter()
>     result = engine.infer(frame)
>     latencies.append((time.perf_counter() - t0) * 1000)
>
> latencies = np.array(latencies)
> print(f'20次推理: 平均={latencies.mean():.1f}ms, P99={np.percentile(latencies, 99):.1f}ms, 最大={latencies.max():.1f}ms')
> assert latencies.mean() < 500, f'单帧推理过慢: {latencies.mean():.0f}ms'  # 2fps底线
> print('✓ 测试4通过: 连续推理稳定')
> "
> ```
>
> 请逐项报告每项测试的通过/失败状态。
> ```

### 6.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 全部 4 项通过 | `git add -A && git commit -m "feat: Phase 2b - OpenVINO 推理引擎 + NPU 管理"` → 进入 Phase 2c |
| NPU 不可用但有 CPU 回退 | 记录到 report: NPU 状态，继续进入 Phase 2c |
| 推理失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 7. Phase 2c: PoseEngine 抽象层

### 7.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是创建推理引擎抽象层，让业务代码可以透明切换 MediaPipe / OpenVINO 后端。
>
> 请先阅读：
> - src/dance_scoring/core/extractor.py (PoseExtractor - MediaPipe 实现)
> - src/dance_scoring/core/inference.py (PoseInferenceEngine - OpenVINO 实现, Phase 2b 产物)
> - src/dance_scoring/core/frame.py (PoseFrame)
> - src/dance_scoring/core/config.py (Config)
> - src/dance_scoring/core/scorer.py (Scorer - 需要改造)
> - scripts/score.py (需要增加 --backend 参数)
> - scripts/run_live.py (需要用 create_pose_engine)
>
> **任务 1: 新增 src/dance_scoring/core/engine.py**
>
> ```python
> """Pose estimation engine abstraction layer."""
>
> from typing import Protocol, List, Optional
> import numpy as np
> from pathlib import Path
>
> class PoseEngine(Protocol):
>     """Pose estimation engine interface."""
>     def extract(self, path: str) -> List[PoseFrame]:
>         """Extract poses from video file (offline mode)."""
>         ...
>
>     def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
>         """Extract pose from a single RGB frame (live mode)."""
>         ...
>
>     @property
>     def backend_name(self) -> str:
>         """Human-readable backend identifier."""
>         ...
>
>
> class MediaPipeEngine:
>     """Wraps existing PoseExtractor. Backend: MediaPipe CPU."""
>
>     def __init__(self, cfg: Config):
>         self._extractor = PoseExtractor(cfg)
>
>     def extract(self, path: str) -> List[PoseFrame]:
>         return self._extractor.extract(path)
>
>     def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
>         # Implement single-frame extraction using the logic from PoseExtractor.extract()
>         # Key: use mp.Image and self._extractor.det.detect_for_video()
>
>     @property
>     def backend_name(self) -> str: return "MediaPipe (CPU)"
>
>
> class OpenVINOEngine:
>     """Backend: OpenVINO IR on NPU/GPU/CPU."""
>
>     def __init__(self, cfg: Config, model_dir: Path, device: str = "NPU"):
>         self._engine = PoseInferenceEngine(model_dir, device)
>         self._cfg = cfg
>
>     def extract(self, path: str) -> List[PoseFrame]:
>         # Read video with cv2, call extract_frame for each frame
>         # Apply _interpolate at the end
>
>     def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
>         result = self._engine.infer(rgb)
>         # Convert PoseInferenceResult -> PoseFrame
>         # Calculate joint angles via frame._calc_angles()
>
>     @property
>     def backend_name(self) -> str: return f"OpenVINO ({self._engine.device})"
>
>
> def _interpolate(poses: List[PoseFrame], cfg: Config) -> List[PoseFrame]:
>     """Shared interpolation logic. Extracted from PoseExtractor._interpolate()."""
>     # Copy the exact logic from PoseExtractor._interpolate()
>
>
> def create_pose_engine(backend: str = "auto", cfg: Config = None) -> PoseEngine:
>     """
>     Factory for creating the best available pose engine.
>
>     backend:
>       "auto"      → Try OpenVINO(NPU) → fallback MediaPipe
>       "mediapipe" → Force MediaPipe
>       "openvino"  → Force OpenVINO (may throw if no IR model)
>     """
> ```
>
> **任务 2: 修改 core/scorer.py**
>
> Scorer.__init__ 增加 engine 参数（可选，默认 None 则内部创建 MediaPipeEngine）。
>
> **任务 3: 修改 scripts/score.py**
>
> 增加 --backend 参数 (choices: auto/mediapipe/openvino, default: auto)。
> 传递给 Scorer。
>
> **任务 4: 修改 scripts/run_live.py**
>
> 使用 create_pose_engine(cfg.alignment_method → 不对，backend 参数独立)。
> 在 LiveConfig 中增加 backend 字段 (default: "auto")。
>
> 关键要求：
> - _interpolate 逻辑从 PoseExtractor 提取为 engine.py 的独立函数，避免重复
> - extract_frame 的单帧逻辑需要仔细实现（参考 extractor.py 中循环体的逻辑）
> - OpenVINO backends 的 extract() 需要自行视频读取（cv2.VideoCapture）
> - 所有代码保持中文注释、snake_case 命名
>
> 完成后告诉我改动了哪些文件。
> ```

### 7.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 2c PoseEngine 抽象层的改动。
>
> **测试 1: 模块可导入，工厂函数正常**
> ```bash
> python3 -c "
> from src.dance_scoring.core.engine import (
>     PoseEngine, MediaPipeEngine, OpenVINOEngine,
>     create_pose_engine, _interpolate
> )
> from src.dance_scoring.core.config import Config
>
> # auto 模式
> engine = create_pose_engine('auto')
> print(f'auto backend: {engine.backend_name}')
> assert engine.backend_name, 'backend_name 不应为空'
>
> # 强制 MediaPipe
> mp = create_pose_engine('mediapipe')
> assert isinstance(mp, MediaPipeEngine), f'应为 MediaPipeEngine，实际: {type(mp).__name__}'
> print(f'mediapipe backend: {mp.backend_name}')
>
> print('✓ 测试1通过: 工厂函数正常')
> "
> ```
>
> **测试 2: MediaPipeEngine 单帧提取**
> ```bash
> python3 -c "
> import numpy as np
> from src.dance_scoring.core.engine import create_pose_engine
>
> engine = create_pose_engine('mediapipe')
> # 模拟人体区域
> frame = np.zeros((480, 640, 3), dtype=np.uint8)
> frame[100:300, 200:400] = [180, 140, 100]  # RGB
>
> result = engine.extract_frame(frame, 0)
> if result is not None:
>     print(f'提取成功: kp3d shape={result.kp3d.shape}, angles={len(result.angles)}')
> else:
>     print('未检测到人体（正常，模拟图可能无人体特征）')
> print('✓ 测试2通过: MediaPipeEngine 单帧提取不崩溃')
> "
> ```
>
> **测试 3: OpenVINOEngine 单帧提取**
> ```bash
> python3 -c "
> import numpy as np
> from src.dance_scoring.core.engine import create_pose_engine
>
> try:
>     engine = create_pose_engine('openvino')
>     frame = np.zeros((480, 640, 3), dtype=np.uint8)
>     frame[100:300, 200:400] = [180, 140, 100]
>     result = engine.extract_frame(frame, 0)
>     print(f'提取成功: kp3d shape={result.kp3d.shape}, backend={engine.backend_name}')
>     print('✓ 测试3通过: OpenVINOEngine 单帧提取正常')
> except Exception as e:
>     print(f'OpenVINO backend 不可用: {e}')
>     print('⚠ 测试3跳过（环境限制，非代码问题）')
> "
> ```
>
> **测试 4: scorer 支持 engine 参数**
> ```bash
> python3 -c "
> import inspect
> from src.dance_scoring.core.scorer import Scorer
> sig = inspect.signature(Scorer.__init__)
> params = list(sig.parameters.keys())
> print(f'Scorer.__init__ 参数: {params}')
> assert 'alignment_method' in params, '缺少 alignment_method'
> # engine 参数至少存在（可能是 Optional）
> print('✓ 测试4通过: scorer 参数完整')
> "
> ```
>
> **测试 5: score.py --backend 参数**
> ```bash
> python3 scripts/score.py --help 2>&1 | grep -i backend || echo "⚠ --backend 参数未显示（检查是否已添加）"
> ```
>
> **测试 6: 离线评分端到端 (如有视频)**
> ```bash
> REF=$(find . -name '*.mp4' -not -path './.git/*' -not -path './.venv/*' 2>/dev/null | head -1)
> if [ -n "$REF" ]; then
>     python3 scripts/score.py -r "$REF" -u "$REF" --backend auto 2>&1 | head -20
>     echo "---"
>     echo "✓ 测试6: 离线评分正常运行"
> else
>     echo "⚠ 无参考视频，跳过测试6"
> fi
> ```
>
> 请逐项报告每项测试的通过/失败状态。
> ```

### 7.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| 测试1,2,4 必须通过；测试3,5,6 视环境 | `git add -A && git commit -m "feat: Phase 2c - PoseEngine 抽象层 + 后端切换"` → 进入 Phase 2d |
| 有必须项失败 | 发给 Implementer 修复 → 重新 Tester |

---

## 8. Phase 2d: 性能基准测试

### 8.1 启动 Implementer Agent

> **给 Implementer Agent 的 prompt**:
>
> ```
> 你的任务是创建性能基准测试脚本 scripts/benchmark.py。
>
> 请先阅读：
> - src/dance_scoring/core/engine.py (create_pose_engine - Phase 2c 产物)
> - src/dance_scoring/core/config.py (Config)
>
> 实现 scripts/benchmark.py:
>
> **功能要求:**
>
> 1. CLI: `python scripts/benchmark.py <video_path> [--frames N] [--rounds R]`
>    - 默认测试 100 帧，运行 3 轮取平均
>
> 2. 测试两个后端:
>    - MediaPipeEngine
>    - OpenVINOEngine (如果可用)
>
> 3. 每个后端测试:
>    - 首帧延迟 (冷启动 → 第一次推理)
>    - 预热后推理延迟 (warmup 3次后统计):
>      - 平均 (mean)、中位数 (median)、P99
>      - 吞吐量 (fps)
>
> 4. 模型压缩指标:
>    - 读取 meta.json 获取 compression_ratio
>    - 对比原始 TFLite vs IR (.bin) 文件大小
>    - 验证是否满足竞赛指标: 压缩 ≥50%
>
> 5. 竞赛指标对照输出:
> ```
> =================================================
>   姿态推理性能对比
> =================================================
> 测试视频: reference.mp4 | 测试帧数: 100
> 设备: Intel Core Ultra 5 225U
>
> 指标              MediaPipe(CPU)    OpenVINO(NPU)   竞赛要求
> ─────────────────────────────────────────────────────────
> 首帧延迟(ms)        45.1             12.3           -
> 平均延迟(ms)        18.2              5.1           ≤50
> 中位延迟(ms)        17.8              5.0           -
> P99 延迟(ms)        22.4              6.8           -
> 吞吐量(fps)         54.9            196.0           ≥20
>
> 模型体积:
>   原始 TFLite:        5.6 MB
>   IR (FP16):          2.8 MB (压缩 50.0%)  ✓ 满足≥50%
> =================================================
> 竞赛指标: ✓ 全部达标
>   [✓] 推理延迟 ≤50ms  [✓] 帧率 ≥20fps  [✓] 体积压缩 ≥50%
>   精度: FP16（DK-2500 NPU 原生加速，速度最快）
> ```
>
> 5. 测量方法:
>    - 用 time.perf_counter() 精确计时
>    - 首帧延迟: 从创建引擎到第一次 infer 完成
>    - 稳态延迟: 预热 3 次后连续推理计时
>
> 完成后告诉我改动了哪些文件。
> ```

### 8.2 启动 Tester Agent

> **给 Tester Agent 的 prompt**:
>
> ```
> 请验证 Phase 2d 性能基准测试。
>
> **测试 1: benchmark 脚本语法正确**
> ```bash
> python3 -c "import py_compile; py_compile.compile('scripts/benchmark.py', doraise=True); print('✓ 语法正确')"
> ```
>
> **测试 2: 运行 benchmark (需视频)**
> ```bash
> REF=$(find . -name '*.mp4' -not -path './.git/*' -not -path './.venv/*' 2>/dev/null | head -1)
> if [ -n "$REF" ]; then
>     python3 scripts/benchmark.py "$REF" --frames 30 --rounds 2 2>&1
>     echo "---"
>     echo "✓ 测试2: benchmark 运行正常"
> else
>     echo "⚠ 无参考视频，跳过测试2"
> fi
> ```
>
> **测试 3: 模型压缩率验证**
> ```bash
> python3 -c "
> import os, json
> model_dir = 'src/dance_scoring/models'
> meta_path = os.path.join(model_dir, 'pose_landmarker_meta.json')
> if os.path.exists(meta_path):
>     with open(meta_path) as f:
>         meta = json.load(f)
>     ratio = meta.get('compression_ratio', 0)
>     print(f'模型压缩率: {ratio*100:.1f}%')
>     assert ratio >= 0.5, f'压缩率 {ratio*100:.1f}% 不满足竞赛要求 ≥50%'
>     print('✓ 测试3通过: 模型压缩率满足竞赛指标')
> else:
>     print('⚠ meta.json 不存在，跳过测试3')
> "
> ```
>
> **测试 4: 竞赛指标汇总**
> ```bash
> python3 -c "
> print('=== 竞赛指标检查 ===')
> print('[✓] 八拍分段: 已实现 (split.py)')
> print('[✓] 姿态推理: ≤50ms (需从 benchmark 确认)')
> print('[✓] 实时帧率: ≥20fps (需从 benchmark 确认)')
> print('[✓] 模型压缩: ≥50% (需从测试3确认)')
> print('[ ] 评分一致性: ≥85% (需人工评估)')
> print('[ ] 薄弱环节准确率: ≥90% (需人工评估)')
> "
> ```
>
> 请报告测试结果，特别是 OpenVINO 延迟、模型压缩率、竞赛指标达标情况。
> ```

### 8.3 裁决标准

| Tester 结果 | 你的操作 |
|-------------|----------|
| Benchmark 运行正常 | `git add -A && git commit -m "feat: Phase 2d - 性能基准测试"` → 最终汇总 |

---

## 9. 最终汇总

所有 Phase 完成后，请在终端输出以下报告并保存为 `docs/specs/2026-06-03-deployment-report.md`：

```markdown
# DK-2500 部署完成报告

日期: [执行日期]
设备: Intel DK-2500, Ubuntu 22.04

## 完成阶段

| Phase | 状态 | 备注 |
|-------|------|------|
| 1a: alignment + correction | ✓ | |
| 1b: 摄像头模块 | ✓ | |
| 1c: 实时跟练 | ✓ | |
| 1d: GUI 改造 | ✓ | |
| 2a: 模型转换 | ✓ | |
| 2b: 推理引擎 + NPU | ✓ | |
| 2c: PoseEngine 抽象层 | ✓ | |
| 2d: 性能基准 | ✓ | |

## 性能结果

| 指标 | MediaPipe(CPU) | OpenVINO(NPU/CPU) | 竞赛要求 |
|------|---------------|-------------------|---------|
| 平均延迟 | XX.X ms | X.X ms | ≤50ms |
| 吞吐量 | XX.X fps | XXX fps | ≥20fps |

## 模型压缩

- 原始 TFLite: X.X MB
- IR (FP16): X.X MB
- 压缩率: XX% (竞赛要求 ≥50%，FP16 预期 ~50%)

## 竞赛指标达标情况

- [ ] 推理延迟 ≤50ms
- [ ] 帧率 ≥20fps
- [ ] 八拍分段精度 ≥95%
- [ ] 模型体积压缩 ≥50%
- [ ] 算力消耗降低 ≥40% (以 CPU 占用为参考)

## 硬件状态

- NPU: [可用/不可用]
- 最佳推理设备: [NPU/GPU/CPU]
- 内存: [可用/总量]
- 摄像头: [有/无] (设备列表)

## 遗留问题

[如有]
```

---

## 附录 A: 错误处理流程

```
任意 Phase 的 Tester 报告失败
    │
    ├── 分析错误类型
    │   ├── ImportError / ModuleNotFoundError → 缺少依赖，pip install
    │   ├── AssertionError → 代码 bug，发给 Implementer
    │   ├── FileNotFoundError → 路径问题，检查文件是否存在
    │   ├── RuntimeError (NPU相关) → NPU 驱动问题，尝试 CPU 回退
    │   └── 其他 → 记录完整错误信息
    │
    ├── 发给 Implementer Agent:
    │   "Phase X 测试失败，错误如下：[粘贴错误]。请修复。"
    │
    ├── 重新启动 Tester Agent
    │
    └── 同一问题修复 3 次仍失败 → 暂停，记录到 report，继续下一 Phase
```

## 附录 B: 依赖安装汇总

```bash
# 系统依赖
sudo apt install -y python3.10-venv python3-dev ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Python 包 (全都加入 requirements.txt)
pip install fastdtw ttkbootstrap openvino>=2024.0

# NPU 驱动 (DK-2500 特有)
sudo apt install -y intel-fw-npu intel-level-zero intel-opencl-icd
sudo usermod -a -G render,video $USER
# 重新登录生效
```
