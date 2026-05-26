# 舞蹈评分系统 v6.2

基于 **MediaPipe 姿态检测** 和 **DTW（动态时间规整）** 算法的舞蹈动作对比评分系统。提取 33 个姿态关键点（世界坐标）并计算 26 个关节角度（面部、四肢、躯干、髋胯、手足），通过角度融合特征进行 DTW 对齐，逐帧非线性评分，支持每段 60 分合格线判定，并可复用或自包含提取 8 拍分段慢动作练习视频。

## v6.2 更新内容

### score_dance.py（v6.0 → v6.2）
- **统一分段逻辑**：移除旧的 `_changes_adaptive` / `_seg_adaptive` 自适应分段，改为 `_seg_by_beats` 固定 BPM 八拍分段，段号与 `split_8beats.py` 输出完全一致
- **新增 `-b` / `--bpm` 参数**：支持指定 BPM，默认 120，需与 `split_8beats.py` 保持一致
- **移除 `kp2d` 图像坐标**：`PoseFrame` 不再存储 2D 图像关键点，仅保留世界坐标 `kp3d`，简化数据结构
- **移除 `KP_WEIGHTS`**：关键点位置权重矩阵移除，评分仅依赖关节角度 (ANGLE_WEIGHTS)
- **自包含兜底提取**：当 `split_8beats.py` 的输出目录不存在或为空时，自动从参考视频实时提取低分/不合格片段，无需手动预处理
- **简化评分档位**：7 档分布统计合并为 5 档判定（优秀/良好/需改进/需重练/不合格）

### split_8beats.py（v2.2 → v2.3）
- **自动尺寸缩放**：`merge_videos` 合并时自动检测并缩放尺寸不一致的片段，避免合并失败
- **beat_times 兼容性修复**：`detect_beats_from_audio` 返回 `.tolist()` 普通列表
- **格式优化**：代码风格统一，变量命名规范

### check_env.py（全面重写）
- 从 3 行 MediaPipe 版本检测重写为完整环境诊断工具
- 检测 numpy / scipy / opencv / mediapipe（含 tasks 子模块）/ librosa / openvino
- 区分核心依赖（必须）与可选依赖（音频/加速）

### requirements.txt
- **新增依赖**：`openvino>=2024.0.0`（NPU 加速）、`madmom>=0.16.1`（备选节拍检测）
- **版本升级**：`mediapipe>=0.10.0` → `>=0.10.32`

## 功能概述

- **姿态提取**：使用 MediaPipe Pose Landmarker 检测 33 个人体关键点（3D 世界坐标），提取 26 个关节角度（上肢、下肢、躯干、髋胯、足部），Z 轴加权压缩
- **特征融合**：每帧向量 = 66 维世界坐标关键点 (XY) + 26 维关节角度，统一使用世界坐标系。丢失关键点通过邻帧插值修复并保持骨骼长度约束
- **动作对齐**：带 Sakoe-Chiba 窗口约束的 DTW 算法（窗口比 = 序列长度的 10%），防止过度扭曲
- **分段评分**：按固定 BPM 八拍等分分段（与 `split_8beats.py` 统一），每段判定合格（≥60）/ 不合格（<60），5 档判定输出总评
- **非线性评分**：3° 容差内满分，3°–15° 按 1.8 分/度扣分，15° 以上按 3.0 分/度加速扣分
- **慢动作练习**：优先输出不合格段，其次输出低于自定义阈值的片段。优先复用 `split_8beats.py` 的输出，缺失时自动从参考视频实时提取
- **8拍分割**：独立工具支持音频节拍检测（librosa）、运动周期性检测（光流）、固定 BPM 三级回退

## 快速开始

### 环境要求

```bash
# 创建并激活虚拟环境
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt

# 检查环境
python check_env.py
```

> **Note**: 首次运行 `score_dance.py` 会自动下载 MediaPipe 姿态模型（~5.6 MB），无需额外配置。

### 基础用法

```bash
# 1. 将参考视频切分为 8 拍慢动作段
python split_8beats.py -r videos/reference.mp4

# 2. 运行评分（需 BPM 与第1步一致，默认 120）
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4

# 3. 指定 BPM（需与 split_8beats 一致）
python split_8beats.py -r videos/reference.mp4 -b 100
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -b 100
```

### 自定义参数

```bash
# 提高低分阈值（低于该分的段输出练习视频，默认 50）
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -t 60

# 自定义输出目录
python split_8beats.py -r videos/reference.mp4 -o my_segments

# 指定 8 拍分段目录
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -s output/segments
```

## 评分体系

| 档位 | 判定条件 | 等级 |
|------|----------|------|
| 优秀 | 良好帧 ≥ 70% 且 差帧 < 3% | ⭐优秀（总评 ≥ 90） |
| 良好 | 合格帧 ≥ 60% 且 差帧 < 12% | 👍良好（总评 ≥ 78） |
| 需改进 | 差帧 15%–25% | ⚠️需改进（总评 ≥ 35） |
| 需重练 | 差帧 ≥ 25% | 💪需重练 |
| 不合格 | 合格帧 < 60% | ❌不合格 |

**每段合格线：60 分。** 存在不合格段时优先输出对应练习视频；全部合格时输出低于自定义阈值的片段。

**关节角度权重**：膝关节 1.5×，肩髋 1.3–1.4×，肘部 0.8×，腕部 0.3–0.5×，足部 0.6–0.8×。

## 输出说明

- `output/segments/` — 8 拍分段慢动作片段（`ref_seg_NN_slow.mp4`）+ 合并文件（`all_segments_merged.mp4`）
- `output/low_score_clips/` — 需练习的低分/不合格片段（`practice_segNN_scoreXX_slow.mp4`）
- 终端输出：逐帧分档统计、各段得分与合格判定、总评等级与分数

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 姿态检测 | MediaPipe Pose Landmarker (33 KP, VIDEO 模式) | 3D 关键点提取 |
| 序列对齐 | DTW + Sakoe-Chiba 窗口 (scipy cdist) | 动作时序对齐 |
| 视频处理 | OpenCV / ffmpeg | 视频读写、慢动作生成 |
| 音频分析 | librosa → madmom 回退 | 节拍/BPM 检测 |
| 模型加速 | OpenVINO | NPU 推理加速（可选） |
| 数值计算 | NumPy / SciPy | 矩阵运算、距离计算 |

## 文件结构

```
├── score_dance.py         # 主评分流水线 (v6.2)
├── split_8beats.py        # 8拍慢动作分割工具 (v2.3)
├── check_env.py           # 环境诊断工具
├── requirements.txt       # Python 依赖
├── pose_landmarker_lite.task  # MediaPipe 模型（自动下载）
├── .vscode/               # VS Code 配置
├── videos/                # 输入视频
│   ├── reference.mp4
│   └── user.mp4
└── output/                # 输出目录
    ├── segments/          # 8拍慢动作片段
    └── low_score_clips/   # 低分练习片段
```
