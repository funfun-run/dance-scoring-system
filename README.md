# 舞蹈评分系统 v0.01

基于 **MediaPipe 姿态检测** 和 **DTW（动态时间规整）** 算法的舞蹈动作对比评分系统。将用户舞蹈视频与参考视频进行逐帧姿态比对，自动生成评分报告，并输出低分片段的慢动作练习视频。

## 功能概述

- **姿态提取**：使用 MediaPipe Pose Landmarker 检测 33 个人体关键点，计算 12 个关节角度（肘、膝、髋、肩等）
- **动作对齐**：通过 DTW 算法将用户动作序列与参考序列进行时间对齐，消除节奏差异
- **分段评分**：基于动作变化检测自动切分舞蹈段落，逐段给出百分制评分
- **慢动作练习**：对低于阈值的片段自动生成 0.8× 慢速视频，方便针对性练习
- **8拍分割**：独立工具支持按 BPM 将参考视频切分为 8 拍段落

## 快速开始

```bash
# 安装依赖
.venv/Scripts/pip install mediapipe scipy numpy

# 运行评分系统
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4

# 自定义评分阈值（默认 50 分）
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -t 60

# 将参考视频按 8 拍分割
python split_8beats.py -r videos/reference.mp4

# 自定义 BPM
python split_8beats.py -r videos/reference.mp4 -b 100
```

## 输出说明

- `output/low_score_clips/` — 低于阈值的练习片段（慢动作）
- `output/segments/` — 8 拍分割结果

## 技术栈

- **姿态检测**：MediaPipe Pose Landmarker
- **序列对齐**：DTW（Dynamic Time Warping）
- **视频处理**：OpenCV / ffmpeg
- **数值计算**：NumPy / SciPy
