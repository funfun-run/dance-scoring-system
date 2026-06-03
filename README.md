# 舞蹈评分系统 v2.0

基于 **MediaPipe 姿态检测**、**fastdtw 时序对齐** 和 **OpenVINO NPU 加速** 的舞蹈动作对比评分系统。提取 33 个姿态关键点并计算 26 个关节角度，通过角度融合特征进行 DTW/fastdtw 对齐，逐帧非线性评分，输出分段成绩与中文纠正建议，支持生成慢动作练习视频。

> 🎯 **竞赛信息**：2026 年英特尔杯大学生电子设计竞赛嵌入式 AI 专题赛参赛项目  
> 📄 **设计文档**：[OpenVINO 加速集成方案](docs/specs/2026-06-03-openvino-integration-design.md) | [DK-2500 执行手册](docs/specs/2026-06-03-execution-manual.md)

## 功能概述

- **姿态提取**：MediaPipe Pose Landmarker 检测 33 个 3D 人体关键点，提取 26 个关节角度，Z 轴加权压缩
- **特征融合**：每帧向量 = 66 维世界坐标 (XY) + 26 维关节角度。丢失关键点通过邻帧插值修复
- **动作对齐**：支持标准 DTW（精确，离线）和 fastdtw（快速，实时）两种对齐算法
- **分段评分**：按固定 BPM 八拍等分分段，每段 60 分合格线，5 档总评判定
- **纠正建议**：自动定位薄弱关节，生成中文纠正提示（如"右肘角度偏差 12.5°，请伸直右肘"）
- **慢动作练习**：自动导出低分/不合格段慢动作练习视频（0.8x）
- **实时跟练**：USB 摄像头实时采集 + 姿态比对 + 即时反馈（开发中，见执行手册）
- **NPU 加速**：OpenVINO IR 推理引擎，FP16 精度，DK-2500 NPU 原生加速（开发中）
- **现代 GUI**：ttkbootstrap 暗色运动风界面，支持骨骼叠加可视化（开发中）

## 快速开始

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 检查环境
python check_env.py
```

> 首次运行 `scripts/score.py` 会自动下载 MediaPipe 姿态模型（~5.6 MB），无需额外配置。

### 基础用法

```bash
# 1. 参考视频八拍分割
python scripts/split.py -r <reference.mp4>

# 2. 离线评分
python scripts/score.py -r <reference.mp4> -u <user.mp4>

# 3. 指定 BPM 和阈值
python scripts/split.py -r <reference.mp4> -b 100
python scripts/score.py -r <reference.mp4> -u <user.mp4> -b 100 -t 60

# 4. 启动 GUI
python src/dance_scoring/gui/app.py
```

## 评分体系

| 档位 | 判定条件 | 等级 |
|------|----------|------|
| 优秀 | 良好帧 ≥ 70% + 差帧 < 3% | ⭐优秀 |
| 良好 | 合格帧 ≥ 60% + 差帧 < 12% | 👍良好 |
| 需改进 | 差帧 15%–25% | ⚠️需改进 |
| 需重练 | 差帧 ≥ 25% | 💪需重练 |
| 不合格 | 合格帧 < 60% | ❌不合格 |

**每段合格线：60 分** | 关节角度权重：膝 1.5× / 肩髋 1.3–1.4× / 肘 0.8× / 腕 0.3–0.5×

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 姿态检测 | MediaPipe Pose Landmarker (33 KP) | 3D 关键点提取 |
| 序列对齐 | DTW (Sakoe-Chiba) + fastdtw | 动作时序对齐 |
| 视频处理 | OpenCV / ffmpeg | 视频读写、慢动作生成 |
| 音频分析 | librosa | 节拍/BPM 检测 |
| 模型加速 | OpenVINO (FP16, IR) | NPU 推理加速 |
| GUI | ttkbootstrap (darkly) | 暗色运动风界面 |
| 数值计算 | NumPy / SciPy | 矩阵运算、距离计算 |

## 输出说明

- `output/segments/` — 八拍分段慢动作片段 + 合并文件
- `output/low_score_clips/` — 低分/不合格练习片段
- 终端输出：分段得分、合格判定、纠正建议、总评等级

## 竞赛指标

| 指标 | 目标 |
|------|------|
| 八拍分段精度 | ≥95% |
| 推理延迟 | ≤50ms |
| 实时帧率 | ≥20fps |
| 模型压缩 | ≥50% |

## 文件结构

```
├── scripts/                    # CLI 入口
│   ├── score.py                # 离线舞蹈评分
│   ├── split.py                # 视频八拍分割
│   ├── run_live.py             # 实时跟练（开发中）
│   ├── convert_model.py        # OpenVINO 模型转换（开发中）
│   └── benchmark.py            # 性能基准测试（开发中）
├── src/dance_scoring/          # 核心包
│   ├── core/                   # AI 推理层（姿态估计/DTW/评分/分段/纠正/推理引擎）
│   ├── video/                  # 数据处理层（节拍检测/切割/合并）
│   ├── camera/                 # 感知采集层（USB/RTSP 摄像头）
│   ├── gui/                    # 交互反馈层（ttkbootstrap 现代界面）
│   ├── platform/               # DK-2500 硬件适配（NPU/GPIO）
│   ├── transfer/               # 数据交换（WiFi/BLE）
│   └── ros2/                   # ROS2 节点层
├── docs/
│   └── specs/                  # 设计文档 & 执行手册
├── tests/                      # 测试
├── check_env.py                # 环境诊断工具
├── requirements.txt            # Python 依赖
└── output/                     # 运行时产出
```

## 部署目标

Intel DK-2500 (Core Ultra 5 225U + NPU) · Ubuntu 22.04 · OpenVINO · HDMI 外接显示
