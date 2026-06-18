# 舞蹈评分系统 v2.0

> 🎯 **2026 年英特尔杯大学生电子设计竞赛嵌入式 AI 专题赛**参赛项目  
> 📄 设计文档：[OpenVINO 加速集成方案](docs/specs/2026-06-03-openvino-integration-design.md) | [DK-2500 执行手册](docs/specs/2026-06-03-execution-manual.md)

## 硬件平台

- **开发板**：Intel DK-2500（Core Ultra 5 225U + NPU）
- **操作系统**：Ubuntu 22.04
- **推理加速**：OpenVINO 2024.x（FP16 IR）
- **外设**：USB 摄像头、HDMI 外接显示

## 使用方法

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 检查环境
python check_env.py

# 1. 参考视频八拍分割
python scripts/split.py -r <reference.mp4>

# 2. 离线评分
python scripts/score.py -r <reference.mp4> -u <user.mp4>

# 3. 指定 BPM 和阈值
python scripts/split.py -r <reference.mp4> -b 100
python scripts/score.py -r <reference.mp4> -u <user.mp4> -b 100 -t 60

# 4. 实时跟练（USB 摄像头 + 滑动窗口评分）
python scripts/run_live.py -r <reference.mp4>

# 5. 启动 GUI
python src/dance_scoring/gui/app.py
```

> 首次运行 `scripts/score.py` 会自动下载 MediaPipe 姿态模型（~5.6 MB），无需额外配置。

## 技术选型

| 组件 | 技术 | 用途 |
|------|------|------|
| 姿态检测 | MediaPipe Pose Landmarker (33 KP) | 3D 人体关键点提取 |
| 序列对齐 | DTW (Sakoe-Chiba) + fastdtw | 动作时序对齐 |
| 视频处理 | OpenCV 4.8 / ffmpeg | 视频读写、慢动作生成、摄像头采集 |
| 音频分析 | librosa 0.10 | 节拍/BPM 检测 |
| 模型加速 | OpenVINO 2024.x (FP16 IR) | NPU/GPU/CPU 异构推理 |
| GUI | ttkbootstrap 1.10 (darkly 主题) | 暗色运动风桌面界面 |
| 本地 LLM | Qwen2.5 1.5B (OpenVINO) + 3B (llama.cpp GGUF) | 中文纠正建议 + AI 教练对话 |
| 数值计算 | NumPy 1.24 / SciPy 1.10 | 矩阵运算、距离计算 |

## 文件结构

```
dance-scoring-system/
├── scripts/                           # CLI 入口脚本
│   ├── score.py                       #   离线舞蹈评分
│   ├── split.py                       #   视频八拍分割（含慢动作导出）
│   ├── run_live.py                    #   实时跟练（USB 摄像头 + 滑动窗口）
│   ├── convert_model.py               #   MediaPipe .task → OpenVINO IR 转换
│   └── benchmark.py                   #   性能基准测试（MediaPipe vs OpenVINO）
│        └── score_bridge.py           #     子进程评分桥接（隔离 MediaPipe C++ crash）
├── src/dance_scoring/                 # 核心 Python 包
│   ├── core/                          # AI 推理核心层
│   │   ├── config.py                  #   全局常量（BPM/权重/阈值/关节定义）
│   │   ├── frame.py                   #   PoseFrame：单帧 33 关键点 + 26 角度数据结构
│   │   ├── extractor.py               #   MediaPipe 姿态提取器（视频/单帧）
│   │   ├── dtw.py                     #   Sakoe-Chiba 约束窗口 DTW（精确，离线）
│   │   ├── alignment.py               #   fastdtw 快速对齐封装（实时模式）
│   │   ├── scorer.py                  #   评分引擎：逐帧非线性评分 → 分段 → 总评
│   │   ├── segments.py                #   八拍分段 + 低分慢动作片段提取
│   │   ├── correction.py              #   规则引擎：关节偏差 → 中文纠正模板
│   │   ├── correction_provider.py     #   纠正建议抽象层（RuleBased / LLM 双后端）
│   │   ├── inference.py               #   OpenVINO IR 推理引擎（预处理/推理/后处理）
│   │   └── engine.py                  #   PoseEngine 双后端工厂（MediaPipe / OpenVINO）
│   ├── video/                         # 数据处理层
│   │   ├── info.py                    #   视频信息提取（fps/分辨率/时长）
│   │   ├── beat_detector.py           #   三级节拍检测（librosa 音频 / 光流运动 / 固定 BPM）
│   │   ├── splitter.py                #   八拍分段 + 0.8× 慢动作视频生成
│   │   └── merger.py                  #   视频片段合并（ffmpeg concat / OpenCV fallback）
│   ├── camera/                        # 感知采集层
│   │   ├── base.py                    #   CameraBase 抽象接口
│   │   ├── usb.py                     #   USB 摄像头（OpenCV → RGB，兼容 MediaPipe）
│   │   └── stream.py                  #   RTSP 网络流（stub）
│   ├── gui/                           # 交互反馈层 — ttkbootstrap 暗色运动风
│   │   ├── app.py                     #   GUI 主入口（全局异常捕获 + 窗口居中）
│   │   ├── hub.py                     #   主 Hub：四卡片导航 + 底部工具栏 + NPU 指示灯
│   │   ├── panels.py                  #   功能面板（评分/回顾/分割/设置/NPU/模型/性能）
│   │   ├── components.py              #   可复用组件（视频导入/得分显示/段进度条）
│   │   ├── worker.py                  #   后台线程（评分/分割，支持进度回调）
│   │   ├── live_view.py               #   实时跟练双画面（摄像头 + 参考视频同步）
│   │   ├── theme.py                   #   HUD 运动风色板（OLED 暗底 + 橙色调）+ 字体
│   │   └── logger.py                  #   全局日志系统（即时 flush，防段错误丢日志）
│   ├── platform/                      # DK-2500 硬件适配层
│   │   ├── npu.py                     #   NPU 设备管理器（可用检测 + 最佳设备选择）
│   │   └── gpio.py                    #   GPIO 管理器（stub，预留 LED/按键）
│   ├── transfer/                      # 数据交换层（stub，预留 WiFi/BLE）
│   └── ros2/                          # ROS2 节点层（stub，预留分布式部署）
├── LLM/                               # 本地大语言模型接入
│   ├── provider.py                    #   LLMProvider 抽象基类（Prompt 构建 + 响应解析）
│   ├── prompts.py                     #   Prompt 模板（针对 1.5B 小模型优化）
│   ├── my_qwen.py                     #   Qwen2.5 1.5B (OpenVINO) + 3B (llama.cpp) Provider
│   └── model_manager.py              #   全局模型生命周期管理器（单例 + 自动卸载）
├── docs/
│   ├── gui-requirements.md            #   GUI 需求规格说明书
│   └── specs/                         #   设计文档（OpenVINO 集成 / 执行手册 / 重构 / HUD 重设计）
├── tests/                             # 单元测试
├── videos/                            # 测试用参考/用户视频
├── output/                            # 运行时产出
│   ├── segments/                      #   八拍慢动作分段片段
│   └── low_score_clips/               #   低分/不合格练习片段
├── pyproject.toml                     # Python 包配置
├── requirements.txt                   # Python 依赖清单
├── check_env.py                       # 环境诊断工具
└── 启动GUI.bat                        # Windows GUI 一键启动
```
