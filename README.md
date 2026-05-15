# 舞蹈评分系统 v6.0

基于 **MediaPipe 姿态检测** 和 **DTW（动态时间规整）** 算法的舞蹈动作对比评分系统。提取 33 个姿态关键点（世界坐标）并计算 26 个关节角度（含面部、四肢、躯干、髋胯、手足），通过加权关键点+角度融合特征进行 DTW 对齐，逐帧非线性评分，支持每段 60 分合格线判定，并复用 8 拍分段结果输出慢动作练习视频。

## 功能概述

- **姿态提取**：使用 MediaPipe Pose Landmarker 检测 33 个人体关键点（3D 世界坐标 + 2D 图像坐标），提取 26 个关节角度（上肢、下肢、躯干、髋胯、足部），Z 轴加权压缩
- **特征融合**：每帧向量 = 66 维世界坐标关键点（3D→2D，Z轴加权压缩）+ 26 维关节角度，统一使用世界坐标系避免坐标空间不一致。关键点按躯干/四肢加权，丢失关键点通过邻帧插值修复并保持骨骼长度约束
- **动作对齐**：带 Sakoe-Chiba 窗口约束的 DTW 算法，窗口比 = 序列长度的 10%，防止过度扭曲
- **分段评分**：基于局部窗口方差的自适应动作变化检测自动切段，每段判定合格（≥60）/ 不合格（<60），7 档分布计算总评
- **非线性评分**：3° 容差范围内满分，3°–15° 按 1.8 分/度扣分，15° 以上按 3.0 分/度加速扣分
- **慢动作练习**：优先输出不合格段，其次输出低于自定义阈值的片段。直接复用 `split_8beats.py` 生成的 8 拍慢动作片段，无需二次提取
- **8拍分割**：独立工具支持音频节拍检测（librosa）、运动周期性检测（光流）、固定 BPM 三级回退

## 快速开始

```bash
# 安装依赖
.venv/Scripts/pip install -r requirements.txt

# 1. 先将参考视频切分为 8 拍慢动作段
python split_8beats.py -r videos/reference.mp4

# 2. 运行评分（需先执行第1步生成 output/segments/）
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4

# 自定义评分阈值（低于该分的段输出练习视频，默认 50）
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -t 60

# 指定 8 拍分段目录
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -s output/segments

# 自定义 BPM（音频检测失败时回退用）
python split_8beats.py -r videos/reference.mp4 -b 100

# 自定义输出目录
python split_8beats.py -r videos/reference.mp4 -o my_segments

# 检查 MediaPipe 安装
python check_env.py
```

## 评分体系

| 档位 | 条件 | 等级 |
|------|------|------|
| 完美 | ≥ 95 分 | — |
| 优秀 | 85–94 分 | ⭐优秀（总评 ≥ 90） |
| 良好 | 75–84 分 | 👍良好（总评 ≥ 78） |
| 一般 | 60–74 分 | 📝一般（总评 ≥ 60） |
| 较差 | 40–59 分 | ⚠️需改进（总评 ≥ 35） |
| 差 | 20–39 分 | 💪需重练 |
| 极差 | < 20 分 | 💪需重练 |

**每段合格线：60 分。** 存在不合格段时优先输出对应练习视频；全部合格时输出低于自定义阈值的片段。

**特征权重**：关键点位置 40% + 关节角度 60%，躯干中心关键点（髋、肩）权重 1.5×，肢体末端（手指、脚趾）权重 0.3–0.5×，膝关节角度权重 1.5×。

## 输出说明

- `output/segments/` — 8 拍分段慢动作片段（`ref_seg_NN_slow.mp4`）+ 合并文件（`all_segments_merged.mp4`）
- `output/low_score_clips/` — 需练习的低分/不合格片段（`practice_segNN_scoreXX_slow.mp4`）
- 终端输出：逐帧分档统计、各段得分与合格判定、总评等级与分数

## 技术栈

- **姿态检测**：MediaPipe Pose Landmarker（33 关键点，VIDEO 模式）
- **序列对齐**：DTW + Sakoe-Chiba 窗口约束（scipy.spatial.distance.cdist）
- **视频处理**：OpenCV / ffmpeg
- **音频分析**：librosa（节拍检测，可选）
- **数值计算**：NumPy / SciPy
