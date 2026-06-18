# Visibility-Filtered Dance Scoring — 设计文档

**日期**: 2026-06-18
**状态**: 已批准
**关联**: [[2026-06-03-openvino-integration-design]]

---

## 问题

MediaPipe 对画面外的身体部位会输出低置信度的猜测坐标。当前评分系统 (`scorer.py`) 对所有 26 个关节角度（`ANGLE_JOINTS`）一视同仁地计算偏差，包括：

- 脸部关节：(1,0,4) 眼-鼻-眼，(7,0,8) 耳-鼻-耳
- 下半身关节：膝、踝、脚跟、脚尖（23-32 号关键点）

当用户视频只拍到上半身时：
- 腿部/脚部关键点置信度极低但系统仍用猜测值打分
- 纠正建议中出现"请调整左脚尖""请调整左耳"等不合理内容
- 总分被不可见关节的无效偏差拉低

## 方案

在评分层注入 visibility 过滤——利用已有的 `PoseFrame.conf`（33 个关键点的 MediaPipe visibility），逐帧判断每个角度是否"足够可见"，不可见的角度不参与评分和纠正。

### 数据流

```
extractor.py                         scorer.py
───────────                         ─────────
PoseFrame.conf[33]  ──传入──→   逐帧: 检查每个角度的 3 个关键点
(已存在，visibility)            是否全部 confidence ≥ 0.5
                                       ↓
                                不可见 → 该帧该角度记为 NaN
                                可见   → 正常计算偏差
                                       ↓
                                逐段聚合:
                                - 每关节统计"可见帧占比"
                                - 占比 < 30% → skipped_joints
                                - 占比 ≥ 30% → 正常偏差
                                       ↓
                                correction_provider:
                                - skipped_joints → 不提纠正
                                - 附加 "⚠️ XXX未入镜" 提醒
```

### 新增常量 (`config.py`)

| 常量 | 值 | 说明 |
|------|-----|------|
| `VISIBILITY_THRESHOLD` | `0.5` | 单帧中关键点置信度低于此值视为不可见 |
| `MIN_VISIBLE_FRAME_RATIO` | `0.3` | 关节在某段中可见帧占比低于此值则跳过 |

### 改动文件

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 2 个常量 |
| `scorer.py` | 逐帧可见性检查 + NaN 聚合 + skipped_joints 输出 |
| `correction_provider.py` | `SegmentInfo` 新增 `skipped_joints` 字段；`RuleBasedProvider` 过滤 + 提醒 |
| `correction.py` | 无改动（底层模板函数保持纯函数） |

### 不修改的文件

- `extractor.py` — `conf` 已在存，无需改动
- `frame.py` — `PoseFrame` 数据结构已完整
- `LLM/prompts.py` — Prompt 模板无需调整（LLM 只会看到已过滤的偏差列表）
- GUI 面板 — 纠正文本自动包含提醒，无需额外改动

## 实现细节

### scorer.py 逐帧过滤

```python
# 对每个 angle_idx, 检查其 3 个关键点
a, b, c = ANGLE_JOINTS[angle_idx]
angle_visible = all(
    ref[ri].conf[idx] >= VISIBILITY_THRESHOLD and
    user[ui].conf[idx] >= VISIBILITY_THRESHOLD
    for idx in (a, b, c)
)
if angle_visible:
    frame_joint_diffs.append(raw_diffs)
else:
    frame_joint_diffs.append(np.nan)  # 标记不可见
```

### scorer.py 段聚合

```python
for angle_idx, (a, b, c) in enumerate(ANGLE_JOINTS):
    diffs = [fd[angle_idx] for fd in seg_diffs]
    valid_diffs = [d for d in diffs if not np.isnan(d)]
    visibility_ratio = len(valid_diffs) / max(len(diffs), 1)
    
    if visibility_ratio < MIN_VISIBLE_FRAME_RATIO:
        seg['skipped_joints'].append(joint_name)
        continue
    
    mean_dev = np.nanmean(diffs)  # np.nanmean 自动忽略 NaN
    # ... 正常生成 Deviation
```

### Segment 输出新增字段

```python
seg['skipped_joints'] = ["左膝", "右脚尖", ...]
seg['joint_visibility'] = {"右肘": 0.95, "左膝": 0.12, ...}
```

### SegmentInfo 新增字段

```python
@dataclass
class SegmentInfo:
    # ... 现有字段不变 ...
    skipped_joints: List[str] = field(default_factory=list)  # 新增
```

### correction_provider.py 提醒文本

```python
# RuleBasedProvider.generate_correction()
# 1. 从 deviations 中排除 skipped_joints 包含的关节
# 2. 如有跳过的关节，末尾追加提醒
if segment.skipped_joints:
    suffix = f"（⚠️ {', '.join(segment.skipped_joints[:3])}等部位未入镜，已跳过）"
    return correction_text + suffix
```

## 边界情况

| 情况 | 行为 |
|------|------|
| 全程未拍到下半身 | 所有腿部关节（髋/膝/踝/跟/尖）标记为 skipped，评分仅基于上半身 |
| 偶尔几帧拍到脚 | `MIN_VISIBLE_FRAME_RATIO=0.3` 过滤掉（30% 以下不参与） |
| 参考全身、用户半身 | 两方都需可见 → 以用户为准（用户不可见则跳过） |
| 所有关节都不可见 | 理论上不会发生（至少 torso 在画面内）；若发生，段得分保持 NaN-safe |
| 插值修复过的帧 | 插值只修坐标不修 confidence，visibility 仍为原始值，不受影响 |
| 人脸关节 (0-10) | 同样被过滤（脸部无舞蹈评分意义），脸可见但 deviation 通常 <3° 已被 threshold 过滤 |

## 验证

```bash
# 1. 默认行为不变（规则引擎 + 全可见视频）
python scripts/score.py -r videos/ref.mp4 -u videos/user.mp4

# 2. GUI 正常启动
python src/dance_scoring/gui/app.py

# 3. 常量可导入
python -c "
from dance_scoring.core.config import VISIBILITY_THRESHOLD, MIN_VISIBLE_FRAME_RATIO
print(f'VISIBILITY_THRESHOLD={VISIBILITY_THRESHOLD}')
print(f'MIN_VISIBLE_FRAME_RATIO={MIN_VISIBLE_FRAME_RATIO}')
"
```
