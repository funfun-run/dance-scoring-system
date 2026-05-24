# check_env.py - 环境检查（修复MediaPipe误报）

import sys

def check():
    print("=" * 50)
    print("舞蹈评分系统 - 环境检查")
    print("=" * 50)
    print(f"Python: {sys.version[:30]}")
    
    ok = True
    
    # numpy
    try:
        import numpy; print(f"✓ numpy {numpy.__version__}")
    except: print("✗ numpy 未安装"); ok = False
    
    # scipy
    try:
        import scipy; print(f"✓ scipy {scipy.__version__}")
    except: print("✗ scipy 未安装"); ok = False
    
    # opencv
    try:
        import cv2; print(f"✓ opencv {cv2.__version__}")
    except: print("✗ opencv 未安装"); ok = False
    
    # mediapipe
    try:
        import mediapipe as mp
        print(f"✓ mediapipe {mp.__version__}")
        # N3修复：检测tasks而非solutions
        try:
            from mediapipe.tasks.python import vision
            print("  ✓ mediapipe.tasks 可用")
        except Exception as e:
            print(f"  ✗ mediapipe.tasks 不可用: {e}")
            ok = False
    except: print("✗ mediapipe 未安装"); ok = False
    
    # librosa
    try:
        import librosa; print(f"✓ librosa {librosa.__version__}")
    except: print("⚠ librosa 未安装（可选，音频节拍检测需要）")
    
    # openvino
    try:
        import openvino as ov
        print(f"✓ openvino {ov.__version__}")
    except: print("⚠ openvino 未安装（可选，NPU加速需要）")
    
    print("=" * 50)
    if ok: print("✅ 核心依赖就绪")
    else: print("❌ 请安装缺失依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    check()