#!/bin/bash
# ================================================================
# DK-2500 NPU 驱动部署脚本 (v2 — 已修复 GPU 库冲突)
# Intel Core Ultra 5 225U (Meteor Lake) + Ubuntu 22.04
#
# ⚠️  v1 的严重 BUG（已修复）:
#     步骤 3/4 安装了 intel-level-zero-gpu + intel-opencl-icd
#     → 导致 libigdgmm12 被替换为 Intel 仓库的非兼容版本
#     → Mesa/GNOME Shell 崩溃，桌面图形界面不可用
#
#     修复: 移除所有 GPU compute 包。NPU 只需要固件 + 权限。
#           intel_vpu 内核驱动已随 Linux 6.8+ 内置加载。
#           OpenVINO 通过自己的 NPU 插件直接与内核驱动通信。
#
# 用法: bash scripts/setup_npu.sh
# 需要 sudo 权限
# ================================================================
set -e

echo "========================================"
echo " DK-2500 NPU 驱动部署 v2 (安全版)"
echo " Intel Core Ultra 5 225U / Meteor Lake"
echo "========================================"
echo ""
echo "⚠️  v1 事故回顾:"
echo "   intel-level-zero-gpu 替换了 libigdgmm12"
echo "   → 导致 Ubuntu 桌面崩溃"
echo "   v2 只安装 NPU 固件 + 配置权限"
echo "   不再安装任何 GPU compute 运行时"
echo ""

# ---- 步骤 1: 添加 Intel GPU/Compute APT 仓库 (仅用于固件) ----
echo "[1/5] 添加 Intel APT 仓库 + 设置低优先级 (防止覆盖系统包)..."

if [ ! -f /usr/share/keyrings/intel-graphics.gpg ]; then
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
        | sudo tee /etc/apt/sources.list.d/intel-gpu.list
    sudo apt update
    echo "✓ Intel 仓库已添加"
else
    echo "✓ Intel 仓库已存在"
fi

# 固定 Intel 仓库为低优先级 (400)，确保永不覆盖系统包
if [ ! -f /etc/apt/preferences.d/intel-gpu-pin ]; then
    cat <<'EOF' | sudo tee /etc/apt/preferences.d/intel-gpu-pin
# 固定 Intel GPU 仓库为低优先级，防止覆盖系统核心库
# (v1 事故: intel-level-zero-gpu 替换了 libigdgmm12 → 桌面崩溃)
Package: *
Pin: origin repositories.intel.com
Pin-Priority: 400
EOF
    echo "✓ Intel 仓库已固定为优先级 400"
else
    echo "✓ Intel 仓库优先级已固定"
fi

# ---- 步骤 2: 仅安装 NPU 固件 ----
echo ""
echo "[2/5] 安装 NPU 固件 (intel-fw-npu)..."
# ⚠️ 仅安装固件！不安装 intel-level-zero-gpu / intel-opencl-icd！
# NPU 使用 intel_vpu 内核驱动 (已加载)，不需要 GPU compute 运行时
sudo apt install -y intel-fw-npu 2>/dev/null || {
    echo "⚠ intel-fw-npu 未找到，尝试 firmware-intel-ivpu..."
    sudo apt install -y firmware-intel-ivpu 2>/dev/null || {
        echo "⚠ 固件包均不可用"
        echo "  Meteor Lake NPU 固件在 kernel 6.8+ 中已内置"
        echo "  检查: ls /lib/firmware/intel/vpu/ 或 dmesg | grep vpu"
    }
}

# ---- 步骤 3: 用户权限 ----
echo ""
echo "[3/5] 配置用户权限..."
sudo usermod -a -G render,video $USER 2>/dev/null || true

# 配置 udev 规则让 render 组可以访问 NPU 设备
if [ ! -f /etc/udev/rules.d/99-intel-npu.rules ]; then
    echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' \
        | sudo tee /etc/udev/rules.d/99-intel-npu.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "✓ udev 规则已添加"
else
    echo "✓ udev 规则已存在"
fi

# ---- 步骤 4: 验证当前状态 ----
echo ""
echo "[4/5] 验证 NPU 状态..."

echo ""
echo "内核驱动:"
if lsmod | grep -q intel_vpu; then
    echo "  ✓ intel_vpu 已加载"
else
    echo "  ✗ intel_vpu 未加载 (检查内核版本 ≥6.6?)"
fi

echo ""
echo "NPU 固件:"
if ls /lib/firmware/intel/vpu/*.bin 2>/dev/null; then
    echo "  ✓ NPU 固件已安装"
elif dmesg 2>/dev/null | grep -qi "vpu.*firmware"; then
    echo "  ✓ 内核已加载内置固件"
else
    echo "  ⚠ 未检测到固件 (可能已内置在内核中)"
fi

echo ""
echo "设备节点:"
if [ -c /dev/accel/accel0 ]; then
    PERMS=$(stat -c "%a %G" /dev/accel/accel0 2>/dev/null || echo "?")
    echo "  ✓ NPU 设备节点存在: /dev/accel/accel0 ($PERMS)"
else
    echo "  ✗ 无 NPU 设备节点"
fi

echo ""
echo "系统库保护:"
if [ -f /etc/apt/preferences.d/intel-gpu-pin ]; then
    echo "  ✓ Intel 仓库已固定为低优先级 (libigdgmm12 受保护)"
else
    echo "  ⚠ 未设置仓库优先级固定"
fi

# ---- 步骤 5: 完成 ----
echo ""
echo "[5/5] 检查 libigdgmm12 版本 (关键!)..."
dpkg -l libigdgmm12 2>/dev/null | grep libigdgmm12 | awk '{print "  libigdgmm12: " $3}' || echo "  (未安装)"
echo "  预期版本: 22.1.2+ds1-1 (Ubuntu 原版)"
echo "  ⚠️  如果是 22.5.x (Intel 仓库版) → 桌面会崩溃！"

echo ""
echo "========================================"
echo " 部署完成！"
echo "========================================"
echo ""
echo " 下一步:"
echo "   1. 重启 使权限和 udev 规则生效: sudo reboot"
echo "   2. 重启后验证 NPU:"
echo "      python3 -c \"from dance_scoring.platform.npu import NPUManager; print('NPU:', NPUManager.available())\""
echo ""
echo " 预期输出: NPU: True  或  NPU: False (如 firmware 仍缺失)"
echo ""
echo " ⚠️  本脚本不再安装 intel-level-zero-gpu / intel-opencl-icd"
echo "    这些是 GPU compute 包，会破坏桌面环境"
echo "    NPU 通过 intel_vpu 内核驱动工作，不需要它们"
echo "========================================"
