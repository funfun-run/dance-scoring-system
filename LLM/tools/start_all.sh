#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 一键启动: LLM Bridge + Dance Scoring System
#
# 用法:
#   ./start_all.sh                        # 启动 bridge + 评分 CLI
#   ./start_all.sh gui                    # 启动 bridge + GUI
#   ./start_all.sh score -r ref.mp4 -u user.mp4  # bridge + 自定义参数
#   ./start_all.sh stop                   # 停止 bridge
#   ./start_all.sh status                 # 查看 bridge 状态
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/dance-scoring-system"
LLM_VENV="$SCRIPT_DIR/ov_npu_env"
DANCE_VENV="$PROJECT_DIR/.venv"
BRIDGE_PORT=8765
BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT"
BRIDGE_PID_FILE="/tmp/llm_bridge.pid"
BRIDGE_LOG="/tmp/llm_bridge.log"

# ─── 颜色 ─────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ─── 函数 ─────────────────────────────────────────────────────

start_bridge() {
    if pgrep -f "llm_bridge.py" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Bridge 已在运行${NC}"
        return 0
    fi

    echo -e "${GREEN}▶ 启动 LLM Bridge...${NC}"
    source "$LLM_VENV/bin/activate"
    nohup python "$SCRIPT_DIR/llm_bridge.py" --port "$BRIDGE_PORT" > "$BRIDGE_LOG" 2>&1 &
    echo $! > "$BRIDGE_PID_FILE"

    # 等待 bridge 就绪
    echo -n "   等待模型加载"
    for i in $(seq 1 30); do
        if curl -s "$BRIDGE_URL/health" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✓ Bridge 就绪${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo -e "\n${RED}✗ Bridge 启动超时！查看日志: $BRIDGE_LOG${NC}"
    return 1
}

stop_bridge() {
    if [ -f "$BRIDGE_PID_FILE" ]; then
        PID=$(cat "$BRIDGE_PID_FILE")
        if kill "$PID" 2>/dev/null; then
            echo -e "${GREEN}✓ Bridge 已停止 (PID=$PID)${NC}"
        fi
        rm -f "$BRIDGE_PID_FILE"
    fi
    # 兜底
    pkill -f "llm_bridge.py" 2>/dev/null || true
}

check_bridge() {
    if curl -s "$BRIDGE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Bridge 运行中 → $BRIDGE_URL${NC}"
        curl -s "$BRIDGE_URL/health" | python3 -m json.tool 2>/dev/null || true
    else
        echo -e "${RED}✗ Bridge 未运行${NC}"
    fi
}

cleanup() {
    echo ""
    echo -e "${YELLOW}正在关闭 Bridge...${NC}"
    stop_bridge
}

# ─── 命令分发 ─────────────────────────────────────────────────

case "${1:-score}" in
    stop)
        stop_bridge
        exit 0
        ;;
    status)
        check_bridge
        exit 0
        ;;
esac

# 捕获退出信号，自动清理
trap cleanup EXIT INT TERM

# 1. 启动 Bridge
start_bridge || exit 1

# 2. 启动 Dance Scoring System
echo -e "${GREEN}▶ 启动 Dance Scoring System...${NC}"
echo ""

source "$DANCE_VENV/bin/activate"
cd "$PROJECT_DIR"

case "${1:-score}" in
    gui)
        echo "  启动 GUI..."
        python src/dance_scoring/gui/app.py
        ;;
    score)
        shift
        echo "  运行评分: python scripts/score.py $@"
        python scripts/score.py "$@"
        ;;
    shell)
        echo "  进入交互 shell (输入 exit 退出)"
        bash --rcfile <(echo "source $DANCE_VENV/bin/activate; cd $PROJECT_DIR; echo 'Dance Scoring System env ready.'")
        ;;
    *)
        echo "  运行: $@"
        "$@"
        ;;
esac
