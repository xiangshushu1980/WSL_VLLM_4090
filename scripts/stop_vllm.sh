#!/bin/bash
# ============================================================
# 停止 vLLM 服务器 — 优雅关闭 + 强制清理
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}  停止 vLLM 服务器${NC}"
echo -e "${YELLOW}============================================${NC}"

# 1. 找到所有 vLLM 相关进程
VLLM_PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true)
VLLM_SERVE_PIDS=$(pgrep -f "vllm serve" 2>/dev/null || true)
MULTIPROC_PIDS=$(pgrep -f "VLLM::EngineCore" 2>/dev/null || true)
RAY_PIDS=$(pgrep -f "ray.*vllm" 2>/dev/null || true)

ALL_PIDS=$(echo "$VLLM_PIDS $VLLM_SERVE_PIDS $MULTIPROC_PIDS $RAY_PIDS" | xargs -n1 | sort -u | tr '\n' ' ')

if [ -z "${ALL_PIDS// /}" ]; then
    echo -e "${GREEN}✅ 没有运行中的 vLLM 进程${NC}"

    # 仍然清理 GPU 残留
    echo ""
    echo "检查 GPU 残留进程..."
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$GPU_PROCS" ]; then
        echo -e "${YELLOW}发现 GPU 上的残留进程: $GPU_PROCS${NC}"
        for pid in $GPU_PROCS; do
            echo "  清理 PID $pid ..."
            kill -9 "$pid" 2>/dev/null || true
        done
        echo -e "${GREEN}✅ GPU 残留已清理${NC}"
    else
        echo -e "${GREEN}✅ GPU 无残留进程${NC}"
    fi
    exit 0
fi

echo "发现以下 vLLM 相关进程:"
echo "$ALL_PIDS" | xargs -n1 | while read pid; do
    if [ -n "$pid" ]; then
        ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || echo "  PID $pid (已退出)"
    fi
done

echo ""

# 2. 优雅关闭: SIGTERM
echo -e "${YELLOW}步骤 1/3: 发送 SIGTERM (优雅关闭)...${NC}"
for pid in $ALL_PIDS; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
done

# 等待进程退出 (最多 10 秒)
for i in $(seq 1 10); do
    remaining=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null | wc -l)
    if [ "$remaining" -eq 0 ]; then
        echo -e "${GREEN}✅ 所有进程已优雅退出 (${i}s)${NC}"
        break
    fi
    sleep 1
done

# 3. 强制关闭: SIGKILL
echo -e "${YELLOW}步骤 2/3: 发送 SIGKILL (强制清理)...${NC}"
for pid in $ALL_PIDS; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
done
sleep 1

# 4. 清理 GPU 上残留的计算进程
echo -e "${YELLOW}步骤 3/3: 清理 GPU 残留...${NC}"
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n "$GPU_PROCS" ]; then
    echo "  发现的 GPU 残留 PID: $GPU_PROCS"
    for pid in $GPU_PROCS; do
        kill -9 "$pid" 2>/dev/null && echo "  已终止 PID $pid" || true
    done
fi

# 5. 最终检查
echo ""
remaining=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo -e "${GREEN}✅ vLLM 已完全停止${NC}"
else
    echo -e "${RED}⚠ 仍有残留进程: $(pgrep -f vllm | tr '\n' ' ')${NC}"
fi

# 6. 显示 GPU 释放情况
echo ""
echo "GPU 显存状态:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "  已用: %s | 空闲: %s\n", $1, $2}'
