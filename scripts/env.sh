#!/bin/bash
# vLLM RTX 4090 共享环境变量 — 所有启动脚本的公共配置
set -euo pipefail

# Conda 环境
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# WSL2 必需
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# CUDA 13 pip 环境
export CUDA_HOME=/home/sean/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/sean/miniconda3/envs/vllm/bin:$CUDA_HOME/bin:$PATH

# CCCL 兼容性 — flashinfer + CUDA 13
export VLLM_USE_FLASHINFER_SAMPLER=0

# JIT 编译工具链
ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-gcc /home/sean/miniconda3/envs/vllm/bin/gcc 2>/dev/null
ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-g++ /home/sean/miniconda3/envs/vllm/bin/g++ 2>/dev/null
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++

# 模型根路径
export MODEL_ROOT="/home/sean/projects/vllm/models"

# 获取 WSL2 实际 IP（不是 localhost）
WSL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
export WSL_IP="${WSL_IP:-127.0.0.1}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印启动信息
print_header() {
    local title="$1"
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  ${title}${NC}"
    echo -e "${CYAN}============================================${NC}"
}

# 检查是否已有 vLLM 在运行
check_running() {
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
        echo -e "${RED}⚠ vLLM 服务器已在运行!${NC}"
        echo -e "  PID: $(pgrep -f 'vllm.entrypoints.openai.api_server' | tr '\n' ' ')"
        echo -e "  端口占用:"
        ss -tlnp 2>/dev/null | grep -E ':8000' || echo "  (未检测到端口占用)"
        echo ""
        read -p "  是否先停止旧实例? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "已取消。手动停止: bash scripts/stop_vllm.sh"
            exit 1
        fi
        echo -e "${YELLOW}正在停止旧实例...${NC}"
        bash "$(dirname "$0")/stop_vllm.sh"
        sleep 2
    fi
}

print_info() {
    echo -e "${BLUE}模型:${NC}     $1"
    echo -e "${BLUE}端口:${NC}     ${2:-8000}"
    echo -e "${BLUE}GPU:${NC}      RTX 4090 (24GB)"
    echo -e "${BLUE}WSL IP:${NC}   ${WSL_IP}"
    echo -e "${BLUE}日志:${NC}     ${3:-/tmp/vllm_server.log}"
    echo ""
    echo -e "${GREEN}API 地址:${NC}   http://${WSL_IP}:${2:-8000}/v1"
    echo -e "${GREEN}API 测试:${NC}   curl http://${WSL_IP}:${2:-8000}/v1/models"
    echo ""
}

# 公共 vllm 参数
# 所有 27B/35B 模型共享的 base flags
COMMON_ARGS="--tensor-parallel-size 1"
COMMON_ARGS="$COMMON_ARGS --gpu-memory-utilization 0.935"
COMMON_ARGS="$COMMON_ARGS --kv-cache-dtype fp8"
COMMON_ARGS="$COMMON_ARGS --quantization awq_marlin"
COMMON_ARGS="$COMMON_ARGS --trust-remote-code"
COMMON_ARGS="$COMMON_ARGS --language-model-only"
COMMON_ARGS="$COMMON_ARGS --reasoning-parser qwen3"
COMMON_ARGS="$COMMON_ARGS --host 0.0.0.0"
