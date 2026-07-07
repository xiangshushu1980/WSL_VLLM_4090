#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ — 生产配置（高并发·短上下文）
# ============================================================
# 场景: 在线服务 / 高吞吐
# 上下文: 4,096 tokens
# 并发:   4 seqs
# 吞吐:   ~167.6 tok/s 总吞吐（实测）
# 单TPS:  ~48 tok/s
# 显存:   模型 19.0G + CUDA Graph 0.48G + KV cache 2.4G
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_production.log"

check_running

print_header "Qwen3.6-27B-AWQ · 生产配置"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "${YELLOW}配置:${NC}   4,096 上下文 | 4 并发 | CUDA Graph ON"
echo -e "${YELLOW}预期:${NC}   ~168 tok/s 总吞吐 | ~48 tok/s 单请求"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
