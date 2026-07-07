#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ — 平衡配置（中等上下文·中等并发）
# ============================================================
# 场景: 日常使用 / 中等长度文档
# 上下文: 8,192 tokens
# 并发:   2 seqs
# 吞吐:   ~83.7 tok/s 总吞吐（实测）
# 单TPS:  ~48 tok/s
# 显存:   模型 19.0G + CUDA Graph 0.44G + KV cache 2.44G
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_balanced.log"

check_running

print_header "Qwen3.6-27B-AWQ · 平衡配置"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "${YELLOW}配置:${NC}   8,192 上下文 | 2 并发 | CUDA Graph ON"
echo -e "${YELLOW}预期:${NC}   ~84 tok/s 总吞吐 | ~48 tok/s 单请求"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
