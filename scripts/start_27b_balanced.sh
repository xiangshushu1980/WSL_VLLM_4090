#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ · 均衡配置 (8Kctx × 2seqs = 84 tok/s)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_balanced.log"

check_running

print_header "Qwen3.6-27B-AWQ · 8Kctx×2并发 · 84 tok/s"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}上下文:${NC}   8,192 tokens"
echo -e "  ${CYAN}并发数:${NC}   2 seqs (KV池 2.44 GiB, 满ctx并发 5.56x)"
echo -e "  ${CYAN}总吞吐:${NC}   83.7 tok/s (实测)"
echo -e "  ${CYAN}单TPS:${NC}    ~48 tok/s"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
