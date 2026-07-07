#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ · 高并发生产 (4Kctx × 4seqs = 168 tok/s)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_production.log"

check_running

print_header "Qwen3.6-27B-AWQ · 4Kctx×4并发 · 168 tok/s"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}上下文:${NC}   4,096 tokens"
echo -e "  ${CYAN}并发数:${NC}   4 seqs (KV池 2.40 GiB, 满ctx并发 8.33x)"
echo -e "  ${CYAN}总吞吐:${NC}   167.6 tok/s (实测)"
echo -e "  ${CYAN}单TPS:${NC}    ~48 tok/s"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
