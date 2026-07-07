#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ · 长上下文 (32Kctx × 2seqs = 87 tok/s)
# ============================================================
# KV池 2.44 GiB, 满ctx并发 2.08x — 刚好装下2个32K请求
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_long.log"

check_running

print_header "Qwen3.6-27B-AWQ · 32Kctx×2并发 · 87 tok/s"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}上下文:${NC}   32,768 tokens (per seq)"
echo -e "  ${CYAN}并发数:${NC}   2 seqs (KV池 2.44 GiB, 满ctx并发 ${YELLOW}2.08x${NC} — 勉强够)"
echo -e "  ${CYAN}总吞吐:${NC}   87.0 tok/s (实测, 2并发)"
echo -e "  ${CYAN}单TPS:${NC}    ~48 tok/s"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
