#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ · 极限上下文 (73Kctx × 1seq = 48 tok/s)
# ============================================================
# KV池 2.48 GiB, 满ctx并发 1.00x — 一个token都不浪费的物理极限
# 首次启动会因 torch.compile 临时显存不足而失败，重启即可
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_maxctx.log"

check_running

print_header "Qwen3.6-27B-AWQ · 73Kctx×1并发 · 48 tok/s (物理极限)"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}上下文:${NC}   73,728 tokens (per seq) — ${RED}物理极限${NC}"
echo -e "  ${CYAN}并发数:${NC}   1 seq (KV池 2.48 GiB, 满ctx并发 ${RED}1.00x${NC} — 一个不剩)"
echo -e "  ${CYAN}单TPS:${NC}    ~48 tok/s"
echo -e "  ${RED}⚠ 首次启动:${NC} torch.compile 临时吃 ~2G → 失败; 重启即正常"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 73728 \
    --max-num-seqs 1 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
