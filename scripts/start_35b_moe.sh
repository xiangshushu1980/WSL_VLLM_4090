#!/bin/bash
# ============================================================
# Qwen3.6-35B-A3B-AWQ · MoE 最佳 (32Kctx × 2seqs = 356 tok/s)
# ============================================================
# MoE: 35B 总参 / 3B 激活, 仅10/40层KV cache → per-token仅12KB (vs 27B的74KB)
# KV池 1.22 GiB, 满ctx并发 3.21x
# 首次启动后务必重启 (torch.compile 缓存陷阱)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen3.6-35B-A3B-AWQ"
LOG_FILE="/tmp/vllm_35b_moe.log"

check_running

print_header "Qwen3.6-35B-A3B · 32Kctx×2并发 · 356 tok/s 🏆"
print_info "Qwen3.6-35B-A3B-AWQ (AWQ 4-bit, MoE)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}架构:${NC}     MoE 256 experts | 35B总参 / 3B激活"
echo -e "  ${CYAN}上下文:${NC}   32,768 tokens (per seq)"
echo -e "  ${CYAN}并发数:${NC}   2 seqs (KV池 1.22 GiB, 满ctx并发 3.21x — 宽裕)"
echo -e "  ${CYAN}总吞吐:${NC}   355.8 tok/s (实测) — ${GREEN}vs 27B@32K: 4.1x${NC}"
echo -e "  ${CYAN}单TPS:${NC}    ~133 tok/s"
echo -e "  ${RED}⚠ 重要:${NC}   首次启动后重启一次 (torch.compile 缓存)"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-35b-a3b \
    --port "$PORT" \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
