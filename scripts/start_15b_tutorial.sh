#!/bin/bash
# ============================================================
# Qwen2.5-1.5B-Instruct · 教程/快速测试 (~3GB, 无需量化)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
LOG_FILE="/tmp/vllm_15b_tutorial.log"

check_running

print_header "Qwen2.5-1.5B · 默认配置 · 教程/测试"
print_info "Qwen2.5-1.5B-Instruct (FP16, ~3GB)" "$PORT" "$LOG_FILE"
echo -e "  ${CYAN}上下文:${NC}   默认 (~32K)"
echo -e "  ${CYAN}并发数:${NC}   默认 (显存充裕, 随便调)"
echo -e "  ${CYAN}显存占用:${NC} 仅 ~6-8 GB / 24GB"
echo -e "  ${CYAN}教程脚本:${NC} python scripts/step3-streaming.py"
echo -e "             python scripts/step4-benchmark.py"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen2.5-1.5b \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    2>&1 | tee "$LOG_FILE"
