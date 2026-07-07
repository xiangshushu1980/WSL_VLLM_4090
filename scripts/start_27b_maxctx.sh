#!/bin/bash
# ============================================================
# Qwen3.6-27B-AWQ — 极限上下文配置
# ============================================================
# 场景: 超长文档 / 全书处理 / 极限单请求
# 上下文: 73,728 tokens（物理极限）
# 并发:   1 seq
# 吞吐:   ~48 tok/s
# 显存:   模型 19.0G + CUDA Graph 0.40G + KV cache 2.48G
# ============================================================
# 注意: 首次启动会因 torch.compile 临时占用 ~2G 额外显存
#       而失败。重启（缓存命中）即可正常。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen/Qwen3.6-27B-AWQ"
LOG_FILE="/tmp/vllm_27b_maxctx.log"

check_running

print_header "Qwen3.6-27B-AWQ · 极限上下文配置"
print_info "Qwen3.6-27B-AWQ (4-bit)" "$PORT" "$LOG_FILE"
echo -e "${RED}⚠ 极限配置:${NC} 73,728 上下文 | 1 并发 | CUDA Graph ON"
echo -e "${RED}⚠ 首次启动:${NC} 可能因 torch.compile 临时显存不足而失败"
echo -e "${RED}⚠ 解决方案:${NC} 失败后再次运行此脚本即可（缓存命中）"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --port "$PORT" \
    --max-model-len 73728 \
    --max-num-seqs 1 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
