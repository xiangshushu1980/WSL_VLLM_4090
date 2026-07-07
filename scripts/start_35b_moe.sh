#!/bin/bash
# ============================================================
# Qwen3.6-35B-A3B-AWQ — MoE 最佳配置
# ============================================================
# 场景: 最佳性价比 / 推荐日常使用
# 架构: MoE, 35B 总参数, 仅 3B 激活/token
# 上下文: 32,768 tokens
# 并发:   2 seqs
# 吞吐:   ~355.8 tok/s 总吞吐（实测，vs 27B 的 ~87 tok/s）
# 显存:   模型 20.3G + CUDA Graph 0.46G + KV cache 1.22G
# ============================================================
# 优势: 仅 10/40 层有传统 KV cache → per-token KV 仅 ~12 KB
#       vs 27B Dense 的 ~74 KB → 长上下文并发能力远超 27B
# ============================================================
# 注意: 首次启动后务必重启一次！
#       torch.compile 编译用临时显存 → 首启 KV cache 仅 0.53 GiB
#       重启后恢复至 1.22 GiB，32K 并发从 1.37x → 3.21x
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen3.6-35B-A3B-AWQ"
LOG_FILE="/tmp/vllm_35b_moe.log"

check_running

print_header "Qwen3.6-35B-A3B-AWQ · MoE 最佳配置"
print_info "Qwen3.6-35B-A3B-AWQ (AWQ 4-bit, MoE)" "$PORT" "$LOG_FILE"
echo -e "${YELLOW}架构:${NC}   MoE 256 experts | top-8 | 35B total / 3B active"
echo -e "${YELLOW}配置:${NC}   32,768 上下文 | 2 并发 | CUDA Graph ON"
echo -e "${YELLOW}预期:${NC}   ~356 tok/s 总吞吐 | ~133 tok/s 单请求（碾压 27B 的 48 tok/s）"
echo -e "${RED}⚠ 重要:${NC}   首次启动后请重启一次（torch.compile 缓存陷阱）"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-35b-a3b \
    --port "$PORT" \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_FILE"
