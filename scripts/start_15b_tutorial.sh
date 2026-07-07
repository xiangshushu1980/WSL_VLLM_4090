#!/bin/bash
# ============================================================
# Qwen2.5-1.5B-Instruct — 教程/快速测试配置
# ============================================================
# 场景: 学习 vLLM / 快速原型 / API 开发测试
# 模型: Qwen2.5-1.5B-Instruct (~3GB, 无需量化)
# 上下文: 32,768 tokens（默认）
# 并发:   默认
# 显存:   仅 ~6-8 GB, 余量极大
# ============================================================
# 特点:
#   - 极小模型, 秒级启动
#   - 不需要量化, 不需要 AWQ
#   - 可以随意测试参数
#   - 适合跑 step3/step4 教程脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PORT=8000
MODEL_PATH="$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
LOG_FILE="/tmp/vllm_15b_tutorial.log"

check_running

print_header "Qwen2.5-1.5B-Instruct · 教程/测试配置"
print_info "Qwen2.5-1.5B-Instruct (FP16, ~3GB)" "$PORT" "$LOG_FILE"
echo -e "${YELLOW}配置:${NC}   默认上下文 | 默认并发 | 无量化"
echo -e "${YELLOW}用途:${NC}   学习 vLLM / API 开发 / 快速测试"
echo -e "${GREEN}教程:${NC}   python scripts/step3-streaming.py"
echo -e "${GREEN}      ${NC}   python scripts/step4-benchmark.py"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen2.5-1.5b \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    2>&1 | tee "$LOG_FILE"
