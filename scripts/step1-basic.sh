#!/bin/bash
# 第1步: 最简启动 — 验证 vLLM 能跑通
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# 先下载小模型（只需跑一次）
MODEL_DIR="./models/Qwen2.5-1.5B-Instruct"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "下载 Qwen2.5-1.5B-Instruct..."
    mkdir -p "$MODEL_DIR"
    hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$MODEL_DIR"
fi

echo "启动 vLLM (最简配置)..."
echo "模型: Qwen2.5-1.5B-Instruct (~3GB)"
echo "端口: 8000"
echo ""

vllm serve "$MODEL_DIR" --host 0.0.0.0 --port 8000
