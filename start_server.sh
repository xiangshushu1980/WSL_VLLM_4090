#!/bin/bash
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/home/sean/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/sean/miniconda3/envs/vllm/bin:$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

# Ensure JIT compilation toolchain is available
ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-gcc /home/sean/miniconda3/envs/vllm/bin/gcc 2>/dev/null
ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-g++ /home/sean/miniconda3/envs/vllm/bin/g++ 2>/dev/null
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++

MODEL_PATH="/home/sean/projects/vllm/models/Qwen/Qwen3.6-27B-AWQ"

echo "启动vLLM OpenAI兼容服务器..."
echo "模型: Qwen3.6-27B-AWQ (4-bit)"
echo "模型路径: $MODEL_PATH"
echo "GPU: RTX 4090 (24GB)"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq_marlin \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.935 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser qwen3
