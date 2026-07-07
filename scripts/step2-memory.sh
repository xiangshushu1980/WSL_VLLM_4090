#!/bin/bash
# 第2步: 理解显存参数 — 逐轮改参数观察日志
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

MODEL_DIR="./models/Qwen2.5-1.5B-Instruct"

echo "=== 显存参数学习 ==="
echo "每轮只改一个参数，观察启动日志中的 'Model loading took X GiB'"
echo ""

# 轮1: 默认 gpu-memory-utilization=0.90
echo "--- 轮1: --gpu-memory-utilization 0.90 (默认) ---"
echo "观察: 模型加载用了多少显存？"
echo ""

# 轮2: 降低到 0.70
echo "--- 轮2: --gpu-memory-utilization 0.70 ---"
echo "观察: 和轮1有区别吗？（小模型不在乎这个参数）"
echo ""

# 轮3: 限制上下文长度
echo "--- 轮3: --max-model-len 512 (限制上下文) ---"
echo "观察: 启动日志中 max_seq_len 变为 512"
echo "实验: 发送超过 512 tokens 的长 prompt，会被截断"
echo ""

echo "你可以修改上面参数手动实验。小模型看不出显存差异，"
echo "用 27B-AWQ 时每个参数的变化会很明显。"
echo ""
echo "启动命令（按需修改参数）:"
echo "vllm serve $MODEL_DIR --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.90"
