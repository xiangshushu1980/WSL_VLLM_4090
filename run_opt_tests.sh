#!/bin/bash
# 自动测试不同 上下文/并发/PrefixCaching 组合的 TPS
# 用法: bash run_opt_tests.sh

source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/home/sean/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/sean/miniconda3/envs/vllm/bin:$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++

MODEL="/home/sean/projects/vllm/models/Qwen/Qwen3.6-27B-AWQ"
BENCH="/home/sean/projects/vllm/benchmark_optimizations.py"
RESULTS_FILE="/home/sean/projects/vllm/docs/optimization_results.md"
WSL_IP="192.168.31.46"

echo "# 优化配置 TPS 测试结果" > "$RESULTS_FILE"
echo "> $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# 测试矩阵: "name|max_model_len|max_num_seqs|extra_flags"
CONFIGS=(
    "A_baseline_4096_4seqs|4096|4|"
    "B_reduced_2048_2seqs|2048|2|"
    "C_prefix_2048_2seqs|2048|2|--enable-prefix-caching"
    "D_medium_3072_3seqs|3072|3|"
    "E_context_2048_4seqs|2048|4|"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name max_len max_seqs extra <<< "$cfg"

    echo ""
    echo "=========================================="
    echo "  测试: $name"
    echo "  max_model_len=$max_len max_num_seqs=$max_seqs"
    echo "  extra: $extra"
    echo "=========================================="

    # 杀掉旧服务
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
    sleep 3

    # 启动新服务
    echo "启动服务..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name qwen3.6-27b \
        --host 0.0.0.0 --port 8000 \
        --quantization awq_marlin \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.935 \
        --max-model-len "$max_len" \
        --max-num-seqs "$max_seqs" \
        --kv-cache-dtype fp8 \
        --trust-remote-code \
        --language-model-only \
        --reasoning-parser qwen3 \
        $extra \
        > /tmp/vllm_test_${name}.log 2>&1 &

    # 等待服务就绪
    for i in $(seq 1 40); do
        sleep 5
        if curl -s --max-time 3 "http://${WSL_IP}:8000/health" > /dev/null 2>&1; then
            echo "  服务就绪 (${i}x5s)"
            break
        fi
        if [ $i -eq 40 ]; then
            echo "  ❌ 启动超时或失败"
            tail -20 /tmp/vllm_test_${name}.log | grep -i "error" || true
            echo "| $name | ❌ 失败 | - | - | - |" >> "$RESULTS_FILE"
            continue 2
        fi
    done

    sleep 3  # 等待稳定

    # 运行基准测试
    echo "  运行单请求测试..."
    SINGLE_RESULT=$(python "$BENCH" single "$max_seqs" 2>&1 | grep "热身后" || echo "失败")
    echo "  $SINGLE_RESULT"

    # 提取 TPS
    TPS=$(echo "$SINGLE_RESULT" | grep -oP '[\d.]+(?= tok/s)' | head -1)
    TTFT=$(echo "$SINGLE_RESULT" | grep -oP 'TTFT [\d.]+' | grep -oP '[\d.]+' | head -1)

    if [ -z "$TPS" ]; then
        TPS="N/A"
        TTFT="N/A"
    fi

    echo ""
    echo "  运行并发测试..."
    CONC_RESULT=$(python "$BENCH" concurrent "$max_seqs" 2>&1 | grep "并发:" || echo "失败")
    echo "$CONC_RESULT"

    # 提取各并发吞吐
    T1=$(echo "$CONC_RESULT" | grep "1并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")
    T2=$(echo "$CONC_RESULT" | grep "2并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")
    T3=$(echo "$CONC_RESULT" | grep "3并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")
    T4=$(echo "$CONC_RESULT" | grep "4并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")

    # 记录结果
    echo "| **$name** | $max_len | $max_seqs | ${extra:-无} | ${TPS} | ${T1} | ${T2} | ${T3} | ${T4} |" >> "$RESULTS_FILE"

    echo "  完成: TPS=$TPS, 并发=$T1/$T2/$T3/$T4"
    echo ""
done

# 汇总表
echo "" >> "$RESULTS_FILE"
echo "## 汇总" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| 配置 | 上下文 | 并发数 | 优化 | 单TPS | 1并发吞吐 | 2并发吞吐 | 3并发吞吐 | 4并发吞吐 |" >> "$RESULTS_FILE"
echo "|------|--------|--------|------|-------|----------|----------|----------|----------|" >> "$RESULTS_FILE"
# 标记今天的日期方便后续查看
echo "" >> "$RESULTS_FILE"
echo "*测试时间: $(date)*" >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "  全部测试完成！结果见: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
