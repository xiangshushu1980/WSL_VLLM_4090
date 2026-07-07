#!/bin/bash
# 测试不同上下文长度对内存和TPS的影响
# 用法: bash test_context_scaling.sh

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
BENCH_SCRIPT="/home/sean/projects/vllm/benchmark_optimizations.py"
WSL_IP="192.168.31.46"
RESULTS="/home/sean/projects/vllm/docs/context_scaling_results.md"

# 测试矩阵: "name|max_model_len|max_num_seqs"
CONFIGS=(
    "F_8K_2seqs|8192|2"
    "G_16K_1seq|16384|1"
    "H_24K_1seq|24576|1"
    "I_32K_1seq|32768|1"
)

echo "# 上下文长度 TPS 测试结果" > "$RESULTS"
echo "> $(date)" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| 配置 | 上下文 | 并发 | 启动? | CUDA graph | KV cache(池) | KV tokens | 单TPS | TTFT | 1并发吞吐 | 2并发吞吐 | 扩展比 |" >> "$RESULTS"
echo "|------|--------|------|-------|-----------|------------|-----------|-------|------|----------|----------|--------|" >> "$RESULTS"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name max_len max_seqs <<< "$cfg"

    echo ""
    echo "=========================================="
    echo "  测试: $name (ctx=$max_len, seqs=$max_seqs)"
    echo "=========================================="

    # 杀旧服务
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
    sleep 3

    # 启动新服务
    LOGFILE="/tmp/vllm_ctx_test_${name}.log"
    echo "  启动服务 (log: $LOGFILE)..."
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
        > "$LOGFILE" 2>&1 &

    SERVER_PID=$!

    # 等待服务就绪 (可能因为 torch.compile 缓存需要几分钟)
    READY=0
    for i in $(seq 1 60); do
        sleep 5
        if curl -s --max-time 3 "http://${WSL_IP}:8000/health" > /dev/null 2>&1; then
            echo "  服务就绪 (${i}x5s = $((i * 5))s)"
            READY=1
            break
        fi
        # 检查是否已经失败
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ❌ 服务进程已退出"
            break
        fi
    done

    if [ $READY -eq 0 ]; then
        echo "  ❌ 启动超时或失败"
        # 提取错误信息
        ERROR_MSG=$(grep -E "(ValueError|OOM|out of memory|CUDA error|RuntimeError)" "$LOGFILE" | tail -1 | cut -c1-200)
        echo "| $name | $max_len | $max_seqs | ❌ 失败 | - | - | - | - | - | - | - | - |" >> "$RESULTS"
        echo "  错误: $ERROR_MSG"

        # 如果失败是因为KV cache不够，后面的更大上下文肯定也不够
        if echo "$ERROR_MSG" | grep -q "KV cache"; then
            echo "  → KV cache 不足，跳过更大上下文"
            kill -9 $SERVER_PID 2>/dev/null
            break
        fi
        kill -9 $SERVER_PID 2>/dev/null
        continue
    fi

    sleep 3  # 稳定

    # 提取服务器端的内存数据
    CUDA_GRAPH=$(grep "Estimated CUDA graph memory" "$LOGFILE" | grep -oP '[\d.]+(?= GiB total)' | head -1)
    KV_CACHE_MEM=$(grep "Available KV cache memory" "$LOGFILE" | grep -oP '[\d.]+(?= GiB)' | head -1)
    KV_TOKENS=$(grep "GPU KV cache size" "$LOGFILE" | grep -oP '[\d,]+(?= tokens)' | sed 's/,//g' | head -1)

    echo "  内存: CUDA graph=${CUDA_GRAPH}GiB, KV cache=${KV_CACHE_MEM}GiB, KV tokens=${KV_TOKENS}"

    # 单请求测试
    echo "  运行单请求基准..."
    SINGLE_OUT=$(python "$BENCH_SCRIPT" single 2>&1)
    echo "$SINGLE_OUT" | grep -E "热身后|tok/s|TTFT" || echo "  (输出: $SINGLE_OUT)"

    TPS=$(echo "$SINGLE_OUT" | grep -oP '[\d.]+(?= tok/s)' | head -1)
    TTFT=$(echo "$SINGLE_OUT" | grep -oP 'TTFT [\d.]+' | grep -oP '[\d.]+' | head -1)
    TPS=${TPS:-"N/A"}
    TTFT=${TTFT:-"N/A"}

    # 并发测试
    echo "  运行并发测试..."
    CONC_OUT=$(python "$BENCH_SCRIPT" concurrent "$max_seqs" 2>&1)
    echo "$CONC_OUT" | grep -E "并发:|吞吐" || echo "  (并发测试完成)"

    T1=$(echo "$CONC_OUT" | grep "1并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")
    T2=$(echo "$CONC_OUT" | grep "2并发" | grep -oP '吞吐 [\d.]+' | grep -oP '[\d.]+' | head -1 || echo "-")

    # 计算扩展比
    if [ "$T1" != "-" ] && [ "$T2" != "-" ] && [ "$(echo "$T1 > 0" | bc -l 2>/dev/null)" = "1" ]; then
        SCALING=$(echo "scale=1; $T2 / $T1" | bc -l 2>/dev/null || echo "-")
    elif [ "$max_seqs" -eq 1 ]; then
        SCALING="N/A (单并发)"
    else
        SCALING="-"
    fi

    echo "| $name | $max_len | $max_seqs | ✅ | ${CUDA_GRAPH:-?} GiB | ${KV_CACHE_MEM:-?} GiB | ${KV_TOKENS:-?} | ${TPS} | ${TTFT}s | ${T1} | ${T2} | ${SCALING}x |" >> "$RESULTS"

    echo "  结果: TPS=$TPS, TTFT=${TTFT}s, 1并发=$T1, 2并发=$T2, 扩展比=${SCALING}x"
    echo ""

    # 保存日志路径供后续查看
    echo "  完整日志: $LOGFILE" >> "$RESULTS"
    echo "" >> "$RESULTS"

    kill -9 $SERVER_PID 2>/dev/null
    sleep 3
done

# 汇总
echo "" >> "$RESULTS"
echo "## 汇总分析" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "*测试时间: $(date)*" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "完整启动日志保存在 /tmp/vllm_ctx_test_*.log" >> "$RESULTS"

echo ""
echo "=========================================="
echo "  全部测试完成！结果见: $RESULTS"
echo "=========================================="
cat "$RESULTS"
