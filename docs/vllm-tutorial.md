# vLLM 新手教程 — 从零到生产

> 适用: RTX 4090 24GB + vLLM 0.23.0 + WSL2  
> 配套脚本: `scripts/` 目录，每步有对应脚本  
> 速查: 参数说明见 [params-reference.md](params-reference.md)  
> 社区: 实战经验见 [rtx4090-guide.md](rtx4090-guide.md)

---

## 路线图

```
第1步: 安装验证 → 小模型跑通 (5分钟)
第2步: 理解核心参数 → 逐步调参 (10分钟)
第3步: 流式输出 → 理解 TTFT/TPOT (5分钟)
第4步: 并发测试 → 理解吞吐量 (10分钟)
第5步: 生产配置 → 27B-AWQ 实战 (15分钟)
```

---

## 第1步: 安装验证 — 跑通第一个请求

### 1.1 检查环境

```bash
# 确认 GPU 可用
nvidia-smi

# 确认 vLLM 已安装
python -c "import vllm; print(vllm.__version__)"
```

### 1.2 安装一个小模型快速验证

> **为什么用小模型？** Qwen2.5-1.5B 只有 ~3GB，加载 5 秒，改参数秒出效果。

```bash
# 下载 Qwen2.5-1.5B-Instruct (约 3GB)
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

# 或用 ModelScope（国内更快）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='models/Qwen2.5-1.5B-Instruct')"
```

### 1.3 启动服务器（最简配置）

```bash
# scripts/step1-basic.sh
vllm serve models/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 8000
```

### 1.4 发送第一个请求

```bash
curl http://<WSL_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "你好，什么是vLLM？"}],
    "max_tokens": 128
  }'
```

> **验证成功标志**: 返回 JSON 包含 `choices[0].message.content`

---

## 第2步: 理解核心参数 — 逐步加参数观察效果

> 每轮只改 **一个参数**，观察服务日志的变化。

### 2.1 控制显存 — `--gpu-memory-utilization`

```bash
# 脚本: scripts/step2-memory.sh
# 轮1: 默认 0.90 — 看启动日志的 "Model loading took X GiB"
# 轮2: 改为 0.70 — 剩余更多给 KV cache
# 轮3: 改为 0.95 — 极限压榨（可能 OOM）

vllm serve models/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.70  # 改这里的值
```

**观察点**: 服务日志中 `Model loading took X GiB memory` 这一行

### 2.2 控制上下文长度 — `--max-model-len`

```bash
# 脚本: scripts/step2-context.sh
# 轮1: 512   — 长文本请求会截断
# 轮2: 4096  — 适中
# 轮3: 32768 — 小模型也能支持长上下文，但 KV cache 增大

vllm serve models/Qwen2.5-1.5B-Instruct \
    --max-model-len 512  # 逐步增大
```

**验证**: 发送一个超过 `max-model-len` 的长 prompt，观察是否截断

### 2.3 控制并发 — `--max-num-seqs`

```bash
# 脚本: scripts/step2-concurrency.sh
# 轮1: --max-num-seqs 1   → 同时发 4 个请求，只有 1 个在处理
# 轮2: --max-num-seqs 4   → 4 个请求同时处理

vllm serve models/Qwen2.5-1.5B-Instruct \
    --max-num-seqs 1  # 逐步增大
```

**验证**: `python scripts/test_concurrent.py` 看 4 请求的完成时间差异

### 2.4 KV Cache 精度 — `--kv-cache-dtype`

```bash
# 轮1: 不设 (默认 auto) → 看显存
# 轮2: --kv-cache-dtype fp8 → 省一半 KV 显存，几乎无损

vllm serve models/Qwen2.5-1.5B-Instruct \
    --kv-cache-dtype fp8
```

### 2.5 CUDA Graph — 默认开启即可

```bash
# CUDA graph 默认开启，不需要额外参数
# 预捕获不同 batch size 的 GPU 运算图
# 27B Dense: 加速 1.3x | 35B MoE: 加速 12x (关键!)
# 开销: ~0.46 GiB 显存

# 除非显存实在不够，否则不要加 --enforce-eager
# --enforce-eager 会同时禁用 CUDA graph + torch.compile
```

---

## 第3步: 流式输出 — 理解响应指标

### 3.1 启动服务器（任何模型都行）

```bash
vllm serve models/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```

### 3.2 运行流式测试脚本

```bash
# 脚本: scripts/step3-streaming.py
python scripts/step3-streaming.py
```

**学习的关键指标:**

| 指标 | 全称 | 含义 |
|------|------|------|
| **TTFT** | Time To First Token | 首 token 延迟：用户体验的核心 |
| **TPOT** | Time Per Output Token | 每个 token 的生成间隔 |
| **TPS** | Tokens Per Second | 单个请求的生成速度 |
| **Throughput** | 总吞吐量 | 所有请求每秒生成的总 token 数 |

### 3.3 对比实验

```
修改 --max-num-seqs 1 → 4，重跑 step3-streaming.py
观察：吞吐量变化，TTFT 变化
```

---

## 第4步: 并发测试 — 理解 vLLM 的核心优势

### 4.1 启动服务器

```bash
vllm serve models/Qwen2.5-1.5B-Instruct \
    --max-num-seqs 8 \
    --host 0.0.0.0 --port 8000
```

### 4.2 跑不同并发数的对比

```bash
# 脚本: scripts/step4-benchmark.py
# 自动跑 1, 2, 4, 8 并发对比
python scripts/step4-benchmark.py
```

**预期观察:**
- 1 并发: TTFT 最低，总吞吐最低
- 4 并发: TTFT 略升，总吞吐 ×3-4
- 8 并发: TTFT 明显上升（排队效应）

### 4.3 理解 PagedAttention

vLLM 的核心技术：KV cache 按"页"管理（类似 OS 的虚拟内存），避免碎片化。
- 传统方法: KV cache 浪费 60-80%
- PagedAttention: 浪费 <4%

**实际效果**: 同样的显存，vLLM 能同时处理的请求数是 Ollama 的 5-10 倍。

---

## 第5步: 生产配置 — 27B-AWQ 实战

### 5.1 下载模型

```bash
# 国内用 ModelScope（快）
python -c "
from modelscope import snapshot_download
snapshot_download('QuantTrio/Qwen3.6-27B-AWQ',
                  cache_dir='models/Qwen/Qwen3.6-27B-AWQ')
"
# 下载后移动到正确位置（ModelScope 会嵌套目录）
```

### 5.2 生产级启动配置

**方案 A: 27B Dense — 高并发短上下文** (167.6 tok/s @4并发)

```bash
# 脚本: start_server.sh（项目根目录）
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen/Qwen3.6-27B-AWQ \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 --port 8000 \
    --quantization awq_marlin \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.935 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser qwen3
```

**方案 B: 35B-A3B MoE — 长上下文** (355.8 tok/s @32K/2并发) 🔥推荐

```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen3.6-35B-A3B-AWQ \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 --port 8000 \
    --quantization awq_marlin \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.935 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser qwen3
```

> ⚠️ 35B-A3B 首次启动后需**重启一次**：torch.compile 编译用临时显存 → KV cache 缩水。缓存命中后恢复正常。

### 5.3 参数解读

```
--quantization awq_marlin（不是 awq！）
  → awq_marlin 用 Marlin 专用 kernel，比 awq 快 7.7x

--gpu-memory-utilization 0.935
  → RTX 4090 WSL2 实测天花板（PyTorch 固定吞 1.12 GiB）
  → 分配 22.43 GiB 给 vLLM

--max-model-len 4096 / 32768
  → 27B: 4096 → 4 并发 @ 167.6 tok/s
  → 35B: 32768 → 2 并发 @ 355.8 tok/s

--max-num-seqs 4 / 2
  → 最多同时处理 N 个请求
  → 超出排队的由 continuous batching 自动调度

--kv-cache-dtype fp8
  → KV 缓存用 fp8，省一半显存

--language-model-only
  → 不加载视觉 encoder

--reasoning-parser qwen3
  → Qwen3.6 默认生成 thinking token
  → 解析器将它们从 content 分离到 reasoning 字段

# 注意：不要加 --enforce-eager！
# MoE 模型关了 CUDA graph 速度跌 12x (133→10 tok/s)
```

### 5.4 验证

```bash
# 健康检查
curl http://<WSL_IP>:8000/v1/models

# 4 并发压力测试
python test_concurrent.py
```

---

## 附录: 故障排查

### OOM 类

| 症状 | 检查 | 调整 |
|------|------|------|
| 加载阶段 OOM | `du -sh models/` | 模型是否太大？换更小量化 |
| 首次请求 OOM | 启动日志 `Model loading took X GiB` | 降低 `--gpu-memory-utilization` |
| 并发请求 OOM | KV cache 爆炸 | 降 `--max-model-len` 或降并发 |

### 编译类

| 症状 | 检查 | 调整 |
|------|------|------|
| `flashinfer: nvcc not found` | `which nvcc` | 设置 `CUDA_HOME` 和 `PATH` |
| `CCCL incompatible` | flashinfer 版本 < 0.6.13 | 升级 + 补丁，或设 `VLLM_USE_FLASHINFER_SAMPLER=0` |

### 网络类

| 症状 | 检查 | 调整 |
|------|------|------|
| `localhost` 连不上 | `ss -tlnp | grep 8000` | WSL2 用 `hostname -I` 获取 IP |
| curl 卡住 | 服务器日志最后几行 | 可能首次请求触发 JIT 编译，等 30 秒 |

---

## 下一步学习

1. **35B-A3B MoE**: 更大的上下文（32K），更快的速度（356 tok/s），见 [build-log.md §11](build-log.md#11-qwen36-35b-a3b-moe-部署)
2. **工具调用**: `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice`
3. **投机解码 (需要 48GB)**: `--speculative-config` 加速 1.5-2.5x（额外 ~1 GB 显存）
4. **多 GPU**: `--tensor-parallel-size 2` 双卡并行
5. **上下文极限**: 见 [context_scaling_results.md](context_scaling_results.md)
