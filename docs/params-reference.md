# vLLM 常用参数速览

> RTX 4090 24GB 单卡参考配置 | vLLM 0.23.0

---

## 一、启动命令速查

```bash
vllm serve /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --quantization awq \
    --kv-cache-dtype fp8
```

## 二、核心参数分类

### 🖥️ 显存控制（最重要）

| 参数 | 默认 | 推荐 | 说明 |
|------|------|------|------|
| `--gpu-memory-utilization` | 0.90 | **0.85-0.95** | GPU 显存分配比例。AWQ 模型可推到 0.90；FP8 可能需降到 0.80 |
| `--max-model-len` | 自动 | **4096-32768** | 最大上下文长度。越长 = KV cache 显存越多 |
| `--max-num-seqs` | 256 | **1-8** | 最大并发序列数。每增加 1 并发 ≈ 增加 KB 级别显存 |
| `--kv-cache-dtype` | auto | **fp8** | KV 缓存精度。fp8 = 省一半显存，几乎无损质量 |
| `--enforce-eager` | 关 | **MoE模型严禁开** | 跳过 CUDA Graph + torch.compile。Dense模型省 ~0.5 GB 但降 10-20% 速度；**MoE 模型降速 12x（133→10 tok/s）** |
| `--language-model-only` | 关 | **文生文必开** | 不加载视觉编码器，省大量显存 |
| `--cpu-offload-gb` | 0 | **0（不推荐）** | CPU offload MoE 专家权重。**与 torch.compile 和 Marlin MoE kernel 均不兼容，会导致崩溃** |

### 📐 模型加载

| 参数 | 说明 |
|------|------|
| `--quantization` | 量化方式：`awq` / `gptq` / `fp8` / `bitsandbytes` |
| `--dtype` | 模型精度：`auto` 自动检测，手动可设 `float16` |
| `--tensor-parallel-size` | 张量并行数。单卡=1，双卡=2 |
| `--trust-remote-code` | 允许加载自定义模型代码（Qwen 系列必须） |
| `--served-model-name` | API 暴露的模型名，可自定义别名 |

### 🚀 推理优化

| 参数 | 默认 | 推荐 | 说明 |
|------|------|------|------|
| `--enable-prefix-caching` | 关 | 多轮对话可开 | 缓存相同前缀的 KV，省计算。加少量显存 |
| `--enable-chunked-prefill` | 开(v1) | 保持默认 | 分块预填充，避免长 prompt 卡顿 |
| `--max-num-batched-tokens` | 自动 | 保持默认 | 每批次最大 token 数 |
| `--speculative-config` | 关 | **27B+单卡关** | 推测解码，省时但费显存 |

### 🔧 高级参数

| 参数 | 说明 |
|------|------|
| `--compilation-config` | JSON 覆盖编译选项。例 `'{"cudagraph_mode": 0}'` 禁用 CUDA graph 但保留 torch.compile |
| `--reasoning-parser` | 推理解析器，Qwen3 用 `qwen3`（必须！），DeepSeek 用 `deepseek_r1` |
| `--tool-call-parser` | 工具调用解析器，Qwen3 Coder 用 `qwen3_coder` |

## 三、环境变量

| 变量 | 用途 |
|------|------|
| `CUDA_VISIBLE_DEVICES=0` | 指定使用的 GPU |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | WSL 必须设置 |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | CUDA 13 兼容问题/省显存时禁用 flashinfer 采样 |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN` | 强制 Triton 注意力（CUDA 13 flashinfer 不兼容时） |

## 四、24GB 显存预算速算

```
可用显存 = 24GB × gpu_memory_utilization
         = 24 × 0.935 = 22.4 GB (实测天花板)

消耗 = 模型权重 + KV Cache + CUDA Graph + 框架开销

模型权重:
  FP16:   参数×2    (27B → 54GB ❌)
  AWQ 4-bit: 实测 (27B Dense → 19.0 GiB, 35B MoE → 20.3 GiB)

KV Cache (per token, fp8):
  = 2 × num_full_attention_layers × num_kv_heads × head_dim
  Qwen3.6-27B  (16/64层全注意力): ~73.8 KB/token
  Qwen3.6-35B  (10/40层全注意力): ~12.3 KB/token

CUDA Graph (FULL_AND_PIECEWISE):
  27B @max_seqs=4: ~0.48 GiB
  35B @max_seqs=2: ~0.46 GiB

框架开销 (PyTorch + CUDA Context): ~1.12 GiB (固定，不可压缩)
```

### 35B-A3B 快速参考

| 配置 | 上下文 | 并发 | 吞吐 | 备注 |
|------|--------|------|------|------|
| **推荐** | 32,768 | 2 | 355.8 tok/s | 长上下文最优 |
| 平衡 | 16,384 | 3 | ~500 tok/s | 中等上下文高并发 |
| 极限 | 32,768 | 3 | ~500 tok/s | 接近 OOM 边缘 |

> ⚠️ 35B-A3B 首次启动后**必须重启一次**：torch.compile 编译占用临时显存→KV cache 从 0.53 GiB 缩水到启动后恢复的 1.22 GiB。详见 [build-log.md §11](build-log.md#11-qwen36-35b-a3b-moe-部署)

### 27B 快速参考

| 配置 | 上下文 | 并发 | 吞吐 | 备注 |
|------|--------|------|------|------|
| 生产 | 4,096 | 4 | 167.6 tok/s | 高并发短上下文 |
| 平衡 | 8,192 | 2 | 83.7 tok/s | 中等上下文 |
| 长上下文 | 32,768 | 2 | 87.0 tok/s | 长文档单请求 |
```

## 五、常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| OOM 加载阶段 | 模型权重 > 显存 | 换更小量化（FP8→AWQ）或更小模型 |
| OOM 推理阶段 | KV cache 爆炸 | 降 `--max-model-len` 或降并发 |
| flashinfer 编译失败 | CCCL/CUDA 版本不兼容 | 见 [build-log.md §3](build-log.md#3-flashinfer--cuda-13-兼容修复) |
| localhost 连不上 | WSL2 网络隔离 | 用 `hostname -I` 获取 WSL IP |
| 35B-A3B 首启 KV cache 小 | torch.compile 占用临时显存 | 重启服务（缓存命中后恢复） |

