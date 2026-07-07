# vLLM on RTX 4090 — 完整构建日志

> 持续更新中 | 最后更新: 2026-07-07

---

## 目录

1. [环境概览](#1-环境概览)
2. [安装过程](#2-安装过程)
3. [flashinfer + CUDA 13 兼容修复](#3-flashinfer--cuda-13-兼容修复)
4. [Qwen3.6-27B-AWQ 部署](#4-qwen36-27b-awq-部署)
5. [性能优化历程](#5-性能优化历程)
6. [基准测试结果](#6-基准测试结果)
7. [已知问题与经验教训](#7-已知问题与经验教训)
8. [待探索优化方向](#8-待探索优化方向)
9. [Prefill 阻塞与 Chunked Prefill](#9-prefill-阻塞与-chunked-prefill)
10. [社区基准对比](#10-社区基准对比)
11. [Qwen3.6-35B-A3B MoE 部署](#11-qwen36-35b-a3b-moe-部署)
12. [多配置脚本体系](#12-多配置脚本体系)
13. [项目总结](#13-项目总结)

---

## 1. 环境概览

```
硬件:   NVIDIA GeForce RTX 4090, 24 GB VRAM
OS:     WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2)
Conda:  /home/sean/miniconda3/envs/vllm/
Python: 3.11
vLLM:   0.23.0
CUDA:   13.3 (cu13, pip package)
GPU驱动: 610.43.02 (WSL2)
```

**WSL2 注意事项:**
- `localhost`/`127.0.0.1` 不走 WSL2 回环 → 必须用 `hostname -I` 获取的 IP
- 需要 `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 需要 `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` (libcuda.so 在此)
- `pin_memory=False` 自动检测（WSL2 限制）

---

## 2. 安装过程

### 2.1 Conda 环境

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install vllm==0.23.0
```

### 2.2 CUDA 13 环境

vLLM 0.23.0 依赖 nvidia-cu13 系列包，自动安装于：
```
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/
├── bin/nvcc       # CUDA 编译器
├── include/       # CUDA 头文件
├── lib/           # CUDA 库 (libcudart.so.13 等)
└── cccl/          # CCCL 头文件
```

### 2.3 flashinfer 升级

```bash
pip install flashinfer-cubin==0.6.13 flashinfer-python==0.6.13
```

---

## 3. flashinfer + CUDA 13 兼容修复

### 问题描述

flashinfer 0.6.13 内置的 CCCL 头文件与 CUDA 13 的 nvcc 不兼容，JIT 编译时出错：
```
error: "CUDA compiler and CUDA toolkit headers are incompatible"
```

### 根因分析

CCCL (C++ CUDA Core Libraries) 在 `flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h` 中检查 CUDA 版本，CUDA 13 不在其"已知兼容"列表中。

**解决方案：** 在 nvcc 编译参数中加 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` 跳过该检查。

### 修复 1: sm89_nvcc_flags 缺少 common flags

**文件**: `flashinfer/jit/core.py:123`

RTX 4090 是 sm_89 架构，但 `sm89_nvcc_flags` 是独立列表，没有继承 `common_nvcc_flags`（其中包含 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`）。

```python
# 修复前:
sm89_nvcc_flags = [
    "-gencode=arch=compute_89,code=sm_89",
    "-DFLASHINFER_ENABLE_FP8_E8M0",
]

# 修复后:
sm89_nvcc_flags = [
    "-gencode=arch=compute_89,code=sm_89",
] + common_nvcc_flags
```

### 修复 2: CompilationContext 的 COMMON_NVCC_FLAGS 缺失

**文件**: `flashinfer/compilation_context.py:28`

Attention kernel 的 JIT 编译用的是 `CompilationContext.get_nvcc_flags_list()`，它有自己独立的 `COMMON_NVCC_FLAGS`，也缺少 CCCL 标志。

```python
# 修复前:
COMMON_NVCC_FLAGS = [
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
]

# 修复后:
COMMON_NVCC_FLAGS = [
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
]
```

### 修复 3: 链接器用 lib64 但 nvidia-cu13 wheel 用 lib/

**文件**: `flashinfer/jit/cpp_ext.py:254`

```python
# 修复前:
ldflags = ["-shared", "-L$cuda_home/lib64", "-L$cuda_home/lib64/stubs", "-lcudart", "-lcuda"]

# 修复后:
ldflags = ["-shared", "-L$cuda_home/lib", "-L/usr/lib/wsl/lib", "-lcudart", "-lcuda"]
```

### 修复 4: 缺失 .so symlink

nvidia-cu13 wheel 只带版本化的 `.so` 文件，缺少无版本后缀的 symlink：

```bash
ln -sf libcudart.so.13 libcudart.so  # $CONDA_PREFIX/.../nvidia/cu13/lib/
ln -sf /usr/lib/wsl/lib/libcuda.so.1 libcuda.so  # 同上目录
```

### 必须的环境变量

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0  # CUDA 13 不兼容 flashinfer sampler
```

### 解决结果

修复前：服务启动成功但首次推理时 flashinfer JIT 编译失败 → 引擎崩溃
修复后：JIT 编译正常完成，推理成功

---

## 4. Qwen3.6-27B-AWQ 部署

### 模型信息

| 属性 | 值 |
|------|-----|
| 名称 | Qwen3.6-27B-AWQ |
| 来源 | ModelScope: QuantTrio/Qwen3.6-27B-AWQ |
| 大小 | 21 GB (8 个 safetensors 分片) |
| 量化 | AWQ 4-bit |
| 架构 | Dense (非 MoE) |
| 层数 | 64 层 (3×DeltaNet + 1×Full Attention 循环) |
| KV Cache | 仅 16/64 层有传统 KV cache |
| hidden_size | 5120 |
| 推理模式 | 带思维链 (reasoning model) |

### 模型加载详情

```
模型加载: 19.05 GiB VRAM, 44 秒
KV cache: 取决于配置
CUDA graphs: 0.48 GiB (可选)
```

### 脚本文件

**start_server.sh** — 生产启动脚本，含所有环境变量和参数

**test_concurrent.py** — 4 并发压力测试，支持 Qwen3.6 reasoning 模型

---

## 5. 性能优化历程

### 优化前配置 (v1 — baseline)

```bash
--quantization awq       # 慢速 AWQ kernel
--enforce-eager           # 禁用 torch.compile + CUDA graphs
VLLM_ATTENTION_BACKEND=TRITON_ATTN  # v0.23.0 中实际被忽略
--gpu-memory-utilization 0.90
--max-model-len 8192
```

**结果**: 单请求 ~6.2 tok/s

### 瓶颈分析

vLLM 启动日志中明确提示：
```
Detected that the model can run with awq_marlin, however you specified
quantization=awq explicitly, so forcing awq. Use quantization=awq_marlin
for faster inference
```

三个性能杀手：
1. `--quantization awq`：使用未优化的 AWQ kernel
2. `--enforce-eager`：禁用 CUDA graphs → decode 无优化
3. `VLLM_ATTENTION_BACKEND=TRITON_ATTN`：v0.23.0 中已忽略，无用代码

### 优化后配置 (v2 — optimized)

```bash
--quantization awq_marlin  # Marlin INT4 kernel → 大幅提升 GEMM
--gpu-memory-utilization 0.935  # 极限利用 VRAM
--max-model-len 4096       # 给 CUDA graphs 留空间
# 移除了 --enforce-eager
# 移除了 VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

**CUDA graphs 显存代价**：0.48 GiB。需要从 `--max-model-len 8192` 降低到 `4096` 才能腾出空间。

**gpu_memory_utilization 精确调优**：
- 0.90 → KV cache 足够但不用 CUDA graphs（6 tok/s）
- 0.93 → 开启 CUDA graphs 但 KV cache 只有 0.18 GiB（不够 4096 上下文）
- 0.935 → KV cache 2.4 GiB，刚好满足 4096 上下文 ✅

### 性能提升

| 指标 | v1 (awq+eager) | v2 (awq_marlin+CUDA) | 提升 |
|------|---------------|---------------------|------|
| 单请求 TPS | 6.2 tok/s | 48.0 tok/s | **7.7x** |
| 4并发总吞吐 | 24.2 tok/s | 163.1 tok/s | **6.7x** |
| TTFT | 0.46–0.73s | 0.23–0.41s | **~50% 改善** |

---

## 6. 基准测试结果

### 6.1 Qwen2.5-1.5B-Instruct (baseline)

| 并发 | 耗时 | TTFT | 单 TPS | 总吞吐 | 加速比 |
|------|------|------|--------|--------|--------|
| 1 | 2.14s | 0.59s | 161.3 | 116.7 | 1.0x |
| 2 | 1.22s | 0.02s | 211.8 | 362.5 | 3.1x |
| 3 | 1.27s | 0.04s | 203.8 | 506.0 | 4.3x |
| 4 | 1.31s | 0.05s | 201.6 | 672.8 | 5.8x |

### 6.2 Qwen3.6-27B-AWQ (optimized)

| 并发 | 耗时 | TTFT | 单 TPS | 总吞吐 | 加速比 |
|------|------|------|--------|--------|--------|
| 1 | 5.60s | 0.27s | 48.0 | 45.7 | 1.0x |
| 2 | 5.83s | 0.23s | 45.8 | 87.8 | 1.9x |
| 3 | 6.04s | 0.24s | 44.3 | 127.2 | 2.8x |
| 4 | 6.28s | 0.41s | 43.7 | 163.1 | 3.6x |

### 6.3 对比总结

| 并发 | 1.5B 吞吐 | 27B-AWQ 吞吐 | 倍数 |
|------|----------|-------------|------|
| 1 | 116.7 | 45.7 | 2.6x |
| 2 | 362.5 | 87.8 | 4.1x |
| 3 | 506.0 | 127.2 | 4.0x |
| 4 | 672.8 | 163.1 | 4.1x |

---

## 7. 已知问题与经验教训

### 踩过的坑

1. **flashinfer CCCL 兼容**：需要分别在 `core.py` 和 `compilation_context.py` 两处加 CCCL disable flag。只修一处不够。
2. **nvidia-cu13 wheel 用 lib/ 不是 lib64/**：flashinfer 硬编码了 `lib64` 路径。
3. **缺 .so symlink**：wheel 只有 `libcudart.so.13`，`-lcudart` 需要 `libcudart.so`。
4. **localhost 在 WSL2 不通**：必须用实际 IP。
5. **CUDA graphs 需要额外 0.48 GiB**：不是免费的！
6. **Qwen3.6 是 reasoning 模型**：流式输出中 content 来自 `model_extra['reasoning']` 而非 `delta.content`。
7. **gpu_memory_utilization 需精确调优**：0.93 不够，0.94 太，0.935 刚好。

### Qwen3.6 推理模型特殊性

- 流式响应：推理 token 在 `delta.model_extra['reasoning']`，内容在 `delta.content`
- 非流式：`message.reasoning` 包含思维链，`message.content` 包含最终回答
- 需要 `--reasoning-parser qwen3 --trust-remote-code` 两个 flag 同时设置

---

## 8. 待探索优化方向

| 方向 | 描述 | 预期提升 | 状态 |
|------|------|---------|------|
| **MTP** | 多头预测，模型自带 | 2-2.5x | ❌ 24GB不够 |
| N-gram GPU | GPU n-gram 推测 | 1.1-1.3x | ❌ Qwen3.6不兼容(挂起) |
| Prefix Caching | KV cache 复用 | 5-30% | ❌ 哈希表~2GiB开销 (2048/2seqs仍差0.05GiB) |
| flashinfer sampler | flashinfer top-k/top-p | 10-20% | ⏳ 待修复 (CUDA 13兼容) |
| ~~降上下文换优化~~ | 降低上下文腾空间 | — | ❌ 杀死并发扩展 (2048: 1.1x vs 4096: 3.8x) |
| GGUF/llama.cpp | 对比框架 | 20-30% vs AWQ | 单请求更快，但无并发优势 |

### 8.2 上下文/并发对 TPS 影响测试 (2026-07-07)

| 配置 | 上下文 | 并发 | 单TPS | 4并发吞吐 | 扩展比 |
|------|--------|------|-------|----------|--------|
| A baseline | 4096 | 4 | 47.9 | 167.6 | **3.8x** |
| B reduced | 2048 | 2 | 46.9 | — | — |
| D medium | 3072 | 3 | 46.9 | — | — |
| E low-ctx | 2048 | 4 | 48.1 | 47.3 | 1.1x |

**关键结论**：单请求 TPS 不受上下文影响，但并发扩展严重依赖上下文长度。4096 是实现线性扩展的最低要求。

### 8.3 上下文极限测试 (2026-07-07) ⭐ 新增

完整测试了 4K→72K 上下文范围。**核心发现：64K-72K 上下文是可行的，但仅支持单并发。**

#### 实测数据

| 配置 | 上下文 | 最大并发 | CUDA graph | KV cache 池 | 池大小(tokens) | 满上下文并发 | 单 TPS | 2并发吞吐 |
|------|--------|----------|------------|------------|---------------|-------------|-------|----------|
| A | 4,096 | 4 | 0.48 GiB | 2.40 GiB | 34,133 | 8.33x | 47.9 | 88.2 (1.9x) |
| F | 8,192 | 2 | 0.44 GiB | 2.44 GiB | 45,511 | 5.56x | 47.7 | 83.7 (1.8x) |
| G | 16,384 | 1 | 0.40 GiB | 2.48 GiB | 59,684 | 3.64x | 47.9 | — |
| H | 24,576 | 1 | 0.40 GiB | 2.48 GiB | 65,967 | 2.68x | 47.1 | — |
| I (1seq) | 32,768 | 1 | 0.40 GiB | 2.48 GiB | 69,632 | 2.12x | 46.8 | — |
| **I2 (2seqs)** | **32,768** | **2** | **0.44 GiB** | **2.44 GiB** | — | **2.08x** | **47.8** | **87.0 (1.91x)** |
| J | 49,152 | 1 | 0.40 GiB | 2.48 GiB | — | 1.46x | 50.6 | — |
| K | 65,536 | 1 | 0.40 GiB | 2.48 GiB | — | 1.13x | 47.6 | — |
| L | **73,728** | 1 | 0.40 GiB | 2.48 GiB | — | **1.00x** | 47.7 | — |

#### 关键发现

1. **单 TPS 恒定 ~48 tok/s**：上下文长度对 decode 速度无影响（memory-bandwidth bound）。Qwen3.6 的特殊架构（仅 16/64 层有传统 KV cache，其余是 DeltaNet）使得 KV cache 带宽开销即使在 64K 上下文也只占 ~10%。
2. **CUDA graph 随并发数减小而缩小**：max_num_seqs 4→2→1，CUDA graph 从 0.48→0.44→0.40 GiB
3. **KV cache 池随之增长**：2.40→2.44→2.48 GiB（CUDA graph 省下的内存自动分配给 KV cache）
4. **绝对最大上下文：~72K tokens**（正好 1.00x 并发，一个 token 都不浪费）
5. **32K 能跑 2 并发**：满上下文 2.08x，实测 1.91x 扩展比，总吞吐 87.0 tok/s
6. **0.935 是 gpu_memory_utilization 物理天花板**：PyTorch CUDA 上下文字吃掉 1.12 GiB，vLLM 看到 free 22.45 GiB，刚好够 0.935×23.99=22.43，差 0.02 GiB 碰壁。关闭 GameViewer、`expandable_segments`、重启系统均无法突破。
7. **首次编译需 "warmup"**：新上下文长度的首次 torch.compile 需 ~2 GiB 临时 GPU 内存，导致首次启动失败。第二次（缓存命中）即可正常启动

#### 实践建议

```
并发优先:  4096 / 4 seqs  → 167.6 tok/s 总吞吐  (在线服务)
平衡:      8192 / 2 seqs  →  83.7 tok/s           (中等上下文+并发)
32K+并发:  32768 / 2 seqs →  87.0 tok/s           (长上下文+有限并发)
长上下文:  65536 / 1 seq  →  47.6 tok/s           (单请求长文档)
绝对极限:  73728 / 1 seq  →  47.7 tok/s           (刚好 1.00x 并发)
```

### 8.4 gpu_memory_utilization 天花板调查 (2026-07-07)

**问题**: 为什么 0.935 是上限？能否突破到 0.94+？

**空载基线**:
```
Total:      24564 MiB = 23.99 GiB
Reserved:     426 MiB =  0.42 GiB (驱动 framebuffer, 不可动)
Free (idle):24138 MiB = 23.57 GiB ← 理论极限
```

**PyTorch CUDA init 吃掉 1.12 GiB**:
```python
torch.cuda.mem_get_info()  # → free=22.45 GiB, total=23.99 GiB
```

**尝试的优化 (全部失败)**:
| 尝试 | 效果 |
|------|------|
| 关闭 Windows GameViewer.exe | 省 13 MiB, 可忽略 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 无变化 (解决碎片化, 不缩小初始池) |
| 重启 WSL2 + Windows | 无变化 (CUDA 上下文开销恒定) |
| gpu_memory_utilization=0.94 | `ValueError: Free memory 22.45 < 22.55 (0.94×23.99)` |

**根因**: 22.45/23.99 = 0.9358。多要 0.02 GiB 都拿不出来。PyTorch 缓存分配器 + CUDA 驱动 + WSL2 PV 层的 1.12 GiB 开销是计算必需的。

**硬件配置**: AMD R7 430 亮机卡已接显示器，RTX 4090 为纯计算模式 (Disp.A: Off)。显示负载已不在 4090 上。

**结论**: 0.935 是物理极限，无法突破。唯一能躲开这个限制的方式是不用 PyTorch (换 llama.cpp/Vulkan 后端)。

### 8.1 MTP 探索记录

**Qwen3.6-27B-AWQ 自带 MTP 预测头**（`"mtp_num_hidden_layers": 1`，未量化，~850 MiB）：
```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

**社区基准**（RTX 4090, 27B-AWQ, MTP=3）：~45 tok/s (2.4x 加速)

**本机尝试结果**：
| 配置 | 结果 | 原因 |
|------|------|------|
| MTP + CUDA graphs + max_seqs=3 | ❌ KV cache -0.18 GiB | MTP 显存超限 |
| MTP + CUDA graphs + max_seqs=2 | ❌ KV cache -0.18 GiB | 同上 |
| MTP + CPU offload 2GiB + CUDA graphs | ❌ torch.compile 崩溃 | CPU 卸载与 dynamo 不兼容 |

**结论**：RTX 4090 24GB 上，**CUDA graphs（7.7x 加速）> MTP（2.4x 加速）**，且二者无法共存。

---

## 9. Prefill 阻塞与 Chunked Prefill

vLLM v1 默认启用 `--enable-chunked-prefill`：
- 长 prompt 被拆分为多个 chunk，与 decode 步骤交错执行
- 短请求可以穿插在长请求的 chunk 之间，避免阻塞
- 我们的 `max_num_seqs=4` + chunked prefill = 无阻塞风险

## 10. 社区基准对比

| 框架 | 量化 | 单请求 TPS | 并发优势 |
|------|------|-----------|---------|
| vLLM + AWQ Marlin (我们) | AWQ 4-bit | 48 tok/s | ✅ continuous batching (163 tok/s @4并发) |
| llama.cpp (LM Studio) | GGUF Q4_K_M | 62-70 tok/s | ❌ 无 continuous batching |
| vLLM + MTP | AWQ 4-bit | ~45 tok/s | 需更多 VRAM (>24GB) |
| SGLang | AWQ 4-bit | 40-50 tok/s | 类似 vLLM |

**单请求 TPS 天花板**: 27B AWQ 在 4090 上理论 ~52 tok/s (memory-bound)。我们 48 tok/s = 理论 92%，接近极限。

## 11. Qwen3.6-35B-A3B MoE 部署

> 日期: 2026-07-07 | 模型: Chunity/Qwen3.6-35B-A3B-AWQ | 26 GB 磁盘

### 11.1 模型架构

- **MoE (Mixture of Experts)**: 256 experts, top-8 routing
- **35B 总参数 / 3B 激活参数** (每 token 只激活 ~3B)
- **混合注意力**: 10 Full Attention + 30 DeltaNet 层
- **KV cache 层数少**: 仅 10/40 层有传统 KV cache → per-token KV 极小 (~12 KB fp8)
- **量化**: AWQ 4-bit (awq_marlin), group_size=128, AutoRound

### 11.2 关键发现

#### CUDA Graphs 对 MoE 的影响是决定性的

| 配置 | 单请求 TPS | CUDA Graph |
|------|-----------|-------------|
| torch.compile + CUDA graph | **133 tok/s** | ON |
| torch.compile only | 10.0 tok/s | OFF |
| 纯 eager | 7.6 tok/s | OFF |

**CUDA graph 加速 ~13x**，对 MoE 模型至关重要（大量小 kernel launch 被合并为单个 graph）。

#### torch.compile 首次编译陷阱（与 27B 相同）

首次启动需要 ~2 GiB 额外 GPU 内存用于 torch.compile → KV cache 池缩小。重启（缓存命中）后恢复。

| 启动次数 | torch.compile | KV cache | KV tokens | 32K 并发 |
|----------|-------------|----------|-----------|---------|
| 首启 | 46.3s (编译) | 0.53 GiB | 44,840 | 1.37x |
| 重启 | **4.4s** (缓存) | **1.22 GiB** | **105,202** | **3.21x** |

### 11.3 最终配置（32K 上下文 + 2 并发）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/sean/projects/vllm/models/Qwen3.6-35B-A3B-AWQ \
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

### 11.4 内存分布

```
GPU 总显存:            23.99 GiB
vLLM 池 (0.935):       22.43 GiB
├── PyTorch 开销:       ~1.1 GiB  (固定)
├── 模型权重:           20.27 GiB (AWQ 4-bit)
├── CUDA graph:         0.46 GiB  (FULL_AND_PIECEWISE)
└── KV cache (fp8):     1.22 GiB → 105,202 tokens
    ├── per-token:       ~12 KB    (10 层 KV cache × fp8)
    ├── 32K per seq:     0.38 GiB
    └── 最大并发@32K:    3.21x
```

### 11.5 性能基准

| 配置 | 上下文 | 并发 | 总吞吐 | vs 27B@4K | vs 27B@32K |
|------|--------|------|--------|----------|-----------|
| **35B-A3B** | **32,768** | **2** | **355.8 tok/s** | **2.1x** | **4.1x** |
| 27B | 4,096 | 4 | 167.6 tok/s | — | — |
| 27B | 32,768 | 2 | 87.0 tok/s | — | — |

### 11.6 试过但失败的方案

| 方案 | 原因 |
|------|------|
| `--cpu-offload-gb 4` | torch.compile 无法 trace UVAOffloader → dynamo 崩溃 |
| `--cpu-offload-gb 4 --enforce-eager` | Marlin MoE kernel `b_scales is not on GPU` — CPU 端 tensor 不兼容 |
| `--enforce-eager` (无 offload) | 速度仅 ~7 tok/s，不可用 |

### 11.7 Qwen3.6 Thinking 模型注意事项

Qwen3.6 系列默认生成 "thinking" token（内部推理过程），然后用 `<｜end▁of▁thinking｜>...` 包裹实际输出。

- **生产环境务必使用 `--reasoning-parser qwen3`**：将 thinking token 从 content 分离到 `reasoning` 字段
- **不带 reasoning parser**：所有输出直接放到 `content`（包括 thinking 过程），用户看到的是推理步骤而非回答
- **`max_tokens` 预算**：thinking token 也计入 `max_tokens`。建议设置充裕的 `max_tokens`（≥1024），确保 thinking 完成后有足够 token 生成实际回答
- **速度测量**：`completion_tokens` 统计所有 token（thinking + 回答），TPS 计算不受 reasoning parser 影响

## 12. 多配置脚本体系

> 日期: 2026-07-07

### 12.1 设计动机

27B 和 35B 两个模型各自支持多种上下文/并发组合。每次手动改参数容易出错，切换配置时容易忘关旧实例。需要一个统一的脚本体系 — 一个配置一个脚本，启动即用。

### 12.2 架构

```
scripts/
├── env.sh                    ← 共享环境 (Conda/CUDA/WSL2/flashinfer/颜色输出)
├── start_27b_production.sh   ← 各自独立配置
├── start_27b_balanced.sh
├── start_27b_long.sh
├── start_27b_maxctx.sh
├── start_35b_moe.sh
├── start_15b_tutorial.sh
└── stop_vllm.sh              ← 统一清理
```

**设计原则:**
- **`env.sh` 是唯一的环境配置来源**：Conda 激活、CUDA 路径、WSL2 变量、flashinfer 兼容标志、`$COMMON_ARGS` 公共参数
- **每个启动脚本极简**：只定义模型路径、上下文、并发数，其余继承 `$COMMON_ARGS`
- **启动前自动检测冲突**：`check_running()` 发现已有 vLLM 进程时提示是否先停止
- **`stop_vllm.sh` 三步保证清理**：SIGTERM (优雅) → SIGKILL (强制) → nvidia-smi 残留清除

### 12.3 配置矩阵

| 脚本 | 模型 | 上下文 | 并发 | 总吞吐 | 满ctx并发 | 适用场景 |
|------|------|--------|------|--------|----------|----------|
| `start_35b_moe.sh` | 35B-A3B | 32,768 | 2 | 355.8 | 3.21x 🟢 | 长文+高吞吐 🏆 |
| `start_27b_production.sh` | 27B | 4,096 | 4 | 167.6 | 8.33x 🟢 | 高并发服务 |
| `start_27b_long.sh` | 27B | 32,768 | 2 | 87.0 | 2.08x 🟡 | 长文档RAG |
| `start_27b_balanced.sh` | 27B | 8,192 | 2 | 83.7 | 5.56x 🟢 | 日常使用 |
| `start_27b_maxctx.sh` | 27B | 73,728 | 1 | 47.7 | 1.00x 🔴 | 极限上下文 |
| `start_15b_tutorial.sh` | 1.5B | 默认 | 默认 | — | — | 学习测试 |

**满ctx并发** = KV 池 token 容量 ÷ 单请求上下文 token。🟢 >3x 宽裕, 🟡 1.5-3x 勉强, 🔴 <1.5x 极限。

### 12.4 经验总结

1. **共享 env.sh 是正确抽象**：不重复写环境变量，修改一处生效全局
2. **check_running + stop_vllm.sh 组合避免了大量手动操作**：以前每次换配置要手动 `kill`、检查 GPU 残留
3. **文件头注释写清 ctx×seqs×throughput 比任何文档都有效**：`bash scripts/` + Tab 就能看到所有选项
4. **stop_vllm.sh 的 `nvidia-smi --query-compute-apps` 清理是关键**：SIGKILL 后 GPU 上可能有残留 compute context，不清理会导致下次启动 OOM

---

## 13. 项目总结

### 13.1 最终成果

在 RTX 4090 24GB 单卡上成功部署 vLLM 0.23.0，支持 2 个模型、6 种生产配置：

**Qwen3.6-35B-A3B (MoE)** — 最佳模型:
- 32K 上下文 × 2 并发 → **355.8 tok/s** 总吞吐
- 单请求 133 tok/s (vs 27B 的 48 tok/s = 2.8x)
- CUDA graph 绝对不能关 (关了 12x 暴跌)

**Qwen3.6-27B (Dense)** — 全场景覆盖:
- 4K×4 并发 (高并发) → 73K×1 并发 (极限上下文)
- 总吞吐范围 48-168 tok/s

### 13.2 关键数字

| 指标 | 值 |
|------|-----|
| 初始单请求速度 | 6.2 tok/s |
| 最终单请求速度 (27B) | 48.0 tok/s (**7.7x**) |
| 最终单请求速度 (35B) | 133.0 tok/s (**21x**) |
| 最大总吞吐 (35B) | 355.8 tok/s |
| 最大并发 (27B) | 4 @ 4K |
| 最大上下文 (27B) | 73,728 tokens |
| 优化手段 | awq→awq_marlin + CUDA graphs |
| 遇到并解决的兼容问题 | 4 个 (flashinfer CUDA 13) |
| 失败的优化尝试 | 5 个 (MTP/PrefixCache/N-gram/CPU offload/enforce_eager) |
| 项目总代码量 | ~2,800 行 (脚本+文档+测试) |

### 13.3 最重要的 3 个教训

1. **CUDA graphs 对 MoE 是决定性的**：关了从 133→10 tok/s (12x)，远超 Dense 模型的影响。MoE 有大量小 kernel launch，graph 捕获将它们合并。
2. **0.935 是物理天花板，无法突破**：PyTorch CUDA 上下文固定开销 1.12 GiB，关任何东西都省不出来。唯一出路是换框架。
3. **KV cache 是共享池，不是按序列分配**：降并发不会"释放"容量给剩余请求。池子几乎不变，只是重新分配。

### 13.4 项目 Git

- 远程: `git@github.com:xiangshushu1980/WSL_VLLM_4090.git`
- `.gitignore` 排除 `models/` (49GB)、`.claude/`
- 3 次提交: 初始文档 → 脚本体系 → 脚本优化
