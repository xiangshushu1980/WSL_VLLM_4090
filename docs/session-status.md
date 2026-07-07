# Session Status — 2026-07-06

> 开启新对话后，将此文档提供给 Claude 即可无缝继续。

---

## 一、我们的目标

在 **RTX 4090 24GB** 上部署 **Qwen3.6-27B-AWQ**，实现 **4 并发** vLLM 推理服务。

---

## 二、当前状态

### ✅ 4 并发测试结果 (2026-07-07 优化后)

| 指标 | 优化前 (awq+eager) | 优化后 (awq_marlin+CUDA) | 提升 |
|------|-------------------|------------------------|------|
| 单请求 TPS | 6.2 tok/s | 48.0 tok/s | 7.7x |
| 4并发总吞吐 | 24.2 tok/s | 167.6 tok/s | 6.9x |
| TTFT | 0.46–0.73s | 0.23–0.41s | ~50% 改善 |

### ✅ 已完成

| 任务 | 状态 |
|------|------|
| vLLM 0.23.0 安装 | ✅ conda env `vllm` 可用 |
| Qwen3.6-27B-AWQ 下载 | ✅ `models/Qwen/Qwen3.6-27B-AWQ/` (21GB) |
| flashinfer CUDA 13 兼容补丁 | ✅ 3处修复 + 1个symlink（详见下方备忘） |
| 生产启动脚本 | ✅ `start_server.sh` |
| 测试脚本 | ✅ `test_concurrent.py` |
| 文档体系 | ✅ `docs/` + `scripts/` |
| 记忆文件 | ✅ `~/.claude/.../memory/` |
| 27B-AWQ 4 并发推理 | ✅ 4/4 成功，167.6 tok/s（优化后） |
| Qwen2.5-1.5B 教程模型 | ✅ 已下载 (~2.9GB) |
| 教程第1-4步 | ✅ 全部完成，并发扩展 3.8x |

### ⏸️ 待完成

> 全部完成 ✅

### 🧪 优化实验记录 (2026-07-07)

| 优化手段 | 结果 | 详情 |
|----------|------|------|
| awq→awq_marlin | ✅ 7.7x | `docs/build-log.md#5` |
| CUDA graphs | ✅ 含上面 | 额外 0.48 GiB |
| Prefix Caching | ❌ OOM | 即使降到 2048/2seqs 也不够 |
| N-gram GPU 推测 | ❌ 不兼容 | Qwen3.6 请求挂起 |
| MTP 推测 | ❌ OOM | 未量化头 ~850 MiB |
| 降上下文 | ⚠️ 杀死扩展 | 2048: 1.1x vs 4096: 3.8x |
| **上下文极限测试** | ✅ 新完成 | 4K-72K 全范围测试 |

**单请求 TPS 天花板**: ~52 tok/s (memory-bandwidth bound)，当前 48 tok/s 已达 92%

### 📏 上下文伸缩测试 (2026-07-07 新增)

| 上下文 | 最大并发 | 单TPS | 并发吞吐 | 备注 |
|--------|----------|-------|----------|------|
| 4,096 | 4 | 47.9 | 167.6 (4并发) | **生产最优** |
| 8,192 | 2 | 47.7 | 83.7 (2并发) | 上下文翻倍，并发减半 |
| 32,768 | **2** | 47.8 | 87.0 (2并发) | **长上下文+有限并发** |
| 65,536 | 1 | 47.6 | — | 64K 长上下文单请求 |
| **73,728** | **1** | 47.7 | — | **绝对极限** (1.00x) |

**核心发现**：
- 单 TPS 不受上下文影响（始终 ~48 tok/s），memory-bandwidth bound
- CUDA graph 随并发数缩小：4 seqs→0.48 GiB, 1 seq→0.40 GiB
- KV cache 池随之增长：2.40→2.48 GiB（省下的 CUDA graph 内存分给 KV cache）
- **0.935 是 gpu_memory_utilization 物理天花板**：PyTorch CUDA 上下文吃掉 1.12 GiB，vLLM 看到 free 22.45 GiB。关 GameViewer、`expandable_segments`、重启系统均无效
- 首次编译新上下文需 ~2 GiB 临时显存→首次失败，第二次（缓存命中）成功
- 详见 `docs/build-log.md#83`，`docs/context_scaling_results.md`

### ❌ 已解决的坑（备忘）

| 问题 | 根因 | 修复文件 | 行号 |
|------|------|----------|------|
| flashinfer CCCL 兼容(CUDA 13) | `sm89_nvcc_flags` 没继承 `common_nvcc_flags` | `flashinfer/jit/core.py:123` | 改 `[...]` → `[...] + common_nvcc_flags` |
| flashinfer CCCL 兼容(attention) | `CompilationContext.COMMON_NVCC_FLAGS` 缺少 CCCL disable | `flashinfer/compilation_context.py:28` | 加 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` |
| flashinfer JIT 链接失败 | nvidia-cu13 wheel 用 `lib/` 非 `lib64/` | `flashinfer/jit/cpp_ext.py:254` | `lib64` → `lib`, 删 `stubs`, 加 `/usr/lib/wsl/lib` |
| libcudart.so 找不到 | 只有 `libcudart.so.13` 无 symlink | symlink | `ln -sf libcudart.so.13 libcudart.so` (在 cu13/lib/) |

---

## 三、关键环境信息

```
GPU:    NVIDIA GeForce RTX 4090, 24 GB VRAM
OS:     WSL2 (Linux 6.6)
Conda:  /home/sean/miniconda3/envs/vllm/
Python: 3.11
vLLM:   0.23.0
CUDA:   13.3 (cu13, at $CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13)
```

## 四、关键技术细节

### 4.1 Qwen3.6-27B 架构（重要！）
- **Dense**（非 MoE）：全部 27B 参数每 token 激活
- **混合架构**：64 层，3×DeltaNet(linea_attn) + 1×Full Attention 循环
- 仅 16/64 层有传统 KV cache → KV 显存节约 75%
- `hidden_size=5120, head_dim=256, num_kv_heads=4`

### 4.2 我们下载的模型
- **源**: QuantTrio/Qwen3.6-27B-AWQ (ModelScope)
- **位置**: `models/Qwen/Qwen3.6-27B-AWQ/`
- **大小**: 21 GB (8 个 safetensors 分片)
- **量化**: AWQ 4-bit
- **加载后显存**: **19.05 GiB**（实测）

### 4.3 flashinfer + CUDA 13 兼容问题
- **问题**: flashinfer 0.6.12/0.6.13 的 CCCL 头文件与 CUDA 13 的 nvcc 不兼容
- **出错信息**: `error: "CUDA compiler and CUDA toolkit headers are incompatible"`
- **已应用的修复**:
  1. `pip install flashinfer-cubin==0.6.13 flashinfer-python==0.6.13`
  2. Patch `flashinfer/jit/core.py` — `common_nvcc_flags` 加了 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
  3. `export VLLM_USE_FLASHINFER_SAMPLER=0`
  4. 清除 `~/.cache/flashinfer/`
- **验证状态**: ✅ 修复完成并验证通过 — 4 并发推理成功，163 tok/s 吞吐

### 4.4 WSL2 网络注意事项
- `localhost`/`127.0.0.1` 不通 WSL2 端口
- 必须用 WSL IP: 运行 `hostname -I` 获取（当前是 `192.168.31.46`）
- API 地址: `http://192.168.31.46:8000/v1`

---

## 五、当前 start_server.sh 内容

```bash
#!/bin/bash
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/home/sean/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/sean/miniconda3/envs/vllm/bin:$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-gcc /home/sean/miniconda3/envs/vllm/bin/gcc 2>/dev/null
ln -sf /home/sean/miniconda3/envs/vllm/bin/x86_64-conda-linux-gnu-g++ /home/sean/miniconda3/envs/vllm/bin/g++ 2>/dev/null
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++

MODEL_PATH="/home/sean/projects/vllm/models/Qwen/Qwen3.6-27B-AWQ"

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
```

## 六、项目文件结构

```
/home/sean/projects/vllm/
├── docs/
│   ├── params-reference.md    # 常用参数速查（分类、显存预算公式）
│   ├── rtx4090-guide.md       # 社区实战（模型推荐、量化对比、性能）
│   └── vllm-tutorial.md       # 5步教程（安装→参数→流式→并发→生产）
├── scripts/
│   ├── step1-basic.sh         # 最简启动 Qwen2.5-1.5B
│   ├── step2-memory.sh        # 显存参数学习
│   ├── step3-streaming.py     # 流式输出 + TTFT/TPS
│   └── step4-benchmark.py     # 1/2/4/8 并发自动对比
├── models/
│   ├── Qwen/
│   │   └── Qwen3.6-27B-AWQ/       # 21GB Dense 模型
│   └── Qwen3.6-35B-A3B-AWQ/        # 26GB MoE 模型 (推荐长上下文)
├── start_server.sh                  # 27B 生产启动脚本
└── test_concurrent.py         # 4并发压力测试
```

## 七、新对话后的操作步骤

```
第1步: 验证环境
  $ nvidia-smi                    # 确认 GPU 空闲
  $ source start_server.sh        # 启动服务

第2步: 检查服务
  $ hostname -I                   # 获取 WSL IP
  $ curl http://<IP>:8000/v1/models   # 验证服务在线

第3步: 跑并发测试
  $ python test_concurrent.py     # 4并发压力测试
  # 如果失败，检查日志中的 flashinfer 错误

第4步（可选）: 走教程
  下载 Qwen2.5-1.5B:
  $ hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
  然后按 docs/vllm-tutorial.md 逐步实验
```

## 八、文档维护约定

> **重要**：新开对话后，除了沿用本文档恢复上下文，还需要：
> 1. 将新发现的问题、解决方案、性能数据更新到 [build-log.md](build-log.md)
> 2. 每次优化的前后对比数据记录下来
> 3. 社区反馈和外部参考资料索引到此文档
>
> build-log.md 是永久记录，session-status.md 是短期上下文恢复。

## 九、如果 flashinfer 仍然失败

备选方案:
```bash
# 完全绕过 flashinfer，用 Triton 后端
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

# 或重新安装 flashinfer 更新版本
pip install --upgrade flashinfer-cubin flashinfer-python
rm -rf ~/.cache/flashinfer/
```

## 十、Qwen3.6-35B-A3B MoE 模型 (2026-07-07)

### 🎯 核心数据

| 指标 | 27B (生产) | 27B (长上下文) | **35B-A3B (推荐)** |
|------|-----------|-------------|------------------|
| 上下文 | 4,096 | 32,768 | **32,768** |
| 最大并发 | 4 | 2 | **2** (最多 3) |
| 总吞吐 | 167.6 tok/s | 87.0 tok/s | **355.8 tok/s** |
| vs 27B@4K | — | — | **2.1x** |
| vs 27B@32K | — | — | **4.1x** |

### 架构要点

- MoE: 256 experts top-8, 35B total / 3B activated
- 10 Full Attention + 30 DeltaNet 层 → per-token KV 仅 ~12 KB (fp8)
- 模型加载: 20.27 GiB GPU (AWQ 4-bit)
- CUDA graph: 0.46 GiB (FULL_AND_PIECEWISE)
- KV cache: 1.22 GiB (105,202 tokens)

### 35B 启动脚本

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

### ⚠️ 重要注意

1. **首次启动必需重启**: torch.compile 编译需 ~46s 且占用临时显存 → KV cache 仅 0.53 GiB。重启后缓存命中（4.4s），KV cache 涨到 1.22 GiB
2. **CUDA graph 决不能关**: 关了速度从 133 tok/s 跌到 10 tok/s（12x 差异！）
3. **`--cpu-offload-gb` 不可用**: UVA offloader 与 torch.compile 和 Marlin MoE kernel 均不兼容
4. **`--reasoning-parser qwen3` 必须加**: Qwen3.6 默认生成 thinking token，不加的话输出全是推理过程
5. **`max_tokens` 设大些**: thinking token 也计入预算，推荐 ≥1024
