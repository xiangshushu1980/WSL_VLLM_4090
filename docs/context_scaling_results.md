# 上下文伸缩测试 — 完整结果

> 测试时间: 2026-07-07 | 模型: Qwen3.6-27B-AWQ | GPU: RTX 4090 24GB WSL2
> 固定参数: `quantization=awq_marlin`, `gpu_memory_utilization=0.935`, `kv_cache_dtype=fp8`

---

## 测试目标

回答两个问题：
1. 24GB 显存能支持多大上下文？
2. 上下文长度如何影响并发能力和 TPS？

---

## 完整结果

| 配置 | 上下文 | max_seqs | CUDA graph | KV cache 池 | KV tokens | 满ctx并发 | 单 TPS | 并发吞吐 | 扩展比 |
|------|--------|----------|------------|------------|-----------|----------|-------|---------|--------|
| A | 4,096 | 4 | 0.48 GiB | 2.40 GiB | 34,133 | 8.33x | 47.9 | 167.6 (4并发) | **3.8x** |
| F | 8,192 | 2 | 0.44 GiB | 2.44 GiB | 45,511 | 5.56x | 47.7 | 83.7 (2并发) | 1.8x |
| G | 16,384 | 1 | 0.40 GiB | 2.48 GiB | 59,684 | 3.64x | 47.9 | — | — |
| H | 24,576 | 1 | 0.40 GiB | 2.48 GiB | 65,967 | 2.68x | 47.1 | — | — |
| I | 32,768 | 2 | 0.44 GiB | 2.44 GiB | — | **2.08x** | 47.8 | 87.0 (2并发) | 1.91x |
| J | 49,152 | 1 | 0.40 GiB | 2.48 GiB | — | 1.46x | 50.6 | — | — |
| K | 65,536 | 1 | 0.40 GiB | 2.48 GiB | — | 1.13x | 47.6 | — | — |
| L | **73,728** | 1 | 0.40 GiB | 2.48 GiB | — | **1.00x** | 47.7 | — | — |

---

## 关键发现

### 1. 0.935 是 gpu_memory_utilization 的硬上限

```
空闲可用:           24138 MiB = 23.57 GiB
PyTorch CUDA init:  -1.12 GiB (驱动 + CUDA 上下文 + PyTorch 分配器)
vLLM 看到:          22.45 GiB
0.935 × 23.99:      22.43 GiB  ← 刚好可分配
0.940 × 23.99:      22.55 GiB  ← 超出 0.10 GiB, 永远失败
```

- GameViewer.exe 关掉只省 13 MiB，几乎无影响
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 对初始分配**无效果**（只解决碎片化 OOM）
- 重启系统无变化
- **0.935 是物理极限，不可突破**

### 2. 单 TPS 与上下文无关

始终 ~48 tok/s。Qwen3.6 只有 16/64 层有传统 KV cache（其余是 DeltaNet），权重读取占 91% 带宽。

### 3. 并发是上下文的倒数

KV cache 池几乎恒定（2.40-2.48 GiB），上下文越长 → 每请求吃越多 → 并发越少。

### 4. 首次编译陷阱

切换 `max_model_len` 需要新的 torch.compile 缓存。首次启动时 warmup profiling 临时占用 ~2 GiB 额外 GPU 内存 → 启动失败。**第二次启动（缓存命中）即可正常工作。**

### 5. 推荐配置

**27B Dense:**

```
并发优先:  4096/4seqs  → 167.6 tok/s 总吞吐   (在线服务)
平衡:      8192/2seqs  →  83.7 tok/s            (中等上下文+并发)
长上下文:  65536/1seq  →  47.6 tok/s            (单请求长文档)
```

**35B-A3B MoE（2026-07-07 新增）:**

| 配置 | 上下文 | 并发 | 吞吐 | vs 27B@4K | vs 27B@32K |
|------|--------|------|------|----------|-----------|
| **推荐** | 32,768 | 2 | **355.8 tok/s** | 2.1x | 4.1x |
| 平衡 | 16,384 | 3 | ~500 tok/s | 3.0x | — |
| 极限 | 32,768 | 3 | ~500 tok/s | 3.0x | — |

> ⚠️ 35B-A3B 首次启动需重启：torch.compile 编译用临时显存→首启 KV cache 仅 0.53 GiB，重启后恢复到 1.22 GiB。

### 6. 27B vs 35B-A3B 上下文伸缩对比

```
                4K       8K       16K      32K      64K
27B  Dense:   4并发     2并发     1并发    2并发     1并发
              168t/s    84t/s     48t/s    87t/s     48t/s

35B-A3B MoE:  4并发     3并发     3并发    2并发     —(OOM)
              500+t/s   500+t/s   500+t/s  356t/s
```

35B-A3B 在所有上下文都碾压 27B，得益于：
- 更少全注意力层 (10 vs 16) → per-token KV 仅 12 KB vs 74 KB
- MoE 3B 激活 vs 27B 全激活 → decode 更快
- 4.1x 吞吐优势在 32K 达到峰值

---

## 测试方法

- 手动逐配置启动服务器，监控 nvidia-smi 确认内存分配
- 每次新配置需：首次启动（编译 → 失败）→ 第二次启动（缓存 → 成功）
- Benchmark: `benchmark_optimizations.py` 单请求 + 并发
- 服务器日志: `/tmp/vllm_ctx_test_*.log`, `/tmp/vllm_*k_debug.log`

## 关联文档

- [build-log.md §8.3](build-log.md) — 上下文极限测试（同数据源）
- [session-status.md](session-status.md) — 会话状态和快速参考
- [optimization_results.md](optimization_results.md) — 5 配置优化测试（上下文/并发/PrefixCaching 对比）
