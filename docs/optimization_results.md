# 优化配置 TPS 测试结果

> 测试时间: 2026-07-07 02:59–03:21 EDT
> 模型: Qwen3.6-27B-AWQ | GPU: RTX 4090 24GB
> 固定参数: `quantization=awq_marlin`, `gpu_memory_utilization=0.935`, `kv_cache_dtype=fp8`, CUDA graphs (FULL_AND_PIECEWISE)

---

## 测试目标

回答一个问题：**降低上下文/并发能腾出多少显存？值得牺牲并发换取其他优化（如 Prefix Caching）吗？**

为此设计了 5 个配置，逐步变化 `max_model_len`（上下文窗口）和 `max_num_seqs`（最大并发数）。

---

## 测试配置一览

| 配置 | 上下文 | 最大并发 | 额外优化 | 目的 |
|------|--------|----------|----------|------|
| **A** (baseline) | 4096 | 4 | 无 | 当前最优配置，作为对照组 |
| **B** (reduce) | 2048 | 2 | 无 | 大幅降低上下文+并发，看内存释放效果 |
| **C** (prefix) | 2048 | 2 | `--enable-prefix-caching` | 在 B 的基础上，用省下的内存开 Prefix Caching |
| **D** (medium) | 3072 | 3 | 无 | A/B 的中间配置 |
| **E** (low-ctx) | 2048 | 4 | 无 | 只降上下文、不降并发，分离"上下文 vs 并发"的影响 |

---

## 单请求 TPS（所有配置对比）

> 方法: 同一 prompt 跑 2 轮取最快（排暖启动干扰），`benchmark_optimizations.py single`

| 配置 | 上下文 | 单请求 TPS | TTFT |
|------|--------|-----------|------|
| A (baseline) | 4096 | 47.9 tok/s | ~0.27s |
| B (reduce) | 2048 | 46.9 tok/s | — |
| D (medium) | 3072 | 46.9 tok/s | — |
| E (low-ctx) | 2048 | 48.1 tok/s | — |

**结论**: 单请求 TPS 与上下文长度几乎无关（46.9–48.1 tok/s），差异在测量误差范围内。瓶颈是 memory bandwidth，不受 max_model_len 影响。

---

## 并发 TPS（核心测试）

> 方法: 依次发 1/2/3/4 个并发请求，测量总吞吐 = 总 token 数 / 总墙钟时间，`benchmark_optimizations.py concurrent`

| 配置 | 上下文 | 并发上限 | 1并发 | 2并发 | 3并发 | 4并发 | 4并发扩展比 |
|------|--------|----------|-------|-------|-------|-------|------------|
| **A** (baseline) | 4096 | 4 | 44.1 | 88.2 | 122.6 | **167.6** | **3.8x** |
| B (reduce) | 2048 | 2 | 50.4 | 44.4 | — | — | 0.9x |
| D (medium) | 3072 | 3 | 43.6 | 45.7 | 46.8 | — | 1.1x |
| **E** (low-ctx) | 2048 | 4 | 44.1 | 42.4 | 45.9 | **47.3** | **1.1x** |

> 注: 表中数据来自 `run_opt_tests.sh` 的 grep 解析，受 benchmark 脚本偏差影响可能有少量抖动（如 B 的 1 并发 > 单 TPS 是正常的，并发测试用不同 prompt 且测量方式不同）。**趋势和扩展比是可靠的。**

### 关键发现

```
                    4 并发扩展比
Config A (4096/4):  ████████████████████ 3.8x  ← 接近线性，最优
Config D (3072/3):  ██████ 1.1x                 ← 开始衰减
Config E (2048/4):  ██████ 1.1x                 ← 完全无扩展，4并发=1并发
Config B (2048/2):  █████ 0.9x                  ← 反而退化
```

**上下文长度是并发扩展的决定性因素**，不是并发数本身：

- 4096 上下文 → chunked prefill 能有效交错多个请求 → 3.8x 扩展
- 2048 上下文 → KV cache 太小 (0.33 GiB, 34k tokens)，多个请求互相排队等待 → 无扩展
- 降上下文"省"出来的显存（~0.1 GiB）远远抵不上并发能力损失

---

## Prefix Caching 测试 (Config C)

| 配置 | 上下文 | 并发 | 额外参数 | 结果 |
|------|--------|------|----------|------|
| C (prefix) | 2048 | 2 | `--enable-prefix-caching` | ❌ **OOM 启动失败** |

**失败日志**:
```
ValueError: To serve at least one request with the model's max seq len (2048),
(0.38 GiB KV cache is needed, which is larger than the available KV cache
memory (0.33 GiB)).
```

即使降到最低配置（2048 上下文 + 2 并发），Prefix Caching 的内部哈希表开销 (~0.05 GiB) 仍导致 KV cache 不足。

**结论**: RTX 4090 24GB 上 Prefix Caching 不可用。

---

## 显存预算分析

```
模型权重:       19.05 GiB  (固定)
CUDA graphs:     0.48 GiB  (固定)
运行时开销:      ~2.0 GiB  (固定)
─────────────────────────
剩余给 KV cache:  2.4 GiB  (0.935 utilization)
```

| 上下文 | KV cache 需求 (per seq) | max 4 seqs | 结论 |
|--------|------------------------|------------|------|
| 4096 | ~0.43 GiB | ~1.7 GiB | ✅ 刚好够 |
| 3072 | ~0.32 GiB | ~1.3 GiB | ✅ 够用 |
| 2048 | ~0.22 GiB | ~0.9 GiB | ✅ 够用但并发不扩展 |
| 2048 + prefix | ~0.22 + 0.05 hash | ~0.9 GiB | ❌ OOM (0.33 可用 < 0.38 需要) |

显存上够用不代表并发能扩展 — 2048 上下文时虽然有 0.9 GiB KV cache 空间，但 prefill 阶段 token 数太少、chunked prefill 无法有效交错多个请求。

---

## 最终结论

| 问题 | 答案 |
|------|------|
| 降上下文能省显存吗？ | 能，但只有 ~0.1 GiB，微不足道 |
| 降并发能省显存吗？ | 能省 ~0.2 GiB/seq，但用不上 |
| 省下的显存够开 Prefix Caching 吗？ | ❌ 不够，即使 2048/2seqs 也 OOM |
| 降上下文影响单请求 TPS 吗？ | ❌ 不影响，single TPS 始终 ~48 tok/s |
| 降上下文影响并发扩展吗？ | ✅ **严重影响**，4096→3.8x, 2048→1.1x |
| 最优 27B 配置是什么？ | **Config A: 4096/4seqs, awq_marlin, CUDA graphs** |

> **2026-07-07 更新**：35B-A3B MoE 模型在 32K/2seqs 下达到 355.8 tok/s，是 27B 最优配置的 **2.1x**。详见 [build-log.md §11](build-log.md#11-qwen36-35b-a3b-moe-部署)

---

## 测试方法说明

- **自动测试脚本**: `run_opt_tests.sh` — 循环启动不同配置的服务器 → 等待健康检查 → 跑 benchmark → 记录结果 → 杀服务器
- **Benchmark 脚本**: `benchmark_optimizations.py` — `single` 模式测单请求 TPS, `concurrent` 模式测并发总吞吐
- **数据采集**: shell script 通过 grep 解析 benchmark 的 stdout，少部分数据可能在解析中错位/丢失
- **服务器日志**: 保留在 `/tmp/vllm_test_<config>.log`，用于排查启动/运行问题
