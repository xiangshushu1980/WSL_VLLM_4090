# RTX 4090 24GB 社区实践指南

> 基于 Reddit r/LocalLLaMA、vLLM Forums、HuggingFace Discussions 的真实反馈整理（2026 年 7 月）

---

## 一、RTX 4090 硬件概览

| 规格 | 数值 |
|------|------|
| 显存 | 24,564 MiB (24 GB) |
| 架构 | Ada Lovelace (SM 8.9) |
| 显存带宽 | 1,008 GB/s |
| FP16 算力 | 82.6 TFLOPS |
| 功耗 | 450W |

## 二、社区公认最佳模型

### 🥇 Qwen3.6-35B-A3B (MoE) — 社区首选

| 指标 | 社区报告 | **我们实测 (vLLM 0.23)** |
|------|---------|----------------------|
| 架构 | MoE, 35B 总参数, **仅 3B 激活** |
| 量化 | AWQ 4-bit ≈ 20.3 GiB 加载后 |
| 推理速度 | 120-150 tok/s (llama.cpp) | **355.8 tok/s @2并发** (vLLM) |
| 并发 | 4 并发 @ 32K (llama.cpp) | **2 并发 @ 32K** (vLLM, 显存限制) |
| 上下文上限 | 32K+ | 32K (实测可 2 并发) |
| 单卡可行性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

> **实测详情**: `Chunity/Qwen3.6-35B-A3B-AWQ`，模型 20.27 GiB + CUDA graph 0.46 GiB + KV cache 1.22 GiB = 逼近 0.935 天花板。首次启动需重启以释放 torch.compile 占用的临时显存。详见 [build-log.md §11](build-log.md#11-qwen36-35b-a3b-moe-部署)

### 🥈 Qwen3.6-27B — 密集型甜点

| 指标 | 数值 |
|------|------|
| 架构 | Dense + DeltaNet 混合（64层，16层全注意力） |
| 量化 | AWQ 4-bit ≈ ~19 GB 加载后 |
| SWE-bench | ~70+ |
| 并发 | 2-4 并发 @ 8K 上下文 |
| 单卡可行性 | ⭐⭐⭐⭐ |

> **我们的实测**: `QuantTrio/Qwen3.6-27B-AWQ`，加载 19.05 GiB，2-4 并发

### 🥉 其他推荐

| 模型 | 量化 | 显存 | 场景 |
|------|------|------|------|
| Qwen2.5-14B-AWQ | ~8 GB | ~12 GB total | 快速原型、高并发 |
| DeepSeek-Coder-V2 Lite | 16B MoE | ~12 GB | 代码专用 |
| Llama 3.1 70B AWQ | ~21 GB | ~24 GB total | 最强单卡 70B（仅 1 并发） |

## 三、社区量化格式选择

| 格式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **AWQ** | vLLM 原生支持，加载快，Marlin 加速 | 文件较大（~20GB） | vLLM 生产部署 |
| **GPTQ** | 灵活、g128 效果好 | 加载稍慢 | vLLM 备选 |
| **GGUF** | llama.cpp 原生，省显存 | vLLM 支持有限 | 单用户本地推理 |
| **FP8** | 质量高 | 27B × 1 byte = 27GB，24GB 不会放 | 多卡部署 |

## 四、vLLM vs llama.cpp 实测对比

> 同一台 4090 跑 Llama 4 Scout 17B：

| 指标 | Ollama | vLLM | llama.cpp |
|------|--------|------|-----------|
| 单用户吞吐 | 40-50 t/s | **485 t/s** | 50-100 t/s |
| 4 并发吞吐 | ~155 t/s | **920 t/s** | — |
| p95 延迟 (4并发) | 18.4s | **2.1s** | — |
| 显存利用率 | 60-80% | **<4% 浪费** | 依赖量化 |

> **社区结论**: 多并发必须 vLLM，单用户 llama.cpp 更简单

## 五、典型错误和解决

| 错误 | 原因 | 修复 |
|------|------|------|
| "FP8 29GB 下载了 OOM" | FP8 权重 27GB > 24GB | 换 AWQ/GPTQ |
| "flashinfer 编译失败" | CUDA 13 + 旧 CCCL | 升级 flashinfer + 补丁 |
| "OOM on first request" | KV cache 分配太大 | 降 `--max-model-len` |
| "模型加载成功但请求卡住" | WSL localhost 不通 | 用 WSL IP |
