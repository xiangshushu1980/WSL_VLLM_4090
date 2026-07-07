# Session Status — 2026-07-07

> 开启新对话后，将此文档提供给 Claude 即可无缝继续。

---

## 一、我们的目标

在 **RTX 4090 24GB** 上多配置部署 vLLM 推理服务，支持 **2 个模型、7 种配置** 灵活切换。

---

## 二、当前状态：所有目标已完成 ✅

### 2.1 模型资产

| 模型 | 位置 | 大小 | 量化 | 状态 |
|------|------|------|------|------|
| Qwen3.6-27B-AWQ | `models/Qwen/Qwen3.6-27B-AWQ/` | 21 GB | AWQ 4-bit, Dense | ✅ 生产就绪 |
| Qwen3.6-35B-A3B-AWQ | `models/Qwen3.6-35B-A3B-AWQ/` | 26 GB | AWQ 4-bit, MoE | ✅ 生产就绪 🏆 |
| Qwen2.5-1.5B-Instruct | `models/Qwen2.5-1.5B-Instruct/` | 2.9 GB | 无 (FP16) | ✅ 教程用 |

### 2.2 启动配置一览

| 脚本 | 模型 | 上下文 | 并发 | 吞吐 | 满ctx并发 | 场景 |
|------|------|--------|------|------|----------|------|
| `start_35b_moe.sh` | 35B-A3B | 32K | 2 | **356 tok/s** | 3.21x 🟢 | 🏆 长文·高吞吐 |
| `start_27b_production.sh` | 27B | 4K | 4 | **168 tok/s** | 8.33x 🟢 | 高并发在线服务 |
| `start_27b_long.sh` | 27B | 32K | 2 | 87 tok/s | 2.08x 🟡 | 长文档·RAG |
| `start_27b_balanced.sh` | 27B | 8K | 2 | 84 tok/s | 5.56x 🟢 | 日常中等任务 |
| `start_27b_maxctx.sh` | 27B | **73K** | 1 | 48 tok/s | 1.00x 🔴 | 极限单请求 |
| `start_15b_tutorial.sh` | 1.5B | 默认 | 默认 | — | — | 学习·快速测试 |

> **满ctx并发** = KV 池能装下多少个满上下文请求。🟢 宽裕 🟡 勉强 🔴 极限。

### 2.3 基础设施

| 文件 | 用途 |
|------|------|
| `scripts/env.sh` | 所有脚本共享的环境变量 (Conda/CUDA/WSL2/flashinfer) |
| `scripts/stop_vllm.sh` | 三步清理: SIGTERM → SIGKILL → GPU 残留清除 |
| `scripts/step1-basic.sh` ~ `step4-benchmark.py` | 4 步渐进教程 |
| `benchmark_optimizations.py` | 优化矩阵测试 (单请求/并发/PrefixCaching) |
| `benchmark_tps.py` | 1.5B vs 27B 并发扩展对比 |
| `test_concurrent.py` | 4 并发压力测试 (Qwen3.6 reasoning) |

### 2.4 版本控制

| 项目 | 状态 |
|------|------|
| Git 仓库 | ✅ `git@github.com:xiangshushu1980/WSL_VLLM_4090.git` |
| `.gitignore` | ✅ 排除 `models/` (49GB)、`.claude/`、`__pycache__/` 等 |
| 模型文件 | ❌ 不上传 (`.gitignore` 中) |

---

## 三、关键环境信息

```
GPU:    NVIDIA GeForce RTX 4090, 24 GB VRAM
OS:     WSL2 (Linux 6.6.114)
Conda:  /home/sean/miniconda3/envs/vllm/
Python: 3.11
vLLM:   0.23.0
CUDA:   13.3 (cu13 pip package)
驱动:   610.43.02 (WSL2)
```

## 四、关键技术细节

### 4.1 两个模型的架构差异

| 特性 | Qwen3.6-27B | Qwen3.6-35B-A3B |
|------|------------|-----------------|
| 架构 | Dense (27B全激活) | MoE (35B总参, 3B激活) |
| 全注意力层 | 16/64 | 10/40 |
| KV per token (fp8) | ~74 KB | **~12 KB** |
| 模型加载显存 | 19.05 GiB | 20.27 GiB |
| CUDA graph | 0.40-0.48 GiB | 0.46 GiB |
| KV cache 池 | 2.40-2.48 GiB | 1.22 GiB |
| 单请求 TPS | ~48 tok/s | **~133 tok/s** |
| CUDA graph 关闭 | 性能暴跌 | **12x 暴跌 (133→10 tok/s)** |

### 4.2 显存分配模型

```
GPU 总显存:                   23.99 GiB
vLLM 池 (0.935):              22.43 GiB
├── PyTorch 开销:             ~1.12 GiB (固定, 不可压缩)
├── 模型权重:                  19-20 GiB (取决于模型)
├── CUDA graph:              0.40-0.48 GiB (取决于 max_num_seqs)
└── KV cache (fp8):          剩余全部
                               27B: 2.40-2.48 GiB
                               35B: 1.22 GiB (MoE per-token 小)
```

**0.935 是物理天花板**: PyTorch CUDA 上下文固定吃掉 1.12 GiB, vLLM 看到 free 22.45 GiB, 0.935×23.99=22.43 GiB。多 0.02 GiB 都拿不出。

### 4.3 KV Cache 池理解

KV cache 是一个**共享池**, 不是按序列独立分配:

```
max_seqs=2, KV池 2.44 GiB → 总容量 ~68K tokens → 可分给 2 个请求
max_seqs=1, KV池 2.48 GiB → 总容量 ~73K tokens → 全给 1 个请求

降并发只让池子大 0.04 GiB (+1.6%), 总容量只多 8%
```

去掉一个并发槽位不会"释放 32K 容量" — 池子大小几乎不变, 只是重新分配给更少的请求。

### 4.4 flashinfer + CUDA 13 兼容修复

已应用 4 处修复（详见 `build-log.md §3`）:
1. `flashinfer/jit/core.py:123` — sm89 flags 继承 common flags
2. `flashinfer/compilation_context.py:28` — 加 CCCL disable flag  
3. `flashinfer/jit/cpp_ext.py:254` — lib64→lib, 加 WSL lib
4. nvidia-cu13/lib/ 下创建 .so symlink

环境变量: `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`

### 4.5 首次编译陷阱 (两个模型都有)

切换 `max_model_len` 或首次启动时，torch.compile 需 ~2 GiB 临时显存 → 首次可能 OOM。**第二次启动（缓存命中）即正常。**

35B-A3B 尤其明显：首启 KV cache 仅 0.53 GiB, 重启后恢复到 1.22 GiB。

### 4.6 WSL2 网络

- `localhost`/`127.0.0.1` 不通 → 用 `hostname -I` 获取 IP
- API 地址: `http://<WSL_IP>:8000/v1`

---

## 五、已完成的所有任务

| 任务 | 状态 | 日期 |
|------|------|------|
| vLLM 0.23.0 安装 (conda) | ✅ | 07-06 |
| flashinfer CUDA 13 兼容修复 | ✅ | 07-06 |
| Qwen3.6-27B-AWQ 下载部署 | ✅ | 07-06 |
| 27B 4并发 167.6 tok/s (awq_marlin+CUDA graph) | ✅ | 07-07 |
| Qwen2.5-1.5B 教程模型下载 | ✅ | 07-06 |
| 4 步渐进教程完成 | ✅ | 07-07 |
| 上下文伸缩测试 (4K-73K) | ✅ | 07-07 |
| gpu_memory_utilization 天花板调查 | ✅ | 07-07 |
| Qwen3.6-35B-A3B-AWQ 下载部署 | ✅ | 07-07 |
| 35B MoE 32K/2并发 355.8 tok/s | ✅ | 07-07 |
| 多配置启动脚本体系 (6 个配置) | ✅ | 07-07 |
| stop_vllm.sh 三步清理脚本 | ✅ | 07-07 |
| Git 仓库初始化 + GitHub 推送 | ✅ | 07-07 |

### 已尝试但失败的优化

| 优化 | 结果 | 原因 |
|------|------|------|
| MTP 推测解码 | ❌ OOM | 未量化头 ~850 MiB, 与 CUDA graph 不共存 |
| Prefix Caching | ❌ OOM | 哈希表 ~2 GiB, 降到 2048/2seqs 都不够 |
| N-gram GPU 推测 | ❌ 不兼容 | Qwen3.6 请求挂起 |
| CPU offload | ❌ 崩溃 | 与 torch.compile + Marlin MoE 均不兼容 |
| `enforce_eager` (35B) | ❌ 不可用 | 速度仅 ~7 tok/s (CUDA graph 加速 13x) |
| gpu_memory_utilization=0.94 | ❌ 永远失败 | 需要 22.55 GiB, 只有 22.45 GiB |

---

## 六、项目文件结构

```
/home/sean/projects/vllm/
├── scripts/                          # 启动 & 管理脚本
│   ├── env.sh                        #   共享环境配置
│   ├── start_27b_production.sh       #   27B: 4K×4并发 168 tok/s
│   ├── start_27b_balanced.sh         #   27B: 8K×2并发 84 tok/s
│   ├── start_27b_long.sh             #   27B: 32K×2并发 87 tok/s
│   ├── start_27b_maxctx.sh           #   27B: 73K×1并发 48 tok/s (极限)
│   ├── start_35b_moe.sh              #   35B: 32K×2并发 356 tok/s 🏆
│   ├── start_15b_tutorial.sh         #   1.5B: 教程/测试
│   ├── stop_vllm.sh                  #   三步清理脚本
│   ├── step1-basic.sh                #   教程1: 最简启动
│   ├── step2-memory.sh               #   教程2: 显存参数
│   ├── step3-streaming.py            #   教程3: 流式 TTFT/TPS
│   └── step4-benchmark.py            #   教程4: 并发扩展
├── docs/                             # 文档
│   ├── build-log.md                  #   完整构建日志 (永久记录)
│   ├── session-status.md             #   本文档 (会话恢复)
│   ├── params-reference.md           #   参数速查 + 显存预算公式
│   ├── context_scaling_results.md    #   上下文伸缩测试完整数据
│   ├── optimization_results.md       #   5 配置优化测试对比
│   ├── rtx4090-guide.md              #   社区实践 + 模型推荐
│   └── vllm-tutorial.md              #   5 步渐进教程
├── benchmark_optimizations.py        # 优化矩阵测试
├── benchmark_tps.py                  # TPS 基准测试
├── test_concurrent.py                # 并发压力测试
├── start_server.sh                   # (旧) 27B 单配置脚本
└── .gitignore                        # 排除 models/ .claude/ 等
```

---

## 七、新对话快速恢复

```bash
# 第1步: 验证环境
nvidia-smi                           # GPU 空闲?

# 第2步: 选配置启动 (推荐 35B MoE)
bash scripts/start_35b_moe.sh        # 最佳综合
# 或: bash scripts/start_27b_production.sh   # 高并发

# 第3步: 测试
curl http://$(hostname -I | awk '{print $1}'):8000/v1/models

# 第4步: 停止
bash scripts/stop_vllm.sh

# 第5步: 同步文档更新
git add -A && git commit -m "..." && git push origin master
```

---

## 八、文档维护约定

- **build-log.md** — 永久技术记录 (问题/修复/性能数据), 大节追加
- **session-status.md** — 本文档, 会话级状态和快速参考, 每次对话结束更新
- **params-reference.md** — 参数和配置的权威来源
- **context_scaling_results.md** — 测试原始数据
- **optimization_results.md** — 优化对比数据
