#!/usr/bin/env python3
"""优化手段 TPS 影响测试矩阵

测试不同 上下文/并发/PrefixCaching 组合对 TPS 的影响。
每次需要手动启动对应配置的服务器。
"""
import asyncio
import time
import sys
from openai import AsyncOpenAI

WSL_IP = "192.168.31.46"
MODEL = "qwen3.6-27b"

client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://{WSL_IP}:8000/v1")

# 通用技术问答 prompt（短，适合低上下文测试）
PROMPTS = [
    "用Python写一个二分查找算法。",
    "解释TCP三次握手的过程。",
    "什么是Docker容器？",
    "写一个SQL查询去重复记录。",
]

# 共享前缀的 prompt（用于 Prefix Caching 测试）
SHARED_SYSTEM = "你是一个资深软件工程师，擅长Python、数据库和系统设计。请用简洁专业的中文回答。"
SHARED_PROMPTS = [
    "用Python写一个二分查找算法。",
    "用Python写一个快速排序算法。",
    "用Python写一个归并排序算法。",
    "用Python写一个冒泡排序算法。",
]


async def single_request(prompt: str, system: str = "", max_tokens: int = 256):
    t0 = time.time()
    ft = None
    tc = 0
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=MODEL, messages=messages,
            max_tokens=max_tokens, temperature=0.0, stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            text = (delta.model_extra.get('reasoning') if delta.model_extra else None) or delta.content or ""
            if text:
                if ft is None:
                    ft = time.time()
                tc += 1
        t1 = time.time()
        return {
            "ttft": ft - t0 if ft else 0,
            "total_time": t1 - t0,
            "tokens": tc,
            "tps": tc / (t1 - ft) if ft and t1 > ft else 0,
            "success": True,
        }
    except Exception as e:
        return {"error": str(e)[:80], "success": False}


async def bench_single(name: str, rounds: int = 2):
    """单请求基准（取最快一轮，排除 warmup 干扰）"""
    best_tps = 0
    best_ttft = 999
    for i in range(rounds):
        r = await single_request(PROMPTS[0])
        if r["success"]:
            if r["tps"] > best_tps:
                best_tps = r["tps"]
                best_ttft = r["ttft"]
        else:
            print(f"  错误: {r.get('error', 'unknown')}")
            return {"name": name, "tps": 0, "ttft": 0}
    print(f"  {name}: {best_tps:.1f} tok/s (TTFT {best_ttft:.2f}s)")
    return {"name": name, "tps": best_tps, "ttft": best_ttft}


async def bench_concurrent(name: str, max_n: int, max_tokens: int = 256):
    """并发测试"""
    print(f"\n--- {name} (max_n={max_n}) ---")
    results_all = []
    for c in [1, 2, 3, 4]:
        if c > max_n:
            break
        prompts = PROMPTS[:c]
        t0 = time.time()
        tasks = [single_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        t1 = time.time()
        ok = [r for r in results if r["success"]]
        if not ok:
            print(f"  {c}并发: 失败")
            continue
        total_tokens = sum(r["tokens"] for r in ok)
        throughput = total_tokens / (t1 - t0)
        avg_tps = sum(r["tps"] for r in ok) / len(ok)
        print(f"  {c}并发: 吞吐 {throughput:.1f} tok/s, 单TPS {avg_tps:.1f}, 耗时 {t1-t0:.1f}s")
        results_all.append({"concurrency": c, "throughput": throughput, "avg_tps": avg_tps})
        await asyncio.sleep(0.5)
    return results_all


async def bench_prefix_caching():
    """Prefix Caching 专项测试：相同前缀 vs 不同前缀"""
    print(f"\n--- Prefix Caching 专项测试 ---")

    # 测试1: 无共享前缀 (不同 prompt)
    print("  无共享前缀:")
    t0 = time.time()
    tasks = [single_request(p) for p in PROMPTS[:4]]
    results1 = await asyncio.gather(*tasks)
    t1 = time.time()
    ok1 = [r for r in results1 if r["success"]]
    tps1 = sum(r["tokens"] for r in ok1) / (t1 - t0) if ok1 else 0

    # 测试2: 共享系统前缀 (相同 system prompt)
    print("  共享系统前缀:")
    t0 = time.time()
    tasks2 = [single_request(p, system=SHARED_SYSTEM) for p in SHARED_PROMPTS[:4]]
    results2 = await asyncio.gather(*tasks2)
    t2 = time.time()
    ok2 = [r for r in results2 if r["success"]]
    tps2 = sum(r["tokens"] for r in ok2) / (t2 - t0) if ok2 else 0

    gain = (tps2 / tps1 - 1) * 100 if tps1 > 0 else 0
    print(f"  无前缀: {tps1:.1f} tok/s | 有前缀: {tps2:.1f} tok/s | 提升: {gain:+.0f}%")
    return {"no_prefix_tps": tps1, "prefix_tps": tps2, "gain_pct": gain}


async def main():
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print("=" * 60)
    print("  优化手段 TPS 影响测试")
    print(f"  Model: {MODEL} | max_concurrency: {max_n}")
    print("=" * 60)

    if test_type in ("all", "single"):
        print("\n📊 单请求基准")
        await bench_single("单请求(冷启动)", rounds=1)
        await bench_single("单请求(热身后)", rounds=1)
        await bench_single("单请求(确认)", rounds=1)

    if test_type in ("all", "concurrent"):
        await bench_concurrent("并发测试", max_n)

    if test_type in ("all", "prefix"):
        await bench_prefix_caching()


if __name__ == "__main__":
    asyncio.run(main())
