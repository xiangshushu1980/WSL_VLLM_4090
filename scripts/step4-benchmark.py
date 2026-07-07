#!/usr/bin/env python3
"""第4步: 并发测试 — 理解 vLLM 的核心优势

自动跑 1/2/4/8 并发对比，观察吞吐量变化

运行: python scripts/step4-benchmark.py
"""
import asyncio
import time
from openai import AsyncOpenAI

WSL_IP = "192.168.31.46"  # 用 hostname -I 获取
MODEL = "/home/sean/projects/vllm/models/Qwen2.5-1.5B-Instruct"

client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://{WSL_IP}:8000/v1")

PROMPTS = [
    "解释什么是快速排序算法。",
    "写一个Python函数计算斐波那契数列。",
    "什么是二叉树？举一个例子。",
    "解释HTTP GET和POST的区别。",
    "写一个SQL查询找出重复记录。",
    "什么是Docker容器？",
    "解释REST API的设计原则。",
    "什么是Git分支？如何使用？",
]


async def benchmark(concurrency: int):
    """跑一轮并发测试"""
    print(f"\n{'='*50}")
    print(f"🔬 测试 {concurrency} 并发")
    print(f"{'='*50}")

    prompts = PROMPTS[:concurrency]

    async def single_request(prompt: str, idx: int):
        t0 = time.time()
        ft = None
        tc = 0
        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.7,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    if ft is None:
                        ft = time.time()
                    tc += 1
            t1 = time.time()
            return {
                "ttft": ft - t0 if ft else 0,
                "total": t1 - t0,
                "tokens": tc,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False, "total": time.time() - t0}

    t_start = time.time()
    tasks = [single_request(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - t_start

    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if ok:
        avg_ttft = sum(r["ttft"] for r in ok) / len(ok)
        total_tokens = sum(r["tokens"] for r in ok)
        throughput = total_tokens / total_time

        print(f"  ✅ 成功: {len(ok)}/{concurrency}")
        print(f"  ⏱  平均 TTFT: {avg_ttft:.2f}s")
        print(f"  🚀 总吞吐量: {throughput:.1f} tok/s")
        print(f"  📊 总 token: {total_tokens}")
        print(f"  ⏱  总耗时: {total_time:.2f}s")
        return {"concurrency": concurrency, "throughput": throughput, "ttft": avg_ttft}
    else:
        print(f"  ❌ 失败: {len(failed)}/{concurrency}")
        return None


async def main():
    print("第4步: 并发测试 — 理解 vLLM 的吞吐优势\n")

    # 检查服务器
    try:
        models = await client.models.list()
        print(f"✅ 服务器在线，模型: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ 无法连接: {e}")
        return

    # 跑不同并发数
    all_results = []
    for c in [1, 2, 4, 8]:
        result = await benchmark(c)
        if result:
            all_results.append(result)
        await asyncio.sleep(2)  # 冷却

    # 总结
    if len(all_results) >= 2:
        base = all_results[0]["throughput"]
        print(f"\n{'='*60}")
        print("📈 并发扩展性总结")
        print(f"{'='*60}")
        print(f"{'并发数':<10} {'吞吐量(tok/s)':<18} {'相对加速':<12} {'平均 TTFT':<12}")
        print("-" * 52)
        for r in all_results:
            speedup = r["throughput"] / base if base > 0 else 0
            print(f"{r['concurrency']:<10} {r['throughput']:<18.1f} {speedup:<12.1f}x {r['ttft']:<12.2f}s")

        print(f"""
{'='*60}
📚 学习要点
{'='*60}

  吞吐量随并发增长:
    - 1→4 并发: 吞吐量应接近线性增长 (2-3x)
    - 超过 max_num_seqs: 请求排队，吞吐不再增长

  TTFT 随并发增长:
    - 并发越大，首 token 越慢（资源竞争）
    - vLLM 的 continuous batching 让 TTFT 增长平缓

  关键参数关系:
    --max-num-seqs: 控制最大并发数
    --max-model-len: 影响每个请求的 KV cache
    两者乘积 ≈ KV cache 总需求
""")


if __name__ == "__main__":
    asyncio.run(main())
