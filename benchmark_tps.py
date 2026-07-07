#!/usr/bin/env python3
"""TPS 基准测试：1.5B vs 27B-AWQ，1/2/3/4 并发对比"""
import asyncio
import time
import sys
import json
from openai import AsyncOpenAI

WSL_IP = "192.168.31.46"
MODEL_15B = "/home/sean/projects/vllm/models/Qwen2.5-1.5B-Instruct"
MODEL_27B = "qwen3.6-27b"
CONCURRENCIES = [1, 2, 3, 4]

client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://{WSL_IP}:8000/v1")

PROMPTS = [
    "用Python写一个冒泡排序算法，并解释其时间复杂度。",
    "什么是Docker容器？它和虚拟机有什么区别？",
    "解释TCP三次握手的过程，每一步的作用是什么？",
    "什么是RESTful API？列出至少3个设计原则。",
]


async def single_request(model: str, prompt: str, idx: int, max_tokens: int = 256):
    t0 = time.time()
    ft = None
    tc = 0
    content_out = ""
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            text = (delta.model_extra.get('reasoning') if delta.model_extra else None) or delta.content or ""
            if text:
                if ft is None:
                    ft = time.time()
                content_out += text
                tc += 1
        t1 = time.time()
        return {
            "ttft": ft - t0 if ft else 0,
            "total_time": t1 - t0,
            "tokens": tc,
            "tps": tc / (t1 - ft) if ft and t1 > ft else 0,
            "content_len": len(content_out),
            "success": True,
        }
    except Exception as e:
        return {"error": str(e), "success": False, "total_time": time.time() - t0}


async def benchmark(model_name: str, model_path: str, max_tokens: int = 256):
    print(f"\n{'='*65}")
    print(f"🚀 测试模型: {model_name}")
    print(f"   路径: {model_path}")
    print(f"   max_tokens: {max_tokens}")
    print(f"{'='*65}")

    all_results = []
    for c in CONCURRENCIES:
        prompts = PROMPTS[:c]

        t_start = time.time()
        tasks = [single_request(model_path, p, i, max_tokens) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - t_start

        ok = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not ok:
            print(f"\n  ❌ {c}并发: 全部失败!")
            all_results.append({"concurrency": c, "error": True})
            continue

        avg_ttft = sum(r["ttft"] for r in ok) / len(ok)
        avg_tps = sum(r["tps"] for r in ok) / len(ok)
        total_tokens = sum(r["tokens"] for r in ok)
        throughput = total_tokens / total_time
        min_tps = min(r["tps"] for r in ok)
        max_tps = max(r["tps"] for r in ok)

        tag = ""
        if failed:
            tag = f"  ⚠️ {len(failed)}个失败"

        print(f"\n  📊 {c}并发 ({len(ok)}/{c}成功){tag}")
        print(f"     总耗时:     {total_time:.2f}s")
        print(f"     平均 TTFT:  {avg_ttft:.2f}s")
        print(f"     单请求 TPS: {avg_tps:.1f} tok/s (min: {min_tps:.1f}, max: {max_tps:.1f})")
        print(f"     总吞吐量:   {throughput:.1f} tok/s")
        print(f"     总 token:   {total_tokens}")

        all_results.append({
            "concurrency": c,
            "total_time": total_time,
            "avg_ttft": avg_ttft,
            "avg_tps": avg_tps,
            "throughput": throughput,
            "total_tokens": total_tokens,
            "success_count": len(ok),
            "fail_count": len(failed),
        })

        await asyncio.sleep(1)

    # 总结表
    base_thru = all_results[0]["throughput"] if all_results and not all_results[0].get("error") else 0
    print(f"\n{'='*65}")
    print(f"📈 {model_name} — 并发扩展性总结")
    print(f"{'='*65}")
    print(f"{'并发':<6} {'耗时(s)':<10} {'TTFT(s)':<10} {'单TPS':<12} {'总吞吐(tok/s)':<16} {'加速比':<8}")
    print("-" * 62)
    for r in all_results:
        if r.get("error"):
            print(f"{r['concurrency']:<6} ❌ 失败")
        else:
            speedup = r["throughput"] / base_thru if base_thru > 0 else 0
            print(f"{r['concurrency']:<6} {r['total_time']:<10.2f} {r['avg_ttft']:<10.2f} {r['avg_tps']:<12.1f} {r['throughput']:<16.1f} {speedup:<8.1f}x")

    return all_results


async def main():
    # 检查服务器
    try:
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"✅ 服务器在线，模型: {model_ids}")
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        print("请先启动: bash start_server.sh")
        return

    results_15b = None
    results_27b = None

    # 判断当前运行的是哪个模型
    if MODEL_15B in model_ids or any("1.5B" in m for m in model_ids):
        results_15b = await benchmark("Qwen2.5-1.5B-Instruct", MODEL_15B, max_tokens=256)

    if MODEL_27B in model_ids or any("27B" in m or "qwen3.6" in m for m in model_ids):
        results_27b = await benchmark("Qwen3.6-27B-AWQ (awq_marlin+cuDNN)", MODEL_27B, max_tokens=256)

    # 对比
    if results_15b and results_27b:
        print(f"\n{'='*65}")
        print(f"⚡ 1.5B vs 27B-AWQ 对比")
        print(f"{'='*65}")
        print(f"{'并发':<6} {'1.5B吞吐':<14} {'27B吞吐':<14} {'27B/1.5B':<10}")
        print("-" * 44)
        for r1, r2 in zip(results_15b, results_27b):
            if not r1.get("error") and not r2.get("error"):
                ratio = r2["throughput"] / r1["throughput"] if r1["throughput"] > 0 else 0
                print(f"{r1['concurrency']:<6} {r1['throughput']:<14.1f} {r2['throughput']:<14.1f} {ratio:<10.2f}x")


if __name__ == "__main__":
    print("=" * 65)
    print("  vLLM TPS 基准测试：1.5B vs 27B-AWQ")
    print("  并发: 1, 2, 3, 4 | 提示词: 技术问答 | temperature: 0.0")
    print("=" * 65)
    asyncio.run(main())
