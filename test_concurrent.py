#!/usr/bin/env python3
"""Qwen3.6-27B-AWQ 并发测试 (RTX 4090 24GB)"""
import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

CONCURRENT_REQUESTS = 4
PROMPTS = [
    "用Python写一个快速排序算法，并解释时间复杂度。",
    "什么是Mamba架构？它和Transformer有什么区别？",
    "解释一下Docker和Kubernetes的关系，以及它们各自的用途。",
    "写一个简单的HTTP服务器，支持GET和POST请求。",
    "解释什么是RESTful API设计原则。",
    "如何优化MySQL查询性能？给出至少5个建议。",
    "解释TCP三次握手和四次挥手的过程。",
    "什么是微服务架构？它有什么优缺点？",
]

async def send_request(prompt: str, request_id: int):
    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        stream = await client.chat.completions.create(
            model="qwen3.6-27b",
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手，请简洁回答问题。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            # Qwen3.6 reasoning model: reasoning in model_extra, content later
            text = (delta.model_extra.get('reasoning') if delta.model_extra else None) or delta.content or ""
            if text:
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += text
                token_count += 1

        end_time = time.time()
        ttft = first_token_time - start_time if first_token_time else 0
        total_time = end_time - start_time
        tps = token_count / (end_time - first_token_time) if first_token_time and end_time > first_token_time else 0

        print(f"[请求 {request_id}] 完成!")
        print(f"  TTFT: {ttft:.2f}s | 总耗时: {total_time:.2f}s")
        print(f"  生成token: {token_count} | 速度: {tps:.1f} tok/s")
        print(f"  回复长度: {len(full_response)} 字符")
        print()

        return {
            "request_id": request_id,
            "ttft": ttft,
            "total_time": total_time,
            "token_count": token_count,
            "tps": tps,
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        print(f"[请求 {request_id}] 失败: {e}")
        return {
            "request_id": request_id,
            "error": str(e),
            "total_time": end_time - start_time,
            "success": False
        }

async def main():
    print(f"=== vLLM 并发测试 (Qwen3.6-27B-AWQ | RTX 4090 24GB) ===")
    print(f"并发数: {CONCURRENT_REQUESTS}")
    print("=" * 50)
    print()

    print("检查服务器状态...")
    try:
        models = await client.models.list()
        print(f"可用模型: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"无法连接到服务器: {e}")
        print("请先启动服务器: bash start_server.sh")
        return

    print()

    tasks = []
    for i, prompt in enumerate(PROMPTS[:CONCURRENT_REQUESTS]):
        print(f"发起请求 {i+1}: {prompt[:40]}...")
        tasks.append(send_request(prompt, i+1))

    print()
    print("=" * 50)
    print("等待所有请求完成...")
    print("=" * 50)
    print()

    start_all = time.time()
    results = await asyncio.gather(*tasks)
    total_all = time.time() - start_all

    success_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    print("=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"总耗时: {total_all:.2f}s")
    print(f"成功: {len(success_results)}/{len(results)}")
    print(f"失败: {len(failed_results)}/{len(results)}")

    if success_results:
        avg_ttft = sum(r["ttft"] for r in success_results) / len(success_results)
        avg_tps = sum(r["tps"] for r in success_results) / len(success_results)
        total_tokens = sum(r["token_count"] for r in success_results)
        throughput = total_tokens / total_all

        print(f"平均TTFT: {avg_ttft:.2f}s")
        print(f"平均单请求速度: {avg_tps:.1f} tok/s")
        print(f"总吞吐量: {throughput:.1f} tok/s")
        print(f"总生成token数: {total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
