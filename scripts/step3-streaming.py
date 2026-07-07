#!/usr/bin/env python3
"""第3步: 理解流式输出和性能指标

学习目标:
  - TTFT (Time To First Token): 首 token 延迟
  - TPS (Tokens Per Second): 单个请求生成速度
  - 流式 vs 非流式的区别

运行: python scripts/step3-streaming.py
"""
import asyncio
import time
from openai import AsyncOpenAI

# WSL 环境需要替换为实际 IP: hostname -I
import socket
WSL_IP = "192.168.31.46"  # 用 hostname -I 获取

client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://{WSL_IP}:8000/v1")


async def compare_streaming_vs_non_streaming():
    prompt = "请用Python写一个二分查找算法，并解释其时间复杂度。"
    model = "/home/sean/projects/vllm/models/Qwen2.5-1.5B-Instruct"

    # ========== 流式请求 ==========
    print("=" * 60)
    print("📡 流式请求 (stream=True)")
    print("=" * 60)

    t0 = time.time()
    first_token_time = None
    token_count = 0
    preview = ""

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - t0
                print(f"  ⏱  TTFT: {ttft:.2f}s (第一个字出现的时间)")
            content = chunk.choices[0].delta.content
            preview += content
            token_count += 1

    t1 = time.time()
    print(f"  📊 总 token数: {token_count}")
    print(f"  ⏱  总耗时: {t1 - t0:.2f}s")
    print(f"  🚀 生成速度: {token_count / (t1 - first_token_time):.1f} tok/s")
    print(f"  📝 回复预览: {preview[:100]}...")
    print()

    # ========== 非流式请求 ==========
    print("=" * 60)
    print("📦 非流式请求 (stream=False)")
    print("=" * 60)

    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.7,
        stream=False,
    )
    t1 = time.time()

    content = response.choices[0].message.content
    token_count = len(content)
    print(f"  ⏱  总耗时: {t1 - t0:.2f}s (一次性返回)")
    print(f"  📊 回复长度: {token_count} 字符")
    print(f"  ⚠️  无法知道 TTFT（一次性返回）")
    print()

    # ========== 总结 ==========
    print("=" * 60)
    print("📚 学习要点")
    print("=" * 60)
    print("""
  TTFT (Time To First Token):
    - 用户感知延迟的核心指标
    - 流式可以减少用户等待感
    - 受 prompt 长度和模型大小影响

  TPS (Tokens Per Second):
    - 单请求的生成速度
    - 受模型大小、量化精度影响

  vLLM 的 continuous batching:
    - 多个请求可以共享同一批次计算
    - 这就是并发吞吐量高的原因
    - 下一步学习！
""")


if __name__ == "__main__":
    print("第3步: 理解流式输出和性能指标\n")
    asyncio.run(compare_streaming_vs_non_streaming())
