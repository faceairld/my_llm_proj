#!/usr/bin/env python3
"""验证 Path A 修复后的模型输出语义
   用法: 在容器里跑(vllm 已启动 + 端口 8001)
         python3 /data/my_vllm_test/show_output.py
   它做的事:
     - 发 4 条 long_context prompts(跟 benchmark 一样,前 99% 共享前缀)
     - 第 1 条不命中(冷启动),其余 3 条触发 prefix-cache + 走 Path A
     - 完整打印每条的输出文本,看语义是否合理
"""
import asyncio
from openai import AsyncOpenAI

LONG_CTX = (
    "Transformer 架构由 Vaswani 等人在 2017 年提出,其核心创新是自注意力机制。"
    "相比传统的 RNN,它能更好地并行化训练,且在长距离依赖建模上表现优异。"
    "encoder-decoder 结构、multi-head attention、位置编码、layer normalization、"
    "残差连接共同构成了它的骨架。后续的 GPT、BERT、T5 等模型都基于这一架构演化而来。"
    "近年来 LLM 的进步主要源于:参数规模的扩张、训练数据的增长、"
    "训练策略(RLHF、DPO)的改进、以及推理优化技术(KV cache、PagedAttention、"
    "speculative decoding)的发展。"
) * 6  # 6 倍重复,约 3600 token,跟 benchmark 一致

PROMPTS = [
    f"请总结以下文本的要点,用三句话:\n\n{LONG_CTX}\n\n问题角度{i+1}"
    for i in range(4)
]


async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")

    # 探活
    try:
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"✓ Connected to vllm, model = {model_id}\n")
    except Exception as e:
        print(f"❌ Cannot connect to vllm: {e}")
        return

    for i, prompt in enumerate(PROMPTS):
        print(f"{'='*60}")
        print(f"Request #{i+1}  (问题角度{i+1})")
        if i == 0:
            print("(冷启动,无 prefix-cache 命中,走原 fresh prefill 路径)")
        else:
            print(f"(应触发 prefix-cache 命中,前 ~3600 token 复用 Request #1 的 cache)")
            print("(★ 这次会进入我们的 PATH-A 分支!)")
        print(f"{'='*60}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. 用简洁的中文回答。"},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await client.chat.completions.create(
                model=model_id, messages=messages, temperature=0.7,
                max_tokens=300,
            )
            text = resp.choices[0].message.content
            print(f"\n输出({len(text)} 字符):")
            print("-" * 40)
            print(text)
            print("-" * 40)
        except Exception as e:
            print(f"❌ Request failed: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
