#!/usr/bin/env python3
"""PD 并发爬坡拆解:并发 K=1/8/32/64 各打一轮,看哪段随并发爆。
- prefill-only(8100):量 prefill 在并发下的耗时(看 prefill 实例饱和没)。
- proxy(8000):量 TTFT / decode-TPOT / 聚合吞吐。
- 搬运+开销 = TTFT - prefill:若随并发暴涨 → mooncake 发送池(num_workers)或代理是瓶颈。
判读:① decode-TPOT 随 K 涨是正常的(批变大);② 关键看"搬运+开销"是否随 K 暴涨。
需 PD 栈已起(prefill 8100 + decode 8200 + proxy 8000)。
"""
import asyncio
import os
import statistics
import time

import httpx

SERVED = "qwen3-32b"
PREFILL = "http://127.0.0.1:8100/v1/completions"
PROXY = "http://127.0.0.1:8000/v1/completions"
LEVELS = [1, 8, 32, 64]
GEN_TOKENS = 64
NW = os.environ.get("AUTOBENCH_PD_NUM_WORKERS", "默认(10)")


def make_prompt(approx_tokens: int = 2048) -> str:
    return ("The quick brown fox jumps over the lazy dog. " * (approx_tokens // 12))


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


async def one_prefill(client, prompt):
    t0 = time.time()
    r = await client.post(PREFILL, json={"model": SERVED, "prompt": prompt,
                                         "max_tokens": 1, "temperature": 0}, timeout=600)
    r.raise_for_status()
    return time.time() - t0


async def one_proxy(client, prompt):
    t0 = time.time(); t_first = None; n = 0
    async with client.stream("POST", PROXY, json={"model": SERVED, "prompt": prompt,
                                                  "max_tokens": GEN_TOKENS, "temperature": 0,
                                                  "stream": True}, timeout=900) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            if "[DONE]" in line:
                break
            if t_first is None:
                t_first = time.time()
            n += 1
    return (t_first - t0 if t_first else None), time.time() - t0, n


async def ramp_level(K, prompt):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
        pres = await asyncio.gather(*[one_prefill(client, prompt) for _ in range(K)])
        wall0 = time.time()
        rres = await asyncio.gather(*[one_proxy(client, prompt) for _ in range(K)])
        wall = time.time() - wall0
    prefill_mean = mean(pres) * 1000
    ttfts = [x[0] for x in rres if x[0] is not None]
    tpots = [(tot - tf) / (n - 1) for (tf, tot, n) in rres if tf and n > 1]
    ttft_mean = mean(ttfts) * 1000
    tpot_mean = mean(tpots) * 1000
    xfer = ttft_mean - prefill_mean
    thr = K / wall
    return prefill_mean, ttft_mean, xfer, tpot_mean, thr


async def main():
    prompt = make_prompt(2048)
    print(f"PD 并发爬坡(输入~2k, 每请求生成 {GEN_TOKENS} token, num_workers={NW})")
    print(f"{'并发K':>5} | {'prefill':>8} | {'TTFT':>8} | {'搬运+开销':>9} | {'decTPOT':>8} | {'吞吐req/s':>9}")
    print("-" * 66)
    # 预热
    try:
        await ramp_level(2, prompt)
    except Exception as e:
        print(f"预热失败: {e}")
    for K in LEVELS:
        try:
            pf, ttft, xfer, tpot, thr = await ramp_level(K, prompt)
            print(f"{K:>5} | {pf:7.0f}m | {ttft:7.0f}m | {xfer:8.0f}m | {tpot:7.1f}m | {thr:8.2f}")
        except Exception as e:
            print(f"{K:>5} | 失败: {e}")
    print("\n判读:")
    print("  · '搬运+开销'(=TTFT−prefill)随并发 K 暴涨 → mooncake 发送池或代理是瓶颈。")
    print("    再跑一次 num_workers=32 对比:若这列明显回落 → 是发送池(连接器);")
    print("    若不回落 → 是 toy 代理(要换 disagg 路由)。")
    print("  · decTPOT 随 K 上升是正常(批变大);prefill 随 K 暴涨 = prefill 实例饱和。")


if __name__ == "__main__":
    asyncio.run(main())
