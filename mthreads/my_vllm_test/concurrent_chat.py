#!/usr/bin/env python3
"""
vllm 并发压测 + 抽样展示客户端。

每输入一个问题，会**并发**向 vllm 发 N 条相同请求（默认 10），
等全部返回后随机挑 3 条展示，同时给出整体并发耗时统计。

============================================================
前置条件
============================================================
  1. vllm serve 已启动（默认监听 0.0.0.0:8000）
       cd /data/my_vllm_test
       nohup bash run.sh /data/SETS/models/qwen3-8b 8 32768 > server.log 2>&1 &
       tail -f server.log    # 看到 "Application startup complete" 即可
  2. 本机有 openai 库（gy_work 容器里已装好；宿主机没装）
       所以推荐在容器内运行此脚本

============================================================
用法（在 gy_work 容器里执行）
============================================================
  # 进容器
  docker exec -it gy_work bash

  # 默认: 10 并发，每次随机展示 3 条
  python /data/my_vllm_test/concurrent_chat.py

  # 32 并发，展示 5 条
  python /data/my_vllm_test/concurrent_chat.py -n 32 --show 5

  # 单轮非交互
  python /data/my_vllm_test/concurrent_chat.py --once "讲一个笑话" -n 16

  # 提高温度看回答多样性
  python /data/my_vllm_test/concurrent_chat.py --temperature 1.0

  # 不在默认 8000 端口
  python /data/my_vllm_test/concurrent_chat.py --port 8001

  # 多轮对话保留历史（默认每轮独立，更适合看并发对比）
  python /data/my_vllm_test/concurrent_chat.py --keep-history

  # 一行直接调用（不进容器交互式 shell）
  docker exec -it gy_work python /data/my_vllm_test/concurrent_chat.py -n 32

============================================================
交互式命令
============================================================
  /n <数>     改并发数（例: /n 16）
  /show <数>  改展示数
  /sys <文本> 修改 system prompt（影响下一次请求）
  /reset      清空多轮对话历史（仅在 --keep-history 模式下有用）
  /exit       退出（Ctrl+C 同效）

============================================================
典型输出
============================================================
  你> 讲一个关于程序员的笑话
  发起 10 个并发请求...

  ─── 并发结果: 10/10 成功, 整体耗时 1.83s ───
    TTFT      avg=215ms  min=180ms  max=298ms
    完整耗时   avg=1.65s   min=1.20s   max=1.83s
    总输出字符 4823   ≈ 2636 char/s (聚合吞吐)

  ─── 随机抽样 3/10 ───

  [#2 ttft=210ms total=1.55s]
  程序员到饭店点菜...
  [#5 ttft=235ms total=1.42s]
  ...

============================================================
典型用途 / 观察点
============================================================
  • 对比 `-n 1` vs `-n 32` 的「整体耗时」—— 看 vllm 是否真在并发
  • TTFT min/max 差距 —— 反映调度公平性 / batch 拼接效率
  • 聚合吞吐 (char/s) —— 反映 GPU 真实利用率
  • 同问题、不同回答 —— 验证 temperature 采样是否生效
"""
import argparse
import asyncio
import random
import sys
import time

from openai import AsyncOpenAI


async def one_request(client, model, messages, temperature, max_tokens, idx):
    """单条请求：返回 (idx, text, ttft_s, total_s, ok)"""
    t0 = time.perf_counter()
    first_at = None
    chunks = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_at is None:
                    first_at = time.perf_counter()
                chunks.append(delta)
        text = "".join(chunks)
        total = time.perf_counter() - t0
        ttft = (first_at - t0) if first_at else float("nan")
        return idx, text, ttft, total, True
    except Exception as e:
        total = time.perf_counter() - t0
        return idx, f"[ERROR] {e}", float("nan"), total, False


async def fire_batch(client, model, messages, n, temperature, max_tokens):
    """并发发 n 条相同请求，返回所有结果"""
    tasks = [
        asyncio.create_task(one_request(client, model, messages, temperature, max_tokens, i))
        for i in range(n)
    ]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    return results, wall


def render_batch(results, wall, n, n_show):
    """展示并发批次的统计 + 随机抽样回复"""
    ok = [r for r in results if r[4]]
    fail = [r for r in results if not r[4]]
    n_ok = len(ok)

    if n_ok == 0:
        print(f"\n  ❌ 全部失败 ({len(fail)}/{n})")
        for idx, text, _, _, _ in fail[:3]:
            print(f"  [#{idx}] {text}")
        return

    ttfts = [r[2] for r in ok if not (r[2] != r[2])]  # 排除 nan
    totals = [r[3] for r in ok]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
    min_ttft = min(ttfts) if ttfts else float("nan")
    max_ttft = max(ttfts) if ttfts else float("nan")
    avg_total = sum(totals) / len(totals)
    min_total = min(totals)
    max_total = max(totals)
    total_chars = sum(len(r[1]) for r in ok)

    print()
    print(f"\033[33m─── 并发结果: {n_ok}/{n} 成功, 整体耗时 {wall:.2f}s ───\033[0m")
    print(f"  TTFT      avg={avg_ttft*1000:.0f}ms  min={min_ttft*1000:.0f}ms  max={max_ttft*1000:.0f}ms")
    print(f"  完整耗时   avg={avg_total:.2f}s   min={min_total:.2f}s   max={max_total:.2f}s")
    print(f"  总输出字符 {total_chars}   ≈ {total_chars/wall:.0f} char/s (聚合吞吐)")
    if fail:
        print(f"  ⚠️  {len(fail)} 条失败: {fail[0][1][:120]}")

    # 随机抽样
    sample = random.sample(ok, min(n_show, n_ok))
    sample.sort(key=lambda r: r[0])  # 按原 idx 排序展示
    print(f"\n\033[33m─── 随机抽样 {len(sample)}/{n_ok} ───\033[0m")
    for idx, text, ttft, total, _ in sample:
        head = f"\033[36m[#{idx} ttft={ttft*1000:.0f}ms total={total:.2f}s]\033[0m"
        print(f"\n{head}\n{text}")


async def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="模型名，默认自动选第一个")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--concurrency", "-n", type=int, default=10, help="每次问题的并发数 (默认 10)")
    ap.add_argument("--show", type=int, default=3, help="每次随机展示的回复数 (默认 3)")
    ap.add_argument("--temperature", type=float, default=0.8, help="提高一点 (0.8) 让并发回答更多样")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--once", default=None, help="单轮非交互模式")
    ap.add_argument("--keep-history", action="store_true", help="多轮对话保留历史 (默认每次独立)")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key)

    # 探活
    try:
        models = await client.models.list()
        available = [m.id for m in models.data]
    except Exception as e:
        print(f"[ERROR] 连不上 {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    if not available:
        print(f"[ERROR] 没有可用模型", file=sys.stderr)
        sys.exit(1)

    model = args.model or available[0]
    if model not in available:
        print(f"[ERROR] 模型 '{model}' 不存在。可用: {available}", file=sys.stderr)
        sys.exit(1)

    # 单轮模式
    if args.once is not None:
        msgs = [{"role": "system", "content": args.system}, {"role": "user", "content": args.once}]
        results, wall = await fire_batch(client, model, msgs, args.concurrency, args.temperature, args.max_tokens)
        render_batch(results, wall, args.concurrency, args.show)
        return

    # 交互式
    print(f"=== vllm 并发 chat ===")
    print(f"  endpoint    : {base_url}")
    print(f"  model       : {model}    (可用: {available})")
    print(f"  concurrency : {args.concurrency}    (用 /n <数> 修改)")
    print(f"  show        : {args.show}    (用 /show <数> 修改)")
    print(f"  temperature : {args.temperature}")
    print(f"  history     : {'保留' if args.keep_history else '每次独立'}")
    print(f"  命令: /n /show /sys /reset /exit\n")

    history = []  # 仅在 keep_history 时使用

    while True:
        try:
            user = input("\033[36m你> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = []
            print("[历史已清空]")
            continue
        if user.startswith("/n "):
            try:
                args.concurrency = max(1, int(user.split()[1]))
                print(f"[并发数 → {args.concurrency}]")
            except (ValueError, IndexError):
                print("[用法: /n <数字>]")
            continue
        if user.startswith("/show "):
            try:
                args.show = max(1, int(user.split()[1]))
                print(f"[展示数 → {args.show}]")
            except (ValueError, IndexError):
                print("[用法: /show <数字>]")
            continue
        if user.startswith("/sys"):
            new_sys = user[4:].strip()
            if not new_sys:
                print(f"[当前 system]: {args.system}")
            else:
                args.system = new_sys
                print(f"[system 已更新]")
            continue

        # 构造 messages
        msgs = [{"role": "system", "content": args.system}]
        if args.keep_history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user})

        print(f"\033[35m发起 {args.concurrency} 个并发请求...\033[0m")
        try:
            results, wall = await fire_batch(
                client, model, msgs, args.concurrency, args.temperature, args.max_tokens
            )
        except KeyboardInterrupt:
            print("\n[已中断]")
            continue

        render_batch(results, wall, args.concurrency, args.show)

        if args.keep_history:
            ok = [r for r in results if r[4]]
            if ok:
                # 保留历史时，把抽样里的第一条当 assistant 回复
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": ok[0][1]})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
