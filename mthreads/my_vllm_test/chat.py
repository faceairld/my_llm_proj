#!/usr/bin/env python3
"""
vllm OpenAI 兼容接口的简易 chat 客户端。

用法:
    python chat.py                                  # 交互式聊天，自动列出可用模型
    python chat.py --port 8001                      # 指定端口
    python chat.py --model qwen3-8b                 # 指定模型名
    python chat.py --once "你好，介绍一下你自己"     # 单轮非交互
    python chat.py --no-stream                      # 关闭流式输出
    python chat.py --system "你是一个翻译助手..."     # 自定义 system prompt

交互式命令:
    /reset    清空对话历史
    /save     保存对话到 chat_history.json
    /sys      查看 / 修改 system prompt
    /exit     退出 (Ctrl+C 同效)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI


def list_models(client: OpenAI) -> list[str]:
    return [m.id for m in client.models.list().data]


def stream_chat(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    """流式输出，返回拼接后的完整回复"""
    t0 = time.time()
    first_token_at = None
    chunks = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_at is None:
                first_token_at = time.time()
            chunks.append(delta)
            print(delta, end="", flush=True)
    print()

    full = "".join(chunks)
    elapsed = time.time() - t0
    ttft = (first_token_at - t0) if first_token_at else float("nan")
    n_tok = len(full)  # 字符数当近似 token 数
    tps = n_tok / elapsed if elapsed > 0 else 0
    print(f"\n  [TTFT={ttft*1000:.0f}ms  total={elapsed:.2f}s  ~{tps:.1f} char/s]")
    return full


def non_stream_chat(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    print(text)
    print(f"\n  [total={elapsed:.2f}s  prompt_tokens={resp.usage.prompt_tokens}  completion_tokens={resp.usage.completion_tokens}]")
    return text


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="vllm 服务地址 (默认 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="vllm 服务端口 (默认 8000)")
    ap.add_argument("--model", default=None, help="模型名 (默认自动选第一个)")
    ap.add_argument("--system", default="You are a helpful assistant.", help="system prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    ap.add_argument("--once", default=None, help="单轮非交互，直接给一句话获取回复")
    ap.add_argument("--api-key", default="EMPTY", help="vllm 不校验，随便填")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    # 探活 + 选模型
    try:
        available = list_models(client)
    except Exception as e:
        print(f"[ERROR] 连不上 {base_url}: {e}", file=sys.stderr)
        print("       请确认 vllm serve 已启动，端口正确。", file=sys.stderr)
        sys.exit(1)

    if not available:
        print(f"[ERROR] {base_url} 没有可用模型", file=sys.stderr)
        sys.exit(1)

    model = args.model or available[0]
    if model not in available:
        print(f"[ERROR] 模型 '{model}' 不存在。可用: {available}", file=sys.stderr)
        sys.exit(1)

    chat_fn = non_stream_chat if args.no_stream else stream_chat

    # 单轮模式
    if args.once is not None:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.once},
        ]
        chat_fn(client, model, messages, args.temperature, args.max_tokens)
        return

    # 交互模式
    print(f"=== vllm chat ===")
    print(f"  endpoint : {base_url}")
    print(f"  model    : {model}    (可用: {available})")
    print(f"  stream   : {not args.no_stream}")
    print(f"  system   : {args.system[:60]}{'...' if len(args.system) > 60 else ''}")
    print(f"  命令: /reset 清空 | /save 保存 | /sys 改 system | /exit 退出\n")

    messages = [{"role": "system", "content": args.system}]

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
            messages = [{"role": "system", "content": args.system}]
            print("[历史已清空]")
            continue
        if user == "/save":
            path = Path("chat_history.json")
            path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[已保存到 {path.resolve()}]")
            continue
        if user.startswith("/sys"):
            new_sys = user[4:].strip()
            if not new_sys:
                print(f"[当前 system]: {messages[0]['content']}")
            else:
                messages[0]["content"] = new_sys
                print("[system 已更新；当前历史保留]")
            continue

        messages.append({"role": "user", "content": user})
        print("\033[32m助手> \033[0m", end="", flush=True)
        try:
            reply = chat_fn(client, model, messages, args.temperature, args.max_tokens)
        except KeyboardInterrupt:
            print("\n[已中断]")
            messages.pop()  # 撤回最后一条 user
            continue
        except Exception as e:
            print(f"\n[ERROR] {e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
