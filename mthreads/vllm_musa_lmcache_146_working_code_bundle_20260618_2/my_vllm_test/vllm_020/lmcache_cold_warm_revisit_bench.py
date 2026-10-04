#!/usr/bin/env python3
"""Cold/warm shared-prefix benchmark for vLLM + LMCache.

This runner intentionally avoids the random shuffle in bench_serving.py.
It sends one cold-fill request for every prefix group, then sends one warm
revisit request for every same group in the same order. Only the warm phase
is intended for LMCache-vs-pure performance comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import requests
from transformers import AutoTokenizer


METRIC_NAMES = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
)


@dataclass
class RequestResult:
    phase: str
    group: int
    prompt_len: int
    output_len: int
    ok: bool
    status: int
    ttft_ms: float | None
    e2e_ms: float
    generated_chars: int
    error: str | None = None


def parse_metric_text(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name in METRIC_NAMES:
            if line.startswith(name + "{") or line.startswith(name + " "):
                parts = line.split()
                if len(parts) >= 2:
                    values[name] = float(parts[1])
    return values


def fetch_metrics(host: str, port: int) -> dict[str, float]:
    url = f"http://{host}:{port}/metrics"
    text = requests.get(url, timeout=10).text
    values = parse_metric_text(text)
    return {name: values.get(name, 0.0) for name in METRIC_NAMES}


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in METRIC_NAMES}


def sample_text(tokenizer: Any, rng: random.Random, token_count: int) -> str:
    vocab_ids = list(tokenizer.get_vocab().values())
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    usable_ids = [tid for tid in vocab_ids if tid not in special_ids]
    ids = rng.choices(usable_ids, k=token_count)
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_requests(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    rng = random.Random(args.seed)

    prefixes = [
        sample_text(tokenizer, rng, args.prefix_len) for _ in range(args.groups)
    ]
    cold_questions = [
        sample_text(tokenizer, rng, args.question_len) for _ in range(args.groups)
    ]
    warm_questions = [
        sample_text(tokenizer, rng, args.question_len) for _ in range(args.groups)
    ]

    rows: list[tuple[str, int, str]] = []
    for group, prefix in enumerate(prefixes):
        rows.append(("cold", group, prefix + "\n\n" + cold_questions[group]))
    for group, prefix in enumerate(prefixes):
        rows.append(("warm", group, prefix + "\n\n" + warm_questions[group]))
    return rows


async def one_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    phase: str,
    group: int,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    generated_chars = 0
    status = 0
    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            async for raw in resp.content:
                now = time.perf_counter()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line == "data: [DONE]":
                    continue
                if not line.startswith("data: "):
                    continue
                if first is None:
                    first = now
                try:
                    item = json.loads(line[len("data: ") :])
                    text = item.get("choices", [{}])[0].get("text", "")
                    generated_chars += len(text)
                except json.JSONDecodeError:
                    pass
        end = time.perf_counter()
        return RequestResult(
            phase=phase,
            group=group,
            prompt_len=prompt_len,
            output_len=output_len,
            ok=200 <= status < 300,
            status=status,
            ttft_ms=None if first is None else (first - start) * 1000,
            e2e_ms=(end - start) * 1000,
            generated_chars=generated_chars,
        )
    except Exception as exc:  # noqa: BLE001 - keep benchmark robust.
        end = time.perf_counter()
        return RequestResult(
            phase=phase,
            group=group,
            prompt_len=prompt_len,
            output_len=output_len,
            ok=False,
            status=status,
            ttft_ms=None,
            e2e_ms=(end - start) * 1000,
            generated_chars=generated_chars,
            error=repr(exc),
        )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(rows: list[RequestResult]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.ok and row.ttft_ms is not None]
    ttft = [row.ttft_ms for row in ok_rows if row.ttft_ms is not None]
    e2e = [row.e2e_ms for row in ok_rows]
    return {
        "requests": len(rows),
        "successful": len(ok_rows),
        "mean_ttft_ms": None if not ttft else statistics.mean(ttft),
        "median_ttft_ms": None if not ttft else statistics.median(ttft),
        "p90_ttft_ms": percentile(ttft, 90),
        "p99_ttft_ms": percentile(ttft, 99),
        "mean_e2e_ms": None if not e2e else statistics.mean(e2e),
        "median_e2e_ms": None if not e2e else statistics.median(e2e),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = build_requests(args)
    encoded_lens = [len(tokenizer.encode(prompt)) for _, _, prompt in rows]
    request_rows = [
        (phase, group, prompt, encoded_lens[i]) for i, (phase, group, prompt) in enumerate(rows)
    ]

    url = f"http://{args.host}:{args.port}/v1/completions"
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    metrics_before = fetch_metrics(args.host, args.port)
    results: list[RequestResult] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for phase in ("cold", "warm"):
            phase_rows = [row for row in request_rows if row[0] == phase]
            for start in range(0, len(phase_rows), args.concurrency):
                batch = phase_rows[start : start + args.concurrency]
                tasks = [
                    one_completion(
                        session=session,
                        url=url,
                        model=args.served_model_name,
                        phase=item_phase,
                        group=group,
                        prompt=prompt,
                        prompt_len=prompt_len,
                        output_len=args.output_len,
                    )
                    for item_phase, group, prompt, prompt_len in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                done = len([row for row in results if row.phase == phase])
                print(f"{phase}: {done}/{len(phase_rows)} done", flush=True)
            if phase == "cold":
                metrics_after_cold = fetch_metrics(args.host, args.port)

    metrics_after_all = fetch_metrics(args.host, args.port)
    cold = [row for row in results if row.phase == "cold"]
    warm = [row for row in results if row.phase == "warm"]

    avg_prompt_len = statistics.mean(encoded_lens) if encoded_lens else 0
    approx_gpu_prefix_capacity = int(args.gpu_kv_tokens // max(1, avg_prompt_len))
    evicted_candidate_count = max(0, args.groups - approx_gpu_prefix_capacity)
    warm_evicted_candidates = [
        row for row in warm if row.group < evicted_candidate_count
    ]

    return {
        "args": vars(args),
        "prompt_len": {
            "min": min(encoded_lens),
            "max": max(encoded_lens),
            "mean": avg_prompt_len,
        },
        "estimated_evicted_candidate_groups": evicted_candidate_count,
        "summary": {
            "all": summarize(results),
            "cold": summarize(cold),
            "warm": summarize(warm),
            "warm_evicted_candidates": summarize(warm_evicted_candidates),
        },
        "metrics": {
            "before": metrics_before,
            "after_cold": metrics_after_cold,
            "after_all": metrics_after_all,
            "cold_delta": metric_delta(metrics_before, metrics_after_cold),
            "warm_delta": metric_delta(metrics_after_cold, metrics_after_all),
            "total_delta": metric_delta(metrics_before, metrics_after_all),
        },
        "results": [asdict(row) for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer", default="/data/SQT-v1.0.5-test/models/qwen3-8b")
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--prefix-len", type=int, default=3072)
    parser.add_argument("--question-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--gpu-kv-tokens", type=int, default=216704)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({k: data[k] for k in ("prompt_len", "estimated_evicted_candidate_groups", "summary", "metrics")}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
