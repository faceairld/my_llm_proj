#!/usr/bin/env python3
"""
Run benchmark cases against the 8 vLLM servers launched by run_list.py.
All 8 models are benchmarked in parallel; each model writes its own CSV.

Usage:
  python3 /home/test_list.py                 # default result dir /home/bench_results
  python3 /home/test_list.py /tmp/results    # custom result dir

The MODELS / BASE_PORT config is imported from run_list.py so the two files
stay in sync.

Outputs:
  <result_dir>/benchmark_<model_name>_<timestamp>.csv   (per model)
  <result_dir>/benchmark_summary_<timestamp>.csv        (all models combined)
  <result_dir>/bench_logs/<model_name>.log              (stdout/err per model)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home")
from run_list import MODELS, BASE_PORT, TP, served_name  # noqa: E402

# ----- Configuration ---------------------------------------------------------

BENCH_PY = "/home/bench_serving.py"
DATASET_PATH = "/home/ShareGPT_V3_unfiltered_cleaned_split.json"
WARMUP_REQUESTS = 200
NUM_PROMPTS = 200

# input_len, output_len, label, request_rate
BENCH_CASES: list[tuple[int, int, str, float]] = [
    (2048, 1024, "2k/1k",   3.0),
    (2048, 1024, "2k/1k",   3.2),
    (2048, 1024, "2k/1k",   3.4),
    (3072, 1024, "3k/1k",   2.0),
    (3072, 1024, "3k/1k",   2.2),
    (3072, 1024, "3k/1k",   2.4),
    (3072, 1024, "3k/1k",   2.6),
    (3584, 1024, "3.5k/1k", 2.0),
    (3584, 1024, "3.5k/1k", 2.2),
    (4096, 1024, "4k/1k",   1.0),
    (4096, 1024, "4k/1k",   1.2),
    (4096, 1024, "4k/1k",   1.4),
    (4096, 1024, "4k/1k",   1.6),
    (4096, 1024, "4k/1k",   1.8),
    (4096, 1024, "4k/1k",   2.0),
    (4096, 1536, "4k/1.5k", 1.0),
    (4096, 1536, "4k/1.5k", 1.2),
    (4096, 1536, "4k/1.5k", 1.4),
]

CSV_HEADER = (
    "model_name,tp,input_len,output_len,io_label,request_rate,num_prompts,"
    "req_tp,in_tok_tp,out_tok_tp,"
    "mean_ttft,median_ttft,p99_ttft,"
    "mean_tpot,median_tpot,p99_tpot,"
    "mean_itl,p99_itl,mean_e2e,"
    "real_concurrency,duration,total_input_tokens,total_output_tokens,status"
)

STAT_FIELDS = [
    "request_throughput", "input_throughput", "output_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "mean_e2e_latency_ms", "concurrency", "duration",
    "total_input_tokens", "total_output_tokens",
]

# ----- Helpers ---------------------------------------------------------------


def check_server(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


def parse_stats(json_path: Path) -> str:
    try:
        content = json_path.read_text(encoding="utf-8").strip()
        data = json.loads(content) if content else {}
    except Exception:
        data = {}
    return ",".join(str(data.get(k, 0)) for k in STAT_FIELDS)


def fail_row(name: str, input_len: int, output_len: int,
             label: str, rate: float) -> str:
    zeros = ",".join(["0"] * len(STAT_FIELDS))
    return (f"{name},{TP},{input_len},{output_len},{label},"
            f"{rate},{NUM_PROMPTS},{zeros},FAIL")


# ----- Per-model worker ------------------------------------------------------


def run_one_model(idx: int, model_path: str, port: int,
                  result_dir: Path, ts: str) -> tuple[str, Path, int, int]:
    name = served_name(model_path)
    csv_path = result_dir / f"benchmark_{name}_{ts}.csv"
    log_path = result_dir / "bench_logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w") as f:
        f.write(CSV_HEADER + "\n")

    ok = fail = 0
    log_f = log_path.open("w")
    log_f.write(f"# model={model_path} port={port} gpu={idx}\n")
    log_f.flush()

    if not check_server(port):
        msg = f"[{name}] server at port {port} not reachable - skipping"
        print(msg)
        log_f.write(msg + "\n")
        log_f.close()
        return name, csv_path, ok, len(BENCH_CASES)

    for input_len, output_len, label, rate in BENCH_CASES:
        json_out = result_dir / f"temp_{name}_{input_len}_{output_len}_r{rate}.json"
        print(f"[{name}] case in={input_len} out={output_len} rate={rate}")
        log_f.write(f"\n>>> case in={input_len} out={output_len} rate={rate}\n")
        log_f.flush()

        cmd = [
            "python3", BENCH_PY,
            "--backend", "vllm",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model", model_path,
            "--served-model-name", name,
            "--tokenizer", model_path,
            "--dataset-name", "random",
            "--dataset-path", DATASET_PATH,
            "--num-prompts", str(NUM_PROMPTS),
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--random-range-ratio", "1.0",
            "--request-rate", str(rate),
            "--burstiness", "102",
            "--warmup-requests", str(WARMUP_REQUESTS),
            "--apply-chat-template",
            "--disable-tqdm",
            "--output-file", str(json_out),
        ]
        try:
            subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                           check=False)
        except Exception as e:
            log_f.write(f"[error] {e}\n")

        if json_out.exists():
            stats = parse_stats(json_out)
            row = (f"{name},{TP},{input_len},{output_len},{label},"
                   f"{rate},{NUM_PROMPTS},{stats},OK")
            ok += 1
            try:
                json_out.unlink()
            except Exception:
                pass
        else:
            row = fail_row(name, input_len, output_len, label, rate)
            fail += 1
        with csv_path.open("a") as f:
            f.write(row + "\n")

    log_f.close()
    print(f"[{name}] done ok={ok} fail={fail} -> {csv_path}")
    return name, csv_path, ok, fail


# ----- Main ------------------------------------------------------------------


def main() -> int:
    result_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/bench_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[step 1] Checking {len(MODELS)} servers ...")
    not_ready = []
    for idx, model in enumerate(MODELS):
        port = BASE_PORT + idx
        if check_server(port):
            print(f"  UP    port={port} {served_name(model)}")
        else:
            print(f"  DOWN  port={port} {served_name(model)}")
            not_ready.append((idx, model))

    if not_ready:
        print("\n[error] some servers are not reachable.")
        print("        start them first with: python3 /home/run_list.py")
        return 1

    print(f"\n[step 2] Running benchmarks across {len(MODELS)} models in parallel")
    print(f"         ({len(BENCH_CASES)} cases per model)")
    print(f"         result dir: {result_dir}")
    t0 = time.time()

    results: list[tuple[str, Path, int, int]] = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(
                run_one_model, idx, model, BASE_PORT + idx, result_dir, ts
            ): (idx, model)
            for idx, model in enumerate(MODELS)
        }
        for fut in as_completed(futures):
            idx, model = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[error] {served_name(model)} (gpu={idx}): {e}")

    summary_path = result_dir / f"benchmark_summary_{ts}.csv"
    with summary_path.open("w") as out:
        out.write(CSV_HEADER + "\n")
        for _, csv_path, _, _ in results:
            if not csv_path.exists():
                continue
            with csv_path.open("r") as src:
                next(src, None)
                for line in src:
                    out.write(line)

    elapsed = time.time() - t0
    print("\n[done] " + f"elapsed={elapsed:.1f}s")
    print(f"       summary csv: {summary_path}")
    for name, csv_path, ok, fail in sorted(results):
        print(f"       {name:40s} ok={ok:3d} fail={fail:3d}  {csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
