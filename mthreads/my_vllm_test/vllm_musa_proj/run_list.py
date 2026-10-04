#!/usr/bin/env python3
"""
Launch 8 vLLM services in parallel, one per GPU, one model each.

Usage:
  python3 /home/run_list.py            # launch all and wait until ready
  python3 /home/run_list.py --no-wait  # launch all and return immediately
  python3 /home/run_list.py status     # poll readiness of each port
  python3 /home/run_list.py stop       # stop all servers launched here

Edit MODELS / BASE_PORT below to change which models and ports are used.
The list index is used as the GPU index, so MODELS[i] -> GPU i, port BASE_PORT+i.

Logs:  /home/run_logs/<idx>_<model_name>.log
PIDs:  /home/run_logs/pids.txt
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ----- Configuration ---------------------------------------------------------

MODELS = [
    "/mnt/seed17/001688/models/Qwen/Qwen3-8B",
    "/mnt/seed17/001688/models/Qwen/Qwen3-8B-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-2B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-2B-Instruct-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-4B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-4B-Instruct-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-8B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-8B-Instruct-FP8",
]

BASE_PORT = 8000
TP = 1
GPU_MEM_UTIL = 0.8
BLOCK_SIZE = 64
READY_TIMEOUT_SEC = 20000
READY_POLL_INTERVAL_SEC = 10

LOG_DIR = Path("/home/run_logs")
PID_FILE = LOG_DIR / "pids.txt"

# MUSA writes core_*.mudmp into the process cwd on crash; redirect them here
# instead of cluttering /home.
CORE_DUMP_DIR = Path("/home/musa_dumps")

COMPILATION_CONFIG = (
    '{"cudagraph_capture_sizes":'
    "[1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30,32,50,64,100,128,256]}"
)

# ----- Helpers ---------------------------------------------------------------


def served_name(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/")).lower()


def build_cmd(model_path: str, port: int) -> list[str]:
    return [
        "vllm", "serve", model_path,
        "--trust-remote-code",
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--served-model-name", served_name(model_path),
        "--block-size", str(BLOCK_SIZE),
        "--tensor-parallel-size", str(TP),
        "--pipeline-parallel-size", "1",
        "--port", str(port),
        "--compilation-config", COMPILATION_CONFIG,
    ]


def sweep_stray_dumps() -> int:
    """Move any existing core_*.mudmp from /home into CORE_DUMP_DIR."""
    CORE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in Path("/home").glob("core_*.mudmp"):
        try:
            src.rename(CORE_DUMP_DIR / src.name)
            moved += 1
        except Exception as e:
            print(f"[sweep] could not move {src.name}: {e}")
    if moved:
        print(f"[sweep] moved {moved} stray core dump(s) to {CORE_DUMP_DIR}")
    return moved


def check_ready(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


# ----- Actions ---------------------------------------------------------------


def launch_all() -> list[tuple[int, int, str]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CORE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    sweep_stray_dumps()

    if len(MODELS) != 8:
        print(f"[warn] MODELS has {len(MODELS)} entries (expected 8). Continuing.")

    started: list[tuple[int, int, str]] = []
    with PID_FILE.open("w") as pid_f:
        for idx, model in enumerate(MODELS):
            port = BASE_PORT + idx
            log_path = LOG_DIR / f"{idx}_{served_name(model)}.log"

            env = os.environ.copy()
            env["VLLM_USE_V1"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            env["MUSA_VISIBLE_DEVICES"] = str(idx)

            cmd = build_cmd(model, port)
            print(f"[launch] gpu={idx} port={port} model={model}")
            print(f"         log={log_path}")
            print(f"         cmd={' '.join(cmd)}")

            log_f = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(CORE_DUMP_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid_f.write(f"{proc.pid}\t{port}\t{model}\n")
            started.append((proc.pid, port, model))

    print(f"\n[ok] {len(started)} servers spawned. pid file: {PID_FILE}")
    return started


def wait_ready() -> bool:
    deadline = time.time() + READY_TIMEOUT_SEC
    pending = [(i, BASE_PORT + i, m) for i, m in enumerate(MODELS)]
    while pending and time.time() < deadline:
        still_pending: list[tuple[int, int, str]] = []
        for idx, port, model in pending:
            if check_ready(port):
                print(f"[ready] gpu={idx} port={port} {served_name(model)}")
            else:
                still_pending.append((idx, port, model))
        pending = still_pending
        if pending:
            ports = ", ".join(str(p) for _, p, _ in pending)
            print(f"[wait] {len(pending)} still loading (ports: {ports})")
            time.sleep(READY_POLL_INTERVAL_SEC)

    if pending:
        print(f"[timeout] not ready after {READY_TIMEOUT_SEC}s: "
              f"{[(i, p) for i, p, _ in pending]}")
        return False
    print("[all-ready]")
    return True


def status() -> None:
    for idx, model in enumerate(MODELS):
        port = BASE_PORT + idx
        state = "UP  " if check_ready(port) else "DOWN"
        print(f"  {state} gpu={idx} port={port} {served_name(model)}")


def stop_all() -> None:
    if not PID_FILE.exists():
        print(f"[stop] no pid file at {PID_FILE}; nothing to do")
        return
    for line in PID_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"[stop] SIGTERM pid={pid}")
        except ProcessLookupError:
            print(f"[stop] pid={pid} already gone")
        except Exception as e:
            print(f"[stop] pid={pid} error: {e}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "stop":
        stop_all()
        return 0
    if args and args[0] == "status":
        status()
        return 0

    launch_all()
    if "--no-wait" in args:
        print("[note] --no-wait given; not polling readiness.")
        return 0
    return 0 if wait_ready() else 1


if __name__ == "__main__":
    sys.exit(main())
