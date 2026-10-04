#!/usr/bin/env python3
"""
Launch 8 vLLM services in parallel, one per GPU, one model each.

Usage:
  python3 /mnt/seed17/001688/models/Qwen/bench/run_list.py            # launch all and wait until ready
  python3 /mnt/seed17/001688/models/Qwen/bench/run_list.py --no-wait  # launch all and return immediately
  python3 /mnt/seed17/001688/models/Qwen/bench/run_list.py status     # poll readiness of each port
  python3 /mnt/seed17/001688/models/Qwen/bench/run_list.py stop       # stop all servers launched here

Edit MODELS / BASE_PORT below to change which models and ports are used.
The list index is used as the GPU index, so MODELS[i] -> GPU i, port BASE_PORT+i.

Logs:  /mnt/seed17/001688/models/Qwen/bench/run_logs/<idx>_<model_name>.log
PIDs:  /mnt/seed17/001688/models/Qwen/bench/run_logs/pids.txt
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

# AUTOBENCH PATCH 2026-05-28: 支持顶层 auto_bench.py 通过环境变量注入模型列表。
# 设了 AUTOBENCH_MODELS(JSON 数组)就用它,否则用下面的默认列表(原手动用法不变)。
import json as _json  # noqa: E402

_DEFAULT_MODELS = [
    "/mnt/seed17/001688/models/Qwen/Qwen3-8B",
    "/mnt/seed17/001688/models/Qwen/Qwen3-8B-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-2B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-2B-Instruct-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-4B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-4B-Instruct-FP8",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-8B-Instruct",
    "/mnt/seed17/001688/models/Qwen/Qwen3-VL-8B-Instruct-FP8",
]

if os.environ.get("AUTOBENCH_MODELS"):
    MODELS = _json.loads(os.environ["AUTOBENCH_MODELS"])
else:
    MODELS = _DEFAULT_MODELS

# per-model 额外启动参数(JSON: {model_path: [extra vllm args...]})。
# auto_bench.py 从 tp1_bench_params.json 的 extra_args 字段注入。
# 用途:给部分模型加 --max-num-seqs / --speculative-config(EAGLE3)等。
EXTRA_ARGS = {}
if os.environ.get("AUTOBENCH_EXTRA_ARGS"):
    EXTRA_ARGS = _json.loads(os.environ["AUTOBENCH_EXTRA_ARGS"])
# END AUTOBENCH PATCH

BASE_PORT = 8000
# AUTOBENCH PATCH 2026-06-11: 支持多卡部署(TP>1)。
# TP 由顶层 auto_bench.py 通过 AUTOBENCH_TP 注入(默认 1,单卡行为不变)。
# 第 idx 个 server 占用卡 [idx*TP, idx*TP+TP-1],见 launch_all 的卡分配。
TP = int(os.environ.get("AUTOBENCH_TP", "1"))
# END AUTOBENCH PATCH
GPU_MEM_UTIL = 0.8
BLOCK_SIZE = 64
READY_TIMEOUT_SEC = 20000
READY_POLL_INTERVAL_SEC = 10

LOG_DIR = Path("/mnt/seed17/001688/models/Qwen/bench/run_logs")
PID_FILE = LOG_DIR / "pids.txt"

# MUSA writes core_*.mudmp into the process cwd on crash; redirect them here
# instead of cluttering /mnt/seed17/001688/models/Qwen/bench/.
CORE_DUMP_DIR = Path("/mnt/seed17/001688/models/Qwen/bench/musa_dumps")

COMPILATION_CONFIG = (
    '{"cudagraph_capture_sizes":'
    "[1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30,32,50,64,100,128,256]}"
)

# AUTOBENCH PD PATCH 2026-06-16: PD 分离(prefill/decode 拆开 + mooncake 搬 KV)。
# 由 auto_bench.py 注入 AUTOBENCH_PD=1 开启。布局:prefill tp占卡[0,TP)、decode tp占卡[TP,2TP)、
# proxy 对外开 PD_PROXY_PORT(=BASE_PORT 8000,test_list 打这个)。
PD_MODE = os.environ.get("AUTOBENCH_PD") == "1"
PD_PROTOCOL = os.environ.get("AUTOBENCH_PD_PROTOCOL", "rdma")  # mooncake 传输协议:rdma / tcp
PD_NUM_WORKERS = os.environ.get("AUTOBENCH_PD_NUM_WORKERS")    # mooncake 发送线程数;None=默认(10)
PD_PREFILL_PORT = 8100
PD_DECODE_PORT = 8200
PD_PROXY_PORT = BASE_PORT  # 8000
BENCH_DIR_PATH = Path("/mnt/seed17/001688/models/Qwen/bench")
PROXY_SCRIPT = BENCH_DIR_PATH / "toy_proxy_server.py"
# END AUTOBENCH PD PATCH

# ----- Helpers ---------------------------------------------------------------


def served_name(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/")).lower()


def build_cmd(model_path: str, port: int, kv_role: str | None = None) -> list[str]:
    cmd = [
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
    # AUTOBENCH PATCH: 追加 per-model 额外参数(--max-num-seqs / --speculative-config 等)
    cmd += [str(a) for a in EXTRA_ARGS.get(model_path, [])]
    # AUTOBENCH PD PATCH 2026-06-16: PD 模式追加 mooncake KV 传输配置
    # (prefill=kv_producer / decode=kv_consumer),走 MooncakeConnector。
    if kv_role:
        extra_cfg = {"mooncake_protocol": PD_PROTOCOL}
        if PD_NUM_WORKERS:                       # 调优:mooncake 发送线程数(默认 10)
            extra_cfg["num_workers"] = int(PD_NUM_WORKERS)
        kv_cfg = _json.dumps({
            "kv_connector": "MooncakeConnector",
            "kv_role": kv_role,
            "kv_connector_extra_config": extra_cfg,
        })
        cmd += ["--kv-transfer-config", kv_cfg]
    # END AUTOBENCH PD PATCH
    return cmd


def sweep_stray_dumps() -> int:
    """Move any existing core_*.mudmp from /mnt/seed17/001688/models/Qwen/bench/ into CORE_DUMP_DIR."""
    CORE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in Path("/mnt/seed17/001688/models/Qwen/bench/").glob("core_*.mudmp"):
        try:
            src.rename(CORE_DUMP_DIR / src.name)
            moved += 1
        except Exception as e:
            print(f"[sweep] could not move {src.name}: {e}")
    if moved:
        print(f"[sweep] moved {moved} stray core dump(s) to {CORE_DUMP_DIR}")
    return moved


def check_ready(port: int, timeout: float = 3.0, path: str = "/v1/models") -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


# AUTOBENCH PD PATCH 2026-06-16: PD 三件套就绪 = prefill/decode(/v1/models)+ proxy(/healthcheck)
def pd_all_ready() -> bool:
    return (check_ready(PD_PREFILL_PORT)
            and check_ready(PD_DECODE_PORT)
            and check_ready(PD_PROXY_PORT, path="/healthcheck"))
# END AUTOBENCH PD PATCH


def iter_matching_vllm_pids() -> list[tuple[int, int, str]]:
    expected = {
        BASE_PORT + idx: model
        for idx, model in enumerate(MODELS)
    }
    matches: list[tuple[int, int, str]] = []

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().split(b"\0")
        except Exception:
            continue
        args = [arg.decode("utf-8", "replace") for arg in cmdline if arg]
        if not args or "vllm" not in " ".join(args) or "serve" not in args:
            continue

        for port, model in expected.items():
            port_arg = str(port)
            has_port = (
                "--port" in args and args[args.index("--port") + 1:args.index("--port") + 2] == [port_arg]
            ) or f"--port={port_arg}" in args
            if has_port and model in args:
                matches.append((int(proc_dir.name), port, model))
                break

    return matches


def terminate_pid_group(pid: int, label: str) -> bool:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"[stop] SIGTERM {label} pid={pid}")
        return True
    except ProcessLookupError:
        print(f"[stop] {label} pid={pid} already gone")
    except Exception as e:
        print(f"[stop] {label} pid={pid} error: {e}")
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
            # AUTOBENCH PATCH 2026-06-11: 多卡部署时每个 server 占 TP 张卡。
            # idx=0 → 卡 [0..TP-1],idx=1 → 卡 [TP..2TP-1] …… TP=1 时退化为单卡(原行为)。
            gpus = ",".join(str(idx * TP + k) for k in range(TP))
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["MUSA_VISIBLE_DEVICES"] = gpus
            # END AUTOBENCH PATCH

            cmd = build_cmd(model, port)
            print(f"[launch] gpu={gpus} port={port} tp={TP} model={model}")
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


# AUTOBENCH PD PATCH 2026-06-16: PD 分离启动 —— prefill(producer) + decode(consumer) + proxy
def launch_pd() -> list[tuple[int, int, str]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CORE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    sweep_stray_dumps()
    if not MODELS:
        print("[error] PD 模式需要至少一个模型(AUTOBENCH_MODELS)")
        return []
    model = MODELS[0]
    if len(MODELS) != 1:
        print(f"[warn] PD 模式只用第一个模型 {model}(共给了 {len(MODELS)} 个)")
    prefill_gpus = ",".join(str(g) for g in range(TP))
    decode_gpus = ",".join(str(TP + g) for g in range(TP))
    started: list[tuple[int, int, str]] = []
    with PID_FILE.open("w") as pid_f:
        for role, port, gpus, kv in [
            ("prefill", PD_PREFILL_PORT, prefill_gpus, "kv_producer"),
            ("decode", PD_DECODE_PORT, decode_gpus, "kv_consumer"),
        ]:
            log_path = LOG_DIR / f"pd_{role}_{served_name(model)}.log"
            env = os.environ.copy()
            env["VLLM_USE_V1"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["MUSA_VISIBLE_DEVICES"] = gpus
            cmd = build_cmd(model, port, kv_role=kv)
            print(f"[launch-pd] role={role} gpu={gpus} port={port} tp={TP} protocol={PD_PROTOCOL}")
            print(f"            log={log_path}")
            print(f"            cmd={' '.join(cmd)}")
            log_f = log_path.open("w")
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(CORE_DUMP_DIR),
                stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
            )
            pid_f.write(f"{proc.pid}\t{port}\t{model}\n")
            started.append((proc.pid, port, role))

        # proxy:对外端点(8000),把请求 prefill→decode 路由
        proxy_log = LOG_DIR / f"pd_proxy_{served_name(model)}.log"
        proxy_cmd = [
            sys.executable, str(PROXY_SCRIPT),
            "--prefiller-host", "127.0.0.1", "--prefiller-port", str(PD_PREFILL_PORT),
            "--decoder-host", "127.0.0.1", "--decoder-port", str(PD_DECODE_PORT),
            "--port", str(PD_PROXY_PORT),
        ]
        print(f"[launch-pd] role=proxy port={PD_PROXY_PORT}")
        print(f"            cmd={' '.join(proxy_cmd)}")
        plog = proxy_log.open("w")
        pproc = subprocess.Popen(
            proxy_cmd, env=os.environ.copy(), cwd=str(BENCH_DIR_PATH),
            stdout=plog, stderr=subprocess.STDOUT, start_new_session=True,
        )
        pid_f.write(f"{pproc.pid}\t{PD_PROXY_PORT}\t__proxy__\n")
        started.append((pproc.pid, PD_PROXY_PORT, "proxy"))

    print(f"\n[ok] PD: {len(started)} 进程已启动(prefill/decode/proxy)。pid 文件: {PID_FILE}")
    return started
# END AUTOBENCH PD PATCH


def wait_ready() -> bool:
    # AUTOBENCH PD PATCH: PD 模式等 prefill+decode+proxy 三者就绪
    if PD_MODE:
        deadline = time.time() + READY_TIMEOUT_SEC
        while time.time() < deadline:
            if pd_all_ready():
                print("[all-ready] PD prefill+decode+proxy 就绪")
                return True
            print(f"[wait] PD prefill={check_ready(PD_PREFILL_PORT)} "
                  f"decode={check_ready(PD_DECODE_PORT)} "
                  f"proxy={check_ready(PD_PROXY_PORT, path='/healthcheck')}")
            time.sleep(READY_POLL_INTERVAL_SEC)
        print("[timeout] PD 三件套未全部就绪")
        return False
    # END AUTOBENCH PD PATCH
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
    # AUTOBENCH PD PATCH: PD 模式 —— 一行逻辑端点(proxy 8000),三件套全就绪才算 UP。
    # 细节用 ✓/✗(不写 "UP"),避免 auto_bench 的 'UP in line' 误判为已就绪。
    if PD_MODE:
        pre = check_ready(PD_PREFILL_PORT)
        dec = check_ready(PD_DECODE_PORT)
        prx = check_ready(PD_PROXY_PORT, path="/healthcheck")
        overall = "UP  " if (pre and dec and prx) else "DOWN"
        model = MODELS[0] if MODELS else "?"
        mk = lambda b: "✓" if b else "✗"
        print(f"  {overall} gpu=0 port={PD_PROXY_PORT} {served_name(model)}  "
              f"[PD P={mk(pre)} D={mk(dec)} Proxy={mk(prx)}]")
        return
    # END AUTOBENCH PD PATCH
    for idx, model in enumerate(MODELS):
        port = BASE_PORT + idx
        state = "UP  " if check_ready(port) else "DOWN"
        print(f"  {state} gpu={idx} port={port} {served_name(model)}")


def stop_all() -> None:
    stopped_pids: set[int] = set()

    if PID_FILE.exists():
        for line in PID_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if terminate_pid_group(pid, "pidfile"):
                stopped_pids.add(pid)
    else:
        print(f"[stop] no pid file at {PID_FILE}; falling back to process scan")

    fallback_matches = [
        (pid, port, model)
        for pid, port, model in iter_matching_vllm_pids()
        if pid not in stopped_pids
    ]
    if not fallback_matches:
        print("[stop] no matching vllm serve processes found by fallback scan")
        return

    for pid, port, model in fallback_matches:
        terminate_pid_group(pid, f"fallback port={port} model={served_name(model)}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "stop":
        stop_all()
        return 0
    if args and args[0] == "status":
        status()
        return 0

    if PD_MODE:          # AUTOBENCH PD PATCH: PD 分离启动
        launch_pd()
    else:
        launch_all()
    if "--no-wait" in args:
        print("[note] --no-wait given; not polling readiness.")
        return 0
    return 0 if wait_ready() else 1


if __name__ == "__main__":
    sys.exit(main())
