#!/usr/bin/env python3
"""
顶层自动化 benchmark 封装(vllm_musa,tp1 单卡一批 8 模型)。

解决交接脚本的三个问题:
  1. 手动改模型路径   -> 从 tp1_bench_params.json 读模型列表,自动写入 run_list
  2. bench_results 覆盖 -> 结果归档到 bench_results/run_<时间戳>/,不覆盖
  3. 固定 rate 测所有模型 -> per-model rate 序列(参数文件驱动)

另外修复 warmup 只热一种输入形状的问题(见 README 6.2):
  正式压测前对每个模型的所有输入长度各预热一遍。

用法:
  python3 auto_bench.py                 # 跑参数文件里的全部模型
  python3 auto_bench.py --models qwen3-8b qwen3-14b   # 只跑指定 served_name
  python3 auto_bench.py --dry-run       # 只打印将要做什么,不真正启动

依赖现有同目录脚本:run_list.py(启动/停止/状态)、test_list.py(压测)、
bench_serving.py(压测引擎)。本脚本通过环境变量把"模型列表 + per-model cases"
传给它们,不改写它们的磁盘文件(run_list/test_list 已改造为可读环境变量)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ----- 路径配置 --------------------------------------------------------------
# 显式硬编码 bench 目录(与 run_list.py / test_list.py 里的硬编码路径保持一致)。
# 不用 Path(__file__) 推断:4.127 上 /mnt/seed17/001688 底层 mount 指向
# /mnt/si0003568lza/default,__file__ 会被解析成后者,与参数文件的 /mnt/seed17
# 前缀不一致(物理同一目录,但路径名混淆)。可用 AUTOBENCH_DIR 覆盖(便于本地测)。
BENCH_DIR = Path(os.environ.get(
    "AUTOBENCH_DIR", "/mnt/seed17/001688/models/Qwen/bench"))
PARAM_FILE = BENCH_DIR / "tp1_bench_params.json"
RUN_LIST = BENCH_DIR / "run_list.py"
TEST_LIST = BENCH_DIR / "test_list.py"
RESULTS_ROOT = BENCH_DIR / "bench_results"
RUN_LOG = BENCH_DIR / "run_list.nohup.log"
TEST_LOG = BENCH_DIR / "test_list.nohup.log"
# run_list.py 把每个服务的 vllm 日志写到 BENCH_DIR/run_logs/<idx>_<served_name>.log,
# load_progress 从这里抓加载进度。须与 run_list.py 的 LOG_DIR 指向同一目录。
LOG_DIR = BENCH_DIR / "run_logs"
RUN_PID = BENCH_DIR / "run_list.nohup.pid"
TEST_PID = BENCH_DIR / "test_list.nohup.pid"

READY_TIMEOUT_SEC = 1800      # 等所有服务就绪的上限(8 模型加载可能较久)
READY_POLL_SEC = 15
TEST_POLL_SEC = 30


# ----- 工具函数 --------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[auto_bench {datetime.now():%H:%M:%S}] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":  # type: ignore
    log(f"ERROR: {msg}")
    raise SystemExit(1)


def load_params(selected: list[str] | None, tp: int = 1) -> dict:
    if not PARAM_FILE.exists():
        fail(f"参数文件不存在: {PARAM_FILE}")
    params = json.loads(PARAM_FILE.read_text(encoding="utf-8"))
    if selected:
        params = {k: v for k, v in params.items()
                  if v.get("served_name") in selected or k in selected}
        if not params:
            fail(f"--models 指定的模型在参数文件里都找不到: {selected}")
    # 多卡部署时每个 server 占 tp 张卡,总卡数 = 模型数 × tp,不能超过 8。
    if len(params) * tp > 8:
        fail(f"本批 {len(params)} 个模型 × tp{tp} = {len(params) * tp} 卡 > 8 卡,"
             f"请分批(用 --models 指定)或减小 --tp")
    return params


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


# ----- 进度展示:ASCII 边框表格 ----------------------------------------------

def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """渲染 ┌┬┐ 边框表格;按显示宽度(中文算 2)对齐。"""
    def w(s: str) -> int:
        return sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))

    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(w(c) for c in col) for col in cols]

    def pad(s: str, width: int) -> str:
        return str(s) + " " * (width - w(s))

    def line(l: str, m: str, r: str) -> str:
        return l + m.join("─" * (wd + 2) for wd in widths) + r

    def row(cells: list[str]) -> str:
        return "│ " + " │ ".join(pad(c, wd) for c, wd in zip(cells, widths)) + " │"

    out = [line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
    for r in rows:
        out.append(row(r))
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


# 模型权重加载进度:从 run_logs/<idx>_<name>.log 抓最新百分比
_LOAD_PAT = re.compile(r"(?:Loading safetensors checkpoint shards|"
                       r"Prefetching checkpoint files):\s*(\d+)%")


def load_progress(params: dict) -> list[list[str]]:
    """返回 [[模型, 加载进度, 状态], ...]。靠 run_list status 判断是否就绪。"""
    # 端口就绪情况
    rows = []
    for idx, (name, v) in enumerate(params.items()):
        log_f = LOG_DIR / f"{idx}_{served_name_of(v['model_path'])}.log"
        pct = "—"
        up = False
        txt = tail(log_f, 200)
        if txt:
            m = list(_LOAD_PAT.finditer(txt))
            if m:
                pct = m[-1].group(1) + "%"
            if "Application startup complete" in txt or "Uvicorn running" in txt \
                    or "Starting vLLM API server" in txt and "init engine" in txt:
                pass
        rows.append([name, pct, ""])  # 状态列稍后由调用方填(就绪/加载中)
    return rows


def served_name_of(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/")).lower()


# 测试 case 进度:直接读 CSV 行数(只增不减,根治"完成数倒退"bug)
def case_progress(result_dir: Path, params: dict) -> tuple[list[list[str]], int, int]:
    """返回 ([[模型, 已完成/总, 状态], ...], 完成总数, case总数)。"""
    rows = []
    done_all = total_all = 0
    for v in params.values():
        name = served_name_of(v["model_path"])
        total = v["num_cases"]
        total_all += total
        # CSV 文件名: benchmark_<served_name>_<ts>.csv;ts 未知,glob 匹配
        csvs = list(result_dir.glob(f"benchmark_{name}_*.csv"))
        done = 0
        if csvs:
            try:
                done = max(0, sum(1 for _ in csvs[0].open()) - 1)  # 减表头
            except Exception:
                done = 0
        done_all += done
        if done >= total:
            status = "✅ 已完成"
        elif done == 0:
            status = "等待/预热"
        elif done >= total * 0.8:
            status = "快收尾"
        else:
            status = f"进行中 {done * 100 // total}%"
        rows.append([name, f"{done} / {total}", status])
    return rows, done_all, total_all


# ----- 阶段 1:启动 8 个服务 --------------------------------------------------

def launch_servers(params: dict, env: dict) -> subprocess.Popen:
    log(f"启动 {len(params)} 个服务 ...")
    # run_list.py 读 AUTOBENCH_MODELS(JSON: [model_path, ...])
    env["AUTOBENCH_MODELS"] = json.dumps([v["model_path"] for v in params.values()])
    # per-model 额外启动参数(extra_args 字段),run_list 按 model_path 拼接
    extra = {v["model_path"]: v["extra_args"]
             for v in params.values() if v.get("extra_args")}
    if extra:
        env["AUTOBENCH_EXTRA_ARGS"] = json.dumps(extra)
        for mp, args in extra.items():
            log(f"  额外参数 {os.path.basename(mp)}: {' '.join(args)}")
    with RUN_LOG.open("w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(RUN_LIST), "--no-wait"],
            stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(BENCH_DIR),
        )
    RUN_PID.write_text(str(proc.pid))
    return proc


def wait_servers_ready(params: dict, env: dict) -> None:
    """轮询直到全部 UP 或超时;用 ASCII 表格显示每个模型加载进度(#14)。"""
    deadline = time.time() + READY_TIMEOUT_SEC
    n = len(params)
    # 端口 → 模型名,用于把 status 的 UP/DOWN 对上行
    port_up = {}
    while time.time() < deadline:
        out = subprocess.run(
            [sys.executable, str(RUN_LIST), "status"],
            capture_output=True, text=True, env=env, cwd=str(BENCH_DIR),
        ).stdout
        # run_list status 每行形如 "  UP   gpu=0 port=8000 name"
        for ln in out.splitlines():
            for idx in range(n):
                if f"port={8000 + idx}" in ln:
                    port_up[idx] = ("UP" in ln)
        up = sum(1 for v in port_up.values() if v)

        # 组装加载进度表格
        rows = load_progress(params)
        for idx, r in enumerate(rows):
            r[2] = "✅ 就绪" if port_up.get(idx) else (
                "加载中" if r[1] != "—" else "启动中")
        print(render_table([f"模型({up}/{n} 就绪)", "加载进度", "状态"], rows),
              flush=True)

        if up >= n:
            log("[all-ready] 全部服务就绪")
            return
        rlog = tail(RUN_LOG, 60)
        for kw in ("Traceback", "CUDA error", "MUSA error", "core dumped",
                   "Address already in use", "OOM", "out of memory"):
            if kw in rlog:
                log(f"!! run_list 日志出现 '{kw}',可能有服务启动失败:")
                print(rlog[-2000:])
                break
        time.sleep(READY_POLL_SEC)
    log("部分服务未就绪,日志末尾:")
    print(tail(RUN_LOG, 80))
    fail(f"等待服务就绪超时({READY_TIMEOUT_SEC}s)")


# ----- 阶段 2:跑压测 --------------------------------------------------------

def run_benchmark(result_dir: Path, env: dict, params: dict) -> None:
    log(f"开始压测,结果目录: {result_dir}")
    # test_list.py 读 AUTOBENCH_PARAM_FILE(per-model cases)+ 结果目录参数
    env["AUTOBENCH_PARAM_FILE"] = str(PARAM_FILE)
    with TEST_LOG.open("w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(TEST_LIST), str(result_dir)],
            stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(BENCH_DIR),
        )
    TEST_PID.write_text(str(proc.pid))

    # 监控直到进程退出;用 ASCII 表格显示 case 进度(#12,直接读 CSV,只增不减)
    while proc.poll() is None:
        time.sleep(TEST_POLL_SEC)
        rows, done_all, total_all = case_progress(result_dir, params)
        pct = done_all * 100 // total_all if total_all else 0
        rows.append(["合计", f"{done_all} / {total_all}", f"{pct}%"])
        print(render_table(["模型", "已完成 / 总", "状态"], rows), flush=True)
        # 错误监控仍看日志(不再用日志数完成数)
        tlog = tail(TEST_LOG, 50)
        for kw in ("Traceback", "CUDA error", "MUSA error", "not reachable"):
            if kw in tlog:
                log(f"!! test_list 日志出现 '{kw}':")
                print(tail(TEST_LOG, 40))
                break

    rc = proc.returncode
    full = tail(TEST_LOG, 200)
    if "[done]" in full or "elapsed=" in full:
        log("压测正常结束([done])")
    if rc != 0:
        log(f"test_list 退出码非零: {rc},日志末尾:")
        print(tail(TEST_LOG, 60))
        fail("压测异常结束")


# ----- 阶段 3:停服务 -------------------------------------------------------

def stop_servers(env: dict) -> None:
    log("停止所有服务 ...")
    subprocess.run([sys.executable, str(RUN_LIST), "stop"],
                   env=env, cwd=str(BENCH_DIR))


# ----- 主流程 --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="只跑指定 served_name(默认全部)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-alive", action="store_true",
                    help="压测后不停服务(便于手动复测)")
    ap.add_argument("--param-file", default=None,
                    help="指定参数文件(默认 tp1_bench_params.json;两轮找拐点用 "
                         "tp1_bench_params_coarse.json / _fine.json)")
    ap.add_argument("--tp", type=int, default=1,
                    help="张量并行卡数(多卡部署,默认 1=单卡)。每个模型占 tp 张卡,"
                         "总卡数 模型数×tp 不能超过 8")
    args = ap.parse_args()

    if args.param_file:
        global PARAM_FILE
        PARAM_FILE = Path(args.param_file)
        if not PARAM_FILE.is_absolute():
            PARAM_FILE = BENCH_DIR / args.param_file
        log(f"使用参数文件: {PARAM_FILE}")

    params = load_params(args.models, args.tp)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"run_{ts}"

    log(f"本批模型({len(params)}),tp={args.tp},占用 {len(params) * args.tp} 卡:")
    for name, v in params.items():
        rates = sorted({c[3] for c in v["cases"]})
        log(f"  {name:16s} port=auto cases={v['num_cases']} "
            f"rate={min(rates)}~{max(rates)}")
    log(f"结果将归档到: {result_dir}")

    if args.dry_run:
        log("--dry-run:仅打印,未启动任何服务。")
        return 0

    result_dir.mkdir(parents=True, exist_ok=True)
    # 把本批参数快照存进结果目录,便于追溯
    (result_dir / "params_snapshot.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2))

    env = os.environ.copy()
    env["AUTOBENCH_TP"] = str(args.tp)   # run_list.py 读它决定每个 server 占几张卡
    try:
        launch_servers(params, env)
        wait_servers_ready(params, env)
        run_benchmark(result_dir, env, params)
    finally:
        if not args.keep_alive:
            stop_servers(env)
        else:
            log("--keep-alive:服务保留运行,记得手动 run_list.py stop")

    log(f"完成。结果在 {result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
