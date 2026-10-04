#!/usr/bin/env python3
"""PD 过载观测采样器:抓 prefill(8100)/decode(8200)的请求队列 + KV 填充率 + 卡 util。
目的:区分"prefill 真算不过来"(卡0-3打满)vs"prefill 空等交接"(卡0-3闲 + decode KV满反压)。
采样:  python3 pd_monitor.py [out.csv]
汇总:  python3 pd_monitor.py --summary out.csv
"""
import csv
import re
import subprocess
import sys
import time
import urllib.request

PF = "http://127.0.0.1:8100/metrics"
DEC = "http://127.0.0.1:8200/metrics"
INTERVAL = 2
M_RUN = "vllm:num_requests_running"
M_WAIT = "vllm:num_requests_waiting"
# KV cache 实际块占用率(不是显存 MiB!)。不同版本字段名可能不同,挨个试。
M_KV_CANDIDATES = [
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage",
]


def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=3).read().decode()
    except Exception:
        return ""


def metric(text, name):
    vals = []
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                vals.append(float(line.rsplit(" ", 1)[1]))
            except Exception:
                pass
    return sum(vals) / len(vals) if vals else float("nan")


def metric_kv(text):
    for name in M_KV_CANDIDATES:
        v = metric(text, name)
        if v == v:  # 命中(非 nan)
            return v
    return float("nan")


def gpu_utils():
    try:
        out = subprocess.run(["mthreads-gmi"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return [float("nan")] * 8
    u = [int(m.group(1)) for m in re.finditer(r"S5000.*?\|\s*(\d+)%", out)]
    while len(u) < 8:
        u.append(float("nan"))
    return u[:8]


def avg(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def sample_loop(out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "pf_wait", "pf_run", "dec_wait", "dec_run",
                    "dec_kv%", "pf_util(0-3)", "dec_util(4-7)"])
        t0 = time.time()
        while True:
            pf, dec, u = fetch(PF), fetch(DEC), gpu_utils()
            w.writerow([f"{time.time()-t0:.0f}",
                        f"{metric(pf, M_WAIT):.0f}", f"{metric(pf, M_RUN):.0f}",
                        f"{metric(dec, M_WAIT):.0f}", f"{metric(dec, M_RUN):.0f}",
                        f"{metric_kv(dec)*100:.0f}",
                        f"{avg(u[0:4]):.0f}", f"{avg(u[4:8]):.0f}"])
            f.flush()
            time.sleep(INTERVAL)


def summary(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("空")
        return

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except Exception:
                pass
        return out
    print(f"采样 {len(rows)} 点(原始逐点数据见 CSV,人工分析)。各列 [最小~最大]:")
    for c in ["pf_wait", "pf_run", "dec_wait", "dec_run", "dec_kv%",
              "pf_util(0-3)", "dec_util(4-7)"]:
        v = col(c)
        if v:
            print(f"  {c:>14}: {min(v):.0f} ~ {max(v):.0f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--summary":
        summary(sys.argv[2])
    else:
        sample_loop(sys.argv[1] if len(sys.argv) > 1
                    else "/mnt/seed17/001688/models/Qwen/bench/pd_monitor.csv")
