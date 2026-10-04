#!/usr/bin/env python3
"""
两轮找拐点 —— 第 2 步:读粗扫 summary,自动定位拐点,生成「精扫」参数文件。

两个相互独立的拐点(因果上 容量拐点 ≤ 性能拐点):
  ① 容量拐点(throughput):goodput 比 = req_tp/request_rate 跌破阈值(默认 0.85)。
       —— 推理框架 + 硬件资源的吞吐上限。健康区 ≈1,饱和后掉下去。不依赖 SLO。
  ② 性能拐点(latency):mean_tpot 相对健康区基线(最低几档最小值)跳变 > 阈值(默认 1.8×)。
       —— 用户感知延迟开始恶化的点(SLO 边界)。
  小模型满载也可能不暴涨 TPOT → 容量拐点存在、性能拐点测不到(在最高 rate 之上)。

用 --knee 选要精扫哪个:
  throughput  只精扫容量拐点区间
  latency     只精扫性能拐点区间
  both        两个拐点各精扫一簇(默认)

用法:
  python3 find_knee.py <coarse_summary.csv>                      # 只打印两拐点报告
  python3 find_knee.py <coarse_summary.csv> --knee both --emit fine.json
  python3 find_knee.py <coarse_summary.csv> --knee throughput --emit fine_cap.json
可调:--gr-thresh 0.85 --tpot-jump 1.8 --fine-points 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def csv_model_name(model_path: str) -> str:
    """CSV 的 model_name 列 = served-model-name = model_path 的 basename 小写。
    注意它可能与参数文件里的 served_name 字段不同(如 Qwen3-8B → qwen3-8b,
    而 served_name 写的是 qwen3-8b-bf16),匹配粗扫结果时必须按 basename。"""
    return os.path.basename(model_path.rstrip("/")).lower()


def load_rows(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    num = ["input_len", "output_len", "request_rate", "req_tp",
           "mean_tpot", "p99_tpot", "mean_ttft", "p99_ttft", "real_concurrency"]
    out = []
    for r in rows:
        if r.get("status") != "OK":
            continue
        try:
            for c in num:
                r[c] = float(r[c])
        except (ValueError, KeyError):
            continue
        out.append(r)
    return out


def _interval(rates: list[float], onset: int | None) -> dict:
    """把 onset 索引转成拐点区间。onset=None 表示全程未触发。"""
    if onset is None:
        return {"status": "above_max", "lo": rates[-1], "hi": None}
    if onset == 0:
        return {"status": "below_min", "lo": None, "hi": rates[0]}
    return {"status": "ok", "lo": rates[onset - 1], "hi": rates[onset]}


def detect_knees(points: list[dict], gr_thresh: float, tpot_jump: float) -> dict:
    """同一 (model, io_label) 下按 rate 升序的行 → 分别返回容量/性能两个拐点。"""
    pts = sorted(points, key=lambda r: r["request_rate"])
    rates = [p["request_rate"] for p in pts]
    base_tpot = min(p["mean_tpot"] for p in pts[:max(1, min(3, len(pts)))])

    gp_onset = None      # 容量拐点:goodput 首次跌破
    tp_onset = None      # 性能拐点:tpot 首次跳变
    for i, p in enumerate(pts):
        gr = p["req_tp"] / p["request_rate"] if p["request_rate"] > 0 else 0.0
        if gp_onset is None and gr < gr_thresh:
            gp_onset = i
        if tp_onset is None and base_tpot > 0 and p["mean_tpot"] > tpot_jump * base_tpot:
            tp_onset = i
    return {
        "base_tpot": base_tpot,
        "rates": rates,
        "throughput": _interval(rates, gp_onset),   # 容量拐点
        "latency": _interval(rates, tp_onset),       # 性能拐点
        "table": [(p["request_rate"], p["req_tp"],
                   (p["req_tp"] / p["request_rate"] if p["request_rate"] > 0 else 0),
                   p["mean_tpot"], p["p99_tpot"], p["p99_ttft"]) for p in pts],
    }


def fine_rates_for(interval: dict, n: int) -> list[float]:
    """单个拐点区间 → 精扫 rate 列表(n 点,2 位小数去重)。"""
    st = interval["status"]
    if st == "ok":
        lo, hi = interval["lo"], interval["hi"]
    elif st == "below_min":
        hi = interval["hi"]
        lo = max(0.05, hi * 0.25)          # 向下延伸 [0.25hi, hi]
    else:  # above_max
        lo = interval["lo"]
        hi = lo * 2.0                       # 向上延伸 [max, 2×max]
    if hi <= lo:
        return [round(lo, 2)]
    step = (hi - lo) / (n - 1)
    return sorted({round(lo + step * k, 2) for k in range(n) if lo + step * k > 0})


def fine_rates(knee: dict, mode: str, n: int) -> list[float]:
    """按 --knee 模式合并需要精扫的 rate。"""
    vals: set[float] = set()
    if mode in ("throughput", "both"):
        vals.update(fine_rates_for(knee["throughput"], n))
    if mode in ("latency", "both"):
        vals.update(fine_rates_for(knee["latency"], n))
    return sorted(vals)


def _desc(interval: dict, label: str) -> str:
    st = interval["status"]
    if st == "ok":
        return f"{label}拐点区间 [{interval['lo']}, {interval['hi']}]"
    if st == "below_min":
        return f"⚠ {label}拐点 < 最低 rate {interval['hi']}(首点即过,需向下延伸)"
    return f"⚠ {label}拐点 > 最高 rate {interval['lo']}(全程未触发,需向上延伸)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="粗扫 benchmark_summary_*.csv")
    ap.add_argument("--knee", choices=["throughput", "latency", "both"],
                    default="both", help="精扫哪个拐点(默认 both)")
    ap.add_argument("--base", default=None,
                    help="精扫参数文件的模型元数据来源(默认同目录 tp1_bench_params_coarse.json)")
    ap.add_argument("--emit", default=None, help="输出精扫参数文件路径")
    ap.add_argument("--gr-thresh", type=float, default=0.85, help="goodput 比拐点阈值(容量)")
    ap.add_argument("--tpot-jump", type=float, default=1.8, help="mean_tpot 跳变倍数阈值(性能)")
    ap.add_argument("--fine-points", type=int, default=5, help="每个拐点区间精扫点数")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        print("没有可用 OK 行")
        return 1

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model_name"], r["io_label"])].append(r)
    knees = {k: detect_knees(v, args.gr_thresh, args.tpot_jump)
             for k, v in groups.items()}

    # ---- 报告(两拐点都打印,无论 --knee)----
    print(f"\n拐点报告 [--knee={args.knee}]  "
          f"容量: goodput<{args.gr_thresh} | 性能: tpot 跳变>{args.tpot_jump}×")
    print("=" * 96)
    cur = None
    for (model, label) in sorted(knees):
        if model != cur:
            print(f"\n■ {model}")
            cur = model
        info = knees[(model, label)]
        print(f"  [{label:>9}] base_tpot={info['base_tpot']:.1f}ms")
        print(f"             容量 | {_desc(info['throughput'], '容量')}")
        print(f"             性能 | {_desc(info['latency'], '性能')}")
        print(f"             精扫 rate → {fine_rates(info, args.knee, args.fine_points)}")

    # ---- 生成精扫参数文件 ----
    if args.emit:
        base_path = Path(args.base) if args.base else \
            Path(__file__).parent / "tp1_bench_params_coarse.json"
        base = json.loads(Path(base_path).read_text(encoding="utf-8"))
        label_shape = {r["io_label"]: (int(r["input_len"]), int(r["output_len"]))
                       for r in rows}
        out = {}
        for name, v in base.items():
            model = csv_model_name(v["model_path"])   # 按 basename 匹配 CSV model_name
            cases = []
            for (m, label), info in knees.items():
                if m != model:
                    continue
                inp, outl = label_shape.get(label, (None, None))
                if inp is None:
                    continue
                for r in fine_rates(info, args.knee, args.fine_points):
                    cases.append([inp, outl, label, r])
            cases.sort(key=lambda c: (c[2], c[3]))
            if not cases:
                continue
            out[name] = {
                "model_dir": v.get("model_dir"),
                "served_name": v["served_name"],
                "num_cases": len(cases),
                "cases": cases,
                "model_path": v["model_path"],
                "extra_args": v["extra_args"],
            }
        Path(args.emit).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tot = sum(m["num_cases"] for m in out.values())
        print(f"\n已生成精扫参数文件: {args.emit}  ({len(out)} 模型, {tot} cases, knee={args.knee})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
