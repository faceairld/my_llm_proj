#!/usr/bin/env python3
"""
benchmark 结果性能分析:从 24 列原始指标算出业界公认的派生性能指标,
并生成浓缩评判表 + 雷达图。

业界指标:
  1. Goodput / SLO 达标最大吞吐:满足 P99 TTFT<阈值 且 P99 TPOT<阈值 的 case 中,
     取最大 req_tp = 该模型的"有效服务容量"(MLPerf/DistServe 范式)。
  2. TGS(Token/GPU/s)= out_tok_tp / tp,跨 TP 配置可比的产能。
  3. 归一化延迟 = mean_e2e / output_len(单 token 端到端体验)。
  4. goodput 比 = req_tp / request_rate(<1 即过载)。
  5. 吞吐-延迟曲线数据(req_tp vs P99 延迟)。

SLO 默认值来自 sglang release 文档(2048~4096 档):TTFT<2000ms,TPOT<50ms。

雷达图 5 个维度(英文标签 → 中文含义,都"越大越好"、按全模型最大值归一化到 0-1):
  EffCapacity  → 有效服务容量:SLO 达标下的最大 req/s(Goodput 范式)
  PeakTGS      → 峰值每卡 token 吞吐:out_tok_tp / tp(Token/GPU/s)
  LowLatency   → 低延迟程度:归一化延迟(mean_e2e/output_len)的倒数
  SLO-PassRate → SLO 达标率:达标 case 数 / 总 case 数
  Goodput      → 抗压能力:goodput 比 req_tp/request_rate(封顶 1.0)

用法:
  python3 analyze.py <summary.csv>                 # 用默认 SLO
  python3 analyze.py <summary.csv> --ttft-slo 2000 --tpot-slo 50
  python3 analyze.py <summary.csv> --radar out.png  # 指定雷达图输出
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# ----- 读取 ------------------------------------------------------------------

def load_rows(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    # 数值列转 float
    numcols = ["tp", "input_len", "output_len", "request_rate", "num_prompts",
               "req_tp", "in_tok_tp", "out_tok_tp", "mean_ttft", "median_ttft",
               "p99_ttft", "mean_tpot", "median_tpot", "p99_tpot", "mean_itl",
               "p99_itl", "mean_e2e", "real_concurrency", "duration",
               "total_input_tokens", "total_output_tokens"]
    out = []
    for r in rows:
        if r.get("status") != "OK":
            continue
        try:
            for c in numcols:
                r[c] = float(r[c])
        except (ValueError, KeyError):
            continue
        out.append(r)
    return out


# ----- 派生指标 --------------------------------------------------------------

def per_model_metrics(rows: list[dict], ttft_slo: float, tpot_slo: float) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_name"]].append(r)

    result = {}
    for model, rs in by_model.items():
        # 每行派生
        for r in rs:
            r["tgs"] = r["out_tok_tp"] / max(r["tp"], 1)
            r["norm_lat"] = r["mean_e2e"] / max(r["output_len"], 1)
            r["goodput_ratio"] = r["req_tp"] / max(r["request_rate"], 1e-9)
            r["slo_ok"] = (r["p99_ttft"] < ttft_slo and r["p99_tpot"] < tpot_slo)

        slo_rows = [r for r in rs if r["slo_ok"]]
        # 有效服务容量 = SLO 达标 case 里的最大 req_tp(及其 out_tok_tp)
        if slo_rows:
            best = max(slo_rows, key=lambda r: r["req_tp"])
            eff_cap = best["req_tp"]
            eff_tgs = best["tgs"]
            best_desc = f"{best['io_label']}@rate{best['request_rate']:.1f}"
        else:
            eff_cap = 0.0
            eff_tgs = 0.0
            best_desc = "无达标 case"

        result[model] = {
            "n_cases": len(rs),
            "n_slo_ok": len(slo_rows),
            "eff_capacity": eff_cap,          # req/s,SLO 达标最大吞吐
            "eff_tgs": eff_tgs,               # 达标点的每卡 token 吞吐
            "peak_tgs": max(r["tgs"] for r in rs),       # 不限 SLO 的峰值 TGS
            "best_norm_lat": min(r["norm_lat"] for r in rs),  # 最优单 token 延迟
            "max_goodput_ratio": max(r["goodput_ratio"] for r in rs),
            "best_desc": best_desc,
            "rows": rs,
        }
    return result


# ----- 打印评判表 ------------------------------------------------------------

def print_table(metrics: dict, ttft_slo: float, tpot_slo: float) -> None:
    print(f"\n性能评判表(SLO: P99 TTFT<{ttft_slo:.0f}ms, P99 TPOT<{tpot_slo:.0f}ms)")
    print("=" * 100)
    hdr = (f"{'模型':<16} {'达标/总':>8} {'有效容量':>10} {'达标TGS':>9} "
           f"{'峰值TGS':>9} {'最优归一延迟':>12} {'最佳达标点':>16}")
    print(hdr)
    print("-" * 100)
    for m, d in sorted(metrics.items()):
        print(f"{m:<16} {d['n_slo_ok']:>3}/{d['n_cases']:<4} "
              f"{d['eff_capacity']:>9.2f}  {d['eff_tgs']:>8.0f} "
              f"{d['peak_tgs']:>8.0f} {d['best_norm_lat']:>11.2f}  "
              f"{d['best_desc']:>16}")
    print("=" * 100)
    print("说明:有效容量=SLO达标下最大 req/s;TGS=token/GPU/s;归一延迟=ms/token(越小越好)")
    print("     达标=0 表示该模型所有 case 都不满足 SLO(过载或太慢)")


# ----- 雷达图 ----------------------------------------------------------------

def make_radar(metrics: dict, out_png: Path, ttft_slo: float, tpot_slo: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 5 个维度,英文标签避免中文字体依赖;中英文对照见文件开头 docstring。
    dims = ["EffCapacity", "PeakTGS", "LowLatency", "SLO-PassRate", "Goodput"]
    models = sorted(metrics.keys())

    # 方向 D:绝对刻度 + 对数映射,不再"除以全模型最大值"。
    # 每个维度给一对固定参考端点 (lo, hi),把绝对值映射到 0-1;吞吐/TGS 跨度大用 log。
    # 好处:① 大小模型都落在真实位置,不被相对归一化掩盖;② 跨数据集可比(端点固定)。
    import math

    def lin(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0

    def logmap(v, lo, hi):
        # 对数刻度:v∈[lo,hi] → 0-1;v<=0 记 0
        if v <= 0:
            return 0.0
        v = max(lo, min(hi, v))
        return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    def axis_value(d):
        eff = d["eff_capacity"]                       # req/s,0~~4
        tgs = d["peak_tgs"]                           # token/GPU/s,~300~2000
        nlat = d["best_norm_lat"]                     # ms/token,越小越好,~20~120
        passrate = d["n_slo_ok"] / max(d["n_cases"], 1)
        # Goodput 维度改用"最佳达标点的 goodput 比"代替封顶值,过载模型才会低
        gp = min(d["max_goodput_ratio"], 1.0)
        return [
            logmap(eff, 0.1, 4.0),       # EffCapacity:对数,0.1~4 req/s
            logmap(tgs, 200, 2000),      # PeakTGS:对数,200~2000 token/GPU/s
            lin(1000.0 / nlat, 1000.0 / 120, 1000.0 / 20) if nlat > 0 else 0,  # LowLatency: 20~120ms/token 线性反向
            passrate,                    # SLO-PassRate:本就是 0-1
            gp,                          # Goodput:0-1
        ]

    norm = {m: axis_value(metrics[m]) for m in models}

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        vals = norm[m] + norm[m][:1]
        ax.plot(angles, vals, label=m, color=cmap(i % 10), linewidth=1.8)
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"vllm_musa tp1 performance radar\n(SLO: TTFT<{ttft_slo:.0f}ms, "
                 f"TPOT<{tpot_slo:.0f}ms; absolute log/linear scale, fixed endpoints)",
                 fontsize=12)
    # 标注各维度的绝对刻度端点,让人知道 0/1 代表什么
    ax.text(0, 1.12, "EffCap 0.1→4 req/s | PeakTGS 200→2000 | "
            "Lat 120→20 ms/tok | Pass 0→1 | Goodput 0→1",
            transform=ax.transAxes, ha="left", fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n雷达图已保存: {Path(out_png).resolve()}")


# ----- 主流程 ----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="benchmark_summary_*.csv 路径")
    ap.add_argument("--ttft-slo", type=float, default=2000,
                    help="P99 TTFT 上限(ms),默认 2000(sglang 文档 2k~4k 档)")
    ap.add_argument("--tpot-slo", type=float, default=50,
                    help="P99 TPOT 上限(ms),默认 50")
    ap.add_argument("--radar", default=None, help="雷达图输出 png(默认同目录 radar.png)")
    ap.add_argument("--export", default=None, help="派生指标导出 CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    if not rows:
        print("没有可用的 OK 行")
        return 1

    metrics = per_model_metrics(rows, args.ttft_slo, args.tpot_slo)
    print_table(metrics, args.ttft_slo, args.tpot_slo)

    radar_png = Path(args.radar) if args.radar else csv_path.parent / "radar.png"
    make_radar(metrics, radar_png, args.ttft_slo, args.tpot_slo)

    if args.export:
        with open(args.export, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n_cases", "n_slo_ok", "eff_capacity_reqs",
                        "eff_tgs", "peak_tgs", "best_norm_lat_ms_per_tok",
                        "max_goodput_ratio", "best_slo_point"])
            for m, d in sorted(metrics.items()):
                w.writerow([m, d["n_cases"], d["n_slo_ok"],
                            f"{d['eff_capacity']:.3f}", f"{d['eff_tgs']:.1f}",
                            f"{d['peak_tgs']:.1f}", f"{d['best_norm_lat']:.3f}",
                            f"{d['max_goodput_ratio']:.3f}", d["best_desc"]])
        print(f"派生指标已导出: {args.export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
