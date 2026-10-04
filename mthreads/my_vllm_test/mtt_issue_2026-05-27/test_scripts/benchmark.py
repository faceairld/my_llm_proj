#!/usr/bin/env python3
"""
vllm 推理性能基准脚本（固定输入，跨模型 / 框架 / 优化对比用）。

============================================================
能测什么
============================================================
  • TTFT (Time To First Token)            首 token 延迟
  • TPOT (Time Per Output Token)          decode 阶段 inter-token 延迟
  • E2E Latency                           端到端请求耗时
  • 聚合吞吐 (Aggregate Throughput)        系统级 token/s（核心指标）
  • KV Cache 占用率                        从 vllm /metrics 拿
  • Prefix Cache 命中率                    从 vllm /metrics 拿
  • 排队等待时间                           从 vllm /metrics 拿
  • GPU 显存占用                           从 mthreads-gmi 拿

============================================================
测试场景（固定输入，用于横向对比）
============================================================
  scenario        说明                                         适合观察
  ─────────────────────────────────────────────────────────────────────────
  short           短 Q&A（50 token 入 / 100 token 出）          基础 TTFT / 调度
  long_context    长上下文（2k token 入 / 200 token 出）         prefill 性能
  long_output     短入长出（50 token 入 / 1024 token 出）        decode 吞吐
  multi_turn      多轮对话（4 轮，每轮约 200 token）             KV cache 增长
  prefix_repeat   同 prompt 重复 N 次                          prefix cache 命中率

每个 scenario 跑两轮：
  1) 串行（concurrency=1）：拿到单请求 baseline
  2) 并发（concurrency=N，可配）：拿到聚合吞吐 + KV/prefix 命中数据

============================================================
用法（在 gy_work 容器里）
============================================================
  # 默认全套（5 scenarios × {串行+并发16}），写报告到 ./bench_<时间>/
  python /data/my_vllm_test/benchmark.py

  # 只跑某些场景
  python /data/my_vllm_test/benchmark.py --scenarios short,long_context

  # 指定并发数
  python /data/my_vllm_test/benchmark.py --concurrency 32

  # 重复次数（每个 scenario 默认跑 32 条请求）
  python /data/my_vllm_test/benchmark.py --requests 64

  # 自定义模型 / 端口
  python /data/my_vllm_test/benchmark.py --port 8001 --model qwen3-32b

  # 只生成 JSON 不打印（用于 CI / 跨次对比）
  python /data/my_vllm_test/benchmark.py --quiet --output bench_a.json

============================================================
横向对比示例（不同模型 / 优化）
============================================================
  # 1. 跑 baseline（无 prefix cache）
  bash run.sh /data/SETS/models/qwen3-8b 8 32768  # 改 run.sh 去掉 --enable-prefix-caching
  python benchmark.py --output bench_no_prefix.json

  # 2. 跑开优化版本
  bash run.sh /data/SETS/models/qwen3-8b 8 32768
  python benchmark.py --output bench_with_prefix.json

  # 3. 对比
  python benchmark.py --compare bench_no_prefix.json bench_with_prefix.json
"""
import argparse
import asyncio
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI

# ============================================================
# 固定测试场景
# ============================================================

# 中文长段落，用于构造长上下文（重复几次就拉长了）
LONG_PARAGRAPH = (
    "Transformer 架构由 Vaswani 等人在 2017 年提出，其核心创新是自注意力机制。"
    "相比传统的 RNN，它能更好地并行化训练，且在长距离依赖建模上表现优异。"
    "encoder-decoder 结构、multi-head attention、位置编码、layer normalization、"
    "残差连接共同构成了它的骨架。后续的 GPT、BERT、T5 等模型都基于这一架构演化而来。"
    "近年来 LLM 的进步主要源于：参数规模的扩张、训练数据的增长、训练策略（RLHF、DPO）的改进、"
    "以及推理优化技术（KV cache、PagedAttention、speculative decoding）的发展。"
)

# 真实非重复中文长文本（~2k 字符），用于公平测试 long_context
# 8 条请求共享这一段为前缀，问题各不相同 → 既能触发 prefix-cache 命中又有可评判的语义输出
LONG_ARTICLE = (
    "人类对时间的感知，是一种奇妙而又令人困惑的体验。"
    "当我们回望童年，那些漫长的暑假仿佛延伸到了无尽的远方，每一天都充满了新鲜的发现和无穷的可能性。"
    "然而随着年龄的增长，时间似乎被一只无形的手不断加速，一年又一年如同翻书一般飞速掠过，让人来不及细细品味便已消逝。"
    "这种主观感受上的巨大差异，长久以来吸引着哲学家、心理学家和神经科学家的关注。"
    "从心理学的角度来看，这种现象可以用「比例理论」来部分解释：对于一个五岁的孩子而言，"
    "一年占据了他全部生命经验的五分之一，因此显得漫长而厚重；而对于一个五十岁的成年人来说，"
    "同样的一年仅仅是生命的五十分之一，自然显得微不足道。但这个解释并不完整，"
    "因为我们对时间的感知还受到许多其他因素的深刻影响，比如情绪状态、注意力的集中程度、以及经历的新鲜度。"
    "当我们沉浸在一项令人愉悦的活动中时，时间仿佛长了翅膀，不知不觉间数小时便已过去。"
    "这就是心理学家所说的「心流」状态——当一个人全神贯注于某项具有适当挑战性的任务时，"
    "自我意识和时间意识都会暂时消退，取而代之的是一种深层的满足感和投入感。与此相反，"
    "当我们焦虑等待、百无聊赖或者身处痛苦之中时，每一分每一秒都变得格外漫长，像是被拉伸的橡皮筋，令人难以忍受。"
    "这种不对称性揭示了一个有趣的悖论：那些在经历时感觉飞逝的快乐时光，在回忆中往往显得丰富而充实；"
    "而那些在经历时度日如年的无聊时刻，在记忆中却往往被压缩成一片模糊的空白。"
    "神经科学的研究为这些现象提供了更深层的生理学基础。大脑中并不存在一个单一的「时间中枢」，"
    "而是由多个脑区协同工作来构建我们的时间感知。基底神经节、小脑、前额叶皮层和海马体都在其中扮演着不同的角色。"
    "多巴胺这种神经递质在时间感知中尤为关键——当多巴胺水平升高时，我们内在的「时钟」会加速运转，"
    "导致我们高估了时间的流逝速度；而当多巴胺水平降低时，内在时钟放慢，时间便显得拖沓。"
    "这解释了为什么在兴奋和期待中时间飞逝，而在沮丧和倦怠中时间却变得黏稠。"
    "记忆与时间感知之间的关系同样耐人寻味。我们对过去时间长度的判断，"
    "在很大程度上取决于我们在那段时间内形成了多少可区分的记忆。"
    "当我们前往一个从未到过的城市旅行时，大脑会贪婪地记录下每一个新奇的细节——"
    "陌生的街道、独特的建筑、意想不到的邂逅。这些丰富的记忆使得短短一周的旅程在回忆中显得无比漫长，"
    "仿佛经历了整整一个月。而当我们日复一日地重复着相同的通勤路线、相同的工作流程、相同的晚餐菜单时，"
    "大脑会自动将这些重复的经历压缩合并，结果就是整整一个月在记忆中被浓缩成了区区几个片段。"
)

# 8 个针对 LONG_ARTICLE 的不同问题，问题不同但前缀相同 → 触发 prefix-cache 命中
LONG_ARTICLE_QUESTIONS = [
    "请用三句话总结上文的主要观点。",
    "什么是「比例理论」？它能完全解释时间感知的差异吗？",
    "什么是「心流」状态？它对时间感知有什么影响？",
    "为什么快乐的时光在回忆中显得充实而无聊的时刻反而显得空白？",
    "大脑中哪些区域参与时间感知？它们各自扮演什么角色？",
    "多巴胺如何影响我们对时间的感知？请举两个例子。",
    "为什么年龄增长后会觉得时间过得越来越快？",
    "如果想在主观上延长生命体验，根据上文应该怎么做？",
]


def build_scenarios() -> dict[str, list[list[dict]]]:
    """返回 scenario_name → list of message_lists（每条 = 一次请求的 messages）"""
    sys_prompt = {"role": "system", "content": "You are a helpful assistant. 用简洁的中文回答。"}

    # ---- short: 50 入 / 100 出 ----
    short_prompts = [
        "什么是 PagedAttention？",
        "Tensor Parallel 和 Pipeline Parallel 的区别？",
        "为什么 Transformer 用 LayerNorm 而不是 BatchNorm？",
        "解释一下 prefix caching 的原理。",
        "FP16、BF16、FP8 在训练和推理中各有什么取舍？",
        "vLLM 是如何提高吞吐的？",
        "RoPE 位置编码相比绝对位置编码有什么优势？",
        "RLHF 中 reward model 是怎么训练的？",
    ]
    short = [[sys_prompt, {"role": "user", "content": p}] for p in short_prompts]

    # ---- long_context: ~2k 入 / 200 出 ----
    # 用真实非重复文本 + 8 个不同的真实问题；前缀共享触发 prefix-cache 命中
    long_context = [
        [sys_prompt, {"role": "user", "content": f"请阅读以下文章并回答问题：\n\n{LONG_ARTICLE}\n\n问题：{q}"}]
        for q in LONG_ARTICLE_QUESTIONS
    ]

    # ---- long_output: 短入长出 ----
    long_output_prompts = [
        "请详细介绍 vLLM 的架构设计与优化技术，至少 1000 字。",
        "解释 MoE（Mixture of Experts）的工作原理及其训练挑战，详尽展开。",
        "请系统讲解大模型推理的各类量化技术（W8A8、W4A16、FP8、KV-cache 量化）。",
        "讲一个 1000 字的科幻短篇小说。",
    ]
    long_output = [[sys_prompt, {"role": "user", "content": p}] for p in long_output_prompts]

    # ---- multi_turn: 4 轮 ----
    multi_turn_seed = [
        ("我是个 Python 初学者，想学机器学习，从哪开始？", None),
        ("好，那我应该先学 NumPy 还是直接 PyTorch？", None),
        ("PyTorch 的 autograd 和 TensorFlow 的 GradientTape 有什么不同？", None),
        ("能给我一个最简单的 PyTorch 训练循环示例吗？", None),
    ]
    # 每条 multi_turn "请求"实际是模拟一个完整 4 轮对话——
    # 但为了 benchmark 简单，我们把它展开成 4 条独立请求，第 N 条带前 N-1 轮的历史
    multi_turn = []
    for session_id in range(4):  # 4 个独立 session
        msgs = [sys_prompt]
        for q, _ in multi_turn_seed:
            msgs = msgs + [{"role": "user", "content": q}]
            multi_turn.append(list(msgs))  # 当前到这里的累积历史就是一条请求
            # 模拟 assistant 回答（用占位，因为真回复要等服务返回，这里只是构造测试输入）
            msgs.append({"role": "assistant", "content": "（模拟历史回复占位）"})
        # session_id 用于让不同 session 的 prompt 略有差异（避免完全前缀相同）
        if session_id > 0:
            multi_turn[-1][1]["content"] = f"（用户{session_id}）" + multi_turn[-1][1]["content"]

    # ---- prefix_repeat: 同 prompt 重复，测 prefix cache ----
    prefix_repeat = [
        [sys_prompt, {"role": "user", "content": "用三句话介绍量子计算的基本原理。"}]
        for _ in range(16)
    ]

    return {
        "short": short,
        "long_context": long_context,
        "long_output": long_output,
        "multi_turn": multi_turn,
        "prefix_repeat": prefix_repeat,
    }


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RequestStat:
    idx: int
    ttft_s: float = float("nan")
    e2e_s: float = float("nan")
    n_tokens_out: int = 0
    n_tokens_in: int = 0
    ok: bool = False
    err: str = ""
    # 验证语义用:用户 prompt 末尾(完整 prompt 太长,只截尾)+ 模型输出
    user_prompt_tail: str = ""
    output_text: str = ""


@dataclass
class RunStat:
    scenario: str
    concurrency: int
    n_requests: int
    wall_s: float
    ok_count: int
    fail_count: int
    ttft_avg_ms: float = float("nan")
    ttft_p50_ms: float = float("nan")
    ttft_p95_ms: float = float("nan")
    ttft_p99_ms: float = float("nan")
    tpot_avg_ms: float = float("nan")
    e2e_avg_s: float = float("nan")
    e2e_p50_s: float = float("nan")
    e2e_p95_s: float = float("nan")
    output_tps_avg: float = float("nan")          # 单请求 token/s 平均
    aggregate_tps: float = float("nan")           # 聚合吞吐
    total_tokens_out: int = 0
    total_tokens_in: int = 0
    # vllm /metrics 增量
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    metrics_delta: dict = field(default_factory=dict)
    # 每条请求的明细(包括生成文本,用于人工验证语义)—— 写入 sidecar .outputs.txt,不进 bench.json
    requests: list = field(default_factory=list, repr=False, compare=False)
    # GPU 显存（每张卡的 used MiB）
    gpu_mem_before: list = field(default_factory=list)
    gpu_mem_after: list = field(default_factory=list)
    # 错误样本
    sample_errors: list = field(default_factory=list)


# ============================================================
# 工具：抓 vllm /metrics 和 mthreads-gmi
# ============================================================

# 关心的 vllm metric 前缀
INTERESTED_METRICS = (
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_queries",
    "vllm:gpu_prefix_cache_hits",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_count",
)


async def fetch_vllm_metrics(base_url: str) -> dict:
    """从 vllm /metrics 抓 Prometheus 风格数据，解析成 {metric_name: value}"""
    url = base_url.replace("/v1", "") + "/metrics"
    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.get(url)
            r.raise_for_status()
    except Exception as e:
        return {"_error": str(e)}

    out = {}
    # 简单 Prometheus 解析：跳过 # 注释，行格式 `metric_name{labels} value`
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)$", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if not any(name.startswith(p) for p in INTERESTED_METRICS):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        # 同名 metric 可能多次出现（不同 label），简单累加
        out[name] = out.get(name, 0.0) + v
    return out


def metrics_delta(before: dict, after: dict) -> dict:
    """计算 after - before 的增量，附带派生指标"""
    delta = {}
    for k in set(before) | set(after):
        if k.startswith("_"):
            continue
        b = before.get(k, 0.0)
        a = after.get(k, 0.0)
        delta[k] = a - b

    # 派生：prefix cache 命中率
    q = delta.get("vllm:gpu_prefix_cache_queries", 0.0)
    h = delta.get("vllm:gpu_prefix_cache_hits", 0.0)
    if q > 0:
        delta["_prefix_cache_hit_rate"] = h / q

    # 派生：平均 TTFT（vllm 内部统计）
    ttft_sum = delta.get("vllm:time_to_first_token_seconds_sum", 0.0)
    ttft_cnt = delta.get("vllm:time_to_first_token_seconds_count", 0.0)
    if ttft_cnt > 0:
        delta["_vllm_ttft_avg_ms"] = ttft_sum / ttft_cnt * 1000

    # 派生：平均 TPOT
    tpot_sum = delta.get("vllm:time_per_output_token_seconds_sum", 0.0)
    tpot_cnt = delta.get("vllm:time_per_output_token_seconds_count", 0.0)
    if tpot_cnt > 0:
        delta["_vllm_tpot_avg_ms"] = tpot_sum / tpot_cnt * 1000

    # 派生：平均排队等待
    q_sum = delta.get("vllm:request_queue_time_seconds_sum", 0.0)
    q_cnt = delta.get("vllm:request_queue_time_seconds_count", 0.0)
    if q_cnt > 0:
        delta["_avg_queue_time_ms"] = q_sum / q_cnt * 1000

    # 当前 KV cache 占用（取 after 的最新值）
    if "vllm:gpu_cache_usage_perc" in after:
        delta["_kv_cache_usage_after"] = after["vllm:gpu_cache_usage_perc"]

    return delta


def query_gpu_mem() -> list[int]:
    """返回每张卡当前 used MiB 列表；mthreads-gmi 不可用时返回空"""
    try:
        out = subprocess.run(
            ["mthreads-gmi"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return []
    used = []
    # 行示例: "0%    70160MiB(81920MiB)"
    for m in re.finditer(r"(\d+)MiB\((\d+)MiB\)", out):
        used.append(int(m.group(1)))
    return used


def inspect_vllm_musa_flash_attn(path: str) -> dict:
    """Inspect flash_attn.py so benchmark output records which implementation ran."""
    info = {
        "path": path,
        "exists": False,
        "sha256": "",
        "has_codex_marker": False,
        "has_path_a_marker": False,
        "actual_impl": "unknown",
        "error": "",
    }
    try:
        p = Path(path)
        info["exists"] = p.exists()
        if not p.exists():
            info["error"] = "file not found"
            return info
        data = p.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        info["sha256"] = hashlib.sha256(data).hexdigest()
        info["has_codex_marker"] = ("CODEX MOD" in text or "CODEX PATH" in text)
        info["has_path_a_marker"] = ("PATH-A" in text or "Path A" in text)
        info["actual_impl"] = (
            "patched" if info["has_codex_marker"] or info["has_path_a_marker"]
            else "original-like"
        )
    except Exception as e:
        info["error"] = str(e)
    return info


# ============================================================
# 单条请求
# ============================================================

async def run_one(
    client: AsyncOpenAI, model: str, messages: list[dict],
    temperature: float, max_tokens: int, idx: int,
    request_timeout: float = 90.0,
) -> RequestStat:
    """单条请求；超过 request_timeout 秒未完整返回则标记失败（避免 vllm 死锁拖死整个 benchmark）"""
    stat = RequestStat(idx=idx)
    t0 = time.perf_counter()
    first_at = None
    n_chunks = 0
    full = []

    async def _do_request():
        nonlocal first_at, n_chunks
        stream = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_at is None:
                    first_at = time.perf_counter()
                full.append(chunk.choices[0].delta.content)
                n_chunks += 1
            if chunk.usage:
                stat.n_tokens_in = chunk.usage.prompt_tokens
                stat.n_tokens_out = chunk.usage.completion_tokens

    try:
        await asyncio.wait_for(_do_request(), timeout=request_timeout)
        stat.e2e_s = time.perf_counter() - t0
        stat.ttft_s = (first_at - t0) if first_at else float("nan")
        if stat.n_tokens_out == 0:
            stat.n_tokens_out = max(1, n_chunks)
        stat.ok = True
        # 捕获用户 prompt 尾巴 + 完整输出文本(便于人工验证语义)
        try:
            user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            stat.user_prompt_tail = user_msg[-150:] if len(user_msg) > 150 else user_msg
            stat.output_text = "".join(full)
        except Exception:
            pass
    except asyncio.TimeoutError:
        stat.e2e_s = time.perf_counter() - t0
        stat.err = f"REQUEST_TIMEOUT after {request_timeout}s (vllm hang?)"
    except Exception as e:
        stat.e2e_s = time.perf_counter() - t0
        stat.err = str(e)[:200]
    return stat


# ============================================================
# 跑一组（一个场景 × 一个并发数）
# ============================================================

async def run_group(
    client: AsyncOpenAI, base_url: str, model: str,
    scenario: str, message_lists: list[list[dict]],
    concurrency: int, temperature: float, max_tokens: int,
    requests: int, request_timeout: float = 90.0,
) -> RunStat:
    # 平铺成 N 条（用 message_lists 循环填）
    flat = [message_lists[i % len(message_lists)] for i in range(requests)]

    # 抓 before metrics & gpu mem
    metrics_before = await fetch_vllm_metrics(base_url)
    gpu_before = query_gpu_mem()

    sem = asyncio.Semaphore(concurrency)

    async def gated(idx, msgs):
        async with sem:
            return await run_one(client, model, msgs, temperature, max_tokens, idx, request_timeout)

    t0 = time.perf_counter()
    stats = await asyncio.gather(*[gated(i, m) for i, m in enumerate(flat)])
    wall = time.perf_counter() - t0

    metrics_after = await fetch_vllm_metrics(base_url)
    gpu_after = query_gpu_mem()
    delta = metrics_delta(metrics_before, metrics_after)

    # 聚合统计
    ok = [s for s in stats if s.ok]
    fail = [s for s in stats if not s.ok]
    ttfts = [s.ttft_s * 1000 for s in ok if s.ttft_s == s.ttft_s]  # 排除 nan
    e2es = [s.e2e_s for s in ok]
    out_tokens = sum(s.n_tokens_out for s in ok)
    in_tokens = sum(s.n_tokens_in for s in ok)
    per_req_tps = [s.n_tokens_out / s.e2e_s for s in ok if s.e2e_s > 0]

    rs = RunStat(
        scenario=scenario, concurrency=concurrency, n_requests=requests,
        wall_s=wall, ok_count=len(ok), fail_count=len(fail),
        total_tokens_out=out_tokens, total_tokens_in=in_tokens,
        metrics_before=metrics_before, metrics_after=metrics_after, metrics_delta=delta,
        gpu_mem_before=gpu_before, gpu_mem_after=gpu_after,
        sample_errors=[s.err for s in fail[:3]],
        requests=stats,   # per-request 明细,用于写 .outputs.txt
    )

    if ttfts:
        rs.ttft_avg_ms = statistics.mean(ttfts)
        rs.ttft_p50_ms = statistics.median(ttfts)
        rs.ttft_p95_ms = percentile(ttfts, 0.95)
        rs.ttft_p99_ms = percentile(ttfts, 0.99)
    if e2es:
        rs.e2e_avg_s = statistics.mean(e2es)
        rs.e2e_p50_s = statistics.median(e2es)
        rs.e2e_p95_s = percentile(e2es, 0.95)
    if per_req_tps:
        rs.output_tps_avg = statistics.mean(per_req_tps)
    if wall > 0:
        rs.aggregate_tps = out_tokens / wall

    # TPOT 估算 = (e2e - ttft) / (n_out - 1)
    tpots = []
    for s in ok:
        if s.n_tokens_out > 1 and s.e2e_s > 0 and s.ttft_s == s.ttft_s:
            tpots.append((s.e2e_s - s.ttft_s) / (s.n_tokens_out - 1) * 1000)
    if tpots:
        rs.tpot_avg_ms = statistics.mean(tpots)

    return rs


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    k = (len(data) - 1) * p
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


# ============================================================
# 报告输出
# ============================================================

def print_run(rs: RunStat) -> None:
    """打印一组结果"""
    print(f"\n\033[33m━━━ {rs.scenario}  (concurrency={rs.concurrency}, requests={rs.n_requests}) ━━━\033[0m")
    print(f"  ✓ {rs.ok_count}/{rs.n_requests}   wall={rs.wall_s:.2f}s   失败={rs.fail_count}")
    if rs.fail_count:
        for e in rs.sample_errors[:2]:
            print(f"    err: {e[:120]}")
    if rs.ok_count == 0:
        return
    print(f"  TTFT      avg={rs.ttft_avg_ms:.0f}ms  p50={rs.ttft_p50_ms:.0f}  p95={rs.ttft_p95_ms:.0f}  p99={rs.ttft_p99_ms:.0f}")
    print(f"  TPOT      avg={rs.tpot_avg_ms:.1f}ms (decode 阶段单 token)")
    print(f"  E2E       avg={rs.e2e_avg_s:.2f}s   p50={rs.e2e_p50_s:.2f}   p95={rs.e2e_p95_s:.2f}")
    print(f"  Tokens    in={rs.total_tokens_in}   out={rs.total_tokens_out}")
    print(f"  Throughput  单请求avg={rs.output_tps_avg:.1f} tok/s   \033[32m聚合={rs.aggregate_tps:.1f} tok/s\033[0m")

    d = rs.metrics_delta
    if d:
        if "_prefix_cache_hit_rate" in d:
            q = d.get("vllm:gpu_prefix_cache_queries", 0)
            h = d.get("vllm:gpu_prefix_cache_hits", 0)
            print(f"  PrefixCache  queries={q:.0f}  hits={h:.0f}  hit_rate={d['_prefix_cache_hit_rate']*100:.1f}%")
        if "_kv_cache_usage_after" in d:
            print(f"  KV Cache    占用率(末态)={d['_kv_cache_usage_after']*100:.1f}%")
        if "_avg_queue_time_ms" in d:
            print(f"  排队等待    avg={d['_avg_queue_time_ms']:.1f}ms")

    if rs.gpu_mem_after:
        diff = [a - b for a, b in zip(rs.gpu_mem_after, rs.gpu_mem_before)]
        print(f"  GPU Mem 增量 (MiB/卡): {diff}")


def print_summary_table(runs: list[RunStat]) -> None:
    print(f"\n\n\033[1m======== Summary ========\033[0m")
    hdr = f"{'scenario':<15} {'cc':>4} {'ok/N':>7} {'wall(s)':>8} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'agg tok/s':>10} {'prefix%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        prefix = r.metrics_delta.get("_prefix_cache_hit_rate")
        prefix_s = f"{prefix*100:.1f}" if prefix is not None else "-"
        print(f"{r.scenario:<15} {r.concurrency:>4} {r.ok_count:>3}/{r.n_requests:<3} "
              f"{r.wall_s:>8.2f} {r.ttft_avg_ms:>9.0f} {r.tpot_avg_ms:>9.1f} "
              f"{r.aggregate_tps:>10.1f} {prefix_s:>8}")


# ============================================================
# 跨次对比
# ============================================================

def cmd_compare(file_a: str, file_b: str) -> None:
    a = json.loads(Path(file_a).read_text())
    b = json.loads(Path(file_b).read_text())
    print(f"\n\033[1m======== Compare ========\033[0m")
    print(f"  A = {file_a}")
    print(f"  B = {file_b}\n")
    by_key_a = {(r["scenario"], r["concurrency"]): r for r in a["runs"]}
    by_key_b = {(r["scenario"], r["concurrency"]): r for r in b["runs"]}
    keys = sorted(set(by_key_a) | set(by_key_b))

    print(f"{'scenario':<15} {'cc':>4}   {'agg A':>10}   {'agg B':>10}   {'B/A':>7}    {'TTFT A':>8}   {'TTFT B':>8}   {'B/A':>7}")
    for k in keys:
        ra, rb = by_key_a.get(k), by_key_b.get(k)
        if not ra or not rb:
            continue
        agg_a, agg_b = ra["aggregate_tps"], rb["aggregate_tps"]
        ttft_a, ttft_b = ra["ttft_avg_ms"], rb["ttft_avg_ms"]
        agg_ratio = agg_b / agg_a if agg_a else float("nan")
        ttft_ratio = ttft_b / ttft_a if ttft_a else float("nan")
        print(f"{k[0]:<15} {k[1]:>4}   {agg_a:>10.1f}   {agg_b:>10.1f}   {agg_ratio:>6.2f}x   "
              f"{ttft_a:>8.0f}   {ttft_b:>8.0f}   {ttft_ratio:>6.2f}x")


# ============================================================
# 主流程
# ============================================================

async def main_run(args) -> None:
    base_url = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    vllm_musa_info = inspect_vllm_musa_flash_attn(args.vllm_musa_flash_attn)
    if args.vllm_musa_impl == "original" and vllm_musa_info.get("actual_impl") != "original-like":
        print(
            "[ERROR] --vllm-musa-impl=original 但 flash_attn.py 仍包含 CODEX/PATH-A 标记，拒绝误测。",
            file=sys.stderr,
        )
        print(json.dumps(vllm_musa_info, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(2)

    # 探活 + 选模型
    try:
        models = (await client.models.list()).data
    except Exception as e:
        print(f"[ERROR] 连不上 {base_url}: {e}", file=sys.stderr)
        sys.exit(1)
    available = [m.id for m in models]
    if not available:
        print(f"[ERROR] 无可用模型", file=sys.stderr); sys.exit(1)
    model = args.model or available[0]
    if model not in available:
        print(f"[ERROR] 模型 '{model}' 不存在。可用: {available}", file=sys.stderr); sys.exit(1)

    print(f"[INFO] endpoint: {base_url}    model: {model}")
    print(f"[INFO] requests/scenario={args.requests}    concurrencies={[1, args.concurrency]}")
    print(f"[INFO] vllm_musa_impl={args.vllm_musa_impl} actual={vllm_musa_info.get('actual_impl')} sha256={vllm_musa_info.get('sha256', '')[:12]}")

    scenarios = build_scenarios()
    selected = args.scenarios.split(",") if args.scenarios else list(scenarios.keys())
    selected = [s.strip() for s in selected if s.strip() in scenarios]
    if not selected:
        print(f"[ERROR] 没有可跑的 scenario。可用: {list(scenarios.keys())}", file=sys.stderr); sys.exit(1)

    # 各 scenario 用对应 max_tokens
    max_tokens_map = {
        "short": 100,
        "long_context": 200,
        "long_output": 1024,
        "multi_turn": 200,
        "prefix_repeat": 200,
    }

    all_runs = []
    for sc in selected:
        msgs = scenarios[sc]
        mt = max_tokens_map.get(sc, args.max_tokens)
        # 串行 baseline
        rs1 = await run_group(
            client, base_url, model, sc, msgs,
            concurrency=1, temperature=args.temperature, max_tokens=mt,
            requests=min(args.requests, 16),  # 串行少跑点省时间
            request_timeout=args.request_timeout,
        )
        if not args.quiet:
            print_run(rs1)
        all_runs.append(rs1)

        # 并发
        rsN = await run_group(
            client, base_url, model, sc, msgs,
            concurrency=args.concurrency, temperature=args.temperature, max_tokens=mt,
            requests=args.requests,
            request_timeout=args.request_timeout,
        )
        if not args.quiet:
            print_run(rsN)
        all_runs.append(rsN)

    if not args.quiet:
        print_summary_table(all_runs)

    if args.output:
        # 1) 写 bench.json:剥掉 per-request 明细(避免输出文本撑大文件)
        def _strip_requests(run_dict):
            d = dict(run_dict)
            d.pop("requests", None)
            return d

        report = {
            "endpoint": base_url, "model": model,
            "concurrency": args.concurrency, "requests": args.requests,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vllm_musa_impl": args.vllm_musa_impl,
            "vllm_musa_flash_attn": vllm_musa_info,
            "runs": [_strip_requests(asdict(r)) for r in all_runs],
        }
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))

        # 2) 写 sidecar .outputs.txt:每条请求的输入尾巴 + 完整输出文本,便于人工验证语义
        output_path = Path(args.output)
        sidecar = output_path.with_suffix(output_path.suffix + ".outputs.txt")
        lines = []
        lines.append(f"# Model outputs from benchmark run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Endpoint: {base_url}  Model: {model}")
        lines.append(f"# Concurrency: {args.concurrency}  Requests/run: {args.requests}")
        lines.append(f"# vllm_musa_impl: requested={args.vllm_musa_impl} actual={vllm_musa_info.get('actual_impl')}")
        lines.append(f"# flash_attn.py: {vllm_musa_info.get('path')} sha256={vllm_musa_info.get('sha256', '')}")
        lines.append(f"# markers: codex={vllm_musa_info.get('has_codex_marker')} path_a={vllm_musa_info.get('has_path_a_marker')}")
        lines.append("")
        for run in all_runs:
            lines.append("=" * 80)
            lines.append(f"Scenario: {run.scenario}    Concurrency: {run.concurrency}")
            lines.append(f"OK: {run.ok_count}/{run.n_requests}    wall={run.wall_s:.2f}s")
            lines.append("=" * 80)
            for stat in run.requests:
                lines.append(f"\n--- Request #{stat.idx} ---")
                if stat.ok:
                    lines.append(f"[STATUS] ok=True  ttft={stat.ttft_s*1000:.1f}ms  e2e={stat.e2e_s:.2f}s  tokens_out={stat.n_tokens_out}")
                else:
                    lines.append(f"[STATUS] ok=False  err={stat.err}")
                if stat.user_prompt_tail:
                    lines.append(f"[USER prompt 尾巴] ...{stat.user_prompt_tail}")
                if stat.output_text:
                    lines.append("[MODEL output]")
                    lines.append(stat.output_text)
                lines.append("")
        sidecar.write_text("\n".join(lines))
        print(f"[INFO] 输出文本已写入 {sidecar}")
        print(f"\n[INFO] 报告已写入 {args.output}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scenarios", default=None,
                    help="逗号分隔，缺省跑全部。可选: short,long_context,long_output,multi_turn,prefix_repeat")
    ap.add_argument("--concurrency", type=int, default=16, help="并发请求数 (默认 16)")
    ap.add_argument("--requests", type=int, default=32, help="每场景请求总数 (默认 32)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--request-timeout", type=float, default=90.0,
                    help="单条请求超时秒数（vllm 死锁时避免无限等待，默认 90 秒）")
    ap.add_argument("--vllm-musa-impl", choices=("current", "original"), default="current",
                    help="记录/校验本次预期的 vllm_musa 实现。original 会要求 flash_attn.py 无 CODEX/PATH-A 标记")
    ap.add_argument("--vllm-musa-flash-attn",
                    default="/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py",
                    help="用于识别当前 vllm_musa 实现的 flash_attn.py 路径")
    ap.add_argument("--quiet", action="store_true", help="只输出 JSON，不打印中间结果")
    ap.add_argument("--output", default=None, help="保存 JSON 报告路径，便于跨次对比")
    ap.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                    help="对比两份 JSON 报告（不会重新跑测试）")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(*args.compare)
        return

    if args.output is None and not args.quiet:
        # 默认放到 ./bench_<时间戳>.json
        args.output = f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"

    asyncio.run(main_run(args))


if __name__ == "__main__":
    main()
