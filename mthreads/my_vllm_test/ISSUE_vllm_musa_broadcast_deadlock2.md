# ISSUE vllm_musa broadcast deadlock —— 续篇（2）：新版 vllm_musa 上配置 LMCache 阶段

> 本文承接 [`ISSUE_vllm_musa_broadcast_deadlock.md`](./ISSUE_vllm_musa_broadcast_deadlock.md)（下称「续1」/「原 ISSUE」）。
> 原 ISSUE 记录的是 **V0 引擎下 APC（prefix-cache）的崩溃 bug（Bug 1 OOB）与加速效果**，当时 V1+LMCache 在 S5000 上跑不起来，LMCache 是纯待办。
> 本文记录的是 **V1+LMCache 真正跑通之后的新阶段**：在新版 vllm_musa（带 V1 引擎）上配 LMCache，并设计「纯版 vs 挂 LMCache」性能对比 benchmark。
> 详细移植排查另见 [`vllm_020/lmcache_on_vllm_musa_020_feasibility.md`](./vllm_020/lmcache_on_vllm_musa_020_feasibility.md)。
>
> **创建日期**：2026-06-16  
> **最近更新**：2026-06-17

---

## 0. 当前环境（2026-06-16 核对）

| 项 | 值 |
|---|---|
| 服务器 | **146**（`10.10.142.146`，root / `Admin@9000`） |
| 容器 | `vllm020_lmcache_test`（Up，新版 vllm_musa，V1 引擎） |
| 模型 | Qwen3-8B（bf16），服务端路径 `/data/SQT-v1.0.5-test/models/qwen3-8b` |
| 端口 19000 | `qwen3-8b-pure-fixed` —— **纯版 vllm_musa，无 LMCache** |
| 端口 19001 | `qwen3-8b-lmcache` —— **挂 LMCache**（`LMCacheConnectorV1` + `LocalCPUBackend`，可参数开关） |
| 发起 benchmark 的机器 | **165 本地**，脚本 `vllm_musa_proj/bench_serving.py` + `auto_bench/`，跨网打 146 |

> LMCache 已确认不是空壳启动：19001 `/v1/models` ready、小 chat 请求成功、日志确认 `LMCacheConnectorV1` + `LocalCPUBackend` + `LMCache hit tokens` 统计字段。

---

## 1. 续1 没记录的核心机制：APC vs prefix-cache vs LMCache

> 这一节是续1 完全没碰的新地盘。续1 时代 V1 跑不起来，只验证过 GPU 端 prefix-cache 的崩溃和加速；LMCache 的实际行为（offload、命中顺序、跨卡）从没记录过。

### 1.1 APC 就是 prefix-cache（同一个东西）

- **APC = Automatic Prefix Caching**，是 vLLM 内置那套 prefix-cache 机制的功能名。续1 里讲的「prefix-cache 三档命中行为」说的就是 APC。
- 工作方式：paged attention 把 KV 切成 block 存在 **GPU HBM**，按内容 hash；新请求前缀 hash 命中 → 直接复用显存里的 block，**跳过这段 prefill 重算**。
- 限制：**只在 GPU 显存内、只在单实例内、块会被 LRU 淘汰**（显存不够给新请求时挤掉旧的）。

### 1.2 LMCache 与 APC 的关系：互补，不是重复

| | APC（GPU 前缀缓存） | LMCache |
|---|---|---|
| KV 存在哪 | GPU HBM | CPU 内存 / 硬盘 / 远程后端（当前 = `LocalCPUBackend` CPU 内存） |
| 复用代价 | **零搬运**（本就在显存） | 要把 KV 从 CPU/盘**搬回 GPU**（PCIe/网络，有耗时） |
| 容量 | 小（受 HBM，本例 216,704 token） | 大很多（CPU 内存几十~几百 GB） |
| 持久性 | 易失，进程重启即没 | 可持久，跨重启 |
| 共享范围 | 单实例单卡 | **可跨卡 / 跨机**（用远程后端时） |

可理解为**缓存金字塔**：APC = L1（小快），LMCache-CPU = L2（大稍慢），LMCache-disk/远程 = L3（巨大最慢）。

### 1.3 ⚠️ 关键陷阱：APC 会「掩盖」LMCache

vLLM v1 调度的查找顺序：
1. **先查本地 GPU APC** → 命中的 token（KV 已在显存，等于免费）；
2. **再问外部连接器（LMCache）**：在 APC 命中之外还能**额外**补多少 token（源码 `get_num_new_matched_tokens`，注意是 "new"）。

**推论**：若前缀整段还在 GPU APC 里，LMCache 没活可干 → `external_prefix_cache_hits` = 0。这**不是 LMCache 坏了**，而是 KV 已在显存（零搬运），从 CPU 再搬一遍反而更慢，系统正确地优先 APC。

**LMCache 只有在 APC 没命中时才现身**——前缀被挤出 GPU：工作集超显存容量、隔了很多别的请求被淘汰、或进程重启。这时 APC miss，LMCache 从 CPU 把 KV 捞回，省掉重算。

→ **直接后果**：小工作集冒烟看不到 LMCache 命中，必须把工作集撑过 GPU KV 容量才能逼出 LMCache。

### 1.4 命中率埋点：vLLM 自带 `/metrics` 就够，且能区分 APC / LMCache

| 计数器 | 含义 |
|---|---|
| `vllm:prefix_cache_hits_total` / `_queries_total` | **GPU APC**（内部）命中 / 查询 |
| `vllm:external_prefix_cache_hits_total` / `_queries_total` | **LMCache**（external connector）命中 / 查询 |

实测快照（2026-06-16）：
- 19001（LMCache）：APC 0/15，external 0/15（尚未真正复用前缀）；
- 19000（纯版）：APC 1024/18433（之前压测留下），external 0/0（纯版无 external，符合预期）。

压测前后各抓一次 `/metrics` 做差，即可把「TTFT 下降」干净归因到 APC 还是 LMCache，**无需解析日志**。

### 1.5 benchmark 指标速查：每个数到底在看什么

下面这些指标会反复出现在 Step1 / Step2 结果里。先把含义说清楚，否则后面表格只是在堆数字。

#### 延迟类指标

| 指标 | 全称 / 含义 | 主要反映哪一段 | 在 LMCache 测试里怎么用 |
|---|---|---|---|
| **TTFT** | Time To First Token，发出请求到收到第一个 token 的时间 | **prefill + 排队 + 调度** | LMCache 最核心指标。前缀 KV 命中后，理论上 prefill 变少，TTFT 应该下降 |
| **Mean TTFT** | 所有请求 TTFT 的平均值 | 整体首 token 延迟 | 容易被少数慢请求拉高，适合看总体负担 |
| **Median TTFT / P50 TTFT** | 一半请求低于这个 TTFT | 典型请求的首 token 延迟 | 比 mean 更抗异常值，适合看普通请求体验 |
| **P99 TTFT** | 99% 请求低于这个 TTFT | 首 token 长尾 | 判断是否有偶发排队、cache miss、调度抖动 |
| **E2E Latency** | End-to-End latency，请求从发出到完整输出结束的时间 | prefill + decode + 排队 | 输出较短时跟 TTFT 更相关；输出较长时会被 decode 稀释 |
| **TPOT** | Time Per Output Token，不含首 token 的平均 decode token 间隔 | **decode 阶段** | LMCache 不直接优化 decode，所以 TPOT 不是判断 LMCache 收益的主指标 |
| **ITL** | Inter-Token Latency，相邻输出 token 间隔 | decode 抖动 | 用来看 decode 稳定性，不适合单独证明 LMCache 有收益 |

为什么 TTFT 最重要：LMCache 省的是「长前缀 prefill 重算」。prefill 发生在第一个 token 出来之前，所以收益首先应该体现在 TTFT，而不是 TPOT。

#### 吞吐类指标

| 指标 | 含义 | 怎么读 |
|---|---|---|
| **Request throughput / req/s** | 每秒完成多少请求 | 综合指标，会同时受 prefill、decode、并发、输出长度影响 |
| **Input token throughput / tok/s** | 每秒处理多少输入 token | 更接近 prefill 能力；前缀复用强时，这个数可能变高 |
| **Output token throughput / tok/s** | 每秒生成多少输出 token | 更接近 decode 能力；LMCache 不直接优化这一段 |
| **Total token throughput** | 输入 token + 输出 token 的综合吞吐 | 可以看整体负载，但不能直接归因到 LMCache |
| **Peak concurrent requests** | benchmark 期间服务端/客户端看到的峰值并发 | 用来判断是否按预期施加并发压力 |
| **Concurrency** | 平均并发 | 如果两组对照并发不一致，延迟/吞吐不能直接比较 |

吞吐指标要小心解读：LMCache 可能让 prefill 变轻，从而提高 req/s 或 input tok/s；但如果 workload 里 cold miss 很多、decode 占比高、或者 CPU→GPU KV 搬运开销较大，吞吐收益会被稀释。

#### cache 命中类指标

| 指标 | 来源 | 含义 | 怎么判断 |
|---|---|---|---|
| `vllm:prefix_cache_queries_total` | vLLM 内部 APC | 查询 GPU prefix cache 的 token 数 | 说明 vLLM 在尝试做 GPU 前缀复用 |
| `vllm:prefix_cache_hits_total` | vLLM 内部 APC | GPU APC 命中的 token 数 | 工作集小于 GPU cache 时，这个通常很高 |
| `vllm:external_prefix_cache_queries_total` | V1 KV connector | 向外部 KV cache（这里是 LMCache）查询的 token 数 | 说明 connector 通路被调用 |
| `vllm:external_prefix_cache_hits_total` | V1 KV connector | LMCache 命中的 token 数 | **判断 LMCache 是否真正工作最关键的计数器** |

这些 counter 是进程级累计值，不能直接看单次压测后的绝对值，必须在压测前后各抓一次 `/metrics`，做 delta：

```text
delta = after - before
```

如果 `external_queries` 增长但 `external_hits=0`，通常说明 LMCache 通路被调用了，但 GPU APC 已经覆盖复用，或者 LMCache 里还没有对应 KV。如果 `external_hits>0`，才说明请求确实从 LMCache 拿回了 KV。

#### 本文里的判断优先级

读结果时按这个顺序判断：

1. **成功率**：请求是否都成功。失败率高时，性能数字没有意义。
2. **external hits**：LMCache 是否真正命中。没有 external hit，就谈不上 LMCache 性能收益。
3. **TTFT**：LMCache-hit 请求理论上应该降低首 token 延迟。
4. **Input token throughput / req/s**：看 prefill 压力是否下降。
5. **TPOT / output throughput**：只作为 decode 稳定性参考，不作为 LMCache 收益主证据。

---

## 2. Step0 实测：GPU KV 容量标尺（已完成）

读 146 容器内 19000 启动日志（`vllm_020/qwen3_8b_pure_fixed_19000.log`）：

| 项 | 值 |
|---|---|
| **GPU KV 容量** | **216,704 tokens**（19000 / 19001 配置一致） |
| Available KV memory | 29.76 GiB |
| gpu_memory_utilization | 0.65 |
| max_model_len | 4096（"Maximum concurrency for 4,096 tokens: 52.91x" → 216704/4096=52.9 ✓） |
| 每 token KV | ≈ **144 KB/token** = 2 × 36层 × 8 KV头 × 128 head_dim × 2字节（bf16）✓ |

**对 Step2 的标尺**：
- 复用前缀工作集需 **> 216,704 tokens**（2~4× → ~43万~87万 tokens）才能逼出 LMCache；
- `max_model_len=4096` 卡住单请求长度（单前缀 ≤ ~3K，留输出）→ 填满需 **~70 条满长前缀**，2× 要 **~140 条** round-robin；
- **CPU cache 须装得下**：2× 工作集 ≈ 43万 token × 144KB ≈ **62 GiB CPU RAM**，Step2 前要确认 LMCache `LocalCPUBackend` 上限够。

> 备注：`gpu_memory_utilization` 只决定「总预算」，不等于 KV 容量。KV 容量(token) = (总HBM×util − 模型权重 − 激活/CUDAgraph/框架开销) / 每token KV字节。vLLM 启动日志里 `GPU KV cache size: N tokens` 就是减完之后的权威值，直接读最准。

---

## 2.5 Step1 实测：LMCache 通路冒烟（已完成，2026-06-16）

在 **146 容器内**用 vLLM 的 venv（`/root/.virtualenvs/sglang-0.5.6/bin/python3`，带 transformers 4.57.1）跑 `bench_serving.py` 打本机 19001。负载 `generated-shared-prefix`：4 组前缀 × 每组复用 8 = 32 请求，`system-prompt-len=2048` / `question-len=128` / `output-len=64`，工作集 = 4×2048 = 8192 token «216,704（远小于容量）。

**性能结果**（32/32 成功，7.68s）：

| 指标 | 值 | 怎么看 |
|---|---:|---|
| Successful requests | 32/32 | 小冒烟所有请求成功，说明服务端和 benchmark 客户端路径正常 |
| Median TTFT | **119 ms** | 典型请求首 token 很快；这里主要来自 GPU APC 命中，不是 LMCache |
| Mean TTFT | 170 ms | 平均首 token 延迟正常，没有被大量慢请求拖垮 |
| P99 TTFT | 583 ms | 仍在亚秒级，没有明显长尾卡死 |
| Input token throughput | **9555 tok/s** | 前缀大量复用后，等效输入处理吞吐很高 |
| Output token throughput | 266 tok/s | decode 吞吐参考值；LMCache 不直接优化这一项 |

**命中率 delta（压测前→后，`/metrics`）**：

| 计数器 | 变化 | 含义 |
|---|---|---|
| APC `prefix_cache_queries` | 15 → 75,718 | vLLM 对 GPU prefix cache 发起了大量查询 |
| APC `prefix_cache_hits` | 0 → **61,824**（≈ **81.7%**） | **GPU APC 强命中**，说明小工作集主要被显存缓存接住 |
| LMCache `external_queries` | 15 → 13,894 | LMCache **被查询**，说明 V1 connector 通路是活的 |
| LMCache `external_hits` | 0 → **0** | **LMCache 零命中**，因为前缀还在 GPU APC 里，没必要从 CPU 搬回 |

**结论**：Step1 三点全达成——①请求正常 ②LMCache 通路活（external_queries 递增）③复用时 TTFT 低 + APC 强命中。`external_hits=0` **正是 §1.3 预测的「APC 掩盖 LMCache」的实测证据**（小工作集前缀全在 GPU，LMCache 没活干，非 bug）。→ 「看到 LMCache 真命中」必须等 Step2 超容量负载。

**附带发现**：venv 的 huggingface_hub 会先联网失败再 fallback 本地 tokenizer（一堆 `Connection reset` 警告，不影响结果）。Step2 加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 消除。

**运维约定**：bench 在 **146 上跑**（依赖齐全；165 默认 python 无 transformers，且 165 资源可能被占用）。

---

## 2.6 2026-06-17 更新：为什么最终先用 40GB LMCache CPU cache

这一段记录的是 **Step1 之后、正式做性能对照之前** 的配置收敛过程。它不是单纯调大一个参数，而是在确认：

1. 146 上新版 `vllm_musa` 的 V1 引擎能稳定跑；
2. LMCache 的 `LocalCPUBackend` 能被 vLLM V1 connector 正确加载；
3. CPU cache 容量足够大，能够承载「GPU APC 装不下但 LMCache 装得下」的 Step2 工作集；
4. 调参过程不能把 146 上已有服务和测试环境弄崩。

### 当前服务拓扑

146 容器 `vllm020_lmcache_test` 里同时保留两套 Qwen3-8B 服务，后续 benchmark 都打本机环回地址，避免跨机器网络抖动混进结果。

| 服务 | 端口 | GPU | served model name | 作用 |
|---|---:|---:|---|---|
| 纯新版 vLLM | 19000 | GPU0 | `qwen3-8b-pure-fixed` | baseline：只用 vLLM 内置 GPU APC，不挂 LMCache |
| LMCache 版 | 19001 | GPU1 | `qwen3-8b-lmcache` | 实验组：V1 引擎 + `LMCacheConnectorV1` + `LocalCPUBackend` |

两边启动参数保持一致，除了 19001 多了 LMCache connector：

```bash
--gpu-memory-utilization 0.65
--block-size 64
--max-model-len 4096
--tensor-parallel-size 1
--pipeline-parallel-size 1
```

19001 额外参数：

```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### GPU KV cache 是什么，为什么它决定 Step2 的下限

从两边启动日志读到的权威值一致：

| 项 | 值 | 怎么理解 |
|---|---:|---|
| Available KV cache memory | 29.76 GiB | vLLM 在扣掉权重、激活、CUDA graph 等开销后，留给 KV cache 的显存预算 |
| GPU KV cache size | 216,704 tokens | vLLM 最终能在 GPU APC 里容纳的 KV token 数 |
| block size | 64 | KV cache 以 64-token block 管理 |
| max model len | 4096 | 单请求最大上下文长度 |

这里最关键的是 **216,704 tokens**。只要复用前缀工作集小于这个数，GPU APC 大概率自己就能接住复用，LMCache 即使启用了也不会贡献 `external_prefix_cache_hits`。所以 Step2 的工作集必须大于 216,704 tokens。

### CPU cache 为什么不能直接设 80GB

Claude 原先的计划是把 LMCache CPU cache 从 20GB 调到 80GB。这个思路本身对：如果要构造 2x GPU KV 工作集，粗略估算需要：

```text
2 × 216,704 tokens × 144 KB/token ≈ 62 GiB
```

所以 80GB 是一个合理目标。但在 146 当前 MUSA runtime + 容器环境下，实测 80GB 和 64GB 都启动失败。

| `LMCACHE_MAX_LOCAL_CPU_SIZE` / `max_local_cpu_size` | 启动结果 | 现象 |
|---:|---|---|
| 80GB | 失败 | `LMCacheEngine marked as init failed: musaHostAlloc failed: 205`，随后 CUDA graph capture 阶段 MUDNN `FillOp` 失败 |
| 64GB | 失败 | 同样的 `musaHostAlloc failed: 205` / MUDNN `FillOp` 失败 |
| 40GB | 成功 | 日志确认 `max_local_cpu_size: 40.0`，并创建 `LocalCPUBackend` |

这个失败点发生在 LMCache 初始化 pinned/host memory 相关资源时，不是模型权重加载失败，也不是 vLLM V1 引擎不能推理。换句话说：

- 纯新版 `vllm_musa` 没问题；
- 19001 的 LMCache connector 能加载；
- 失败是 **CPU cache 设得太大后，MUSA host allocation 失败**；
- 失败后有可能留下 `VLLM::EngineCore` 残留进程，占住 GPU 显存，需要手动清理后再重启。

因此后续所有 Step2 设计都先按 **40GB CPU cache** 这个已验证上限来做，而不是继续沿用 80GB 方案。

### 40GB 配置下为什么要重跑 Step1

改了 LMCache CPU cache 容量后，要先重跑小冒烟，确认不是「服务能启动但请求路径坏了」。Step1 的目的仍然只有三个：

1. 19001 能正常接 OpenAI compatible completion 请求；
2. vLLM V1 会调用 external connector，也就是 `external_prefix_cache_queries_total` 会增长；
3. 小工作集下不应该强求 LMCache hit，因为 GPU APC 足够装下。

Step1 复测负载：

| 参数 | 值 | 含义 |
|---|---:|---|
| dataset | `generated-shared-prefix` | 使用共享前缀合成数据集，而不是普通 random 请求；只有共享前缀才能测试 APC/LMCache |
| groups | 4 | 生成 4 个不同 shared prefix；每个 group 代表一个可复用前缀 |
| prompts/group | 8 | 每个 shared prefix 生成 8 个不同问题，所以同一个 prefix 会被访问 8 次 |
| system prompt len | 2048 | 每个 shared prefix 的目标 token 长度 |
| question len | 128 | 每个请求独有后缀长度；用于模拟“相同系统提示 + 不同用户问题” |
| output len | 64 | 每个请求最多生成 64 token；Step1 不关注 decode，保持短输出 |
| shared-prefix 工作集 | 4 × 2048 = 8192 tokens | 所有不同 shared prefix 的总规模，用来和 GPU KV cache 容量比较 |

8192 tokens 远小于 GPU KV cache 的 216,704 tokens，所以它验证的是 **通路健康**，不是 LMCache 性能收益。

Step1 复测结果：

| 指标 | 值 | 解释 |
|---|---:|---|
| 成功请求 | 32/32 | 服务和 benchmark 路径正常 |
| Median TTFT | 190.81 ms | 小工作集复用场景下首 token 正常 |
| Mean TTFT | 272.89 ms | 有少量波动，但无失败 |
| P99 TTFT | 461.68 ms | 没有秒级异常长尾 |
| APC hits delta | 61,824 | GPU 内部 prefix cache 命中强 |
| APC queries delta | 75,703 | vLLM 确实在查 prefix cache |
| LMCache external queries delta | 13,879 | vLLM V1 确实调用了 LMCache connector |
| LMCache external hits delta | 0 | 预期结果：前缀都还在 GPU APC 里，LMCache 没机会命中 |

结论：40GB 配置不是空启动，LMCache connector 通路是活的；但 Step1 不能说明 LMCache 有性能优势，因为它没有制造 GPU APC 淘汰。

---

## 2.7 Step2 首轮超容量对照（2026-06-17）

Step2 的目标不是「随便跑一个大 benchmark」，而是构造一个能回答下面问题的负载：

> 当共享前缀工作集 **大于 GPU APC 容量**、但理论上 **小于 LMCache CPU cache 容量** 时，挂 LMCache 的 19001 是否能从 CPU cache 找回已经被 GPU 淘汰的 KV？

如果这个问题都回答不了，后面的吞吐曲线、goodput 曲线没有意义。因此首轮 Step2 先选一个轻量但能过容量线的 case。

### 为什么选 80 groups × 3072 tokens

已知：

- GPU KV cache：216,704 tokens；
- LMCache CPU cache：40GB；
- Qwen3-8B bf16 KV 粗算：约 144KB/token；
- 单请求 `max_model_len=4096`，不能把 shared prefix 设得太接近 4096，否则还要留 question/output 空间。

首轮参数：

| 参数 | 值 | 含义 | 选择原因 / 调整影响 |
|---|---:|---|---|
| dataset | `generated-shared-prefix` | 合成共享前缀数据集 | 必须使用共享前缀，否则 APC/LMCache 没有复用对象 |
| groups | 80 | 不同 shared prefix 的数量 | `groups × system_prompt_len` 决定不同前缀工作集；增大 groups 更容易挤爆 GPU APC，但也更吃 LMCache CPU cache |
| prompts/group | 2 | 每个 shared prefix 被访问几次 | 至少要 2：第一次 cold fill，第二次 warm revisit；再增大会增加复用次数，但也会拉长测试 |
| system prompt len | 3072 | shared prefix 目标 token 长度 | 接近 `max_model_len=4096`，但给 question/output 留空间；增大会放大 prefill 成本和 LMCache 收益信号 |
| question len | 128 | 每次请求独有问题长度 | 保证同一 group 内不是完全相同请求，而是“相同前缀 + 不同问题” |
| output len | 64 | 最大输出 token 数 | 保持短输出，让 TTFT/prefill 占主要比例；输出越长，decode 越会稀释 LMCache 收益 |
| num prompts | 160 | 总请求数 = `groups × prompts/group` | 80×2；包含 cold/warm 混合请求，但此轮会被随机 shuffle |
| max concurrency | 8 | 客户端最多同时在途请求数 | 控制压测压力；太高会引入排队/调度抖动，太低测试耗时很长 |
| request rate | `inf` | 所有请求尽快发出，受 max concurrency 限制 | 沿用 benchmark 默认突发模式；适合施压，但不适合做干净冷热分离 |

理论 shared-prefix 工作集：

```text
80 groups × 3072 tokens = 245,760 tokens
```

这个值约等于：

```text
245,760 / 216,704 ≈ 1.13 × GPU KV cache
```

也就是说，它只比 GPU APC 容量大 13%。这是一个保守选择：足够越过 GPU 容量线，但不至于把 40GB LMCache CPU cache 撑爆。首轮目标是 **先看到 external hit**，不是一次性测出最大收益。

实际生成数据后，benchmark 统计到总输入为 538,610 tokens。这个值比 245,760 大，是因为它包含：

- 每个请求的 shared system prompt；
- 每个请求自己的 question；
- tokenizer 实际编码长度与目标长度的轻微差异；
- 每个 group 有 2 个请求，所以总输入会包含重复前缀。

判断是否超 GPU 容量时，看的是 **不同 shared prefix 的工作集**，不是所有请求 input tokens 的简单总和。

### 当前 benchmark 脚本的一个重要限制

`bench_serving.py` 的 `generated-shared-prefix` 会生成 `groups × prompts_per_group` 个请求后执行 `random.shuffle(input_requests)`。

这意味着当前首轮 Step2 不是严格的：

```text
先冷启动灌满所有 group，再按同样顺序 warm 重访所有 group
```

而是混合随机顺序。结果会带来两个影响：

1. 有些第二次访问可能离第一次很近，仍然被 GPU APC 命中；
2. 有些第二次访问可能被挤出 GPU，才会落到 LMCache external hit；
3. 冷请求和 warm 请求混在总平均里，会稀释 LMCache 对 TTFT 的收益。

所以这轮结果适合判断 **LMCache 是否真的参与命中**，但还不是最终性能结论。最终要做更干净的收益评估，需要定制请求顺序或改 benchmark：冷/热分离、强制 round-robin、只统计 warm 重访部分。

### LMCache arm：19001 结果

命令打的是 19001，served model name 为 `qwen3-8b-lmcache`。结果文件在 146 容器内：

```text
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/lmcache40_step2_overcap_20260617_081346.json
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/lmcache40_step2_overcap_20260617_081346.log
```

性能结果：

| 指标 | 值 | 怎么看 |
|---|---:|---|
| Successful requests | 160/160 | 负载完整跑完，无请求失败 |
| Benchmark duration | 60.46 s | 总耗时略短于 baseline |
| Request throughput | 2.65 req/s | 略高于 baseline |
| Input token throughput | 8908.57 tok/s | 略高于 baseline |
| Output token throughput | 169.37 tok/s | 略高于 baseline |
| Mean E2E | 3015.71 ms | 略低于 baseline |
| Median E2E | 3003.70 ms | 略低于 baseline |
| Mean TTFT | 817.78 ms | 比 baseline 差 |
| Median TTFT | 748.80 ms | 比 baseline 差 |
| P99 TTFT | 1768.32 ms | 比 baseline 差 |
| Mean TPOT | 34.89 ms | 比 baseline 好；但 TPOT 主要是 decode，不是 LMCache 的核心指标 |

metrics 前后差：

| 计数器 | delta | 解释 |
|---|---:|---|
| APC `prefix_cache_hits_total` | 188,864 | GPU APC 仍然命中了大量 token |
| APC `prefix_cache_queries_total` | 541,962 | 本轮总 prefix-cache 查询量 |
| LMCache `external_prefix_cache_hits_total` | 67,328 | **关键结果：LMCache 确实命中了 CPU cache 中的 KV** |
| LMCache `external_prefix_cache_queries_total` | 353,098 | APC 未完全覆盖时，vLLM 向 LMCache 查询的 token 数 |

`external_prefix_cache_hits_total=67,328` 是这轮最重要的正向证据：这说明 19001 不只是挂了参数，而是真正通过 V1 connector 从 LMCache 拿到了 KV。

### Pure arm：19000 结果

命令打的是 19000，served model name 为 `qwen3-8b-pure-fixed`。使用同一个 generated-shared-prefix 缓存文件，保证请求数据一致。结果文件在 146 容器内：

```text
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/pure_step2_overcap_20260617_081935.json
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/pure_step2_overcap_20260617_081935.log
```

性能结果：

| 指标 | 值 | 怎么看 |
|---|---:|---|
| Successful requests | 160/160 | baseline 也完整跑完 |
| Benchmark duration | 62.43 s | 比 LMCache 慢约 1.97s |
| Request throughput | 2.56 req/s | 略低于 LMCache |
| Input token throughput | 8627.00 tok/s | 略低于 LMCache |
| Output token throughput | 164.02 tok/s | 略低于 LMCache |
| Mean E2E | 3113.79 ms | 略高于 LMCache |
| Median E2E | 3129.26 ms | 略高于 LMCache |
| Mean TTFT | 632.59 ms | 比 LMCache 好 |
| Median TTFT | 654.10 ms | 比 LMCache 好 |
| P99 TTFT | 1574.61 ms | 比 LMCache 好 |
| Mean TPOT | 39.38 ms | 比 LMCache 差 |

metrics 前后差：

| 计数器 | delta | 解释 |
|---|---:|---|
| APC `prefix_cache_hits_total` | 188,864 | 和 LMCache arm 一样，说明两边 GPU APC 行为非常接近 |
| APC `prefix_cache_queries_total` | 541,962 | 和 LMCache arm 一样 |
| LMCache `external_prefix_cache_hits_total` | 0 | 纯版没有 external connector，符合预期 |
| LMCache `external_prefix_cache_queries_total` | 0 | 纯版不会查询 LMCache |

### LMCache vs pure 首轮对比该怎么读

把两边关键指标并排看：

| 指标 | LMCache 19001 | Pure 19000 | 表面差异 | 正确解读 |
|---|---:|---:|---:|---|
| Successful requests | 160/160 | 160/160 | 持平 | 两边服务都能承载这轮负载，结果可比较 |
| Benchmark duration | 60.46 s | 62.43 s | LMCache 快 1.97 s | 整体完成时间略好，但差距不大 |
| Request throughput | 2.65 req/s | 2.56 req/s | LMCache +3.5% | 整体吞吐略好，但不能单独证明 LMCache 收益 |
| Input token throughput | 8908.57 tok/s | 8627.00 tok/s | LMCache +3.3% | prefill 相关吞吐略好，方向符合 LMCache 可能收益 |
| Median E2E | 3003.70 ms | 3129.26 ms | LMCache 好约 126 ms | 端到端略好，但包含 decode 和排队 |
| Median TTFT | 748.80 ms | 654.10 ms | LMCache 差约 95 ms | 首 token 延迟没有变好，说明当前 workload 还不能证明性能收益 |
| P99 TTFT | 1768.32 ms | 1574.61 ms | LMCache 差约 194 ms | 长尾也没有改善，可能受搬运/调度/冷热混合影响 |
| Mean TPOT | 34.89 ms | 39.38 ms | LMCache 好 | 主要是 decode 阶段差异，不能作为 LMCache 命中收益主证据 |
| LMCache external hits | 67,328 | 0 | LMCache 有命中 | **功能验证通过**：LMCache 确实参与了 KV 复用 |

这张表的重点是区分两类结论：

- **功能结论**：成立。`external_prefix_cache_hits_total` 增长到 67,328，说明 LMCache V1 connector + LocalCPUBackend 已经真实命中。
- **性能结论**：暂不成立。TTFT 没有改善，而 TTFT 才是 LMCache 应该优先改善的指标；吞吐/E2E 略好只能说明有趋势，不能直接下结论。

### 为什么「同一批测试数据」下 pure 不一定天然更差

直觉上看，同一批 shared-prefix 请求打两套服务：

```text
pure：    GPU APC 命中就复用；APC miss 就重算 prefill
LMCache： GPU APC 命中就复用；APC miss 还有 LMCache 兜底
```

所以好像 LMCache 应该天然不差于 pure。但这个推论隐含了一个前提：

> LMCache 兜底的额外成本必须小于它省掉的 prefill 重算成本。

实际系统里这个前提不一定总成立，因为 LMCache 不是免费的第三层缓存。

#### 三种请求路径的 TTFT 成本不同

把请求按 cache 行为拆开看，TTFT 成本大致是：

| 请求类型 | KV 在哪里 | 做了什么 | 预期 TTFT |
|---|---|---|---|
| GPU APC hit | GPU HBM | 直接复用 GPU 上已有 KV | 最快 |
| LMCache hit | CPU cache | 从 CPU 把 KV 搬回 GPU，再继续 decode | 通常比重算快，但比 GPU APC hit 慢 |
| cold miss / full recompute | 没有缓存 | 完整 prefill，必要时再写入 cache | 通常最慢 |

因此 LMCache 只应该在第三种场景里赢：**pure 需要重算 prefill，而 LMCache 能从 CPU cache 找回 KV**。它不应该比 GPU APC hit 更快，因为 GPU APC hit 不需要搬运。

#### 本轮为什么没有天然变快

本轮数据里，两边 GPU APC 命中完全一样：

```text
APC prefix_cache_hits_total delta = 188,864 tokens
APC prefix_cache_queries_total delta = 541,962 tokens
```

这说明大量复用仍然被 GPU APC 接住。对这些请求来说，pure 和 LMCache 都走最快路径，LMCache 没有额外收益。

LMCache 额外发生的是：

```text
external_prefix_cache_queries_total delta = 353,098 tokens
external_prefix_cache_hits_total    delta = 67,328 tokens
```

这组数字说明两件事：

1. vLLM 向 LMCache 查询了很多 token；
2. 但只有其中一部分真正命中。

未命中的 external query 仍然有 lookup 开销；命中的部分还要承担 CPU → GPU KV 搬运开销。也就是说，19001 相比 pure 多了：

- external lookup；
- LMCache hit 后的 CPU→GPU KV 搬运；
- cold fill 阶段可能把 KV 写入 CPU cache；
- LMCache 相关调度/同步开销。

如果这些额外开销叠加起来，超过了本轮 external hit 省下的 prefill 重算时间，那么总 TTFT 就可能不降反升。

#### 为什么必须做冷热分离

当前 `bench_serving.py` 把请求随机打乱后一起统计，最终的 median/mean TTFT 混合了三类请求：

```text
cold fill：第一次见到某前缀，pure 和 LMCache 都要完整 prefill
GPU APC hit：前缀仍在 GPU，pure 和 LMCache 都很快
LMCache warm hit：GPU APC miss，但 LMCache 命中，LMCache 才可能赢
```

这三类请求混在一起平均，会把真正需要比较的部分稀释掉。

正确要比较的是 **warm revisit**：

```text
pure warm revisit：
  目标前缀已经被 GPU APC 淘汰，只能重算 prefill

LMCache warm revisit：
  目标前缀已经被 GPU APC 淘汰，但 CPU LMCache 里还有 KV，可以取回
```

只有这个对比才回答核心问题：

> LMCache 从 CPU 搬 KV 回 GPU，是否比 vLLM 重新 prefill 这段前缀更快？

所以本轮看到 external hit 只能证明 LMCache 起到了功能作用；要证明性能收益，下一轮必须把 cold fill 和 warm revisit 拆开，至少在统计上只看 warm revisit 请求。

### 这轮结果到底说明了什么

这轮可以明确说明三件事：

1. **LMCache 已经真正工作**：19001 的 `external_prefix_cache_hits_total` 增加了 67,328 tokens。
2. **首轮 workload 确实越过了 GPU APC 容量线**：否则 external hit 大概率仍为 0。
3. **当前测试还不能宣称 LMCache 性能收益成立**：LMCache 的吞吐/E2E 略好，但 TTFT 明显比 pure 差。

TTFT 没有变好并不等于 LMCache 没用，当前更可能是测试设计还不够干净：

- 工作集只比 GPU KV 容量大 13%，APC 仍然覆盖了大部分复用；
- 请求顺序被 `random.shuffle` 打乱，冷请求和热请求混在一起；
- LMCache 从 CPU 搬 KV 回 GPU 有搬运开销，短输出/短前缀下可能抵消一部分 prefill 省下来的时间；
- 总平均 TTFT 同时包含 cold miss、APC hit、LMCache hit 三类请求，无法单独看 LMCache-hit 请求的收益；
- 这轮并发为突发模式，调度和排队也会影响 TTFT。

因此这轮 Step2 的定位应该写成：

> **LMCache external hit 验证通过；性能收益尚未验证，需要下一轮做冷热分离和更强淘汰顺序。**

### 下一轮应该怎么改（已按该思路完成，见 §2.8）

为了让一个没接触过项目的人能继续，下一轮建议按下面顺序做：

1. 固定 19000/19001 当前配置，不再先动服务端。
2. 生成同一批 shared-prefix 请求，但不要随机 shuffle。
3. 先发一轮 cold fill：访问所有 group 的第 1 个 prompt，把 KV 写入 APC/LMCache。
4. 再发足够多干扰请求或 round-robin 访问其他 group，使 GPU APC 容量被挤压。
5. 最后只统计 warm revisit：访问每个 group 的第 2 个 prompt。
6. 对比 warm revisit 的 TTFT，并同时看：
   - pure：APC hit 有多少，TTFT 是否回到重算水平；
   - LMCache：external hit 有多少，TTFT 是否低于 pure。

如果不改 `bench_serving.py`，也可以先把工作集继续加大一点，例如 90 groups × 3072 tokens（276,480 tokens，约 1.28× GPU KV）。但 40GB CPU cache 边界不宽，继续加大要注意 LMCache 自身也可能开始淘汰。更稳的做法是先改请求顺序，而不是盲目扩大规模。

---

## 2.8 Step2 冷热分离复测：LMCache 性能收益验证（2026-06-17）

上一轮 `bench_serving.py` 随机 shuffle 的问题是：cold miss、GPU APC hit、LMCache hit 混在一起统计，TTFT 不能直接解释。为了解决这个问题，新增了一个专用脚本：

```text
/data/my_vllm_test/vllm_020/lmcache_cold_warm_revisit_bench.py
```

脚本逻辑：

1. 生成 `N` 个不同 shared prefix；
2. **cold 阶段**：每个 prefix 只访问一次，把 KV 写入 APC/LMCache，但不拿这一阶段判断收益；
3. **warm 阶段**：按相同 group 顺序再次访问每个 prefix，但换不同 question；
4. 分别记录 cold/warm 的 TTFT/E2E；
5. 在压测前、cold 后、warm 后各抓一次 `/metrics`，所以能单独看到 warm 阶段的 APC/LMCache 命中。

### 测试用例怎么构造的（先看这个，再看参数）

上面 5 步比较抽象。这一节用具体例子说明脚本到底生成了什么样的请求，让没接触过的人能看懂 `group` / `cold` / `warm` 实物长什么样。

**第 1 层：三种"积木"**

脚本先用固定 seed 随机造三类 token 片段（`sample_text()`：从模型词表里随机取 token，再 decode 成文本）：

| 积木 | 数量 | 长度 | 作用 |
|---|---|---|---|
| `prefix[g]` 共享前缀 | groups 个（本轮 80） | 各 3072 token | 每个 group 一段，是"会被复用"的部分，模拟一段长系统提示 / 长文档 |
| `cold_question[g]` | groups 个（80） | 各 128 token | cold 阶段每个 group 用的问题 |
| `warm_question[g]` | groups 个（80） | 各 128 token | warm 阶段每个 group 用的问题，**故意和 cold 的不一样** |

> 为什么用随机 token 而不是真实文章？为了保证 group 之间**绝不会意外共享前缀**——前缀复用必须完全由我们控制，否则真实语料里的相似句子会污染命中率统计。

**第 2 层：一条请求长什么样**

一条请求 = 某个 group 的共享前缀 + 这一阶段的问题，拼成一个 prompt 发给 `/v1/completions`：

```
┌──────────────────────────────────────────────┬────────────────────┐
│  prefix[g]   (3072 token, 该 group 固定不变)     │ question (128 token) │
└──────────────────────────────────────────────┴────────────────────┘
   ↑ 会被复用的部分（APC / LMCache 命中的就是这段）      ↑ 每次都换、必须现算
prompt = prefix[g] + "\n\n" + question          max_tokens=32, temperature=0, stream
```

实测编码后单条输入约 3366 token（3072+128 再加分隔符/tokenizer 边界）。**关键**：可复用的只有前缀那 3072 token，问题 128 token 每次都是新的、必须 prefill。这也解释了为什么"命中"后 TTFT 不是 0——还得现算新问题 + 把前缀 KV 搬回来。

**第 3 层：完整请求序列（共 160 条 = 80×2，严格按 group 顺序，不打乱）**

```
cold 阶段（80 条，只负责灌 cache，不计入收益）:
  req   1: prefix[0]  + cold_question[0]
  req   2: prefix[1]  + cold_question[1]
   ...
  req  80: prefix[79] + cold_question[79]
  ──→ 抓 /metrics（after_cold）

warm 阶段（80 条，只统计这一段做对比）:
  req  81: prefix[0]  + warm_question[0]   ← 和 req1 同一个前缀，换了问题
  req  82: prefix[1]  + warm_question[1]
   ...
  req 160: prefix[79] + warm_question[79]
  ──→ 抓 /metrics（after_all）
```

**第 4 层：一个 group 的生命周期（以 group 0 为例）**

| 时刻 | 这条请求 | GPU APC | LMCache(CPU) | 这次怎么算 |
|---|---|---|---|---|
| cold | prefix[0]+cold_q[0] | 空 → 写入 | 空 → 写入 | 完整 prefill（~3.3K token），顺便把前缀 KV 存进两层 cache |
| cold 继续 | prefix[1..79] 陆续灌入 | 装不下 → **把 group0 挤出去** | 都留着（CPU 大） | —— |
| warm | prefix[0]+warm_q[0] | **miss**（已被挤出） | **hit** | 前缀 KV 从 CPU 搬回（省掉 prefill），只现算新问题 128 token |

纯版（19000）没有 LMCache 那一列，warm 时 GPU APC 也 miss → 只能把 3072 前缀**整段重算**。这就是 pure warm TTFT 高（728ms）、LMCache warm 低（262ms）的根本原因。

**第 5 层：为什么这个顺序能保证 warm 时 APC 已经淘汰**

- cold 灌入 80×3072 = 245,760 token 前缀，超过 GPU APC 容量 216,704 约 13%；GPU 只装得下约 `216704 / 3366 ≈ 64` 个前缀。
- 所以 cold 结束时，最早灌的至少 `80 − 64 = 16` 个 group（脚本算出的 `estimated_evicted_candidate_groups = 16`）必然已被挤出 GPU。
- 实测更彻底：warm 阶段每发一条又把一个前缀搬回 GPU、挤掉另一个，连环淘汰，最终 **warm 全部 80 条 APC hits 都是 0**（两臂 `warm_delta` 都证实）。脚本还单独汇总了这 16 个"铁定被淘汰"的 group（`warm_evicted_candidates`），它们的 TTFT（LMCache 264.86ms / pure 724.70ms）和整体 warm 几乎一致 → 说明结论稳健，不是被少数残留的 APC 命中拉偏的。

> **一句话理解整个设计**：cold 把前缀灌到溢出 GPU，warm 再回头访问——此时前缀只剩在 LMCache 的 CPU 里，于是"LMCache 兜底 vs 纯版重算"的差距被干净地隔离出来测到。

### 测试参数

| 参数 | 值 | 含义 | 为什么这样设 / 改动影响 |
|---|---:|---|---|
| groups | 80 | 不同 shared prefix 的数量；每个 group 是一个独立可复用前缀 | 这是最关键参数之一。`groups × prefix_len` 决定不同前缀工作集，本轮为 245,760 tokens，略大于 GPU KV cache 216,704 tokens，从而逼出 APC miss |
| cold requests | 80 | cold 阶段请求数，每个 group 访问一次 | 等于 groups。用途是把每个 prefix 的 KV 写进 APC/LMCache，不用于判断收益 |
| warm requests | 80 | warm 阶段请求数，每个 group 再访问一次 | 等于 groups。只统计这一阶段，比较 pure 重算 vs LMCache 取回 |
| prefix len | 3072 | 每个 shared prefix 的目标 token 长度 | 接近单请求上限但留出 question/output 空间；越长 prefill 越贵，LMCache 命中收益越明显，但也更占 cache |
| question len | 128 | 每次请求独有后缀 token 长度 | cold/warm 用不同 question，模拟“相同系统提示 + 不同用户问题”；如果设 0，会退化成完全重复请求 |
| output len | 32 | 最大生成 token 数 | 分离测试关注 TTFT，不关注长 decode，所以比上一轮 64 更短；输出越长，E2E 越受 decode 影响 |
| concurrency | 4 | 客户端并发数，每批同时发 4 个请求 | 折中速度和可解释性；并发太高会放大排队/调度噪声，太低测试耗时过长 |
| seed | 2026061702 | 随机数种子，控制 prefix/question 生成 | pure 和 LMCache 必须使用同一个 seed，才能保证两边请求内容一致 |
| GPU KV cache | 216,704 tokens | vLLM 日志读到的 GPU APC 容量 | 判断工作集是否超过 APC 容量的基准 |
| 理论 shared-prefix 工作集 | 80 × 3072 = 245,760 tokens | 不同 shared prefix 的总 token 数 | 比 GPU KV cache 高约 13%，目标是让 warm 阶段 APC 不能完整保留所有 prefix |

这几个参数的关系必须一起看：

```text
总请求数 = groups × 2
cold requests = groups
warm requests = groups
理论 shared-prefix 工作集 = groups × prefix_len
```

其中 `groups` 和 `prefix_len` 决定是否能压过 GPU APC 容量；`question_len` 和 `output_len` 决定单请求是否还落在 `max_model_len=4096` 内；`concurrency` 决定测试压力和 TTFT 噪声；`seed` 决定 pure/LMCache 两边是否可比。

实际 tokenizer 编码后，平均 prompt 长度约 3.3K tokens：

| arm | min prompt len | max prompt len | mean prompt len |
|---|---:|---:|---:|
| LMCache 19001 | 3320 | 3446 | 3366.64 |
| Pure 19000 | 3311 | 3423 | 3375.94 |

这说明本轮工作集确实超过 GPU KV cache，且 warm 阶段有机会观察「GPU APC miss 后怎么办」。

### LMCache arm：19001 冷/热结果

结果文件：

```text
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/lmcache40_cold_warm_20260617_19001.json
```

| 阶段 | requests | Median TTFT | Mean TTFT | P99 TTFT | Median E2E | 解释 |
|---|---:|---:|---:|---:|---:|---|
| cold | 80/80 | 912.74 ms | 875.51 ms | 925.66 ms | 1597.89 ms | 第一次访问，LMCache 还没有 KV，需要完整 prefill，并写入 cache |
| warm | 80/80 | **262.70 ms** | **261.53 ms** | 271.19 ms | 949.46 ms | 第二次访问，GPU APC 未命中，但 LMCache 从 CPU cache 取回 KV |

metrics 分阶段 delta：

| 阶段 | APC hits | APC queries | LMCache external hits | LMCache external queries | 解释 |
|---|---:|---:|---:|---:|---|
| cold | 0 | 269,306 | 0 | 269,306 | 通路被查询，但 cache 里还没有这些新 prefix |
| warm | 0 | 269,357 | **245,760** | 269,357 | 关键结果：warm 阶段 LMCache 命中了完整 shared-prefix 工作集 |

这里 `external_prefix_cache_hits_total=245,760` 非常干净，正好等于：

```text
80 groups × 3072 prefix tokens = 245,760 tokens
```

说明 warm 阶段命中的不是偶然碎片，而是我们刻意构造的 shared prefix。

### Pure arm：19000 冷/热结果

结果文件：

```text
/data/my_vllm_test/vllm_020/bench_results_lmcache_step2/pure_cold_warm_20260617_19000.json
```

| 阶段 | requests | Median TTFT | Mean TTFT | P99 TTFT | Median E2E | 解释 |
|---|---:|---:|---:|---:|---:|---|
| cold | 80/80 | 729.64 ms | 648.48 ms | 842.08 ms | 1504.06 ms | 第一次访问，需要完整 prefill |
| warm | 80/80 | **728.62 ms** | **649.70 ms** | 841.58 ms | 1504.28 ms | 第二次访问时 GPU APC 没有命中，也没有 LMCache 兜底，所以仍接近完整 prefill |

metrics 分阶段 delta：

| 阶段 | APC hits | APC queries | LMCache external hits | LMCache external queries | 解释 |
|---|---:|---:|---:|---:|---|
| cold | 0 | 270,057 | 0 | 0 | 纯版没有 external connector |
| warm | 0 | 270,094 | 0 | 0 | warm 阶段没有 GPU APC 命中，只能重算 |

### 关键对比：只看 warm revisit

这一次的核心是只比较 warm 阶段，因为 cold 阶段只是灌 cache。

| 指标 | LMCache warm | Pure warm | 差异 | 结论 |
|---|---:|---:|---:|---|
| Median TTFT | **262.70 ms** | 728.62 ms | LMCache 降低约 **465.92 ms**（约 64%） | LMCache 明显降低首 token 延迟 |
| Mean TTFT | **261.53 ms** | 649.70 ms | LMCache 降低约 **388.18 ms**（约 60%） | 平均首 token 延迟也明显改善 |
| P99 TTFT | **271.19 ms** | 841.58 ms | LMCache 降低约 **570.39 ms** | 长尾首 token 延迟也被压低 |
| Median E2E | **949.46 ms** | 1504.28 ms | LMCache 降低约 **554.83 ms** | 短输出场景下端到端也改善 |
| APC hits | 0 | 0 | 持平 | 两边都不是靠 GPU APC 赢 |
| LMCache external hits | **245,760** | 0 | LMCache 命中完整 shared-prefix 工作集 | 收益可以归因到 LMCache |

这轮终于能把结论写清楚：

> 在 GPU APC 已经无法命中的 warm revisit 场景下，LMCache 从 CPU cache 取回 shared-prefix KV，显著降低 TTFT。  
> 这证明 LMCache 不只是功能链路可用，而且在该构造负载下确实带来了性能收益。

### 为什么这轮比上一轮更可信

上一轮随机 shuffle 的结果是：

- LMCache 有 external hit；
- 但 TTFT 比 pure 差；
- 因为 cold miss、APC hit、LMCache hit 混在一起，无法单独解释。

本轮分离测试消除了这个问题：

1. cold 和 warm 分开统计；
2. warm 阶段两边 APC hits 都是 0；
3. pure warm 没有 external connector，只能重算；
4. LMCache warm external hits 正好等于 shared-prefix 工作集；
5. warm TTFT 从 pure 的 728.62 ms 降到 LMCache 的 262.70 ms。

因此本轮已经能证明 LMCache 对「APC miss 后的 shared-prefix warm revisit」有明确收益。

---

## 3. 性能指标考虑：这些指标能测出 LMCache 收益吗？

> 评判脚手架沿用 `vllm_musa_proj/reademe.md` 的指标体系（24 列 CSV + `analyze.py` 派生指标 + 找拐点方法论），但**负载和埋点要改**。

### 3.1 哪些指标真正反映 LMCache 收益

LMCache 优化的是 **prefill 阶段**（命中→跳过前缀 prefill）：

| 指标 | 能否反映 LMCache 收益 | 原因 |
|---|---|---|
| **TTFT**（mean/median/p99） | ✅ 头号指标 | 命中→跳过前缀 prefill→首 token 大幅提前 |
| **in_tok_tp**（输入吞吐/prefill 能力） | ✅ | 省的就是 prefill |
| **req_tp / Goodput / 吞吐-延迟拐点曲线** | ✅ 前缀密集负载下 | prefill 算力省出来→吞吐升；拐点曲线最有说服力 |
| **E2E latency** | ⚠️ 间接 | 长前缀+短输出时跟 TTFT 降；长输出被 decode 稀释 |
| **TPOT / ITL** | ❌ 无效 | decode 阶段，LMCache 不动，几乎不变（别用它判断收益） |
| ok/fail、duration、显存 | ❌ | 健康/容量指标 |

### 3.2 不能「完全监测」——缺的三类东西

1. **负载必须有前缀复用**：reademe 里 `range-ratio=1.0` + random 语料，每请求输入独立随机 → **命中率恒 0**，测不出收益。必须换 `generated-shared-prefix`。
2. **命中率埋点**（已解决，见 1.4）：用 `/metrics` 的 external vs internal 计数器，否则 TTFT 降了无法归因。
3. **冷热分离**：第一遍灌 cache（两边都 miss），第二遍才分化；混在一起平均会冲淡收益。

### 3.3 最大陷阱：APC 干扰 → 负载怎么设计

见 1.3：纯版 vllm_musa 自带 APC，**工作集 < GPU 容量时纯版也命中**，导致「纯 vs LMCache」几乎没差，不是 LMCache 没用而是没暴露差异。

**设计目标**：让被复用的前缀工作集**大到 GPU 装不下、CPU 装得下**。
- 工作集 < 容量：APC 全包，LMCache external hits ≈ 0（这是预期，不是 bug）；
- 工作集 > 容量：APC 被迫淘汰，重访 APC miss（重算），**LMCache 从 CPU 捞回**——这才是 LMCache 唯一赢的地方。

**模型选择的额外考虑**：大模型（如 14B）权重吃更多 HBM → KV 容量更小（更好造超容量场景）+ prefill 更贵（LMCache 省下的绝对 TTFT 更大、信号更强）。代价是要重启服务 + 重验 LMCache 通路 + KV 搬运字节也增大（短前缀可能出现「搬运比重算还慢」的反转点，需实测）。**当前先用 8B 走完整流程，再测 14B。**

---

## 4. Task 计划（4 步）

| # | 任务 | 状态 | 说明 |
|---|---|---|---|
| **Step0** | 量出 GPU KV 容量标尺 | ✅ 完成 | 216,704 tokens（见 §2） |
| **Step1** | LMCache 通路冒烟验证（工作集 < 容量） | ✅ 完成 | 见 §2.5：TTFT median 119ms，APC 81.7% 命中，external hits=0 实证 §1.3 |
| **Step2** | 构造超容量差异化 benchmark + 三臂对照 | ✅ 分离复测完成 | 先用随机 shuffle 首轮验证 external hit，再用 cold/warm 分离复测证明收益：warm 阶段 LMCache external hit=245,760 tokens，Median TTFT 262.70ms vs pure 728.62ms |
| **埋点** | 给 bench 流程加 cache 命中率采集 | ✅ 完成 | `/metrics` 前后做差已验证（external=LMCache，internal=APC，见 §1.4 / §2.5） |

依赖：Step1 依赖 Step0；Step2 依赖 Step0 / Step1 / 埋点。

### 当前卡点 / 注意事项（2026-06-17）

- **benchmark 实际在 146 容器内跑**：容器里有 vLLM venv 和 transformers，路径为 `/root/.virtualenvs/sglang-0.5.6/bin/python3`；
- 146 容器内 benchmark 脚本路径是 `/data/my_vllm_test/vllm_020/bench_serving.py`，不是 165 本机的 `/data/my_vllm_test/vllm_musa_proj/bench_serving.py`；
- 165 本机仍保留原始 benchmark/auto_bench 脚本，可作为参考，但直接打 146 容器本机端口能减少网络变量；
- 40GB 是当前验证可启动的 LMCache CPU cache；64GB/80GB 失败，继续加大前要先解决 `musaHostAlloc failed: 205`；
- `bench_serving.py` 当前会随机打乱 `generated-shared-prefix` 请求顺序，适合做首轮 external hit 验证；严谨验证性能收益时，以 §2.8 的冷/热分离脚本为准。

### bench_serving.py 关键参数（共享前缀负载）

`bench_serving.py` 的共享前缀模式由 `--dataset-name generated-shared-prefix` 启用。它的核心概念是：

```text
一个 group = 一个 shared system prompt + K 个不同 question
```

所以：

```text
总请求数 = gsp_num_groups × gsp_prompts_per_group
不同 shared-prefix 工作集 = gsp_num_groups × gsp_system_prompt_len
```

关键参数：

| 参数 | 含义 | 为什么重要 |
|---|---|---|
| `--dataset-name generated-shared-prefix` | 使用共享前缀合成数据集 | 不用这个模式，普通 random 请求几乎没有可复用前缀，测不出 APC/LMCache |
| `--gsp-num-groups N` | 生成 N 个不同 shared prefix | 决定不同前缀数量；N 越大，越容易超过 GPU APC 容量 |
| `--gsp-prompts-per-group K` | 每个 shared prefix 生成 K 个请求 | K=1 只有 cold，没有复用；K≥2 才有 warm revisit |
| `--gsp-system-prompt-len L` | 每个 shared prefix 的目标 token 长度 | 和 N 相乘决定工作集大小；L 越大，prefill 越贵，LMCache 收益越可能明显 |
| `--gsp-question-len Q` | 每个请求独有 question 的目标 token 长度 | 模拟同一个长系统提示下不同用户问题；避免完全重复请求 |
| `--gsp-output-len O` | 每个请求最大输出 token 数 | 输出越长，E2E 越受 decode 影响；测 LMCache 时通常要短输出，突出 TTFT |
| `--num-prompts` | 实际参与 benchmark 的请求数 | 应与 `N×K` 对齐；否则可能截断或采样请求 |
| `--max-concurrency` | 客户端最大并发请求数 | 控制压测强度；过高会引入排队噪声，过低耗时长 |
| `--request-rate` | 请求到达速率 | `inf` 表示尽快发出；适合施压，但不适合严格冷/热分离 |
| `--seed` | 随机种子 | 保证生成数据可复现；做 pure/LMCache 对照时必须一致 |

注意：`bench_serving.py` 当前会在生成后 `random.shuffle(input_requests)`。这会打乱 cold/warm 顺序，所以它适合做“是否有 external hit”的首轮验证，不适合直接证明性能收益。要证明收益，应使用 §2.8 的冷/热分离脚本，或修改 `bench_serving.py` 让请求顺序可控。

### lmcache_cold_warm_revisit_bench.py 关键参数

§2.8 新增脚本的参数含义如下：

| 参数 | 含义 | 与 `bench_serving.py` 的对应关系 |
|---|---|---|
| `--groups` | 不同 shared prefix 数量 | 类似 `--gsp-num-groups` |
| `--prefix-len` | shared prefix 目标 token 长度 | 类似 `--gsp-system-prompt-len` |
| `--question-len` | cold/warm 请求各自的独有后缀长度 | 类似 `--gsp-question-len` |
| `--output-len` | 最大输出 token 数 | 类似 `--gsp-output-len` |
| `--concurrency` | 每批同时发送的请求数 | 类似 `--max-concurrency`，但脚本按 cold/warm phase 分批执行 |
| `--seed` | 控制 prefix/question 生成 | pure 和 LMCache 两边必须一致 |
| `--gpu-kv-tokens` | GPU APC 容量，用于估算工作集是否超容量 | 默认 216,704，来自 vLLM 启动日志 |
| `--served-model-name` | OpenAI 请求中的 model 字段 | 19001 用 `qwen3-8b-lmcache`，19000 用 `qwen3-8b-pure-fixed` |
| `--port` | 服务端端口 | 19001 = LMCache，19000 = pure |

这个脚本固定执行两段：

```text
cold phase: group 0..N-1 各访问一次
warm phase: group 0..N-1 再各访问一次
```

它不会随机打乱请求，因此 warm 阶段的 TTFT 和 metrics 可以单独解释。

backend 用 `vllm`（走 `/v1/completions`）。在 146 容器内跑时，建议：

- `--host 127.0.0.1`
- `--port 19001` 测 LMCache，`--port 19000` 测 pure baseline
- `--model /data/SQT-v1.0.5-test/models/qwen3-8b`
- `--served-model-name qwen3-8b-lmcache` 或 `qwen3-8b-pure-fixed`
- `--tokenizer /data/SQT-v1.0.5-test/models/qwen3-8b`

这样 tokenizer/config 都走本地路径，避免 huggingface_hub 把 served model name 当成远程 repo 去联网查询。

### 一键复现（在 146 容器内执行）

两臂必须用**同一个 seed**，warm 阶段才可比。先跑 LMCache（19001），再跑 pure baseline（19000）：

```bash
# 进 146 容器
sshpass -p 'Admin@9000' ssh root@10.10.142.146
docker exec -it vllm020_lmcache_test bash

# 用 vLLM 那个带 transformers 的 venv（容器默认 python3 没有 transformers）
PY=/root/.virtualenvs/sglang-0.5.6/bin/python3
SCRIPT=/data/my_vllm_test/vllm_020/lmcache_cold_warm_revisit_bench.py
OUT=/data/my_vllm_test/vllm_020/bench_results_lmcache_step2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 免得联网查 tokenizer

# ① LMCache 臂（19001）
$PY $SCRIPT --host 127.0.0.1 --port 19001 \
  --served-model-name qwen3-8b-lmcache \
  --tokenizer /data/SQT-v1.0.5-test/models/qwen3-8b \
  --groups 80 --prefix-len 3072 --question-len 128 --output-len 32 \
  --concurrency 4 --seed 2026061702 --gpu-kv-tokens 216704 \
  --output $OUT/lmcache40_cold_warm_$(date +%Y%m%d)_19001.json

# ② pure 臂（19000）—— 换 port + served-model-name，其余参数（尤其 seed）保持一致
$PY $SCRIPT --host 127.0.0.1 --port 19000 \
  --served-model-name qwen3-8b-pure-fixed \
  --tokenizer /data/SQT-v1.0.5-test/models/qwen3-8b \
  --groups 80 --prefix-len 3072 --question-len 128 --output-len 32 \
  --concurrency 4 --seed 2026061702 --gpu-kv-tokens 216704 \
  --output $OUT/pure_cold_warm_$(date +%Y%m%d)_19000.json
```

跑完看结果 JSON 里的 `summary.warm`（warm 阶段 TTFT）和 `metrics.warm_delta`（warm 阶段 APC/LMCache 命中）即可对比。判定 LMCache 生效的两个标志：
1. LMCache 臂 `warm_delta` 里 `external_prefix_cache_hits_total` ≈ `groups × prefix_len`（本轮 245,760）；
2. LMCache 臂 `summary.warm.median_ttft_ms` 明显低于 pure 臂。

> 换模型/换容量复测时：先重启服务并从启动日志重读 `GPU KV cache size`，把 `--gpu-kv-tokens` 改成新值，并确保 `groups × prefix_len` 仍 > 新的 GPU 容量、且 < LMCache CPU cache 容量。

---

## 5. 多卡 / 多机扩展方向（当前是单卡，下一步可以往哪走）

### 5.0 现状先说清楚

当前**还是单卡**：从启动日志看两个服务都是 `tensor_parallel_size=1`，只是分别放在 GPU0（19000 纯版）和 GPU1（19001 LMCache）——是**两个独立的单卡服务**，不是多卡并行。

硬件资源（2026-06-17 核对）：146 有 **8× MTT S5000，每张 80GB，当前全空闲**。所以下面三种多卡方案硬件上都够。

> 注意「多卡」对 LMCache 有**三种完全不同的含义**，测的是不同东西，别混为一谈。

### 5.1 方案① TP 张量并行（一个模型拆到多张卡）

| 项 | 说明 |
|---|---|
| 是什么 | 把单个模型按张量并行拆到 N 张卡（tp2/4/8）。8 张卡最多 tp8 |
| 测什么 | LMCache 在 TP 下能不能正常工作；以及给**大模型**（14B/32B）省 prefill 的收益（大模型 prefill 更贵，LMCache 绝对收益更大） |
| 要改什么 | 启动加 `--tensor-parallel-size N`（benchmark 封装见 `reademe.md` §7.7 的 `--tp`）；每个 TP rank 各自把自己那份 KV **分片** offload 到 LMCache CPU |
| 当前差什么 | 没起过 TP 服务，需重启验证；TP 下每 rank 的 LMCache CPU 分配是独立的，注意 §2.6 的 `musaHostAlloc` 上限对每 rank 仍可能触发 |
| 优先级 | **最容易做，和「测完 8B 测 14B/32B」天然组合**，建议作为下一步首选 |

### 5.2 方案② 跨实例 / 跨卡 KV 共享（LMCache 的杀手锏，APC 永远做不到）

| 项 | 说明 |
|---|---|
| 是什么 | 多个**独立 vLLM 实例**（如 GPU0、GPU1 各一个）**共享同一份 KV**：GPU0 算过的前缀，GPU1 直接复用，不重算 |
| 测什么 | LMCache **独有**、APC 根本做不到的能力——APC 是每实例自己的 GPU 显存，互相看不见 |
| 要改什么 | 当前 `LocalCPUBackend` 是**每实例独占的 CPU 内存，不共享**（日志确认 `enable_p2p=False`、`enable_controller=False`、`remote_url=None`）。要换**远程/共享后端**（Redis / mooncake 等）或开 LMCache 的 P2P / controller 模式 |
| 当前差什么 | 缺远程后端的搭建与验证（要起一个 remote KV server）；这是额外基础设施 |
| 优先级 | **最能体现「LMCache 比 APC 强在哪」**，但工程量最大，建议放在 TP 之后 |

### 5.3 方案③ PD 分离（prefill / decode 拆到不同卡，LMCache 当中间 KV 通道）

| 项 | 说明 |
|---|---|
| 是什么 | prefill 在一组卡、decode 在另一组卡，LMCache 负责把 prefill 算好的 KV 搬给 decode 节点 |
| 测什么 | LMCache 作为 prefill→decode 的 KV 传输层（disaggregated serving 场景） |
| 要改什么 | `reademe.md` §7.7.2 有 `--pd` 支持；LMCache 配置里有 `enable_pd / pd_role / pd_buffer_*` 等字段（当前 `enable_pd=False`）；codex 在 2026-06-16 碰过 PD 分离 |
| 当前差什么 | 需要 prefill/decode 双服务 + proxy 编排（`toy_proxy_server.py`），是更专门的拓扑 |
| 优先级 | 最窄的专门场景，作为后续探索 |

### 5.4 建议扩展顺序

与 `vllm_020/lmcache_on_vllm_musa_020_feasibility.md` §16 一致：

1. 先保持当前单卡 Qwen3-8B 基线不变，复跑 §2.8 冷/热脚本确认环境没漂；
2. 再扩工作集 / 并发，观察 `external_prefix_cache_hits_total` 与 TTFT 的变化趋势；
3. 再上 **Qwen3-14B/32B + 方案① TP**（信号更强，工程量小）；
4. 最后再研究 **方案② remote backend / 方案③ PD / 多机**，不要把这些和本地 CPU cache 主路径混在同一轮排查。
