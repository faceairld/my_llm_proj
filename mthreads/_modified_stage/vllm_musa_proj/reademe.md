# vLLM 多模型并行服务与压测

## 测试环境（2026-06-03 核对）

| 项 | 值 |
| --- | --- |
| 服务器 | `192.168.4.127`（k8s pod,SSH `ssh -p 30001 root@192.168.4.127`） |
| GPU | MTT S5000 ×8,单卡 80GB(81920MiB) |
| 驱动 | Driver 3.3.5-server;`mthreads-gmi` 2.3.2 |
| Python | 3.10.12 |
| vllm | `0.20.1.dev0+g88d34c640.d20260519`（**仅 V1 引擎**,见下) |
| vllm_musa | 0.1.1(摩尔线程 MUSA 适配层) |
| torch / torch_musa | 2.9.0 / 2.9.0 |
| 虚拟环境 | `/root/.virtualenvs/sglang-0.5.6`(`PY=/root/.virtualenvs/sglang-0.5.6/bin/python`) |
| sglang(仅参考数据来源) | 0.5.6.post2 |
| bench 目录 | `/mnt/seed17/001688/models/Qwen/bench`(底层 mount = `/mnt/si0003568lza/default/...`) |

> ⚠️ **V0 引擎已被删除,本版本只有 V1。** 源码实锤(2026-06-03 核对):
> - `vllm/worker/` 目录**整个不存在**(V0 的 model_runner/worker 执行层已物理删除);
> - `vllm/engine/llm_engine.py` 仅剩 7 行,`LLMEngine = vllm.v1.engine.llm_engine.LLMEngine` 的**别名壳**(只为兼容老 import),`async_llm_engine.py` 同理指向 V1 的 `AsyncLLM`;
> - 环境变量 `VLLM_USE_V1` 在整个包里**搜不到**,已移除——设了也无效。
> - **后果**:`run_list.py` 里 `env["VLLM_USE_V1"]="0"` 是从老版本继承的**死代码**(不管设什么,启动日志都是 `Initializing a V1 LLM engine`)。所有"V0/V1 切换"相关的旧理解在本环境一律作废,只按 V1 行为分析。
> - 因此第四章列的默认优化(prefix caching / chunked prefill 等)都按 **V1 默认值**理解。

---

在 8 张 GPU 上各启动一个 vLLM 服务（一卡一模型），端口 `8000`–`8007`。编排脚本为 `run_list.py`，压测脚本为 `test_list.py`。


| GPU | 端口   | 模型（示例路径）                 |
| --- | ---- | ------------------------ |
| 0   | 8000 | Qwen3-8B                 |
| 1   | 8001 | Qwen3-8B-FP8             |
| 2   | 8002 | Qwen3-VL-2B-Instruct     |
| 3   | 8003 | Qwen3-VL-2B-Instruct-FP8 |
| 4   | 8004 | Qwen3-VL-4B-Instruct     |
| 5   | 8005 | Qwen3-VL-4B-Instruct-FP8 |
| 6   | 8006 | Qwen3-VL-8B-Instruct     |
| 7   | 8007 | Qwen3-VL-8B-Instruct-FP8 |


模型路径与端口在 `run_list.py` 的 `MODELS` / `BASE_PORT` 中配置。

---

## 启动 8 个服务

```bash
cd /home
nohup python3 -u run_list.py > /home/run_list.nohup.log 2>&1 &
echo $! > /home/run_list.nohup.pid
```

- `-u`：关闭 Python 输出缓冲，`tail -f` 可实时看到进度。
- `run_list.nohup.log`：汇总日志（`[launch]` / `[ready]` / `[wait]` / `[all-ready]`）。
- 每个 vLLM 实例的详细日志：`/home/run_logs/<idx>_<name>.log`
- 子进程 PID 记录：`/home/run_logs/pids.txt`

### 查看就绪状态

```bash
# 等待全部就绪，日志中出现 [all-ready] 即可
tail -f /home/run_list.nohup.log

# 或随时查询各端口
python3 /home/run_list.py status
```

前台直接运行（不经过 nohup）：

```bash
python3 /home/run_list.py              # 启动并轮询直到就绪
python3 /home/run_list.py --no-wait    # 仅启动，不等待就绪
```

---

## 运行压测

需先确认 8 个服务均已就绪，再执行：

```bash
cd /home
nohup python3 -u test_list.py > /home/test_list.nohup.log 2>&1 &
echo $! > /home/test_list.nohup.pid
tail -f /home/test_list.nohup.log
```

结果默认写入 `/home/bench_results/`（单模型 CSV、汇总 CSV、各模型 `bench_logs/`）。

---

## 停止服务

```bash
python3 /home/run_list.py stop
```

上述命令会按 `pids.txt` 向各 vLLM 进程组发送 `SIGTERM`，是**正确**的关停方式。

若仅需结束 nohup 父进程（**不会**关掉 8 个 vLLM）：

```bash
kill "$(cat /home/run_list.nohup.pid)" 2>/dev/null || true
```

> **说明**：`run_list.py` 启动子进程时使用了 `start_new_session=True`，因此 kill 掉 nohup 父进程后，8 个 vLLM 仍会保留在后台；只有 `python3 /home/run_list.py stop` 才会真正关闭它们。

---

## 相关文件


| 文件                 | 说明                   |
| ------------------ | -------------------- |
| `run_list.py`      | 并行启动 / 状态查询 / 停止     |
| `test_list.py`     | 对 8 个端口并行跑 benchmark |
| `run.sh`           | 单模型手动启动示例            |
| `bench_serving.py` | 压测核心逻辑               |
| `download.py`      | 从 ModelScope 批量下载模型  |


<br>

---
---

# 附录:项目结构说明

> 以下为对本项目文件、调用层次与测试类型的补充说明,与上文使用指南相互独立。

## 一、各文件作用

### 核心编排脚本

- **`run_list.py`** — 服务启动/管理器
  - 在 8 张 GPU 上各拉起一个 vLLM 服务,一卡一模型(Qwen3-8B 系列 + Qwen3-VL 多模态系列,含 FP8 量化版),端口 `8000–8007`。
  - 关键环境设置:`VLLM_USE_V1=0`(强制走 V0 引擎)、按索引绑定 `CUDA_VISIBLE_DEVICES` / `MUSA_VISIBLE_DEVICES`。
  - 子进程用 `start_new_session=True` 独立成会话组,因此杀 nohup 父进程不会连带杀掉 8 个服务,必须用 `stop` 子命令按进程组发 SIGTERM。
  - 子命令:无参=启动并轮询就绪 / `--no-wait`=只启动 / `status`=查端口状态 / `stop`=关停全部。
  - 启动前会把 `/home` 下散落的 MUSA `core_*.mudmp` 崩溃转储扫进 `/home/musa_dumps`,并把子进程 cwd 设到该目录(防止 dump 污染 /home)。

- **`test_list.py`** — 并行压测调度器
  - 从 `run_list.py` 导入 `MODELS` / `BASE_PORT` 保持配置同步。
  - 用 `ThreadPoolExecutor` 对 8 个端口**并行**跑 benchmark,每模型 18 个 case(不同 input/output 长度组合 2k~4k,扫不同 request_rate)。
  - 每模型输出独立 CSV + 一份汇总 CSV,日志在 `bench_logs/`。压测前先检查全部服务可达,否则报错退出。

### 压测引擎

- **`bench_serving.py`** — 实际打流器,改编自 vLLM/SGLang 官方 `benchmark_serving.py`。异步发请求,统计 TTFT、TPOT、ITL、吞吐、并发等指标。`test_list.py` 和 `test.sh` 都调用它。

### 辅助/示例脚本

- **`run.sh`** — 单模型手动启动示例(`vllm serve`,TP=1,带 cudagraph 捕获尺寸配置)。
- **`test.sh`** — 单模型串行压测的 bash 版(对应 `test_list.py` 的单机版,case 列表相同),跑完输出一份 CSV。

### 数据

- **`ShareGPT.json`** (≈672 MB) — 压测用真实对话数据集。注意脚本里写死的路径是 `/home/ShareGPT_V3_unfiltered_cleaned_split.json`,与此处文件名不同,需重命名或软链。

---

## 二、调用层次结构

```
vllm_musa_proj/
│
├── 【文档】
│   └── reademe.md ──────────────── 使用说明(启动/状态/压测/停止全流程)
│
├── 【服务层】启动 8 个 vLLM 服务,一卡一模型
│   ├── run_list.py ★ 编排器 ────── 启动/status/stop,8 GPU × 端口 8000~8007
│   │       │                       (VLLM_USE_V1=0, 绑 MUSA/CUDA_VISIBLE_DEVICES)
│   │       └── 调用 vllm serve
│   │
│   └── run.sh ──────────────────── 单模型手动启动示例(run_list.py 的简化版)
│
├── 【压测层】对已起的服务打流测性能
│   ├── test_list.py ★ 调度器 ───── 8 端口并行压测,每模型 18 个 case
│   │       │  └─ import MODELS/BASE_PORT from run_list.py(配置同步)
│   │       │  └─ ThreadPoolExecutor 并行
│   │       └── 每个 case 调用 ↓
│   │
│   ├── test.sh ────────────────── 单模型串行压测(test_list.py 的 bash 版)
│   │       └── 每个 case 调用 ↓
│   │
│   └── bench_serving.py ◆ 引擎 ─── 实际异步打流 + 指标统计
│                                   (TTFT/TPOT/ITL/吞吐/并发)
│                                   改编自 vLLM/SGLang 官方脚本
│
└── 【数据】
    └── ShareGPT.json (672MB) ───── 压测用对话数据集
            ⚠ 脚本期望名: /home/ShareGPT_V3_unfiltered_cleaned_split.json
```

调用关系一句话概括:

```
                 配置同步(import)
   run_list.py ──────────────────► test_list.py
        │                                │
        ▼ spawn                          ▼ 每 case subprocess
   8× vllm serve ◄──── HTTP 打流 ──── bench_serving.py
   (8000~8007)                            ▲
                                          │ 读取
                                    ShareGPT.json
```

两条独立路径:

- **批量路径(主)**:`run_list.py`(起服务)→ `test_list.py`(并行压测)→ `bench_serving.py`(引擎)
- **单机路径(示例)**:`run.sh`(起一个)→ `test.sh`(串行压测)→ `bench_serving.py`(同一引擎)

标记:★ = 入口脚本,◆ = 底层引擎,⚠ = 路径/命名需注意。

---

## 三、实际启动/运行的服务类型

**每个服务都是单卡单模型**——整体是「8 个独立单卡服务同时跑」,不是多卡协同。

证据(三个脚本一致):

- `run_list.py` `TP = 1`、`--pipeline-parallel-size 1`,且 `CUDA_VISIBLE_DEVICES=str(idx)` / `MUSA_VISIBLE_DEVICES=str(idx)` —— 每个 vLLM 进程只能看到 1 张卡。
- `run.sh` `TP=1` + `--pipeline-parallel-size 1`。
- `test.sh` `export TP=1`。

两个维度要分清:

| 维度 | 本项目 | 未覆盖 |
| --- | --- | --- |
| 单服务并行度 | 单卡(TP=1, PP=1) | 多卡张量并行 TP>1 / 流水线并行 PP>1 |
| 整机利用方式 | 8 卡各跑一个独立模型,互不通信 | 一个大模型跨多卡切分 |

结论:它测的是**「满载场景下每张卡单独服务一个模型时的吞吐/延迟表现」**(8 路并行施压,看整机能否扛住、各模型 TTFT/TPOT 如何),而**不测卡间通信**(TP/PP 的 all-reduce / broadcast 路径)。

如需覆盖多卡 TP 场景,需改三处:调大 `TP`、给每个服务分配一组卡(如 `CUDA_VISIBLE_DEVICES=0,1`)、并相应减少并发服务数。

---

## 四、优化方案:已开启 / 未开启

> ⚠ **更正(2026-06-03)**:本章早期版本基于 5/28 的 V0 引擎测试写成,误把 prefix caching / chunked prefill 标为"未开"。**实际本次跑的是新版 vllm_musa 0.20 的 V1 引擎**,经 run_logs 的 config 行核实,这两项 V1 **默认就开着**。下表已按 V1 实际状态更正。

启动命令为 `run_list.py` 的 `build_cmd`(与 `run.sh` 一致)。除显式配的几项外,V1 引擎还默认启用了一批优化。

### 默认就有(vLLM 核心 + V1 默认,非显式配置)

| 优化 | 本次状态 | 证据 / 说明 |
| --- | --- | --- |
| **PagedAttention** | ✅ 启用 | 核心实现,天然启用 |
| **Continuous batching** | ✅ 启用 | 调度器默认行为 |
| **FLASH_ATTN v3** | ✅ 启用 | MUSA 平台默认 attention 后端(run_logs: "FLASH_ATTN with version 3"),= sglang `fa3` |
| **Prefix Caching** | ✅ **默认开** | run_logs: `enable_prefix_caching=True`(V1 默认。随机数据集下复用率低,但确实开着) |
| **Chunked Prefill** | ✅ **默认开** | run_logs: `enable_chunked_prefill=True`(V1 默认) |

### ✅ 显式配置 / 已调参

| 优化 | 参数 | 说明 |
| --- | --- | --- |
| CUDA / MUSA Graph | `--compilation-config '{"cudagraph_capture_sizes":[1,2,...,256]}'` | 未加 `--enforce-eager`,graph 生效(run_logs: `enforce_eager=False`),自定义 23 个 batch size 档位捕获 |
| KV cache block size | `--block-size 64` | 默认通常 16,调大以减少元数据开销 |
| 显存利用率 | `--gpu-memory-utilization 0.8` | 留 20% 余量(sglang 用 0.9) |
| 批 token 预算 | `--max-num-batched-tokens`(本次默认 =8192) | 见下方专门说明 |

### 🔶 Attention 计算后端(`--attention-backend`)

attention 的具体 kernel 实现可选,这是一个**独立的性能维度**(此前漏列,补上)。vllm_musa 在 MUSA 平台支持:

| backend | 说明 |
| --- | --- |
| **FLASH_ATTN**(v3) | MUSA 平台**默认**,本次实际使用(run_logs 确认 "FLASH_ATTN with version 3")。等价于 sglang 的 `fa3` |
| TRITON_ATTN | Triton 实现,备选 |
| TORCH_SDPA | PyTorch 原生 SDPA,兜底 |

> 本次 vllm_musa 用 FLASH_ATTN v3,和 sglang 的 `--attention-backend fa3` **是同一个东西**——attention 后端这块两边一致,不是性能差距来源。注意:MUSA 上 FLASH_ATTN 暂不支持 FP8(日志有 "Cannot use FLASH_ATTN with FP8 on MUSA")。

### ❌ 可显式开启但**未开启**(参数名来自 vllm 0.20 源码 `arg_utils.py`,已核对)

| 优化 | 参数 | 状态 / 备注 |
| --- | --- | --- |
| **★ 投机解码(Speculative Decoding)** | `--speculative-config`(EAGLE3 等) | ✗ —— **本次最大遗漏项**。sglang 的 8B 开了 EAGLE3(吞吐 ×2~3),vllm_musa 没开,是 6.5 节过载的决定性根因。vllm_musa **已适配**(有 `eagle_full_loop_runner.py` / `tree_attn.py`),能开,需 draft model |
| **★ 调度并发上限** | `--max-num-seqs` | ✗ —— 没设用默认,导致并发卡 ~60;sglang 用 256。见 6.5。**最该补的低成本项** |
| FP8 KV Cache | `--kv-cache-dtype fp8` | ✗ —— 且 **MUSA FLASH_ATTN 不支持 FP8 KV**(日志明说),基本不能用 |
| torch.compile 编译级别 | `--compilation-config` 的 `level` | ✗ —— 只设了 graph 捕获,未设编译优化等级 |
| Tensor / Pipeline 并行 | `--tensor-parallel-size>1` 等 | ✗ —— 本期 TP=1, PP=1 |
| CPU offload | `--cpu-offload-gb` / `--offload-*` | ✗ |
| 启动期量化 | `--quantization` | ✗ —— FP8 来自 checkpoint 自身,非启动期量化 |

> **vllm_musa 专属、但不适用本期 Qwen dense 模型**:`FLASHMLA` / `TRITON_MLA`(给 MLA 架构如 Deepseek)、`TURBOQUANT`、`DeepGEMM`(特定量化/MoE)。这些 vllm_musa 注册了,但 Qwen3 dense 用不上。

### 📌 `--max-num-batched-tokens` 是什么(本次 =8192)

**限制"一次前向(一个 step)里所有请求加起来最多处理多少 token"**,是 token 层面的 batch 预算上限。它和 `--max-num-seqs`(请求条数上限)并存,**谁先到顶受谁限**。

- **在 prefill 阶段最关键**:每个请求 prefill 要一次处理它的全部输入 token。例:`8192 / 4096(4k输入)= 一个 batch 只能塞 2 个请求的 prefill` → 新请求进不来排队 → TTFT 暴涨。这是长输入档过载的直接机制之一(2k 输入能塞 4 个,所以 2k 档表现明显更好)。
- **权衡旋钮**:调大 → batch 塞更多 token,吞吐↑、prefill 并发↑,但显存压力↑;调小 → 单 step 算得快,TTFT↓,但吞吐↓。
- **与 chunked prefill 的关系**(本次开着):长 prompt 被切成 ≤ 此值的块分步处理,所以它还兼任"每块多大"——值越大块越大、吞吐越高但单步延迟越高。等价于 sglang 的 `--chunked-prefill-size 8192`(本次两边一致)。
- **对本期**:并发瓶颈主因是 `--max-num-seqs` 没调(默认小),不是这个值。重测提并发主要调 `--max-num-seqs`。

### ⚠ 易误判点

1. **FP8 模型 ≠ FP8 KV cache**:模型名带 `-FP8` 只是权重格式;未加 `--kv-cache-dtype fp8`,KV cache 仍按默认精度存(且 MUSA FLASH_ATTN 暂不支持 FP8 KV)。
2. **attention 后端 ≠ attention 优化开关**:`--attention-backend` 选的是"用哪个 kernel 实现"(FLASH_ATTN/Triton/SDPA),和 prefix caching 这类"要不要开某优化"是两类配置,容易混。
3. **投机解码和并发上限是这个项目里最关键的两个变量**(决定 vllm_musa vs sglang 的可比性),不是可有可无的次要开关——详见第六章 6.5。

> 小结:V1 引擎默认已启用 PagedAttention + continuous batching + FLASH_ATTN v3 + **prefix caching + chunked prefill**,外加显式配的 CUDA Graph。真正没开、且对本期 Qwen 模型有意义的是 **投机解码 + 并发上限(`--max-num-seqs`)** 两项,正是与 sglang 拉开差距的主因。

### 🧩 投机解码(Speculative Decoding)是什么

**用一个小模型先"猜"出后面几个 token,再让大模型一次性批量"验证",对的就直接用——从而一次前向吐出多个 token,加速生成。**

- **为什么能加速**:大模型生成是逐 token 串行的,每生成 1 个 token 都要把几十 GB 权重从显存搬一遍(decode 瓶颈是显存带宽,不是算力)。投机解码让大模型**一次前向并行验证多个候选 token**——权重只搬一次,产出从 1 个变多个。
- **流程(草稿 + 验证)**:① draft 小模型快速猜接下来 N 个 token ② 大模型一次并行验证这 N 个 ③ 从头比对,认可的保留(全对→一次吐 N 个;第 k 个错→保留前 k-1 个重来)。
- **不掉精度**:验证用的是大模型本身,只接受大模型认可的 token,**最终输出和大模型逐个生成完全一致**,纯加速。草稿猜错只是浪费一次,不产生错误结果。
- **EAGLE3**:目前最先进的投机算法之一。不用独立小模型,而是复用大模型中间特征训个**极轻量草稿头**,接受率高、开销小,加速比常达 2~3 倍。`Qwen3-8B_eagle3` 就是给 Qwen3-8B 训的 EAGLE3 草稿模型。
- **本项目关联**:sglang 的 8B 开了 EAGLE3(rate 能压到 3.6 不崩),vllm_musa 没开(同 rate 全过载)——这是 6.5 节"过载根因"的决定性因素。需要 draft model,本批只有 8B 有。
- **关键参数**:`--speculative-config '{"method":"eagle3","model":"<draft路径>","num_speculative_tokens":N}'`,N=一次猜几个 token(草稿步数,常用 3)。
- **比喻**:普通解码=秘书一个字一个字记;投机解码=实习生先猜好一整句,你扫一眼"前半句对、后半句不对",对的直接用,一次顶好几个字。猜得越准越快。

---

## 五、本期 tp1 自动化测试:每个模型要改的参数

> 背景:把 SGLang v0.5.6-post2 release 文档里标 **tp1(单卡)** 的纯文本模型,在 vllm_musa 上重测一遍。共 **8 个模型一批跑**(填满 8 卡)。多模态 Qwen2.5-VL-7B 暂不纳入(需图像数据集)。

> 给不熟悉压测的人:一句话——**模拟很多用户同时向大模型服务发请求,测它扛得住多大压力、响应多快、吞吐多高。** 分两步:① 启动服务(加载模型到 GPU,起 HTTP 服务)② 压测(按设定速率发请求,统计指标)。下面 5.1~5.5 先讲每个参数是什么,5.6 起再说本期哪些要改、哪些不改。

### 5.1 启动服务的参数(run_list.py)—— 模型怎么部署在卡上

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `--tensor-parallel-size`(TP) | 1 | **用几张卡切一个模型**。TP=1=单卡跑整个模型;TP=4=把权重切到 4 卡一起算(大模型单卡放不下才用)。本期全是 tp1 |
| `--pipeline-parallel-size`(PP) | 1 | 流水线并行,按"层"切到多卡。1=不切 |
| `--gpu-memory-utilization` | 0.8 | **允许用多少显存**(80%,留 20% 余量)。这块显存大部分拿来当 KV cache |
| `--block-size` | 64 | **KV cache 的分页块大小**。KV cache 按块管理,64=每块存 64 个 token 的 K/V,类似内存分页 |
| `cudagraph_capture_sizes` | [1,2,…256] | **CUDA Graph 预捕获的 batch 尺寸**。提前把常见 batch 的计算图录下来,运行时直接重放,省调度开销→更快 |
| `--served-model-name` | 模型名 | 服务对外的模型名,请求时要对上 |
| `--port` | 8000~8007 | 每个模型一个端口 |

**关键概念 KV cache**:大模型生成时,前面每个 token 算出的 K/V 向量要存起来给后面用(否则每生成一字要重算全部历史)。这个存储就是 KV cache,**它占多少显存直接决定能同时处理多少请求**。`gpu-memory-utilization` 和 `block-size` 都在管它。

### 5.2 压测参数(test_list.py / bench_serving.py)—— 怎么发请求

**控制"请求长什么样":**

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `--dataset-name` | random | 用随机生成的请求(非真实对话),从语料随机采 token 拼成指定长度 |
| `--dataset-path` | ShareGPT.json | 语料采样源(只提供真实文本的 token 分布) |
| `--random-input-len` | 2048 等 | **每个请求的输入长度**(prompt 多少 token),模型要"读"多少 |
| `--random-output-len` | 1024 等 | **每个请求生成多少 token**,模型要"写"多少 |
| `--random-range-ratio` | 1.0 | 长度浮动比例。1.0=每条严格等于设定长度,不浮动(保证可控可比) |
| `--apply-chat-template` | — | 按模型对话模板格式化 prompt(加 `<\|im_start\|>` 等标记) |

> 为什么扫不同输入输出组合:**input 长**→影响 prefill(读 prompt)→主要拖慢 **TTFT**;**output 长**→影响 decode(逐字生成)→主要影响 **TPOT/吞吐**。不同组合覆盖不同负载形态。

**控制"发多少、多快":**

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `--num-prompts` | 100 | **这一轮总共发多少个请求**(整数,是"个数") |
| `--request-rate`(QPS) | 0.4~3.6 不等 | **★平均发压速率**(见 5.3,可小数) |
| `--burstiness` | 102 | 请求到达的**均匀程度**(见 5.4) |
| `--warmup-requests` | 500 | **正式测前先发的预热请求**(不计入统计),让 CUDA Graph/cache 进入稳态,避免冷启动污染数据 |

### 5.3 ★ request-rate(QPS)—— 最重要的参数

**request-rate 是平均速率,不是"每秒整好几个请求"**,它决定"两个请求之间平均隔多久发":

```
平均请求间隔 = 1 / request-rate  秒
```

| request-rate | 平均间隔 | 含义 |
| --- | --- | --- |
| 3.0 | 0.333 s | 平均每隔 1/3 秒发一个 |
| 3.4 | 0.294 s | 平均每隔 0.294 秒发一个 |
| 0.4 | 2.5 s | 平均每隔 2.5 秒才发一个 |

**所以 rate 有小数完全合理**——它是速率/频率(像"水流 3.4 升/分钟"),不是请求个数。请求个数是 `--num-prompts`(整数)。两者别混:**rate=速率(可小数),num-prompts=总数(整数)**。

**为什么要"从低到高扫一组 rate"(mentor 说的"慢慢增大")**:单个 rate 没意义,要画**性能 vs 负载曲线**找拐点——

```
rate 低   → 服务轻松,延迟低,但吞吐没拉满(卡没吃饱)
rate 升   → 吞吐上升,延迟略增(健康区)
rate 到拐点 → 吞吐到顶
rate 再升 → 请求排队,TTFT 暴涨,吞吐不再涨(过载)
```

那个拐点就是**这张卡这个模型的服务能力上限**。这也是为什么每个模型 rate 范围不同(见 5.7):拐点位置不同,慢模型(14B,拐点 ~1)用快模型的高 rate 会全程过载、白测。

### 5.4 ★ burstiness —— 实际间隔围绕平均值的波动程度

rate 只规定了**平均**间隔,但真实间隔可均匀可爆发,burstiness 控制这个:

```
rate=3.4(平均间隔 0.294s)不变,burstiness 不同:

burstiness 大(→均匀):  |--0.29--|--0.29--|--0.29--|   每个间隔≈平均值,平稳
burstiness 小(→爆发):  |-0.05-||-0.6-|-0.02-||--0.9--|  忽长忽短,扎堆又空闲
```

两种平均速率都是 3.4,但对服务冲击不同:均匀=压力平稳好测稳态;爆发=瞬时过载又空闲,更接近真实流量。

技术上请求到达建模成 **Gamma 分布**,burstiness 是形状参数:
- =1 → 标准泊松过程(完全随机,中等爆发)
- >1 → 越大越**均匀**(趋向固定间隔)
- <1 → 越小越**爆发**

本期用 **burstiness=102**(很大)→ 请求几乎按固定间隔平稳到达。刻意选的:测拐点要稳定可比的负载,不要爆发噪声干扰。

### 5.5 输出指标(CSV 里的列)—— 测出来的结果

CSV 每行 **24 列**(与 `test_list.py` 的 `CSV_HEADER` 严格一致,已核对实际输出文件)。分**前置参数列**(本 case 的配置)和**结果列**(测出来的数据)两部分。

**前 7 列 —— 前置参数(标识这是哪个 case):**

| # | CSV 列名 | 含义 |
| --- | --- | --- |
| 1 | `model_name` | served-model-name(如 qwen3-8b) |
| 2 | `tp` | tensor-parallel-size(本期均为 1) |
| 3 | `input_len` | 输入长度(token) |
| 4 | `output_len` | 输出长度(token) |
| 5 | `io_label` | 输入输出档位标签(如 2k/1k) |
| 6 | `request_rate` | 发压速率 QPS(见 5.3) |
| 7 | `num_prompts` | 本 case 总请求数(本期 100) |

**后 17 列 —— 结果指标(测出来的数据)。** 不按 CSV 顺序平铺,按**性能维度分四组**,每组先说"反映哪方面性能、受谁影响"。

> 先厘清:这套 benchmark 测的是**服务/系统性能**(吞吐、延迟),**不测模型质量**(准确率/聪不聪明)。这里所有"性能"都是 **模型 + 推理引擎 + 这张卡** 三者的联合结果——所以横向比模型(8b vs 14b)差异来自模型/量化,横向比引擎(vllm_musa vs sglang)差异来自推理架构。本项目正是后者:固定模型和卡,对比两个引擎。

**① 吞吐类(产能)—— `req_tp` / `in_tok_tp` / `out_tok_tp`(列 8/9/10)**

| CSV 列 | 含义 |
| --- | --- |
| `req_tp` | **请求吞吐**(req/s):每秒实际处理完成多少请求。对照 `request_rate`——`req_tp ≈ request_rate`=跟得上;`<`=跟不上、排队(过载)。**req_tp 封顶不再涨 = 拐点** |
| `in_tok_tp` | 输入 token 吞吐(token/s),反映 prefill 处理能力 |
| `out_tok_tp` | 输出 token 吞吐(token/s),反映 decode 生成产能,衡量产出的核心 |

> **反映**:整机服务**产能上限**(每秒能干多少活)。**受谁影响**:模型大小、量化(FP8>BF16)、引擎 batching 调度、卡算力共同决定。同卡同引擎只换模型,req_tp 上限就差几倍(8b≈3、14b≈1)——所以它对模型/量化高度敏感,是横向对比的核心指标。

**② 首字延迟类(响应速度)—— `mean/median/p99_ttft`(列 11/12/13)**

| CSV 列 | 含义 |
| --- | --- |
| `mean_ttft` / `median_ttft` / `p99_ttft` | **TTFT(首字延迟)**:发请求到吐第一个 token 的时间,的 均值/中位数/P99 |

> **反映**:**交互响应感**(用户多久看到第一个字)。**受谁影响**:主要是 **prefill 阶段**——input 越长越慢;模型越大越慢;还受卡算力和调度排队影响。过载时 ttft 会因排队暴涨,是拐点的敏感信号。

**③ 生成速度类(出字流畅度)—— `mean/median/p99_tpot` + `mean/p99_itl`(列 14~18)**

| CSV 列 | 含义 |
| --- | --- |
| `mean_tpot` / `median_tpot` / `p99_tpot` | **TPOT**:每个输出 token 的平均耗时,的 均值/中位数/P99。`1/TPOT`=单请求出字速度 |
| `mean_itl` / `p99_itl` | **ITL**:相邻两 token 的实际间隔(TPOT 是它的平均),看抖动/毛刺 |

> **反映**:**生成阶段的流畅度**(文字往外蹦多快、稳不稳)。**受谁影响**:主要是 **decode 阶段**——模型大小、**显存带宽**、量化;并发越高被摊薄越多。ITL 的 P99 高但 TPOT 正常 → 大部分流畅、偶有卡顿。

**④ 并发与汇总类 —— `mean_e2e` / `real_concurrency` / `duration` / `total_*` / `status`(列 19~24)**

| CSV 列 | 含义 |
| --- | --- |
| `mean_e2e` | 端到端延迟均值:单请求从发出到完全返回的总时间(≈ TTFT + 生成总时长) |
| `real_concurrency` | **实际并发数**:任一时刻平均多少请求在飞。满足 Little's Law `并发 ≈ req_tp × mean_e2e`。未饱和时低且稳,**过载时猛涨**(堆积) |
| `duration` | 本 case 总耗时(s) |
| `total_input_tokens` / `total_output_tokens` | 本 case 总输入/输出 token 数 |
| `status` | OK / FAIL(失败时结果列全 0) |

> **反映**:**并发承载力**与整体规模。**受谁影响**:`real_concurrency` 主要由**引擎的 KV cache 管理 + 显存容量**决定(能同时容纳多少请求),最能体现"推理架构"差异;`mean_e2e` 是延迟的综合;`duration/total_*` 是规模核对量。

> 补充:TTFT/TPOT/ITL 均提供 **mean / median / P99** 多档(ITL 只有 mean/p99)。P99=99% 的请求好于此值(看最差情况,比平均更反映真实体验)。

**怎么用这些列判断过载 / 拐点**(配合 5.3):

| 信号组合 | 含义 |
| --- | --- |
| `req_tp ≈ request_rate` + `real_concurrency` 低 + ttft 低 | ✅ 健康,未饱和 |
| `req_tp` 封顶(不再随 rate 涨) + `real_concurrency` 猛涨 + ttft 暴涨 | ❌ 过载,已过拐点 |

本质是一回事:req_tp 到产能极限 → 多发的请求只能排队 → concurrency 堆积 → ttft 暴涨。判拐点时这几个列联动,看任一个都能佐证。例(qwen3-8b `2k/1k`):`request_rate=3.0` 时 `req_tp` 只有 1.35(远跟不上 3.0)、ttft 高 → 该档位已过载。

**一个完整 case 串起来看**(以 `Qwen3-8B-FP8, 2k/1k, rate=3.4` 为例):
> 起 Qwen3-8B-FP8 单卡服务 → 先发 500 个预热请求 → 然后以平均每隔 0.294 秒一个的速度,发 100 个请求,每个输入 2048 token、生成 1024 token → 统计这 100 个请求的 req_tp / TTFT / TPOT / 并发等 → 写一行 CSV。换 rate 再来一轮,扫完得到完整性能曲线。

#### 5.5.1 派生性能指标(业界公认,用于评判与横向对比)

24 列原始指标单看片面(吞吐高可能全是超时请求、延迟低可能没压满),业界用**派生/组合指标**来评判性能。分析脚本 `analyze.py` 从原始列算出这些指标,并出评判表 + 雷达图。

**核心范式:SLO 达标下的最大吞吐(Goodput)** —— MLPerf Inference / DistServe 等的金标准。

> 给定延迟约束(SLO,如 P99 TTFT<2000ms 且 P99 TPOT<50ms),不断加 rate,找到**仍满足 SLO 的最大 req_tp**,即该模型的"有效服务容量"。一个数就能横向比模型/比引擎,远比看 20 行原始数据清晰。"吞吐高但全超时"没意义,Goodput 只算"既处理完又达标"的请求,最能反映真实可用产能。

**可从单点算出的派生指标:**

| 派生指标 | 公式 | 含义 / 用途 | 业界叫法 |
| --- | --- | --- | --- |
| **有效服务容量** | SLO 达标 case 中的 max `req_tp` | 满足延迟约束下的最大请求吞吐 | Goodput / effective capacity |
| **TGS**(每卡 token 吞吐) | `out_tok_tp / tp` | 单卡产能,跨 TP 配置可比 | Token/GPU/s(xlsx 里就有此列) |
| **归一化延迟** | `mean_e2e / output_len` | 每个输出 token 摊到的端到端时间(ms/token,越小越好) | Normalized Latency |
| **goodput 比** | `req_tp / request_rate` | 实际处理/期望处理,<1 即过载 | goodput ratio |
| **并发效率** | `out_tok_tp / real_concurrency` | 每个并发槽的产出 | per-slot throughput |

**SLO 阈值标准**(来自 sglang release 文档,按输入长度分档):

| 输入长度档 | TTFT 上限 | TPOT 上限 |
| --- | --- | --- |
| ~512 | 1000ms | 33/50/100ms 三梯度 |
| 1024 | 2000ms | 同上 |
| **2048~4096(本期档位)** | **2000ms** | **33/50/100ms** |
| 8192 | 2500ms | 同上 |

> 本期默认 SLO 取 **TTFT<2000ms、TPOT<50ms**(2k~4k 档的中间梯度)。`analyze.py --ttft-slo / --tpot-slo` 可改。

**为什么不简单"加权打一个总分"**:延迟和吞吐是 trade-off,加权和会掩盖曲线形状。业界看**曲线**而非单点——

- **吞吐-延迟曲线**(throughput-latency curve):X 轴 req_tp,Y 轴 P99 延迟,曲线"膝盖点(knee)"是最优工作点。两个引擎曲线叠一起,谁更靠右下谁更好——**对比 vllm_musa vs sglang 最权威的图**。
- **雷达图**:把 有效容量 / 峰值TGS / 低延迟 / SLO达标率 / Goodput 五维归一化叠加,多模型一图对比强弱(`analyze.py` 输出 `radar.png`)。

**用法**:
```bash
python3 analyze.py <benchmark_summary_*.csv> \
    --ttft-slo 2000 --tpot-slo 50 \
    --radar radar.png --export derived_metrics.csv
```

### 5.6 哪些参数要按模型改 / 哪些全局统一

| 参数层 | 位置 | 本期处理 |
| --- | --- | --- |
| **模型列表** | `run_list.py` 的 `MODELS` | ✅ 换成本期 8 个模型路径 |
| 启动参数(TP / 显存 / block-size / cuda-graph) | `run_list.py` 全局常量 | ❌ 全局统一即可(都是 tp1,`TP=1`/`GPU_MEM_UTIL=0.8`/`BLOCK_SIZE=64` 不变) |
| **压测 rate 序列** | `test_list.py` 的 `BENCH_CASES` | ✅ **必须 per-model**(每个模型 rate 范围不同,见下表),由参数文件 `tp1_bench_params.json` 驱动 |
| 压测其它参数(num_prompts / warmup / range-ratio / burstiness) | `test_list.py` 全局常量 | ❌ 全局统一(`NUM_PROMPTS=100`、`WARMUP_REQUESTS=500`、`range-ratio=1.0`、`burstiness=102`) |

**为什么 rate 要 per-model**:`request-rate`(QPS,每秒发多少请求)要"从低到高逐档加压",观察 TTFT/TPOT/吞吐何时饱和,找性能拐点。8 个模型快慢差异大,拐点位置不同——慢模型(14B)用快模型(3.5-0.8B)的高 rate 会全程过载、测不到拐点;反之压不满。所以每个模型用各自在 release 文档里测过的 rate 序列。

### 5.7 本期 8 个模型 + 各自 rate 序列(来自 release 文档)

输入输出组合统一为 5 档:`2k/1k`、`3k/1k`、`3.5k/1k`、`4k/1k`、`4k/1.5k`(input/output token 长度,`range-ratio=1.0` 固定不浮动)。每档下扫不同 rate:

| 模型 | 目录 | rate 范围 | cases 数 |
| --- | --- | --- | --- |
| Qwen3-8B(BF16) | `Qwen3-8B` | 1.0 ~ 3.0 | 20 |
| Qwen3-8B-FP8 | `Qwen3-8B-FP8` | 1.0 ~ 3.6 | 18 |
| Qwen3-14B(BF16) | `Qwen3-14B` | 0.4 ~ 1.4 | 14 |
| Qwen3-14B-FP8 | `Qwen3-14B-FP8` | 0.4 ~ 1.8 | 16 |
| Qwen3.5-9B | `Qwen3.5-9B` | 0.3 ~ 2.0 | 58 |
| Qwen3.5-4B | `Qwen3.5-4B` | 0.3 ~ 2.0 | 59 |
| Qwen3.5-2B | `Qwen3.5-2B` | 1.4 ~ 4.0 | 31 |
| Qwen3.5-0.8B | `Qwen3.5-0.8B` | 1.6 ~ 4.0 | 25 |

模型路径前缀均为 `/mnt/seed17/001688/models/`。完整逐档 rate 序列见参数文件 `tp1_bench_params.json`。

#### 各模型逐档 rate 明细

**Qwen3-8B(BF16)** — `2k/1k`:2.0/2.2/2.4/2.6/2.8 · `3k/1k`:2.0/2.2/2.4/3.0 · `3.5k/1k`:2.0/2.2 · `4k/1k`:1.0/1.2/1.4/1.6/1.8 · `4k/1.5k`:1.0/1.2/1.4/2.0

**Qwen3-8B-FP8** — `2k/1k`:3.0/3.2/3.4/3.6 · `3k/1k`:2.0/2.2/2.4/2.6/2.8 · `3.5k/1k`:2.0/2.2/2.4 · `4k/1k`:2.0/2.2 · `4k/1.5k`:1.0/1.2/1.4/1.6

**Qwen3-14B(BF16)** — `2k/1k`:1.0/1.2/1.4 · `3k/1k`:0.4/0.6/0.8/1.0 · `3.5k/1k`:0.4/0.6/0.8 · `4k/1k`:0.4/0.6/0.8 · `4k/1.5k`:0.4

**Qwen3-14B-FP8** — `2k/1k`:1.0/1.2/1.4/1.6/1.8 · `3k/1k`:1.0/1.2 · `3.5k/1k`:1.0/1.2 · `4k/1k`:0.4/0.6/0.8/1.0 · `4k/1.5k`:0.4/0.6/0.8

**Qwen3.5-9B** — `2k/1k`:1.1~2.0(步进0.1) · `3k/1k`:0.6~1.5 · `3.5k/1k`:0.4~1.7 · `4k/1.5k`:0.3~1.5 · `4k/1k`:0.6~1.7

**Qwen3.5-4B** — `2k/1k`:1.1~2.0 · `3k/1k`:0.6~1.5 · `3.5k/1k`:0.4~1.7 · `4k/1.5k`:0.3~1.6 · `4k/1k`:0.3/0.8~1.8

**Qwen3.5-2B** — `2k/1k`:1.8/1.9/2.0/2.5/3.0/4.0 · `3k/1k`:1.5/2.0~2.5 · `3.5k/1k`:1.7/2.0~2.4 · `4k/1.5k`:1.4~2.0 · `4k/1k`:2.0~2.4

**Qwen3.5-0.8B** — `2k/1k`:2.8/2.9/3.0/4.0 · `3k/1k`:2.0/2.5~2.9 · `3.5k/1k`:2.5~2.8 · `4k/1.5k`:1.6~1.9/2.5/2.8 · `4k/1k`:2.5~2.9

> 注:Qwen3.5-9B / 4B 的 rate 点很密(58/59 个 case),整批跑耗时较长,可按需精简。

#### 第二轮(2026-06-11):多卡 Qwen3-32B（tp4 集中式）

tp1 单卡 8 模型跑完后,转向 Qwen3-dense 里**唯一有多卡数据的 Qwen3-32B**。release 文档对 dense 多卡只测了两种部署:**tp4 集中式**(4 卡)和**单机 PD 分离**(prefill tp4 + decode tp4 = 8 卡)。本轮先做 tp4 集中式;PD 分离需新写启动层,另行处理(见第七章)。

| 模型 | 目录 | 部署 | rate 范围(粗扫) | cases 数 |
| --- | --- | --- | --- | --- |
| Qwen3-32B-BF16 | `Qwen3-32B` | tp4 + EAGLE3 投机 | 0.3 ~ 2.6 | 27 |
| Qwen3-32B-FP8 | `Qwen3-32B-FP8` | tp4 + EAGLE3 投机 | 0.3 ~ 2.6 | 27 |

- **8 卡同时跑**:BF16 占卡 0-3(port 8000)、FP8 占卡 4-7(port 8001),由 `auto_bench.py --tp 4` 驱动(每个 server 占 tp 张卡,多卡支持见第七章)。⚠ 两组 tp4 并发会共享卡间互联(allreduce),与 sglang 单部署逐个测的条件不完全对齐,结论里需标注"并发测"。
- **投机解码**:两个都开 EAGLE3,共用 draft `/mnt/seed17/001688/models/Qwen3-32B_eagle3`(对齐 sglang;sglang 的 BF16/FP8 tp4 启动指令均带 `--speculative-algorithm EAGLE3`);`--max-num-seqs 256`。
- **rate 来源**:对齐 sglang Qwen3-dense 表的 tp4 发压QPS 并向两侧扩(粗扫夹拐点)。sglang 参考:BF16 tp4 ≈ 1.0~1.4(`4k/1.5k` 0.4~0.8)、FP8 tp4 ≈ 1.0~1.6(`4k/1.5k` 0.4~1.0)。

**逐档 rate(BF16 / FP8 同一套粗扫网格)** — `2k/1k`:0.6/1.0/1.4/1.8/2.2/2.6 · `3k/1k`:0.6/1.0/1.4/1.8/2.2 · `3.5k/1k`:0.6/1.0/1.4/1.8/2.2 · `4k/1k`:0.6/1.0/1.4/1.8/2.2 · `4k/1.5k`:0.3/0.5/0.7/0.9/1.1/1.3

> 参数文件:`tp4_bench_params_32b_coarse.json`。这是**粗扫**轮,跑完用 `find_knee.py` 定位拐点后再生成精扫文件跑第二遍(两轮找拐点流程见 5.9.3)。

### 5.8 语料文件(所有模型共用一份)

压测用 `--dataset-name random` 模式,语料文件 `ShareGPT_V3_unfiltered_cleaned_split.json` 只作为 token 采样源,真正决定测试内容的是 input/output 长度参数。**8 个纯文本模型共用同一份语料**,无需按模型区分。(仅多模态模型需要单独的图像数据集,本期不涉及。)

### 5.9 拐点定义与找拐点方法(2026-06-04 确立)

压测的核心目标是找"拐点":随 request-rate 增大,系统从"跟得上"翻转到"过载"的临界点。实践中发现拐点不是一个,而是**两个相互独立的拐点**。

#### 5.9.1 两个拐点的定义

| 拐点 | 判据 | 含义 |
| --- | --- | --- |
| **容量拐点(throughput knee)** | **goodput 比 = req_tp / request_rate** 跌破 ~0.85 | 推理框架 + 硬件资源的**吞吐上限**(每秒最多处理多少请求)。**不依赖 SLO 阈值**,是系统固有属性。 |
| **性能拐点(latency knee)** | **mean_tpot** 相对健康区基线(最低几档最小值)**跳变 > ~1.8 倍** | **用户感知延迟开始恶化**的点(SLO 边界)。 |

**因果关系:容量拐点 ≤ 性能拐点**(实测处处成立)。延迟恶化是"吞吐已饱和、请求开始堆积/并发上涨"的**后果**,所以一定是先吞吐饱和(req_tp 压平),之后才轮到延迟暴涨。
- 小模型极端例子:Qwen3-8B 吞吐 ~1.6 就饱和(容量拐点),但因为模型快、满载也不让 TPOT 暴涨,性能拐点远在其后甚至测不到。
- 两个拐点回答不同问题:容量拐点 = "硬件框架能推多少"(真正该卡的运营点);性能拐点 = "延迟还能撑到多高 rate"。

#### 5.9.2 为什么用 req_tp 平台当主判据(而非 SLO 阈值)

- **SLO 阈值(如 P99 TTFT<2000、TPOT<50)是拍脑袋的**,设严了会在真容量拐点之前误判过载,设松了又漏。我们一开始并不知道合适阈值是多少。
- **req_tp 平台 / goodput 比是无参数的**:goodput≈1 是天然参考点(吞吐跟得上),跌破即饱和——不需要先验知识。
- 正确顺序:**先用 req_tp 平台定位容量拐点,顺手量到拐点处实际 TTFT/TPOT,再反推合理 SLO 阈值**,而不是反过来。
- 性能拐点用 mean_tpot 跳变(比 TTFT 干净,TTFT 跟队列/测试时长耦合、噪声大)。两个判据各测各的,不要求一致——快模型上两者会在不同 rate 触发。

#### 5.9.3 找拐点方法:两轮 + 自动定位(摒弃二分)

- **为什么不用二分**:① 拐点附近测量亚稳/有噪声,单点硬判左右会带偏;② 二分反复扎过载边界(最贵区域),墙钟未必省;③ 要两个拐点 + 整条曲线(雷达图/对比需要),二分只给标量;④ 范围已收窄,对数效率优势没了。
- **两轮法**:
  1. **第一轮(粗/锚定)**:每个模型扫一段 rate,**对数间隔**(拐点按倍数分布,对数布点相对分辨率一致;等间隔会"该密的疏、该疏的密")。
  2. **`find_knee.py` 自动定位**:读 summary,对每个(模型×档)算 goodput 比 + tpot 跳变,输出容量/性能两个拐点区间(`--knee throughput/latency/both`),并生成第二轮精扫配置。
  3. **第二轮(精扫)**:只在拐点区间内**等间隔**密扫(窄区间内均匀分辨率最优)。
- **过载侧别扫深**:req_tp 一压平、再采 1-2 点确认即停,不深入雪崩区(256 并发下深度过载单 case 要几分钟)。

#### 5.9.4 参数分析:各参数控制什么(找拐点时怎么用)

| 参数 | 控制什么 | 找拐点时 |
| --- | --- | --- |
| **request-rate** | **每秒到达多少请求 = 压力本身(横轴/自变量)** | 扫它找拐点。它是横轴,各系统各扫各的,**不需要和 sglang 取一样的值**(对比的是拐点 vs 拐点)。 |
| **num-prompt** | 一个 case **发多少条 = 测多久**(时长≈num-prompt÷rate),**不改每秒压力** | 它**不决定压力**,但要够大才能让过载充分暴露(队列堆起来要时间)。实测 **100 已足够**(上一轮过载暴露彻底);500 只在逐行对 sglang 同一 rate 的确切延迟时才必要(而那场景全过载、无意义);别用 20 那种(过载测不出、拐点虚高)。 |
| **input/output 长度** | 定义"档位"的工作负载 | **必须和 sglang 一致**(本期 5 档 2k~4k 已对齐 dense/3.5 sheet)。 |
| **--max-num-seqs** | 并发上限 / 背压 | **保持 256 对齐 sglang**。它只在过拐点之后才起作用(拐点处真实并发 ~20-30,远不到 256),所以拐点位置与它无关;256 下 req_tp 照样压平、拐点照样可定位。 |

**与 sglang 对齐的核对结论(2026-06-04)**:
- ✅ 输入输出档位:一致(我们 5 档 = dense/3.5 sheet)。
- ✅ 扫法:一致(都是固定 num-prompt 扫 rate;sglang 的 8B-EAGLE3 专项 sheet 是 `num-prompt=rate` 的并发突发测法,与我们不同,但**那不是我们对标的 sheet**)。
- ⚠ num-prompt:sglang 这 8 个模型用的 **500**,我们用 100。但容量拐点是系统固有属性、与 num-prompt 无关(100 已够暴露过载),故找拐点用 100 即可。
- 关键认知:**上一轮(run_20260603)用的 rate 就是 sglang 发压QPS 的原值,结果几乎全过载——证明 vllm_musa 的拐点在 sglang 范围之下**。所以对比要做的是"拐点 vs 拐点"(sglang 拐点 ≈ 它测的范围上沿,从它数据读出;vllm_musa 拐点往低 rate 扫才测得到),而非在 sglang 的 rate 下逐行比(那只会得到"哪都过载")。

#### 5.9.5 解读拐点:吞吐与延迟会解耦,看哪个拐点取决于"背压落在哪"

> 本节是**与卡数无关的通用判读规则**,单卡/多卡(TP/PP/更大模型)同样适用;下面的单卡模型只作举例,真正要记的是规律,不是具体数值。

**① 容量拐点和性能拐点会解耦,不能只看 goodput 判"是否可用"。**
随 rate 增大,goodput(req_tp/rate)先掉(吞吐到顶),但**延迟(TTFT/TPOT)未必同步恶化**。会出现"goodput 已明显 <1、但 TTFT/TPOT 仍健康"的区间——此时系统是"吞吐到顶但体验无碍",不是"过载到不可用"。所以判断一个工作点能不能用,要**吞吐和延迟一起看**,不能只凭 goodput 一条线。

**② 为什么会解耦:背压(backpressure)落在 TTFT 还是 TPOT,由「并发上限 vs 在飞请求数」决定。**
- 推理服务有个并发上限(`--max-num-seqs`,即最多同时运行多少条序列)。
- **在飞请求数 ≤ 并发上限**:没有请求在"准入队列"里等名额 → 每条一到就开跑 → **TTFT 只含 prefill 本身、不含排队**。吞吐跟不上(goodput<1)就表现为"大家都在 batch 里一起慢慢跑完",压力体现在 **TPOT / 吞吐天花板**,而非 TTFT。
- **在飞请求数 > 并发上限**:超出的请求在准入队列里等 → 这些请求的 **TTFT 暴涨**。压力体现在 **TTFT / 排队**。
- 一句话:**并发上限设得高 → 背压压在 TPOT;设得低 → 背压压在 TTFT(排队)**。这也是为什么对齐被测系统(如 sglang)的并发上限很重要——它直接改变"过载长什么样"。

**③ 但"没排队"不等于"延迟一定低",还要看硬件是否真饱和。**
即使在飞数没超并发上限(无准入排队),如果**算力已被占满**,新请求的 prefill 会被在跑的 decode 饿着(chunked prefill 交织),**TTFT 仍会涨**;反之算力没满则 TTFT 保持低。
> 举例(本期单卡):同样"在飞数<并发上限、无准入排队",小模型(Qwen3.5-0.8b)因算力没压满,goodput 掉到 0.4 时 TTFT 仍 ~90ms;重模型(Qwen3.5-9b)因算力饱和、prefill 被饿,TTFT 直接飙到几十秒。**模型/并行规模越重,越容易在这里崩**——多卡大模型尤其要盯这一点。

**④ 有限测试的 caveat:小 num-prompt 会"掩盖"过载。**
压测每个 case 只发有限条(num-prompt)。当 **num-prompt ≤ 并发上限**时,在飞数永远到不了上限、准入队列根本没机会形成 → 过载的排队效应测不出来,测出的"延迟很好"**不能外推到持续过载的生产**(真实持续高负载下在飞数会超并发上限、队列形成、TTFT 爬升)。要把排队暴露出来,需 **num-prompt > 并发上限**。能安全外推的是**吞吐天花板(req_tp 平台值)**,它是系统固有属性。

**⑤ 由此得到的实践判读规则(找拐点时怎么取舍):**
- **容量拐点(吞吐天花板)是可外推、可比的真数**,优先以它为准;它常能直接从 req_tp 压平的位置读出,不必非要精确卡到 goodput=0.85 那一点。
- **若某档容量拐点落在最低测试 rate 之下、但延迟全程健康** → 该档在测试范围内是"吞吐到顶但体验无碍",**再往更低 rate 补测意义不大**(吞吐天花板已能读出,且不存在延迟问题)。
- **真正要警惕、值得深挖的是延迟也崩的档**(TTFT/TPOT 同步暴涨)——那才是会影响用户体验的真过载。
- 异常孤点(某单一档延迟远超相邻档)优先怀疑测量故障/坏路径,**单独复测一格**确认,而非顺着往下扫。

---

## 六、已知问题与修复

> 交接脚本(`run_list.py` / `test_list.py`)在使用中暴露的两个问题,基于历史数据(`bench_results_v1/v2/v3`,2026-05-26~27 三轮)分析确认。自动化封装会一并修掉。

### 6.1 误区澄清:"重复跑一轮比一轮好" —— 数据不支持

前任使用者反馈"同一模型跑几轮后效果比前几轮好,怀疑是 warmup 问题"。**核实结论:整轮重跑结果稳定,没有逐轮变好。**

证据:qwen3-8b 与 qwen3-8b-fp8 在 v1→v2→v3 三轮逐 case 对比,ttft/tpot 差异均在 ±5% 随机抖动内,无系统性改善。例(qwen3-8b `2k/1k rate=3.0` 的 mean_ttft):

| 轮次(真实时间序) | v1(5/26) | v2(5/27) | v3(5/27) |
| --- | --- | --- | --- |
| mean_ttft(ms) | 938.6 | 924.0 | 930.6 |
| mean_tpot(ms) | 53.22 | 53.24 | 53.27 |

三轮纹丝不动——这正是 benchmark 该有的可复现性。**重跑哪一轮都一样,"取后面几轮"是错误做法。**

### 6.2 真正的问题:每个输入长度档的"第一个 case"偏慢(warmup 未覆盖多形状)

使用者看到的"变好",真相是**一次运行内、每个输入长度档的第一个 rate case 偏慢**,从第二个 case 起恢复正常——被误记成了"一轮比一轮好"。

硬证据(qwen3-8b-fp8,同一 `2k/1k` 档,rate 递增,三轮稳定复现):

| case | mean_ttft(ms) | 说明 |
| --- | --- | --- |
| `2k/1k` rate=3.0 | 600 | ← 该档**第一个** case |
| `2k/1k` rate=3.2 | 119 | ← 压力更大,ttft 却暴降 80% |
| `2k/1k` rate=3.4 | 117 | 稳定 |

**rate 升高(负载更重)ttft 反而暴跌,违反物理规律**——唯一解释:该档第一个 case 仍在冷启动(CUDA/MUSA Graph 现场捕获、显存分配器预热)。

**根因**:`bench_serving.py` 的 warmup 只用 `input_requests[0]` 一条请求的形状重复预热(见 `bench_serving.py:2016` 附近)。而一个模型要扫 5 种输入长度(2k/3k/3.5k/4k),**warmup 只热了第一种**;每换一个长度档,该档首个 case 又要现场捕获 graph → 偏慢。

**修复**(自动化封装中落地):每个模型正式测前,**对所有要测的输入长度各预热一遍**(而非只热第一种),消除每个档位首 case 偏慢。

### 6.3 附带问题:固定 rate 序列对慢模型导致过载

历史数据里 qwen3-8b 的长输入档(如 `4k/1k`)ttft 高达 2 万+ ms,是 rate 设太高导致的过载排队假象,非真实性能。根因是**所有模型共用一套固定 rate 序列**(见 5.6)。修复:per-model rate(`tp1_bench_params.json`)。

### 6.4 (2026-06-02 实测)直接套用 sglang 的 rate,在 vllm_musa 上整体偏高 → 87% case 过载

**现象**:2026-06-02 用 auto_bench 跑完 8 模型 241 case(全 OK、0 FAIL),但用 `analyze.py` 分析发现 **210/241(87%)case 处于过载**(`req_tp` 远跟不上 `request_rate`,ttft 高达 20~33 秒,且 concurrency 几乎全卡在 ~60)。

**SLO 达标情况**(P99 TTFT<2000ms、TPOT<50ms):

| 模型 | SLO 达标 / 总 | 有效容量(req/s) |
| --- | --- | --- |
| qwen3.5-0.8b | 25/25 ✅ | 1.65 |
| qwen3.5-2b | 31/31 ✅ | 1.56 |
| qwen3-8b-fp8 | 3/18 | 1.57 |
| qwen3-8b | **0/20** ❌ | 0 |
| qwen3-14b-fp8 | **0/16** ❌ | 0 |
| qwen3-14b | 2/14 | 0.34 |
| qwen3.5-9b | 2/58 | 0.36 |
| qwen3.5-4b | 4/59 | 0.44 |

只有两个最小模型(0.8b/2b)全达标,其余 6 个 rate 都偏高、有效容量≈0。

**根因(推断,待 #15 验证)**:这批 rate 是从 sglang release 文档抄的,而 sglang 那次启动开了 **EAGLE3 投机解码**(`--speculative-algorithm EAGLE3`,一次吐多 token,产能高得多);vllm_musa 这边**没开投机解码**,产能低一截,所以 sglang 扛得住的 rate,vllm_musa 全过载。另注:concurrency 普遍顶在 ~60,疑似 `--max-num-seqs` 或调度并发上限,待查。

**真实拐点证据**:14B `3k/1k` 档 rate=0.6→0.8 之间 ttft 从 565ms 暴涨到 5338ms —— 真实拐点在 rate≈0.6,而文档给的序列(0.4~1.4)大部分点都在拐点之后。

**结论**:数据本身真实准确(脚本无 bug,过载是真过载),但 rate 取点系统性偏高,采到的几乎都是过载区,**这批数据可用性低**。需:① 查清是否要对齐引擎配置(开投机解码)#15;② 把 rate 序列整体下移到健康区重测 #16。

### 6.5 (2026-06-03 查清)过载根因 = sglang 与 vllm_musa 启动配置差异大,rate 直接套用不公平

对照 xlsx 里 sglang 的启动指令(W 列)与本次 vllm_musa 的 `run_list.py`,根因有三:

| 配置 | sglang(xlsx) | vllm_musa(本次) | 影响 |
| --- | --- | --- | --- |
| **投机解码** | 8B/8B-FP8 开 **EAGLE3**(`--speculative-algorithm EAGLE3` + draft `Qwen3-8B_eagle3` + `num-steps 3`,`SGLANG_ENABLE_SPEC_V2=1`) | ❌ 没开 | **决定性**:EAGLE3 一次验证吐 3+ token,吞吐高 2~3 倍。vllm_musa 裸产能低,同 rate 全过载 |
| **并发上限** | `--max-running-requests 256` | ❌ 没设 `--max-num-seqs`(用默认),`max_num_batched_tokens=8192` | 长输入下 2~3 个请求就占满 batch token 预算 → 并发被压到 **~60**,排队 ttft 暴涨。解释了 concurrency 全卡 60 |
| 显存 | `--mem-fraction-static 0.9` | `--gpu-memory-utilization 0.8` | sglang 多 10% 显存做 KV cache |
| attention / sampling | `fa3` / `flashinfer` | musa 默认 | — |

> 注:14B 那两个 sglang **没开 EAGLE3**(改用 `--disable-radix-cache --chunked-prefill-size 8192`);EAGLE3 draft model 只有 8B 有(`/mnt/seed17/001688/yaoxi/Qwen3-8B_eagle3`)。

**核心含义**:直接套 sglang 的 rate 本就不公平——那套数字是"开投机解码 + 256 并发"的产能,裸配置的 vllm_musa 达不到。

**重测前的两个待决策点(取决于测试目的):**

1. **目的是什么?**
   - **vllm_musa vs sglang 公平 PK**(比引擎)→ 需对齐配置(并发 + 投机解码),否则是"裸 vllm" 比 "全武装 sglang"。
   - **摸 vllm_musa 标准配置基线**(给自己定基准)→ 不用对齐,但 rate 要下移到它能扛的范围。

2. **并发上限**:`run_list.py` 加 `--max-num-seqs 256`(对齐 sglang)?——改动小、风险低、收益明确(解开 60 并发瓶颈),基本应该调。与投机解码是两件独立的事。

3. **投机解码(EAGLE3)**:要对齐需先确认 ① vllm_musa 0.20 是否支持 EAGLE3 ② 那个 draft model 能否被 vllm 加载。且只有 8B 有 draft,其它模型 sglang 本来也没开。**待确认**。

**下一步(#16)**:据上述决策,调整启动配置 + 把 rate 序列整体下移到健康区(参考本次过载数据反推的真实拐点,如 14B `3k/1k` 拐点在 rate≈0.6),重跑得到可用曲线。

### 6.6 (2026-06-03)过载明细 + 关键修正:多数"过载"是并发上限造成的假象

> 用 `analyze.py` 的 goodput 比(`req_tp/request_rate`,<0.7 记过载 ⚠)逐 case 统计,并结合 concurrency/ttft 复核,得到一个**推翻早期判断**的结论:大部分模型的"过载"不是算力到顶,而是 `--max-num-seqs` 默认值把并发锁死在 ~60~73 造成的假象。

**各模型过载统计(本次 2026-06-02 数据,默认配置,未开投机/未调并发):**

| 模型 | 过载/总 | 健康区最高 rate | 是否真过载 |
| --- | --- | --- | --- |
| qwen3-8b | 20/20 | 无 | ❓ 疑假象(待验证) |
| qwen3-8b-fp8 | 18/18 | 无 | ❓ 疑假象 |
| qwen3-14b | 9/14 | 0.6 | ✅ **真过载**(ttft 暴涨) |
| qwen3-14b-fp8 | 15/16 | 0.4 | ✅ 真过载 |
| qwen3.5-9b | 47/58 | 0.7 | ❌ **假象**(见下) |
| qwen3.5-4b | 48/59 | 0.8 | ❌ 假象 |
| qwen3.5-2b | 28/31 | 1.8 | 部分假象 |
| qwen3.5-0.8b | 25/25 | 无 | ❌ 假象(最小模型却全过载,反常) |

**关键证据——区分"真过载"和"并发假象"靠看 ttft 是否暴涨:**

- **真过载(qwen3-14b `3k/1k`)**:rate 0.6→1.0,ttft `566 → 12072 ms` 暴涨 20 倍 → 算力真的到顶,请求排长队。
- **并发假象(qwen3.5-9b `2048/1024`)**:rate 1.1→2.0,**ttft 全程稳在 ~310 ms**(没暴涨!),但 req_tp 卡在 ~0.65 上不去,concurrency 爬到 ~73 就封顶。系统明明很轻松(ttft 低),却被判"过载"。

**机制分析(为什么并发上限会制造"假过载"):**

```
req_tp = real_concurrency / mean_e2e_latency
```

`--max-num-seqs` 默认值把 `real_concurrency` 锁死在 ~60~73(并发槽位满)。一旦封顶:
- req_tp 也跟着封顶(分子上不去)→ 你发的高 rate 它"接不下",goodput 比 <0.7 被判过载;
- 但请求是被**挡在并发槽位外**(没进来),不是进来后算不动 → 所以**已接收请求的 ttft 依然很低**。

这是"门口排队"(并发限流)而非"厨房做不过来"(算力过载)。最小模型 qwen3.5-0.8b 反而 25/25 全过载,正是铁证:它算力最强,绝不可能因算力过载,只可能是并发被锁死。

**提高 concurrency(`--max-num-seqs 60→256`)对 goodput 比的预期影响:**

| 模型类型 | 现状 | 调到 256 后预期 |
| --- | --- | --- |
| 假象型(0.8b/2b/4b/9b/8b) | concurrency 卡 ~73,req_tp 封顶,goodput<0.7 | concurrency 放开 → req_tp 跟着 rate 上升 → goodput 比回升、**真实拐点上移**,原 rate 可能反而不够高 |
| 真过载型(14b/14b-fp8) | ttft 已暴涨,算力到顶 | 改善有限,但拐点也会**略升**(并发放开后能多塞几个) |

**结论(修正早期判断)**:此前以为"小模型并发瓶颈、大模型算力瓶颈"的二分是**错的**。实测显示**几乎所有模型的拐点都被 `max-num-seqs≈60` 压低了**(连慢的 9b/4b 都没到真算力拐点)。因此现有这批 rate 数据**整体被并发上限污染**,必须放开并发(256)重测才能得到任何真实拐点;不能凭这批数据盲目下移 rate。

**已就绪的配置(待重测验证,见 7.4.1)**:全部 8 模型 `--max-num-seqs 256`;Qwen3-8B/8B-FP8 加 EAGLE3 投机。重测后预期:假象型模型显出真实曲线(可能需上调 rate),真过载型(14B)维持低 rate 健康区。

### 6.7 (2026-06-04)放开并发 + 投机重测结果:6.6 假设被推翻,放开并发暴露真过载

**本次参数改动(对齐 sglang,见 7.4.1 / 测试配置 MD)**:
- ① 全部 8 模型加 `--max-num-seqs 256`(原版未设,实测被压到 ~60-73);
- ② 两个 8B 加 EAGLE3 投机解码(`--speculative-config`,num_speculative_tokens=3,draft=`Qwen3-8B_eagle3`);
- ③ rate 序列仍沿用 6.6 的(未下移,目的是先看放开并发后真实曲线)。

**结果**:`bench_results/run_20260603_173409/`,241 case 全 OK,耗时约 7h。`analyze.py` 评判(SLO: P99 TTFT<2000ms 且 P99 TPOT<50ms):

| 档 | 模型 | SLO 达标 | 表现 |
|---|---|---|---|
| 健康 | Qwen3.5-0.8b / 2b | 25/25、31/31 | 真快,测试 rate 区间在健康区,TPOT 26-44ms |
| 临界 | Qwen3-8b / 8b-fp8(投机) | 0/20、1/18 | mean_tpot 不错(~33ms)但 **P99_tpot 压线 48-65、P99_ttft 刚超 2000(2400-2800)**,rate 略高 |
| 真过载 | Qwen3-14b/-fp8、3.5-9b、3.5-4b | 1-2 / 几十 | **TPOT 暴涨 100-300ms,TTFT 飙到 30-157 秒**,rate 远超拐点 |

**关键结论:放开并发没有让"假过载"消失,反而暴露了"真过载"——6.6 的假设方向是错的。**

- 6.6 猜:解开 ~73 并发上限后,假过载模型会显出真实曲线、**可能需要更高 rate**。实测**相反**。
- 真相:那个 `--max-num-seqs` 上限**其实在帮重模型做背压(admission control)**,把在飞请求数压住,TTFT 才稳。设成 256 撤掉背压后,4b/9b/14b 的队列**直接跑飞**(TTFT 几十秒~上百秒),暴露出可持续吞吐其实只有 **0.3-0.4 req/s**。
- 拐点是**悬崖式**:concurrency 越过 ~20-30 后 TPOT/TTFT 失控发散(同一 rate 下有的 case concur 13 健康、有的 concur 53 雪崩,典型饱和双稳态)。例:Qwen3.5-4b `4096/1024@rate0.3`→concur 7.8/TPOT 27ms ✅,`3584/1024@rate0.4`→concur 53/TPOT 207ms ❌。
- 所以 4b/9b/14b 需要的是**大幅下移 rate**,不是上移。

**对两个改动的回看**:
- `--max-num-seqs 256`:**保持 256,继续对齐 sglang(sglang 用的就是 `--max-running-requests 256`),不要降**。⚠️ 修正早前判断:256 vs 64 **只在"过拐点之后"才有区别**;拐点及健康区的真实并发远不到 64(这次重模型拐点在 concur ~20-30),所以**拐点位置和健康区数据两者完全一样**。而且实测在 256 下 **req_tp 照样压平**(4b 在 rate 0.3~1.8 一大片都卡在 ~0.3 req/s),拐点完全可定位。256 只是让过拐点后从"吞吐压平的膝盖"变成"延迟雪崩的悬崖",膝盖位置没变。**所以这次失败的根因不是 256,是 rate 整段都在拐点上方、下方一个点都没采到**。
- 投机解码(8B):mean_tpot 33ms 看着好,但**无法判定是投机的功劳还是拖累**——投机在低负载(小 batch)受益、高并发(concur 50+)时 draft+verify 额外开销可能拖慢 P99。这次 8b 正好卡在 concur 50-55、P99_tpot 压线,**怀疑高并发下投机帮倒忙,需开/关投机各跑一遍低 rate 做 A/B 才能定论**。

**下一步:两轮自适应找拐点(2026-06-04 定方案)**
- **并发**:8 模型一律保持 `--max-num-seqs 256`(对齐 sglang),投机 8B 保留。
- **rate**:废弃 per-model 手调,改**统一一套对数间隔 rate(0.1~6,10 点)**套全部模型——低 rate 端密照顾重模型、高 rate 端疏照顾小模型,一套就能把所有模型拐点夹住,摆脱"先知道拐点才能设 rate"的鸡蛋问题。
- **两轮**:① 粗扫(统一对数 rate)→ ② `find_knee.py` 自动定位每个(模型×输入档)的拐点区间 → ③ 精扫(只在拐点区间用 0.05 步长密扫)。
- **拐点判据(无参数优先)**:主用 **req_tp 平台 / goodput 比 req_tp/rate 跌破 ~0.85**(容量的直接定义,不依赖 SLO 阈值),佐证用 **mean_tpot 跳变 >~1.8 倍**,两信号一致才确认。**不再用 SLO 阈值当主判据**(2000/50 是拍脑袋的,严了会在真拐点前误判);改为先用 req_tp 法定位拐点、量到拐点处实际 TTFT/TPOT,再反推合理 SLO。
- **过载区别扫深**:req_tp 一压平、再采 1-2 点确认即停,不深入雪崩区(256 下深度过载单 case 要几分钟)。
- ④ 8b 投机开/关 A/B(低 rate)。

### 6.8 (2026-06-04)往 sglang 范围下方重测(round2):拐点采到了

按 6.7 + 5.9 的方法,第二轮往每个模型 sglang 最低 rate **下方**扫(配置/脚本改动见 7.6),`run_20260604_175959`,**200 case 全 OK,~3.5h**。round2(低 rate 侧)+ round1(高 rate 侧)合并成完整曲线后:

- **拐点采到了**:8 模型各档的容量拐点 + 性能拐点 + 拐点处延时,见归档 `my_vllm_test_proj/拐点数据_20260604/拐点表_汇总.md`(数据 `summary_merged_round1+2.csv`)。
- **验证 5.9.1 的因果**:**容量拐点 ≤ 性能拐点** 处处成立。
- **投机有效**:**8B EAGLE3 是 TPOT 赢家**——8b-BF16 base TPOT 15ms,比 2B(20ms)还低,只有投机加速才可能(此前 6.7 担心"高并发下投机帮倒忙"被否定)。
- **遗留**:11 个 `<min` 档(多为短输入档 + Qwen3.5-0.8b)容量拐点仍在更低 rate,延时栏取的是已过载点的值;待 round3 往下补齐。异常档 `8b-fp8 4k/1k` 最低点 TTFT 6.5s,严重过载,拐点远低于测试范围。

### 6.9 (2026-06-11)多卡 32B tp4:req_tp 变缓 / goodput 掉破 0.85 都不能当拐点(有限测试 artifact)

第二轮转 Qwen3-32B tp4 集中式(BF16 占卡 0-3、FP8 占卡 4-7 并发,均开 EAGLE3,见 5.7 第二轮),用 tp1 的扫描套路(锚定 sglang rate、向下扫、goodput<0.85 判拐点)跑出来对不上。深挖后发现一个**比"方向反了"更根本的问题**:在重模型 + 短测试下,**req_tp 变缓和 goodput 跌破 0.85 这两个判据,本身就是测量算术造的假象,跟硬件饱和无关**。

**先看现象**(BF16 2k/1k,rate 0.6→2.6):req_tp 一路升 0.52→1.37 但增量递减(像在"压平"到 ~1.4);goodput 一路降 0.87→0.53(rate≈0.7 跨破 0.85);而 **TTFT(~170ms)、TPOT(~22ms)、e2e(~25s)全程平,并发只爬到 35 / 256(14%)**。按 tp1 的判据会判"容量拐点在 rate≈0.7";但延迟、并发都说明系统**根本没被压饱和**。

**根因 = 有限测试 artifact**。测试发 `N=100` 条、每条 e2e≈25s,总时长 ≈ 发包时间 + 排空尾巴 = `N/rate + e2e`,于是:

```
req_tp  = N / (N/rate + e2e)
goodput = req_tp/rate ≈ 1 / (1 + rate·e2e/N)
```

这两个式子**和服务器快慢无关**:只要 rate 升、e2e 固定、N 有限,req_tp 就会弯向天花板 `N/e2e`(=100/25=4),goodput 就会单调下滑。代入"完美无限快服务器"(e2e 恒 25s):它的 goodput 也在 **rate=0.71** 跨破 0.85、req_tp 也弯平——**和我们实测几乎逐点重合**(实测/理想 req_tp = 0.88~1.00)。所以:

- **req_tp 变缓 ≠ 吞吐天花板**:哪怕服务器无限快,只发 100 条、每条活 25s,req_tp 也会弯平。这是"100 条 + 固定排空尾巴"的算术形状,不是能力边界。
- **goodput 跨 0.85 ≠ 容量拐点**:完美服务器也在同一 rate 跨破,这个交点**不含任何饱和信息**。tp1 时它"有时管用"只是因为那些 case 的真饱和点恰好落在 artifact 也开始咬的附近;重模型 + 长 e2e 下真拐点远在上方,artifact 先咬,规则就误报。**0.85 不是普适判据**(此前 5.9.1/5.9.2 把它当主判据,需按本节修正理解)。

**真饱和该看什么(物理信号,本轮全没出现 → 区间内无拐点)**:
- **延迟抬头**:撞墙时每条 e2e 会自己变长 → **TPOT/TTFT 上升**(性能拐点)。本轮 e2e/TPOT/TTFT 全平。
- **并发逼近 `--max-num-seqs`**:在飞并发顶到上限(256)→ 准入排队 → TTFT 暴涨(机制见 5.9.5)。本轮并发才 14%,离墙很远。
- 想用吞吐信号,要看**实测 req_tp 明显偏离"理想服务器 req_tp"曲线**(扣掉 artifact 后的真缺口),而不是看 req_tp 绝对值压平。

**纠正后的多卡找拐点原则**:
1. **判据只认两个物理信号**:延迟(TPOT/TTFT/e2e)抬头 + 并发逼近 max-num-seqs。**别用 req_tp 压平、也别用 goodput<0.85** 当拐点——重模型短测试下它俩都是 artifact。
2. **必须加大 num_prompts**:要让"发包时间 ≫ 排空尾巴"(`N/rate ≫ e2e`)artifact 才消。e2e~25s、rate~2 时需 `N≫50`,实测 100 远不够;**提到 ~500**(sglang 这些模型用的正是 500)。
3. **rate 往上压到延迟真抬头 / 并发真逼近 256 为止**,而不是锚定 sglang 向下扫;粗扫上界宁可给够。

**附带观察**:① FP8 的 TPOT 反而比 BF16 高(FP8 ~29-33ms vs BF16 ~20-25ms,五档稳定复现),反常,疑似 MUSA FP8 反量化开销 / 或 FP8 在卡 4-7 与 BF16 抢卡间互联(本轮并发跑的代价);② 在 rate 1.4 这个 sglang 也测过的点上,vllm_musa 延迟更好(TTFT 172 vs 225ms);吞吐能否对齐要等加大 num_prompts、压到真拐点后才能下结论(本轮 req_tp 受 artifact 压制,不能直接比)。

### 6.10 (2026-06-16~17)PD 分离 ≈ 集中式(只低 ~15%):排查全过程 + 修正结论

第三轮跑 Qwen3-32B 单机 PD 分离(prefill tp4 卡0-3 + decode tp4 卡4-7 + toy proxy,mooncake rdma 搬 KV,num_prompts=500;数据 `my_vllm_test_proj/拐点数据_20260604/run_20260616_144509/`)。初看反常(PD 8 卡像比集中式 4 卡差很多),**逐步排查后修正:PD 其实只比集中式低 ~15%,那个"差很多"是拐点放大的假象。** 排查链值得留作方法论。

**① 第一印象(被 TTFT 误导)**:PD 2k/1k 运营点 TTFT 805ms / TPOT 85ms,集中式才 406ms / 29ms,像差 3 倍;且 PD 多数档在最低 rate 就雪崩。一度判"PD 裸配置不行"。

**② 排查中陆续否掉几个猜测**(都靠数据,不靠猜):
- **闭环并发爬坡探测**(`pd_probe.py`,K=1/8/32/64):并发=1 时 PD 每段都健康、甚至比集中式快(TTFT 176 vs 406ms、decode 25 vs 29ms);KV 搬运全程才几十~两三百 ms;`num_workers` 从 10 调到 32 没用。→ **连接器/mooncake、num_workers 排除**。
- **同 QPS 跨档对比**(用户提出):rate 1.0 下 2k/1k 健康、3.5k/1k 雪崩,**除输入长度外全相同**。proxy 按请求路由、与输入长度无关 → **proxy、连接器配置排除**(它们不随输入变)。
- **开环过载 + 外部观测**(`pd_monitor.py` 抓两实例 `/metrics` 队列 + KV 块占用率 + 卡 util,压 3.5k/1k @1.0 ×500;**注意 KV 用量看 vllm 的 `gpu_cache_usage_perc` 块占用率,不是 gmi 显存——显存启动就预分配满了**):逐点读完整份采样(352 点,每 2s 一行)的时间序列如下:

| 阶段(t 秒) | decode 卡util(4-7) | prefill 卡util(0-3) | KV 填充% | decode 队列 dec_wait | 说明 |
|---|---|---|---|---|---|
| t0-50 起压 | 96-100% | 14-36% | 0→14 | 0 | decode 立刻满载 |
| t50-251 堆积 | **99-100%** | 多在 0-30(常 0) | 29→**100** | 0 | decode 满、KV 爬升、prefill 闲 |
| t251-664 雪崩 | **99-100% 钉死** | 多是 0、偶尔 30 | **100 钉死** | **0→64 一路涨** | KV 撑满后 decode 队列才开始涨 → TTFT 雪崩 |
| t676+ 排空 | 99% | 0 | 100→0 | 归 0 | 收尾排空 |

活跃窗口统计:`dec_util` 均 99 / 峰 100、`pf_util` 均 21、`dec_kv%` 峰 100、`pf_wait` 全程 0、`dec_wait` 峰 64、`dec_run` 峰 176(被 KV 满卡住,未到 max-num-seqs 256)。

**三个铁证**:① decode 卡从头钉 99-100% → decode 算力是硬瓶颈;② KV 在 t≈251 填满 100% 后 `dec_wait` 才开始涨 → 时序证明"decode 太慢 → 序列堆积 → KV 满 → 排队 → TTFT 雪崩";③ prefill 卡均 21%、`pf_wait` 恒 0 → prefill 彻底闲、那 4 卡纯浪费。→ **瓶颈 = decode 计算**;KV 满是 decode 太慢的后果;prefill / proxy / 连接器全无关。

**③ 修正结论(看吞吐而非 TTFT)**:

| 3.5k/1k | 集中式 tp4 | PD 分离 |
|---|---|---|
| req_tp 天花板 | **0.95** | **0.81**(只低 ~15%) |
| 过载段 TPOT(rate≥1.6) | ~185ms | ~185-217ms(几乎相同) |
| rate 1.0 的 goodput | **0.95**(发 1.0 出 0.95,勉强跟上)→ TTFT 566ms | **0.71**(发 1.0 出 0.71,明显跟不上)→ TTFT 48s |

- **两边 decode 速度其实一样**(都过载时 TPOT 都 ~185ms;都是 4 卡 decode,本就该一样);吞吐只差 ~15%。
- **那个几十倍的 TTFT 差是"拐点放大"**:集中式容量 ≈1.0、PD ≈0.81,**rate 1.0 正好卡在两者拐点之间**——对集中式刚好够(goodput 0.95)、对 PD 已超(goodput 0.71);拐点附近容量差一点点,开环过载下 TTFT 就差几十倍。**不是 decode 慢几十倍。**(之前用"运营点 TPOT 29 vs 85ms"说事也是同一 artifact:集中式在拐点下=小批=29ms,PD 在拐点上=大批=85ms。)

**④ 为什么 PD 没赢(那 ~15% 从哪来)**:负载是 **decode 主导**(输出 1024-1536),decode 在 PD/集中式里**都只有 4 卡**;PD 多的 4 张 prefill 卡(21% util)对 decode 瓶颈帮不上忙、纯浪费,还多背搬运/协调开销,且 decode-only 少了集中式"prefill 交错填补 decode 内存带宽空隙"的收益 → 净亏 ~15%。**PD 的主场是 prefill 主导(长输入短输出),我们这批正相反。**

**⑤ 验证"PD 没开投机=放水"的猜想 —— 推翻了:投机对容量(吞吐)无效,过载段甚至有害**。原对比集中式开了 EAGLE3、PD 没开,疑似不公平。给 PD 也开 EAGLE3(同款 draft,`spec=on`)重跑同负载(3.5k/1k @1.0 ×500):

| | 无投机 | 带投机(EAGLE3) |
|---|---|---|
| req_tp | 0.71 | **0.70**(没涨) |
| TTFT 均 | 48s | **80s**(更差) |
| ITL Max | — | 82s(抖动巨大) |
| decode 卡 util | 99% | 98-100%(没缓解) |

投机**确实生效**(日志有 SpecDecoding metrics),但**接受率在过载下崩了**:平均接受长度 2.84→1.57、接受率 61%→18.8%(逐位 0.80/0.61/0.42 → 0.40/0.13/0.04)。机制:**投机是"用额外算力换更少串行步",是低负载/延迟受限时的优化**;PD 在 rate 1.0 已过载、大批(batch ~187)、**算力受限**——每步多验证 3×187 个 draft token、接受率才 19%(draft 3 收 0.5,2.5 个算力纯浪费)→ 吞吐没涨、TTFT 更差。**结论:找拐点/比容量时投机是错的杠杆(低负载降 TPOT、过载降吞吐);集中式那 0.95 也是带投机在过载段拿的,投机对它同样没加成。所以 PD ≈ 集中式低 ~15-25% 是真实 decode 吞吐差,不是投机不公平造成的。**

**⑥ 真正的大差距是 vs sglang,且与 PD 无关**:sglang 在该档 QPS ~3 量级,我们 PD 0.81、集中式 0.95——**sglang 比我们俩都快 ~3-4×**。这是 **vllm_musa 整体 serving/decode 效率比 sglang 低**(kernel/调度),**PD 和集中式一起落后**,不是 PD 架构 / proxy / 连接器的问题。

**方法论留档**:① 别被 TTFT 单点吓到,拐点附近它会放大几十倍,**比能力要看吞吐(req_tp)和过载段 TPOT**;② 定位瓶颈靠**外部观测各实例的队列 + 块占用率 + 卡 util**,别在组件内部插检查点(测不到组件自身排队);③ KV 用量看 `gpu_cache_usage_perc`,不是显存。

**提容量的配置杠杆盘点**:① 投机 EAGLE3 —— **✗ 对容量无效**(见⑤,已验证);② **FP8 权重(`Qwen3-32B-FP8`)—— 未试,唯一还可能有用的**(decode 受显存带宽限,FP8 权重每步读字节减半 → 每步更快 → 容量或可上);③ decode kernel 效率 —— 框架层,调不动。

**遗留/下一步**:PD 对 decode-heavy 负载结论已清楚(≈集中式、无优势,且投机帮不上)。最后可试 **FP8 权重**那轮(`pd_monitor_run.sh` 把 model 换 `Qwen3-32B-FP8`、`spec=off`);若 FP8 也提不动,则定死:**PD/集中式都卡在 vllm_musa 的 decode 效率上,配置层没救,需摩尔线程优化 kernel**(追 sglang 3-4× 的唯一出路)。相关脚本:`pd_probe.py`(闭环爬坡)、`pd_monitor.py`/`pd_monitor_run.sh`(开环外部观测,支持 `spec` on/off)、`pd_smoke_test.sh`(冒烟),均独立、不动主流程。

---

## 七、自动化封装(2026-05-28 初版,06-04 找拐点工具链,06-11 多卡,06-16 PD 分离)

> 本章是在**原交接脚本之上**做的自动化封装。四批改动:
> - **2026-05-28 初版**:解决"手动改路径 / 结果覆盖 / 固定 rate / warmup 不全"四个问题(顶层入口 `auto_bench.py` + 参数文件驱动 + 归档不覆盖)。
> - **2026-06-04 增补(7.6)**:为支撑两轮找拐点(5.9),加 `--param-file`、`AUTOBENCH_NUM_PROMPTS`、`find_knee.py` 和多份参数文件,并修了两个运行期 bug。
> - **2026-06-11 多卡(7.7)**:加 `--tp`,支持 TP>1 部署(每个 server 占 tp 张卡);用于 Qwen3-32B tp4 集中式(见 5.7 第二轮、6.9)。
> - **2026-06-16 PD 分离(7.7)**:加 `--pd`,支持 prefill/decode 分离 + mooncake 搬 KV + proxy 路由;用于 Qwen3-32B 单机 PD(对应 sglang dense 多卡条目②③)。
>
> **与第二节(原始脚本说明)分开看**:第二节讲交接来的原版怎么工作;本章讲在它之上加了什么、改了什么。原版已备份在 `_orig_backup_20260528/`;多卡/PD 改动前的版本备份在 `_backup_20260611_premulticard/`、`_backup_20260616_prePD/`,可随时回滚。

### 7.1 文件构成(在原 bench 目录里新增/修改)

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `auto_bench.py` | ★新增 | 顶层一键入口,串起 启动→就绪→压测→归档→停服务,含 log 监控、ASCII 进度表格、加载进度百分比。**06-04 增** `--param-file`(7.6.2);**06-11 增** `--tp`(多卡);**06-16 增** `--pd`/`--pd-protocol`(PD 分离,7.7);修了 `LOG_DIR`/`params` 两个 bug(7.6.1) |
| `analyze.py` | ★新增 | 结果性能分析:从 24 列算业界派生指标(Goodput/TGS/归一化延迟)+ 评判表 + 雷达图。可配 SLO(`--ttft-slo/--tpot-slo`)。见 5.5.1 |
| `find_knee.py` | ★新增(06-04) | 读 summary 自动定位**容量/性能两个拐点**、生成精扫配置。详见 7.6.3。⚠ 重模型上 goodput<0.85 判据会误报,见 6.9 |
| `toy_proxy_server.py` | ★新增(06-16) | PD 分离的 proxy(官方 example 原样拷入):对外开 8000,把请求 prefill(max_tokens=1)→decode 路由。见 7.7 |
| `tp1_bench_params.json` | ★新增 | **基线**参数文件:8 模型 + 各自 rate(= sglang 发压QPS 原值)+ `extra_args`(投机/并发,见 7.4.1) |
| `tp1_bench_params_round2.json` | ★新增(06-04) | tp1 第二轮低 rate 侧粗扫配置(往 sglang 范围下方扫)。详见 7.6.3 |
| `tp4_bench_params_32b_coarse.json` / `_round2.json` | ★新增(06-11) | Qwen3-32B tp4 集中式(BF16+FP8)粗扫 / round2(num_prompts=500 上探)。见 5.7 第二轮、6.9 |
| `tp_pd_params_32b_bf16.json` | ★新增(06-16) | Qwen3-32B PD 分离粗扫(BF16,rate 上探)。配 `--pd` 用 |
| `pd_smoke_test.sh` | ★新增(06-16) | PD 冒烟测试:拉 prefill+decode+proxy 跑一条 curl,验证 mooncake 在 MTT 上通不通(独立于 auto_bench)。见 7.7 |
| `run_list.py` | 改 | ① `MODELS` 环境变量注入 ② `EXTRA_ARGS` per-model 额外参数 ③ **06-11** `AUTOBENCH_TP` 多卡卡分配 ④ **06-16** `AUTOBENCH_PD` PD 模式(`launch_pd`:prefill/decode/proxy)。原默认行为保留为 fallback |
| `test_list.py` | 改 | ① per-model cases ② warmup 覆盖所有输入形状 ③ **06-04** `AUTOBENCH_NUM_PROMPTS`(7.6.2)④ **06-16** PD 下预检改打 `/healthcheck`(proxy 无 `/v1/models`) |
| `bench_serving.py` | 不动 | 压测引擎,原样(PD 也走它,经 proxy 打 `/v1/completions`) |
| `_orig_backup_20260528/` | ★新增 | 原版脚本备份 |
| `_backup_20260611_premulticard/` / `_backup_20260616_prePD/` | ★新增 | 多卡 / PD 改动前的脚本备份(回滚用) |
| `tp1_bench_params.json.20260604bak` | ★新增(06-04) | 基线参数文件的备份(改 rate 前留存) |

> **雷达图归一化(方向 D)**:`analyze.py` 的雷达图用**绝对刻度 + 对数映射**(固定端点,非"除以全模型最大值")——吞吐类(EffCapacity 0.1→4 req/s、PeakTGS 200→2000)用对数压缩跨度,这样大小模型都落在真实位置、且跨数据集可比(以后叠加 sglang 数据可直接对比)。5 维中英文对照见 `analyze.py` 开头 docstring。
>
> **⚠ 排错记录(draft model 路径)**:xlsx 里 sglang 用的 EAGLE3 draft 路径 `/mnt/seed17/001688/yaoxi/Qwen3-8B_eagle3` **已失效**;实际可用的在 `/mnt/seed17/001688/models/Qwen3-8B_eagle3`。本批仅 Qwen3-8B 有 draft(8B-FP8 共用),其余无。

### 7.2 调用层次(修改版)

> 下图是**单轮**的调用链(启动→压测→归档)。参数文件默认 `tp1_bench_params.json`,可用 `--param-file` 指定;**两轮找拐点是把这条链跑两遍 + 中间用 `find_knee.py` 定位**,完整流程见 7.6.4。
>
> **三种部署模式**(决定 ① 启动服务这步的拓扑,见图下方对比;压测/归档/停服流程一致):
> - **单卡**(默认):N 个模型 → N 个 server,第 i 个占卡 i、端口 8000+i。
> - **多卡 tp**(`--tp T`):每个 server 占 T 张卡 [i·T, i·T+T);如 BF16+FP8 各 tp4 占满 8 卡(端口 8000/8001)。见 7.7。
> - **PD 分离**(`--pd --tp T`):1 个模型拆 prefill(卡[0,T),producer)+ decode(卡[T,2T),consumer)+ proxy;对外只有 proxy 8000。见 7.7。

```
【第 0 阶段:参数设置(测前一次性,人工/AI 做,不是脚本)】
  SGLang v0.5.6-post2 Release文档.xlsx  ◀── 参数来源(sglang 各模型的 TP/发压QPS/长度档/启动指令)
     │  人工(或让模型)按测试目标提取:
     │    · 筛出本批要测的模型(如 TP=1 的 8 个 dense)
     │    · 每个模型的 io 长度档(input/output,必须与 sglang 对齐才可比)
     │    · 每个模型的 rate 序列(= sglang 发压QPS;找拐点时再向下/向上扩,见 5.9)
     │    · 每个模型的启动配置 extra_args(tp / 投机 / 并发上限,见 7.4.1)
     ▼  写成参数文件 ↓
  tp1_bench_params.json   ◀── 参数文件(脚本的唯一输入;换批测试就换/改这个文件)
     │   ⚠ 这一步没有自动脚本:xlsx → 参数文件是人工/AI 完成,
     │      auto_bench 只负责"读已写好的参数文件"并照着跑。
     ▼
─────────────────────────────────────────────────────────────
auto_bench.py  ★顶层封装(新增入口)
  │  读参数文件(默认 tp1_bench_params.json,--param-file 可指定 round2/fine)
  │
  ├─① 启动服务:设环境变量 AUTOBENCH_MODELS / AUTOBENCH_TP / AUTOBENCH_PD …
  │     └─ 调 run_list.py --no-wait(拓扑随模式不同,见下方对比图)
  │           └─ run_list 起 vllm serve(单卡/多卡)或 prefill+decode+proxy(PD)
  │           └─ 日志 run_list.nohup.log
  │
  ├─② 等就绪:轮询 run_list.py status,监控 run log 抓崩溃关键字
  │           (Traceback / CUDA error / MUSA error / OOM / 端口占用 …)
  │           PD 模式 status 改查 proxy /healthcheck(proxy 无 /v1/models)
  │
  ├─③ 跑压测:设环境变量 AUTOBENCH_PARAM_FILE=参数文件
  │     └─ 调 test_list.py <result_dir>
  │           └─ test_list 读参数文件 → 每个模型按 model_path 取自己的 cases
  │           └─ 先对每个模型的所有输入形状各 warmup 一遍(★修复点)
  │           └─ 再 ThreadPoolExecutor 并行,每 case 调 bench_serving.py
  │           └─ 日志 test_list.nohup.log(完成时输出 [done])
  │           └─ 监控 test log 抓错误
  │
  ├─④ 归档:结果写 bench_results/run_<时间戳>/(★不覆盖)
  │           + params_snapshot.json(本批参数快照,便于追溯)
  │
  └─⑤ 停服务:调 run_list.py stop(--keep-alive 可跳过)
```

**①「启动服务」这步三种拓扑对比**(其余步骤一致):

```
单卡(默认)        多卡 tp(--tp 4)               PD 分离(--pd --tp 4)
─────────         ──────────────────           ──────────────────────────
模型A→卡0:8000    模型A(BF16)→卡0-3:8000       1 个模型拆两角色:
模型B→卡1:8001    模型B(FP8) →卡4-7:8001         prefill 卡0-3 :8100 (producer)
  ⋮                 (各 1 个 tp4 server)            decode  卡4-7 :8200 (consumer)
模型H→卡7:8007                                      proxy        :8000  ← test_list 打这里
                                                          │
test_list 打       test_list 打                    bench_serving→proxy:8000/v1/completions
8000+idx           8000+idx(2 个)                    proxy: 先 prefill(max_tokens=1)→ mooncake 搬 KV → decode 流式
                                                    KV 传输:MooncakeConnector(P2P 握手,rdma/tcp)
```

### 7.3 相对原版改了什么(逐条对应已知问题)

| # | 原版问题 | 修改 | 改动位置 |
| --- | --- | --- | --- |
| 1 | 每次手动改 `MODELS` 路径 | 参数文件驱动,环境变量注入 | `run_list.py` MODELS + `auto_bench.py` |
| 2 | `bench_results` 被下一次覆盖 | 每次归档到 `run_<时间戳>/` 子目录 | `auto_bench.py` |
| 3 | 所有模型共用一套固定 rate(慢模型过载/快模型压不满,见 6.3) | per-model rate 序列,按 model_path 匹配 | `test_list.py` `CASES_BY_PATH` |
| 4 | warmup 只热一种输入形状,每个长度档首 case 偏慢(见 6.2) | 正式测前对每个输入形状各预热一遍 | `test_list.py` `run_one_model` |
| 5 | 无统一入口、无错误检测 | 顶层封装 + log 监控抓崩溃关键字 | `auto_bench.py` |
| 6 | 只支持单卡(一模型一卡) | `--tp T`:每 server 占 T 卡 [i·T, i·T+T);卡数守卫 模型数×T≤8 | `run_list.py` `AUTOBENCH_TP` + `auto_bench.py`(见 7.7) |
| 7 | 不支持 PD 分离 | `--pd`:`launch_pd` 起 prefill+decode+proxy、`build_cmd` 加 `--kv-transfer-config`、就绪查 proxy `/healthcheck` | `run_list.py` `AUTOBENCH_PD` + `test_list.py` + `auto_bench.py`(见 7.7) |

> 所有改动在文件内用 `# AUTOBENCH PATCH ... # END AUTOBENCH PATCH`(多卡/PD 是 `AUTOBENCH PD PATCH 2026-06-16`)标记,且向后兼容:不设环境变量时,`run_list.py`/`test_list.py` 行为与原版一致。

### 7.4 用法

> **先改配置,再跑指令**。下面的命令都不用动;要调测试内容,改对应的文件即可(改完直接重跑 `auto_bench.py`,无需动脚本):
>
> | 想改什么 | 改哪个文件 / 字段 | 说明 |
> | --- | --- | --- |
> | 投机解码(EAGLE3 开/关) | `tp1_bench_params.json` → 某模型的 `extra_args` 加/删 `--speculative-config` | 见 7.4.1;需 draft model |
> | 并发上限 `--max-num-seqs` | `tp1_bench_params.json` → 某模型的 `extra_args` 里改数值 | 见 7.4.1 |
> | 其它 vllm 启动参数(per-model) | `tp1_bench_params.json` → `extra_args` | 任意 `vllm serve` 参数,拼到命令尾部 |
> | 每个模型的 rate 序列 | `tp1_bench_params.json` → 某模型的 `cases`(每行第 4 个数) | rate=QPS,见 5.3 |
> | 输入/输出长度档(io_label) | `tp1_bench_params.json` → `cases`(每行前 3 个:input/output/label) | 见 5.2 |
> | 测哪几个模型 | 命令行 `--models <served_name...>`,或改 `tp1_bench_params.json` 顶层键 | served_name 见各模型 `served_name` 字段 |
> | 全局启动参数(8 卡通用) | `run_list.py`:`GPU_MEM_UTIL` / `BLOCK_SIZE` / `BASE_PORT` 等 | 所有模型统一,非 per-model |
> | **多卡 tp 部署** | 命令行 `--tp 4` | 每 server 占 4 卡;模型数×tp≤8。见 7.7 |
> | **PD 分离** | 命令行 `--pd --tp 4`(+ `--pd-protocol rdma/tcp`) | 1 模型拆 prefill+decode+proxy,占 2×tp 卡。见 7.7 |
> | SLO 阈值 / 雷达图 | 跑 `analyze.py --ttft-slo --tpot-slo --radar`(分析阶段,不在本流程) | 见 5.5.1 |
>
> 一句话:**per-model 的东西(投机、并发、rate、长度)全在 `tp1_bench_params.json`;全局通用的启动参数在 `run_list.py`。**
>
> **文件绝对路径(都在 bench 目录下)**:
> - 参数文件:`/mnt/seed17/001688/models/Qwen/bench/tp1_bench_params.json`
> - 启动脚本:`/mnt/seed17/001688/models/Qwen/bench/run_list.py`
> - 顶层入口:`/mnt/seed17/001688/models/Qwen/bench/auto_bench.py`
> - 分析脚本:`/mnt/seed17/001688/models/Qwen/bench/analyze.py`

```bash
cd /mnt/seed17/001688/models/Qwen/bench
PY=/root/.virtualenvs/sglang-0.5.6/bin/python

# 预览将要跑什么(不启动服务)
$PY auto_bench.py --dry-run

# 跑全部 8 个模型(后台 + 看日志)
nohup $PY -u auto_bench.py > auto_bench.nohup.log 2>&1 &
tail -f auto_bench.nohup.log

# 只跑指定模型(按 served_name)
$PY auto_bench.py --models qwen3.5-0.8b qwen3-14b

# 压测后保留服务不停(便于手动复测)
$PY auto_bench.py --keep-alive

# 多卡 tp4(如 Qwen3-32B BF16+FP8 各占 4 卡,见 7.7)
AUTOBENCH_NUM_PROMPTS=500 $PY auto_bench.py --param-file tp4_bench_params_32b_round2.json --tp 4

# PD 分离(1 模型,prefill 卡0-3 + decode 卡4-7 + proxy,见 7.7)
AUTOBENCH_NUM_PROMPTS=500 $PY auto_bench.py --param-file tp_pd_params_32b_bf16.json --models qwen3-32b-bf16 --pd --tp 4
```

结果在 `bench_results/run_<时间戳>/`:每模型 CSV + `benchmark_summary_*.csv` + `bench_logs/<模型>.log` + `params_snapshot.json`。

#### 7.4.1 per-model 启动参数(投机解码 / 并发上限)

每个模型可带自己的额外 vllm 启动参数,配在 `tp1_bench_params.json` 的 **`extra_args`** 字段(字符串数组)。`auto_bench.py` 注入环境变量 `AUTOBENCH_EXTRA_ARGS`,`run_list.py` 的 `build_cmd` 按 model_path 拼到 `vllm serve` 命令尾部。

**本期配置(2026-06-03,对齐 sglang)**:

| 模型 | extra_args |
| --- | --- |
| 全部 8 个 | `--max-num-seqs 256`(对齐 sglang `--max-running-requests 256`,解开 ~60 并发瓶颈) |
| Qwen3-8B / 8B-FP8 | 额外 `--speculative-config '{"method":"eagle3","model":"/mnt/seed17/001688/models/Qwen3-8B_eagle3","num_speculative_tokens":3}'` |

**开启方法(改 `extra_args` 即可,无需动脚本)**:

```jsonc
// tp1_bench_params.json 里某模型条目
"Qwen3-8B-BF16": {
  "model_path": "/mnt/seed17/001688/models/Qwen3-8B",
  ...
  "extra_args": [
    "--max-num-seqs", "256",
    "--speculative-config",
    "{\"method\":\"eagle3\",\"model\":\"/mnt/seed17/001688/models/Qwen3-8B_eagle3\",\"num_speculative_tokens\":3}"
  ]
}
```

**关键约束**:
- **投机解码需要 draft model**。本批只有 `Qwen3-8B` 有 EAGLE3 draft(`/mnt/seed17/001688/models/Qwen3-8B_eagle3`),8B-FP8 共用之;14B / Qwen3.5 系列**无 draft,不能开**(留空 speculative 即可)。
- `--speculative-config` 的值是 **JSON 字符串**:`method` 用 `eagle3`,`model` 填 draft 路径,`num_speculative_tokens` 是一次猜几个 token(草稿步数,常用 3)。
- `--max-num-seqs` 是请求条数上限;配合 `--max-num-batched-tokens`(token 预算,见第四章)共同决定并发。
- 改 `extra_args` 后直接重跑 `auto_bench.py`,无需改 `run_list.py`/`test_list.py`。

### 7.5 回滚

```bash
cd /mnt/seed17/001688/models/Qwen/bench
cp _orig_backup_20260528/run_list.py  run_list.py
cp _orig_backup_20260528/test_list.py test_list.py
```
(`auto_bench.py` / `tp1_bench_params.json` 是新增文件,删掉即可,不影响原版运行。)

### 7.6 两轮找拐点:脚本改动 + 新增文件详解(2026-06-04)

为支撑 5.9 的两轮找拐点方法,在原封装之上做了以下改动。**全部向后兼容**:不传 `--param-file` / 不设环境变量时,行为与之前一致。

#### 7.6.1 Bug 修复(运行中发现,#12/#14 进度显示代码遗留)

| 文件 | 问题 | 修复 |
| --- | --- | --- |
| `auto_bench.py` | `load_progress()` 引用未定义的 `LOG_DIR` → 等就绪时崩 | 补 `LOG_DIR = BENCH_DIR / "run_logs"` |
| `auto_bench.py` | `run_benchmark()` 用了 `params` 但函数没收该形参 → 压测一开始 `NameError` | 加 `params` 形参,`main` 调用处传入 |

> 已用 AST 静态检查器扫 `auto_bench.py`/`run_list.py`/`test_list.py`,确认无其它未定义名字。

#### 7.6.2 新增能力

| 改动 | 文件 | 说明 |
| --- | --- | --- |
| `--param-file <json>` | `auto_bench.py` | 指定本次用哪个参数文件(默认 `tp1_bench_params.json`)。两轮各用各的 json,**不必覆盖原文件**。 |
| `AUTOBENCH_NUM_PROMPTS` 环境变量 | `test_list.py` | 覆盖 num-prompt(默认 100)。粗扫想加速可调小;不设则同原版。用法:`AUTOBENCH_NUM_PROMPTS=40 $PY auto_bench.py ...` |

#### 7.6.3 新增文件详解

**① `find_knee.py`(★找拐点自动定位 + 生成精扫配置)**

读一份 benchmark_summary CSV,对每个(模型 × 输入档)分别检测两个拐点(定义见 5.9.1):
- **容量拐点**:goodput 比 `req_tp/request_rate` 跌破 `--gr-thresh`(默认 0.85)的区间;
- **性能拐点**:`mean_tpot` 相对健康区基线跳变 > `--tpot-jump`(默认 1.8)倍的区间;
- `--knee throughput|latency|both`:选精扫哪个拐点(默认 both,两个都报、各精扫一簇);
- 区间状态:`ok`(夹住了)/ `below_min`(拐点比最低 rate 还低,需向下延伸)/ `above_max`(全程未触发,需向上延伸);
- `--emit out.json`:把每个拐点区间内**等间隔**密扫的 rate 写成精扫参数文件;
- 注意:CSV 的 `model_name` 列 = `model_path` 的 basename 小写(如 `qwen3-8b`),可能与参数文件的 `served_name` 字段(如 `qwen3-8b-bf16`)不同,`find_knee` 按 basename 匹配,避免漏模型。

用法:
```bash
$PY find_knee.py <summary.csv> --knee both                 # 只打印两拐点报告
$PY find_knee.py <summary.csv> --knee both --emit fine.json # 同时生成精扫配置
```

**② 参数文件家族(都是 `auto_bench.py --param-file` 的输入,格式同 `tp1_bench_params.json`)**

| 文件 | 角色 | rate 怎么排 |
| --- | --- | --- |
| `tp1_bench_params.json` | **基线(= sglang 发压QPS 原值)** | 每模型每档照搬 sglang 的 rate。上一轮 run_20260603 用它,证明拐点在此范围**之下**。 |
| `tp1_bench_params_round2.json` | **第二轮 / 低 rate 侧粗扫** | 每模型每档:从 sglang 最低 rate `smin` **往下**对数扫到 `max(0.1, smin×0.5)`,5 点。目的:把"健康→过载"的拐点夹在中间(顶点 smin 已知过载,底部探健康)。 |
| `tp1_bench_params_coarse.json` | (备用)统一盲扫 | 8 模型统一对数 rate 0.1~6.0 × 10 点。**已弃用**——后来改成锚定 sglang 往下扫(round2),更省、更可比。 |
| `_fine.json`(由 `find_knee --emit` 生成) | **第三轮 / 精扫** | 只在 find_knee 定位出的拐点区间内等间隔密扫。文件名自取。 |

> `model_dir` / `served_name` / `model_path` / `extra_args`(投机 + `--max-num-seqs 256`)这些每模型元数据,各 json 都从基线继承、保持一致;**不同的只有 `cases` 里的 rate**。

#### 7.6.4 两轮(可三轮)完整流程

```bash
cd /mnt/seed17/001688/models/Qwen/bench
PY=/root/.virtualenvs/sglang-0.5.6/bin/python

# 第二轮:低 rate 侧粗扫(num-prompt 默认 100)
nohup $PY -u auto_bench.py --param-file tp1_bench_params_round2.json > auto_bench.round2.log 2>&1 &

# 合并 round1(高侧)+ round2(低侧)的 summary → 完整曲线,再定位拐点
$PY find_knee.py <合并后的summary.csv> --knee both --emit tp1_bench_params_fine.json

# 第三轮(可选):精扫,把拐点定准
nohup $PY -u auto_bench.py --param-file tp1_bench_params_fine.json > auto_bench.fine.log 2>&1 &
```

#### 7.6.5 回滚

新增文件(`find_knee.py` / `tp1_bench_params_round2.json` / `_coarse.json` / `_fine.json`)删掉即可;`tp1_bench_params.json` 未改(另存 `.20260604bak`);`auto_bench.py`/`test_list.py` 的改动向后兼容,不影响原用法。

### 7.7 多卡 / PD 分离支持(2026-06-11 / 06-16)

从 tp1 单卡转向 Qwen3-dense 多卡(Qwen3-32B,见 5.7 第二轮)。sglang dense 多卡只有两种部署:**tp4 集中式** 和 **单机 PD 分离**,本节支持这两种。所有改动标 `# AUTOBENCH PD PATCH 2026-06-16`,改前已备份(`_backup_20260611_premulticard/`、`_backup_20260616_prePD/`)。

#### 7.7.1 多卡 tp(`--tp T`,06-11)

- **run_list.py**:`TP` 改读 `AUTOBENCH_TP`(默认 1);第 i 个 server 占卡 `[i·T, i·T+T)`(原来每 server 只给 1 张)。`--tensor-parallel-size` 由 `TP` 驱动,**别再往 `extra_args` 塞 `--tensor-parallel-size`**(会重复冲突)。
- **auto_bench.py**:`--tp` 注入 `AUTOBENCH_TP`;卡数守卫从"模型数≤8"改为"模型数×tp≤8"。
- 用法:`--tp 4` + 参数文件放 1~2 个模型(2 个 tp4 = 满 8 卡)。test_list 仍打 `8000+idx`,无需改。
- ⚠ 两组 tp4 并发会共享卡间互联(allreduce),与 sglang 单部署条件不完全对齐,结论需标注(见 6.9)。

#### 7.7.2 PD 分离(`--pd`,06-16)

vllm_musa 自带 PD 方案:**MooncakeConnector**(`vllm_musa/distributed/kv_transfer/.../mooncake_connector.py`)+ 官方 example。三件套:

| 组件 | 启动 | 卡 / 端口 |
| --- | --- | --- |
| prefill 实例 | `vllm serve … --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer",…}'` | 卡 [0,T)、8100 |
| decode 实例 | 同上,`"kv_role":"kv_consumer"` | 卡 [T,2T)、8200 |
| proxy | `toy_proxy_server.py --prefiller-port 8100 --decoder-port 8200 --port 8000` | 8000(对外) |

- **数据流**:bench_serving → proxy:8000/`v1/completions` → prefill(强制 max_tokens=1)→ **mooncake 搬 KV** → decode 流式返回。proxy 用 `kv_transfer_params` 在两实例间传 block 信息。
- **mooncake**:`P2PHANDSHAKE` 初始化(**不需要外部 metadata/master 服务**);协议默认 `rdma`(MTT 有 IB,sglang 也走它),可 `--pd-protocol tcp` 兜底。
- **改动**:
  - `run_list.py`:`AUTOBENCH_PD=1` → `launch_pd()` 起 prefill/decode/proxy,三 pid 都写 `pids.txt`(`stop` 一并杀);`build_cmd(…, kv_role=)` 追加 `--kv-transfer-config`;`status`/`wait_ready` 改查 prefill+decode(`/v1/models`)+ proxy(`/healthcheck`)。
  - `test_list.py`:PD 下预检改打 proxy 的 `/healthcheck`(proxy 无 `/v1/models`);压测请求不变(走 `/v1/completions`),`run_one_model` 打 8000=proxy。
  - `auto_bench.py`:`--pd`/`--pd-protocol` 注入 `AUTOBENCH_PD`/`AUTOBENCH_PD_PROTOCOL`;守卫"PD 只能 1 模型 + 2×tp≤8"。
- **冒烟验证**:`bash pd_smoke_test.sh [model] [rdma|tcp]` —— 独立拉起三件套跑一条 curl,确认 mooncake 在 MTT 上通不通(末行打 ✅/❌,日志在 `pd_smoke_logs/`)。
- **限制 / TODO**:① PD 一个模型就吃满 8 卡,不能像集中式那样 BF16+FP8 并发;② 当前 PD 不开投机(对应 sglang 条目②基础 PD);prefill 开 EAGLE 的变体③后续加;③ rate 序列见 `tp_pd_params_32b_bf16.json`(PD 吞吐应高于集中式,rate 上探)。

### 7.8 PD 诊断脚本(2026-06-16~17,定位 PD 瓶颈)

为查清 PD 为何不及集中式(过程与结论见 6.10),写了一组**独立诊断脚本**——都不动 benchmark 主流程、不改源码,可单独跑、可删。设计上分三层,从"能不能跑"到"卡在哪一段":

| 脚本 | 类型 | 设计 / 测什么 |
|---|---|---|
| `pd_smoke_test.sh` | 冒烟 | 拉 prefill+decode+proxy 跑一条 curl,验证 mooncake KV 搬运在 MTT 上通不通(独立于 auto_bench)。支持 `rdma`/`tcp` 切换 |
| `pd_probe.py` + `.sh` | **闭环**拆解 | 固定并发 K=1/8/32/64,各打一轮,把单请求拆成 **prefill / 搬运+decode首token / decode** 三段计时;带 `num_workers` A/B(默认 10 vs 32)。**闭环=固定在飞并发**,看每段随并发怎么涨,排除连接器/`num_workers` 用。⚠ 闭环不会复现开环雪崩(并发有上限),只看分段 |
| `pd_monitor.py` + `pd_monitor_run.sh` | **开环**外部观测 | 真 `bench_serving` 发开环过载负载(默认 3.5k/1k @1.0 ×500),期间每 2s 抓 **prefill(8100)/decode(8200) 的 `/metrics` 队列(`num_requests_waiting/running`)+ KV 块占用率(`gpu_cache_usage_perc`)+ 卡 util(mthreads-gmi 的 %GPU)**,逐点落带时间戳的 CSV。**只观测、不在组件内部插检查点**(避免测不到组件自身排队);判读靠人工读 CSV 时序,程序不下结论 |

**关键设计取舍(踩过的坑,留记录)**:
- **为什么不在 proxy/连接器内部插检查点**:若瓶颈是组件自身排队(事件循环/线程池队列),排队发生在检查点括住的范围之外 → 会测出"各段都正常"的假象。改为**从外部抓每个实例自己的队列计数**,天然包含排队;且"两实例都不忙却慢"就反证是 proxy。
- **KV 用量必须看 `gpu_cache_usage_perc`(块占用率),不能看 gmi 显存**:vllm 启动时按 `gpu-memory-utilization` 把 KV cache 池**一次性预分配占满**,显存 MiB 从头到尾不变,反映不了实际填了多少;块占用率才是 0→100% 随负载变的真信号。
- **看能力要看吞吐 req_tp + 过载段 TPOT,别被单点 TTFT 吓到**:拐点附近 TTFT 会放大几十倍(开环过载队列无限涨),容量只差 15% 也能让 TTFT 差几十倍。
- **要复现开环雪崩得用真 `bench_serving`(开环、num_prompts 够大)**,闭环探测复现不了(并发有上限,堆不出无限 backlog)。

---

## 八、当前进度与待办(2026-06-03)

**已完成**:
- 自动化封装 `auto_bench.py`(per-model rate + warmup 多形状修复 + 结果不覆盖归档 + ASCII 进度表格 + 加载进度百分比 + log 监控)
- 性能分析 `analyze.py`(Goodput/TGS/归一化延迟 + 可配 SLO + 雷达图)
- 首轮实测(2026-06-02):8 模型 241 case 全 OK,但 87% 过载;经分析查清根因(6.4/6.5/6.6)
- 投机+并发配置就绪(7.4.1):全部 `--max-num-seqs 256`,8B/8B-FP8 加 EAGLE3 投机,已部署待重测

**待办(下一步)**:
- **重测(#16)**:用已就绪的"开投机 + 256 并发"配置重跑一次,**保持原 rate**(理由见 6.6:现有 rate 数据被并发上限污染,放开并发后假过载模型会显出真实曲线,可能需上调 rate;真过载的 14B 维持低 rate 健康区)。重测后用 `analyze.py` 出评判表 + 雷达图。
- 重测前可先单起 Qwen3-8B + EAGLE3 验证投机能跑通(没踩过这个组合)。
- 重测后据真实拐点二次调整 per-model rate 序列。

**关键环境信息**:4.127 容器 bench 目录 `/mnt/seed17/001688/models/Qwen/bench`;venv `/root/.virtualenvs/sglang-0.5.6`;vllm_musa 0.20(V1 引擎,torch2.9);8 卡 S5000。EAGLE3 draft = `/mnt/seed17/001688/models/Qwen3-8B_eagle3`。


