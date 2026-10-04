# 工作记录梳理（2026.04 – 2026.06）

> 本文档按时间顺序梳理近几个月的工作，简要介绍每个阶段做的事情、产出与结论。
> 资料来源：`work_record/work_record.txt`、`work_record/file_guide.txt`、`vllm_musa_proj/reademe.md`、
> `ISSUE_vllm_musa_broadcast_deadlock.md`、`ISSUE_vllm_musa_broadcast_deadlock2.md`，以及
> `work_record/rtp_llm/`、`work_record/exllamav2/` 下的报告。

---

## 总览

这几个月的工作大致分为四条主线，时间上前后衔接：

1. **大模型推理框架选型与基准测试**（RTP-LLM）—— 4 月初
2. **exllamav2 在国产卡（MTT S5000）上的性能排查与算子优化** —— 4 月中下旬
3. **若干视觉库的验证与 demo** —— 4 月下旬（穿插）
4. **vLLM MUSA 移植版（vllm_musa）的 bug 排查 + 性能基准 + LMCache 集成** —— 5 月起至今（主线，占比最大）

核心阵地是**摩尔线程 MTT S5000（arch 310 / PH1）8 卡平台**上的大模型推理优化。

---

## 一、4.07 – 4.09　RTP-LLM 框架选型与 Qwen3 基准测试

> 详见 `work_record/rtp_llm/` 下的两份 benchmark 报告（报告日期 4.09）。
> 任务：选定一个大模型推理框架做性能验证，最终选用阿里的 **alibaba-rtp-llm**，在 **NVIDIA A30（24GB）** 上跑 Qwen3 系列基准。

### 1. 环境搭建（踩坑过程）

- 完成服务器环境配置、agent 设置，确定 rtp-llm v0.2.0 + torch 2.6.0+cu126 + CUDA 12.6 + Ubuntu 22.04 Docker 的技术栈。
- **主要障碍**：拉取 alibaba-rtp-llm 库时无法连接阿里云服务器。绕过办法是用旧版镜像 + whl 包更新镜像，过程中逐个解决依赖包版本冲突，最终把环境配通。

### 2. 基准测试设计

- 数据集 **ShareGPT_V3**（多轮对话，输入输出长度由真实对话决定），`num-prompts=100`、`request-rate=inf`（一次性发完，压极限吞吐）、`max-batch-size=32`、贪心解码（top_k=1）、FP16、单卡 TP=1。
- 覆盖 **Qwen3-0.6B / 4B / 8B** 三档参数量，形成性能梯度对比；并对 0.6B 补了单并发测试。

### 3. 结果与分析

| 指标 | Qwen3-0.6B | Qwen3-4B | Qwen3-8B |
|---|---:|---:|---:|
| 请求吞吐 | 11.56 req/s | 3.57 req/s | 2.16 req/s |
| 总 token 吞吐 | 4,513 tok/s | 1,395 tok/s | 843 tok/s |
| 平均请求延迟 | 1.16 s | 3.55 s | 5.70 s |
| 每输出 token 延迟 | 8.01 ms | 25.73 ms | 40.05 ms |

- **核心规律**：吞吐下降幅度始终**小于**参数量增长幅度（0.6B→8B 参数涨 13.3×，吞吐只降 5.4×；4B→8B 参数翻倍，吞吐仅降 1.7×）——说明 rtp-llm 的 batching 优化在大模型上效率更高，GPU 计算密度提升带来效率增益。
- **显存评估**：A30 24GB 下 0.6B（~1.2GB）/ 4B（~8GB）显存绰绰有余；8B（~16GB 权重）单卡可跑但留给 KV Cache 的空间只剩 ~8GB，高并发场景偏紧。
- **结论分档**：0.6B 吞吐极高但质量有限（适合轻量/嵌入式）；4B 吞吐与质量较均衡；8B 质量最佳，高并发下建议 INT8/INT4 量化或多卡 TP。

> 这一阶段的 ShareGPT 数据集压测、`request-rate` 扫描、吞吐/延迟指标体系，为后面 6 月的 vllm_musa 系统性基准测试（第六章）打下了方法论基础。

---

## 二、4.14 – 4.16　exllamav2 在 MTT S5000 上的性能排查与算子优化（重点）

> 详见 `work_record/exllamav2/` 下的 benchmark / profiling / optimization 三份报告（报告日期 4.09 起）。
> 任务：把量化 LLM 推理框架 **exllamav2** 在 **MTT S5000（PH1 / mp_31）** 上跑起来并对齐 NVIDIA A30 的性能，
> 测试模型为 Llama-3.1-8B / Qwen3-1.8B 的 GPTQ INT4 量化版。

### 1. 现状基准（exllamav2_benchmark_report）

完成 MUSA 移植后做了端到端基准对比，结论是**移植可用但性能与稳定性差距明显**：

- **推理速度**：S5000 仅为 A30 的 **17–21%**（128–192 token 区间约 9.3 t/s vs A30 的 44–57 t/s）。
- **模型加载**：S5000 约为 A30 的 **4.3 倍**（15s vs 3.5s）。
- **稳定性**：S5000 生成超过 192 token 后出现 **Segfault / 卡死（GPU 利用率 0%）**，最大稳定长度仅 192 token；A30 可稳定生成 1024+。
- **计算精度**：两平台 GEMM 内核精度一致，24/24 形状全部通过，数值正确性无问题（说明问题不在计算正确性，而在性能/稳定性）。
- 顺带修了一个兼容性问题：torch_musa 2.5.0 不支持 `is_privateuseone()`，改用 `.is_musa` 属性。

### 2. 瓶颈定位（exllamav2_profiling_report，module 级 → kernel 级）

用 monkey-patch 插桩做了**逐层 / 逐 kernel** 的剖析，层层下钻定位根因：

- **Module 级**：MLP 慢 **12.6×**（占 S5000 总耗时 73.2%），是最大瓶颈；Attention 慢 4.5×；而走 muBLAS 的 **Head_Linear 反而 S5000 更快（0.6×）**——说明 **BLAS 库本身没问题**。
- **MLP 内部拆 7 步**：瓶颈精确落在三个量化矩阵投影 **gate_proj / up_proj / down_proj（各慢 10–11×）**，它们走的是 exllamav2 自定义的 `q_gemm_kernel_gptq` kernel。
- **组件级 micro-benchmark**（关键矛盾）：单独测时 S5000 全面占优——裸 muBLAS hgemm 快 1.2–1.6×、裸 reconstruct（INT4→FP16）快 1.8×、大 kernel 纯执行基本持平；**唯独 kernel launch+sync 固定开销慢 4.3×**（每次多 ~50–60us）。
- **根因**：① 自定义 GPTQ fused kernel 用的 NVIDIA intrinsics（`__hfma2`、`atomicAdd on half2`、INT4 128-bit 向量化 load）在 MUSA PH1 架构上严重退化，效率仅理论值 **1%**；② 一次 decode forward 要发射 300+ 个 kernel，**launch overhead 累积**成第二大瓶颈。

### 3. 优化措施与成效（optimization_report，Qwen3-1.8B GPTQ INT4）

基于上述定位，做了两步代码改动（均在 `q_gemm.mu`），decode 吞吐 **9.07 → 21.21 t/s（2.34×）**：

- **优化 1 — decode 绕过自定义量化 kernel，改走 muBLAS 路径**：把 `size_m > MAX_Q_GEMM_ROWS(32)` 的阈值放开为 `size_m >= 1`，让 decode（M=1）也走 "reconstruct(INT4→FP16) + muBLAS hgemm"。**吞吐 9.07 → 15.42 t/s（+70%）**，MLP 提速 2.1×。
- **优化 2 — M=1 时用 `at::mm` 替代 `mublasHgemm`**：C++ musaEvent 插桩发现同形状下 `mublasHgemm` 比 `torch.mm` 慢 3.4×；用 profiler 追踪发现 `torch.mm` 对 M=1 会自动 dispatch 到 MUSA DNN 的专用 `batch_gemv` kernel，而 `mublasHgemm` 走通用 GEMM kernel、对 M=1 严重低效。于是对 `size_m <= 2` 改用 `at::mm`。**吞吐 15.42 → 21.21 t/s（+37.5%）**，MLP -21%、Attention -35%。

### 4. 当前差距与后续方向

- 优化后 S5000 ≈ A30 的 **2.07× 慢**（MLP 4.7×、Attention 2.9×），剩余差距主要来自：3× reconstruct 的额外开销（A30 走 fused kernel 不需要）、launch overhead 累积、以及 **S5000 无 FlashAttention（走 O(n²) 标准 attention）**。
- 后续方向（正交两条线）：A 路线（短期、低风险）= MUSA Graph 降低 launch overhead + 验证 muBLAS stream capture；B 路线（长期、上限最高）= 为 S5000 适配 FlashAttention / 重写 fused 量化 kernel。
- 这一阶段的 fused kernel 优化经验，也直接延续到了 4 月下旬继续的 exllama fused kernel 工作（见第三章）。

---

## 三、4.21 – 4.29　视觉库验证与 demo 构建（穿插进行）

- **human-pose-estimation（人体姿态估计）**：可正常工作，完成 demo。
- **yolov4**：跑通，能生成展示图片，完成 demo。
- **图像风格迁移**：因原 Torch 库部分函数未移植到 PyTorch，暂未跑出可展示效果。
- 4.29 完成 yolov4 与 human pose 的 demo 构建与测试。
- 同期继续推进 **exllama fused kernel 优化**，并初步分析瓶颈：
  1. MUSA 的 kernel launch 额外延时较多；
  2. S5000 上走的是普通 attention，未像 A30 那样启用 flash attention；
  3. exllama decoding 阶段的 fused kernel 表现不佳。

---

## 四、5.07 – 5.12　过渡：算子优化收尾 + 转向 LMCache / vllm_musa

- **5.07**：继续优化 exllama fused kernel，计划测试 `srush-gpu-puzzles` 学习库（巩固 GPU 编程）。
- **5.12**：测试 **lmcache** 库，readme 中的基本流程跑通，准备开始测试 **vllm_musa 下的加速情况**——由此正式转入 vllm_musa 主线。

---

## 五、5.14 – 5.28　vllm_musa 长 prompt 卡死 Bug 攻坚（重点）

> 详见 `ISSUE_vllm_musa_broadcast_deadlock.md`。
> 环境：MTT S5000 ×8 @ node165，vllm_musa 0.9.3.dev（torch_musa 2.7.1，MUSA SDK 4.3.5，MuDNN 3.1.5），测试模型 Qwen2.5-14B，TP=8。
> 这是这几个月持续最久、最硬的一块攻坚：从一个看似随机的「多卡卡死」，两周内层层下钻到一个**闭源 C++ 算子的越界写**，并落地了可用的 workaround。

### 1. 现象与初步误判

- **5.14**：配置 vllm_musa 多卡测试环境，发现 **多卡 + 长 prompt 下必卡死**，开始排查（中途服务器被回收，进度受阻）。
- 现象表现五花八门：「broadcast 死锁」、「MuDNN AsmKernel Error」、「shm_broadcast 超时」、「Fill::Run failed / Permute::Run failed」、各种 illegal memory access——一度按「broadcast 死锁」方向排查（首次发现时甚至误诊为通信死锁）。
- **5.19**：镜像升级到新 build 后仍崩，用 **8 组对照实验**锁定三个**必要条件**：`TP=8` + `prefix-cache=on` + 长 prompt，三者缺一不崩。关掉 prefix cache 后长 prompt 可正常推理 → 矛头指向 prefix-cache 命中路径。

### 2. 逐层下钻定位（5.19 – 5.21）

这是本阶段方法论上最关键的部分——**不轻信表层栈、用替换法逐层证伪**：

- **5.19 py-spy 抓栈**：指向 `flash_attn.py:388`，但后来证明这只是 sticky error 的「信使」（Python 等 GPU 的下一个 sync 点），不是真正出错位置。
- **5.20 完整 checkpoint 链（FULL-CKPT）**：在函数体内插 12 个 sync + try/except 点，发现**我们函数体内所有 sync 都通过，但下一层 sdpa 入口 sync 必然失败** → 确认 bug 在更底层。
- **5.21 STUB 全替换实验**：把整个 sdpa 函数换成直接返回 `query.reshape` 的 stub —— **不崩**，确认 bug 100% 在该函数体内。
- **5.21 三轮 bisection（关键里程碑）**：B1（只 `varlen_fa_seqlen_pad`）OK → B2（+SDPA）OK → B3（只 `varlen_fa_seqlen_unpad`）**崩**。**100% 锁定元凶**：`_kernels.so` 里的闭源 C++ 算子 `ops.varlen_fa_seqlen_unpad`，在 prefix-cache 部分命中（`sum_seq < max_seq × batch_size`）时**越界写 output 缓冲区之后的相邻显存**，污染 GPU context 进入 sticky error。之前看到的所有花式报错都是这一个根因在下游不同 op 上的次生症状。

### 3. prefix-cache 三档命中行为（理解 bug 触发条件的关键）

排查中厘清了 vLLM prefix-cache 不是「命中/未命中」二元，而是按重合度分三档，**只有中间一档触发 bug**：

| 重合度 | 走哪条路径 | 是否触发 Bug 1 |
|---|---|---|
| 0%（全新 prompt，fresh prefill） | 原 prefill 路径，`sum_seq == max_seq×bs` 无不对称 | ❌ 不触发 |
| **部分重合**（共享前缀 + 新尾部） | 进 prefill，`sum_seq < max_seq×bs` | ✅ **触发越界写** |
| 100%（整段已在 cache） | 跳过 prefill 直接 decode | ❌ 不触发 |

- 推论：**多轮对话天然就是部分重合**（每轮新 user 消息 + 共享历史），生产避不开，所以必须修。这也解释了为什么短 prompt benchmark 测不出来（两轮要么 0% 要么 100%，没有一条进 partial 命中）。

### 4. 三条 workaround 路径与抉择

| 路径 | 思路 | 结果 |
|---|---|---|
| **A. Python concat + SDPA** | Python 层用 `block_tables` 从 paged KV cache 拉 cached K/V，拼上新 K/V，调稳定的 SDPA，**完全旁路** `varlen_fa_seqlen_pad/unpad` | ✅ **选用**（唯一能自主控制、不依赖外部修复） |
| B. 改 MTT C++ unpad 加边界检查 | 几行 C++ 改动 | ❌ `_kernels.so` 闭源，需 MTT 发版 |
| C. vllm 上游 Triton paged kernel | import `context_attention_fwd` | ❌ MUSA 检测到「2 active drivers」把 Triton 换成 placeholder，缺 API 跑不通 |

- Path A patch 只加 **1 个分支**（partial 命中那档），另外两档走原版稳定路径，改动面最小、失败模式安全、回滚成本低。

### 5. Path A 验证与「乱码」误判翻转

- **5.21 初判**：Path A「不崩但输出乱码」，一度以为实现错了。
- **5.26（里程碑·翻转）**：经一组对照实验发现 **5.21 的「乱码」是被 benchmark 自身的坏 prompt 误导**（long_context prompt 是同一段话 6 倍重复 + 无意义后缀，让模型脱轨）。改用**真实长文章 + 8 个真问题**后，Path A **8/8 全过、输出语义正确**，且有三类硬证据：① `enable_prefix_caching=True` 确认；② 诊断 print 显示 `cached_lens=640`（695 token 里 92% 从 paged KV cache 命中）；③ **TTFT 319ms → 60ms（降到 1/5）**，prefix-cache 加速可见。**Bug 1 实质解决**。

### 6. 衍生发现与残留 bug

排查过程中又分出两个与 Bug 1 平行的独立问题，形成完整 **bug 地图（Bug 1/2/3）**：

- **Bug 2（stop-token / chat template 泄漏）✅ 5.27 已修**：根因是 Qwen2.5 的 `generation_config.json` 把 `eos_token_id` 只配了 `<|endoftext|>`(151643)，漏了 chat template 实际用来结束每轮的 `<|im_end|>`(151645)。用 `--override-generation-config '{"eos_token_id":[151645,151643]}'` 补上，撞 max_tokens 的请求从「几乎全部」降到 4/16。
- **Bug 3（长 prompt 数值/语义偶发不稳）⚠️ 仍在**：表现为生僻跨语言 token、末尾胡言乱语，是 logit 分布被数值噪声污染（非 temperature 问题）。候选根因为 MuDNN flash SDPA 的 BF16 累积精度 / RoPE / TP all-reduce 精度等，多在闭源 kernel 里，列为待 MTT 排查项。
- **5.27 附带发现**：试 V1 引擎绕开 Bug 1，但 **V1 在 S5000（arch 310）上当前跑不通**——先撞 PH1+V1 的 block_size 代码不一致（已 patch 绕过），再撞底层 `flash_attn_varlen_func` kernel 「not support on MUSA arch 310」（闭源，改不了）。所以「切 V1」这条路当时走不通。

### 7. 阶段成果与交付

- **5.28**：长 prompt + prefix cache 卡死基本消除，模型生成意外截断的问题也通过修 eos 基本解决。
- 产出：完整的诊断报告（STUB 二分定位 + 三档命中分析）、Path A patch（`patches/path_a_python_concat.py`）、可立即复现的测试指令、以及给 MTT 的诉求清单（修 unpad 越界写 / 适配 V1 arch 310 kernel / 查 Bug 3 后端精度）。
- 攻坚中沉淀的几条教训也很有价值：不要相信「sync 通过就没事」（MUSA 错误常延迟到下游 op 暴露）、不要把 padding 区 NaN 当 bug 源、不要相信单层 py-spy 栈、改完 patch 必须验证它真生效再推论。

---

## 六、6.02 – 6.09　vllm_musa 各模型性能基准测试（重点）

> 详见 `vllm_musa_proj/reademe.md`。任务：把 SGLang release 文档里 tp1（单卡）的 Qwen3 系列模型，在新版 vllm_musa 上系统重测一遍，产出与 sglang 可比的性能数据。

### 1. 新版环境与架构变化

- 拿到比攻坚阶段（0.9.3 / torch2.7）新约 2 个月的 **vllm_musa 0.20.1（torch 2.9）**，部署在 **192.168.4.127** k8s pod（MTT S5000 ×8）。
- **关键架构变化**：新版是**纯 V1 引擎**——上游 vLLM 已把 V0 整段删除（`vllm/worker/` 目录不存在、`llm_engine.py` 只剩 7 行别名壳、`VLLM_USE_V1` 开关全包搜不到）。这意味着第五章 Bug 1 所在的 V0 `varlen_fa_seqlen_pad/unpad` 路径**永久消失**，prefill 改走 V1 的 `flash_attn_varlen_func`。本批 Qwen3 已能在 V1 + S5000 上正常 serve（第五章遗留的 arch 310 kernel 问题对这些模型不再是阻塞）。

### 2. 压测工程搭建

搭了一套 **8 卡一卡一模型**的自动化并行压测工程：

- `run_list.py` — 服务编排器：8 张卡各拉一个 vLLM 服务（Qwen3-8B 系列 + Qwen3-VL 多模态 + FP8 量化版），端口 8000–8007，子进程独立成会话组，支持 启动/status/stop。
- `test_list.py` — 调度器：用 `ThreadPoolExecutor` 对 8 个端口并行压测，每模型多 case（不同 input/output 长度 2k~4k × 不同 rate），配置由 `tp1_bench_params.json` 驱动。
- `bench_serving.py` — 打流引擎：异步发请求，统计 TTFT / TPOT / ITL / 吞吐 / 并发等指标，改编自 vLLM/SGLang 官方脚本。

### 3. 关键调参与方法论

- **6.02 – 6.04**：发现原始脚本参数设置有问题，与 sglang 对齐重设（block-size=64、gpu-mem-util=0.8、CUDA Graph 23 档捕获、max-num-batched-tokens=8192 等），并改写成更自动化的测试方式，开始批量测试。
- **6.09**：上一版没测到性能拐点。改进 **request-rate 扫描策略**——每个模型用各自的 per-model rate 序列（快慢模型拐点位置不同，慢模型用快模型的高 rate 会全程过载、白测），从低到高加压绘制**吞吐-延迟曲线找拐点**（`req_tp` 封顶 + `real_concurrency` 猛涨 + TTFT 暴涨 = 过载）。
- 建立了一套**派生指标 + 评判范式**：以 SLO 达标下的最大吞吐（**Goodput**，TTFT<2000ms & TPOT<50ms）为金标准，配合 TGS（每卡 token 吞吐）、归一化延迟等，并出 吞吐-延迟曲线 / 雷达图，作为 vllm_musa vs sglang 最权威的对比方式。

### 4. 优化项盘点（vllm_musa vs sglang 差距归因）

梳理清楚 V1 引擎的优化状态，定位与 sglang 的差距来源：

- **默认已开**：PagedAttention、continuous batching、FLASH_ATTN v3（= sglang fa3）、prefix caching、chunked prefill。attention 后端两边一致，不是差距来源。
- **关键未开（差距主因）**：① **投机解码（EAGLE3）**——sglang 的 8B 开了，吞吐 ×2~3，vllm_musa 没开，是高 rate 下过载的决定性根因；② **并发上限 `--max-num-seqs`**——没设导致并发卡在 ~60，sglang 用 256，是最该补的低成本项。
- 注意点：MUSA 上 FLASH_ATTN 暂不支持 FP8 KV cache；FP8 模型名只代表权重格式，不等于 FP8 KV。

---

## 七、6.16 – 6.17　多卡部署测试 + LMCache 集成与性能验证（重点）

> 详见 `ISSUE_vllm_musa_broadcast_deadlock2.md`（续篇 2）。任务：在新版 vllm_musa（V1 引擎）上把 LMCache 真正跑通，并设计「纯版 vs 挂 LMCache」的性能对照 benchmark。

### 多卡部署

- 测试 **8 卡 PD 分离** 的 vllm_musa 性能，发现相比单卡性能下降，正在排查原因。
- 单卡部分基准基本完成，正在测试多卡 TP/PP 部署方式。

### 1. 概念厘清：APC vs LMCache（续1 完全没碰的新地盘）

- **APC（= prefix-cache）**：KV 切 block 存 **GPU HBM**，前缀 hash 命中就零搬运复用，但容量小（本例 216,704 token）、易失、只在单实例单卡。
- **LMCache**：KV 存 **CPU 内存 / 盘 / 远程**（本例 `LocalCPUBackend`），容量大、可持久、可跨卡跨机，但命中要把 KV 从 CPU 搬回 GPU（有耗时）。
- 二者是**缓存金字塔**（APC=L1，LMCache-CPU=L2），互补不重复。
- **关键陷阱「APC 掩盖 LMCache」**：vLLM 调度先查 GPU APC，再问 LMCache 还能**额外**补多少。若前缀整段还在 GPU APC 里，LMCache 没活干 → `external_hits=0`，这**不是 bug**。所以**必须把工作集撑过 GPU KV 容量才能逼出 LMCache 命中**。
- 命中归因用 vLLM 自带 `/metrics`：`prefix_cache_hits`（GPU APC）vs `external_prefix_cache_hits`（LMCache），压测前后做 delta。

### 2. 分步验证（Step0 → Step2）

- **Step0（标尺）**：读启动日志得 **GPU KV 容量 = 216,704 tokens**（≈144 KB/token），作为构造超容量工作集的基准。
- **Step1（通路冒烟）**：4 组前缀 × 8 = 32 请求、工作集仅 8192 token（远小于容量）。结果 32/32 成功、Median TTFT 119ms、APC 强命中 81.7%，但 **LMCache `external_hits=0`**——正是「APC 掩盖 LMCache」的实测证据（非 bug）。
- **配置收敛**：原计划 CPU cache 设 80GB（按 2× 工作集 ≈62GiB 估算合理），但实测 80GB / 64GB 都启动失败（`musaHostAlloc failed: 205` + MUDNN FillOp 失败），最终用**已验证上限 40GB**，并在 40GB 下重跑 Step1 确认通路健康。
- **Step2 首轮（超容量，随机 shuffle）**：80 groups × 3072 = 245,760 token（≈1.13× GPU 容量）。首次拿到 **LMCache `external_hits=67,328`** 的硬证据（功能验证通过）；但 TTFT 反而比纯版差——因为请求被随机打乱，cold miss / APC hit / LMCache hit 三类混在一起统计，LMCache 收益被稀释。**结论：功能成立，性能收益未验证**。

### 3. Step2 冷热分离复测（关键结论）

针对随机 shuffle 的问题，专门写了脚本 `lmcache_cold_warm_revisit_bench.py`，把缓存灌入和重访严格分开：

- **设计**：80 个随机 token 共享前缀（各 3072 token，保证 group 间绝不意外共享）。**cold 阶段**每个前缀访问一次灌满 cache（245,760 token 超 GPU 容量 13%，灌完最早的前缀必被挤出 GPU）；**warm 阶段**换不同问题再访问一次，此时前缀只剩在 LMCache 的 CPU 里。三次抓 `/metrics` 单独统计 warm 阶段。
- **结果（只看 warm revisit）**：

| 指标 | LMCache warm (19001) | Pure warm (19000) | 改善 |
|---|---:|---:|---|
| Median TTFT | **262.70 ms** | 728.62 ms | ↓ 约 64% |
| Mean TTFT | **261.53 ms** | 649.70 ms | ↓ 约 60% |
| P99 TTFT | **271.19 ms** | 841.58 ms | ↓ 约 570ms |
| Median E2E | **949.46 ms** | 1504.28 ms | ↓ 约 555ms |
| APC hits | 0 | 0 | 两边都不靠 GPU APC |
| LMCache external hits | **245,760** | 0 | 命中完整工作集 |

- LMCache warm 阶段 `external_hits=245,760` 正好等于 80×3072，命中的是刻意构造的完整 shared prefix，非偶然碎片。
- **结论**：在 GPU APC 已无法命中的 warm revisit 场景下，LMCache 从 CPU 取回前缀 KV，**TTFT 降约 64%**——干净地证明了 LMCache 不只是功能链路可用，而且确实带来性能收益。这比首轮可信，因为它把 cold/warm 分离、只统计真正能体现 LMCache 价值的 warm 重访。

---

## 当前进行中 / 待办

- vllm_musa **多卡（TP/PP、PD 分离）** 性能下降原因排查。
- LMCache 性能收益的更大规模、更严格淘汰场景验证。
- 把新版 LMCache 收益与单卡基准数据整合，形成对外的性能结论。
- 待 MTT 侧：修 `varlen_fa_seqlen_unpad` 越界写根因、把 V1 `flash_attn_varlen_func` 适配 arch 310、排查 Bug 3 后端数值精度。
