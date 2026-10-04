# 实习工作记录（简历用）

> 实习时间：2026.04 – 2026.06
> 核心方向：摩尔线程国产 GPU（MTT S5000 / arch 310）上的大模型推理框架移植、性能优化与问题攻坚
> 技术栈：vLLM / SGLang / exllamav2 / LMCache、torch_musa、MUSA SDK、CUDA、Python / C++ kernel

1. 大模型推理框架选型与 Qwen3 基准测试：选用阿里 alibaba-rtp-llm，在 NVIDIA A30（24GB）和 MS5000 上对 Qwen3-0.6B / 4B / 8B 三档模型用 ShareGPT_V3 真实多轮对话数据集做端到端压测。搭建环境时因无法连接阿里云服务器拉取镜像、且依赖包版本层层冲突，环境长时间跑不起来，于是改用旧版镜像 + whl 包增量更新的方式绕过网络限制，排查并锁定 torch 2.6.0+cu126 / CUDA 12.6 的兼容版本把环境跑通，同时设计了一套吞吐 / 延迟 + request-rate 扫描的指标体系。测试了三档模型的性能并进行了分析

2. exllamav2 在MS5000上的性能优化：负责将量化 LLM 推理框架 exllamav2 移植到 MTT S5000 并对齐 A30 性能，测试 Llama-3.1-8B / Qwen3-1.8B 的 GPTQ INT4 量化模型，移植后发现性能仅为 A30 的 17–21% 且生成超过 192 token 出现 Segfault / 卡死、无法稳定服务的问题。现在 module 级到 kernel 级的逐层做了插桩分析，锁定瓶颈在 MLP 的三个量化矩阵投影（gate/up/down_proj，各慢 10–11×），再通过组件级benchmark测试确认根因是自定义 GPTQ fused kernel 所用的 NVIDIA intrinsics 在 MUSA 架构上严重退化，叠加单次 decode 发射 300+ kernel 的 launch 过载；在 `q_gemm.mu` 做了两步改动——让 decode 绕过自定义量化 kernel 改走reconstruct + muBLAS路径，并针对 M=1 用 `at::mm` 替代低效的 `mublasHgemm`。最终把 decode 吞吐从 9.07 提升到 21.21 t/s（2.34×）。

3. 视觉库验证与 Demo 构建：负责在国产卡平台上验证多个视觉库的可用性并构建可展示 demo，覆盖人体姿态估计、目标检测（YOLOv4）和图像风格迁移；其中图像风格迁移因所依赖的原 Torch 库部分函数未移植到当前 PyTorch 而无法直接跑出效果，便定位出缺失算子并明确需移植的函数清单，对可正常工作的 pose 与 YOLOv4 完成 demo 构建、测试与展示图产出，验证了平台对主流视觉模型的支持能力。

4. vLLM MUSA 多卡长 prompt 卡死 Bug 分析与解决：在 MTT S5000 ×8 平台、Qwen2.5-14B / TP=8 环境下排查并解决 vllm_musa多卡 + 长 prompt 卡死的 bug。该问题体现为（broadcast 死锁、MuDNN AsmKernel Error、shm 超时、各种 illegal memory access），表层调用栈一度误导排查方向，于是先用 8 组对照实验锁定触发条件（TP=8 + prefix-cache + 长 prompt），再用STUB替换排查问题函数，最终发现根因是闭源 C++ 算子 `ops.varlen_fa_seqlen_unpad` 在 prefix-cache 部分命中时越界写显存、污染 GPU context；由于算子闭源无法直接改，因此在 Python 层用 block_tables 从 paged KV cache 拉取并拼接 K/V、改走稳定的 SDPA 完全旁路问题算子，且只新增 partial 命中这一个分支、改动面最小。最终长 prompt 卡死基本消除，TTFT 由 319ms 降至 60ms，同时附带修复了 eos_token 配置导致的截断 bug。

5. vllm_musa新版多模型性能测试：在新版 vllm_musa（0.20.1 / torch 2.9，纯 V1 引擎）上对 Qwen3-8B 系列、Qwen3-VL 多模态及 FP8 量化版做系统性单卡基准测试，产出与 SGLang 可比的性能数据。针对原始压测脚本参数与 SGLang 不可比、测不到性能拐点、且快慢模型混用同一 request-rate 会导致慢模型全程过载白测的问题，搭建了自动化并行压测工程（服务编排 + 线程池调度 + 异步打流），与 SGLang 对齐关键参数，并把 request-rate 扫描改为 per-model 序列、从低到高加压绘制吞吐-延迟曲线找拐点，同时建立以 Goodput（SLO 达标下最大吞吐）为标准的派生指标。最终产出 vllm_musa vs SGLang 对比数据，并归因出与 SGLang 的主要差距来源是推理框架性能的差距，其次是投机解码（EAGLE3）未开与并发上限 `--max-num-seqs` 未设。

6. 多卡部署测试 + LMCache 集成与性能验证：在新版 vllm_musa V1 引擎上把 LMCache 库成功移植，并进行了原版和LMCache的性能对照 benchmark。过程中遇到首轮测试 LMCache 反而更慢、收益无法测得，CPU cache 配 80GB/64GB 启动失败，以及APC 掩盖 LMCache的问题的问题；为此先读启动日志拿到 GPU KV 容量基准（216,704 token）并据此构造 1.13× 超容量工作集逼出命中，把 CPU cache 设置为 40GB，再针对首轮收益被随机请求稀释的问题设计了冷热分离脚本，把缓存灌入（cold）与重访（warm）分开、只统计真正体现 LMCache 价值的 warm 重访阶段并用 `/metrics` 的 delta 做命中判别。最终在 GPU APC 已无法命中的 warm revisit 场景下，LMCache 命中完整工作集，Median TTFT 由 728ms 降至 263ms（↓约 64%）。

> **当前进行中**：多卡（TP/PP、PD 分离）性能下降原因排查；LMCache 更大规模严格淘汰场景验证；整合对外性能结论。
