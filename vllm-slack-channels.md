# vLLM Slack 频道活跃度速查

> 数据采集于 **2026-08-14**，来源 vLLM 开发者 Slack（slack.vllm.ai）。
> 覆盖 `#feat-` 33 个 + `#sig-` 37 个，共 **70** 个频道，全部分级。

## 怎么读这张表

用两个数一起判断，只看其中一个都会误判：

| 指标 | 含义 |
|---|---|
| **最后** | 距最后一条消息多少天。只说明"有没有人来过" |
| **间隔** | 最近 6 条消息的平均间隔天数。说明"**来的频率**" |

**成员数基本无用** —— 频道成员只增不减，是历史积累量。610 人的频道可能三周才一条，18 人的频道可能一天两条。

由此分出六类：

| 标记 | 判据（间隔 / 最后） | 含义 |
|---|---|---|
| 🔥 **热** | < 4 天 / < 14 天 | 真的在讨论，发言有人接 |
| 🌤 **温** | 4–20 天 / < 60 天 | 有人看，回复要等 |
| 📋 **稀** | > 20 天 / < 60 天 | **看着活，实际是公告板**。发言大概率没人理 |
| 💀 **猝死** | < 14 天 / > 60 天 | 曾经密集讨论后戛然而止 —— 项目黄了或讨论搬家了 |
| ⚰️ **死** | > 14 天 / > 60 天 | — |
| ⚫ **空** | 无实质消息 | — |

---

## 🔥 热（16 个）

真正能聊起来的地方。**16 个里 14 个是 `#sig-`，只有 2 个是 `#feat-`。**

| 频道 | 人数 | 最后 | 间隔 | 讨论什么 |
|---|---:|---:|---:|---|
| `#sig-ci` | 582 | 今天 | **0.3 天** | CI 救火：主干挂了、机器排队、flaky test。全场最高频 |
| `#feat-sparse-kv-offloading` | 18 | 4 天 | **0.6 天** | 稀疏 KV + offloading，TP sharing / prefill 侧卸载。小而极活 |
| `#sig-release-management` | 517 | 今天 | 0.7 天 | 版本发布流程与 release blocker |
| `#feat-kimi-k3` | 117 | 10 天 | 0.8 天 | Kimi K3 支持（2.8T MoE，1M 上下文）。新模型突发型 |
| `#sig-amd` | 381 | 2 天 | 1.5 天 | ROCm / Instinct MI3xx，Quark 量化权重 |
| `#sig-agentic-api` | 81 | 今天 | 1.6 天 | agentic 场景 API 设计与 crate 发布 |
| `#sig-spec-decode` | 614 | 今天 | 1.8 天 | 投机解码：EAGLE / MTP / DFlash、Speculators |
| `#sig-omni` | 502 | 7 天 | 2.0 天 | vllm-omni 多模态生成，有固定 sync |
| `#sig-quantization` | 534 | 2 天 | 2.6 天 | 在线量化、LLM Compressor、MXFP4 / NVFP4 |
| `#sig-large-scale-serving` | 275 | 1 天 | 2.6 天 | PD 分离、wide EP、弹性伸缩，对接 llm-d / Dynamo |
| `#sig-model-performance` | 368 | 3 天 | 3.0 天 | nightly 性能评测、trace 分析、性能回归 |
| `#sig-reinforcement-learning` | 404 | 今天 | 3.4 天 | RL 场景：权重同步、rollout 性能 |
| `#sig-perf-eval` | 33 | 4 天 | 3.5 天 | perf-eval 流水线本身的维护 |
| `#sig-tpu` | 292 | 5 天 | 3.7 天 | TPU 后端 |
| `#sig-cpu` | 127 | 3 天 | 3.8 天 | CPU 后端 |
| `#sig-multi-modality` | 590 | 3 天 | 4.0 天 | 多模态模型支持，有 community sync |

## 🌤 温（16 个）

有人看，但别指望即时回复。

| 频道 | 人数 | 最后 | 间隔 | 讨论什么 |
|---|---:|---:|---:|---|
| `#sig-struct-out-tool-calling` | 7 | 今天 | 6.2 天 | 结构化输出 × tool calling 的交叉 bug。7 个人在干活 |
| `#sig-core` | 223 | 1 天 | 7.1 天 | 引擎核心：Scheduler、KV Cache Manager、MRV2 |
| `#sig-frontend` | 83 | 16 天 | 7.3 天 | OpenAI 兼容层、Responses API、scale-out render |
| `#feat-tool-calling` | 162 | 13 天 | 7.4 天 | tool calling / function calling |
| `#feat-kv-connectors` | 167 | 9 天 | 7.9 天 | KV / EC connector 抽象层 |
| `#sig-torch-compile` | 432 | 37 天 | 9.5 天 | torch.compile 集成、fusion pass、编译时间 |
| `#feat-startup-ux` | 194 | 21 天 | 11.6 天 | 启动速度与启动体验 |
| `#feat-structured-output` | 288 | 1 天 | 12.4 天 | 约束解码：JSON schema、grammar、backend 插件化 |
| `#feat-deepseek-v4` | 116 | 3 天 | 13.8 天 | DeepSeek V4，B300 上的 KV 并发退化 |
| `#feat-dllm` | 37 | 22 天 | 13.8 天 | diffusion LLM 插件（LLaDA2、DiffusionGemma） |
| `#feat-gptoss-support` | 163 | 44 天 | 14.3 天 | OpenAI gpt-oss 模型支持 |
| `#feat-ssm-hybrids` | 86 | 30 天 | 15.6 天 | Mamba / Jamba 这类 SSM 混合架构 |
| `#feat-extensible-hardware` | 131 | 4 天 | 16.4 天 | 插件化硬件后端，非 GPU 平台的 MRV2 复用 |
| `#feat-context-parallel` | 76 | 15 天 | 17.0 天 | 上下文并行（长序列切分） |
| `#feat-router` | 149 | 32 天 | 18.6 天 | 请求路由 / 前置调度 |
| `#sig-tokens-in-out` | 39 | 44 天 | 19.4 天 | token 级 I/O API 语义 |

## 📋 稀 —— 看着活，其实是公告板（11 个）

**这一类最容易骗人。** 最后一条消息可能就在今天，但平均要等半个月到几个月才有下一条。在这里提问，大概率石沉大海。

| 频道 | 人数 | 最后 | 间隔 | 讨论什么 |
|---|---:|---:|---:|---|
| `#feat-prefill-disaggregation` | **610** | **今天** | **21 天** | PD 分离。全场人数第一，但三周才一条 |
| `#feat-elastic-ep` | 70 | 16 天 | 22.5 天 | 弹性 EP + 容错 |
| `#sig-batch-invariant` | 102 | 31 天 | 23.0 天 | 批次不变性 |
| `#sig-benchmark` | 271 | 3 天 | 23.3 天 | 基准测试方法 |
| `#sig-docs` | 93 | 31 天 | 25.0 天 | 文档 |
| `#feat-kvcache-offloading` | **353** | **1 天** | **26.5 天** | KV cache 卸载到 CPU / 磁盘 |
| `#feat-free-threaded-python` | 46 | 58 天 | 26.6 天 | 无 GIL Python（PEP 703） |
| `#feat-lora` | 172 | 10 天 | 33.1 天 | LoRA / multi-LoRA |
| `#feat-afd` | 93 | 18 天 | 45.6 天 | Attention-FFN 分离（蚂蚁在推的那个） |
| `#feat-transformers` | 57 | 29 天 | **114 天** | vLLM + HF transformers 后端 |
| `#sig-events` | 113 | 52 天 | **172 天** | meetup / 会议组织，非技术 |

## 💀 猝死 —— 密集讨论后戛然而止（12 个）

间隔很小说明当时聊得很热，但已经两个月以上没人了。**这类通常不是慢慢冷掉，是项目黄了或讨论整体搬家了**，值得先查清去向再决定要不要跟。

| 频道 | 人数 | 最后 | 间隔 | 发生了什么 |
|---|---:|---:|---:|---|
| `#sig-mrv2` | 69 | 116 天 | 1.3 天 | Model Runner V2 —— 讨论迁到了 `#sig-core` |
| `#sig-pooling` | 18 | 127 天 | 1.6 天 | pooling / embedding 模型 |
| `#feat-deepseek-support` | **483** | 97 天 | 1.7 天 | 整体迁到 v3.2 / v4 分频道 |
| `#feat-vllm-ir` | 81 | 93 天 | 4.2 天 | **最后一条就是它的讣告**（见下） |
| `#sig-nvidia` | 118 | 150 天 | 4.5 天 | NVIDIA 相关（FlashInfer 等） |
| `#feat-fused-moe-refactor` | 38 | 63 天 | 6.6 天 | fused MoE kernel 重构 |
| `#feat-eplb` | 12 | 81 天 | 6.7 天 | Expert Parallelism Load Balancer |
| `#sig-omni-rl` | 27 | 181 天 | 7.9 天 | omni + RL |
| `#sig-streaming-input` | 85 | 144 天 | 8.1 天 | 流式输入 |
| `#feat-v1-cpu-offloading` | 154 | 72 天 | 10.7 天 | V1 引擎 CPU offloading |
| `#feat-pipeline-parallel` | 93 | 137 天 | 11.0 天 | 流水线并行 |
| `#feat-deepseek-v32-support` | 126 | 100 天 | 13.5 天 | DeepSeek V3.2，Sparse MLA |

## ⚰️ 死（11 个）

| 频道 | 人数 | 最后 | 间隔 | 讨论什么 |
|---|---:|---:|---:|---|
| `#feat-ray-support` | 126 | 163 天 | 20.0 天 | Ray 分布式执行后端 |
| `#feat-xla-rearch` | 31 | **551 天** | 30 天 | XLA / TPU 后端重构 |
| `#feat-hybrid-alloc-kv-connector` | 65 | 163 天 | 39.6 天 | 混合 KV allocator × connector |
| `#sig-hybrid-memory-allocator` | 75 | 149 天 | 44.5 天 | 混合内存分配器（HMA） |
| `#feat-llama-support` | 78 | 238 天 | 49 天 | Llama 系列支持 |
| `#feat-attention-backends` | **168** | 88 天 | **58 天** | attention 后端选型。**两个月才一条** |
| `#sig-guidellm` | 71 | 164 天 | 65 天 | guidellm 压测工具 |
| `#sig-maca` | 22 | 168 天 | 88.5 天 | 沐曦 MetaX 后端插件 |
| `#sig-metax` | 14 | 357 天 | 样本不足 | 沐曦，与 `#sig-maca` 重复 |
| `#sig-helix-parallel` | 11 | 281 天 | 样本不足 | Helix 并行，只有一条"开个小组" |
| `#sig-cd` | 28 | 602 天 | 样本不足 | 持续交付 |

## ⚫ 空（4 个）

`#sig-spyre`（23 人，只有改频道描述的记录）· `#ext-vllm-sig-rl`（5 人）· `#sig-post-training`（2 人）· `#feat-`（1 人，建频道时名字打空了）

---

## 值得单独记的结论

**1. SIG 比 feat 健康一个量级。**
16 个"热"频道里 14 个是 SIG。`#feat-` 是跟着单个特性开的临时频道，特性做完或黄了就静；SIG 是官方常设工作组，多数有固定例会撑着流量。**要找活人优先去 SIG。**

**2. 人数和活跃度呈现明显的反相关。**
`#feat-prefill-disaggregation` 610 人 / 21 天一条，`#feat-kvcache-offloading` 353 人 / 26.5 天一条，`#feat-attention-backends` 168 人 / 58 天一条。而 `#feat-sparse-kv-offloading` 18 人 / 0.6 天一条，`#sig-struct-out-tool-calling` 7 个人却天天有动静。大频道是历史沉淀，小频道才是当前战场。

**3. `#feat-vllm-ir` 的最后一条消息是它自己的讣告。**
Luka Govedič 公告：vLLM 决定移除 model-level fullgraph torch.compile 集成，改用手写 fusion + eager breakable cudagraph，模型定义按硬件平台特化。原话 *"the fate of vLLM IR is still up in the air"*。之后频道彻底停摆。

**4. 讨论会随特性演进搬家，别把"搬家"误判成"黄了"。**
DeepSeek 总频道 → v3.2 / v4 分频道；MRV2 从 `#sig-mrv2` → `#sig-core`；vLLM IR → `#sig-torch-compile`。看到"猝死"型频道，先找去向。

**5. vLLM Slack 里没有任何摩尔线程 / MUSA 频道。**
沐曦有两个（都已死），华为昇腾在 `#feat-afd` / `#feat-extensible-hardware` 很活跃，IBM Spyre 有频道（空的），Intel Gaudi 有独立 repo 和 CI 流水线。摩尔线程是零。

---

## 方法说明与已知偏差

**采集方式**：直接调 Slack Web API `conversations.history`，每频道取最近 15 条、过滤掉 join/leave/改描述这类系统消息后保留 6 条，计算时间跨度。脚本在会话临时目录，不在本仓库。

**四个局限**：

1. **样本只有 6 条**，突发讨论会把间隔算小。`#feat-kimi-k3` 的 0.8 天来自一次集中讨论，不代表长期节奏。
2. **不含 thread 回复**。`conversations.history` 只返回顶层消息，讨论主要在 thread 里的频道会被系统性低估。
3. **消息不等于有效讨论**。`#sig-ci` 的 0.3 天里相当比例是报错和求 merge，不是技术探讨。
4. **`conversations.history` 在高并发调用下会静默返回空**（疑似限流），空结果不能直接当作"没消息"。本表的空频道都用 `search` 复查过。

**一个 PowerShell 陷阱**：PS 5.1 的 `Invoke-RestMethod` 会丢弃手动设置的 `Cookie` header，导致 Slack 返回 `invalid_auth`。必须用 `WebRequestSession` 加 `System.Net.Cookie` 对象传 `d` cookie。

**复现**：由 Claude Code 读取生成，只读，未在任何频道发言。
