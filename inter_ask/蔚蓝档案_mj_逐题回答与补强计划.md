# 蔚蓝档案面试复盘：逐题回答与补强计划

整理日期：2026-09-18。

依据：[面试回忆原文](蔚蓝档案_mj.md)、当前 Qwen2.5 工程源码、项目实验记录，以及文中链接的论文和官方文档。问题按原记录出现顺序拆分；记不清的原话不补造。文中的“口述回答”是下一次可采用的表达，不代表你当时已经这样回答。

**阅读约定：**“代码事实”表示当前源码可以确认；“记录结果”表示旧实验文件中有这个数字；“待验证”表示需要运行测试或采集性能数据。本文没有运行 GPU 测试，也没有修改模型源码。公式全部用纯文本，避免 LaTeX 显示问题。

## 先明确项目边界

你完成的是一个有针对性的推理流程与 CUDA 算子实践，涉及框架集成、模型执行、调度开销和部分算子优化。它有实际工作量，但目前还不能直接等同于经过充分验证、广泛调优的生产级 Attention 实现。

面试最容易出问题的三种混淆：

1. “手写 Attention”不等于“已经用 Tensor Core 做高性能 Attention”。
2. “自己的第二版”不等于论文里的 FlashAttention-2。
3. “端到端生成速度变快”不等于“某个 kernel 达到了硬件性能上限”。

准确讲清实现范围，会让你更容易接住追问。不要为了证明项目有价值，把没有实现、没有测量的东西说成已有结论。

## 1. Attention 计算用了 Tensor Core 吗？

**口述回答：**

> 我手写的 Attention 核心计算目前没有使用 Tensor Core。QK 和 PV 使用普通浮点乘加与 warp 规约，重点实现了分块、在线 softmax 和融合，并改进了线程分工。模型里的 QKV 和输出投影仍调用 PyTorch Linear；这些库算子是否使用 Tensor Core，需要另外查看实际 kernel 或指令，不能和我的 Attention kernel 混为一谈。

**代码事实：**[my_flash_attention_v2.cu](../Qwen2_5_0_5B/csrc/my_flash_attention_v2.cu) 中，QK 的内层循环是 `float(q) * float(k)`；PV 也由普通乘加和 shuffle 规约组成。该实现没有 WMMA、MMA 或 cuBLAS 调用。普通 CUDA 标量乘加循环不能因为输入是 FP16，就默认认定会变成 Tensor Core 运算。

**追问“为什么不用”时：**

> 当时这个版本先完成了融合 Attention 的计算流程和接入，Tensor Core 的 tile 布局与指令实现还没有做。这是目前实现的性能边界。对于 prefill，QK 和 PV 是适合继续尝试 Tensor Core 的矩阵计算；下一步我会在固定形状下实现和测量，而不是直接宣称它一定更快。

不要临时编造“我测过，Tensor Core 更慢”。只有做过相同形状的对照实验，才能这样回答。

## 2. 使用 Tensor Core 要写 SASS 吗？PTX、SASS 是什么关系？

**口述回答：**

> 不需要手写 SASS。可以通过 CUDA C++ 的 WMMA 接口，或内联 PTX 的 MMA 指令使用 Tensor Core，也可以使用 CUTLASS、cuBLAS 等实现。PTX 是中间指令表示，SASS 是具体 GPU 的机器指令。我可以查看生成的 SASS 来确认执行路径，但查看 SASS 和手写 SASS 是两回事。

```text
CUDA C++ / WMMA / 内联 PTX
          ↓ 编译与汇编
       GPU 机器指令 SASS
```

例如适用架构上的 `mma.sync` 是 PTX 指令，不应说成“我写了 SASS”。架构不同，矩阵指令及输入布局也不同。[NVIDIA WMMA 介绍](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

## 3. 是调用 cuBLAS 来完成 Attention 吗？

你的手写 Attention 核心没有调用 cuBLAS。需要区分两种方案：

```text
拆分实现：
GEMM(Q, K转置) → softmax → GEMM(P, V)

融合实现：
在一个 kernel 内分块计算 QK、softmax、PV，避免大中间矩阵落显存
```

前者可以用高效 GEMM 库，但仍需考虑中间结果、额外访存和 kernel 发射。后者也可以在内部使用 Tensor Core 矩阵指令，并不意味着“融合就不能用 Tensor Core”。你的版本是融合思路，但矩阵计算尚未接 Tensor Core。

原文中 `attention计算XXX` 没有完整记录，无法还原面试官的具体疑问。可以围绕“投影 Linear”和“attention 核心 QK/softmax/PV”这两个边界准备。

## 4. FlashAttention 做了什么优化？原理和优势是什么？

**口述回答：**

> 普通拆分 Attention 会生成并读写很大的分数矩阵和 softmax 概率矩阵。FlashAttention 通过分块，将当前需要的 Q、K、V 放在片上，利用在线 softmax 维护归一化状态，并直接累积输出，避免把完整的中间矩阵写回显存。它主要减少显存读写和中间存储，不改变 dense attention 的数学目标。

```text
Q：[Lq, D]
K：[Lkv, D]
V：[Lkv, Dv]

S = Q @ K.T / sqrt(D)       # [Lq, Lkv]
P = softmax(S, axis=-1)     # [Lq, Lkv]
O = P @ V                  # [Lq, Dv]
```

融合后只保留当前 tile 和必要的行状态，不完整物化 S、P。dense attention 的主要算术量仍随序列长度呈二次增长，不能说它“把计算复杂度从平方变成线性”。它是精确 attention 的实现优化；浮点舍入顺序可能造成小数值差异，不保证逐 bit 相同。训练反向还涉及重计算，以换取更少中间存储。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

## 5. GEMM 的 M、N、K 怎么划分？和 FlashAttention 有什么关系？

先写清一次 GEMM：

```text
C[M, N] = A[M, K] @ B[K, N]

沿 M/N 切：产生不同输出位置，通常可独立计算。
沿 K 切：产生同一输出的部分和，需要合并。
```

套到 Attention 的两次矩阵乘法，忽略 batch/head：

| 运算 | GEMM M | GEMM N | GEMM K（归约维） |
|---|---|---|---|
| Q @ K.T | Lq | Lkv | head_dim |
| P @ V | Lq | Dv | Lkv |

**特别容易混：Attention 的 K 张量，不等于 GEMM 名字里的 K 归约维。**

按 Q 的行分块，得到不同 query 的输出，可以独立算。沿 KV 序列切开，则切开了每个 query 的 softmax 范围：不同块的局部输出不能直接相加或平均，必须按各自的归一化状态重新加权。

prefill 有很多 Q 行，容易沿 Q 分配工作；单 token decode 的 Lq=1，Q 方向缺少并行度，这就是后面 Flash-Decoding 引入 KV 分块并行的背景。

## 6. “你没看过论文吗？”怎么回答？

是否读过、读到什么程度，按事实回答。没有完整读过时可以说：

> 我之前主要依据算法说明和实现资料完成了分块、在线 softmax 的代码，但没有完整掌握论文所有优化，尤其是 Tensor Core 工作划分。这部分是我目前的不足。我能先解释自己实现的计算和线程映射，也会把它与论文实现区分开。

后续阅读至少要能回答：避免了哪些中间读写、online softmax 怎么合并、FA2 为什么改 warp 分工、decode 为什么要另找并行维度。只记“用了 shared memory，所以快”不够。

## 7. Online softmax 优化了什么？

**口述回答：**

> 它允许逐块处理分数，同时维护全局最大值和指数和，避免必须先保存整行分数才能归一化。用于 Attention 时，还可以同步维护加权 V 的未归一化输出，从而把 softmax 与后面的 PV 融合起来。

对一行 query，维护：

```text
m：已经处理过的最大分数
l：以 m 为基准的指数和
u：以 m 为基准的未归一化输出向量

初始：m=-∞，l=0，u=0

读入新块的分数 s 和 V：
m_new = max(m, max(s))
r = exp(m - m_new)
p = exp(s - m_new)

l_new = r*l + sum(p)
u_new = r*u + p @ V

最终：O = u / l
```

新最大值改变后，旧的 l 和 u 必须一起乘 r，才能保持同一尺度。有效数据为空的块要专门处理，避免 `-∞ - (-∞)`。

**边界：**单独的 online softmax 通常仍要输出整行概率；不是“任何 softmax 都只读一次就能输出全部结果”。Attention 可以融合 PV，只保留 u，而不输出完整概率矩阵，收益更大。[Online softmax 论文](https://arxiv.org/abs/1805.02867)

## 8. 为什么有 Flash-Decoding？为什么不直接用 FlashAttention？

**口述回答：**

> FlashAttention 的思路仍然有用，但单 token decode 的 query 长度为 1，若 batch 和 head 提供的工作量不够，GPU 并行度会不足。Flash-Decoding 进一步沿 KV 序列拆分任务，各块独立计算局部 attention，再合并归一化结果，以增加并行度。

```text
一个 query 对很长 KV：
chunk 0 → (m0, l0, u0)
chunk 1 → (m1, l1, u1)
...

m = max(mg)
l = Σg exp(mg-m) * lg
u = Σg exp(mg-m) * ug
O = u/l
```

这里 ug 是未归一化输出。若存的是各块已经除过 lg 的输出 Og，合并时必须乘回对应权重，不能套用同一公式而漏掉 lg。

典型实现需要计算局部结果和最终合并两个阶段；额外工作是否值得，取决于上下文长度、batch/head 数和硬件。[作者对 Flash-Decoding 的说明](https://pytorch.org/blog/flash-decoding/)

## 9. 为什么不一直使用普通 decoding？Flash-Decoding 一定更快吗？

**口述回答：**

> 短上下文时，一个融合 kernel 直接算完，可能比拆成多个分块再归约更便宜；长上下文、小 batch 时，增加 KV 方向并行度可能更重要。因此需要按实际形状选择，不能仅凭算法名称判断。

**你的记录结果：**[测试时间记录.txt](../Qwen2_5_0_5B/测试时间记录.txt) 中，约 300 上下文附近记下 normal decode 84.78 tok/s、分块 decode 82.61 tok/s。这是整段生成吞吐，不是两个 Attention kernel 的独立耗时；还存在手动去掉最大值、Graph 路径与正确性待核对的问题。

因此只能说“旧记录里短上下文的普通路径更快，值得做独立验证”。还不能据此确定“差异完全由第二个 kernel 的 launch 导致”，更不能把 256 当作已经测准的最佳切换阈值。

## 10. 你的 grid/block 怎么设计？一个线程究竟算什么？

以下来自当前源码，不把固定形状实现说成通用算子。

| 算子 | grid | block | 主要分工 |
|---|---|---|---|
| 自写 prefill v1 | `(ceil(Lq/16), Hq, B)` | `(64,16)` | 一个 block 处理 16 个 Q；每个 Q 由两个 warp 协作 |
| 自写 prefill v2 | `(ceil(Lq/32), Hq, B)` | `(32,32)` | 一个 block 处理 32 个 Q；每个 Q 对应一个 warp |
| normal decode | `(Hq,B)` | `(512,1)` | 一个 block 处理一个 query head，线程分担 KV 位置 |
| 分块 decode 第一阶段 | `(8,Hq,B)` | `(64,1)` | 固定 8 个 KV 分块，每块覆盖 64 个位置 |
| 分块 decode 合并阶段 | `(64,Hq,B)` | `(8,1)` | 一个 block 对某个输出通道合并 8 个分块 |
| RMSNorm | `(B*L,1,1)` | `(896,1,1)` | 一个 block 处理一行，线程对应 hidden 元素 |

对 prefill v2：

```text
blockIdx.x：Q 的第几个 32 行块
blockIdx.y：query head
blockIdx.z：batch
threadIdx.y：块内的 query 行，同时确定 warp
threadIdx.x：lane 0..31

每个 KV tile 有 64 行。
每个 lane 算两个 score：lane 和 lane+32 对应的 K 行。
每个 score 的 head_dim=64 点积，由该线程循环累加。
```

**不要说成“32 个 lane 分摊一个 QK 点积的 64 个乘法”。你的这份 v2 不是这样分工的。** 它是 lane 分摊不同 K 行，每个 lane 自己算完整点积，再在 warp 内做 softmax/PV 的规约。

decode 第一阶段的 `8×64` 对应当前 512 长缓存，不是任意长上下文实现；头数、head_dim、GQA 比例也有硬编码。源码：[prefill v1](../Qwen2_5_0_5B/csrc/my_flash_attention.cu)、[prefill v2](../Qwen2_5_0_5B/csrc/my_flash_attention_v2.cu)、[normal decode](../Qwen2_5_0_5B/csrc/my_decode_attention.cu)、[分块 decode](../Qwen2_5_0_5B/csrc/my_flash_decoding.cu)。

## 11. 你自己的 v2 相比 v1 改了什么？

**口述回答：**

> 第一版每行 Q 用两个 warp，softmax 和 PV 的部分结果需要通过 shared memory 跨 warp 合并。第二版改成一行 Q 对应一个 warp，每个 lane 承担两个 KV 位置，把这些规约收进 warp 内；同时将一个 block 处理的 Q 行数从 16 增加到 32，并给共享数组加 padding，改善特定访问模式。

| 对比 | 自写 v1 | 自写 v2 |
|---|---|---|
| block 线程数 | 64×16=1024 | 32×32=1024 |
| 每行 Q 的 warp 数 | 2 | 1 |
| 每个 lane 的 score 数/64 行 KV tile | 1 | 2 |
| 跨 warp 中间缓冲 | `mim_data[16][2]` | 相关规约不再需要 |
| shared 数组行宽 | 64 | 66 |
| Tensor Core | 未使用 | 未使用 |

**进一步追问时：**这不是线程数减少，而是线程承担工作的方式改变了。收益可能来自同步、shared 访问和数据复用变化；需要拆分改动进行消融实验。v2 仍有逐输出通道循环规约、1024 线程 block 等明显优化空间。

同步数量减少不代表所有同步都能删，当前共享缓冲区循环复用的正确性风险见第 34 节。

## 12. 论文 FlashAttention-2 相比 FlashAttention-1 改进了什么？

原文的“v1 相比 v2”可能是回忆时写反；通常问题是第二版如何改进第一版。

**口述回答：**

> FA2 延续了分块和在线 softmax，主要改进是减少非矩阵乘法运算、增加单个 head 内沿 query 序列的 block 并行，并改进 warp 工作划分。典型变化是从切分 K 的协作方式转向切分 Q，让不同 warp 负责各自输出行，减少跨 warp 的中间结果通信。

**不要回答成：**“FA1 没有 Tensor Core，FA2 才有”，或“FA2 才不保存完整 attention 矩阵”。这些不是两者的区别。

你自己“每行两个 warp 改成一个”的动机与减少 warp 通信相通，但不能据此宣称完整复现 FA2。[FA2 论文](https://arxiv.org/abs/2307.08691)、[作者的工作划分说明](https://crfm.stanford.edu/2023/07/17/flash2.html)

## 13. “你之前主要做架构吗？”怎么回答？

“架构”可能指模型结构、推理软件架构或 GPU 硬件架构，应先简短对齐含义。

> 我的工作主要在推理软件和 GPU 算子这两层：既搭建过模型执行流程，也做过具体算子和国产卡适配，并不是设计 GPU 硬件架构。相比把单个 GEMM 优化到极限，我之前在端到端推理和故障定位上投入更多。

这比因为本科专业或课程背景而笼统回答“我主要做架构”更能表达实际经历。

## 14. 简单介绍 Qwen2.5 的架构

**口述回答：**

> 我实现的是 Qwen2.5-0.5B 的 decoder-only 推理流程。输入先经过 embedding，然后重复 24 层 decoder。每层先做 RMSNorm 和带 RoPE 的 GQA attention，加残差；再做 RMSNorm 和 SwiGLU MLP，再加残差。最后经过 RMSNorm 和词表投影得到 logits。

```text
token IDs → embedding
  → 重复 24 次：
      x1 = x + Attention(RMSNorm(x))
      x2 = x1 + MLP(RMSNorm(x1))
  → final RMSNorm → lm_head → logits → 选 token

MLP(x) = down( SiLU(gate(x)) * up(x) )
```

你的这份模型是 hidden=896、Q heads=14、KV heads=2、head_dim=64、MLP intermediate=4864。不要把这些小模型参数说成所有 Qwen2.5 型号都相同。

代码：[my_qwen2.py](../Qwen2_5_0_5B/my_qwen2.py)、[模型配置](../Qwen2_5_0_5B/modeel_dir/config.json)。

## 15. 是不是漏了 Q/K/V 三个 Linear？RoPE 在哪里？

**口述回答：**

> 刚才我把 Attention 当成整个模块概括了，展开后先有 Q、K、V 三个线性投影，再对 Q 和 K 施加 RoPE，写入或读取 KV cache，接着计算 softmax(QK转置/√D)V，最后经过输出投影。三个 Linear 在 attention 核心计算之前，但在软件实现上通常属于 Attention 模块内部。

```text
归一化后的 hidden states
        ├→ Q Linear → reshape → RoPE ─────────┐
        ├→ K Linear → reshape → RoPE → cache ┤
        └→ V Linear → reshape ───────→ cache ┤
                                            ↓
                                   QK → softmax → PV
                                            ↓
                                     合并 heads → O Linear
```

代码确实在 `MyQwenAttention` 内定义了 q/k/v/o 四个 Linear。当前 RoPE 应用于 Q/K，不应用于 V。GQA 中多个 Q head 共享一个 KV head；你的比例是 7:1。

被指出口头遗漏时，直接补全计算顺序即可，不必争论“算不算 Attention 前面”。

## 16. 为什么手写 RMSNorm 替换原来的实现？

**口述回答：**

> 原来由多个 PyTorch 操作组成，包括平方、求均值、加 epsilon、倒平方根和逐元素乘法。对于小 batch，它们的发射和中间张量开销比较明显。我将一行的归约与归一化融合在一个 kernel 中，让输入值在寄存器里复用，减少发射次数和中间读写。

```text
r = rsqrt( sum_i(x[i]^2)/H + eps )
y[i] = x[i] * r * weight[i]
```

你的 [rmsnorm_kernel.cu](../Qwen2_5_0_5B/csrc/rmsnorm_kernel.cu) 固定 H=896，每行 896 个线程。先在 warp 内求和，28 个 warp 的部分和写 shared，再合并。

**实现边界：**这不是任意 hidden size 的通用 RMSNorm；896 线程也没有经过完整 block-size 搜索。还应比较 128/256 线程、每线程处理多个元素的版本。不能把模块 NVTX 区间耗时直接当成单个 `rms_kernel` 的 GPU 时间。

## 17. CUDA kernel 常见哪两类瓶颈？RMSNorm 属于哪类？

常用分类是计算受限和访存带宽受限，但还会有低并行度、发射延迟、依赖链、同步或某类指令吞吐限制。

```text
算术强度 AI = FLOPs / 搬运字节数
粗略吞吐上限 ≈ min(相关计算单元峰值, 内存带宽 × AI)
```

RMSNorm 每个元素的算术较少，大规模数据时通常容易受带宽限制；batch=1、只有一行时可能只启动一个 block，不能简单说已经打满显存带宽。框架拆分版还可能主要受 launch 影响。

**口述回答：**

> RMSNorm 通常是低算术强度算子，但我会结合形状判断。很多行时关注有效带宽；单行小工作量时，更关注发射、归约和并行度，最后用 NCU 数据验证。

优化 FP32 CUDA Core 算子时，也不能把 Tensor Core 的峰值算力直接拿来当该实现的 roofline 上限。

## 18. PagedAttention 是什么？

**口述回答：**

> 它将每个请求逻辑上连续的 KV cache 分成固定 token 数的块，再通过 block table 映射到不要求连续的物理块。Attention kernel 根据映射访问缓存，减少为每个请求预留大块连续空间的浪费，也便于缓存块共享和复用。

```text
假设每块 16 token，某请求 block_table=[17,4,23]
token 34：逻辑块 34//16=2，块内偏移 34%16=2
实际访问：物理块 23 的第 2 个位置
```

KV cache block 和 CUDA thread block 不是一个概念。PagedAttention 不等于 CPU 内存换入换出；prefix caching 也不是它的同义词。

你的 Qwen 手写引擎当前是固定连续 KV buffer，不是自己实现了 PagedAttention。你对 paged KV 的实践主要来自 vLLM 项目和根据 block table 读取缓存的修复方案。[vLLM 官方说明](https://docs.vllm.ai/en/latest/design/paged_attention/)

## 19. 你具体怎样用 NSYS？

**口述回答：**

> 我用 nsys 启动被测程序，采集 CUDA 活动，在模型各模块加 NVTX 标签以关联调用阶段。先观察 CPU 发射、GPU kernel、拷贝和同步的时间线，定位间隙和热点，再选择具体 kernel 深入分析。

与当前代码设计相配的命令：

```powershell
cd E:\vscode\cuda_proj\SNN_proj1\Qwen2_5_0_5B
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -o qwen_profile E:\envs\cuda_env\python.exe benchmark2.py
nsys stats --report cuda_gpu_kern_sum qwen_profile.nsys-rep
```

代码通过 `cudaProfilerStart/Stop` 指定采集窗口，`emit_nvtx()` 给 PyTorch 操作添加标记；模型里还有 `RMSNorm`、`MLP`、`Attention_generate` 等手工标签。`.nsys-rep` 用 GUI 打开。

**边界：**当年的完整启动命令没有留存，上面是配套用法；当前 benchmark 接口还需先修正，见第 34 节。NVTX 是 CPU 侧范围，长度不能直接当作异步 GPU kernel 的耗时；看到 GPU 空闲也不能不排查依赖、拷贝和同步，就断言都是 CPU 发射慢。[NSYS 文档](https://docs.nvidia.com/nsight-systems/UserGuide/)

## 20. NSYS 能看单个 kernel 吗？NCU 看什么？wave 又是什么？

**口述回答：**

> NSYS 可以看单个 kernel 的名称、持续时间、启动配置和前后依赖，不是看不了 kernel。需要分析内部瓶颈时，我再用 NCU 看带宽、SM 吞吐、寄存器、occupancy、shared bank conflict 和调度停顿等指标。

建议在独立 microbenchmark 中筛选少量 launch，避免不加筛选地对整个模型采集全部指标。下面是待编写 microbenchmark 的命令模板，不是历史运行记录：

```powershell
ncu --set full --kernel-name "regex:rms_kernel.*" --launch-count 1 -o rms_profile E:\envs\cuda_env\python.exe bench_rmsnorm.py
```

`bench_rmsnorm.py` 此次没有创建。正式比较时间应另做不带 profiler 的重复计时，因为采集与 replay 会影响运行。[NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

关于 wave，可用简化模型理解：

```text
S = SM 数
R = 当前 kernel 每个 SM 可同时驻留的 block 数
G = grid 中总 block 数

同时驻留容量约为 S*R
wave 数量尺度约为 G/(S*R)
```

这不是全 GPU 每轮一起启动、一起结束的同步批次。block 完成后硬件可以继续分配后续工作。大量完整 wave 很正常；任务过少或末尾只剩少量 block 时，可能产生利用不足。warp、wave、block、thread 四个概念不能混用。[NCU 性能分析说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 21. CUDA 一般有哪些优化方法？

先按瓶颈组织回答，比罗列术语更清楚：

| 观察到的问题 | 候选措施 | 必须关注的代价 |
|---|---|---|
| 数据搬运过多 | 融合、分块复用、减少中间结果 | 寄存器/shared 占用增大 |
| 访存请求分散 | 改线程映射、布局、对齐 | 额外重排成本 |
| 矩阵算术慢 | 适当精度下使用 Tensor Core、改 tile | padding、布局转换、数值误差 |
| GPU 工作量不足 | 增加独立 tile、split reduction | 额外归约和中间结果 |
| 同步/通信多 | warp 内规约、改变工作划分 | 不能删掉正确性必需的同步 |
| launch 占比高 | 算子融合、CUDA Graph | 动态 shape 和控制流限制 |
| 拷贝与计算串行 | pinned memory、异步传输、多 stream | 需要独立工作与正确依赖 |

接着举一个自己做过的例子：RMSNorm 融合，或 Attention 两个 warp 合并为一个。不要把所有技巧都说成“越多越快”。

## 22. “内存连续性”到底怎么优化？

核心是一个 warp 的一次内存指令所请求的地址集合，而不只是 tensor 的 `is_contiguous()`。

```cpp
// float 元素，基址适当对齐
x[base + lane]        // 32 个 lane 覆盖连续 128 字节
x[base + lane * 32]   // 每个 lane 相隔 128 字节，访问分散
```

可以把 lane 映射到物理连续维度，选择合适的行列布局，并通过 shared memory 重排来兼顾 global 合并访问和后续计算。

**你自己的具体例子：**normal decode 中，一个线程负责一个 KV 位置，在点积内层固定 i 时，相邻 lane 读取 `K[(pos+lane)*64+i]`。FP16 下相邻地址相差 128 字节。这暴露了值得测量的访问效率问题，即使单个线程后续循环会连续读取其对应行。缓存可能缓解部分开销，所以先测 sector/缓存行为，再评价优化收益。

## 23. L1 的搬运机制是什么？最小一定 128 byte 吗？

针对这里讨论的 NVIDIA GPU，可区分：

```text
L1 cache line：128 byte
sector：32 byte
一条 line 含 4 个 sector
```

**一条 line 是 128B，不等于每次访问都必须从下级搬满 128B。** 请求覆盖哪些 sector、哪些已经命中缓存，会影响实际下级流量。也不要把 sector 大小直接等同于物理 DRAM 的全部突发传输细节。[NCU 对 cache line/sector 的定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference)

例如 warp 的 32 个 lane 各读一个 float，连续且对齐时覆盖 4 个 sector；起点偏移可能让它跨到第 5 个 sector。访问计数和真正 DRAM 字节数也不是同一个指标。

## 24. 必须相邻 thread 访问相邻地址吗？

不是硬性要求。连续访问是容易实现高效合并的常见方式，真正看的是同一次 warp 指令覆盖了多少地址区域。

例如同一 warp 的 lane 反向访问同一段连续数据，覆盖的 sector 集合可以不变。多个线程读同一个地址也不是“线程不连续所以效率一定差”。相反，每个线程各自读一串连续数据，如果相邻 lane 的起点相距很远，单次指令仍可能不合并。

**口述回答：**

> 我会先写出地址关于 lane ID 的表达式，再数它覆盖的 sector，而不是只看代码里有没有连续数组下标。

全局内存 coalescing 与 shared bank conflict 是不同层面的分析。[CUDA 访存优化指导](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

## 25. 哪些情况适合用 shared memory？

**口述回答：**

> shared memory 适合 block 内的数据复用、线程间协作、归约中间结果以及数据重排。比如 GEMM 将 A/B tile 搬入后供多个线程复用，或者转置时先合并读入，再按另一种访问模式输出。

它保存的是需要复用或交换的数据，不是笼统“保存大量重复计算”。同一线程的复用优先考虑寄存器；warp 内交换可考虑 shuffle；多 warp 协作才更常需要 shared。

没有复用的 vector add 一般不需要先搬 shared。使用 shared 要同时评估容量、读写次数、bank conflict 和同步成本。

## 26. 把量化权重反量化后存 shared，再复用，可行吗？

可行，但要先说明量化格式和计算路径。

```text
W4A16 常见实现之一：
global INT4 → shared INT4 → 寄存器解包/反量化 → FP16 MMA

另一候选方案：
INT4 → 寄存器反量化 → shared FP16 → 多个 warp 读取
```

后者可减少跨 warp 重复反量化，但 FP16 权重 tile 比 INT4 大 4 倍，并增加 shared 写入和同步。前者也能把反量化后的 fragment 留在寄存器里复用，不是每次乘加都必须重新解包。

如果讨论的是原生 FP8×FP8 Tensor Core 路径，提前展开成 FP16 会改变计算方案。先确认两方讨论的是 W4A16 还是 FP8，不能只用“量化”一个词混在一起。

**面对“这样就失去意义”可回答：**

> 我刚才指的是 W4A16 权重量化，global 仍保存 INT4，因此压缩读取的收益还在；至于片上反量化放在哪里，需要权衡复用和资源占用。如果讨论的是原生 FP8 Tensor Core 路径，我同意应优先保持低精度输入来计算。

## 27. FP8 输入 Tensor Core、输出后反量化是什么流程？

以每个输入 tensor 一个缩放系数为例，定义：

```text
X ≈ sx * X8
W ≈ sw * W8

Y ≈ (sx * sw) * (X8 @ W8)
```

可以先使用 FP8 输入做矩阵乘法、用高精度累加，再应用 scale 恢复数值尺度；输出可按需要保存成 FP16/BF16/FP32，或重新量化成 FP8。尺度恢复可以融合到 GEMM 收尾，不要求独立 kernel。

```text
X=[1,2]，W=[3,4]ᵀ，正确结果为 11
sx=0.5，sw=0.25
X8=[2,4]，W8=[12,16]ᵀ

乘法结果 P=88，哪怕以 FP32 保存仍然是 88
恢复尺度：88*0.5*0.25=11
```

把 FP16 转 FP32 不等于反量化；恢复尺度也不恢复舍入损失。如果 scale 沿 K 分组变化，需要对部分和分组缩放再合并，或由支持块缩放的硬件处理，不能所有组相加后只乘一个系数。[cuBLAS FP8 缩放定义](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#tensorwide-scaling-for-fp8-data-types)

RTX 3060 的 Ampere Tensor Core 不提供原生 FP8 MMA；在你的卡上学习 Tensor Core 可以先用 FP16 输入。不能拿“3060 没有 FP8”解释“为什么 FP16 Attention 没用 Tensor Core”。[Ampere 数据格式说明](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations)

## 28. Bank conflict 是什么？是 16 个 bank 吗？

针对你的 Ampere GPU，shared memory 是 **32 个 bank**。以常见的 32-bit word 映射理解：

```text
bank_id = floor(byte_address / 4) % 32
```

同一个 warp 的同一次 shared 指令，如果不同地址的字落到同一 bank，可能需要拆成多次服务。多个线程读取同一个 word 可以广播，不按普通冲突处理；分析 half、向量指令时还要考虑字内位置和请求拆分。[CUDA shared memory 说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#shared-memory-5-x)

最常用的 float 示例：

```text
float tile[32][32]
warp 各 lane 读 tile[lane][0]
相邻 lane 相隔 32 个 float → bank 全相同

float tile[32][33]
warp 各 lane 读 tile[lane][0]
相邻 lane 相隔 33 个 float → bank 依次错开
```

**联系你自己的代码：**v2 将 `[...][64]` 改为 `[...][66]`。对 FP16，固定列、lane 取不同的行时：

```text
行宽 64 half：每行 128 byte = 32 个 word，bank 步长 0
行宽 66 half：每行 132 byte = 33 个 word，bank 步长 1
```

所以这能改善对应的跨行访问。但是 FP32 时，66 个 float 的 bank 步长是 2，32 lane 会重复使用 16 个 bank；不能不分 dtype 就说“加 2 完全消除了所有冲突”。也要分析实际指令对应的读写模式，并用 NCU 验证。

## 29. 介绍 GPU 架构，以 RTX 3060 / Ampere 为例

**口述顺序：**

```text
整张 GPU：多个 SM + 共享 L2 + 显存控制器/显存
单个 SM：warp 调度、寄存器、shared/L1、运算与访存单元
执行单元：浮点/整数执行管线、Tensor Core、LD/ST、特殊函数等
```

GA10x SM 划分为四个处理分区，每个分区有 warp scheduler 和部分寄存器、执行资源。寄存器合计 **64K 个 32-bit 寄存器，即 256 KiB/SM**，不是 64 KiB/SM；每个分区 64 KiB。[NVIDIA GA10x 白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

对 compute capability 8.6，面试常用资源上限：

| 项目 | 上限/说明 |
|---|---|
| warp 大小 | 32 threads |
| 每 block 最大 threads | 1024 |
| 每 SM 最大驻留 warps | 48，即 1536 threads |
| 每 SM 最大驻留 blocks | 16 |
| 寄存器 | 64K 个 32-bit 寄存器/SM |
| shared memory | 最大约 100 KiB/SM；单 block 最大约 99 KiB，具体申请需满足配置限制 |

这些是架构限制，不是某个 kernel 自动能同时用满的资源。shared/L1 配置、具体 GPU 启用的 SM 数、显存带宽和时钟要查询设备，不要将 3060 桌面版、Laptop 版和 A100 的数字混用。[Ampere 调优指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)

## 30. Grid、block、warp 如何映射到 GPU？

**口述回答：**

> 一次 kernel launch 定义一个 grid，里面有多个 block。普通 CUDA 执行模型下，一个 block 的线程在同一个 SM 上执行，SM 可以同时驻留多个 block。block 内线程被划分成 warp，由 SM 的 warp scheduler 从就绪 warp 中选择指令发射。

关键细节：

- block 不会被平均拆到多个 SM；不同 block 的执行顺序一般不能假设。
- 一个 block 有 1024 个线程，不等于同一个时钟周期执行 1024 次标量运算。
- `blockDim=(32,32)` 中 x 维优先线性化，因此一个固定 y 的 32 个 x 线程形成一个 warp。
- `__syncthreads()` 同步的是同一 block，不是整个 grid。
- CUDA Graph 负责组织和重放执行任务，并不替代 kernel 内的 block/warp 调度。

不同 warp 等待内存或依赖时，调度器可发射其他就绪 warp，以隐藏延迟；驻留很多 warp 也不保证它们都随时可发射。

## 31. 一个 SM 能驻留多少 block，由什么限制？

主要取各项资源允许数量的最小值：

```text
resident_blocks <= min(
  block 数量架构上限,
  thread/warp 容量允许的 block 数,
  寄存器容量允许的 block 数,
  shared memory 容量允许的 block 数
)
```

寄存器与 shared 分配存在粒度、分区等约束，不能只用总量除法当精确答案。实际可结合编译报告、occupancy API 和 NCU 判断。

**直接联系你的实现：**CC 8.6 的 1536 threads/SM，放入一个 1024-thread block 后，即使寄存器/shared 还有空间，也无法再放第二个同样大的 block。最多 32 个驻留 warp，相对 48 的架构上限，理论 occupancy 约 66.7%，还可能受其他条件限制。这个数字不等于“算力只能发挥 66.7%”。

四个 warp scheduler 不等于只能驻留四个 block；scheduler 数量主要影响发射能力，不能简单当成 block 数量上限。

## 32. 写一个 vector add kernel

最基本的面试版本：

```cpp
#include <cuda_runtime.h>
#include <cstddef>

__global__ void vector_add(const float* a, const float* b,
                           float* c, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 假设 a/b/c 是有效 device 指针，n 对应合法 grid 大小。
// n==0 时不 launch。
void launch_add(const float* a, const float* b, float* c,
                std::size_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n - 1) / threads + 1);
    vector_add<<<blocks, threads, 0, stream>>>(a, b, c, n);
    // 调用方应检查 cudaGetLastError()；测试时再同步检查执行错误。
}
```

解释：相邻线程访问相邻 float；尾部有边界检查；每个输出独立，不需要 shared 或同步。忽略缓存等影响，每元素读两次 float、写一次 float，理想数据量约 `12*n` byte，算术只有一次加法，规模足够大时通常适合用有效带宽评价。

若面试官要求限制 grid 大小，可改成 grid-stride loop：

```cpp
for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
     i < n;
     i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    c[i] = a[i] + b[i];
}
```

文中代码是回答示例，本次没有编译或运行。生产封装还需处理参数、grid 上限、错误传播等。

## 33. 如何回应“你是不是没有把一个 kernel 优化到极致”？

**可以直接承认范围：**

> 是的，我之前更多做端到端推理、算子接入和问题定位，没有把某一个 kernel 在多种形状下持续优化到接近硬件上限。我的 Attention 完成了分块和规约优化，但还没有 Tensor Core 版本和完整的 shape sweep。下一步我会补独立 benchmark、正确性验证和 NCU 对比，先把一个算子做扎实。

不必说“我的项目都是玩具”，也不必用“我以后都会做”替代当前证据。当前优势是系统链路和排障经验；单算子深度需要继续积累，这两点可以同时成立。

## 34. 当前源码和实验记录中，优先需要复核的事项

下面是本次阅读发现或与旧项目复盘交叉核对的事项。它们比立即添加更复杂的优化更优先。只做静态检查，未运行验证；不能用当前快照反推所有旧版本都存在相同问题。

### 34.1 Graph 调用参数和模型接口不一致：当前源码已确认

[benchmark2.py](../Qwen2_5_0_5B/benchmark2.py) 的 capture 调用传入 `update_seq_len=False`，但当前 `Qwen2Model.forward` 没有这个参数。直接执行到该路径会因关键字不匹配而失败，需要先统一接口。

不要只删除该关键字就认为修复完成：还需要检查下一项长度更新问题。

### 34.2 Graph 中的长度更新与 Python 分支：需要一起修复设计

当前 forward 收到非 None 的 `current_seq_len` 时，会执行：

```python
self.seq_len_t.fill_(current_seq_len)
```

capture 时传入 Python 整数，会把常量填充操作录入图；如果没有额外机制阻止它，每次 replay 都可能覆盖图外写入的新长度。这是按代码和 Graph 语义得到的判断，不是本次运行的测量结果。

同时 `current_seq_len > 256` 的 Python 分派只在 capture 时执行，replay 不重新判断。应将“选择哪个 kernel”和“kernel 读取当前有效长度”分开设计。可按长度区间选择不同 graph，或让固定 kernel 读取设备长度并处理边界。

### 34.3 Prefill v2 循环复用 shared 的同步风险：待 racecheck 验证

当前 v2 在每轮加载 K/V 后有 `__syncthreads()`，但计算完本轮、下一轮开始覆盖同一 shared K/V 之前，没有看到 block 级保护。

各 warp 虽各自负责不同 Q 行，但会读取共享的 K/V tile。较快的 warp 可能开始写下一轮，而较慢的 warp 仍在读本轮。warp 内 shuffle 不能保证跨 warp 的这类缓冲区安全。

应针对多 KV tile、不同尾块和 causal 分支做正确性测试，并用 `compute-sanitizer --tool racecheck` 检查；必要时在缓冲区覆写前增加正确同步或采用有依赖保护的流水线。这是高优先级风险，不在没有测试的情况下宣称已确认发生数据污染。[Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

### 34.4 当前算子支持范围较窄

RMSNorm 固定 896；Attention 多处固定 head_dim=64、14 Q heads/2 KV heads；decode 缓存固定 512，分块固定 8×64。封装还需要检查 dtype、device、shape、contiguous/stride，处理 PyTorch 当前 CUDA stream，并检查 launch 错误。

这些是下一步工程化工作。面试时可以明确说“针对该模型配置优化”，不能说支持任意模型和形状。

### 34.5 吞吐口径需要修正

当前 50 token 的计时窗口覆盖 prefill、decode 和一次 capture；不能称纯 decode 吞吐。若遇 EOS 提前结束，还应按实际生成数统计。历史记录中的 78.83 tok/s 与同时写下的 0.5626 秒也不满足 `50/time`，需要查原始日志。

21.36 到 82.41 的倍率可以从简历数字算出，但当前文本记录不足以完整重建首个基线、所有中间版本和一致测试条件。不要用倍率替代可复现的实验。

### 34.6 速度与正确性必须分开验证

至少检查：单算子与参考结果、每层输出或 logits、prefill+多步 decode、eager 与 Graph 的一致性、有效长度增长，以及缓存边界。低精度不要求逐 bit 相同，但误差容限必须有依据。能生成通顺句子不等于计算正确。

## 35. 为什么项目做了不少，还是容易被追问住？

最主要的差距是下面这条链还没有闭合：

```text
我实现了什么
  → 数学上为什么正确
  → 线程与数据怎么映射
  → 实际受什么限制
  → 哪项测量支持判断
  → 修改改变了哪个指标
  → 哪些形状有效，哪些无效
```

### 35.1 项目覆盖广，但单点深度分布不均

你做过移植、故障排查、缓存集成、压测、手写算子。它们积累的是不同能力。系统项目做得多，不会自动让你熟悉 MMA fragment 布局或写出高性能 GEMM。

### 35.2 端到端加速容易掩盖单算子的不足

基线有较多 Python 调度或小 kernel 时，融合和 Graph 就可能带来明显端到端提升。但这不证明 Attention 的 QK/PV 执行效率高。面试官问 Tensor Core，是在确认算术实现还剩多少空间。

### 35.3 项目命名与结论比现有证据走得更远

“FlashAttention v2”“纯 decode 提升”“消除了 bank conflict”“确认根因为某条 intrinsic”都容易引发更严格的追问。需要明确是自己的版本号、端到端结果、某种访问模式的改善，还是已经有硬件计数器和指令级对照的结论。

### 35.4 看懂术语，不等于能现场推导

真正需要练的是：给一段下标算出 lane 地址，给一组维度画出 tile，给一个 warp 解释哪个线程产生哪个结果。把这个能力建立起来，面试官改变问法时也能继续推导。

### 35.5 通过一面说明已有能力被看见，但不替代后续补强

按你的记录，面试官认可了 CUDA/GPU 理解，HR 也确认通过。合理判断是“基础与实践达到了一面的认可，深入优化还有缺口”，而不是“项目没有价值”或“已经不需要补”。

## 36. 现在按什么顺序补？

### P0：先让一个现有结果可信

优先选 RMSNorm 或当前 Attention 的固定形状，建立能单独运行的测试。先处理第 34 节接口、Graph 长度和共享缓冲风险，不新增复杂功能。

需要留下的产物：

- 一个明确支持范围的算子接口。
- 一组参考输出、边界用例与误差结果。
- 一个独立 benchmark，记录版本、设备、shape、dtype、warmup、重复次数和统计方式。
- 一份不含 profiler 干扰的耗时，以及对应的 NSYS/NCU 报告。

“没有显著加速，但正确、可解释、可复现”也比一个来源不明的大倍率有用。

### P1：补一个 Tensor Core GEMM，再决定如何接 Attention

在 3060 上先做 FP16 输入、FP32 累加的 WMMA GEMM，弄清楚：

1. block tile、warp tile、MMA tile 的关系。
2. A/B 在 global、shared 和寄存器 fragment 中的布局。
3. 尾块与对齐，warp 协作调用的要求。
4. 为什么调用 Tensor Core 后仍可能受访存、同步或小矩阵利用率限制。

先和 cuBLAS/明确的参考计算比较正确性及时间，再学习内联 MMA 或 CUTLASS/CuTe。FP16 cuBLAS 基线与 SIMT FP32 累加基线要记录精度差异，不能只比较一个数字。

接入 Attention 时，不是简单替换一个乘法：QK fragment 要配合按行 softmax，P 又要重新组织成 PV 的操作数。在线状态、布局转换和同步是重要工作。可以先完成单个固定 head_dim 的版本。

### P2：把一个算子做成完整优化案例

建议先把 RMSNorm 做成相对容易闭合的案例，再深入 Attention：

| 版本 | 只改变什么 | 想验证什么 |
|---|---|---|
| 基线 | 当前实现 | 正确性、当前瓶颈 |
| A | 896 threads 改为较小 block，每线程多元素 | 资源占用、归约与并行度的取舍 |
| B | 合适对齐下向量化读写 | 指令数/访存效率是否改善 |
| C | 调整规约和参数处理 | 同步或冗余计算是否下降 |

不预先保证 B/C 更快。每次只改变可解释的因素，保留没有收益的结果，并说明原因。

对 Attention 补：自己的 v1/v2、padding 单独开关、不同 Q tile、普通 decode/分块 decode、最后再加 Tensor Core。不能同时改所有变量后把收益全部归给某一个技巧。

### P3：补形状范围和性能解释

区分三类测试：

- microbenchmark：固定输入形状，测单算子。
- 单请求推理：分别测 prefill、稳定 decode、端到端生成。
- 服务压测：控制并发、请求速率、输入输出长度和缓存状态，测 TTFT/TPOT/吞吐。

当前支持域内先测正确，不要在仅支持 512 的缓存上直接跑 4K。要研究长上下文，先扩展缓存与 kernel 的分块索引和边界。

时间统计优先用预先规定的重复次数、中位数和波动范围，避免看到异常后任意删最快或最慢值。既保存有收益的形状，也保存无收益的形状。

### P4：把材料整理成面试时能打开的一页证据

每个重点项目准备：问题、baseline、支持范围、核心代码、三项最重要的测量、一个失败尝试、一个尚未完成的方向。

你最值得先准备的两个例子：

1. Attention：一行 Q 的线程分工如何改变，哪些同步能省、哪些必须保留。
2. exllamav2：同一设备、同一形状下，切换计算路径带来了什么结果；把已证明的路径性能差异与尚未证明的指令级根因分开。

## 37. 下一次被问“为什么不用 Tensor Core”的完整回答模板

> 我手写的 Attention 核心目前没有使用 Tensor Core，QK 和 PV 还是普通浮点乘加与规约。这个阶段主要完成了分块、在线 softmax、融合执行和模型接入，也调整了 warp 分工。
>
> 对 prefill 来说，Tensor Core 是很明确的后续方向，我还没有做，不能把当前版本说成充分优化的实现。要继续做，需要把 QK 和 PV 映射到 MMA tile，处理中间 softmax 的布局与数值，再与现有版本和库实现做同形状比较。
>
> 对单 token decode，我会另外看上下文长度和 batch/head 提供的并行度，不能只因为有矩阵乘法就认定 Tensor Core 一定有收益。我当前能拿出来讲的是已经实现的线程分工和测量结果，未实现的部分我会明确说明。

如果已经补完实验，就把“还没有做”换成真实数据和结论；不要提前套用一个不存在的优化理由。

## 38. 复习检查表

- [ ] 不看代码画出 Qwen decoder，包含 QKV、RoPE、输出投影和残差。
- [ ] 写出 QK 与 PV 的 M/N/K，区分矩阵名字 K 和归约维 K。
- [ ] 用一个小例子推导 online softmax 的尺度修正。
- [ ] 解释 split-KV 的归一化合并，不能直接平均局部输出。
- [ ] 画出自己 prefill v1/v2 的 lane、warp、Q 行、KV 行对应关系。
- [ ] 区分自写 v2 与论文 FA2。
- [ ] 用地址算 sector，用地址算 bank；分别分析 FP16 和 FP32。
- [ ] 准确说出 Tensor Core 调用方式，以及当前代码哪里用、哪里没用。
- [ ] 解释 NSYS 能看单 kernel、NCU 提供更深指标，避免把两者说成互斥。
- [ ] 分清 kernel 时间、端到端时间、prefill、decode、capture。
- [ ] 解释一个优化为什么无收益，并提供可复现的对照。
- [ ] 能直接承认尚未实现的部分，同时说清已实现部分的价值和边界。

## 本地资料索引

- [面试记录](蔚蓝档案_mj.md)
- [Qwen 项目已有详细复盘](Qwen2.5推理引擎_项目细节.md)
- [量化方法与算子开发速成路径](量化方法与算子开发_四小时速成路径.md)
- [模型实现](../Qwen2_5_0_5B/my_qwen2.py)
- [生成与计时脚本](../Qwen2_5_0_5B/benchmark2.py)
- [历史时间记录](../Qwen2_5_0_5B/测试时间记录.txt)
- [历史 kernel 统计](../Qwen2_5_0_5B/vllm比对.txt)

外部依据已在相关回答中就近链接。涉及你当前实现的结论以本地代码为准，涉及算法标准含义的结论以论文和官方文档为准。
