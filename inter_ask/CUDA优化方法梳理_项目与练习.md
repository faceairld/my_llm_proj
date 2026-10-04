# CUDA 优化方法梳理：项目实现、练习与面试表述

整理日期：2026-09-22。

本文根据 SNN、Qwen2.5 推理引擎和 `cuda_pratice` 中的实际代码整理。重点补充多 stream、CUDA Graph、算子融合之外，自己已经使用过的 CUDA 优化方法。

**证据口径：**“代码已实现”不代表已经证明性能提升；历史性能数据来自已有实验记录，本次只核对源码，没有重新编译、测试或采集 profiler。练习中的改动不能直接说成已经集成到 Qwen/SNN 项目中。

## 1. 一页速览

| 优化方法 | 实际例子 | 主要解决的问题 | 当前证据 |
|---|---|---|---|
| Shared memory 分块 | 手写 GEMM、SNN 卷积、FlashAttention | 减少重复 global load，复用输入 | 源码存在 |
| 寄存器分块 | GEMM 每线程计算 8×8 输出 | 复用 shared memory 读出的数据，增加独立累加链 | 源码存在 |
| Shared memory 布局调整 | GEMM 的 B tile 转置存储、transpose 的 padding | 减少 bank conflict，改善读写布局 | 源码存在；GEMM 有历史实验记录 |
| 向量化访存 | RMSNorm 的 half2，向量加/reduction/softmax 的 float4 | 减少访存指令数量，提高搬运效率 | 源码存在；half2 有历史对照记录 |
| Warp shuffle 分层归约 | RMSNorm、reduction、attention | 减少 shared memory 中转及 block 同步 | 源码存在 |
| 循环展开 | GEMM、RMSNorm 练习 | 降低循环控制开销，帮助局部数组寄存器化 | 源码存在；有编译结果对照记录 |
| 寄存器保留输入 | Qwen RMSNorm 的 original_data | 避免归约后再次读取输入 | 源码存在 |
| 数学表达式调整 | 平方使用乘法/FMA、倒平方根使用 rsqrtf | 避免不必要的通用数学运算 | 源码存在；有历史对照记录 |
| 调整 warp 分工 | FA v1 的双 warp 一行改为 v2 的单 warp 一行 | 减少跨 warp 合并 | 两版源码存在 |
| 拆分 KV 维度增加并行度 | Flash-Decoding 分块再合并 | 增加小 batch decode 的可并行任务 | 源码存在，收益依赖形状 |
| 因果剪枝 | FA 限制 KV tile 遍历范围 | 跳过完全被 causal mask 遮挡的计算 | 源码存在 |
| 在线 attention 与延迟归一化 | FA 维护 max/sum/输出累加器 | 避免完整分数/概率矩阵落显存 | 源码存在 |

## 2. GEMM：shared memory 分块与寄存器分块

代码：[cuda_pratice/GEMM/gemm.cu](../cuda_pratice/GEMM/gemm.cu)，约第 101～174 行。

当前配置：

```text
BM = 128
BN = 128
BK = 8
每个 block：256 个线程
每个线程：负责 8×8 个输出
```

数据复用分成两层：

```text
global memory 中的 A/B
    ↓ 协作加载 tile
shared memory 中的 A/B tile
    ↓ 每线程加载局部片段
寄存器中的 A_temp[8]、B_temp[8]
    ↓ 组合成 64 次乘加
寄存器中的 data[8][8]
```

关键代码：

```cpp
float data[8][8] = {};
// 每个 K 位置，取 A 的 8 个值、B 的 8 个值。
data[y][x] = fmaf(A_temp[y], B_temp[x], data[y][x]);
```

与一个线程只计算一个输出相比，读入的 A/B 值能在多个输出之间复用，减少 shared memory 访问。多个输出累加器还提供了相对独立的计算链。

**代价：**寄存器需求增大，可能减少驻留 block 数，或引起 spilling。不是每线程输出块越大越好。

**面试表述：**

> 我做了两级数据复用：先把 A/B tile 加载到 shared memory，再让每个线程维护 8×8 的输出块。每次读取两组 8 个元素，用外积方式更新 64 个累加器，并结合寄存器占用和 occupancy 调整分块。

## 3. Shared memory 布局调整：减少 bank conflict

### 3.1 GEMM 中 B tile 的布局

同一文件中，B 的全局输入按 `[N,K]` 提供，加载到 shared memory 时改成 `[K,N]`：

```cpp
B_mem[offset_x * BN + offset_y] = b_data;
```

计算时：

```cpp
B_temp[i] = B_mem[idx_BK * BN + threadIdx.x * 8 + i];
```

这里要分析的是：**同一 warp 执行同一条 shared load 时，各线程地址映射到哪些 bank。** 不是看到“转置”或“连续”就能直接断言无冲突。

历史记录写有 B 布局调整前后的 bank 映射推导，以及 `857 → 2374 GFLOP/s` 的实验数字。该数字属于当时的特定实验，不能和其他形状、频率或版本的数值混作一条优化链；本次未复测。

记录：[CUDA 手写练习进度与待办](../cuda_pratice/进度与待办.md)，第四节 bank conflict。

### 3.2 Transpose 中的片内重排

代码：[cuda_pratice/Transposition/trans.cu](../cuda_pratice/Transposition/trans.cu)，约第 67 行。

```cpp
__shared__ half s_mem[32][66];
// 写入时使用偶数列。
s_mem[i / 32][(i % 32) * 2] = temp;
// 转置方向读取。
output[...] = s_mem[i % 32][(i / 32) * 2];
```

目的包括：

- 让读取 global input 的地址连续。
- 在 shared memory 中改变数据排列。
- 让写出 global output 的地址也连续。
- 通过 shared memory 的间隔与行步长调整 bank 映射。

**面试表述：**

> 我会从实际地址表达式推导 bank 映射，区分同地址广播和同 bank 不同 word 的冲突，再调整片内布局；同时检查是否改善了全局读写的合并访问。

## 4. 向量化访存：half2 与 float4

代码：

- [RMSNorm 练习](../cuda_pratice/RNSNorm/rnsnormal.cu)，约第 75～126 行。
- [向量加](../cuda_pratice/vector_add/vector_add.cu)，约第 70 行。
- [Reduction](../cuda_pratice/reduction/reduction2.cu)。
- [Softmax](../cuda_pratice/softmax/softmax.cu)。

RMSNorm 中使用：

```cpp
const half2* h2 = reinterpret_cast<const half2*>(input);
half2 temp = h2[index];
// 然后使用 temp.x、temp.y。
```

历史实验比较过“分别读取成员”与“先读取完整 half2 对象”，记录了加载指令数量变化和约 13%～14% 的收益。记录也明确说明，造成整体加速的具体微架构机制仍未完全解释清楚。

应区分：

```text
half2 / float4 向量化访存：打包读取或写出多个元素。
half2 算术：用相应指令同时处理两个 half。
Tensor Core：矩阵乘累加硬件路径。
```

这三件事不是同一件事。当前 RMSNorm 练习使用 half2 搬运，但主要用 float 运算累加。

**边界：**向量化需要处理对齐和尾部；源码使用 float4 不保证最终一定产生预期宽度的指令，应该检查 SASS。更宽也不保证更快，已有记录中就有 128-bit 方案不如 32-bit 方案的情况。

**面试表述：**

> 我尝试过 half2、float4 打包访存，并检查实际加载指令。RMSNorm 的一个对照中，整体读取 half2 减少了加载指令数并有可复现收益；但不会把这个结果直接推广到所有 kernel。

## 5. Warp shuffle 与分层归约

代码：[Qwen RMSNorm kernel](../Qwen2_5_0_5B/csrc/rmsnorm_kernel.cu)，约第 12～46 行。

结构：

```text
每个线程形成局部结果
    ↓
warp 内通过 __shfl_down_sync 归约
    ↓
每个 warp 写一个部分结果到 shared memory
    ↓
第一个 warp 合并部分结果
    ↓
发布最终结果给 block 内线程
```

Qwen RMSNorm 有 896 个线程，也就是 28 个 warp，因此用 `sum[28]` 保存 warp 部分结果。

优化点是让多数中间交换发生在 warp 内，减少 shared memory 中转和 block 级 barrier。跨 warp 的部分结果合并仍需要正确同步，不能简单删除所有 `__syncthreads()`。

**面试表述：**

> 我使用 warp shuffle 做局部归约，只把每个 warp 的一个结果写入 shared memory，再进行第二级归约，减少共享内存访问和 block 同步。

## 6. 循环展开与局部数组寄存器化

代码：[GEMM](../cuda_pratice/GEMM/gemm.cu) 第 139、162 行附近，以及 [RMSNorm 练习](../cuda_pratice/RNSNorm/rnsnormal.cu)。

```cpp
#pragma unroll
for (int k = 0; k < BK; ++k) {
    // 固定形状的片段计算
}
```

循环展开可能带来：

1. 减少循环计数、判断和跳转。
2. 让数组索引成为编译期常量，帮助局部数组拆成寄存器。
3. 给编译器提供更大的指令调度空间。

历史记录中，对比过展开前后的 stack frame、`LDL/STL` 和浮点计算指令。

**不能绝对化：**动态索引不必然导致 local memory，编译器可能进行变换；展开也可能增加代码体积和寄存器压力。应以 ptxas/SASS 和实测为准。

还要区分：

```text
局部数组本来就被放进 local memory
和
寄存器不足导致 spill 到 local memory
```

两者都可能产生 local load/store，但原因不同。看到 stack frame 不能直接断言寄存器溢出。

## 7. 保留输入与简化数学计算

### 7.1 Qwen RMSNorm：保留原始输入

```cpp
float data_in = static_cast<float>(input_data[idx]);
float original_data = data_in;
// 计算平方和及归约……
out_data[idx] = (original_data * rsqrtf(mean_square + eps)) * weight;
```

归约完成后直接复用线程局部的 `original_data`，避免再次读取 global input。是否最终一直保存在寄存器里，仍取决于编译结果和资源压力。

### 7.2 数学表达式

RMSNorm 练习用 `fmaf(x,x,sum)` 做平方累加，用 `rsqrtf` 得到倒平方根。历史记录有 `powf` 与乘法/FMA 写法的对照，记录中 `powf` 版本耗时约为另一版的 1.72 倍。

这个数字只适用于当时的编译和测试条件。不能说编译器在所有版本、所有选项下都不会优化 `powf(x,2)`。数学函数替换、FMA 与 fast math 也可能改变舍入行为，需要正确性验证。

## 8. 调整并行分工：减少通信，或增加可并行任务

### 8.1 Prefill attention：双 warp 一行改成单 warp 一行

代码：[FA v1](../Qwen2_5_0_5B/csrc/my_flash_attention.cu)、[FA v2](../Qwen2_5_0_5B/csrc/my_flash_attention_v2.cu)。

```text
v1：两个 warp 分担同一 Q 行对 KV 的计算，之后合并部分结果。
v2：一个 warp 负责一 Q 行，每个 lane 处理两个 KV 位置。
```

这样可以减少跨 warp 的统计量和输出结果合并。这是改变任务划分带来的优化，不只是把 shared memory 读写换成 shuffle。

两版都已经按 Q tile 分配 block。不能仅因为命名为 v1/v2，就说它们完整对应论文 FA1/FA2；当前手写 QK/PV 也不是 Tensor Core 实现。

### 8.2 Decode attention：沿 KV 拆成多个 block

代码：[my_flash_decoding.cu](../Qwen2_5_0_5B/csrc/my_flash_decoding.cu)。

```text
第一阶段：多个 block 分别计算不同 KV 分块。
第二阶段：结合各分块 max、sum 和输出部分结果，合并 attention。
```

目的是增加小 batch decode 下可调度的 block 数。代价是多一个 kernel、临时结果读写和合并工作。当前代码是固定分块实现，不能宣称已经覆盖任意长上下文；历史测试也不能支持“所有长度都更快”。

## 9. FlashAttention：跳过无效块与减少中间结果落显存

代码：[my_flash_attention_v2.cu](../Qwen2_5_0_5B/csrc/my_flash_attention_v2.cu)，约第 58～164 行。

### 9.1 因果剪枝

根据当前 Q tile 的位置限制 KV tile 遍历范围。完全处于未来的 KV 块不参与计算，边界块内部再判断具体位置是否被 mask。

### 9.2 Online softmax 与输出累加

处理每块 KV 时维护：

```text
m：已处理分数的最大值
l：按当前最大值缩放的指数和
u：按同样尺度累加的加权 V
```

新的 tile 到来后，修正旧统计量的尺度并累加；最终输出 `u/l`。

这样避免把完整的 QK 分数矩阵和 softmax 概率矩阵写到显存再读取。它减少的是中间数据搬运，并没有改变完整 dense attention 的二次计算复杂度。

### 9.3 延迟归一化

保持未归一化的输出累加器，最后统一除以归一化和，减少循环内部重复归一化。这个思路在你的两版 FA 中已经存在，不能全部归因于 v2 新增。

## 10. 前面已经讲过的主项目优化

这些也属于自己的优化经历，可与上面的补充项一起组织，但不要重复计算收益。

| 方法 | 项目中的例子 |
|---|---|
| Kernel 融合 | SNN 卷积/FC 与膜电位更新、发放、重置；Qwen RMSNorm |
| 权重布局重排 | SNN FC 将同一输入对应的不同输出权重连续存放 |
| Constant memory 广播 | SNN 小卷积权重，warp 内线程使用同一权重 |
| Pinned memory + 双 stream | SNN 不同 batch 的传输与计算流水 |
| 双套缓冲区 | SNN 各 stream 独立维护输入、中间状态和 workspace |
| CUDA Graph | 捕获固定执行序列，降低 CPU 逐项提交开销 |
| 库实现替换 | SNN FC1 使用 cuBLASLt，融合 bias 和历史膜电位累加 |
| 维度补齐 | FC1 的 120 个输出补到 128，再取有效输出 |
| 精度与计算模式 | SNN FC1 允许 TF32；Qwen 用 FP16 存储、部分统计量 FP32 累加 |
| 工作量裁剪 | Qwen 生成时只对最后一个位置执行 LM Head |

其中允许 TF32 不代表每次都已确认采用 Tensor Core；最后应以选中 kernel 或 profiler 为证据。双 stream 也需要时间线确认重叠，而不是仅根据 stream 数量判断。

## 11. 哪些目前不能说成已完成的优化

- **cp.async 流水线：**记录列为后续方向，当前检查的 GEMM 中没有对应实现。
- **手写 mma/ldmatrix、WMMA 或 CUTLASS GEMM：**本次检查未发现实现。通过 cuBLASLt 使用 Tensor Core 不等于写过 Tensor Core kernel。
- **GEMM 双缓冲：**历史记录提到试验，但当前 `gemm_duble_buffer.cu` 仍是单 shared-buffer 结构。可以说做过相关实验，不能仅凭文件名说当前版本实现了双缓冲。
- **所有形状都加速：**已有练习和项目存在固定形状假设，不支持这种结论。
- **给每种技巧都分配一个加速倍数：**缺少独立消融时，只能报告整体结果，不能把多项收益逐个编造或相乘。

旧进度文档中“float4 向量加还没写”“transpose 尚未完成”等状态已落后于当前源码；但出现了代码，也不自动代表完整正确性和性能验证已经完成。

## 12. 面试时如何组织

### 12.1 一段概括

> 我的优化主要分为数据复用、访存布局、线程分工和执行调度。数据复用方面，在 GEMM 中使用 shared memory 分块和每线程 8×8 寄存器分块；访存方面调整权重和 shared memory 布局，并尝试 half2/float4 向量化；归约使用 warp shuffle 加分层合并，attention 则通过调整 warp 分工减少跨 warp 通信。此外用在线 attention 减少中间结果落显存，执行层面使用 pinned memory、双 stream 和 CUDA Graph。具体收益会结合形状、编译结果和 profiling 验证。

### 12.2 优先展开三个例子

1. **GEMM 两级分块：**讲清 global → shared → register 的复用链条，以及寄存器占用的代价。
2. **Bank conflict：**从线程索引推导访问地址和 bank，再解释为什么改变布局。
3. **RMSNorm half2 对照：**讲源码、实际加载指令、正确性和性能记录，承认尚未完全解释的机制。

### 12.3 每项准备四个问题

- 原来的开销在哪里，有什么证据？
- 改动后少了哪些读写、指令、同步或重复计算？
- 增加了什么代价，例如寄存器、shared memory 或额外 kernel？
- 测试了哪些形状和精度，哪些结论还不能推广？

目标是能解释自己的取舍和证据，不是背出尽可能多的 CUDA 优化名词。
