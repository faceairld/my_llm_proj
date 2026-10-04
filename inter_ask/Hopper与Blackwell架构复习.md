# Hopper 与 Blackwell 架构复习：从 Ampere 到异步 Tensor Core

整理日期：2026-09-29。

本文围绕此前的 CUDA、FlashAttention 和矩阵乘讨论整理。主线为数据中心 GPU 的 A100（Ampere）→ H100（Hopper）→ B200（Blackwell）。RTX 3060 与 A100 都属于 Ampere，但硬件配置不同；RTX Blackwell 与 B200 的指令路径也不能直接等同。代码为教学示例，本次没有在对应 GPU 上编译运行。

## 1. 三代架构首先比较什么

对算子开发，重点看四件事：矩阵乘如何发起、数据如何搬运、累加结果存在哪里、不同计算怎样重叠。对多卡系统，还要看显存容量/带宽、互联和资源隔离。

| 维度 | Ampere：A100 | Hopper：H100 | Blackwell：B200 |
| --- | --- | --- | --- |
| Tensor Core | 第三代 | 第四代，支持 FP8 | 第五代，进一步支持 FP4/FP6 等低精度路径 |
| 代表性矩阵指令 | warp 级 `mma.sync` | warpgroup 级异步 WGMMA | 单线程发起的异步 `tcgen05.mma` |
| 上述指令的累加器 | 寄存器 | 寄存器 | TMEM |
| 数据搬运 | `cp.async` | 新增 TMA | 延续 TMA，与新 MMA 流水线配合 |
| 跨 block 协作 | 主要借助全局内存等机制 | Thread Block Cluster、分布式共享内存 | 进一步提供 2-CTA MMA 等机制 |

表中列的是代表性路径，不表示某代只支持这一种指令。支持的数据类型、矩阵形状、吞吐和配置限制要以具体 GPU 与指令为准。支持低精度也不意味着所有模型都能无损地直接切换低精度。

参考：[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)、[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)、[Blackwell GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。

## 2. Ampere 已经有的基础能力

### 2.1 Tensor Core 与 warp 级 MMA

这里讨论的 `mma.sync`、`wmma::mma_sync` 是 Tensor Core 矩阵运算路径：一个 warp 的 32 个线程按规定协作完成一次矩阵乘加，结果分布在线程寄存器中。较大的 GEMM tile 可由多次矩阵指令和多个 warp 的计算组成。

普通代码 `acc += a * b` 通常生成普通浮点运算指令，由 CUDA Core 执行；并不会因为 32 个线程同时执行，就自动变成 Tensor Core GEMM。

MMA 是 Matrix Multiply-Accumulate：

```text
D = A × B + C
```

不要把逻辑线程协作粒度理解为物理硬件的独占分配：一个 warp 不等于独占一个 Tensor Core，一个线程也不固定绑定一个 CUDA Core。

### 2.2 异步搬运不是 Hopper 才有

Ampere 的 `cp.async` 已支持 global memory → shared memory 异步拷贝，可避免传统搬运中经线程寄存器中转数据，并与计算重叠。

因此，双缓冲、异步拷贝、计算与搬运重叠不是 Hopper 才首次出现。Hopper 的 TMA 是在此基础上提供更强的批量张量搬运能力。

### 2.3 精度与稀疏

Ampere 支持 TF32、BF16 等 Tensor Core 路径，以及满足特定模式时的结构化稀疏加速。TF32 可用于部分以 FP32 张量为输入的矩阵乘，但乘法输入精度不等于完整 FP32；不能说 Tensor Core 统一只用 TF32。

## 3. Hopper：TMA、WGMMA 和线程分工

### 3.1 TMA 是什么

TMA：Tensor Memory Accelerator，张量内存加速器。它根据张量描述符和坐标处理批量搬运，支持最高 5 维张量，提供 global/shared 等存储之间的相应传输能力。

```text
线程准备描述符与坐标、发起搬运
                ↓
TMA 处理批量地址计算与数据传输
                ↓
通过完成通知和同步机制，让消费者安全使用数据
```

“不需要 thread 搬运”的准确说法是：不需要大量线程逐元素完成地址计算与数据搬运，但仍需要线程发起操作、维护流水线、等待正确的完成条件。

这使 producer-consumer 分工更方便：

```text
producer：提交下一块 K/V 的 TMA 搬运
consumer：计算已经到达 shared memory 的当前块
```

参考：[CUDA 异步拷贝与 TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

### 3.2 WGMMA 的全称与协作粒度

WGMMA：Warpgroup Matrix Multiply-Accumulate，warpgroup 级矩阵乘加。

```text
1 个 warp = 32 个线程
1 个 warpgroup = 4 个连续 warp = 128 个线程
```

128 个线程按指令要求共同执行一次 WGMMA；不是每个 warp 各自重复算完整矩阵，也不是四个 warp 分别绑定四个专属 Tensor Core。

其异步性发生在 GPU kernel 内部：提交矩阵乘后，可以继续执行不依赖该结果的指令；读取累加器或复用相关存储前，再按要求等待。这与 CPU 异步发射整个 kernel 是两个不同层次。

Hopper WGMMA 的 B 操作数来自 shared memory，A 可按指令形式来自 shared memory 或寄存器，累加器位于寄存器。相关布局、同步和生命周期有严格要求。

参考：[NVIDIA WGMMA 编程说明](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html)。

### 3.3 指令 tile 与整个 GEMM tile 不同

例如 FP16 输入的一种指令形状为 `m64n64k16`：

```text
[64,16] × [16,64] → 累加到 [64,64]
```

如果实际计算 `[64,128] × [128,64]`，可以沿 K 维拆成 8 次这样的乘加。不是一条 WGMMA 自动完成任意大小矩阵乘。

可用的 M/N/K 组合由指令与数据类型限定，不能任意填。WGMMA 的优势包括较大的指令级协作块、异步执行和 shared-memory 操作数路径；Ampere 本来就能通过多个 warp 和多条 MMA 组成较大的 block tile，因此不能只说“以前不能算大 tile”。

### 3.4 FP8、Transformer Engine 与其他能力

Hopper 增加 FP8 Tensor Core 能力，配合 Transformer Engine 的软件与硬件支持管理缩放和精度选择。Transformer Engine 不是一个把整个 Transformer 全部包办的单独计算单元。

其他变化包括面向动态规划的 DPX 指令、更强的 NVLink、显存系统及隔离能力。具体容量、带宽与可用特性取决于产品形态，不应把某张 H100 的参数推广到所有 Hopper 产品。

参考：[Hopper 架构介绍](https://developer.nvidia.com/blog/?p=45555)。

## 4. 为什么要 Ping-pong：两组都做矩阵乘不行吗

两个 warpgroup 当然可以都发起矩阵乘。Ping-pong 是优化调度策略，不是硬件规定同一时刻只能一组做矩阵乘。

两组共享 SM 上有限的 Tensor Core 吞吐；多一组提交工作，不等于多出一套 Tensor Core。如果两组阶段很接近，可能先一起争用矩阵计算资源，再一起做 Softmax，让 Tensor Core 暂时缺少后续工作。

```text
阶段相近：
    WG-A 矩阵乘 + WG-B 矩阵乘
    随后 WG-A Softmax + WG-B Softmax

错开阶段：
    WG-A 矩阵乘 + WG-B Softmax
    随后 WG-A Softmax + WG-B 矩阵乘
```

Softmax 的普通运算和指数计算使用其他执行资源，所以错开阶段有机会提高整体利用率。硬件调度器原本也会尝试重叠；软件同步进一步引导顺序，真实时间线不会和示意图完全一样。

### 4.1 两组是否共用同一份中间结果

以两组处理不同 Q 子块为例：K/V tile 可以共享，但各自的 score、max/sum 和输出累加器不同。WG-B 不是替 WG-A 做 Softmax；两组分别完成自己的输出。

相关 K/V 缓冲区必须等消费者用完，才能覆盖。这里的 Ping-pong 指计算阶段交替，不要与“两块内存轮流读写”的双缓冲完全等同。

### 4.2 组内跨迭代流水线是否与 Ping-pong 矛盾

不矛盾。固定一个 Q 子块，可以在处理当前 score 的 Softmax 时，异步计算下一个 Key tile 的 score：

```text
已经得到 S_0 = Q × K_0ᵀ
提交 S_1 = Q × K_1ᵀ
处理 S_0 的 Softmax，同时 Tensor Core 计算 S_1
需要 S_1 时再等待
```

这里迭代的是 Key 序列分块，不是生成下一 token，也不是 head_dim 的归约分块。

| 机制 | 独立工作来自哪里 |
| --- | --- |
| 组间 Ping-pong | 另一组负责的 Q 子块 |
| 组内跨迭代流水线 | 本组下一个 K/V tile |

异步并不保证总有独立工作，也不保证 Softmax 与矩阵乘耗时匹配。两种策略可以结合；若其中一种已充分利用硬件，另一种可能收益不大，甚至因寄存器压力和同步而变慢。

参考：[FA3 作者讲解](https://pytorch.org/blog/flashattention-3/)、[FA3 论文](https://arxiv.org/abs/2407.08608)。

## 5. Thread Block Cluster 与分布式共享内存

### 5.1 cluster 是什么层次

CTA（Cooperative Thread Array）就是 CUDA thread block。

```text
Grid
  → Cluster
    → Block / CTA
      → Warp
        → Thread
```

Hopper 引入可选的 cluster 层次，同一 cluster 的 block 得到同一 GPC 内的协同调度保证，并可以进行硬件支持的同步。cluster 中的 block 可访问彼此的 shared memory，这种能力称为 Distributed Shared Memory（分布式共享内存，DSMEM）。

每个 block 仍有各自的 shared memory 分配，不是任意 GPU block 都能直接共享一块普通数组，也不是自动得到一个连续大数组。远程访问需要地址映射及正确同步。

### 5.2 编译期设置 cluster 的代码示例

下面让两个 block 组成一组，各自读取另一个 block 的 shared memory。需要支持相应 cluster 功能的设备与工具链，RTX 3060 不支持这个示例路径。

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1)
read_neighbor(int* output)
{
    __shared__ int data[128];

    auto cluster = cg::this_cluster();
    unsigned rank = cluster.block_rank();  // cluster 内编号 0 或 1

    data[threadIdx.x] = 100 * blockIdx.x + threadIdx.x;
    cluster.sync();  // 全部 block 初始化完成后再读取

    unsigned peer = 1 - rank;
    int* peer_data = cluster.map_shared_rank(data, peer);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = peer_data[0];
    }

    // 确保对方完成远程访问后，本 block 才能退出。
    cluster.sync();
}

// CPU 侧调用片段：d_output 已在设备上分配至少 4 个 int。
// read_neighbor<<<4, 128>>>(d_output);
```

```text
4 个 block，每个 128 线程
每个 cluster 有 2 个 block，共 2 个 cluster

cluster 0：block 0、block 1
cluster 1：block 2、block 3

上述调用的预期 output：[100, 0, 300, 200]
```

注意：

- `map_shared_rank(data, peer)` 返回目标 block 中对应 shared 数组的映射地址，不是把整块数据复制过来。
- `__syncthreads()` 只同步本 block，不能代替 `cluster.sync()`。
- 所有相关线程必须按要求参与同步，不能只有 thread 0 调用 cluster barrier。
- 在远程访问完成前，提供 shared memory 的 block 不能退出。
- 此写法的 grid 维度仍按 block 数计，必须满足 cluster 维度的整除要求；示例数组还要求每个 block 恰好 128 线程。
- 正式程序应检查设备支持、启动返回值和完成状态；本示例省略分配与错误处理，只展示机制。

### 5.3 运行时设置 cluster

若 kernel 声明没有固定 `__cluster_dims__`，可以通过 `cudaLaunchKernelEx` 设置。以下为 host 侧配置片段，`runtime_cluster_kernel` 应是另一个未固定 cluster 大小、实现相同读取逻辑的 kernel：

```cpp
cudaLaunchConfig_t config = {};
config.gridDim = dim3(4, 1, 1);
config.blockDim = dim3(128, 1, 1);

cudaLaunchAttribute attr = {};
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim.x = 2;
attr.val.clusterDim.y = 1;
attr.val.clusterDim.z = 1;

config.attrs = &attr;
config.numAttrs = 1;

// cudaLaunchKernelEx(&config, runtime_cluster_kernel, d_output);
```

编译期固定的 cluster 大小不能在启动时随意覆盖。cluster 大小也不是越大越好，会影响可驻留数量和调度。

参考：[CUDA Thread Block Clusters](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#thread-block-clusters)。

## 6. Blackwell：TMEM 与单线程发起 MMA

### 6.1 TMEM 不等于 TMA

| 名称 | 全称 | 用途 |
| --- | --- | --- |
| TMA | Tensor Memory Accelerator | 搬运数据 |
| TMEM | Tensor Memory | Tensor Core 使用的专用片上存储 |

以 B200 的新矩阵路径为例，MMA 的累加器可直接保存在 TMEM，减轻线程寄存器压力，有利于更大的 tile 和更深的流水线。后续 Softmax 等普通计算仍可能把数据从 TMEM 读入线程寄存器，并不是所有运算都在 TMEM 中完成。

### 6.2 单线程发起不等于单线程完成全部工作

```text
某个线程发出 tcgen05.mma
    ↓
Tensor Core 异步执行矩阵乘加
    ↓
结果累加到 TMEM
    ↓
完成同步后，线程进行结果处理
```

它改变的是 MMA 的发起粒度与累加器存储，不是把全部计算变成一个普通线程执行。搬运、TMEM 管理、结果读取和同步仍有各自的协作要求，不能把所有相关指令都理解为单线程指令。

### 6.3 更低精度与系统改进

Blackwell 增加更低位宽的矩阵计算路径以及微块缩放支持。低位宽可减少数据量、提高吞吐，但缩放、类型转换、精度和算子支持会影响实际收益。

数据中心产品也增强 HBM 和 NVLink 等能力，面向大模型容量及多卡扩展。具体规格应按产品查，不用把所有型号的数值混背在一起。

参考：[tcgen05 编程说明](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)、[Blackwell 架构介绍](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)。

## 7. 2-CTA MMA 为什么不同于普通多 block GEMM

### 7.1 普通多 block 也能算一个大矩阵

例如计算：

```text
A：[256,K]
B：[K,256]
C：[256,256]
```

两个独立 block 沿 M 维划分输出时：

```text
block 0：A 上半 [128,K] + 完整 B [K,256] → C 上半
block 1：A 下半 [128,K] + 完整 B [K,256] → C 下半
```

实际沿 K 循环装载 tile。对每个 K tile，两个 block 通常都在各自 shared memory 中准备完整的对应 B tile，独立执行矩阵指令。

### 7.2 双 CTA 模式在硬件矩阵指令层面协作

一种 2-CTA 布局如下：

```text
CTA 0 的 shared memory：A 上半、B 左半
CTA 1 的 shared memory：A 下半、B 右半

两个 CTA 配对执行 2-CTA MMA

CTA 0 的 TMEM：C 上半
CTA 1 的 TMEM：C 下半
```

每个输出行仍需完整 B。硬件根据双 CTA 模式使用双方提供的数据，让两边的计算使用完整操作数；每个 CTA 不必在本地重复准备整个 B tile。

新模式不是“让两个普通 block 分别发一条单 CTA MMA”。需要配置相应 cluster、`tcgen05.mma` 的 `cta_group::2` 形式，以及匹配的 TMEM 和同步机制。指令由配对 CTA 中符合规则的单线程发起，双方共同提供资源并保持相应生命周期。

### 7.3 优势与限制

可能的收益：减少重复的 B tile 准备、降低每 CTA 的本地存储负担、组织更大的合作计算块，在某些场景缓解 shared-memory 流量瓶颈。

不能直接宣称全局显存流量减半或性能翻倍：普通 block 的重复读取可能命中缓存，双 CTA 也有跨 SM 数据交换、配对驻留和同步成本，实际收益取决于 shape、数据布局和瓶颈。

还要区分：

- cluster 配成 2 个 block，不会自动把普通代码变成 2-CTA MMA。
- 2-CTA MMA 是同一 GPU 内的协作，不是多 GPU TP。
- 上述例子不是 split-K：不是两组各算一部分 K 后再归约，而是协作提供操作数、分别保存输出行。
- 两 CTA 的合作 tile 也不一定是一条指令就全部完成，实际可能继续沿 K 等维度循环。

参考：[NVIDIA 双 CTA GEMM 教程](https://nvidia.github.io/tilus/latest/tutorials/matmul-blackwell/v6.html)、[Blackwell GEMM 说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。

## 8. 为什么用 cuBLASLt 时没设置 warp 和 thread

之前项目中的调用层次是：

```text
CPU main()
  → 配置矩阵布局、计算类型、bias、workspace
  → 通过 heuristic 选择算法
  → cublasLtMatmul(...)
  → 库内部启动 GPU kernel
  → kernel 中的线程执行矩阵指令
```

线程组织、tile、流水线等由所选算法的内部实现负责，所以你没写 `kernel<<<grid,block>>>`。CPU 主函数不是直接驱动某个 Tensor Core 单元做乘法。

自己写 WMMA/WGMMA/tcgen05 kernel 时，才需要显式设计参与线程、输入布局、指令形状、缓冲区和同步。可以控制编程结构，不能直接指定某个 warp 独占几个物理 Tensor Core。

设置允许 TF32，也不意味着某个 cuBLASLt 调用必然采用预期 Tensor Core 算法；仍应通过实际选用的算法与 profiling 确认。

参考：[cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/)。

## 9. 架构变化怎样对应 FA2、FA3、FA4

```text
Ampere：warp 级 MMA + 异步拷贝
    → FA2：优化分块、工作划分、归约和非矩阵乘开销

Hopper：TMA + WGMMA
    → FA3：充分重叠搬运、矩阵乘、Softmax，并支持 FP8 路径

Blackwell B200：TMEM + tcgen05.mma + 更高矩阵吞吐
    → FA4：重构流水线，缓解指数计算与 shared-memory 等新瓶颈
```

这些是理解论文重点的对应关系，不是说各代 FlashAttention 只能用于这一种 GPU。

FA4 前向使用 FMA 多项式近似分担部分指数计算，并采用条件重缩放减少非矩阵乘操作；反向利用 TMEM 与双 CTA 等手段减少部分存储流量和归约。Tensor Core 更快，不代表整个 Attention 等比例加速。

关于指数近似、online softmax 延迟重缩放为什么不需保存每个 tile 的 max，见 [byte_mj_解答.md](byte_mj_解答.md) 第 18—19 节。那里的 r、L、U 示例用于说明保持共同指数基准的不变量。

参考：[FA4 作者说明](https://tridao.me/blog/2026/flash4/)。

## 10. 型号边界与面试口述

### 10.1 不要只看架构商品名

- RTX 3060 可用于学习 warp 级矩阵指令、shared-memory tiling 和 Ampere 异步拷贝等。
- Hopper 的 TMA、WGMMA、cluster 要使用支持这些特性的硬件和编译目标。
- 本文 TMEM、tcgen05 和双 CTA MMA 的主线是 B200 等数据中心 Blackwell。
- RTX 50 系列 SM120 使用不同矩阵指令路径，不能直接套用 B200 SM100 的 kernel；同为 Blackwell 不代表支持完全相同的编程接口。

参考：[CUTLASS SM120 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。

### 10.2 简短回答

> Ampere 已有 warp 级 Tensor Core MMA 和异步 global-to-shared 拷贝。Hopper 进一步引入 TMA、warpgroup 级异步 WGMMA，以及 cluster 和分布式共享内存，使搬运与计算、矩阵乘与 Softmax 更容易重叠。B200 的 Blackwell 又引入 TMEM 和单线程发起的 tcgen05 MMA，减轻寄存器压力，并支持双 CTA 协作。低精度和矩阵吞吐持续提升后，优化重点也会转向指数计算、shared-memory 流量与同步调度。具体使用时必须区分数据中心和消费级产品的指令能力。
