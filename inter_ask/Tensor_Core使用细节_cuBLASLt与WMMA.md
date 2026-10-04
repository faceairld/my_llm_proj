# Tensor Core 使用细节：从 SNN 的 cuBLASLt 到手写 WMMA

整理日期：2026-09-26。

依据：本次讨论、当前本地 SNN 代码，以及文末 NVIDIA 官方资料。本文区分项目已配置的行为与需要 profiling 才能确认的行为；没有重新运行 SNN，也没有实测本文 WMMA 示例的性能。公式采用纯文本。

## 阅读导航

- SNN 当时如何调用、启发式是否保证 Tensor Core：第 1～3 节。
- 行列主序、转置、leading dimension：第 4 节。
- thread/warp 与 Tensor Core 的关系：第 5 节。
- fragment 参数、数据类型、MNK 形状：第 6～7 节。
- 32×32 GEMM 完整分块示例：第 8 节。
- load/store 指针、对齐和累加器初始化：第 9 节。
- 乘积、累加器、输出精度：第 10 节。
- sync/async、其他常用操作：第 11～12 节。
- 易错点与面试口述：第 13～14 节。

## 1. 有哪些使用 Tensor Core 的层次

| 方式 | 由自己控制什么 | 主要用途 |
|---|---|---|
| cuBLAS / cuBLASLt | 数学问题、类型、布局、算法选择约束等 | 调用已有高性能 GEMM；本项目使用 cuBLASLt |
| CUTLASS / CuTe | 通过模板与抽象组织 tile、流水、布局和 epilogue | 构建或定制高性能矩阵算子 |
| CUDA WMMA | warp 负责的 tile、数据加载、K 循环和输出处理 | 学习和编写显式 Tensor Core kernel |
| PTX `mma.sync` 等 | 更底层的矩阵指令与寄存器操作数组织 | 更精细的架构相关优化 |

显式使用 Tensor Core 指令不等于自动比库快。数据搬运、复用、寄存器/shared memory 占用、边界处理和流水都需要设计。WMMA 不是通过 cuBLASLt 的启发式间接调用，但它仍会由编译器映射为目标架构指令；不能假设一次 WMMA 等于一条硬件指令或一个时钟周期。

## 2. SNN 中实际使用的 cuBLASLt 路径

参考：[SNN_proj1.cu](../SNN_proj1.cu)、[SNN_proj1_dstream.cu](../SNN_proj1_dstream.cu)。

### 2.1 代码实际配置了什么

当前 `SNN_proj1.cu` 的核心设置是：

```cpp
cublasLtMatmulDescCreate(
    &fc3_op_desc,
    CUBLAS_COMPUTE_32F_FAST_TF32,
    CUDA_R_32F);
```

- `CUBLAS_COMPUTE_32F_FAST_TF32`：允许 FP32 输入使用 TF32 Tensor Core 计算路径。
- 第二个参数类型位置中的 `CUDA_R_32F` 是 scale type，即 alpha/beta 的类型；A/B/C/D 的存储类型另外由 matrix layout 描述。
- 变量虽然叫 `fc3_op_desc`，这里实际用于 FC1。
- 输入/权重/输出 layout 都使用 FP32，配置 bias epilogue。
- 启发式查询发生在 Graph 捕获之前，只请求一个候选，后续传入同一个 `heuristic_resuit.algo`。
- 两条 stream 分别使用独立的 32 MiB workspace。

这能证明代码允许 TF32 并使用了 cuBLASLt，不能单独证明每次实际选中的实现一定使用 Tensor Core。需要结合 profiler 或算法实现属性核验。

### 2.2 描述符各管什么

| 对象 | 作用 |
|---|---|
| `cublasLtHandle_t` | cuBLASLt 库句柄 |
| `cublasLtMatmulDesc_t` | 运算配置：计算类型、alpha/beta 类型、转置、epilogue 等 |
| `cublasLtMatrixLayout_t` | 各矩阵的存储类型、行列数、leading dimension、布局等 |
| `cublasLtMatmulPreference_t` | 算法搜索偏好与约束，例如最大 workspace、指针对齐保证等 |
| `cublasLtMatmulHeuristicResult_t` | 候选算法及相关结果信息 |

`cublasLtMatmulDescSetAttribute(desc, attribute, buffer, size)` 中：

- `desc` 是被修改的描述符。
- `attribute` 指定修改哪个属性，不是修改描述符的 C++ 类型。
- `buffer` 指向该属性的值；不同属性可能是 enum、整数、指针等。
- `size` 是值的字节数，用于这种通用接口的类型/大小校验。

例如 `cublasOperation_t` 和 `cublasLtEpilogue_t` 是不同配置项的枚举类型。统一的 setter 使用 `void*` 风格的缓冲区，不能从这个指针直接推导数据类型和长度。

### 2.3 为什么 bias 指针在描述符中，A/B 指针却不在

```cpp
cublasLtMatmulDescSetAttribute(
    fc3_op_desc,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
    &d_fc1_b,
    sizeof(d_fc1_b));
```

`d_fc1_b` 是 CPU 侧变量，保存 GPU bias 地址。`&d_fc1_b` 让 setter 读取这个地址值；这里传的是指针大小，不是整个 bias 张量大小，也不是把 CPU bias 数据复制到 GPU。

A/B/C/D 地址由执行函数 `cublasLtMatmul()` 单独接收；bias 是可选 epilogue 的附加输入，因此这个 API 将其挂在运算描述符上。这是接口设计，不是 Tensor Core 硬件要求 bias 必须采用特殊地址机制。

Epilogue 指矩阵乘后的输出处理，可包含 bias、部分激活等，具体组合由版本和算法支持决定。本项目使用 bias epilogue，后续脉冲/复位操作仍由 `fc1_sum2out` 完成。

### 2.4 Preference、workspace 与启发式

`cudaMalloc` 分配实际 scratch 空间；`MAX_WORKSPACE_BYTES` 告诉算法搜索允许使用多少；真正执行时仍需把 workspace 指针和大小传给 Matmul。Preference 自己不分配显存，也不是每次 GEMM 必须传入的执行对象。

Preference 的主要职责就是算法选择约束，并不只有 workspace 一项。具体属性以所用 CUDA 版本为准。声明某种指针对齐保证不会自动把实际指针变成那个对齐。

`AlgoGetHeuristic(..., 1, &result, &count)` 中的 `1` 是最多请求一个候选；`count` 是返回数量，不是耗时。若没有合法候选，应停止该执行路径或选择合法回退方案。当前代码只打印 `no return result`，不应将这种处理当作完善的错误恢复。

## 3. “允许 TF32”不等于强制；启发式也不是每次随机选

要分开三个问题：

1. **是否允许 TF32？** 由计算配置、运行环境等决定。
2. **当前算法是否采用 Tensor Core？** 看算法信息和实际指令/指标。
3. **它是否最快？** 需要实测，启发式只是预测。

同一环境下查询一次、保存算法、后续重复传入，并不是每次 GEMM 都重新随机选择。GPU、CUDA 版本、尺寸、布局、对齐、epilogue 或 workspace 改变，则可能需要重新选型。

更可靠的工程流程：请求多个合法候选 → 核验状态及实际 workspace/指针条件 → 预热 → 检查结果 → CUDA Event 计时 → 保存合适算法。Nsight Systems 适合看整体时间线；确认 Tensor Core 指令和执行指标通常进一步用 Nsight Compute/SASS。cuBLASLt 的 numerical implementation flags 也能辅助区分 FMA/HMMA 等路径。

### 3.1 120 改为 128 不能直接证明什么

两个本地文件都已使用 `CUBLAS_COMPUTE_32F_FAST_TF32`。`SNN_proj1_dstream.cu` 的 FC1 layout 使用 120，`SNN_proj1.cu` 使用 128。

128 padding 可能改善 tile 适配、步长/地址对齐和候选算法选择。但不能说“120 绝对不能用 Tensor Core，128 才能用”，也不能仅据现存代码断言当时性能变化一定来自 SIMT→Tensor Core 切换。需要当时的运行记录或重新做对照 profiling。

当前模型有效 FC1 输出仍是 120；128 是计算缓冲区的 padding，不等于将网络结构改成了 128 个有效神经元。

## 4. SNN 的矩阵转置与内存布局

### 4.1 原数学运算与库看到的运算

```text
原运算：X[64,256] @ W[256,128] = Y[64,128]

转置恒等式：Y^T = W^T @ X^T

库视角：W^T[128,256] @ X^T[256,64] = Y^T[128,64]
```

对应本地 layout：

```cpp
cublasLtMatrixLayoutCreate(&sumdesc,  CUDA_R_32F, 256, 64, 256);
cublasLtMatrixLayoutCreate(&convdesc, CUDA_R_32F, 128, 256, 128);
cublasLtMatrixLayoutCreate(&outdesc,  CUDA_R_32F, 128, 64, 128);
```

默认列主序，实际 Matmul 的 A 是权重 `convdesc`，B 是输入 `sumdesc`；二者的 `trans` 都是 `CUBLAS_OP_N`。

行主序的 X[64,256] 与列主序的 X^T[256,64] 可以使用同一段数据，因此这一步可以是描述方式与操作数顺序的调整，而不是每次 GEMM 都启动一个转置 kernel。权重加载时的重排是另一件事。

### 4.2 列主序不是“矩阵乘按列相乘”

数学定义始终是：`C[i,j] = sum_k A[i,k] * B[k,j]`。

```text
行主序地址：base + row * ld + col
列主序地址：base + row + col * ld
```

布局规定逻辑元素在哪里；并不规定计算变成“列与列相乘”，也不规定硬件必须逐列串行读取。实际 kernel 可以合作加载 tile，再通过 shared memory 和寄存器组织计算。

`MatrixLayoutCreate` 最后一个参数是 leading dimension，单位是元素数。列主序通常是相邻列起点间隔，必须足以容纳一列；行主序则是相邻行起点间隔。

因此，默认列主序的 `[256,64]` 不能写 `ld=64`；那会让列重叠。正确紧凑列主序为 `ld=256`。若显式采用行主序 `[256,64]`，紧凑布局才是 `ld=64`，但表达的是另一种矩阵解释。

Tensor Core 不要求所有矩阵只能列主序。cuBLASLt/WMMA 支持的布局需按具体接口与算法核验。

## 5. thread、warp、block 与 Tensor Core 的关系

thread/warp/block 是程序执行的组织方式；CUDA Core、Tensor Core 是 SM 内的硬件执行单元。线程不永久绑定一个 CUDA Core，也不需要程序员为它选择一个 Tensor Core 编号。

```text
CPU：kernel<<<grid, block>>>()
           ↓
block 被安排到 SM
           ↓
线程组成 warp，执行编译后的指令
  ├─ 普通标量运算 → 对应算术执行单元
  ├─ load/store   → 访存执行单元
  └─ 矩阵乘加     → Tensor Core 执行通路
```

普通 SIMT GEMM 由程序员规定每个线程负责哪些输出元素；WMMA 由程序员规定一个 warp 负责哪些输出 tile，tile 内的片段映射由接口处理。

32 个线程共同执行 `mma_sync`，逻辑上完成一个 warp 级矩阵乘加，不是 32 次重复的完整矩阵乘，也不是只让 lane 0 算。不能写成只有 `threadIdx.x == 0` 才调用 WMMA。分支中的 WMMA 调用必须满足整个 warp 一致参与的要求。

一个 warp 可以维护多个输出 fragment，循环计算更大的区域；“一个 warp 只算一个 16×16”是示例中的工作划分，不是永久限制。

## 6. fragment 声明的每个参数

```cpp
wmma::fragment<
    wmma::matrix_a,
    16, 16, 16,
    half,
    wmma::row_major
> a;
```

通用形式：`fragment<用途, M, N, K, 元素类型, 布局>`。

| 参数 | 含义 |
|---|---|
| `matrix_a` | 左操作数 A，逻辑形状 M×K |
| `matrix_b` | 右操作数 B，逻辑形状 K×N |
| `accumulator` | 累加输入/输出，逻辑形状 M×N |
| M/N/K | 整个 warp 级矩阵乘加的尺寸，不是线程数/blockDim |
| `half` / `float` 等 | 该 fragment 对应的元素/精度类型 |
| `row_major` / `col_major` | A/B 在内存中的布局解释 |

`matrix_a`、`matrix_b`、`accumulator` 是 WMMA 库提供的类型标签，不是 C++ 语言关键字。`fragment<...>` 是模板实例化的类型，`a` 才是自己起的变量名；声明本身不会发射 GEMM。

累加器常写成：

```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
```

累加器不在模板中指定行列主序；从内存加载/向内存写回时再指定。

每个线程拥有自己的 fragment 变量，通常保存于寄存器，共同表示整个 tile。元素到线程寄存器的映射不透明，不能假设 thread 0 持有第一行，也不能将 `acc.x[i]` 的 i 直接当作矩阵列号。

## 7. MNK 与数据类型不能任意搭配

以下是经典 WMMA 的常用组合，不是所有架构的全部 Tensor Core 指令能力表：

| A/B 输入类型 | 累加类型 | M×N×K |
|---|---|---|
| FP16 | FP32 或 FP16 | 16×16×16、32×8×16、8×32×16 |
| BF16 | FP32 | 16×16×16、32×8×16、8×32×16 |
| TF32 | FP32 | 16×16×8 |
| INT8/UINT8 | INT32 | 16×16×16、32×8×16、8×32×16 |

还要核对目标 GPU 支持。不能任意填写 17×23×31，也不能因硬件支持某个新精度，就认为可以直接塞进经典 WMMA 模板；例如 FP8 的可用指令和接口应另按架构核验。

同一次运算的 A/B/累加器 fragment 必须使用匹配的 M/N/K。具体矩阵大小分别是：A[M,K]、B[K,N]、输出[M,N]。

### 7.1 长方形 tile 的用途

```text
M=32,N=8,K=16：A[32,16] @ B[16,8] → C[32,8]
M=8,N=32,K=16：A[8,16] @ B[16,32] → C[8,32]
```

前者是高瘦的输出 tile，后者是矮胖的输出 tile。例如 C[128,8] 可考虑 32×8，C[8,128] 可考虑 8×32，减少无效边界计算。但选择还取决于 block/warp 分工、访存、数据复用和实际效率；方阵也可能采用长方形 warp tile。

TF32 使用 `wmma::precision::tf32`，输入仍以 float 存储，但需按接口要求用 `__float_to_tf32()` 转到 TF32 精度。这些转换与加载由直接写 WMMA 的程序负责；使用 cuBLASLt 时由库路径组织。

## 8. 128 个线程计算 32×32 GEMM：完整教学示例

设一个 block=128 threads=4 warps。A、B 都为行主序 FP16 的 32×32，C 为行主序 FP32 的 32×32。每个 warp 负责 C 的一个 16×16 tile。

```text
              C 列0～15       C 列16～31
行0～15       warp0：C00      warp1：C01
行16～31      warp2：C10      warp3：C11

warp0 的完整任务：A[0:16,0:32] @ B[0:32,0:16]
                 [16,32]     @ [32,16]

拆 K：
C00 = A[0:16,0:16]  @ B[0:16,0:16]
    + A[0:16,16:32] @ B[16:32,0:16]
```

所以每个 warp 调两次 `mma_sync`，并保留同一个 FP32 累加器。无需每次写回显存，也不需要跨 warp 合并，因为四个 warp 的输出区域互不重叠。

下面是尺寸固定的教学 kernel，不是通用/调优后的 GEMM；省略主机端分配、初始化、错误检查和结果验证。

```cpp
#include <mma.h>
#include <cuda_fp16.h>

namespace wmma = nvcuda::wmma;

// 要求：<<<1, 128>>>，一维 block。
// A/B 各有 32*32 个 half；C 有 32*32 个 float。
// 假定 A/B/C 为各自 cudaMalloc 得到的对齐起始地址。
__global__ void gemm32_wmma(const half* A, const half* B, float* C)
{
    int warp_id = threadIdx.x / 32;
    int row = (warp_id / 2) * 16;
    int col = (warp_id % 2) * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16,
                   float> acc;

    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < 32; k += 16) {
        wmma::load_matrix_sync(a, A + row * 32 + k, 32);
        wmma::load_matrix_sync(b, B + k * 32 + col, 32);
        wmma::mma_sync(acc, a, b, acc);
    }

    wmma::store_matrix_sync(
        C + row * 32 + col, acc, 32, wmma::mem_row_major);
}

// 主机端调用示意：gemm32_wmma<<<1, 128>>>(d_A, d_B, d_C);
```

同 warp 的 `warp_id/row/col/k` 相同，所以每次 load/store 的指针和步长一致。接口自己分配片段，不要再给这些指针加 `lane_id`。

这个例子直接从 global 加载，只为说明映射。实际高性能 GEMM 常先用 shared memory 做 block 内复用，再加载 fragment。多个 warp 复用 A/B tile、双缓冲等，需要另行设计。

## 9. load/store、初始化与地址约束

### 9.1 tile 起始指针还不够，需要实际步长

```text
fragment：决定 tile 大小、数据类型和 A/B 布局
指针：决定 tile 从哪里开始
ldm：相邻行/列的真实内存间隔，单位是元素
```

从行主序 A[64,64] 读取第16行、第32列开始的 16×16 tile：

```cpp
wmma::load_matrix_sync(a, A + 16 * 64 + 32, 64);
```

读取范围为 A[16:32,32:48]。相邻行仍隔64个元素，所以传64。若先把 tile 紧凑复制到 shared[16][16]，再从 shared 加载，步长才变成16；若 shared 做了 padding，则用 padding 后的真实步长。

经典 WMMA load/store 要求内存指针32字节对齐，half 的 ldm 是8的倍数、float 的是4的倍数，同时必须满足合法布局和完整 tile 可访问性。不能因 cudaMalloc 基址对齐就忽略偏移后的 tile 指针对齐。边界不足需补零或采用其他处理路径，WMMA 不会自动屏蔽越界元素。

### 9.2 load/store 都是目标在前、源在后

```cpp
wmma::load_matrix_sync(a, A_ptr, lda);
//                    目标  源
// 内存 → fragment

wmma::store_matrix_sync(C_ptr, acc, ldc, wmma::mem_row_major);
//                     目标   源
// fragment → 内存
```

并不是源目标两个指针反过来：fragment 以对象引用传递，内存以指针传递。

### 9.3 标准调用顺序

```text
声明 fragment
    ↓
初始化 acc：fill_fragment 清零，或 load 已有 C
    ↓
沿 K 循环：load A/B → mma_sync 累加
    ↓
可选：本地逐元素输出处理
    ↓
store_matrix_sync
```

`fill_fragment` 与 A/B 的加载谁先谁后不是重点；第一次使用 acc 前必须有合法初值。不可每轮 K 都清零，否则丢失部分和。

计算 D=A@B+C 时可以从 C 加载累加器。若计算 alpha*A@B+beta*C，还需要正确组织 alpha/beta 处理；`mma_sync` 本身没有 cuBLAS 那样的 alpha/beta 参数。

## 10. 乘积精度、累加精度和输出精度

### 10.1 三个不同位置

```text
C[i,j] = sum_k A[i,k] * B[k,j]

单项乘积：A[i,k] * B[k,j]
累加器：已经加到当前 k 的部分和
最终输出：完成整个 K 后写回的值
```

手写 GEMM 中的 `float sum`、`float data[8][8]` 若用于不断 `+=`，就是 FP32 累加器。因此以前说“矩阵乘结果存在 float 中”，如果指这种变量，本来就在说累加精度。

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    float a = __half2float(A[i * K + k]);
    float b = __half2float(B[k * N + j]);
    sum = fmaf(a, b, sum);
}
```

WMMA 同样直接做矩阵乘加，不先暴露一个可供程序访问的“所有单项乘积矩阵”。程序维护的是输出部分和 fragment。

### 10.2 FP16 输入、FP32 累加不是“乘积先截断成 FP16”

PTX 对 FP16 WMMA 规定：元素乘法至少使用单精度，FP32 累加路径也至少以单精度累加。不是先把每个乘积舍入成 half，再扩成 float 求和。具体累加顺序、部分舍入和非正规数行为不完全由接口规定，不能保证与逐项标量 FP32 FMA 逐位一致。

没有一个独立的 WMMA 参数让程序员随意选择“中间乘积精度”；操作数类型、累加类型与指令形式共同决定行为。不能将 FP16 的说明不加区分地推广到所有 BF16/TF32/FP8 指令。

输入已经是 FP16 所丢失的信息，也不能靠 FP32 累加恢复。

### 10.3 两种舍入损失都要避免混淆

```cpp
half product = __hmul(a_half, b_half);
float product_f = __half2float(product);
```

上面乘积已舍入为 half，再转 float 不能恢复丢失的信息。同理，仅把一个表达式赋给 float，并不普遍保证表达式前面的运算已按 FP32 执行；应检查操作数和所用运算接口。

反过来，乘积精度足够、每次累加却存回 half，也会让部分和反复舍入。FP32 累加保护的是整个求和过程。

### 10.4 FP32 累加不要求最终输出一定是 FP32

```text
FP16 输入 → FP32 累加 → 最终保留 FP32
                     或完成求和后转换成 FP16
```

最终转换一次与每一步都在 FP16 中累加不同。经典 WMMA 的 float 累加器直接 store 对应 float 存储，不能把指针强转为 half* 期待它自动完成数值转换；需要显式转换/其他输出处理方案。

TF32 也不是给普通 CUDA `float` 运算替换了一个全局类型：它保留类似 FP32 的指数范围，但有效精度更低，通常用于 Tensor Core 输入，FP32 累加。严格 FP32 数值要求不能与“允许 TF32”混为一谈。

## 11. sync 与 async：同步范围和架构要区分

| 接口/机制 | 同步或异步的范围 |
|---|---|
| WMMA `*_sync` | warp 内共同参与对应操作，不是整个 GPU 同步 |
| `__syncthreads()` | block 内线程同步与相应内存可见性保证 |
| `cudaDeviceSynchronize()` | 主机等待设备此前工作完成 |
| kernel 内 `cp.async` 等 | global→shared 的异步数据搬运 |
| 主机 `cudaMemcpyAsync` | 在 stream 中提交复制，是否对主机真正异步还取决于内存与调用条件 |

不要把 WMMA 的 sync 当作跨 warp shared memory 的通用屏障。若一个 warp 消费其他 warp 写入的数据，需要匹配的同步；异步复制也需要其完成等待协议，不能只看到 `sync` 后缀就认为一切数据已就绪。

### 11.1 Ampere：搬运与 WMMA 计算重叠

Ampere 支持硬件加速的 global→shared 异步复制，可通过相应 `cuda::memcpy_async`、pipeline 等接口或 PTX 组织。

```text
当前 tile：shared → fragment → mma_sync
                 与
下一 tile：global → shared 的另一缓冲区
                 重叠
```

读下一块 shared 前必须确认复制完成；复用缓冲区前还要确认旧数据已消费。常见优化是双缓冲/多级流水，而不是把 `load_matrix_sync` 简单改名成 `load_matrix_async`。

### 11.2 异步矩阵计算有其他架构接口

Hopper 的 `wgmma.mma_async` 是 warp group 级的异步矩阵运算接口，涉及4个warp/128线程以及专门的同步规则。它不是普通 WMMA 的异步重载，3060 不能使用这条 Hopper 路径。其他架构还有各自接口，不能把新硬件能力直接套到 Ampere。

对于当前学习，先理解 Ampere 的 WMMA/`mma.sync` 与异步搬运，再了解更复杂的异步矩阵流水即可。

## 12. WMMA 附近的常用操作与优化

### 12.1 对 fragment 做统一逐元素操作

```cpp
for (int i = 0; i < acc.num_elements; ++i) {
    acc.x[i] *= alpha;
    acc.x[i] = fmaxf(acc.x[i], 0.0f);
}
```

每个线程处理自己的部分，整个 warp 对整块输出做同样的缩放/ReLU。这些逐元素运算不是又一次 Tensor Core 矩阵乘加。

按列不同的 bias 不能直接用 i 查表，因为 i 不是公开的矩阵列索引；可先 store 到可明确索引的内存，再处理，或采用能控制布局的更底层实现/库 epilogue。

### 12.2 其他常见细节

- `mma_sync` 的可选 `satf`：对 Inf/NaN 等采用规定的饱和到有限值行为，不是量化 scale 配置，也不是常规必须开启的选项。
- shared memory 复用：降低多个 warp 重复从 global 读相同 A/B 的成本。
- 合并访存、bank conflict、padding/swizzle：要同时满足实际布局和所用加载接口的约束，不能任意 swizzle 后仍按普通线性 layout 加载。
- 多份 acc fragment：一个 warp 负责多个输出 tile，提高复用，同时增加寄存器压力。
- K 流水与双缓冲：隐藏搬运延迟，但要处理好生产/消费同步。
- 边缘 tile：补零或专门分支，不能让 warp 内只有部分线程参加 WMMA。

## 13. 高频误区对照

| 误区 | 正确理解 |
|---|---|
| 允许 TF32 就一定选中 Tensor Core | 只是允许路径，需要检查实际算法/执行 |
| 启发式每次 Matmul 都随机选 | 保存并传入算法后重复使用；配置改变再核验 |
| 120 不能用 Tensor Core，128 才能 | padding 可能改善适配，不能仅靠尺寸证明执行路径 |
| Tensor Core 只能列主序 | 布局取决于接口支持；数学乘法规则不随布局改变 |
| 一个 thread 调用一次完整 WMMA | 一个 warp 共同执行，数据分散在各线程 |
| 一个 warp 永远只能算16×16 | 可以循环并维护多份 fragment；单次支持形状有限 |
| M/N/K 可以任意填写 | 类型、形状、目标架构必须匹配 |
| load 只需给左上角指针 | 还需真实步长、正确布局、对齐与有效范围 |
| tile16×16，所以 ldm=16 | ldm 是所在内存布局的真实步长 |
| 每个 thread 给 load 不同元素地址 | 同 warp 应传相同 tile 指针，由接口分配片段 |
| load/store 的源目标顺序不一致 | 都是目标在前、源在后；fragment 与指针的角色交换 |
| 每轮 K 都清零 acc | 只在开始初始化，之后保留部分和 |
| FP16 输入意味着乘积先舍入成 FP16 | 不能这样理解 FP16→FP32 的 Tensor Core 路径 |
| FP32 累加意味着最终输出必须 FP32 | 可在求和结束后另做输出类型转换 |
| mma_sync 会同步全部 GPU | 是 warp 级协作接口，不等于设备同步 |
| WGMMA 是给 WMMA 换成 async 后缀 | 不同架构和协作粒度的接口 |

## 14. 面试口述与复习顺序

### 14.1 描述 SNN 中的实际工作

> SNN 的 FC1 使用 cuBLASLt 实现，FP32 存储下允许 TF32 计算，配置 bias epilogue、workspace，并在初始化时查询算法，后续在 stream/Graph 中复用。代码还将120个有效输出通道补到128以改善计算布局。具体是否采用 Tensor Core、padding 的收益来自哪里，需要根据对应运行的 profiler 证据确认，不能只凭 computeType 或维度判断。

### 14.2 描述 WMMA 的机制

> WMMA 仍然在 CUDA kernel 的 block/thread 模型里执行，只是以一个warp协作完成矩阵tile乘加。先声明A、B、累加器fragment并初始化累加器，再沿K维加载子块、调用mma_sync累加，最后写回。M/N决定warp负责的输出区域，K决定要循环多少次；fragment内部的元素到线程映射由接口处理。

### 14.3 建议复习顺序

1. 先用第8节的固定32×32例子，手算四个warp的指针和两轮K循环。
2. 再区分 fragment tile尺寸与内存ldm，解释为何示例传32而不是16。
3. 画出输入精度→乘积→部分和→最终输出，理解FP32累加。
4. 最后再学shared复用、异步复制和更底层mma；无需一开始就背所有架构指令。

## 15. 官方参考资料

- [CUDA WMMA 接口、对齐、线程协作与形状表（12.2）](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-c-programming-guide/index.html#warp-matrix-functions)：本文经典 WMMA 的接口依据。
- [NVIDIA：Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)：warp协作、fragment和GEMM分块；历史硬件数量/吞吐数据不能套用到3060。
- [PTX ISA（11.0）](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html)：FP16/BF16/TF32矩阵运算的数值语义。
- [cuBLAS/cuBLASLt（12.2.2）](https://docs.nvidia.com/cuda/archive/12.2.2/cublas/index.html)：计算类型、描述符、算法查询和实现标记。
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)：异步global→shared搬运与流水。
- [CUDA Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：异步执行/访存模型，按目标架构阅读。
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：block、warp、指令级tile与流水组织。
