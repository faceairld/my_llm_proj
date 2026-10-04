# LeetGPU 刷题计划与笔记

> 记录于 2026-09-04。背景：AI infra / 大模型推理优化方向，已有手写 Qwen2.5 推理引擎（RMSNorm、Decode Attention、Flash Attention V1/V2、FlashDecoding、CUDA Graph）的项目经验。当前目标是**限时、无 IDE 一次写对**的手撕能力，不是刷题量。

---

## 一、之前给过的建议（原始出处）

在 2026-08-17 讨论字节 / vLLM 相关 JD 的对话中，关于算法与手撕部分的原话概括是：

> 手撕 CUDA 常见题：**reduce、softmax、GEMM、transpose、flash attention 简化版**
>
> 每天 40 分钟，一半 LeetCode 一半白板写 kernel，持续到面试。

这是一句概括性建议，**不是**针对 LeetGPU 题目列表逐题挑选的。下面是把它落到具体题目上的映射。

---

## 二、当前进度

LeetGPU 上已完成（有勾）的题目：

- Vector Addition
- Matrix Multiplication
- Reduction
- Softmax
- RMS Normalization
- Multi-Head Attention

这六道已经覆盖了上面那条建议的大半。

---

## 三、还需要补的题（第一梯队，必刷）

| 题目 | 难度 | 考点 |
|---|---|---|
| **Matrix Transpose** | Easy | shared memory 分块 + padding 避 bank conflict。面试问 transpose 基本都在问这个，是唯一还没做的基础必刷题 |
| **General Matrix Multiplication (GEMM)** | Medium | 和已完成的 Matrix Multiplication 不同，这道要真正写 tiling |
| **Causal Self-Attention** | Hard | 即「flash attention 简化版」，项目里写过，这里限时再写一遍 |
| **Prefix Sum** | Medium | 扫描类模式的代表，和 reduce 是一对 |

---

## 四、第二梯队（与推理优化方向直接对口，性价比高）

这些基本是把手写 Qwen2.5 引擎里的算子拆开单独考，写起来快，面试聊项目时能直接引用：

- Fused Residual Add and RMS Norm
- Rotary Positional Embedding
- SwiGLU MLP Block
- Grouped Query Attention
- Layer Normalization
- INT8 Quantized MatMul
- Top-p Sampling
- INT8 KV-Cache Attention

---

## 五、建议跳过

要么是同一模板换维度刷计数，要么是图论 / 仿真，与推理优化岗关系不大：

- Count Array Element 1D / 2D / 3D
- Subarray Sum 系列（1D / 2D / 3D / Max）
- Rainbow Table、Color Inversion
- Monte Carlo Integration
- K-Means Clustering、Multi-Agent Simulation
- BFS Shortest Path、All-Pairs Shortest Paths

**总量约 12 道，不是七十多道。** 重点在写法质量和限时手感，不在题数。

---

## 六、专题笔记：Vector Add

### 为什么它值得认真对待

Vector Add 本身没有算法，但它是**纯 memory-bound 的极限案例**，考的是"知不知道该拿什么当天花板"。

- 每个元素：读 8 字节（a、b），写 4 字节（c），做 1 次加法
- 算术强度 = 1 FLOP / 12 Byte，比任何 GPU 的 ridge point 低两个数量级
- 结论：**唯一目标是把带宽跑满**，任何与访存无关的优化都是白费

### 两种基本写法

```cuda
// 写法一：一线程一元素，grid 铺满
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) c[i] = a[i] + b[i];
// grid = (N + 255) / 256
```

```cuda
// 写法二：grid-stride loop，grid 固定
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < N;
     i += gridDim.x * blockDim.x)
    c[i] = a[i] + b[i];
// grid = smCount * 每SM的block数（例如 32 * 8）
```

写法二的好处：grid 大小与 N 解耦、可按硬件调、避免超大 N 时 grid 维度爆掉，工业代码更常见。两者带宽利用率差不多。

**反模式**：在 host 侧写循环反复 launch kernel。每次 launch 有几微秒开销，且没有任何收益——vector add 本来就不需要多次 pass。循环要放在 kernel 内部（即 grid-stride loop）。

### 真正拉开差距的：向量化访存

```cuda
__global__ void vecAdd4(const float4* __restrict__ a,
                        const float4* __restrict__ b,
                        float4* __restrict__ c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y,
                           x.z + y.z, x.w + y.w);
    }
}
```

- 一条 `LDG.E.128` 顶四条 `LDG.E.32`，指令数砍到 1/4
- 同时在途的访存请求更多（MLP，memory level parallelism）
- 大 N 下通常能从 ~80% 峰值带宽推到 ~92% 以上

**两个坑**：

1. `N % 4 != 0` 的尾巴要单独处理（最后一个线程收尾，或独立的小 kernel）
2. float4 要求 16 字节对齐。`cudaMalloc` 返回的基址是 256 字节对齐的，所以安全；但如果传进来的是带偏移的指针就不一定，工程代码要判一下

### 面试高频追问

**Q：为什么不用 shared memory？**
因为每个元素只被读一次，没有任何数据复用。shared memory 的价值在 reuse，这里用了只会白白多一次拷贝。这是送分题，很多人会条件反射地往上套 tiling。

**Q：怎么证明写得好？**
报**有效带宽**，不报 GFLOPS：

```
有效带宽 = 3 * N * sizeof(float) / time
```

然后和 `cudaGetDeviceProperties` 里的理论峰值比。能说出"跑到了理论带宽的 90%，剩下的差距是 DRAM 刷新和 ECC 开销"，比说"我用了 float4"有说服力得多。

**Q：blockDim 选多少？**
256 或 512。对 memory-bound kernel 这个参数基本不敏感，够填满就行。

### 行动项

LeetGPU 上最朴素的版本就能过，float4 那版是给面试用的。**把两版的带宽都测一下并记录数据**，以后聊 memory-bound 优化时能直接拿出来用。

---

## 七、执行原则

- 每天固定时间，不固定题数
- **限时、不开 IDE 写**——会写不是问题，一次写对才是
- 每道题写完记录：有效带宽 / 加速比、用到的优化手段、踩到的坑
