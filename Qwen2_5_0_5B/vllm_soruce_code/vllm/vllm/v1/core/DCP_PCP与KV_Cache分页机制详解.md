# DCP、PCP 与 KV Cache 分页机制详解

本文把阅读 `single_type_kv_cache_manager.py` 时最容易混淆的几组概念放到同一套模型里解释：

- Tensor Parallelism（TP）与 Context Parallelism（CP）是什么关系；
- Decode Context Parallelism（DCP）与 Prefill Context Parallelism（PCP）到底哪里不同；
- 它们与 FlashAttention、online softmax 的关系；
- rank、world size、CUDA block 分别处在哪个层级；
- vLLM 里的 page、block、block_size、slot 和 block table 是什么；
- 为什么 `single_type_kv_cache_manager.py` 会把 `block_size` 乘以 `dcp_world_size * pcp_world_size`；
- “逻辑 block 跨 GPU”和“本地 page 位于一张 GPU”为什么不矛盾。

> 版本说明：本文以当前目录中的这份 vLLM 源码为准。当前 checkout 已有 DCP 的实际支持；PCP 已有配置、并行组与 KV 映射框架，但 attention backend 仍处于开发阶段，具体证据见本文第 6.6 节。

> 阅读说明：这不是只列定义的速查摘要。后文会把“一个请求如何进入 KV Cache”“每张 GPU 实际做什么”“局部 softmax 为什么能合并”“一个 token 最后落到 tensor 的哪个位置”逐步推演出来。第 1 节只是先给地图，真正的解释从第 2 节开始。

---

## 1. 先记住这几条结论

1. **TP 和 CP 是正交的并行维度，可以同时使用。**TP 通常切权重和 head 维度，CP 切 token/sequence 维度。
2. **DCP 与 PCP 都可以使用分布式 online softmax，但数据所有权和通信调度不同。**
3. **DCP 的典型形态是“同一批小 Q，对不同 KV 分片计算，再合并同一批 Q 的部分结果”。**
4. **PCP 的典型形态是“不同 GPU 负责不同的大 Q 分片，通过收集或轮转 KV 完成本地 Q 的结果”。**
5. **vLLM 的 page/physical block 是应用层 KV Cache 分配单位，不是 CUDA thread block，也不是 GPU 虚拟内存硬件页。**
6. **`block_size` 的单位是 token/page，表示一个本地 page 能存多少个 token 的 KV。**
7. **逻辑/虚拟 block 可以由多张 GPU 上的多个本地 page 共同表示，但每个本地 page 自身只位于一张 GPU。**
8. **在 `SingleTypeKVCacheManager` 中，同名的 `self.block_size` 在开启 CP 后被提升成了全局记账大小，语义不再等于单卡 page 的容量。**这是理解源码的关键。

---

## 2. 为什么 Prefill 和 Decode 要分别设计 CP

自回归模型的推理主要分为两个阶段。

### 2.1 Prefill

假设 prompt 有 `T` 个新 token。Prefill 要为这些 token 生成大量 Q、K、V，并完成 causal attention：

```text
Q[0:T] × K[0:T] -> attention output[0:T]
```

它的特点通常是：

- query 数量大；
- attention 计算量大；
- 长 prompt 的 TTFT（Time To First Token）容易很高；
- 有足够多的 Q 行可分给多张 GPU 并行计算。

因此 PCP 的主要目标是：**把长 prompt 的 prefill 计算摊到多张 GPU，降低 TTFT。**

### 2.2 Decode

进入 decode 后，每一步通常只新增一个 token（使用 speculative decoding/MTP 时也只是少量 token），但它要读取很长的历史 KV Cache：

```text
Q_new × K/V_history -> output_new
```

它的特点通常是：

- query 数量很小；
- 历史 KV 很大；
- 读取 KV 的显存带宽和 KV Cache 容量更容易成为瓶颈；
- 不像 prefill 那样有大量 Q 行可以直接分摊。

因此 DCP 的主要目标是：**沿历史 token 维度切分 KV Cache，降低每张 GPU 的 KV 容量和读取压力，并扩大可容纳的请求批次。**

“Prefill 更偏计算瓶颈、Decode 更偏带宽瓶颈”是常见工作负载的概括，不是所有模型、长度和硬件下都绝对成立。

vLLM 自带文档也按这两个目标分别介绍 CP，见 [context_parallel_deployment.md](../../../docs/serving/context_parallel_deployment.md)。

### 2.3 从一个请求的完整生命周期看两者插在哪里

假设用户输入一段 prompt，然后模型继续生成 token。整个过程可以按下面的时间线理解。

#### 阶段一：请求进入调度器

调度器先看到的是 token ID 序列，而不是已经算好的 K/V：

```text
prompt token IDs = [t0, t1, t2, ..., tT-1]
```

它根据当前可用 KV page、请求优先级、prefix cache 命中等信息，决定本轮能调度多少 token，并给请求分配 logical block 对应的 block ID。

此时 `SingleTypeKVCacheManager` 管的是“请求占哪些 block、还需要几个 block、哪些 block 可以复用或释放”。它本身并不执行 attention kernel，也不直接装 K/V 数值。

#### 阶段二：Prefill 计算 prompt

模型逐层处理这批 prompt token。每一层大致经历：

```text
hidden states
    -> Q/K/V projection
    -> attention
    -> output projection
    -> MLP 等后续计算
```

这一阶段会把 prompt token 的 K/V 写进已经分配好的 KV page。若启用 PCP，主要是在这里把大量 token/Q 行分给不同 PCP rank，并安排 K/V 的收集或轮转。

#### 阶段三：产生第一个输出 token

当所有 prompt token 的 prefill 完成后，模型得到最后位置的 logits，采样出第一个新 token。此时 prompt 的历史 K/V 已经保存在 KV Cache 中。

#### 阶段四：进入逐 token Decode

后续每一步只有少量新 query，但要读取全部历史 K/V：

```text
第 1 步：Q_new1 读取 prompt 的 KV
第 2 步：Q_new2 读取 prompt + new1 的 KV
第 3 步：Q_new3 读取 prompt + new1 + new2 的 KV
...
```

每生成一个 token，KV Cache 尾部就多写入一个 slot。一个 page 填满后，调度器再给该请求追加新的 page。

若启用 DCP，主要是在这个阶段让各 DCP rank 只读取自己本地的历史 KV 分片，再合并同一批新 query 的局部 attention 结果。

所以可以把两者放在同一条链上：

```text
长 prompt 到来
    -> PCP 主要并行化“大量新 Q/K/V 的 prefill”
    -> 得到首 token
    -> DCP 主要并行化“少量新 Q 对大量历史 KV 的 decode”
```

---

## 3. TP 和 CP：不是二选一，而是切不同维度

忽略 batch 和 layer，一层 KV Cache 可以简化成：

```text
K/V: [T, H_kv, D]

T    = token/sequence 长度
H_kv = KV head 数量
D    = 每个 head 的维度
```

### 3.1 两者通常切什么

```text
TP：主要切模型权重，也通常切 attention/KV head 维度 H_kv
CP：主要切 token/sequence 维度 T
```

所以它们可以构成二维切分：

```text
                           token 维度（CP）
                    CP rank 0       CP rank 1
                  ┌──────────────┬──────────────┐
TP rank 0/head组0 │ 一块 KV 数据 │ 一块 KV 数据 │
                  ├──────────────┼──────────────┤
TP rank 1/head组1 │ 一块 KV 数据 │ 一块 KV 数据 │
                  └──────────────┴──────────────┘
```

权重显存和 KV Cache 显存确实是不同的 allocation，但 TP 仍会影响本 rank 的 KV Cache 布局，因为 TP 决定本 rank 保存多少个 KV head。

### 3.2 为什么 DCP 又和 TP 有紧密关系

假设模型有 8 个 KV head，`TP=2`，通常可以把 head 分成两组：

```text
TP rank 0：KV head 0~3
TP rank 1：KV head 4~7
```

但 MQA/MLA 模型可能只有 1 个 KV head，而部署使用 `TP=8`。一个 KV head 无法继续按 head 维度切成 8 份，于是它可能在多个 TP rank 上重复保存。

DCP 会复用 TP group 中已有的 GPU，再沿 `T` 维切分 KV：

```text
没有 DCP：多个 TP rank 可能保存重复的完整 token 范围
启用 DCP：这些 rank 分别保存不同 token 范围，减少重复
```

因此：

- TP 和 DCP 不是互斥关系；
- 当前 vLLM 的 DCP 有意复用 TP GPU；
- 增大 DCP 不增加进程总数，但要求 TP size 能被 DCP size 整除；
- DCP 越大，KV 重复越少，但通信开销越高。

对应配置注释见 [parallel.py](../../config/parallel.py#L307-L340)。

### 3.3 PCP 是否增加 GPU/world size

当前配置中的总 world size 计算包含 PCP：

```python
world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size
```

DCP 不再额外乘进去，因为它复用 TP group 中的 GPU；PCP 则是一个会进入总 world size 的并行维度。源码见 [parallel.py](../../config/parallel.py#L710-L716)。

### 3.4 一个数字化例子：TP 为什么会影响 KV，DCP 又如何消除重复

假设某层使用 MQA，参数为：

```text
H_kv = 1
T = 32768
head_size = 128
KV dtype = FP16，每个元素 2 bytes
```

这一层完整 K/V 的数据量是：

```text
2 * T * H_kv * head_size * dtype_bytes
= 2 * 32768 * 1 * 128 * 2
= 16 MiB
```

现在使用 `TP=8`。权重当然可以按 TP 切，但这里只有 1 个 KV head，无法沿 head 维均匀切成 8 份。简化理解时，8 个 TP rank 可能各自保存相同 token 范围的这个 KV head：

```text
每个 rank：约 16 MiB/层
8 个 rank 合计：约 128 MiB/层
```

如果在同一组 TP GPU 上启用 `DCP=8`，让每个 rank 只保存八分之一 token：

```text
每个 rank：约 2 MiB/层
8 个 rank 合计：约 16 MiB/层
```

这说明：

- TP 切权重与 DCP 切 token 确实是不同维度，可以同时使用；
- 但 TP 对本地 KV head 的分配会影响 KV 是否重复，因此二者不是完全互不相干；
- DCP 在这里没有增加 GPU 数，而是改变已有 TP rank 上 KV token 的所有权。

实际显存还会受到 page padding、cache dtype、层分组和 backend layout 影响，这个例子只用于说明数量级和切分关系。

---

## 4. rank、world size 和 CUDA block 分别是什么

### 4.1 分布式 rank

rank 是一个进程在某个通信 group 内的编号。典型部署中通常一个 worker 进程控制一张 GPU：

```text
进程/rank 0 <-> GPU 0
进程/rank 1 <-> GPU 1
进程/rank 2 <-> GPU 2
进程/rank 3 <-> GPU 3
```

同一个进程相对于不同通信组，可以同时拥有多种 rank：

```text
global rank
TP rank
PP rank
DCP rank
PCP rank
```

例如，一个进程的 global rank 可能是 6，但它在某个 DCP group 内的 `dcp_rank` 可能是 0。

### 4.2 world size

world size 表示对应通信组里一共有多少个 rank：

```text
dcp_world_size = 一个 DCP group 里的 rank 数
pcp_world_size = 一个 PCP group 里的 rank 数
```

### 4.3 rank 不是 CUDA block

这两个概念位于不同层级：

```text
分布式层级：机器 -> 进程/rank -> GPU
GPU kernel 层级：grid -> CUDA block/CTA -> thread
```

一个 DCP rank 通常会在自己的 GPU 上启动很多 CUDA block，处理不同 request、head、query tile 和 KV tile。因此不能把“一个 DCP rank”理解为“一个 CUDA block”。

### 4.4 为什么 DCP 数和 PCP 数相乘

DCP 与 PCP 形成一个二维坐标：

```text
(pcp_rank, dcp_rank)
```

假设：

```text
pcp_world_size = 2
dcp_world_size = 2
```

一共有四种坐标：

| PCP rank | DCP rank | 压平后的 total_cp_rank |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 2 |
| 1 | 1 | 3 |

源码使用：

```python
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

这里的乘法是在枚举二维坐标的所有组合，不是说一个 page 自己突然扩大成了多卡 allocation。对应实现见 [backend.py](../attention/backend.py#L702-L735) 和 [block_table.py](../worker/block_table.py#L141-L164)。

---

## 5. DCP：同一批小 Q，对不同 KV 分片计算

### 5.1 最直观的例子

假设 decode 当前只计算一个新 token，历史 KV 有 32K token，`DCP=2`：

```text
                     同一个 Q_new
                    /            \
GPU/rank 0：Q_new × KV[0:16K]   GPU/rank 1：Q_new × KV[16K:32K]
                |                              |
        局部 attention 结果 0          局部 attention 结果 1
                \                              /
                  按 LSE/online softmax 精确合并
                                 |
                       Q_new 的完整 attention 输出
```

两张 GPU 都在为同一批 query 行工作，只是各自扫描不同的 KV/token 分片。这可以理解成分布式的 split-KV attention。

### 5.2 为什么不能直接把两个局部 output 相加

Attention 是：

```text
O = softmax(QK^T) V
```

softmax 的分母依赖所有 KV token。每个 rank 只看到一部分 logits，所以必须同时保留归一化统计量。

对第 `r` 个 KV 分片，定义：

```text
m_r = max(local_logits)
l_r = sum(exp(local_logits - m_r))
z_r = sum(exp(local_logits - m_r) * V_r)
```

跨 rank 合并：

```text
m = max_r(m_r)
l = sum_r(exp(m_r - m) * l_r)
z = sum_r(exp(m_r - m) * z_r)
O = z / l
```

这就是 FlashAttention 使用的 online softmax 结合律扩展到多 GPU 后的核心数学基础。LSE（LogSumExp）可以写成：

```text
LSE = m + log(l)
```

当前 vLLM 会检查 DCP attention backend 是否能在 decode 时返回 LSE，见 [cp_utils.py](../worker/cp_utils.py#L14-L44)。FlashAttention backend 取得局部 output/LSE 后调用 DCP combine，见 [flash_attn.py](../attention/backends/flash_attn.py#L920-L939)。

### 5.3 DCP 的通信方式

当前配置列出了两类 DCP 通信 backend：

- `ag_rs`：AllGather + ReduceScatter；
- `a2a`：交换局部 attention output 和 LSE，再做精确的 LSE 加权合并。

`a2a` 实现的文件头明确说明了它试图减少每层的 NCCL 调用次数，见 [dcp_alltoall.py](../attention/ops/dcp_alltoall.py)。

无论具体 backend 怎么组织通信，数学目标相同：**让同一个 query 对所有 KV 分片的局部结果合并成与单卡完整 attention 等价的结果。**

### 5.4 DCP 优化了什么，又付出了什么

收益：

- 每张 GPU 只保存部分 token 的 KV；
- 每张 GPU 每步只读取部分历史 KV；
- 减少 MQA/GQA/MLA 在大 TP 下产生的 KV 重复；
- 腾出 KV Cache 空间以容纳更大 batch。

代价：

- 每一层、每个 decode step 都需要跨 rank 通信；
- decode 的 query 很少，计算量未必足以掩盖通信延迟；
- DCP 太大时，通信收益可能抵不过额外延迟；
- 更适合高速互联，尤其是节点内 NVLink/NVSwitch 场景。

### 5.5 DCP 不是“一个 CUDA block 负责一个 softmax 分片”

可以把 DCP 在数学上理解为“不同 GPU 负责 softmax 的不同 KV 区间”，但具体 CUDA 映射由 attention backend 决定。一张 GPU 通常会启动大量 CTA：

```text
rank 0 / GPU 0
├── CTA 0：某 request、某 head、某 tile
├── CTA 1：另一个 head/tile
├── CTA 2：另一个 request/tile
└── ...
```

DCP 负责的是 GPU/进程级数据所有权与通信；FlashAttention kernel 负责的是单 GPU 内部的 tile、shared memory、warp 和 CUDA block 组织。

### 5.6 一次 DCP decode 在一层里逐步做什么

下面以 `DCP=2` 为例。不同 backend 可能融合、重排某些通信，但逻辑依赖基本相同。

#### 第 1 步：历史 KV 已经分片存放

在进入本次 decode 前，历史 token 的 KV 已经按 CP 映射存入两张 GPU：

```text
rank 0：本 rank 拥有的历史 K0/V0
rank 1：本 rank 拥有的历史 K1/V1
```

这里的 `K0/K1` 不是不同 head 的名字，而是 token 维上的不同本地分片。

#### 第 2 步：生成本轮 query

当前层根据新 token 的 hidden state 生成 Q。由于 DCP 复用 TP rank，Q 在进入局部 attention 前可能需要 all-gather、all-to-all，或者由 backend 按它需要的布局交换。核心要求是：每个负责某段 KV 的 rank 必须拿到计算相应局部 attention 所需的 Q。

#### 第 3 步：每个 rank 只扫描本地 KV page

```text
rank 0：local_attention(Q, K0, V0) -> O0, LSE0
rank 1：local_attention(Q, K1, V1) -> O1, LSE1
```

PagedAttention/FlashAttention kernel 会根据本 rank 的 block table 读取本地 page，不需要把历史 K0/V0 搬到 rank 1，也不需要把 K1/V1 搬到 rank 0。

#### 第 4 步：合并同一 query 的局部结果

`O0` 和 `O1` 都只是同一个 Q 在一部分 KV 上的结果。通信 backend 交换 output/LSE，按全局 softmax 分母重新加权：

```text
(O0, LSE0) + (O1, LSE1) -> O_global
```

这不是普通求和，也不是简单平均。

#### 第 5 步：进入 attention 后续计算

得到与“单卡看到完整 KV”数学等价的 `O_global` 后，才继续 output projection、残差连接、下一层等操作。下一层又重复相同过程，因此 decode 对通信延迟非常敏感：每个生成步、几乎每个 attention 层都会走一次局部计算与组合。

### 5.7 用具体数字验证 LSE 合并不是近似

为了把 online softmax 说透，先把 V 简化成标量。假设一个 query 的 logits 和 V 被分成两片：

```text
rank 0:
    logits0 = [1, 2]
    V0      = [10, 20]

rank 1:
    logits1 = [0, 3]
    V1      = [30, 40]
```

rank 0 的局部统计量：

```text
m0 = 2
l0 = exp(1-2) + exp(2-2)
   = exp(-1) + 1
   ≈ 1.3679

z0 = exp(1-2)*10 + exp(2-2)*20
   ≈ 23.6788
```

rank 1 的局部统计量：

```text
m1 = 3
l1 = exp(0-3) + exp(3-3)
   = exp(-3) + 1
   ≈ 1.0498

z1 = exp(0-3)*30 + exp(3-3)*40
   ≈ 41.4936
```

全局最大值：

```text
m = max(m0, m1) = 3
```

必须把 rank 0 的统计量从基准 `m0=2` 换算到全局基准 `m=3`：

```text
l = exp(m0-m)*l0 + exp(m1-m)*l1
  = exp(-1)*1.3679 + 1*1.0498
  ≈ 1.5530

z = exp(m0-m)*z0 + exp(m1-m)*z1
  = exp(-1)*23.6788 + 1*41.4936
  ≈ 50.2046

O = z/l ≈ 32.33
```

如果直接在一张 GPU 上对 `[1, 2, 0, 3]` 做完整 softmax，再与 `[10, 20, 30, 40]` 加权，也会得到约 `32.33`。因此这是利用 softmax 结合律进行的精确重组，只有浮点舍入差异，不是近似抽样。

这也解释了为什么 DCP backend 必须返回 LSE：仅有局部归一化后的 `O0/O1`，无法知道两个分片的 softmax 分母分别有多大，因而不能正确恢复全局输出。

---

## 6. PCP：不同 GPU 拥有不同的大 Q 分片

### 6.1 基本划分

Prefill 有 `T` 个新 token 时，可以沿 token 维切 Q/K/V：

```text
GPU/rank 0：Q0, K0, V0
GPU/rank 1：Q1, K1, V1
...
```

与 DCP 最重要的区别是：**PCP 的不同 rank 主要拥有不同的 query 行和不同的最终 output 行，而不是共同为同一小批 query 产生最终结果。**

### 6.2 策略一：Partial Q + Full KV

先让每张 GPU 生成自己的 Q/K/V 分片，然后收集 K/V：

```text
GPU 0：Q0 + 完整 K/V -> O0
GPU 1：Q1 + 完整 K/V -> O1
```

特点：

- Q 仍然是分片的；
- 每张 GPU 最终能看到完整 KV；
- 每张 GPU 独立完成自己 Q 分片的 attention；
- 同一 query 行通常不需要再跨 GPU 合并 output；
- 代价是每张 GPU 都要临时持有或访问完整 K/V。

适用于序列较长、希望加速 prefill，但完整 K/V 仍能放下的情况。

### 6.3 策略二：Partial Q + Partial KV / Ring Attention

当完整 K/V 太大，不希望复制到所有 GPU 时，可以让 K/V 分片在 ring 中轮转。

初始状态：

```text
GPU 0：Q0, KV0
GPU 1：Q1, KV1
```

第一轮：

```text
GPU 0 计算 Q0 × KV0
GPU 1 计算 Q1 × KV1
```

交换 KV 后第二轮：

```text
GPU 0 收到 KV1，计算 Q0 × KV1
GPU 1 收到 KV0，计算 Q1 × KV0
```

每轮都不保存完整 attention matrix，而是更新本地 Q 的 online-softmax 累加状态：

```text
(m, l, z) <- merge((m, l, z), 当前 KV 分片产生的结果)
```

所有需要访问的 KV 分片轮转完成后，本 rank 拥有的 Q 分片也得到完整输出。

所以把 PCP/Ring Attention 理解为“把 FlashAttention 的分块 online softmax 提升到多 GPU”是合理的。但是，多 GPU 层面的重点不再只是“一个 CUDA block 算几个 query”，而是 **Q 的所有权、KV 的通信顺序、因果掩码和负载均衡**。

### 6.4 PCP 为什么不能只理解成“大一号 FlashAttention”

单 GPU FlashAttention 主要解决：

- Q/K/V tile 如何进入 shared memory/SRAM；
- 如何避免物化完整 attention matrix；
- warp/CTA 如何处理 query 和 KV tile；
- 如何在单 GPU 上维护 online softmax。

PCP 还必须解决：

- 哪些 Q 行属于哪个 GPU；
- K/V 是 all-gather 还是 ring 轮转；
- 如何把 KV 通信与当前 attention 计算重叠；
- causal attention 的三角计算量如何在 GPU 间均衡；
- 哪些 Q-K 分块因 causal mask 可以直接跳过；
- 一轮通信没完成时如何使用双缓冲预取下一块 KV；
- 最终 hidden states 如何进入下一层的并行布局。

因此，“一个 CTA 负责多个 query”属于单卡 kernel tiling；“哪张 GPU 拥有哪些 Q、KV 如何在 rank 间移动”才是 PCP 的并行算法。

### 6.5 PCP 常见的计算与通信优化

#### 优化一：避免 causal 三角形负载不均

causal attention 中，靠后的 query 能看到更多 KV：

```text
Q0  -> K0
Q1  -> K0 K1
Q2  -> K0 K1 K2
Q3  -> K0 K1 K2 K3
```

如果简单把前半段 Q 给 GPU 0、后半段 Q 给 GPU 1，那么 GPU 1 的计算量更大。可使用交错、zigzag 或其他重新排列方式，使每张 GPU 同时拥有较早和较晚的 Q 区间。

#### 优化二：通信与计算重叠

使用两个缓冲区：

```text
buffer A：计算当前 KV 分片
buffer B：同时接收下一 KV 分片
```

当前计算结束后交换 A/B，尽量隐藏 P2P/NCCL 通信时间。

#### 优化三：跳过不可能可见的 Q-K tile

如果一个 query tile 的所有位置都早于某个 key tile，那么 causal mask 会把整个 tile 屏蔽，可以不执行对应 attention kernel。

#### 优化四：稳定并融合 online softmax

- 使用足够精度（常见为 FP32）保存最大值和归一化累计量；
- 尽量把局部 attention 和 `(m, l, z)` 更新融合；
- 避免频繁把中间结果写回 HBM。

#### 优化五：只在足够长的 prefill 上使用

PCP 有固定的通信与调度开销。短 prompt 的计算量不足以覆盖这些成本，可能还不如单卡/普通 TP。它更适合长上下文和 TTFT 受限的场景。

### 6.6 当前这份源码中 PCP 的实现状态

必须区分“算法设计”和“当前 checkout 已可运行的 backend”。

当前源码中：

```python
supports_pcp: bool = False
```

定义在 [backend.py](../attention/backend.py#L678-L690)。当 `pcp_size > 1` 时，[cp_utils.py](../worker/cp_utils.py#L39-L44) 会断言 attention implementation 必须支持 PCP。

在当前仓库中搜索 `supports_pcp`，只有默认定义和这处检查，没有 attention backend 将它覆盖成 `True`。同时，官方本地文档也把 PCP 的两种方案标记为 under active development。

因此，在这份 checkout 中可以看到：

- PCP 配置字段；
- PCP process group/rank；
- `total_cp_world_size` 和 `total_cp_rank`；
- KV block/slot 的通用 CP 映射；

但看不到一个已经声明可用的 PCP attention backend。本文关于 Ring Attention、负载均衡和通信重叠的内容，是 PCP 的算法设计与优化方向，不应误认为当前版本已经全部落地。

### 6.7 DCP 和 PCP 的最终对照

| 项目 | DCP | PCP |
|---|---|---|
| 主要阶段 | Decode | Prefill |
| 典型 Q 数量 | 很少 | 很多 |
| Q 所有权 | 同一批 Q 被复制/收集到多个 DCP rank | Q 沿 token 维分片 |
| KV 所有权 | 历史 KV 沿 token 维分片 | 可收集成完整 KV，也可保持分片并轮转 |
| 局部结果关系 | 多个 rank 为同一 Q 行产生部分结果 | 每个 rank 主要负责不同 Q 行 |
| 最终组合 | 对同一 Q 的 output/LSE 做精确合并 | 本地 Q 遍历必要 KV 后直接拥有完整输出 |
| 主要收益 | KV 容量、KV 带宽、batch capacity | 长 prompt 计算并行、降低 TTFT |
| 主要代价 | 每层每步的低延迟通信 | 大量 KV 通信、causal 负载均衡和布局转换 |

### 6.8 四张 GPU 上的 PCP Ring Attention 完整推演

设 prompt 被分成四个连续 chunk：

```text
C0 = token   0 ~  999
C1 = token 1000 ~ 1999
C2 = token 2000 ~ 2999
C3 = token 3000 ~ 3999
```

初始所有权：

```text
rank 0：Q0, K0, V0
rank 1：Q1, K1, V1
rank 2：Q2, K2, V2
rank 3：Q3, K3, V3
```

如果暂时忽略 causal mask，每个 Q chunk 都要看四个 KV chunk。最朴素的 ring 时间线可以想成：

```text
轮次 0：每个 rank 计算本地 Q × 本地 KV
轮次 1：KV 向下一个 rank 发送；计算本地 Q × 收到的 KV
轮次 2：KV 再轮转一次；继续更新本地 Q 的 (m,l,z)
轮次 3：最后一片 KV 到达；完成本地 Q 输出
```

rank 2 的视角例如是：

```text
Q2 × KV2 -> accumulator_2
Q2 × KV1 -> merge accumulator_2
Q2 × KV0 -> merge accumulator_2
Q2 × KV3 -> merge accumulator_2
```

具体收到 KV 的顺序可以不同，因为 online softmax 的合并满足结合律；只要所有允许参与的 KV 都被合并，最终结果一致。

加入 causal mask 后，连续 chunk 的可见关系是：

| 本地 Q chunk | 完全或部分需要访问的 KV chunk |
|---|---|
| Q0 | KV0（同 chunk 内仍是三角 causal） |
| Q1 | KV0、KV1 |
| Q2 | KV0、KV1、KV2 |
| Q3 | KV0、KV1、KV2、KV3 |

于是简单连续分块会产生明显不均衡：

```text
rank 0 的 Q0 工作最少
rank 3 的 Q3 工作最多
```

这就是为什么真实 PCP 设计不能只说“每张卡分一段 Q”就结束。常见改进会把序列切得更细，再让一个 rank 同时拿到较早和较晚的 Q 小块，例如概念上的 zigzag 配对：

```text
rank 0：较早块 + 较晚块
rank 1：次早块 + 次晚块
...
```

这样每个 rank 的“可见 KV 面积”更接近，避免一张卡计算三角形顶部、另一张卡计算三角形底部造成长尾。

### 6.9 Ring 中怎样重叠通信和计算

每个 rank 可以准备两个 KV 接收缓冲区：

```text
时间片 t：
    GPU kernel 使用 buffer A 计算 Q_local × KV_current
    NCCL/P2P 同时向 buffer B 接收 KV_next

时间片 t+1：
    交换 A/B
    GPU kernel 使用 buffer B 计算
    通信引擎向 buffer A 接收再下一片 KV
```

理想情况下：

```text
每轮计算时间 >= 下一片 KV 的通信时间
```

通信就能大部分隐藏在计算后面。若 prompt 太短，单轮 attention 很快，通信暴露出来，PCP 反而可能变慢。这就是 PCP 通常需要长度阈值或调度策略，而不是对所有 prefill 无条件启用的原因。

### 6.10 PCP 的输出为什么通常不做 DCP 那种最终归约

在上面的 ring 设计中，Q 不移动，KV 移动：

```text
rank 0 始终拥有 Q0，最终也拥有 O0
rank 1 始终拥有 Q1，最终也拥有 O1
...
```

每个 rank 在本地 accumulator 中先后合并各 KV 分片。因此所有 KV 都看完后，本地 `O_i` 已经完整，不需要把同一个 Q 行的多个最终 output 再做一次跨 rank reduce。

而 DCP 的常见布局是多个 rank 同时为同一批小 Q 产生局部结果，所以必须让这些结果重新组合。两者都用了 online softmax，但“谁固定、谁移动、最终 output 属于谁”不同。

PCP 得到的 `O0/O1/...` 仍然沿 token 维分片。LayerNorm、逐 token MLP 等操作可以在本地 token 上继续执行；若下一操作需要其他并行布局，则还要做相应的 all-gather、reduce-scatter 或 layout transition。TP 自身所需的 collective 与 PCP 的序列布局可以同时存在。

---

## 7. page、block、block_size 和 slot

这里的 `block` 与 CUDA block 没有关系。

### 7.1 用“抽屉柜”建立直觉

把单层、单个 rank 上预分配的 KV Cache 想成一个大抽屉柜：

```text
KV Cache 存储池
├── 抽屉/page 0
├── 抽屉/page 1
├── 抽屉/page 2
├── ...
└── 抽屉/page num_blocks-1
```

在 vLLM/PagedAttention 的语境里，一个这样的抽屉通常称为：

- page；
- physical block；
- KV Cache block。

这几个词强调的侧面不同，但通常都指一个固定容量的应用层 KV Cache 分配单位。

如果：

```text
block_size = 16
```

每个抽屉有 16 个格子：

```text
page 37
├── slot 0：一个 token 的 K/V
├── slot 1：一个 token 的 K/V
├── ...
└── slot 15：一个 token 的 K/V
```

关系是：

```text
KV Cache 存储池
    包含 num_blocks 个 page

一个 page/physical block
    包含 block_size 个 slot

一个 slot
    保存一个 token 在当前层、当前 rank 所负责 head 上的 K 和 V
```

### 7.2 block_size 为什么以 token 为单位

`block_size=16` 的意思是：

```text
每个本地 page 最多保存 16 个 token 的 KV
```

它不表示 16 字节，也不表示一共只有 16 个 block。

源码中的 `KVCacheSpec` 直接把它注释为 `number of tokens in a block`，见 [kv_cache_interface.py](../kv_cache_interface.py#L68-L84)。

### 7.3 一个 slot 里到底有什么

假设当前 rank 有：

```text
num_kv_heads = 8
head_size = 128
```

一个 token 的 K 和 V 分别是：

```text
K[token]: [8, 128]
V[token]: [8, 128]
```

所以一个 slot 可以理解为：

```text
slot
├── K: [num_kv_heads, head_size]
└── V: [num_kv_heads, head_size]
```

一个 page 有 `block_size` 个 slot，因此单个 page 的概念形状是：

```text
[2, block_size, num_kv_heads, head_size]
 ↑
 K/V
```

### 7.4 整个 KV Cache tensor 的形状

FlashAttention backend 返回的典型布局是：

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

见 [flash_attn.py](../attention/backends/flash_attn.py#L129-L143)。

FlashInfer 的维度顺序略有不同：

```text
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

见 [flashinfer.py](../attention/backends/flashinfer.py#L346-L358)。

因此，“page/block 的概念”是稳定的，但不同 backend 可以选择不同的物理维度排列。

### 7.5 “大 tensor 中由 block_id 索引出的切片”翻译成人话

假设 FlashAttention 的 KV Cache 形状是：

```text
[2, 1000, 16, 8, 128]
```

含义是：

```text
K/V 两份数据
× 1000 个抽屉/page
× 每个抽屉 16 个 token 格子
× 每个 token 8 个 KV head
× 每个 head 128 个元素
```

`block_id=37` 就是选择第 37 号抽屉：

```python
page_37 = kv_cache[:, 37, :, :, :]
```

得到的概念形状是：

```text
[2, 16, 8, 128]
```

这里的“切片”不是切割显存，也不是复制数据，只是**定位到原大 tensor 中代表第 37 号抽屉的那段地址**。

继续选择 `slot=5`：

```python
token_kv = kv_cache[:, 37, 5, :, :]
```

得到：

```text
[2, 8, 128]
```

也就是第 37 号 page 的第 5 个 token 格子中的 K/V。

### 7.6 page 是 GPU 虚拟内存页吗

不是。它是 vLLM 自己定义的应用层分配单位。

“PagedAttention”借用了操作系统分页的思想：请求看到连续的逻辑 token，底层可以映射到不连续的 KV page。但它不要求每个 vLLM page 对应：

- 一个 CUDA Virtual Memory Management page；
- 一个 GPU 硬件页表 page；
- 一个操作系统 4 KiB page。

底层完全可以是一整个普通的预分配 GPU tensor，vLLM 通过 `block_id` 和 offset 计算某个 token 应写到这个 tensor 的哪里。

因此，“physical page”更安全的叫法是：

```text
rank-local KV Cache page
```

它强调这块存储在当前 rank/GPU 的 KV Cache pool 中真实存在，但不声称它对应某种硬件物理页。

### 7.7 一个 page 占多少字节

对普通 full-attention KV Cache，一个本地 page 的实际数据量大约是：

```text
page_bytes =
    2
    * block_size
    * num_kv_heads
    * head_size
    * dtype_bytes
```

其中 `2` 代表 K 和 V。对应源码见 [kv_cache_interface.py](../kv_cache_interface.py#L136-L144)。

例如：

```text
block_size = 16
num_kv_heads = 8
head_size = 128
dtype = FP16 = 2 bytes
```

一个 token slot：

```text
2 * 8 * 128 * 2 = 4096 bytes = 4 KiB
```

一个 page：

```text
16 * 4 KiB = 64 KiB
```

如果当前层、当前 rank 有 1000 个 page，这部分约为：

```text
1000 * 64 KiB = 62.5 MiB
```

总 KV Cache 还会受层数、不同 KV Cache group、量化格式和 padding 等因素影响。

### 7.8 为什么要预分配大 tensor，而不是每来一个 token 就申请显存

如果每生成一个 token 都执行一次独立的 GPU 内存申请，系统会遇到：

- `cudaMalloc`/allocator 路径带来的频繁管理开销；
- 大量大小不一、生命周期不同的 allocation 造成碎片；
- batch 中请求不断进入、结束、抢占时，地址和元数据难以稳定管理；
- attention kernel 需要追踪海量小指针，访问不规则；
- prefix cache 很难以固定粒度共享和引用计数。

vLLM 的做法是启动时先申请一大块 KV Cache pool，再把它规则地编号：

```text
一大块 tensor
    -> page 0
    -> page 1
    -> page 2
    -> ...
```

运行时的“申请 page”主要是从空闲 block ID 队列中取一个编号，而不是再次向 CUDA 申请相同大小的底层显存；“释放 page”主要是把这个编号归还到可复用队列。

固定 page 大小是一种折中：

| block_size 较小 | block_size 较大 |
|---|---|
| 请求尾部浪费较少 | block table 项更少 |
| prefix-cache 粒度更细 | 元数据和间接寻址开销较低 |
| page 数和表项更多 | 最后一个未填满 page 可能浪费更多 slot |

假设 `block_size=16`，一个 17-token 请求要两个 page，第二个 page 只用了 1 个 slot，暂时有 15 个尾部空位。这属于固定分页的内部碎片；相比为每个请求申请连续且不断扩展的大块显存，它通常更容易控制。

### 7.9 `KVCacheBlock` Python 对象不是那块 K/V tensor 数据

源码中的 `KVCacheBlock` 类注释写的是 `KV-cache block metadata`，见 [kv_cache_utils.py](./kv_cache_utils.py#L110-L127)。它主要保存：

```text
block_id       哪一个 page 编号
ref_cnt        有多少请求/引用正在使用
block_hash     prefix cache 的内容标识
prev/next      空闲队列中的链表关系
is_null        是否为特殊 null block
```

这个 Python 对象本身并不装 `[2, block_size, num_kv_heads, head_size]` 那些 K/V 浮点数。

可以把两层关系理解成：

```text
CPU/调度器元数据：KVCacheBlock(block_id=37, ref_cnt=1, ...)
                              |
                              | block_id 作为索引
                              v
GPU/worker 数据：kv_cache tensor 中的本地 page 37
```

`SingleTypeKVCacheManager` 主要操作上面那层元数据：分配哪个 block ID、请求引用哪些 block、何时释放、hash 是否命中。attention backend 才根据这些 ID 和 slot mapping 访问下面那层 GPU K/V 数据。

这也解释了为什么解析 manager 时看不到一行行 K/V tensor 写入代码：manager 是“仓库管理员”，不是“搬运 K/V 数值的 CUDA kernel”。

### 7.10 把五维 tensor 的每一级索引彻底展开

仍以 FlashAttention 布局为例：

```text
kv_cache.shape = [2, 1000, 16, 8, 128]
```

选择层级可以逐步展开：

| 表达式 | 选中了什么 | 剩余概念形状 |
|---|---|---|
| `kv_cache` | 整个本地 KV pool | `[2,1000,16,8,128]` |
| `kv_cache[:, 37]` | 第 37 号 page 的 K 和 V | `[2,16,8,128]` |
| `kv_cache[:, 37, 5]` | page 37 中第 5 个 token slot | `[2,8,128]` |
| `kv_cache[0, 37, 5]` | 该 token 的 K | `[8,128]` |
| `kv_cache[1, 37, 5]` | 该 token 的 V | `[8,128]` |
| `kv_cache[0,37,5,2]` | 该 token 的第 2 个本地 KV head 的 K | `[128]` |
| `kv_cache[0,37,5,2,9]` | 上述 head 中第 9 个标量 | 标量 |

真实 CUDA kernel 通常不会真的逐层执行这些 Python 索引，而是根据 stride、block ID 和 slot offset 一次算出地址。表格只是把地址中每个维度的含义展开。

---

## 8. logical block、physical page 和 block table

### 8.1 为什么还需要 logical block

假设一个请求有 35 个 token，`block_size=16`。从请求自己的连续序列看，可以分为：

```text
logical block 0：token 0~15
logical block 1：token 16~31
logical block 2：token 32~34
```

logical block 表示“请求序列中的第几段”，不表示它已经位于 GPU 的哪个 page。

调度器可以把它们分配到任意空闲 page：

```text
logical block 0 -> physical page/block 83
logical block 1 -> physical page/block 7
logical block 2 -> physical page/block 291
```

记录这种映射的表就是 block table：

```text
请求看到的连续序列             GPU KV Cache pool
logical block 0  ----------->  page 83
logical block 1  ----------->  page 7
logical block 2  ----------->  page 291
```

这样即使显存中的空闲 page 不连续，请求仍可逻辑上不断追加 token。这是 PagedAttention 解决 KV Cache 碎片和动态分配问题的核心思想之一。

### 8.2 slot_mapping

当已经知道某个 token 属于哪一个 physical page 时，它的扁平 slot 地址通常可理解为：

```text
slot_id = block_id * local_block_size + offset_in_local_block
```

例如：

```text
block_id = 12
local_block_size = 16
offset = 3

slot_id = 12 * 16 + 3 = 195
```

slot mapping 的作用就是把本批 token 的位置转换成 attention backend 能用于写入 KV Cache 的本地 slot 编号。

### 8.3 page table 类比到底类比了什么

操作系统分页与 PagedAttention 的相似点可以这样对应：

| 操作系统概念 | vLLM/PagedAttention 中的类比 |
|---|---|
| 进程看到的连续虚拟地址 | 请求看到的连续 token position |
| virtual page number | logical block index |
| page table | block table |
| physical frame number | KV Cache block/page ID |
| page 内 offset | token 在 block/page 内的 offset |

例如请求逻辑上连续使用：

```text
token 0, 1, 2, ..., 34
```

其 logical block 可能映射到：

```text
logical 0 -> page 83
logical 1 -> page 7
logical 2 -> page 291
```

attention kernel 不要求 page 83、7、291 在编号或地址上连续，只要 block table 给出的映射正确即可。

类比到这里就应该停止：vLLM 的 block table 是模型服务软件显式管理的数据结构，不等于 GPU MMU 的硬件页表；vLLM page 的大小也不必等于 GPU 硬件页大小。

---

## 9. CP 下的“逻辑大 block”到底是什么

这是 `single_type_kv_cache_manager.py` 中最容易让人困惑的地方。

### 9.1 同一个名字 `block_size` 出现了两种语义

初始化代码是：

```python
self.block_size = kv_cache_spec.block_size
self.dcp_world_size = dcp_world_size
self.pcp_world_size = pcp_world_size
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

见 [single_type_kv_cache_manager.py](./single_type_kv_cache_manager.py#L34-L55)。

这里必须区分：

```text
kv_cache_spec.block_size
    = 单个 rank-local physical page 的 token 容量 B

manager.self.block_size（开启 CP 后）
    = 调度器/manager 使用的全局逻辑记账容量
    = B * pcp_world_size * dcp_world_size
```

变量名相同，但层级不同。manager 并没有真的把每张 GPU 上的 page 扩大；它只是把自己的记账单位提升成了“所有 CP rank 的本地 page 合在一起能覆盖多少个全局 token”。

### 9.2 为什么是乘积

设：

```text
B = 每个 rank-local page 的 slot 数
P = pcp_world_size
D = dcp_world_size
N = P * D = total_cp_world_size
```

每个 CP rank 对同一个 `block_id` 都有一个容量为 `B` 的本地 page。整个 CP group 合起来可容纳：

```text
B * N = B * P * D 个全局 token
```

因此 manager 用：

```text
virtual/logical block_size = B * P * D
```

就可以继续使用普通的：

```python
num_logical_blocks = cdiv(num_tokens, manager.block_size)
```

而不必在每个申请/释放逻辑里重复理解 CP 分片。

### 9.3 “逻辑 block 跨 GPU”和“本地 page 在单 GPU”不矛盾

假设 `DCP=2`，同一个 `block_id=12`：

```text
全局逻辑/虚拟 block 12
├── GPU/rank 0 上的本地 page 12，容量 B
└── GPU/rank 1 上的本地 page 12，容量 B
```

两个本地 page 使用相同的数字 ID，是为了让各 worker 使用同步的 block table；但它们位于不同 GPU 的地址空间，是两块完全独立的显存。

准确说法是：

> 一个全局逻辑 block 所表示的 token/KV 内容，被分片到多个 rank-local page；任何一个 rank-local page 自身都只存在于一张 GPU。

错误理解是：

```text
存在一块 CUDA allocation，它的一半地址在 GPU 0、另一半地址在 GPU 1
```

vLLM 这里表达的不是这种跨设备单 allocation。

### 9.4 默认 token 交错示例

设：

```text
本地 block_size B = 4
PCP = 1
DCP = 2
total_cp_world_size = 2
virtual_block_size = 4 * 2 = 8
cp_kv_cache_interleave_size = 1（默认）
```

一个虚拟 block 覆盖 8 个全局 token。默认按 token 交错：

```text
全局 token： 0  1  2  3  4  5  6  7
所属 rank：  0  1  0  1  0  1  0  1
本地 slot： 0  0  1  1  2  2  3  3
```

于是：

```text
rank 0 的本地 page：slot 0~3 保存全局 token 0, 2, 4, 6
rank 1 的本地 page：slot 0~3 保存全局 token 1, 3, 5, 7
```

### 9.5 block 级交错示例

如果设置：

```text
cp_kv_cache_interleave_size = block_size = 4
```

则同一个虚拟 block 内会先填满前一个 rank 的本地 page，再填下一个：

```text
rank 0 的本地 page：全局 token 0, 1, 2, 3
rank 1 的本地 page：全局 token 4, 5, 6, 7
```

因此以前常见的：

```text
GPU 0 保存 token 0~15
GPU 1 保存 token 16~31
```

只对应 `interleave_size == block_size` 的块级分布，不是当前默认 `interleave_size=1` 的 token 级交错。

配置注释见 [parallel.py](../../config/parallel.py#L327-L340)。实际 slot kernel 使用：

```python
virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
```

并根据 `TOTAL_CP_RANK` 和 `CP_KV_CACHE_INTERLEAVE_SIZE` 判断 token 是否属于本 rank，见 [block_table.py](../worker/block_table.py#L350-L372)。

### 9.6 为什么 manager 只拿一个 block_id 就够了

调度器维护的是全局逻辑 block table，例如为请求分配 `block_id=12`。每个 CP worker 收到同一份编号后：

1. 在自己的本地 KV Cache tensor 中找到本地 page 12；
2. 根据 token 位置和 `total_cp_rank` 判断这个 token 是否属于自己；
3. 属于自己时，计算本地 slot offset 并写入；
4. 不属于自己时，slot mapping 写成无效/PAD slot，不在本 rank 保存该 token。

所以调度器不需要分别维护：

```text
GPU 0 page id
GPU 1 page id
GPU 2 page id
...
```

同一个逻辑 block ID 加上 rank-local 地址空间，就足以定位每张 GPU 自己的那一份 page。

### 9.7 当前版本中 PCP 对乘积的影响

通用代码写成：

```text
total_cp_world_size = P * D
```

是为了统一描述 PCP/DCP 的二维所有权。如果两个维度都可用，`(pcp_rank, dcp_rank)` 唯一标识一个 CP owner。

但是当前 checkout 的 PCP attention backend 尚未可用，因此当前常见可执行路径是：

```text
P = 1
virtual_block_size = B * D
```

阅读 `single_type_kv_cache_manager.py` 时，要把 `* P` 理解成通用/演进中的 PCP 框架，而不是认为当前 backend 已经完成所有 PCP 计算路径。

### 9.8 从全局 token 一直算到本地 slot：完整映射表

现在把 manager、block table、CP rank 和本地 tensor 串起来。设：

```text
local block_size B = 4
PCP = 1
DCP = 2
total_cp_world_size = 2
interleave_size = 1
```

因此一个 manager logical block 覆盖：

```text
virtual_block_size = B * P * D = 8 个全局 token
```

假设调度器给这个请求分配了：

```text
logical block 0 -> block_id 83
logical block 1 -> block_id 7
```

前 16 个全局 token 的映射如下：

| 全局 position | logical block index | block_id | owner CP rank | 本地 offset | 该 owner 上的 slot_id |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 83 | 0 | 0 | `83*4+0 = 332` |
| 1 | 0 | 83 | 1 | 0 | `83*4+0 = 332` |
| 2 | 0 | 83 | 0 | 1 | `83*4+1 = 333` |
| 3 | 0 | 83 | 1 | 1 | `83*4+1 = 333` |
| 4 | 0 | 83 | 0 | 2 | `83*4+2 = 334` |
| 5 | 0 | 83 | 1 | 2 | `83*4+2 = 334` |
| 6 | 0 | 83 | 0 | 3 | `83*4+3 = 335` |
| 7 | 0 | 83 | 1 | 3 | `83*4+3 = 335` |
| 8 | 1 | 7 | 0 | 0 | `7*4+0 = 28` |
| 9 | 1 | 7 | 1 | 0 | `7*4+0 = 28` |
| 10 | 1 | 7 | 0 | 1 | `7*4+1 = 29` |
| 11 | 1 | 7 | 1 | 1 | `7*4+1 = 29` |
| 12 | 1 | 7 | 0 | 2 | `7*4+2 = 30` |
| 13 | 1 | 7 | 1 | 2 | `7*4+2 = 30` |
| 14 | 1 | 7 | 0 | 3 | `7*4+3 = 31` |
| 15 | 1 | 7 | 1 | 3 | `7*4+3 = 31` |

表中 rank 0 和 rank 1 可能出现相同的数字 `slot_id=332`，但不冲突，因为它们索引的是两张 GPU 上各自独立的 KV Cache tensor：

```text
rank 0 的 slot 332：GPU 0 地址空间中的数据
rank 1 的 slot 332：GPU 1 地址空间中的数据
```

对 position 5，kernel 中的推导可以逐步写成：

```text
virtual_block_size = 4 * 2 = 8
block_index = 5 // 8 = 0
block_number = block_table[0] = 83
virtual_block_offset = 5 - 0*8 = 5

owner = (5 // interleave_size) % total_cp_world_size
      = (5 // 1) % 2
      = 1

local_offset = (5 // (2*1))*1 + (5 % 1)
             = 2

slot_id = 83*4 + 2 = 334
```

所以 position 5 只在 CP rank 1 有效；rank 0 对这个 token 会得到 PAD/无效 slot。源码中的 Triton kernel 正是在批量做这套计算。

如果把 `interleave_size` 改成 4，同一个 logical block 的 owner 就会变成：

```text
position 0~3 -> rank 0，本地 offset 0~3
position 4~7 -> rank 1，本地 offset 0~3
```

这时才对应“GPU 0 先保存连续四个 token，GPU 1 再保存连续四个 token”的块级例子。

---

## 10. 从请求 token 到 GPU KV 地址的完整路径

可以把整个流程串成下面这条链：

```text
请求中的全局 token position
        |
        | 除以 virtual_block_size
        v
请求的 logical block index
        |
        | 查询 block table
        v
统一分配的 block_id
        |
        | 根据 total_cp_rank/interleave 判断是否属于本 rank
        v
本 rank 的 local offset
        |
        | slot_id = block_id * local_block_size + local_offset
        v
本 GPU KV Cache tensor 中的 K/V 地址
```

其中：

- scheduler/manager 主要处理 logical block 和 block ID；
- worker 的 block table/slot mapping 负责把 token 位置变成本地 slot；
- attention backend 根据 slot mapping 写入或读取实际 K/V tensor；
- DCP attention 还需要将不同 rank 的局部 attention output/LSE 合并。

---

## 11. 推荐源码阅读顺序

1. [single_type_kv_cache_manager.py](./single_type_kv_cache_manager.py)：看 manager 为什么把 `block_size` 提升成 CP 逻辑大小，以及如何按逻辑 block 申请/释放。
2. [kv_cache_interface.py](../kv_cache_interface.py)：看 `KVCacheSpec.block_size` 与 `page_size_bytes` 的定义。
3. [flash_attn.py](../attention/backends/flash_attn.py#L129-L143) 或 [flashinfer.py](../attention/backends/flashinfer.py#L346-L358)：看真实 KV Cache tensor shape。
4. [parallel.py](../../config/parallel.py#L307-L340)：看 DCP 复用 TP、CP rank 压平和 interleave 的配置语义。
5. [block_table.py](../worker/block_table.py#L141-L164)：看 `total_cp_rank` 如何传入 slot mapping kernel。
6. [block_table.py](../worker/block_table.py#L350-L372)：逐行推导全局 token 如何映射成本地 slot。
7. [backend.py](../attention/backend.py#L678-L735) 与 [cp_utils.py](../worker/cp_utils.py#L14-L44)：看 DCP/PCP backend 能力检查。
8. [dcp_alltoall.py](../attention/ops/dcp_alltoall.py)：看 DCP output/LSE 通信与精确合并的设计。
9. [context_parallel_deployment.md](../../../docs/serving/context_parallel_deployment.md)：对照项目对 PCP/DCP 的总体定位。

---

## 12. 术语速查

| 术语 | 含义 | 不要混淆成 |
|---|---|---|
| TP | 切权重/head 等张量维度的并行方式 | 与 CP 二选一的方案 |
| CP | 沿 token/sequence 维度做并行的总称 | 只切 KV、不涉及 Q 的固定算法 |
| DCP | 面向 decode 的 CP，核心是切历史 KV 并合并同一批小 Q 的局部结果 | 一个 CUDA block |
| PCP | 面向 prefill 的 CP，核心是切大量 Q，并收集或轮转 KV | 单纯把一个 CTA 做大 |
| rank | 某个分布式通信组内的进程编号 | CUDA thread/block 编号 |
| page/physical block | 一个 rank 本地的固定容量 KV Cache 分配单位 | GPU 硬件虚拟内存页 |
| block_size | 一个本地 page 可容纳的 token 数 | page 数量或字节数 |
| num_blocks | 本地 KV Cache pool 中 page 的数量 | 每页 token 数 |
| slot | 本地 page 内保存一个 token K/V 的位置 | 一个标量或 CUDA thread |
| block_id | 本地 KV Cache pool 中的 page 编号 | token ID |
| logical block | 请求/manager 用来按 token 段记账的逻辑单位 | 单张 GPU 上必然连续的一段显存 |
| block table | logical block 到 block_id 的映射 | attention 权重矩阵 |
| slot mapping | 当前 token 到本 rank 扁平 KV slot 的映射 | 请求 token ID 到词表 ID 的映射 |

最后可以用一句话概括全文：

> vLLM 在每张 GPU 上预分配由许多固定容量 page 组成的 KV Cache pool；调度器用 block table 管理请求的逻辑 token 段，CP 再把一个全局逻辑 block 的 token 分摊到多个 rank-local page。DCP 让多个 rank 为同一批小 Q 扫描不同 KV 分片并合并 LSE，PCP 则让不同 rank 负责不同的大 Q 分片，并通过收集或轮转 KV 完成 prefill。
