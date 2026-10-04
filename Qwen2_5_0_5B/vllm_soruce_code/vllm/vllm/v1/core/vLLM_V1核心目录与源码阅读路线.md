# vLLM V1 核心目录、主调用链与源码阅读路线

> 本文用于回答两个问题：`vllm/v1/core` 是否就是 vLLM 的全部核心，以及学习 vLLM V1 时应该重点阅读哪些文件。

## 一、先给出结论

当前阅读的目录确实是：

```text
vllm/v1/core
```

它非常重要，但更准确地说：

> `vllm/v1/core` 是 vLLM V1 中“请求调度和 KV Cache 资源管理”的核心，并不是整个 vLLM 的全部核心。

这个目录主要回答下面这些问题：

- 当前有很多请求时，本轮应该选择哪些请求运行？
- 每个请求本轮允许计算多少个 token？
- GPU 的 KV Cache 空间是否足够？
- 一个请求应该分配哪些 KV block？
- 哪些 prefix-cache block 已经命中，可以直接复用？
- KV Cache 不足时，应该暂停、抢占或释放哪个请求？
- 不同 Attention 类型的 KV Cache 应该如何协调管理？

但是，它通常不负责下面这些工作：

- Qwen2/Qwen2.5 每一层具体怎样进行前向计算；
- Linear、RMSNorm、RoPE、MLP 等算子怎样实现；
- FlashAttention kernel 怎样执行；
- K、V 数据具体写入 GPU 张量的哪个地址；
- CUDA Graph 怎样捕获和重放；
- Tensor Parallel、Pipeline Parallel 和 NCCL 通信怎样执行；
- GPU 上的 token sampling 怎样实现。

因此，可以把 vLLM V1 粗略分成两部分：

```text
控制与资源管理部分
    engine + scheduler + KV Cache manager

GPU 数据与计算部分
    executor + worker + model runner + model_executor + attention backend
```

`vllm/v1/core` 主要属于第一部分。

---

## 二、一次请求在 vLLM V1 中经过的主要链路

理解 vLLM 源码时，最重要的不是孤立地记住每个文件，而是先建立一条完整的请求执行链：

```text
用户请求 / OpenAI API
        ↓
AsyncLLM / LLMEngine
        ↓
EngineCore.step()
        ↓
Scheduler.schedule()
        ├── 决定本轮运行哪些请求
        ├── 决定每个请求本轮计算多少 token
        └── 向 KVCacheManager 申请 KV block
                ↓
          KVCacheCoordinator
                ↓
      SingleTypeKVCacheManager
                ↓
             BlockPool
        ↓
生成 SchedulerOutput
        ↓
Executor.execute_model()
        ↓
GPUWorker
        ↓
GPUModelRunner.execute_model()
        ├── 更新请求批次
        ├── 准备 input_ids 和 positions
        ├── 准备 block table 和 slot mapping
        ├── 执行 Qwen2ForCausalLM.forward()
        ├── 调用 Attention Backend
        └── 对模型输出进行采样
        ↓
Scheduler.update_from_output()
        ↓
OutputProcessor / Detokenizer
        ↓
向用户返回生成结果
```

这条链路基本就是学习 vLLM V1 源码时最重要的主干。后面遇到一个类或函数时，可以先判断它位于这条链的哪一层，再阅读其内部实现。

---

## 三、`vllm/v1/core` 目录主要负责什么

可以把 `vllm/v1/core` 理解成 vLLM 的“调度和 KV Cache 资源管理中心”。

### 3.1 `sched/scheduler.py`

源码：[sched/scheduler.py](sched/scheduler.py)

这是 vLLM V1 最重要的文件之一。它主要负责：

- 管理 waiting、running 等请求队列；
- 从等待队列和运行队列中选择本轮可以执行的请求；
- 决定每个请求本轮应该计算多少 token；
- 处理 prefill、decode 和 chunked prefill；
- 检查本轮可使用的 token budget；
- 检查 KV Cache 是否还有足够的空间；
- 调用 `KVCacheManager.allocate_slots()` 为请求分配 KV block；
- 在资源不足时进行抢占或重新调度；
- 处理 prefix caching、LoRA、Encoder/多模态输入等调度条件；
- 模型执行完成后，根据输出更新请求状态；
- 判断请求是否已经结束，并释放它所占用的资源。

其中最值得优先阅读的方法是：

```python
Scheduler.schedule()
```

它把“有哪些请求”和“本轮有多少计算资源”转换为一个 `SchedulerOutput`。这个输出随后会交给 Executor 和 GPU Worker。

如果只看 KV Cache 管理器，却没有先看 `Scheduler.schedule()`，就容易不知道：

- KV block 为什么在这个时间点申请；
- `num_new_tokens` 是谁计算的；
- 为什么同一个请求会多次调用分配函数；
- 为什么一次调用既可能包含已有 block，也可能包含新 block；
- 为什么某些 block 可以被驱逐。

### 3.2 `kv_cache_manager.py`

源码：[kv_cache_manager.py](kv_cache_manager.py)

`KVCacheManager` 是 Scheduler 操作 KV Cache 的上层入口，主要负责：

- 查询请求已经命中的 prefix-cache block；
- 获取请求当前拥有的 KV block；
- 根据本轮新增 token 数量分配 block；
- 释放一个请求占用的 KV block；
- 重置 prefix cache；
- 在 Scheduler 和底层 KV Cache 管理策略之间提供统一接口；
- 将不同 KV Cache 组的复杂逻辑交给 `KVCacheCoordinator`。

一个典型调用关系是：

```text
Scheduler.schedule()
    ↓
KVCacheManager.get_computed_blocks()
    ↓
查询 prefix cache 命中

Scheduler.schedule()
    ↓
KVCacheManager.allocate_slots()
    ↓
计算并分配本轮需要的 KV block
```

之前研究的 `get_num_blocks_to_allocate()`，只是整个分配流程中的一个底层容量计算步骤。理解它时，需要将它放回 `KVCacheManager.allocate_slots()` 和 `Scheduler.schedule()` 的调用背景中。

### 3.3 `kv_cache_coordinator.py`

源码：[kv_cache_coordinator.py](kv_cache_coordinator.py)

`KVCacheCoordinator` 负责协调一个或多个 KV Cache group。

之所以需要 Coordinator，是因为某些模型不一定所有层都使用完全相同的 Attention 和缓存规格。例如模型中可能同时存在：

- Full Attention；
- Sliding Window Attention；
- Local Attention；
- Mamba 或其他具有不同状态结构的层。

这些层可能具有不同的：

- block size；
- KV Cache page 大小；
- token 覆盖范围；
- block 数量计算规则；
- prefix-cache 处理规则。

因此，vLLM 使用 Coordinator 对它们进行统一管理。大致关系如下：

```text
KVCacheManager
    ↓
KVCacheCoordinator
    ├── 某种 Attention 对应的 SingleTypeKVCacheManager
    ├── 另一种 Attention 对应的 SingleTypeKVCacheManager
    └── 所有类型共享或协调使用的 BlockPool
```

对于普通、所有层 KV Cache 规格一致的模型，Coordinator 的结构看起来可能有些绕；但在支持 Hybrid KV Cache 的模型中，它是必要的抽象层。

### 3.4 `single_type_kv_cache_manager.py`

源码：[single_type_kv_cache_manager.py](single_type_kv_cache_manager.py)

这是当前正在重点研究的文件。

它负责“某一种 KV Cache 类型”的 block 管理策略，主要处理：

- 一个请求在该种 Attention 类型下需要多少 block；
- prefix cache 已经命中了多少 block；
- 当前请求已经持有哪些 block；
- 本轮新增 token 还需要申请多少 block；
- 哪些已缓存 block 可以作为可驱逐 block；
- 哪些逻辑 block 在当前 KV Cache group 中需要被跳过；
- DCP/PCP 等场景下逻辑 block 数量和物理分配之间的换算。

必须注意：

> 这个文件主要管理 block 的逻辑关系、分配策略和元数据，并不直接执行 Attention，也不直接运行 FlashAttention kernel。

它一般也不负责真正创建 GPU 上存放 K、V 数据的巨大张量。GPU 侧 KV Cache 的初始化、block table 构造和 kernel 访问在 Worker、Model Runner 和 Attention Backend 中完成。

所以只看这个文件时，很容易产生下面这些疑问：

- block ID 最终怎样对应 GPU 内存？
- block table 在哪里生成？
- 一个 token 怎样映射到 block 内的 slot？
- Attention kernel 怎样根据 block table 读取 K 和 V？

这些问题需要继续查看 `v1/worker/block_table.py` 和实际 Attention Backend。

### 3.5 `block_pool.py`

源码：[block_pool.py](block_pool.py)

`BlockPool` 是 KV block 的公共资源池，主要管理：

- 空闲 block 队列；
- 已分配 block；
- block ID；
- block 的引用计数；
- block hash；
- prefix cache 的哈希映射；
- block 的分配和释放；
- 哪些 block 可以被驱逐；
- block 的 touch/访问更新操作。

可以把它通俗地理解为：

```text
所有可用 KV page/block 的“号码和使用状态管理处”
```

例如，假设 GPU 为 KV Cache 预留了 1000 个固定大小的物理 page，那么可以为这些 page 编号：

```text
block_id = 0, 1, 2, ..., 999
```

`BlockPool` 主要记录：

- 哪些编号还空闲；
- 哪些编号已经分配给请求；
- 某个编号被多少请求或缓存条目引用；
- 某个编号中保存的完整 token block 对应什么 hash；
- 在没有空间时，哪个缓存 block 可以回收。

它重点管理的是 block 元数据。实际 K、V 数值仍保存在 GPU Worker 初始化的 KV Cache 张量中。

### 3.6 `kv_cache_utils.py`

源码：[kv_cache_utils.py](kv_cache_utils.py)

这个文件主要提供 KV Cache 相关的数据结构和辅助函数，例如：

- `KVCacheBlock` 等 block 元数据结构；
- block hash 的计算和组合；
- KV Cache 配置计算；
- 根据显存预算计算可创建的 block 数量；
- KV Cache group 的构造；
- 不同 Attention/KV Cache 规格之间的辅助计算。

阅读 `BlockPool` 或 Coordinator 时，如果遇到不熟悉的 block 数据结构和 cache config，可以回到该文件查定义。

### 3.7 `encoder_cache_manager.py`

源码：[encoder_cache_manager.py](encoder_cache_manager.py)

这个文件负责 Encoder 或多模态相关缓存，例如：

- Encoder 输出；
- 图片、音频等多模态输入经过 Encoder 后得到的特征；
- Encoder 输入的缓存预算和释放。

它不是普通 Decoder Self-Attention KV Cache 的主要管理文件。只学习纯文本 Decoder 模型时，可以稍后再看。

### 3.8 其他辅助文件

- [kv_cache_metrics.py](kv_cache_metrics.py)：KV Cache 使用情况和命中率等指标；
- [sched/output.py](sched/output.py)：Scheduler 输出给执行侧的数据结构；
- [sched/interface.py](sched/interface.py)：Scheduler 相关接口；
- [sched/request_queue.py](sched/request_queue.py)：请求队列实现；
- [../kv_cache_interface.py](../kv_cache_interface.py)：KV Cache spec、group 和配置相关的公共接口。

---

## 四、`vllm/v1/engine`：驱动整个推理循环

`vllm/v1/engine` 是 V1 Engine 的主控制层。

### 4.1 `engine/core.py`

源码：[../engine/core.py](../engine/core.py)

其中的 `EngineCore` 是理解 V1 架构最重要的类之一。源码将它描述为 vLLM Engine 的内部循环。

最核心的方法是：

```python
EngineCore.step()
```

它的主干逻辑可以简化为：

```python
scheduler_output = self.scheduler.schedule()

model_output = self.model_executor.execute_model(
    scheduler_output,
)

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

也就是：

```text
Scheduler 生成本轮计划
        ↓
Executor/Worker 执行模型
        ↓
Scheduler 根据结果更新所有请求
```

因此，理解 V1 主流程时，应该先从 `EngineCore.step()` 开始，再进入 `Scheduler.schedule()`，最后深入 KV Cache 管理器。

一个容易混淆的命名是：

> 真正名为 `EngineCore` 的类位于 `vllm/v1/engine/core.py`，而不是 `vllm/v1/core` 目录。

### 4.2 Engine 的其他重要文件

- [../engine/async_llm.py](../engine/async_llm.py)：异步引擎接口，适合在线服务；
- [../engine/llm_engine.py](../engine/llm_engine.py)：同步 Engine 封装；
- [../engine/output_processor.py](../engine/output_processor.py)：处理模型输出、停止条件和请求结果；
- [../engine/detokenizer.py](../engine/detokenizer.py)：将 token 增量转换为文本。

在多进程服务架构中，API/前端进程和 EngineCore 进程之间通常还需要进行请求与结果通信。

---

## 五、`vllm/v1/executor`：把执行计划交给 Worker

入口源码：[../executor/abstract.py](../executor/abstract.py)

Executor 负责把 Scheduler 产生的执行计划发送给一个或多个 Worker，并收集执行结果。

它主要解决：

- 当前是单进程还是多进程执行；
- 有多少个设备和 Worker；
- 怎样在 Worker 上初始化模型和 KV Cache；
- 怎样向所有相关 Worker 分发 `execute_model` 调用；
- 怎样收集各个 Worker 的结果；
- 不同部署后端使用哪一种 Executor。

常见实现包括：

- `UniProcExecutor`：单进程或单设备执行；
- `MultiprocExecutor`：本机多进程、多 GPU 执行；
- Ray Executor：通过 Ray 管理分布式 Worker；
- 其他硬件平台对应的 Executor。

Executor 自己通常不定义 Qwen2 的数学计算，其主要职责更接近：

```text
SchedulerOutput
        ↓
将任务发送给一个或多个 GPU Worker
        ↓
等待并收集 Worker 的执行结果
```

---

## 六、`vllm/v1/worker`：GPU 执行侧的核心

### 6.1 `gpu_worker.py`

源码：[../worker/gpu_worker.py](../worker/gpu_worker.py)

`GPUWorker` 主要负责：

- 初始化当前 CUDA device；
- 设置当前 Worker 的 rank 和分布式环境；
- 加载模型；
- 测量或估计可用于 KV Cache 的 GPU 显存；
- 创建实际的 GPU KV Cache；
- 编译、warmup 或捕获 CUDA Graph；
- 接收 Executor 的调用并执行模型；
- 管理当前 GPU 上模型运行所需的状态。

可以将其理解为“一张 GPU 对应的主要执行管理对象”。

### 6.2 `GPUModelRunner`

当前源码中可以看到两套相关实现：

- [../worker/gpu_model_runner.py](../worker/gpu_model_runner.py)
- [../worker/gpu/model_runner.py](../worker/gpu/model_runner.py)

`gpu_worker.py` 会根据 `use_v2_model_runner` 等配置选择实际使用的 Model Runner。刚开始学习时不需要同时阅读两套实现，应当先看 `gpu_worker.py` 的选择逻辑，再跟进当前配置真正启用的实现。

Model Runner 主要负责：

- 将 Scheduler 下发的请求加入或更新到 GPU 执行批次；
- 准备 `input_ids`；
- 准备 `positions`；
- 准备 Attention metadata；
- 构建或更新 block table；
- 构建 token 到 KV Cache 位置的 slot mapping；
- 调用模型的 `forward()`；
- 处理 logits；
- 执行采样；
- 管理 CUDA Graph 所需的固定形状和缓冲区。

### 6.3 `worker/block_table.py`

源码：[../worker/block_table.py](../worker/block_table.py)

这个文件与 KV Cache 学习密切相关。它负责把 Scheduler 分配的 block ID 转换为 GPU 执行时需要的 block table 和 slot mapping。

可以这样区分各层职责：

```text
v1/core 下的 KV Cache 管理器
    管理“一个请求拥有哪些 block ID”

v1/worker/block_table.py
    管理“这些 block ID 在当前 GPU 执行批次中怎样组织和索引”

Attention Backend / kernel
    根据 block table 和 slot mapping 真正读写 K、V 数据
```

这也解释了为什么只阅读 `SingleTypeKVCacheManager` 不能看到真正的 GPU 内存访问：它处理的是调度层和资源层的 block，而不是 Attention kernel 的数据搬运过程。

---

## 七、`vllm/v1/attention`：高性能 Attention 和 KV Cache 访问

Attention 接口入口：[../attention/backend.py](../attention/backend.py)

该目录主要负责 V1 Attention Backend 的抽象、选择和具体集成。

不同 Backend 需要处理的内容包括：

- KV Cache 张量布局；
- Attention metadata；
- prefill 和 decode 所需的输入信息；
- 怎样将新 token 的 K、V 写入 paged KV Cache；
- 怎样根据 block table 读取历史 K、V；
- 怎样调用具体的 CUDA/Triton Attention kernel；
- DCP、PCP、Sliding Window 等场景下的特殊处理。

Backend 选择逻辑可以从下面的文件开始看：

[../attention/selector.py](../attention/selector.py)

具体实现可能包括：

- FlashAttention；
- FlashInfer；
- Triton Attention；
- ROCm Attention；
- 针对其他硬件平台的 Attention Backend。

因此，`SingleTypeKVCacheManager` 只会告诉系统应该为请求保留哪些 block；实际 K、V 怎样写入和读取，需要继续跟到当前启用的 Attention Backend。

---

## 八、`vllm/model_executor`：模型结构和通用算子层

这个目录不在 `vllm/v1` 下面，但 V1 仍然大量复用它。`vllm/v1` 并不是一套完全独立、重新实现所有模型层的代码。

### 8.1 Qwen2/Qwen2.5 模型

源码：[../../model_executor/models/qwen2.py](../../model_executor/models/qwen2.py)

Qwen2.5 的文本模型实现主要复用 Qwen2 模型结构，其中的重要类可以概括为：

```text
Qwen2ForCausalLM
└── Qwen2Model
    └── 多个 Qwen2DecoderLayer
        ├── Qwen2Attention
        ├── Qwen2MLP
        └── RMSNorm
```

这些类负责定义模型层面的计算结构，例如：

- embedding；
- Q、K、V 投影；
- RoPE；
- Attention 层调用；
- MLP；
- residual；
- RMSNorm；
- LM Head。

### 8.2 `model_executor/layers`

目录：[../../model_executor/layers](../../model_executor/layers)

这里提供模型共用的高性能层和组件，例如：

- Linear；
- RMSNorm；
- RoPE；
- Attention Layer；
- Vocabulary Parallel Embedding；
- Quantization；
- MoE；
- logits 处理。

需要区分下面三个层次：

```text
v1/worker
    决定怎样把许多请求组织成一个 GPU 批次，并调用模型

model_executor/models/qwen2.py
    定义 Qwen2/Qwen2.5 每一层的模型结构和前向过程

v1/attention
    决定 Attention 最终使用哪个高性能 Backend 和 kernel
```

---

## 九、`vllm/distributed`：多 GPU 并行和通信

入口之一：[../../distributed/parallel_state.py](../../distributed/parallel_state.py)

这个目录主要负责：

- Tensor Parallel；
- Pipeline Parallel；
- Data Parallel；
- Decode Context Parallel；
- Process Group 管理；
- rank 和 world size；
- NCCL 或其他设备通信；
- all-reduce、all-gather 等 collective operation。

当研究 DCP、TP、多 GPU KV Cache 或多进程 Worker 时，需要将这里的并行组定义和 Worker/Attention 中的使用方式结合起来阅读。

---

## 十、各核心目录的一句话定位

```text
v1/engine
    驱动整个推理循环，连接请求、Scheduler、Executor 和输出

v1/core/sched
    决定本轮哪些请求运行，以及每个请求运行多少 token

v1/core 下的 KV Cache 文件
    决定请求需要、持有和释放哪些 KV block

v1/executor
    把 Scheduler 的执行计划发送给一个或多个 Worker

v1/worker
    准备 GPU 输入、block table，并真正调用模型

model_executor
    定义 Qwen2 等模型结构以及通用模型层

v1/attention
    执行高性能 Attention，并根据 block table 读写 KV Cache

distributed
    管理多进程、多 GPU 并行组和通信

entrypoints
    提供离线 LLM 接口和 OpenAI 兼容的在线服务入口
```

---

## 十一、推荐的源码阅读顺序

目前是从 `SingleTypeKVCacheManager` 开始阅读的，相当于直接进入了比较深的 KV Cache 策略层。这里的函数会假定读者已经知道请求怎样被 Scheduler 调度，因此单独阅读时容易感觉许多变量和分支缺少背景。

建议暂时向上退两层，再按照下面的顺序重新进入 KV Cache。

### 第一阶段：先建立完整主流程

1. 阅读 [vLLM 架构说明](../../../docs/design/arch_overview.md)，重点关注 V1 的进程和组件结构；
2. 阅读 [EngineCore](../engine/core.py) 的初始化过程；
3. 重点阅读 `EngineCore.step()`；
4. 阅读 [Scheduler](sched/scheduler.py) 的初始化过程；
5. 重点阅读 `Scheduler.schedule()`；
6. 阅读 [Request](../request.py)；
7. 阅读 [SchedulerOutput](sched/output.py)。

这一阶段的目标是回答：

```text
请求从哪里进入 EngineCore？
Scheduler 保存了哪些请求状态？
SchedulerOutput 中包含什么？
Executor 接收到的到底是什么？
一次 EngineCore.step() 为什么只推进请求的一部分 token？
```

### 第二阶段：完整理解 KV Cache

建议按照下面的调用层次阅读：

1. `Scheduler.schedule()` 中调用 KV Cache 的位置；
2. `KVCacheManager.get_computed_blocks()`；
3. `KVCacheManager.allocate_slots()`；
4. `KVCacheCoordinator`；
5. `SingleTypeKVCacheManager`；
6. `BlockPool`；
7. `KVCacheBlock` 和 block hash；
8. `v1/worker/block_table.py`；
9. 当前启用的 Attention Backend 怎样使用 block table。

这样，下面这些概念才能形成一条完整链路：

```text
请求 token
    ↓
逻辑 token 位置
    ↓
请求需要的逻辑 block
    ↓
Scheduler 分配的 block ID
    ↓
Worker 中的 block table
    ↓
token 对应的 slot mapping
    ↓
Attention kernel 访问实际 K、V 数据
```

之前遇到的 `A`、`H`、`num_evictable_blocks`、slot、page、block ID、DCP 和 PCP 等概念，也应当放在这条链路中理解。

### 第三阶段：理解 GPU 执行

1. 阅读 [Executor 抽象接口](../executor/abstract.py)；
2. 根据运行模式选择 `UniProcExecutor` 或 `MultiprocExecutor`；
3. 阅读 [GPUWorker](../worker/gpu_worker.py)；
4. 确认当前使用哪一套 GPUModelRunner；
5. 阅读 `GPUModelRunner.execute_model()`；
6. 跟踪输入张量、block table 和 Attention metadata 的构造；
7. 最后再看 CUDA Graph、sampling 和其他优化。

### 第四阶段：理解模型和 Attention

1. 阅读 [Qwen2 模型实现](../../model_executor/models/qwen2.py)；
2. 重点跟踪 `Qwen2ForCausalLM → Qwen2Model → Qwen2DecoderLayer → Qwen2Attention`；
3. 阅读 `model_executor/layers` 中 Qwen2 使用的通用层；
4. 阅读 [AttentionBackend 接口](../attention/backend.py)；
5. 阅读 [Attention Backend 选择逻辑](../attention/selector.py)；
6. 只深入当前环境实际启用的 FlashAttention、FlashInfer 或其他 Backend。

### 第五阶段：按兴趣补充分布式和高级功能

完成主链路后，再阅读：

- Tensor Parallel 和通信；
- DCP/PCP；
- Speculative Decoding；
- Structured Output；
- Prefix Cache 的淘汰策略；
- KV Cache Offload；
- 多模态 Encoder Cache；
- LoRA 调度；
- 不同硬件 Backend。

---

## 十二、当前最值得先看的几个文件

如果暂时不想一次阅读太多目录，可以先只看下面这些主干文件：

1. [../engine/core.py](../engine/core.py)：理解整个 Engine 的一次循环；
2. [sched/scheduler.py](sched/scheduler.py)：理解请求怎样被选择和推进；
3. [kv_cache_manager.py](kv_cache_manager.py)：理解 Scheduler 怎样申请 KV Cache；
4. [kv_cache_coordinator.py](kv_cache_coordinator.py)：理解不同 KV Cache group 怎样协调；
5. [single_type_kv_cache_manager.py](single_type_kv_cache_manager.py)：理解单一 Cache 类型的 block 计算；
6. [block_pool.py](block_pool.py)：理解 block 元数据、空闲队列和 prefix hash；
7. [../worker/block_table.py](../worker/block_table.py)：理解 block ID 怎样进入 GPU 执行；
8. [../worker/gpu_worker.py](../worker/gpu_worker.py)：理解一张 GPU 上的执行管理；
9. 当前启用的 GPUModelRunner：理解实际模型执行批次；
10. [../../model_executor/models/qwen2.py](../../model_executor/models/qwen2.py)：理解 Qwen2.5 模型前向；
11. [../attention/backend.py](../attention/backend.py)：理解 Attention Backend 接口。

---

## 十三、学习方向总结

如果学习目标是理解：

> vLLM 为什么可以同时调度大量请求、复用 prefix cache，并在有限显存中高效管理 KV Cache？

那么当前阅读 `vllm/v1/core` 的方向完全正确。

如果学习目标是理解：

> Qwen2.5 的一次 forward 最终怎样落到 GPU、FlashAttention 和 CUDA kernel 上？

那么还必须继续阅读：

```text
v1/executor
    ↓
v1/worker
    ↓
GPUModelRunner
    ↓
model_executor/models/qwen2.py
    ↓
v1/attention
    ↓
具体 CUDA/Triton kernel
```

当前最合适的下一步，是先回到 `EngineCore.step()` 和 `Scheduler.schedule()`，建立“一个调度 step”从开始到结束的完整认识，然后顺着 `KVCacheManager.allocate_slots()` 再次进入已经研究过的 `SingleTypeKVCacheManager`。这样底层变量就不再是孤立的计算，而能对应到真实的请求调度过程。
