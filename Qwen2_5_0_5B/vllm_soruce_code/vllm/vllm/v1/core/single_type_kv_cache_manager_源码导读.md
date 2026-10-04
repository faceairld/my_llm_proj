# `single_type_kv_cache_manager.py` 源码导读

> 目标源码：`vllm/v1/core/single_type_kv_cache_manager.py`  
> 本文基于当前工作区中的 1131 行版本整理。行号均指这一版本，后续升级 vLLM 后可能发生偏移。  
> 阅读目的：理解 vLLM v1 如何针对不同注意力类型，选择不同的 KV Cache 查找、分配、回收和前缀复用策略。

## 1. 先给结论：这个文件处在什么位置

这个文件不是实际执行 Attention 的地方，也不直接保存 GPU 上的 K/V 张量。它是 **KV Cache 的策略层**：

~~~text
Scheduler
  │  决定本轮调度哪些请求、多少 token
  ▼
KVCacheManager                         v1/core/kv_cache_manager.py
  │  处理一个请求完整的 cache 生命周期
  ▼
KVCacheCoordinator                     v1/core/kv_cache_coordinator.py
  │  一个模型可能有多个 KV cache group，逐 group 分发
  ▼
SingleTypeKVCacheManager 及其子类      ← 本文件
  │  根据 Full / Sliding / Mamba 等类型决定具体策略
  ▼
BlockPool                              v1/core/block_pool.py
  │  维护空闲队列、引用计数、哈希索引、分配和释放
  ▼
KVCacheBlock 元数据 + Worker 中的真实 GPU Cache
~~~

可以把四层职责概括为：

| 层 | 主要问题 |
|---|---|
| `KVCacheManager` | “这个请求本轮需要多少槽位？整个操作是否能完成？” |
| `KVCacheCoordinator` | “模型有多个 cache group，应该把操作分发给哪些 manager？” |
| 本文件中的 manager | “这种注意力类型哪些旧块还能用、哪些能丢、怎样找最长命中？” |
| `BlockPool` | “具体拿哪个物理 block，引用计数怎么变，缓存哈希表怎么维护？” |

因此，本文件里的对象主要保存 **block 元数据和请求到 block 的映射**。真实的 K/V 或 Mamba state 由模型执行侧按 block ID 访问。

## 2. 文件中一共有 7 个类

其中 1 个抽象基类、6 个具体管理器：

| 类 | 对应 `KVCacheSpec` | 一句话职责 |
|---|---|---|
| `SingleTypeKVCacheManager` | `KVCacheSpec` | 抽象基类；实现通用的记账、容量预估、分配、缓存、释放和跳过旧块逻辑 |
| `FullAttentionManager` | `FullAttentionSpec`、`MLAAttentionSpec` | 完整注意力；必须保留整个有效前缀，按哈希链从左到右查连续命中 |
| `SlidingWindowManager` | `SlidingWindowSpec` | 滑动窗口注意力；窗口左侧的块可释放，用 null block 保持逻辑位置 |
| `ChunkedLocalAttentionManager` | `ChunkedLocalAttentionSpec` | 分块局部注意力；完整旧 chunk 可视为已处理，只查当前 chunk 内的缓存 |
| `MambaManager` | `MambaSpec` | 管理 Mamba/线性状态而非传统 K/V；支持 `none`、`all`、`align` 三种模式及推测解码 |
| `CrossAttentionManager` | `CrossAttentionSpec` | 编码器—解码器模型的 cross-attention cache；按请求静态分配，不做跨请求前缀缓存 |
| `SinkFullAttentionManager` | `SinkFullAttentionSpec` | 在 Full Attention 行为上额外永久预留全局 sink blocks |

文件末尾还有：

- `spec_manager_map`：spec 精确类型到 manager 类的映射；
- `get_manager_for_kv_cache_spec()`：根据 spec 创建 manager 的工厂函数。

## 3. 阅读本文件前必须理解的几个概念

### 3.1 Token、block、page 与 block table

vLLM 不按单个 token 动态申请显存，而是把若干 token 的 cache 放进一个 block/page。

假设 `block_size = 4`：

~~~text
逻辑 token 位置:   0  1  2  3 | 4  5  6  7 | 8  9 ...
逻辑 block 索引:       0      |      1      |   2
物理 block ID:        17      |      3      |  26
~~~

请求的 block table 只需记录：

~~~python
req_to_blocks["request-A"] = [block_17, block_3, block_26]
~~~

逻辑 block 索引与物理 block ID 可以完全不同。manager 管理的是这张逻辑表，模型执行侧再用 block ID 定位真实 GPU page。

### 3.2 KV cache group 与 spec

`KVCacheSpec` 是一组层的静态 cache 规格，至少包含 `block_size`。Full、Sliding、Mamba 等子类还带有各自参数。

`KVCacheCoordinator` 会遍历 `kv_cache_config.kv_cache_groups`，为每个 group 调用本文件末尾的工厂，创建一个 single-type manager。多个 manager 共享同一个 `BlockPool`，但各自维护本 group 的请求 block table。

这也是类名中 “SingleType” 的含义：一个实例只处理一种 cache spec/group 的策略。

### 3.3 `KVCacheBlock` 只是元数据

`KVCacheBlock` 的关键字段是：

| 字段 | 含义 |
|---|---|
| `block_id` | GPU block 的编号，范围通常是 `0 .. num_gpu_blocks-1` |
| `ref_cnt` | 当前有多少活跃引用；大于 0 时不能作为可驱逐空闲块使用 |
| `block_hash` | 完整块加入 prefix cache 后使用的哈希键，实际还包含 group ID |
| `is_null` | 是否为特殊 null block |

重要语义：

- `ref_cnt > 0`：当前至少有请求正在引用它；
- `ref_cnt == 0`：它可能仍带有 `block_hash`、仍能被 prefix cache 查到，但已进入可驱逐的 free queue；
- “free” 不一定意味着立刻删除缓存内容，而是把活跃引用减掉；
- 真正再次分配该物理块时，`BlockPool` 才可能驱逐旧哈希并清空元数据。

### 3.4 block hash 为什么是“链式前缀哈希”

请求的 block hash 不只依赖本块 token，还依赖前一个 block 的 hash。可抽象为：

~~~text
h0 = H(NONE, tokens[0:B], extra_keys)
h1 = H(h0,   tokens[B:2B], extra_keys)
h2 = H(h1,   tokens[2B:3B], extra_keys)
~~~

因此第 `i` 个 hash 间接包含它之前的全部前缀信息。Full Attention 查缓存时，只要中间一个块 miss，后面的链式 hash 也不可能属于同一个请求前缀，所以可以立刻停止。

`BlockPool.get_cached_block(hash, group_ids)` 还要求给出的所有 group ID 都命中；任一 group miss，整组结果就是 `None`。

### 3.5 null block 的作用

滑动窗口、chunked local attention 和 Mamba 会释放已经不需要的旧状态，但不能直接删除列表元素，否则后面的逻辑 block 索引会整体左移。

因此代码把旧位置替换为同一个 `null_block`：

~~~text
释放前: [B10, B11, B12, B13]
释放后: [NULL, NULL, B12, B13]
          位置 0  位置 1  位置 2  位置 3
~~~

这样同时满足：

1. block table 长度仍能表达“已经处理到第几个逻辑 block”；
2. 后续真实 block 的位置不变；
3. null block 不会被缓存，也不会作为普通 block 放回 free queue；
4. 返回的 cache-hit 长度仍可用 `len(blocks) * block_size` 计算。

`null_block` 的引用计数没有普通 block 的严格语义，不应依赖它的 `ref_cnt`。

### 3.6 几个容易混淆的数量

| 名称 | 含义 |
|---|---|
| `num_tokens` | 本次要求拥有槽位的 token 总数，包含已分配部分 |
| `num_required_blocks` | `cdiv(num_tokens, block_size)`，装下这些 token 至少需要的逻辑块数 |
| `num_req_blocks` | 当前 `req_to_blocks[request_id]` 的列表长度，null 位置也计数 |
| `new_computed_blocks` | 本轮刚从本地 prefix cache 查到、尚未归入该请求的块 |
| `num_local_computed_tokens` | 本地已经算好或本地缓存命中的 token |
| `num_external_computed_tokens` | 由 KV connector 等外部来源提供、需要本地槽位承接的 token |
| `total_computed_tokens` | local 与 external 之和，用于计算当前窗口左边界 |
| `num_tokens_main_model` | 主/目标模型需要的 token 数，不包含单独的 lookahead 槽位 |
| `num_cached_block[id]` | 已经处理过缓存注册的“逻辑块位置游标”，其中可以包含 null 位置 |

最后一项不要机械理解为“非 null 且一定在哈希表中的实际块数”。例如 skipped prefix 用 null 填充后，这些位置也会包含在游标中。

## 4. 抽象基类 `SingleTypeKVCacheManager`

源码位置：第 28—416 行。

它把所有注意力类型都需要的生命周期逻辑集中起来，把两个真正依赖注意力语义的问题留给子类：

- `find_longest_cache_hit()`：怎样才算最长可复用前缀；
- `get_num_common_prefix_blocks()`：是否支持为 cascade attention 计算公共前缀。

`get_num_skipped_tokens()` 不是抽象方法，默认返回 0；有局部窗口语义的子类再覆写。

### 4.1 初始化字段

`__init__()` 接收：

~~~python
kv_cache_spec
block_pool
enable_caching
kv_cache_group_id
dcp_world_size = 1
pcp_world_size = 1
~~~

主要成员如下：

| 成员 | 作用 |
|---|---|
| `block_size` | 本 manager 的逻辑 block token 数 |
| `dcp_world_size` / `pcp_world_size` | Decode/Prefill Context Parallelism 的并行规模 |
| `kv_cache_spec` | 本 group 的静态规格 |
| `block_pool` | 所有 manager 共享的底层块池 |
| `enable_caching` | 是否允许 prefix caching |
| `new_block_ids` | 自上次读取后，新分配且需要上层关注的 Full Attention block ID |
| `req_to_blocks` | `request_id -> 逻辑 block 列表` |
| `num_cached_block` | `request_id -> 缓存注册位置游标` |
| `kv_cache_group_id` | 当前 group 的编号，用于哈希隔离 |
| `_null_block` | 从 `BlockPool` 取得的特殊占位块 |

当 `dcp_world_size * pcp_world_size > 1` 时：

~~~python
self.block_size = spec.block_size * dcp_world_size * pcp_world_size
~~~

也就是说，manager 用一个放大后的“逻辑块”进行 token 数量记账，使分配单位与上下文并行后的整体进度一致。`kv_cache_spec.block_size` 本身没有被修改。

### 4.2 两张最重要的请求状态表

#### `req_to_blocks`：所有权/位置表

~~~python
defaultdict[str, list[KVCacheBlock]]
~~~

它回答“这个请求在本 group 的每个逻辑位置引用哪个物理 block”。列表可能包含：

- 新分配、尚未写满的普通块；
- 从 prefix cache 命中的共享块；
- 已经缓存并带 hash 的块；
- 用来保留逻辑位置的 null block。

#### `num_cached_block`：缓存游标

~~~python
dict[str, int]
~~~

它表示从列表开头算起，有多少个位置已经作为“已缓存前缀”处理过。后续 `cache_blocks()` 只处理游标之后新变成完整块的区间。

成员是否存在也被用作快速路径判断：若 request ID 在字典中，代码认为它不会在运行途中突然得到新的本地 prefix-cache hit。但要注意，刚开始运行且还没有任何完整块的新请求，可能暂时还没有这个键；所以它更准确地说是“缓存状态已建立”的快速标记，而不是完整的请求状态机。

### 4.3 `_get_num_evictable_blocks()`

源码位置：第 74—76 行。

~~~python
sum(block.ref_cnt == 0 and not block.is_null for block in blocks)
~~~

它统计给定命中块中有多少块目前是可驱逐候选。

为什么 prefix hit 还要消耗 free capacity？

- 一个已缓存块可以 `ref_cnt == 0`，此时它仍在哈希表里，同时位于 free queue；
- 请求认领它时会调用 `BlockPool.touch()`；
- `touch()` 会把它从 free queue 移出并将 `ref_cnt` 加一；
- 所以虽然没有创建新物理块，可用空闲块数量仍会减少 1。

因此容量预估必须把这类命中块算进去。

### 4.4 `get_num_blocks_to_allocate()`：容量预估核心

源码位置：第 78—140 行。

这个函数名容易让人误以为返回值一定等于稍后 `get_new_blocks()` 的数量。更准确地说，它返回：

> 为完成本次操作，需要从 free queue 消耗多少个 block 额度。

返回值由两部分组成：

~~~text
真正需要新分配的块
  +
命中后会被 touch、因而从 free queue 移出的可驱逐块
~~~

#### 快速路径：缓存状态已建立的运行请求

~~~python
if request_id in self.num_cached_block:
    assert len(new_computed_blocks) == 0
    return max(num_required_blocks - num_req_blocks, 0)
~~~

运行中的请求不会再次做一次全新的 prefix-cache 认领，因此要求 `new_computed_blocks` 为空。

这里使用 `max(..., 0)` 是为了推测解码：此前可能为 draft token 多分了块，draft 被拒绝后，当前需要的块数反而小于已经持有的块数。

#### 通用/慢速路径

关键公式：

~~~python
num_required_blocks = cdiv(num_tokens, block_size)
num_skipped_blocks = get_num_skipped_tokens(total_computed_tokens) // block_size
num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks

num_new_blocks = max(
    num_required_blocks
    - max(num_skipped_blocks, num_local_computed_blocks),
    0,
)
~~~

为什么是 `max(skipped, local_computed)`，而不是二者相加？

因为二者都描述从逻辑位置 0 开始的一段前缀：

- `skipped`：已经在注意力窗口左侧、不再需要真实存储的前缀；
- `local_computed`：已经有本地计算结果或本地命中块的前缀。

它们可能重叠，真正不需要新分配的范围是两者覆盖到的更远位置，所以取最大值。

随后代码只统计没有被 skip 掉的 `new_computed_blocks` 中，可驱逐的那些：

~~~python
num_skipped_new_computed_blocks = max(
    0, num_skipped_blocks - num_req_blocks
)
num_evictable_blocks = count(
    new_computed_blocks[num_skipped_new_computed_blocks:]
)
~~~

最终：

~~~python
return num_new_blocks + num_evictable_blocks
~~~

#### 一个算例

假设：

- `block_size = 4`；
- 需要覆盖 24 token，即 `num_required_blocks = 6`；
- 前 8 token 已离开窗口，即 `num_skipped_blocks = 2`；
- 请求已有 1 个逻辑位置，新命中 3 个块，因此 `num_local_computed_blocks = 4`；
- 扣除会被 skip 的第一个新命中后，保留下来的 2 个命中块都满足 `ref_cnt == 0`。

则：

~~~text
真正新块 = 6 - max(2, 4) = 2
touch 会消耗的 free queue 额度 = 2
容量检查返回 = 4
~~~

稍后的 `allocate_new_blocks()` 只会新取 2 块；另外 2 个额度是在 `touch()` 阶段消耗的。

### 4.5 `allocate_new_computed_blocks()`：认领命中块和外部块

源码位置：第 142—213 行。

它发生在容量检查通过之后，按以下顺序处理。

#### 第一步：运行请求快速返回

若 `request_id` 已有缓存游标，则断言没有新命中并直接返回。

#### 第二步：按注意力窗口裁掉已经 skip 的命中

~~~python
num_total_computed_tokens = local + external
num_skipped_blocks = get_num_skipped_tokens(total) // block_size
new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
~~~

如果窗口已经移动到 external token 区间，代码还会收缩 `num_external_computed_tokens`，只保留窗口内真正需要在本地落槽的外部状态。

#### 第三步：touch 仍需使用的 prefix-hit blocks

启用缓存时调用：

~~~python
block_pool.touch(new_computed_blocks)
~~~

这会增加引用计数，并把 `ref_cnt == 0` 的命中块从 free queue 中移出，防止它们在本请求使用时被驱逐。

禁用 prefix caching 时，代码断言命中列表必须为空。

#### 第四步：构造请求的逻辑 block table

~~~python
req_blocks.extend([null_block] * num_skipped_blocks)
req_blocks.extend(new_computed_blocks)
num_cached_block[request_id] = len(req_blocks)
~~~

被 skip 的位置用 null 占位；真正命中块接在后面。游标直接移动到列表末尾，避免后续重复给已带 hash 的命中块设置 hash。

#### 第五步：为 external computed tokens 预留本地槽位

外部 KV 虽然已经计算好，但数据要传到本地，仍必须有本地物理块承接：

~~~python
needed = cdiv(total_computed_tokens, block_size) - len(req_blocks)
allocated = block_pool.get_new_blocks(needed)
req_blocks.extend(allocated)
~~~

当前代码只在 `type(kv_cache_spec) is FullAttentionSpec` 时，把这些 ID 加入 `new_block_ids`。这是精确类型判断，`MLAAttentionSpec` 和 `SinkFullAttentionSpec` 不会进入这一分支。

### 4.6 `allocate_new_blocks()`：补齐本轮新 token 的槽位

源码位置：第 215—242 行。

逻辑很直接：

~~~python
num_required_blocks = cdiv(num_tokens, block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
~~~

- `num_new_blocks <= 0`：已有容量足够，返回空列表；
- 否则从 `BlockPool` 取新块、追加到请求表，并只返回本次新增块。

返回“新增块”而不是完整 block table，是因为上层要把本轮新增映射增量传给执行侧。

与上一方法相同，只有 spec 的精确类型是 `FullAttentionSpec` 时才记录 `new_block_ids`。

### 4.7 `take_new_block_ids()`：读取并清空增量

源码位置：第 244—248 行。

这是典型的 drain 操作：

~~~python
ids = self.new_block_ids
self.new_block_ids = []
return ids
~~~

上层 scheduler 汇总这些 ID 放进 `SchedulerOutput.new_block_ids_to_zero`；worker 在使用前清零对应 GPU cache，避免旧数据或 NaN 污染计算。

### 4.8 `cache_blocks()`：只注册完整块

源码位置：第 250—274 行。

~~~python
num_full_blocks = num_tokens // block_size
~~~

这里是向下取整，不是 `cdiv`。原因是 prefix cache 只能安全共享完整块；末尾尚未写满的部分不能注册为完整 cache entry。

若游标还没到 `num_full_blocks`，就调用：

~~~python
block_pool.cache_full_blocks(
    request,
    blocks,
    num_cached_blocks,
    num_full_blocks,
    block_size,
    kv_cache_group_id,
)
~~~

`BlockPool` 会：

1. 取得 Request 预先计算好的 block hashes；
2. 为每个非 null 完整块设置带 group ID 的 hash；
3. 插入全局 `cached_block_hash_to_block` 索引；
4. 跳过 null block；
5. 最后 manager 把游标更新为 `num_full_blocks`。

上层 `KVCacheManager.allocate_slots()` 会把可提交 token 数限制在 `request.num_tokens` 内，因此未确认、可能被拒绝的 draft token 不会提前成为可共享缓存。

### 4.9 `free()`：释放请求引用，不等于立刻抹掉缓存

源码位置：第 276—291 行。

~~~python
req_blocks = req_to_blocks.pop(request_id, [])
block_pool.free_blocks(reversed(req_blocks))
num_cached_block.pop(request_id, None)
~~~

反向释放的目的，是让序列尾部的块先进入驱逐顺序。尾部 hash 表示更长、更具体的前缀，通常比短前缀更适合先被淘汰。

`BlockPool.free_blocks()` 主要做引用计数减一；当普通块降到 0 时，把它放回 free queue。若该块已经缓存，它的 hash 可以继续保留，直到物理块真正被复用时再驱逐。

请求在尚未分配任何块时被 abort 也安全，因为 `pop(..., [])` 会返回空列表。

### 4.10 `remove_skipped_blocks()`：释放窗口外的完整块

源码位置：第 358—399 行。

流程：

1. 由多态方法 `get_num_skipped_tokens(total_computed_tokens)` 算窗口左侧 token 数；
2. 用整除得到可以完整释放的块数；
3. 上限裁到当前 block table 长度；
4. 从 skip 区间最右侧向左遍历；
5. 释放普通块，并把原位置替换为 null；
6. 遇到已有 null 就停止，因为更左侧按不变量也已处理过。

~~~python
for i in range(num_skipped_blocks - 1, -1, -1):
    if blocks[i] == null_block:
        break
    removed_blocks.append(blocks[i])
    blocks[i] = null_block
~~~

只释放完整块非常关键。即使窗口左边界已经进入某个块内部，该边界块仍可能包含窗口内 token，必须继续保留。

### 4.11 三个可覆写接口

#### `find_longest_cache_hit()`

抽象类方法。输入请求的 block hash 链、最大命中长度、group IDs、spec、EAGLE 和对齐信息；输出是：

~~~python
tuple[list[KVCacheBlock], ...]
~~~

tuple 的每一项对应一个 group ID。返回列表可以含 null placeholder，因此列表长度表示逻辑已计算前缀，非 null 数量才表示实际拿到的物理缓存块。

#### `get_num_common_prefix_blocks()`

抽象实例方法。用于 scheduler 计算运行请求间可用于 cascade attention 的公共前缀块数。

#### `get_num_skipped_tokens()`

默认返回 0，表示 Full Attention 一类策略不会在请求结束前主动丢掉历史 token。

#### `new_step_starts()`

默认无操作。scheduler 每个新 step 开始都会经 coordinator 调用它，Mamba 用它清理“本 step 新生成缓存”的临时集合。

## 5. `FullAttentionManager`

源码位置：第 419—477 行。

它只覆写两个抽象接口，其他分配、缓存、释放行为全部复用基类。

### 5.1 适用范围

工厂映射中：

- `FullAttentionSpec -> FullAttentionManager`；
- `MLAAttentionSpec -> FullAttentionManager`。

`MLAAttentionSpec` 继承自 `FullAttentionSpec`，虽然每页的物理布局不同，但从“历史是否能丢、前缀如何连续命中”的管理语义看仍属于 Full Attention。

方法里的断言还允许 `ChunkedLocalAttentionSpec`，说明该命中算法本身可被相关上层路径复用；正常工厂映射仍会为 Chunked Local 创建专用 manager。

### 5.2 `find_longest_cache_hit()`

算法是从左向右查最长连续前缀：

~~~python
for block_hash in first_max_num_blocks:
    cached = block_pool.get_cached_block(block_hash, group_ids)
    if cached:
        append_to_each_group(cached)
    else:
        break
~~~

特征：

- 最大只检查 `max_length // effective_block_size` 个完整块；
- DCP/PCP 开启时，查找使用放大后的有效 block size；
- 任一 group miss，就停止整个共同前缀；
- 链式 block hash 保证了“第 i 块 miss 后，不必继续检查 i+1”。

#### EAGLE 处理

若 `use_eagle` 且至少命中一个块，删除最后一个命中块：

~~~python
for computed in computed_blocks:
    computed.pop()
~~~

原因是 EAGLE drafting head 还需要上一段的 hidden states；完全复用最后一块会缺少生成 draft 所需的中间结果，因此强制重算一块。

#### 混合 block size 对齐

Hybrid coordinator 会把各 group block size 的最小公倍数传为 `alignment_tokens`。若当前命中 token 数不是其整数倍，就继续从尾部 pop，直到所有 group 都能在完整块边界上表达同一个命中长度。

### 5.3 `get_num_common_prefix_blocks()`

代码拿任意一个运行请求的块表，从头检查：

~~~python
if block.ref_cnt == len(req_to_blocks):
    num_common_blocks += 1
else:
    break
~~~

如果某个 block 的引用数等于 manager 当前记录的请求数，说明所有这些请求都引用该物理块。连续满足的开头部分就是公共前缀。

它在第一个不公共的块处停止，因为 cascade attention 需要连续公共前缀。

### 5.4 这个类的核心语义

- 历史上下文全部可能参与未来注意力，`get_num_skipped_tokens() = 0`；
- 请求结束前不会因窗口移动而回收旧块；
- prefix hit 必须是从 block 0 开始的连续哈希链；
- 当前是唯一真正返回公共前缀块数的具体类。

## 6. `SlidingWindowManager`

源码位置：第 480—616 行。

### 6.1 初始化

在基类字段之外保存：

~~~python
self.sliding_window = kv_cache_spec.sliding_window
~~~

### 6.2 为什么它的 cache hit 不要求从 block 0 连续

对下一 token 做滑动窗口注意力时，只需要最近 `sliding_window` 范围内的历史。窗口左侧即使 cache miss，也不影响当前 token。

因此它可以返回：

~~~text
[NULL, NULL, cached_8, cached_9]
~~~

前两个 null 表示逻辑上已有更长前缀，但旧 KV 已经不需要；真正必须连续命中的是窗口覆盖的尾部块。

### 6.3 `find_longest_cache_hit()`

当前实现不支持 DCP/PCP，二者都必须为 1。

先计算窗口所需的连续块数：

~~~python
K = cdiv(sliding_window - 1, block_size)
~~~

减 1 是因为待计算的输入 token 本身也属于窗口，历史 cache 只需覆盖此前的 `window - 1` 个 token。

若开启 EAGLE，先令 `K += 1`，确保最后再 pop 一块后，仍留下足够的窗口状态。

#### 反向扫描

算法从允许命中的最右侧 block 向左扫描：

1. 初始结果是长度为 `max_num_blocks` 的全 null 列表；
2. 命中时把对应位置的 null 换成真实块；
3. 记录当前连续命中长度；
4. miss 时把连续长度清零；
5. 一旦找到至少 K 个连续命中，截掉其右侧无关内容并返回；
6. 若始终找不到完整窗口，则只保留从序列起点开始的那段连续命中。

之所以能容忍中间 miss，是因为只要在某个更靠后的逻辑位置找到完整窗口，miss 以前的状态已经不参与注意力。

#### 对齐要求

开始一段候选连续区间时，会检查该区间右端对应的 token 长度是否满足 `alignment_tokens`。Hybrid 模型必须让所有 group 报告同一个完整块边界。

EAGLE pop 之后还会再次做对齐，因为删掉一块可能破坏原有 LCM 对齐。

### 6.4 `get_num_skipped_tokens()`

公式：

~~~python
max(0, num_computed_tokens - sliding_window + 1)
~~~

例：`sliding_window = 4`、已经计算 7 个 token，下一 token 的逻辑位置是 7。

~~~text
token:       0 1 2 3 4 5 6 | 7
可跳过:     0 1 2 3
当前窗口:           4 5 6 | 7
~~~

所以返回 4。基类随后只释放其中能组成完整 block 的部分。

### 6.5 公共前缀为什么直接返回 0

Sliding Window 的块表开头通常已经变成 null。不能像 Full Attention 那样仅靠真实 block 的 `ref_cnt` 计算公共前缀；当前也没有实现 sliding window 与 cascade attention 的组合，因此返回 0 保证正确性。

## 7. `ChunkedLocalAttentionManager`

源码位置：第 619—766 行。

这种注意力把序列分成固定大小的 chunk。下一 token 只关注它所在 chunk 的局部范围，而不是一个每 token 连续滑动的窗口。

### 7.1 初始化

~~~python
self.attention_chunk_size = kv_cache_spec.attention_chunk_size
~~~

### 7.2 `find_longest_cache_hit()`

当前限制：

- 不支持 EAGLE；
- 不支持 DCP；
- 不支持 PCP；
- `kv_cache_spec.block_size` 必须等于 `alignment_tokens`，即当前不支持与不同 block size 的 group 做这种混合命中。

先求当前 chunk 的开始 token：

~~~python
local_attention_start_idx = (
    max_length // attention_chunk_size * attention_chunk_size
)
~~~

它之前的完整 chunk 已在当前局部窗口之外，可直接用 null 标记为逻辑已计算：

~~~python
local_attention_start_block_idx = (
    local_attention_start_idx // block_size
)
computed = [null_block] * local_attention_start_block_idx
~~~

然后只在当前 chunk 内从左向右查：

~~~python
for i in range(local_attention_start_block_idx, max_num_blocks):
    if all_groups_hit(block_hashes[i]):
        append_real_blocks()
    else:
        break
~~~

当前 chunk 内仍要求连续前缀命中；一旦 miss 就停止。

#### 源码中的典型例子

`chunk_size = 8`、`block_size = 4`、`max_length = 15`：

~~~text
旧 chunk:      token 0..7    -> [NULL, NULL]
当前 chunk:    token 8..15
完整可查块:    token 8..11   -> 命中则追加 block
未完整尾部:    token 12..14  -> 不能作为完整 cache block 命中
~~~

结果可能是 `[NULL, NULL, hit_block]`，也可能只有 `[NULL, NULL]`。

### 7.3 `get_num_skipped_tokens()`

~~~python
(num_computed_tokens // attention_chunk_size) * attention_chunk_size
~~~

也就是跳过当前 token 所在 chunk 左侧的所有完整 chunk。

- 已算 7 token、chunk=8：仍在第一个 chunk，返回 0；
- 已算 8 token：进入下一个 chunk，返回 8；
- 已算 13 token：仍返回 8。

### 7.4 公共前缀

当前不支持 chunked local attention 的 cascade attention，因此固定返回 0。

## 8. `MambaManager`

源码位置：第 769—1039 行。

这是本文件最复杂的类。原因是它管理的不是传统 Attention 的逐 token K/V，而是能概括历史的 recurrent state。一个较晚位置的 Mamba state 已经包含此前序列信息，因此缓存查找和释放规则都与 Full Attention 不同。

### 8.1 三种 cache mode

结合 `CacheConfig` 的定义：

| 模式 | 含义 |
|---|---|
| `none` | prefix caching 关闭时使用 |
| `all` | 缓存每个 `i * block_size` 位置的 Mamba state；支持的模型开启 prefix caching 时优先使用 |
| `align` | 只缓存 scheduler step 的最后 state，以及恰好位于 block 边界的 state |

`all` 用更多 state checkpoint 换取更灵活的 prefix hit；`align` 通过移动少量运行 state block 节省显存，但分配和状态复制更复杂。

### 8.2 初始化新增状态

~~~python
self.cached_blocks_this_step = set()
self.mamba_cache_mode = spec.mamba_cache_mode
self.num_speculative_blocks = spec.num_speculative_blocks
~~~

`align` 模式额外维护：

| 成员 | 作用 |
|---|---|
| `last_state_block_idx[request_id]` | 记录上一轮/更早运行 state 所在的逻辑索引，便于稍后释放 |
| `_allocated_block_reqs` | 标记哪些请求已经走过 Mamba 实际分配；决定是首次分配还是复用上一轮 speculative blocks |

### 8.3 `find_longest_cache_hit()`：从右向左找一个最新 state

当前不支持 DCP/PCP。

Full Attention 需要整条连续 KV 前缀；Mamba 只需一个能够代表此前历史的最新 state。因此它：

1. 从 `max_num_blocks - 1` 向 0 扫描；
2. 找到所有目标 group 都命中的最右 state；
3. 确保该 state 结束位置满足 `alignment_tokens`；
4. 在它前面补 `i` 个 null；
5. 追加真实命中 state；
6. 立即停止。

若命中逻辑索引 5：

~~~text
[NULL, NULL, NULL, NULL, NULL, state_at_5]
~~~

列表长度是 6，因此上层仍能算出“前 6 个逻辑 block 已计算”；实际只需要最后一个 Mamba state。

此方法没有直接使用 `use_eagle` 做 pop。Hybrid 场景的共同命中上界通常由 coordinator 和其他 attention group 的结果共同收缩。

### 8.4 为什么同一 scheduler step 内的新 Mamba cache 不能立刻被另一请求使用

`cache_blocks()` 会把本 step 新注册的非 null block hash 加进：

~~~python
cached_blocks_this_step
~~~

之后，若另一个请求的 `new_computed_blocks[-1].block_hash` 属于这个集合，`get_num_blocks_to_allocate()` 返回：

~~~python
block_pool.num_gpu_blocks + 1
~~~

这个值必然大于总容量，上层会认为本轮无法分配，从而把请求推迟到下一 step。

设计原因是：Mamba 不能在同一个执行 step 中依赖另一个请求尚未真正产出的 recurrent state。虽然调度侧已经注册了 block 元数据，GPU 计算尚未完成。

每个新 step 开始时：

~~~python
cached_blocks_this_step.clear()
~~~

上一 step 已完成的 state 此时就可以安全命中。

### 8.5 `remove_skipped_blocks()`

首先保守修正异步调度中的 speculative token：

~~~python
num_computed_tokens = max(
    0, num_computed_tokens - num_speculative_blocks
)
~~~

它假设上一轮 draft 全部可能被拒绝，避免把真实仍需使用的 state 过早释放。

然后调用基类；由于 Mamba 的 `get_num_skipped_tokens(N) = N - 1`，语义是只需保留最后计算位置对应的 state。

`align` 模式还会检查 `last_state_block_idx`。当这个较旧 state 已早于当前所需位置时：

- 调用 `block_pool.free_blocks()`；
- 把该逻辑位置改为 null。

代码注释描述了三代 state 的关系：

1. 当前新块将接收 state；
2. 上一 step 的块用于复制到当前块；
3. 再早一 step 的块已不需要，可以释放。

### 8.6 `get_num_blocks_to_allocate()`：非 align 模式

若存在 speculative blocks，先扩充 token 需求：

~~~python
num_tokens += block_size * num_speculative_blocks
~~~

然后交给基类计算。等价于为每个 speculative state 预留一个额外逻辑块的容量。

### 8.7 `get_num_blocks_to_allocate()`：align 模式

align 模式忽略单独的 lookahead 槽位，改用：

~~~python
num_tokens = num_tokens_main_model
num_required_blocks = (
    cdiv(num_tokens, block_size) + num_speculative_blocks
)
~~~

代码注释给出的前提是当前 draft model 不含 Mamba 层，所以 lookahead token 不需要在这里破坏 Mamba block 对齐。

初步需求：

~~~python
num_new_blocks = (
    num_required_blocks
    - len(new_computed_blocks)
    - len(req_to_blocks[request_id])
)
~~~

若确实需要增加：

- 已经分配过的旧请求：最多再要 1 个新块，因为上一轮 speculative blocks 可以移动复用；
- 首次分配：要 1 个运行 state block，加 `num_speculative_blocks` 个 speculative blocks。

最后仍要加上命中块中 `ref_cnt == 0` 的数量，因为认领它们会消耗 free queue 额度。

### 8.8 `allocate_new_blocks()`：非 align 模式

与容量预估保持一致：先给 `num_tokens` 加上 speculative block 对应容量，再调用基类实际分配。

### 8.9 `allocate_new_blocks()`：align 模式状态迁移

这是类中最值得慢读的方法。可以按以下阶段理解。

#### 阶段 A：计算目标布局

~~~python
num_required_blocks = (
    cdiv(num_tokens_main_model, block_size)
    + num_speculative_blocks
)
~~~

若当前列表长度已经相等，不做任何事；否则断言目标只能向前增长。

#### 阶段 B：记住旧运行 state

- 旧请求：运行 state 总在旧列表尾部往前 `1 + S` 的位置，即 `prev_len - 1 - S`；
- 首次实际分配但已 prefix hit：命中的最后一个块就是需要复制的 state。

这个索引写入 `last_state_block_idx`，供后续释放。

#### 阶段 C：为已经越过的位置补 null

~~~python
num_skipped_blocks = num_required_blocks - S - 1
~~~

运行 state 之前的旧逻辑位置不再需要真实 state，用 null 补到正确索引，保持 block table 的逻辑长度。

#### 阶段 D：移动复用上一轮 speculative blocks

旧请求会检查上一轮尾部的 S 个 speculative blocks。如果它们原来的位置现在已经属于 skipped 区，就：

1. 把同一个真实 block 追加到列表新尾部；
2. 把旧位置替换为 null。

这不是复制 GPU block，而是移动 block table 中的所有权位置，从而避免每 step 为所有 speculative state 重新申请。

#### 阶段 E：只补真正缺少的物理块

- 旧请求断言最多新取 1 块；
- 首次分配断言最多新取 `S + 1` 块；
- 把请求加入 `_allocated_block_reqs`；
- 返回相对 `prev_block_len` 新增的 block-table 片段。

### 8.10 其他覆写

#### `free()`

align 模式先清理 `_allocated_block_reqs` 和 `last_state_block_idx`，再调用基类释放整个请求 block table。

#### `get_num_skipped_tokens()`

~~~python
return num_computed_tokens - 1
~~~

因为 recurrent state 已概括历史，理论上只需保留最后 state。真正释放时基类仍会按完整 block 边界向下取整。

#### `cache_blocks()`

先调用基类注册完整块，再比较调用前后的缓存游标，只遍历新注册区间：

- null block 跳过；
- 普通块必须已经有 hash；
- hash 加入 `cached_blocks_this_step`，用于阻止同 step 依赖。

#### `new_step_starts()`

清空 `cached_blocks_this_step`。

### 8.11 MambaManager 的核心不变量

- 一个较晚的 state 可以代表整段历史，所以命中只需最右侧一个 state；
- 旧逻辑位置仍由 null 保持索引；
- speculative state 不能按普通 token KV 的方式随意提前释放；
- align 模式中至少要保留“用于复制的上一 state”和“当前目标 state”，再加 speculative blocks；
- 当前 step 新注册的 state 到下一 step 才能作为其他请求的可靠输入。

## 9. `CrossAttentionManager`

源码位置：第 1042—1088 行。

它服务 encoder-decoder 模型，例如 decoder 的 cross-attention 需要读取本请求的 encoder states。

### 9.1 分配方式

`KVCacheCoordinator` 对这个 manager 有特殊处理：

- 容量预估和实际分配使用 `num_encoder_tokens`；
- 是一次按 encoder 输入长度进行的静态分配；
- 普通 decoder token 增长不决定 cross-attention block 数。

它没有覆写 `allocate_new_blocks()`，所以实际物理块分配仍复用基类。

### 9.2 为什么禁用 prefix caching

源码给出三个原因：

1. encoder states 通常是请求特有的，例如不同音频或图像；
2. encoder states 每请求计算一次，不像 decoder 前缀那样增量增长；
3. 不存在可跨不同多模态输入安全复用的公共前缀。

因此：

- `allocate_new_computed_blocks()` 断言命中列表为空；
- `cache_blocks()` 直接抛出 `ValueError`；
- `find_longest_cache_hit()` 校验 spec 后抛出 `NotImplementedError`；
- `get_num_common_prefix_blocks()` 返回 0。

注意方法注释中有“返回 empty blocks”的描述，但当前实际实现是抛异常。正确调用路径必须避免对 Cross Attention 做 prefix-cache 查找或注册。

请求结束时仍继承基类 `free()`，正常释放它专属的 encoder cache blocks。

## 10. `SinkFullAttentionManager`

源码位置：第 1091—1112 行。

它继承 `FullAttentionManager`，所以 prefix hit、公共前缀、普通请求块分配等行为都与 Full Attention 相同。唯一新增逻辑发生在初始化。

### 10.1 sink block 预留

先验证：

~~~python
sink_len is not None
sink_len > 0
sink_len % effective_block_size == 0
~~~

然后：

~~~python
num_sink_block = sink_len // block_size
self.sink_blocks = free_block_queue.popleft_n(num_sink_block)
~~~

这批块直接从 free queue 取走，不放进任何请求的 `req_to_blocks`，也不会随普通请求 `free()` 释放，因此在 manager 生命周期内被永久保留。

### 10.2 为什么直接从 queue 取，而不是 `get_new_blocks()`

这些不是某个请求独占的普通块，而是静态 sink attention 的全局固定区域。当前本地实现的 `static_sink_attention.py` 会把 block ID `1..num_sink_blocks` 固定放在每个请求 block table 的开头；`BlockPool` 初始化时已经把 ID 0 取作 null block，随后这里从有序 free queue 取出的正是紧接着的一段 sink IDs。

这个类的构造逻辑负责确保这些固定 ID 不会被普通请求再次分配。

### 10.3 一个容易忽略的细节

基类记录 `new_block_ids` 时使用：

~~~python
type(self.kv_cache_spec) is FullAttentionSpec
~~~

`SinkFullAttentionSpec` 是子类但不是精确同一类型，所以普通 Sink manager 新块不会进入该记录分支。

## 11. 工厂映射与创建流程

源码位置：第 1115—1131 行。

~~~python
spec_manager_map = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}
~~~

工厂：

~~~python
manager_class = spec_manager_map[type(kv_cache_spec)]
return manager_class(kv_cache_spec, **kwargs)
~~~

这里是 `type(...)` 精确查表，不是 `isinstance(...)`：

- 已登记的 spec 子类可以正确创建；
- 新增一个 `KVCacheSpec` 子类时，必须同步把精确类型加入映射；
- 否则会得到 `KeyError`，不会自动退回某个父 spec 的 manager。

`KVCacheCoordinator.__init__()` 为每个 group 传入同一个 `BlockPool`，以及各自的 `kv_cache_group_id`、spec 和并行参数。

## 12. 一次请求的完整生命周期

结合上层 `KVCacheManager.allocate_slots()`，可以把本文件的方法串成以下流程。

### 12.1 新请求进入

1. Request 已经拥有按 token 内容计算的 `block_hashes`；
2. coordinator 调对应类的 `find_longest_cache_hit()`；
3. 得到每个 group 的逻辑命中块列表和共同命中 token 长度。

### 12.2 调度前释放不再需要的历史

~~~python
remove_skipped_blocks(request_id, total_computed_tokens)
~~~

Full 不释放；Sliding、Chunked、Mamba 根据各自窗口/state 语义释放并填 null。

### 12.3 做容量预检

~~~python
needed = get_num_blocks_to_allocate(...)
if needed > block_pool.get_num_free_blocks():
    本轮不调度
~~~

由于预估同时包含新块和需要 touch 的可驱逐命中块，后续操作可以作为一个整体安全执行。

### 12.4 认领已计算前缀

~~~python
allocate_new_computed_blocks(...)
~~~

它 touch 本地命中，构造 null/hit 前缀，并为外部 KV 预留槽位。

### 12.5 为新 token 和 lookahead 补槽

~~~python
allocate_new_blocks(...)
~~~

普通 manager 按 `cdiv(total_tokens, block_size)` 补齐；Mamba 根据 mode 做额外 state 布局。

### 12.6 注册已经完整且可提交的块

~~~python
cache_blocks(request, num_tokens_to_cache)
~~~

只处理完整 block。Mamba 还记录本 step 新缓存的 hash。

### 12.7 本 step 执行

worker 收到 block IDs，必要时先清零新块，再由 attention/Mamba kernel 写入或读取真实 cache。

### 12.8 step 切换与请求结束

- 新 step：`new_step_starts()`；普通类无操作，Mamba 清临时集合；
- 请求结束、abort 或 preempt：`free(request_id)`，释放活跃引用并清除请求级记账。

## 13. 各类 cache-hit 算法对比

| manager | 扫描方向 | 命中条件 | null 的意义 | EAGLE |
|---|---|---|---|---|
| Full | 左 → 右 | 从 block 0 开始连续命中 | 通常不用 | 命中后删最后一块并重对齐 |
| Sliding Window | 右 → 左 | 找到覆盖当前窗口的 K 个连续块；早期阶段可退化为起点前缀 | 窗口左侧已处理但无需真实 KV | 先多要求一块，再删最后一块并重对齐 |
| Chunked Local | 当前 chunk 内左 → 右 | 当前 chunk 起点开始连续命中 | 以前的完整 chunk 已不参与当前局部注意力 | 明确不支持 |
| Mamba | 右 → 左 | 最新的一个对齐 recurrent state 命中 | 更早 state 已被最后 state 概括 | 方法内没有直接 pop |
| Cross Attention | 不查 | 不支持跨请求 prefix cache | 不适用 | 不适用 |
| Sink Full | 与 Full 相同 | 与 Full 相同 | 与 Full 相同 | 继承 Full |

## 14. 方法覆写矩阵

“继承”表示直接使用基类实现。

| 类 | hit 查找 | skip 计算/回收 | 分配预估/实际分配 | cache/free | 公共前缀/step hook |
|---|---|---|---|---|---|
| Full | 覆写 | 继承，永不 skip | 继承 | 继承 | 覆写公共前缀 |
| Sliding | 覆写 | 覆写 skip，继承回收 | 继承 | 继承 | 公共前缀返回 0 |
| Chunked Local | 覆写 | 覆写 skip，继承回收 | 继承 | 继承 | 公共前缀返回 0 |
| Mamba | 覆写 | skip 与回收都覆写 | 两者都覆写 | cache、free 都覆写 | 公共前缀 0；覆写 step hook |
| Cross | 禁止调用 | 继承默认 skip=0 | 继承实际分配；禁止认领 hit | 禁止 cache；继承 free | 公共前缀 0 |
| Sink Full | 继承 Full | 继承 | 继承，初始化额外预留 | 继承 | 继承 Full |

## 15. 关键设计不变量

阅读或修改代码时，应持续检查这些不变量。

1. `req_to_blocks[id][i]` 始终代表逻辑 block 位置 `i`；释放旧块必须填 null，不能删除中间元素。
2. 普通非 null block 被活跃请求引用时，`ref_cnt > 0`。
3. `ref_cnt == 0` 的缓存块仍可能在哈希表中，同时位于 free queue，属于可驱逐缓存。
4. prefix hit 在加入请求前必须 `touch()`，防止调度后被其他分配驱逐。
5. 只有完整块才能通过 `cache_full_blocks()` 注册 prefix cache。
6. `num_cached_block` 是逻辑位置游标；命中的带 hash 块和 null 位置都不能被重复缓存。
7. 多 group 查找必须返回同一个 token 长度，Hybrid coordinator 用 `alignment_tokens` 和迭代收缩保证这一点。
8. 容量预估必须在改变引用计数或真正取新块之前完成。
9. 释放请求块时使用反向顺序，以维持期望的缓存驱逐优先级。
10. Mamba 本 step 刚注册的 state 不能被同 step 的另一个请求依赖。

## 16. 容易误读的地方

### 16.1 “free block” 不一定是空内容

free queue 中既有从未使用或已驱逐的块，也有 `ref_cnt == 0`、但 hash 仍在 prefix cache 中的可驱逐块。

### 16.2 返回列表里的 null 也计算命中长度

Sliding、Chunked 和 Mamba 返回的 null 表示“这段逻辑前缀不必再计算或不必保留真实状态”。上层使用列表长度计算 logical hit length，这是刻意设计，不是伪造物理 cache hit。

### 16.3 `get_num_blocks_to_allocate()` 不是单纯的新块数

它还包含 touch 可驱逐命中块时损失的 free queue 额度。实际 `get_new_blocks()` 数量可以更小。

### 16.4 `num_cached_block` 不是简单的真实缓存块计数

它是位置游标，可能跨过 null；并且某个刚开始运行但尚无完整块的请求可能暂时没有字典键。

### 16.5 `num_req_blocks` 中的 “req” 是 request

它是“请求当前 block table 的长度”，不是 “required blocks” 的缩写。

### 16.6 Cross Attention 的实现是抛异常

注释说“返回 empty blocks”，但当前 `find_longest_cache_hit()` 实际抛 `NotImplementedError`；`cache_blocks()` 也抛 `ValueError`。上层必须保证禁用这条 prefix-cache 路径。

### 16.7 工厂和部分分支使用精确类型

`spec_manager_map[type(spec)]` 以及 `type(spec) is FullAttentionSpec` 都不是多态的 `isinstance` 语义。新增 spec 子类时必须逐处检查。

### 16.8 当前支持限制

| 类型 | 限制 |
|---|---|
| Sliding Window | cache-hit 查找不支持 DCP/PCP |
| Chunked Local | 不支持 EAGLE、DCP、PCP，也不支持不同 block size 的 hybrid 对齐 |
| Mamba | cache-hit 查找不支持 DCP/PCP |
| Sliding / Chunked / Mamba / Cross | 当前公共前缀块数固定为 0 |

## 17. 推荐的源码阅读顺序

如果目标是学习 vLLM v1 的整体 KV Cache 结构，建议按这个顺序：

1. `kv_cache_utils.py`：先看 `KVCacheBlock`、`FreeKVCacheBlockQueue` 和 block hash；
2. `block_pool.py`：看 `get_new_blocks`、`touch`、`free_blocks`、`cache_full_blocks`；
3. 本文件的基类：重点看容量预估、认领命中、cache cursor 和 null 替换；
4. `FullAttentionManager`：理解最标准的连续前缀命中；
5. `SlidingWindowManager`：理解“逻辑前缀已处理，但不再保存真实 KV”；
6. `ChunkedLocalAttentionManager`：对比 sliding window 与固定 chunk 的差别；
7. `MambaManager`：最后读，因为它把 recurrent state、推测解码和 step 依赖都叠加在一起；
8. `kv_cache_coordinator.py`：看单一/混合 group 如何组合这些策略；
9. `kv_cache_manager.py` 的 `get_computed_blocks()` 与 `allocate_slots()`：把整条请求生命周期串起来；
10. `scheduler.py`：看这些接口在每个 scheduling step 的真实调用时机。

对于当前目录中的 Qwen2.5-0.5B，如果模型全部是 Full Attention，实际最常走的是：

~~~text
FullAttentionSpec
  -> FullAttentionManager.find_longest_cache_hit()
  -> 基类 get_num_blocks_to_allocate()
  -> 基类 allocate_new_computed_blocks()
  -> 基类 allocate_new_blocks()
  -> 基类 cache_blocks()
  -> 基类 free()
~~~

先把这条主线弄清楚，再把 Sliding、Chunked 和 Mamba 看成对“哪些历史仍需要保存”这个策略点的不同覆写，会比从头逐行硬读更容易建立整体结构。
