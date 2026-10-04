# single_type_kv_cache_manager.py 详细解析

> **相关文档**
>
> *按函数（专项详解）*
> - [allocate_new_computed_blocks 详解](./allocate_new_computed_blocks_详解.md) —— 挂已算好的块（local 命中 / external 落地）
> - [allocate_new_blocks / take_new_block_ids / cache_blocks 详解](./allocate_new_blocks与cache_blocks_详解.md) —— 分配空块 → 登记待清零 → 注册进前缀缓存
> - [find_longest_cache_hit 详解](./find_longest_cache_hit_详解.md) —— 查前缀命中（各子类算法、eagle、alignment）
> - [free / remove_skipped_blocks 详解](./free与remove_skipped_blocks_详解.md) —— 块释放机制（倒序 = 驱逐优先级）
> - [get_num_common_prefix_blocks 详解](./get_num_common_prefix_blocks_详解.md) —— cascade attention
>
> *按子类*
> - [SlidingWindowManager 详解](./SlidingWindowManager_详解.md) —— 滑窗管理器的全部 4 个方法
>
> *跨文件 / 主题*
> - [KVCacheSpec 与 KV Cache 物理布局 详解](./KVCacheSpec与物理布局_详解.md) —— page_size_bytes / merge / group 划分 / block_id→slot 寻址 / 每层张量
> - [vLLM KV Cache 已知局限与困境](./vLLM_KVCache_已知局限与困境.md) —— DCP/PCP、cascade、prefix caching 等未实现项与取舍

## 一、文件定位

在 vLLM v1 的 KV cache 调用链里属于**中间层**：

```
KVCacheManager           ← 对外接口，scheduler 调它
        ↓
KVCacheCoordinator       ← 管理多个 group（混合注意力模型）
        ↓
★ SingleTypeKVCacheManager ← 每个 "attention 类型" 一个实例（本文件）
        ↓
BlockPool                ← 物理 block 仓库
        ↓
KVCacheBlock（kv_cache_utils.py）
```

**核心定位**：模型里如果只有 full attention，就只用 1 个 FullAttentionManager；像 Gemma2/Mistral 这种混合了 sliding window、Sink、Mamba、Cross attention 的模型，每种 attention 类型的 KV cache 释放/复用策略不一样，就需要为每种类型起一个 manager。本文件给出"每种 attention 类型 → 一个 KV cache 管理策略类"的完整集合。

### 关键概念：spec / layer / group 的关系

`kv_cache_spec` 描述的是"一组规格相同的层"的统一 KV cache 格式，**一个 spec 对应一种 attention 类型（一个 KV cache group），不是单个 layer**。

不是"一层有多种 attention"，而是"**很多层共用一个 spec**"。看 `KVCacheGroupSpec`（kv_cache_interface.py L522）：

```python
@dataclass
class KVCacheGroupSpec:
    """一组共享同一张 KV cache block table 的模型层，
    在 KV cache manager 眼里被当成一层。"""
    layer_names: list[str]      # 这一组包含哪些层（复数 list）
    kv_cache_spec: KVCacheSpec  # 这一组共用的那一个 spec
```

三个概念的层级关系：

```
模型 layer（物理层）         几十层，每层一种 attention 计算
        ↓ 按"规格是否相同"分组
KV cache group（逻辑组）     把规格相同的层归到一组，共用一张 block table
        ↓
KVCacheSpec（规格）          一个 group 一个 spec
        ↓
SingleTypeKVCacheManager    一个 group 一个 manager
```

举例：
- **Qwen2.5-0.5B（纯 full attention，24 层）**：24 层全是 FullAttentionSpec、规格一样 → 1 个 group → 1 个 FullAttentionManager
- **Gemma2（交替 full / sliding window，26 层）**：偶数层 full → group A，奇数层 sliding window → group B → 2 个 group → 2 个 manager（因为两种层的 KV 释放策略不同，滑窗层可以丢窗口外 block，full 层不能丢）

判据：什么样的层能归进同一个 group？类型相同 + 关键参数相同（`merge()` / `is_uniform_type()`）。比如都是 SlidingWindowSpec **且窗口大小相同**才能合并。

把"规格相同的 N 层"折叠成"1 个逻辑层"后，manager 只需维护一张 block table：一个请求在 group 里分配了第 5、6、7 号 block，意味着这一组里**所有物理层**的第 5、6、7 号 block 都一起分配/释放。

---

## 二、KVCacheSpec 是什么

`kv_cache_spec` 是一个 `KVCacheSpec` 对象，描述"某一类 attention 层的 KV cache 长什么样、占多少显存"。定义在 `kv_cache_interface.py`。它是**静态的格式描述，不含任何实际 KV 张量**，也不知道哪个 block 给了哪个请求。

继承体系（和各 manager 一一对应）：

```
KVCacheSpec（L69）  ← 最基础，只有 block_size
├── AttentionSpec（L114）        加上 num_kv_heads / head_size / dtype / 量化模式
│   ├── FullAttentionSpec（L148）        full attention → FullAttentionManager
│   │   ├── MLAAttentionSpec（L249）     MLA(DeepSeek)→ 也走 FullAttentionManager
│   │   └── SinkFullAttentionSpec（L383) StreamingLLM → SinkFullAttentionManager
│   ├── ChunkedLocalAttentionSpec（L288) Llama4 局部 → ChunkedLocalAttentionManager
│   ├── SlidingWindowSpec（L307）        滑窗 → SlidingWindowManager
│   ├── EncoderOnlyAttentionSpec（L363）
│   └── CrossAttentionSpec（L370）       enc-dec 的 cross attn → CrossAttentionManager
└── MambaSpec（L333）            Mamba state → MambaManager
```

manager 主要从它取 `block_size`，所有"token 数 ↔ block 数"换算都靠它。子类还会取更具体字段（如 `sliding_window`、`sink_len`）。

它的另一用途是**启动时算显存**：`max_memory_usage_bytes()` 被用来推算"这块 GPU 能放多少 block"，即 BlockPool 的 num_gpu_blocks 来源。

---

## 三、内部结构

### 3.1 抽象基类 SingleTypeKVCacheManager（L28）

所有具体策略的父类，定义统一接口。

#### __init__ 的 6 个参数

```python
def __init__(
    self,
    kv_cache_spec: KVCacheSpec,    # 这组层的规格说明书（block 多大、什么 attention）
    block_pool: BlockPool,         # 底层物理 block 仓库，manager 把活儿派给它
    enable_caching: bool,          # 是否启用 prefix caching（前缀复用）
    kv_cache_group_id: int,        # 我是第几组，用于 hash 隔离不同 group
    dcp_world_size: int = 1,       # 解码时序列切几张卡（默认不切）
    pcp_world_size: int = 1,       # 预填充时序列切几张卡（默认不切）
) -> None:
```

| 参数 | 角色 | 说明 |
|---|---|---|
| `kv_cache_spec` | 配置 | 这组层的格式 |
| `block_pool` | 依赖 | 真正管显存的仓库；manager 不持有显存，只记账，申请/释放都委托它 |
| `enable_caching` | 开关 | 要不要做前缀复用 |
| `kv_cache_group_id` | 身份 | 第几组；做 prefix cache 时 block hash 要拼 group_id 组成 BlockHashWithGroupId，保证不同 group 即使内容相同也不会错误复用彼此的 block |
| `dcp_world_size` | 并行度 | Decode Context Parallelism（解码上下文并行）|
| `pcp_world_size` | 并行度 | Prefill Context Parallelism（预填充上下文并行）|

`__init__` 里还顺手取了 `self._null_block = block_pool.null_block`（block 0，用于给 sliding window 跳过的位置占位）。

#### 关于 dcp/pcp 与"逻辑大 block"

```python
self.block_size = kv_cache_spec.block_size
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

**Context Parallelism（上下文并行）**：把同一条序列的 token 沿序列维度分散到 N 张卡上，每张卡只存一部分 token 的 KV。

- **物理 block**：真实显存，大小永远是 spec 里的 block_size，每张卡各自持有
- **逻辑 block**：manager 的记账单位，= N 个物理 block 捆在一起，装 `block_size × N` 个 token

例子（block_size=16，dcp=2，序列 64 token）：

```
GPU 0：block0(tok 0~15)  block1(tok 32~47)
GPU 1：block0(tok 16~31) block1(tok 48~63)

逻辑 block 0 = {GPU0.block0 + GPU1.block0} → 覆盖 tok 0~31，共 16×2=32 个 token
逻辑 block 1 = {GPU0.block1 + GPU1.block1} → 覆盖 tok 32~63，共 32 个 token
```

manager 把 block_size 乘以并行度后，用同一套代码 `cdiv(num_tokens, self.block_size)` 就能算对"整条序列要几个逻辑 block"，无需感知切卡。
- 单卡：cdiv(64,16)=4 个 block
- 2 卡 CP：cdiv(64,32)=2 个逻辑 block

**为什么 dcp 和 pcp 要分开**：prefill 和 decode 计算特征相反——
- **Prefill**：query 多、算力瓶颈（compute-bound）。PCP 把长 prompt 沿序列切到 N 卡分摊**计算**（类似 Ring Attention，卡间反复传 KV）
- **Decode**：query 只有 1 个、显存带宽瓶颈（memory-bound）。DCP 把 KV cache 沿序列切到 N 卡，每卡只存/读 1/N 的 KV，再归约局部结果（类似跨卡 flash-decoding）

两者瓶颈、通信模式、最优并行度都不同，且常物理分离（P/D 分离部署），所以独立配置。代码里乘的是两者乘积，因为从 KV 记账角度只关心"序列被沿 token 维切成几份"。

> 以上保留原文的概括和示意。关于 TP 与 CP 的关系、DCP/PCP 的执行过程、online softmax、rank、page/block/slot，以及默认 token 级交错与上述连续块示例的适用条件，详细说明参考 [DCP、PCP 与 KV Cache 分页机制详解](./DCP_PCP与KV_Cache分页机制详解.md)。

#### 关键状态（三张表）

```python
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
self.num_cached_block: dict[str, int] = {}
self._null_block = block_pool.null_block
```

**1. `req_to_blocks`：块表（block table）的内存表示**
- key = request_id，value = 该请求按 token 顺序占用的 block 列表
- `defaultdict(list)`：访问不存在的 key 自动建空列表，新请求也能直接 extend
- value 是**有序列表**，下标 i 的 block 装第 `i*block_size ~ (i+1)*block_size-1` 个 token
- 逻辑上连续的 token，物理上可散落在显存任意位置（block id 可乱序）——这是 paged attention 的精髓
- 管"占了哪些 block"

```
req_to_blocks["abc"] = [ Block(5), Block(2), Block(9), Block(7) ]
                          ↑          ↑          ↑          ↑
                       tok 0~15    tok 16~31  tok 32~47  tok 48~63
```

**2. `num_cached_block`：已缓存 block 数游标 + running 标志**
- key = request_id，value = 该请求 block 列表里前多少个已被缓存（写满 + 算好 hash + 登记进 prefix cache 哈希表）
- 因为 token 顺序填充，"已缓存"永远是列表的一个**前缀**，用一个数字表示边界
- 核心用途：避免重复缓存，`cache_blocks` 用它当游标，只缓存新增区间
- 注释强调：**只追踪 RUNNING 请求，被抢占的不追踪**
- "key 在不在里面"被当作"是否已是运行态请求"的判据

**3. `_null_block`**：全局唯一的空白占位 block（block 0）

与 `req_to_blocks` 关系：
```
req_to_blocks["abc"]  = [ B5, B2, B9, B7, B3 ]   ← 占了 5 个 block
num_cached_block["abc"] = 3                       ← 其中前 3 个已缓存
                          [ B5, B2, B9 | B7, B3 ]
                            已缓存(可复用)  还在写/没满
```

#### block_pool 与 req_to_blocks 的分工
- **block_pool**：全局、所有请求/所有 group 共享的物理仓库，管"哪些 block 空闲、缓存、ref_cnt"（仓库总账）
- **req_to_blocks**：每个 manager 私有，只记"我管的请求各占了哪些"（明细账）

---

### 3.2 基类主要方法

| 方法 | 行号 | 作用 |
|---|---|---|
| `_get_num_evictable_blocks` | L75 | 数一组 block 里有几个"可驱逐"的 |
| `get_num_blocks_to_allocate` | L78 | 算分配请求要从空闲池抽走几个 block |
| `allocate_new_computed_blocks` | L142 | 把 prefix cache 命中的 block 认领过来（**[→ 专项详解](./allocate_new_computed_blocks_详解.md)**：逐行拆解 + local/external 来源与顺序、skip、min 削减、cdiv 相减、new_block_ids 清零等细节辨析）|
| `allocate_new_blocks` | L215 | 从 block_pool 申请新的空白 block（**[→ 专项详解 1 章](./allocate_new_blocks与cache_blocks_详解.md)**：逐行拆解 + `num_tokens_main_model`(投机解码)、为何重算不复用 ② 的数、`touch` vs `get_new_blocks`）|
| `take_new_block_ids` | L244 | 取走自上次以来新分配的 block id（**[→ 专项详解 2 章](./allocate_new_blocks与cache_blocks_详解.md)**：清零链路、驱逐只删索引不擦内存、为何传 id、drain 语义）|
| `cache_blocks` | L250 | 把写满的 block 注册到 prefix cache（**[→ 专项详解 3 章](./allocate_new_blocks与cache_blocks_详解.md)**：cache=建索引不搬数据、两个游标、forward 前就登记、`delay_cache_blocks`、"连续"指下标）|
| `free` | L276 | 请求结束时逆序释放所有 block（**[→ 专项详解 1 章](./free与remove_skipped_blocks_详解.md)**：`reversed` 的驱逐优先级、僵尸态与复活、游标 pop 退回慢路径）|
| `get_num_common_prefix_blocks` | L294 | (abstract) 所有 running 请求共享的前缀长度（**[→ 专项详解](./get_num_common_prefix_blocks_详解.md)**：cascade attention、`ref_cnt == len(req_to_blocks)` 妙用、未调度请求污染的保守缺陷）|
| `find_longest_cache_hit` | L311 | (abstract) 给定 hash 链找最长命中（**[→ 专项详解](./find_longest_cache_hit_详解.md)**：per-group 返回结构、islice/海象/zip、全或无、eagle pop、`alignment_tokens`/LCM、各子类算法对比）|
| `remove_skipped_blocks` | L358 | 把滑出窗口的 block 换成 null 并释放（**[→ 专项详解 2 章](./free与remove_skipped_blocks_详解.md)**：`min` cap 为何必要、倒序循环 + break 的两个好处）|
| `get_num_skipped_tokens` | L401 | 默认返回 0（full attention 不丢）（**[→ 专项详解 3 章](./free与remove_skipped_blocks_详解.md)**：为何以"续算起点"为参考）|
| `new_step_starts` | L414 | 默认空操作 |

#### _get_num_evictable_blocks（L75）

```python
@classmethod
def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
    return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)
```

数一组 block 里有几个"可驱逐"（evictable）的。条件**同时满足**：
- `ref_cnt == 0`：当前没有活跃请求在用它（躺在 free_block_queue 里当驱逐候选，但内容还在、还挂在哈希表里）
- `not blk.is_null`：不是空白占位 block

`sum(布尔 for ...)` 利用 True==1、False==0 直接数个数。

"evictable" 状态：
```
ref_cnt > 0  ：有请求在用 → 不可动
ref_cnt == 0 且内容还在：在 free queue 里当驱逐候选 → evictable
              ├─ 被 touch 走 → 复用其缓存内容
              └─ 被 get_new_blocks 取走 → 驱逐、清空、重分配
```

> 关于 @classmethod：Python 默认实例方法调用时自动塞实例作第一个参数，与函数体用没用 self 无关。要让它"别塞实例、改绑类、能用类直接调"，必须显式写装饰器——这是行为开关不是注释。此处 cls 其实闲置，换成 @staticmethod 功能等价、更准确；作者选 @classmethod 是为了和同类的 find_longest_cache_hit（真正用 cls 做多态分发）风格一致。

#### get_num_blocks_to_allocate（L78）—— ★ 核心

**不真正分配**，只回答："为了装下 num_tokens 个 token，要从空闲池（free_block_queue）额外抽走多少个 block？" scheduler 拿返回值和"当前空闲 block 数"比较，判断够不够、要不要抢占。

返回值语义 = **对空闲池的消耗量**（不是"还差几个 block"）。

参数：
```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,                          # 哪个请求
    num_tokens: int,                          # 总共要给多少 token 留槽位（含已分配）
    new_computed_blocks: Sequence[KVCacheBlock],  # 刚命中 prefix cache 的 block
    total_computed_tokens: int,               # 已算 token 数（本地+远端）
    num_tokens_main_model: int,               # 投机解码用，基类没用到（给子类覆盖）
) -> int:
```

开头两个量：
```python
num_required_blocks = cdiv(num_tokens, self.block_size)        # 目标：总共需要几个 block
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))   # 现状：已持有几个 block
```

**快路径**（running 请求，L108）：
```python
if request_id in self.num_cached_block:
    assert len(new_computed_blocks) == 0          # running 请求不会再有新命中
    return max(num_required_blocks - num_req_blocks, 0)
```

为什么用 `request_id in self.num_cached_block` 做判据：这张表语义就是"running 请求追踪器"。
- 不在表里 = 新请求（或抢占后恢复）→ 走慢路径查命中
- 在表里 = 已运行 → prefix 查找只在接纳时做一次，之后只往后生成 token，不会再有新命中（所以 assert new_computed_blocks 为空）

为什么 `num_required_blocks - num_req_blocks` 会非 0：`num_required_blocks` 从 `num_tokens` 算，而序列每步在变长。大多数步差值为 0（当前 block 没写满），每当序列长度**跨过 block 边界**（16/32/48… 的整数倍）就要补 1 个新块。`max(...,0)` 兜底投机解码草稿 token 被拒、已持有反而多于需要的情况。

> 用 num_cached_block 而非永久 bool 的好处：free 时会 pop 掉它，所以被抢占的请求恢复时自动退回慢路径、重查缓存命中（它之前算的 KV 可能还在缓存里）。

**慢路径**（新请求，L116+）：
```python
num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
num_skipped_blocks = num_skipped_tokens // self.block_size
num_new_blocks = max(
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0
)
num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)
num_evictable_blocks = self._get_num_evictable_blocks(
    new_computed_blocks[num_skipped_new_computed_blocks:]
)
return num_new_blocks + num_evictable_blocks
```

**返回 num_new_blocks + num_evictable_blocks 的原因**：空闲池会被"抽走"block 的来源有两条——
1. **num_new_blocks**：全新块，通过 `get_new_blocks()` 从 free_block_queue **pop** 出，每个消耗 1 个
2. **num_evictable_blocks**：prefix 命中块里**此刻躺在 free queue（ref_cnt==0）**的那些。认领它们时调 `touch`，会把它们**从 free queue pop 出**（ref_cnt 0→1），同样消耗空闲池 1 个

只算 evictable 的、不算全部命中块，因为命中块分两种：
| 命中块 | 在空闲队列 | touch 时 | 消耗空闲池 |
|---|---|---|---|
| ref_cnt>0（别人占着）| 否 | 仅 ref_cnt+1 | 不消耗（共享即可）|
| ref_cnt==0（evictable）| 是 | 从队列 pop | 消耗 1 个 |

切片 `new_computed_blocks[num_skipped_new_computed_blocks:]` 先剔除"滑窗跳过、会被 null 替换"的命中块，只对真正要 touch 的命中块数 evictable。

若漏掉 num_evictable_blocks：scheduler 会低估消耗，实际分配时透支空闲池导致崩溃/状态不一致。

#### num_new_blocks 里那个 max 为什么这么写

```python
num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
```

关键前提：**三者都是"从 block 0 开始的连续前缀"**，只是长度不同：
- `num_req_blocks`（已持有）：[0, num_req) 连续前缀
- `new_computed_blocks`（命中）：[0, k] 连续前缀（命中链式 hash，遇 miss 即停）
- `num_skipped_blocks`（跳过）：[0, skip) 连续前缀

skipped 前缀（null 占位，不占显存）和 computed 前缀（已有内容，不用重算）**都不需要新分配真实显存**，但它俩都从 block 0 起、互相重叠，不能相加（会重复扣）。取 `max`——谁延伸得更远，[max, num_required) 才是真正要申请新块的部分。

#### get_num_skipped_tokens 与 skip 机制（L401）

**"skip"指已经滑出注意力窗口、未来再也用不到的 token——它们的 KV block 可提前释放。**

- **full attention**（Qwen）：每个新 token attend 前面所有 token，KV 一旦算出直到请求结束都要留 → 基类返回 0，一个都不跳
- **sliding window**（Mistral）：每个新 token 只 attend 最近 sliding_window 个，更早的不看 → 可释放

SlidingWindowManager 实现（L607）：
```python
return max(0, num_computed_tokens - self.sliding_window + 1)
```

例子（window=4, num_computed_tokens=7，即 0~6 已算、下一个算索引 7）：
```
token 索引:    0    1    2    3    4    5    6    7
             [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ]  7 ←下一个要算的
               └─── skip: 0~3 ───┘  └─ 窗口: 4~7 ──┘
```
索引 7 的窗口 = [4,7]，前 4 个（0~3）滑出窗口、永远用不到 → 可释放。`get_num_skipped_tokens(7)=4`。

**为什么用 total_computed_tokens 而不是整体 num_tokens**：这是正确性问题。skip 边界由"本步要算的、最靠前的那个 token"的窗口左边界决定，那个位置正好是 total_computed_tokens。

chunked prefill 反例（window=4，已算 7，本步要算 7~10）：
- 用 total=7 算：skip=7-4+1=4，丢 0~3 ✓（位置 7 窗口[4,7]还要 4~6，必须留）
- 用整体 num_tokens=11 算：skip=11-4+1=8，丢 0~7 ✗（位置 7、8、9 还要 attend 4~7，KV 被提前删，出错）

所以必须用 total_computed_tokens（本地+远端，因远端传来的 KV 也是前缀的一部分，同样决定窗口位置）。

返回值是个数量（int）即可定位，因为 skip 永远是序列最前面的连续前缀，"跳过 N 个 token" = "跳过开头 N 个" = "跳过前 N//block_size 个 block"。

skip 数被用在两处：
1. **分配时**（get_num_blocks_to_allocate）：跳过的整 block 用 _null_block 占位，不需真实显存，从"要新申请"里扣掉
2. **运行时**（remove_skipped_blocks L358）：真的把滑出窗口的 block 在块表换成 _null_block 并 free_blocks 还给 block_pool

对 full attention，num_skipped_tokens 恒 0，整套 skip 机制不触发。

#### 慢路径的两个边界情况辨析

前提：sliding window 的 `new_computed_blocks` **前面是 null 占位**（find_longest_cache_hit L522 先全填 null，再往命中位置填真块）：
```
new_computed_blocks = [ null,null,null,null,null | B8,B3,B9 ]
                        └─ 跳过区（null 占位）──┘  └ 窗口内真命中 ┘
```
`num_new_blocks`（全新块，get_new_blocks pop）和 `num_evictable_blocks`（命中块，touch pop）数的是**物理不相交**的两类块，位置上也首尾相接（命中在前缀、全新在后缀），所以**永不重复计算**。

**Q：slice `new_computed_blocks[num_skipped_new_computed_blocks:]` 会越界报错吗？**
不会。Python 切片对越界起点返回空列表（`[1,2,3][5:] == []`），最坏 `num_evictable_blocks = 0`，不崩溃。

**Q：skip 占主导时会重复算 evictable 吗？**
不会，evictable 会**自动归零**。skip 真正超过本地命中，通常因 external（远端）computed tokens——窗口用 `total = local + external` 算，被推到比本地命中更靠后。此时 `num_skipped_new_computed_blocks ≥ len(new_computed_blocks)`，切片恰好变空 → evictable=0，正是"只需 num_new_blocks"的精确表达。

例 A（skip 占主导，external 推动）：
```
num_required_blocks=10, num_skipped_blocks=7（total 含大量 external）
len(new_computed_blocks)=3（本地命中少，都在跳过区内）
num_local_computed_blocks = 3+0 = 3
num_new_blocks = max(10 - max(7,3), 0) = 3
num_skipped_new_computed_blocks = max(0, 7-0) = 7
new_computed_blocks[7:] = []   → 越界安全返回空 → num_evictable_blocks = 0
返回 = 3 + 0 = 3               （只剩全新块）
```

例 B（computed 占主导，full attention / 窗口内命中）：
```
num_required_blocks=10, num_skipped_blocks=0
len(new_computed_blocks)=4，其中 2 个 evictable(ref_cnt=0)
num_local_computed_blocks = 4
num_new_blocks = max(10 - max(0,4), 0) = 6
num_skipped_new_computed_blocks = max(0, 0-0) = 0
new_computed_blocks[0:] = 全部4个 → num_evictable_blocks = 2
返回 = 6 + 2 = 8              （命中在前缀、全新在后缀，不重叠）
```

`max(0, ...)` + 切片越界返回空，合起来就是让"external 把窗口推过本地命中""滑窗跳过盖过命中"这些边缘情况都能正确退化。

---

### 3.3 各 attention 类型的具体子类

| 类（行号）| 对应 spec | 关键差异 |
|---|---|---|
| `FullAttentionManager`（L419）| FullAttentionSpec / MLAAttentionSpec | 普通 full attention，永不丢 block；find_longest_cache_hit 顺 hash 链一直匹配 |
| `SlidingWindowManager`（L480）| SlidingWindowSpec | 窗口外可丢；get_num_skipped_tokens 返回窗口外 token 数；find_longest_cache_hit 反向扫描允许中间 miss（**[→ 专项详解](./SlidingWindowManager_详解.md)**：全部 4 个方法、eagle +1 推演、两个出口、返回列表=块表语义、`sliding_window` 三条下游线）|
| `ChunkedLocalAttentionManager`（L619）| ChunkedLocalAttentionSpec | 按 chunk 切（Llama4），只在 chunk 边界考虑命中 |
| `MambaManager`（L769）| MambaSpec | Mamba state cache（非 KV），只占 1~2 固定 block，几乎重写所有方法 |
| `CrossAttentionManager`（L1042）| CrossAttentionSpec | enc-dec 的 encoder KV，不支持 prefix caching |
| `SinkFullAttentionManager`（L1091）| SinkFullAttentionSpec | 继承 full attention，额外预留 sink block 永不释放（StreamingLLM）|

#### find_longest_cache_hit 与"命中为连续前缀"

`FullAttentionManager.find_longest_cache_hit`（L446）：
```python
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(block_hash, ...):
        computed.append(cached)
    else:
        break        # 一旦某 block 没命中，立刻停止
```

从第 0 个 block 顺着往后查，碰到第一个 miss 就 break。**prefix cache 命中只能是从头开始的连续前缀**，因为 block hash 是**链式**的。

#### block hash 的链式特性（重要纠正）

`hash_block_tokens`（kv_cache_utils.py L535）：
```python
return BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
```

hash 输入是三元组：
| 输入 | 作用 | 让 hash 依赖什么 |
|---|---|---|
| ① parent_block_hash | **链式串联** | **前面所有 block 的内容**（透过父链传递）|
| ② curr_block_token_ids | 本块内容 | 本 block 自己的 token |
| ③ extra_keys | 正交上下文 | 多模态 / LoRA / cache_salt 等 |

**让 block hash"和前面内容相关"的是 ① parent_block_hash（链式），不是 ③ extra_keys。** extra_keys 是另一回事——隔离 mm/lora/salt 上下文，让"token 相同但上下文不同"时算出不同 hash，防误复用。

request_block_hasher（L589）每算新 block 都把上一个 block 的 hash 当 parent 传入：
```
block0_hash = H(NONE,        tokens0, extra)
block1_hash = H(block0_hash, tokens1, extra)   ← 含 block0
block2_hash = H(block1_hash, tokens2, extra)   ← 间接含 block0
```

因为这条父链，只要前面某 block 内容不同，它的 hash 变，后面所有 block hash 全变 → 命中必然是 [0,k] 连续前缀，中间不可能有洞。这正是 get_num_blocks_to_allocate 里 max 比较能成立的基石。

---

### 3.4 命中检测：靠内容 hash，不靠 request_id

系统里有**两套完全不同的索引**：

| | 索引键 | 作用 | 跨请求共享 |
|---|---|---|---|
| 全局 prefix cache 表（block_pool 的 BlockHashToBlockMap）| **block 内容 hash** | 查"这段 token 之前有没有人算过" | ✓ 所有请求共享 |
| req_to_blocks / num_cached_block | request_id | 记"这个请求自己占了哪些/缓存到第几个" | ✗ 每请求私有 |

命中检测走第一套：`find_longest_cache_hit(request.block_hashes, ...)` 用 token 内容 hash 去全局表查。**两个不同 request_id 但 prompt 前缀相同的请求，会命中同一批 block**——这是 prefix caching 跨请求复用的精髓。

#### new_computed_blocks vs num_req_blocks 的区别

两者内容上都是"已算好 KV 的 block"，区别在**归属和时机**：

| | num_req_blocks | new_computed_blocks |
|---|---|---|
| 是什么 | 当前已持有的 block 数 | 刚从全局缓存查到的命中 block |
| 来源 | len(req_to_blocks[id])——我的块表 | find_longest_cache_hit——全局表查内容 hash |
| 归属 | **已经是我的** | **还不是我的，正要认领** |
| 在我块表里 | ✓ | ✗ |
| ref_cnt | 已算我头上(>0) | 可能躺在 free queue(==0,待认领) |

是**同一批 block 在不同阶段的两个名字**：认领前叫 new_computed_blocks，认领后（allocate_new_computed_blocks 把它们 extend 进 req_to_blocks）并入 num_req_blocks。

为什么分开算：对空闲额度消耗不同——num_req_blocks 早不在 free queue（不消耗）；new_computed_blocks 里 evictable 的要从 free queue 捞回（消耗）。

#### local vs external computed tokens

`total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens`：
- **local**：本机就能拿到的已算 token（本地 prefix 命中 + 自己算过的）
- **external**：别的节点算好、要通过 KV connector 跨机器传来的（P/D 分离 + kv_transfer_params）

要加在一起，因为滑窗"窗口在哪、谁滑出"由整条序列前缀长度决定，不管 KV 是本地还是远端来的。单机场景 external 恒 0。

---

### 3.5 易混变量名的彻底辨析（重点）

这份代码最劝退的地方就是变量名高度相似。先立三条**解读规则**，看名字就能猜对一半：

**规则1：`num_XXX_tokens` vs `num_XXX_blocks` —— 单位不同，永远别混。** blocks 版 ≈ `cdiv(tokens 版, block_size)`，但二者常出现在不同函数里，别当成同一个变量。

**规则2：`req` = request，不是 required！**
- `num_**required**_blocks` = "**所需**块数"（目标）
- `num_**req**_blocks` = "**request** 的块数"（现状）

**规则3：`computed` = 「KV 已就位、不用再算」，不等于「命中」。** 它有**两个来源**：自己算过的 + 新命中的。只有 `num_**new**_computed_tokens` 才特指"命中"。

#### tokens 家族（全是 int，单位 = token）

| 名字 | 含义 | 来源 | prefill 首块 | decode |
|---|---|---|---|---|
| `request.num_computed_tokens` | 本请求**自己已算过**的 | Request 对象 | 0 | **N** |
| `num_new_computed_tokens` | 这次**新命中前缀缓存**的 | `find_longest_cache_hit` | 48 | **0** |
| `num_local_computed_tokens` | ↑ **两者之和**（本地已有 KV 的） | kv_cache_manager.py L354 | 48 | **N** |
| `num_external_computed_tokens` | **远端**能提供的（需搬入） | KV connector | 32 | **0** |
| `total_computed_tokens`<br>`num_total_computed_tokens` | local + external = **免算的前缀总长** | L357 / L174 | 80 | **N** |
| `num_new_tokens` | 这一步**要现算**的 | scheduler.py L665 | 20 | **1** |
| `num_lookahead_tokens` | **投机草稿** token 数 | spec decode 配置 | 0 | k |
| `num_tokens_main_model` | `total_computed + num_new`（**不含投机**） | L361 | 100 | N+1 |
| `num_tokens_need_slot` | `main_model + lookahead`（**含投机**，要占槽位） | L362 | 100 | N+1+k |
| `num_skipped_tokens` | 滑出窗口、不用再留 KV 的 | `get_num_skipped_tokens` | — | 随窗口增长 |

> ⚠️ **`total_computed_tokens` 和 `num_total_computed_tokens` 是同一个东西**，只是 `get_num_blocks_to_allocate` 的参数名没带 `num_` 前缀，而 `allocate_new_computed_blocks`（L174）内部重算时带了。纯命名不一致。

#### blocks 家族

| 变量 | 类型 | 一句话含义 | 方向 |
|---|---|---|---|
| `num_required_blocks` | 局部 int | **需要**：装下 num_tokens 总共要几个 block = `cdiv(num_tokens, block_size)` | 目标 |
| `num_req_blocks` | 局部 int | **已有**：我块表里现在持有几个 block = `len(req_to_blocks[id])` | 现状 |
| `new_computed_blocks` | 参数 **list** | 这次从**全局缓存命中**、还没收进我块表的 block | cache → 我 |
| `num_local_computed_blocks` | 局部 int | `len(new_computed_blocks) + num_req_blocks` = 内容已算好的块总数 | — |
| `num_new_blocks` | 局部 int | 要**新捞的空块**数（`get_new_blocks` pop） | — |
| `num_evictable_blocks` | 局部 int | 命中块里**躺在空闲队列**的（`touch` 也消耗额度） | — |
| `num_skipped_blocks` | 局部 int | `num_skipped_tokens // block_size` | — |
| `num_skipped_new_computed_blocks` | 局部 int | `max(0, num_skipped_blocks - num_req_blocks)` | — |
| `num_cached_block[id]` | dict 的**值** | 我持有的 block 里，前多少个已**写满+注册进全局缓存** | 我 → cache |
| `id in num_cached_block` | 当 bool 用 | 我是不是 running 请求 | 标志 |
| `num_cached_blocks` | 局部 int | `cache_blocks` 里读出的起点游标（= 上面 dict 的值，**注意差个 s**） | — |
| `num_full_blocks` | 局部 int | `num_tokens // block_size` = 本次能登记到第几块 | — |
| `new_block_ids` | 成员 **list[int]** | 新分配块的 **id**（传 worker 清零用） | 我 → worker |

#### 辨析① new_computed_blocks ≠ num_cached_block（"命中" ≠ "已缓存"）

最大的混淆源：**"命中(hit)" 和 "已缓存(cached)" 不是一回事，方向相反**。

- **new_computed_blocks = 命中(hit)**：我拿自己的 token hash 去**全局缓存表查**，发现这段 token 别人/我以前算过，把现成 block **拿来复用**。方向：**缓存 → 我**（取）
- **num_cached_block = 已注册(cached/contributed)**：我自己算出的 block **写满后**（调 `cache_blocks`）**注册进全局缓存表**，让别人以后能命中它。方向：**我 → 缓存**（存）

一个是"我从缓存里拿了几个现成块"，一个是"我把自己几个块贡献进了缓存"，几乎相反。类型也不同：前者是传入的 block 列表，后者是 manager 的成员字典 `dict[str, int]`。

#### 辨析② num_req_blocks ≠ num_required_blocks（命名陷阱）

名字长得像是 vLLM 起名的锅，看词根：
- `num_**required**_blocks` = "**所需**块数"，装下 num_tokens 需要多少块（**目标**）
- `num_**req**_blocks` = "**req**uest 的块数"，请求当前已持有多少块（**现状**）

`req` 是 **request** 缩写，**不是 required**！一个"要几个"，一个"有几个"。快路径 `num_required_blocks - num_req_blocks` = 需要 − 已有 = 还要补几个，正因不同才相减有意义。

#### 辨析③ 为什么能 num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks

能相加的前提：**两批 block 此刻互不相交（disjoint）**。
- `num_req_blocks`：**已经在** `req_to_blocks[id]` 里的块
- `new_computed_blocks`：**还没进** `req_to_blocks[id]` 的块（这次刚查到的命中，马上由 `allocate_new_computed_blocks` 收进块表）

两批无重叠 → 加起来 = "内容已经算好的块总数"（已持有 + 这次命中要认领）。

代码证据：`allocate_new_computed_blocks`（L172-173）开头有 `assert len(req_blocks) == 0`——**全新请求带命中块进来时，req_to_blocks 是空的，num_req_blocks = 0**。所以最常见的新请求场景里这个加法就是 `len(new_computed_blocks) + 0`。写 `+ num_req_blocks` 是为了让公式在"已持有一些块"的边缘情况（如和滑窗 null 占位交互，L129-132 注释）下也通用。

#### 辨析④ num_req_blocks 和 num_cached_block 不矛盾（总数 vs 子集）

它俩是**同一批持有块的两个不同度量**：

```
req_to_blocks["abc"] = [ B5,  B2,  B9,  B7,  B3 ]
                         └──── 5 个，全是"已持有" ────┘   → num_req_blocks = 5（总数）
                         [ B5,  B2,  B9 | B7,  B3 ]
                          └─写满+注册─┘  └ 还在写/没满 ┘
                                3 个                      → num_cached_block["abc"] = 3（子集）
```

- `num_req_blocks = 5`：持有的 block **总数**（含最后那个没写满的）
- `num_cached_block["abc"] = 3`：这 5 个里前 3 个**已写满并注册进缓存**

关系永远 `num_cached_block ≤ num_req_blocks`，差额就是末尾还没写满、还不能注册的块。是**总量和它的前缀子集**，不是两个互斥定义，所以不矛盾。

**num_cached_block 字典身兼两职**（之前的混淆点）：
1. **它的值**（3）= 缓存游标：已写满注册的块数
2. **它的键存不存在**（`id in num_cached_block`）= running 标志：请求是不是已在运行

快路径 `if request_id in self.num_cached_block` 用的是**职责 2**（键在不在），跟值是 3 还是 5 无关。讲快路径说的"running 标志"和讲缓存说的"已写满块数"，是**同一字典的两种用法**，不打架。

#### 辨析⑤ num_cached_block / num_cached_blocks / num_full_blocks（差一个 s 的陷阱）

```python
# cache_blocks（L250）里：
num_cached_blocks = self.num_cached_block.get(request.request_id, 0)   # 起点游标（读成员，成员是【单数】block）
num_full_blocks   = num_tokens // self.block_size                       # 终点游标（整除 → 只登记【写满】的块）
→ 本次登记区间 = blocks[num_cached_blocks : num_full_blocks]
```

- **成员** `num_cached_block`（**单数**）：manager 的状态字典
- **局部** `num_cached_blocks`（**复数**）：从字典里读出来的那一次的值
- `num_full_blocks`：**整除**得到，半满的尾块被丢掉——只有写满的块内容才定型、才能算 hash、才能登记（函数名 `cache_**full**_blocks` 的 full 就是这意思）

#### 辨析⑥ new_* 三兄弟

| 名字 | 是什么 | 怎么来的 | 后续 |
|---|---|---|---|
| `new_computed_blocks` | **命中**的块（已有 KV） | `find_longest_cache_hit` | `touch` 复用，不消耗新块 |
| `new_blocks` / `num_new_blocks` | **新捞**的空块 | `get_new_blocks`（LRU pop + 驱逐原主） | 是**脏块**，必须清零 |
| `new_block_ids` | 新捞块的 **id** | 分配时登记（仅 FullAttentionSpec） | 传 worker `_zero_block_ids` |

#### 调用方 → 被调方的改名映射（极易踩坑）

同一个值在 `kv_cache_manager.allocate_slots` 里叫一个名，进了 manager 函数换个名：

```
kv_cache_manager.allocate_slots 里                →  manager 函数签名里
──────────────────────────────────────────────────────────────────
num_tokens_need_slot                             →  num_tokens          ★
num_local_computed_tokens + num_external_...     →  total_computed_tokens
num_tokens_main_model                            →  num_tokens_main_model
```

★ **最坑的一个**：`get_num_blocks_to_allocate` / `allocate_new_blocks` 里那个朴素的 `num_tokens`，其实是**含投机 token 的 `num_tokens_need_slot`**（kv_cache_manager.py L379 / L406 传入）。

#### 用一条时间线串起全部变量

`block_size=16`，新请求，prompt 100 token，local 命中前 48（=3 块），external 32，无投机：

```
【prefill 首块】id 不在 num_cached_block → 慢路径
  ── tokens 侧 ──
  request.num_computed_tokens  = 0
  num_new_computed_tokens      = 48            ← 命中
  num_local_computed_tokens    = 0 + 48 = 48
  num_external_computed_tokens = 32
  total_computed_tokens        = 80            ← 免算 80 个
  num_new_tokens               = 20            ← 100-80，本步要现算
  num_tokens_main_model        = 80 + 20 = 100
  num_tokens_need_slot         = 100           ← 传进 manager 就叫 num_tokens ★

  ── blocks 侧（② get_num_blocks_to_allocate）──
  num_required_blocks       = cdiv(100,16) = 7    ← 需要（目标）
  num_req_blocks            = 0                   ← 已有（现状，新请求为空）
  new_computed_blocks       = [B5,B2,B9]          ← 命中 3 块（还没进块表）
  num_local_computed_blocks = 3 + 0 = 3
  num_new_blocks            = max(7 - max(0,3), 0) = 4
  num_evictable_blocks      = 2                   ← 那 3 块里 2 块躺在空闲队列
  return 4 + 2 = 6                                ← 对空闲池的总消耗

  ── 提交 ──
  ④ allocate_new_computed_blocks: 挂 [B5,B2,B9] → external 分 2 个落地块 → req_blocks=5
       num_cached_block["abc"] = 3                ← 我→缓存：命中那 3 块已注册
  ⑤ allocate_new_blocks: cdiv(100,16) - 5 = 2    ← 再捞 2 个空块 → req_blocks=7
       req_to_blocks = [B5,B2,B9, N1,N2, N3,N4]
  ⑥ cache_blocks: num_cached_blocks=3, num_full_blocks=100//16=6 → 登记 blocks[3:6]
       num_cached_block["abc"] = 6

【decode 一步】id 在 num_cached_block 了 → 快路径
  request.num_computed_tokens  = 100           ← 变成"自己的历史"了！
  num_new_computed_tokens      = 0             ← 不传，默认 0（running 不会再命中）
  num_local_computed_tokens    = 100 + 0 = 100
  num_external_computed_tokens = 0             ← 不传，默认 0
  total_computed_tokens        = 100
  num_new_tokens               = 1
  num_tokens_need_slot         = 101

  ② 快路径: assert len(new_computed_blocks) == 0
       return max(cdiv(101,16) - 7, 0) = max(7-7, 0) = 0   ← 当前块没写满，不用新块
  ④ 快路径直接 return
  ⑤ num_new = cdiv(101,16) - 7 = 0 → 返回 []
  ……一直到第 113 个 token：cdiv(113,16)=8 → 补 1 块
```

收束：**"命中(拿)"≠"已缓存(存)"；`req`(已有)≠`required`(需要)；`computed`(KV 已就位)≠`hit`(命中)；命中块与已持有块此刻不相交所以能相加；`num_req_blocks`(持有总数)与 `num_cached_block`(其中已写满数)是总量与子集、非互斥。** 两个关键：① `num_cached_block` 这个字典"键当标志、值当游标"的双重身份；② **decode 时所有 `*_computed_*` 量都退化成"这个请求自己的历史"**，`new_computed_blocks` 恒为空、external 恒为 0。

---

### 3.6 工厂（文件末尾 L1115）

```python
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}

def get_manager_for_kv_cache_spec(kv_cache_spec, **kwargs):
    return spec_manager_map[type(kv_cache_spec)](kv_cache_spec, **kwargs)
```

工厂靠 `type(kv_cache_spec)` 查表决定 new 哪个 manager。上层 KVCacheCoordinator 拿到模型的 kv_cache_groups 后，对每个 group 调一次工厂，得到该 group 对应的 manager。

---

## 四、建议阅读顺序

1. 基类 `__init__`、`allocate_new_blocks`、`free`、`cache_blocks` —— 抓住"申请/释放/缓存"三件事
2. `FullAttentionManager` —— 最简单特例，看清 find_longest_cache_hit 怎么逐 hash 查 block_pool.get_cached_block
3. `SlidingWindowManager` —— 对比看 sliding window 怎么靠 null_block 占位、实现"窗口外 block 复用"
4. 其它子类按需扩展（Qwen2.5-0.5B 纯 full attention，通常只触发 FullAttentionManager）

读完后回看 kv_cache_manager.py 和 kv_cache_coordinator.py，就能看清"多 group 的请求如何并行触发多个 single-type manager 的 allocate/free"。

---

## 五、关键设计点速查

| 设计点 | 原因 |
|---|---|
| block_size *= dcp×pcp | 把记账单位从物理 block 抬到逻辑 block，让 manager 无视切卡 |
| num_cached_block 当 running 标志 | 语义就是"running 追踪器"，且随 free pop → 抢占恢复自动退回慢路径 |
| 返回 num_new + num_evictable | 二者都从 free_block_queue 抽走（一个 pop 新块、一个 touch 旧块）|
| 只算 evictable 命中块 | ref_cnt>0 的命中块不在队列、共享即可、不占额度 |
| max(skipped, computed) | 两者都是从 block0 起的重叠前缀，取较远者 = 不用新分的前缀长度 |
| 用 total_computed_tokens 算 skip | 对应本步最早待算 token 的窗口左边界，用整体会过度删除中间还要用的 KV |
| block hash 链式（parent_block_hash）| 让命中必为连续前缀；extra_keys 只管 mm/lora/salt 隔离 |
| 命中靠内容 hash 查全局表 | 跨请求共享前缀；req_to_blocks 只是私有记账，不负责找命中 |
