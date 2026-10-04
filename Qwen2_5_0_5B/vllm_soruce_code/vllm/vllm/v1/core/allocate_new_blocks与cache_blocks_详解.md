# allocate_new_blocks / take_new_block_ids / cache_blocks 详解

> 本文记录 `single_type_kv_cache_manager.py` 中三个函数：`allocate_new_blocks`（L215-242）、`take_new_block_ids`（L244-248）、`cache_blocks`（L250-274）。它们正好构成一条链：**分配空块 → 登记待清零 → 注册进前缀缓存**。
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 姊妹篇：[allocate_new_computed_blocks 详解](./allocate_new_computed_blocks_详解.md)

---

## 〇、三者在整条链路上的位置

`allocate_slots`（kv_cache_manager.py L373-425）一步之内的完整顺序：

```
scheduler (CPU 侧)
  ① remove_skipped_blocks           先释放滑窗外的旧块（减少后面驱逐）
  ② get_num_blocks_to_allocate      【预演·算数】这一单吃掉几个空闲块？不改状态
  ③ if 消耗 > 空闲: return None      【容量检查】不够就干净退出，一个块没动
  ④ allocate_new_computed_blocks    【提交1】挂已算好的块（local touch / external 落地块）
  ⑤ allocate_new_blocks         ★   【提交2】给"这步要现算的 token"分空块
  ⑥ cache_blocks                ★   【登记】把写满的块注册进 prefix cache（token id 已知，不需 KV）
  → SchedulerOutput（含 new_block_ids_to_zero，来自 take_new_block_ids ★）
        ↓ 跨进程
worker (GPU 侧)
  ⑦ _zero_block_ids                 把新分的脏块清零
  ⑧ forward: reshape_and_cache 写 KV → 再跑 attention 读
```

三类"块的来源"各司其职：

| 需求 | 归谁 |
|---|---|
| 要不要收这个请求？ | `get_num_blocks_to_allocate`（预演，见 解析.md 3.2 节）|
| 已在别处算好、命中前缀 | `allocate_new_computed_blocks`（touch 复用 / external 落地）|
| **这一步要现算** | **`allocate_new_blocks`** |
| **新块要清零** | **`take_new_block_ids`** → worker |
| **写满的块让别人以后能复用** | **`cache_blocks`** |

---

## 一、allocate_new_blocks（L215-242）

### 1.1 定位

一句话：**给"这一步要现算的 token"从池子里申请空块，挂到请求的块表上。** 它是 `allocate_new_computed_blocks` 的对偶：

| | 处理什么 | 块的状态 |
|---|---|---|
| `allocate_new_computed_blocks` | 已算好的前缀（local 命中 + external）| 已有 KV / 待搬入 |
| **`allocate_new_blocks`** | **要现算的后缀 token** | **空块**，KV 等 forward 时写 |

### 1.2 逐行

```python
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)   # 装下 num_tokens 共需几块
    num_new_blocks = num_required_blocks - len(req_blocks)    # 减去已持有的
    if num_new_blocks <= 0:
        return []                                             # 已有的够用，不分配
    else:
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)  # 从空闲池捞空块
        req_blocks.extend(new_blocks)                                # 挂进块表
        if type(self.kv_cache_spec) is FullAttentionSpec:
            self.new_block_ids.extend(b.block_id for b in new_blocks)  # 登记待清零（见 2 章）
        return new_blocks
```

- **`num_tokens` 是累计目标，不是增量**（docstring："including tokens that are already allocated"）。所以 `num_new_blocks = 需要的总块数 − 已持有块数`。
- **`num_new_blocks <= 0` 返回 `[]`**：当前块还没写满，不用新块 —— **decode 绝大多数步都走这里**，只有序列长度跨过 block 边界（16/32/48…）那一步才真分 1 块。
- 末尾登记 `new_block_ids` 与 `allocate_new_computed_blocks` L212-213 完全一致。

**两个例子**（block_size=16）：
```
① prefill 首块：prompt 100，前缀命中 48（3块已由 ④ 挂上）
   num_tokens=100 → num_required=cdiv(100,16)=7，已持有 3 → 分 4 个空块

② decode：序列 100→101 token
   num_required=cdiv(101,16)=7，已持有 7 → num_new=0 → 返回 []
   一直到第 113 个 token：cdiv(113,16)=8 → 才分 1 块
```

### 1.3 `num_tokens_main_model` 是什么

**投机解码（speculative decoding）才有意义的参数**。docstring（L226-228）：
- **无投机解码**：`num_tokens_main_model == num_tokens`（相等，此参数无作用）
- **有投机解码**：`num_tokens_main_model = num_tokens − num_lookahead_tokens`

背景：投机解码用一个**小草稿模型**一口气猜 `k` 个未来 token（lookahead），再让**主模型（target）一次前向**验证，接受一部分、拒绝一部分。所以一步之内序列混了两类 token：

```
[ 主模型的真实 token ...... | 草稿猜的 k 个投机 token ]
└────── num_tokens_main_model ──────┘
└──────────────── num_tokens（含投机）────────────────┘
                              num_lookahead_tokens = k
```

- `num_tokens` 把投机的 k 个也算进去 —— 验证时要给它们 KV 槽位。
- `num_tokens_main_model` 是"抛开投机、主模型真正的序列长度"。

调用方计算（kv_cache_manager.py L361-365）：
```python
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)
```

**为什么要区分**：

- **基类 / 全注意力：不区分**，按 `num_tokens`（含投机）分配即可。草稿 token 验证时需要槽位；被拒的话槽位下一步直接覆盖复用。所以**基类接收 `num_tokens_main_model` 却完全不用它** —— 存在纯粹是**接口统一**，好让子类的重写版能拿到。
- **唯一真正使用者：MambaManager 的 "align" 模式**（L900-906 / L945-951）：
  ```python
  # We don't allocate blocks for lookahead tokens in align mode, because if
  # x * block_size tokens are scheduled, num_tokens is
  # x * block_size + num_lookahead_tokens and breaks the alignment.
  # We can ignore lookahead tokens because current draft models don't have mamba layers.
  num_tokens = num_tokens_main_model
  ```
  两条原因：① **对齐约束**：Mamba/SSM 的 block 要严格按 `block_size` 对齐，`x*block_size + k` 会破坏对齐，所以改用去掉投机的 `num_tokens_main_model`，投机部分单独用 `num_speculative_blocks` 预留；② **草稿模型不含 mamba 层**，lookahead token 压根不产生 mamba state。

### 1.4 为什么要重算，不复用 `get_num_blocks_to_allocate` 的结果

常见疑问：② 不是已经算过要几块了吗，直接传下来不行？**不行**，三个硬原因：

**① 两个函数算的根本不是同一个数**

| | `get_num_blocks_to_allocate`（②）| `allocate_new_blocks`（⑤）|
|---|---|---|
| 算什么 | `num_new_blocks + num_evictable_blocks` | 只算 `num_new_blocks` |
| 含义 | **对空闲池的总消耗**（新空块 + 命中块里 touch 时会被捞出来的）| **只有要现 `get_new_blocks` 的空块** |
| 目的 | 回答"够不够分" | 真的去池子里捞块 |

② 里的 `num_evictable_blocks` 是**命中块**的账 —— 它们由 ④ 的 `touch` 从空闲队列捞出，**不是 ⑤ 干的**。⑤ 若拿 ② 的数去 `get_new_blocks`，会把命中块**重复分配一遍**。

**② ③→⑤ 之间状态被改过了（最致命）**

④ 把命中块 + external 落地块 + null 占位**统统 extend 进了 `req_to_blocks`**，所以轮到 ⑤ 时 `len(req_blocks)` 已经**变大**。⑤ 必须读**当前**的 `len(req_blocks)` 才能算对还差几个空块；② 那时算的数是"改动之前"的，是**过期数据**。

**③ 这是预演→提交的事务模式，② 故意不改状态**

② 是**无副作用的 dry-run**：coordinator 要把**所有 group** 的 ② 结果**加起来**（kv_cache_coordinator.py L99-115），和空闲总数比一次（③），**够了才提交，不够就 `return None` 干净退出**。这个检查绝不能有副作用，否则请求被拒时状态就脏了。而提交阶段从 ground truth 重算 —— 经典的 check/commit 分离。

补充：⑤ 的重算是 `cdiv() − len()` 一次 **O(1) 减法**，便宜到可忽略；为省这点算力把数从 ② 传到 ⑤，反而引入耦合和过期数据的坑。

### 1.5 `touch` vs `get_new_blocks`：复用 vs 重新分配

| | `touch`（block_pool.py L391）| `get_new_blocks`（block_pool.py L322）|
|---|---|---|
| 意图 | **复用**：我要的就是这块里的**内容**（前缀命中）| **要一块地**：不在乎里面原来是什么 |
| 拿哪块 | **你指定的**（`find_longest_cache_hit` 查出来的）| **队列头部**（`popleft_n`，LRU）|
| 内容 / hash | **原封不动保留** | 若原本有缓存 → **驱逐、`reset_hash()`** |
| 查不查缓存 | 是命中的结果 | 注释明说 "we **do not check block cache** in this function" |
| 对 free queue | `remove(block)` 定点摘除 | `popleft_n` 从头批量弹出 |
| ref_cnt | `+= 1` | `0 → 1` |
| 消耗空闲额度 | **仅当 ref_cnt==0 时**消耗 1 | **每个都**消耗 1 |

**touch 全程没碰 `block_hash`、没碰内容** —— KV 原样留着给你用。

**get_new_blocks 会顺手驱逐**：
```python
if num_blocks > self.get_num_free_blocks():
    raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")   # ← 透支直接抛异常
ret = self.free_block_queue.popleft_n(num_blocks)
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)   # ← 原有缓存内容被驱逐
        block.ref_cnt += 1
```

**两个由此串起的关键点：**

1. **预检判据和 touch 行为逐字对应**：
   ```python
   touch:                     if block.ref_cnt == 0 and not block.is_null:   # 才 remove
   _get_num_evictable_blocks: sum(blk.ref_cnt == 0 and not blk.is_null ...)  # 才计数
   ```
   条件一模一样 —— 这就是"预检精确镜像提交行为"的实锤。

2. **L333 那句 `raise ValueError` 就是"透支崩溃"的实锤**：若 ② 漏算 `num_evictable_blocks` 让 scheduler 以为够，走到这里直接抛异常。不是理论风险。

---

## 二、take_new_block_ids（L244-248）

### 2.1 drain（取走并清空）语义

```python
def take_new_block_ids(self) -> list[int]:
    """Drain and return block IDs allocated since the last call."""
    ids = self.new_block_ids
    self.new_block_ids = []      # ← 清空
    return ids
```
它是增量收集器 `self.new_block_ids`（L58 初始化）的出口。

### 2.2 返回的 id 干什么用：一路传到 worker 去**清零 GPU 显存**

```
① L213 / L241   self.new_block_ids.extend(b.block_id for b in ...)   # 分配时登记
② L244          take_new_block_ids()                                  # 取走并清空
③ kv_cache_manager.py:543-547                                         # 汇总所有 group
④ scheduler.py:908-909   new_block_ids_to_zero = take_new_block_ids() or None
⑤ SchedulerOutput.new_block_ids_to_zero        (output.py:239)
⑥ gpu_model_runner.py:1084-1085  → self._zero_block_ids(...)          # GPU 上清零
```
worker 那头的注释（gpu_model_runner.py:1082）：
> Zero GPU memory for freshly allocated cache blocks to prevent **stale NaN/data** from corrupting attention or SSM computation.

### 2.3 为什么非清不可：**驱逐只删索引，不擦内存**

`_maybe_evict_cached_block`（block_pool.py L354-386）：
```python
block_hash = block.block_hash
if block_hash is None: return False
if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None: return False
block.reset_hash()
```
它只做两件事：**从哈希表摘掉索引** + **清掉 `block_hash` 元数据**。**从头到尾没碰过一个字节的 GPU 显存！**

> "驱逐"的真实含义是 **「忘记这块是什么」，不是「擦掉这块的内容」**。

所以 `get_new_blocks` 从 LRU 队头 pop 给你的块，显存里**原封不动躺着上一个请求的 KV**（或开机以来从没写过的未初始化 NaN）。**新分配的块 = 脏块。** 这就是登记 id → 清零的根本理由。

关于"块本来就要接着写，为什么还要清零"、"为什么用 0 而非 fp16-min"、"NaN 为什么 mask 挡不住"，见 [allocate_new_computed_blocks 详解](./allocate_new_computed_blocks_详解.md) 的 3.8 / 3.9 节。

### 2.4 为什么传 **id**，不传块对象

**scheduler 和 worker 是分离的**：scheduler（CPU 侧）只做逻辑记账，`KVCacheBlock` 是它的 Python 对象；worker（GPU 侧）持有真实 KV 张量。两者靠 `SchedulerOutput` 通信，必须**可序列化** —— 传不了对象引用。而 `block_id` 正是**物理 KV cache 张量里的索引**，worker 拿它就能定位显存。

### 2.5 为什么是 "take"（取走并清空）而非只读

**不只是效率，是正确性**：
```
step N:   分配 B7 → 登记 id 7 → 取走并清空 → worker 清零 B7 → forward 往 B7 写真实 KV
step N+1: 没新分配 → 取走得到 [] → 不清零 ✓

若不清空列表：
step N+1: 又取到 id 7 → worker 再清零一次 B7
          → 把 step N 刚写进去的真实 KV 全擦掉！💥
```
每块**只能在"刚被分配、还没写入"的那一步清零一次**。drain 语义保证每个 id 恰好出现一次、在正确时机。

### 2.6 为什么只有 FullAttentionSpec 登记

```python
if type(self.kv_cache_spec) is FullAttentionSpec:      # 精确类型，不是 isinstance
```
继承关系（kv_cache_interface.py）：
```
FullAttentionSpec(AttentionSpec)              ← L148 纯全注意力
├── MLAAttentionSpec(FullAttentionSpec)        ← L249 子类
└── SinkFullAttentionSpec(FullAttentionSpec)   ← L383 子类
SlidingWindowSpec / ChunkedLocalAttentionSpec / CrossAttentionSpec(AttentionSpec)  ← 兄弟
```
`type() is FullAttentionSpec` 只认纯全注意力，**故意排除 MLA、SinkFullAttention 两个子类**（`isinstance` 会把它们捞进来）。其它类型不走这条链路：MLA/Sink 显存布局与 kernel 不同；滑窗/局部注意力窗口外靠 null 占位；Mamba 有独立管理（全文件 `new_block_ids` 从不为它写入）。刻意收窄。

---

## 三、cache_blocks（L250-274）

### 3.1 定位：**"cache" = 建索引，不是存数据**

最大的认知陷阱。这里的 "cache" **不是**"把数据存进某个缓存区"，而是 **「给这些块登记索引，让别人以后能查到并复用它」**。

看 `block_pool.cache_full_blocks` 循环体真正干的事（block_pool.py L258-272）：
```python
for i, blk in enumerate(new_full_blocks):
    if blk.is_null: continue                 # 滑窗的 null 占位块跳过
    assert blk.block_hash is None
    block_hash = new_block_hashes[i]
    block_hash_with_group_id = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    blk.block_hash = block_hash_with_group_id                              # ① 给块打上 hash 标签
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)  # ② 插进全局哈希表
```
**就这两件事。KV 数据纹丝未动。** 所以不需要传"数据"——数据没动地方，动的只是"有没有人索引它"。

**一个漂亮的对称**：

| | 干什么 | 碰 GPU 显存吗？|
|---|---|---|
| `cache_full_blocks` | 设 `block_hash` + `insert(hash, blk)` → **插索引** | ❌ |
| `_maybe_evict_cached_block` | `pop(block_hash)` + `reset_hash()` → **删索引** | ❌ |

整个 prefix cache 机制**全是元数据操作**，KV 字节从写进去到被覆盖，一次都没搬过家。这正是 paged attention 高效的原因之一。

### 3.2 为什么只传两个游标，不传块列表

```python
self.block_pool.cache_full_blocks(
    request=request,
    blocks=self.req_to_blocks[request.request_id],   # ← 整个块列表传过去了
    num_cached_blocks=num_cached_blocks,             # ← 起点游标
    num_full_blocks=num_full_blocks,                 # ← 终点游标
    ...
)
```
`cache_full_blocks` 第一句就切片（block_pool.py L239）：
```python
new_full_blocks = blocks[num_cached_blocks:num_full_blocks]   # ← "哪些块"在这里
```
**为什么两个数字够**：块按 token 顺序填充，"已缓存"永远是列表的一个**连续前缀**，边界一个数字就能表达。

**`num_full_blocks = num_tokens // self.block_size` 的整除是刻意的**：**只有写满的块才能缓存**。半满的尾块内容还没定型，现在算 hash 以后还会变，整除直接把它丢掉。函数名 `cache_**full**_blocks` 的 "full" 就是这个意思。

**hash 从哪来**：docstring（block_pool.py L225-226）写明
> The block hashes values are **computed by the Request object** immediately when it is created and when new tokens are appended.

`request.block_hashes` 是 Request 早就算好的链式 hash，`cache_full_blocks` 只是读出来用。**这就是为什么要传 `request`** —— 它携带"内容的身份"。所以严格说：**"数据的身份"传了，只是"数据的字节"不用传**。

### 3.3 调用时机：主路径在 **forward 之前**！

`cache_blocks` 有两个调用点：

| 调用点 | 时机 | 场景 |
|---|---|---|
| **kv_cache_manager.py:421-425**（在 `allocate_slots` **里面**）| **forward 之前** | 主路径 |
| scheduler.py:2056 / 2066（`update_from_output` 侧）| 远程 KV 收完后 | P/D 分离补登记 |

主路径：
```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,   # ← 包含本步要算、还没算的 token！
    request.num_tokens,
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

**为什么可以提前登记？因为 block hash 只由 token id 决定，不由 KV 数值决定。** 回顾 hash 输入：
```python
hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
```
三个输入**全是 token id 和上下文，没有一个是 KV 张量**。而 token id 提前就知道。所以"建索引"完全可以先于"填数据"——索引的 key 不依赖数据。

注释（L416-420）还显示了谨慎之处：
> must exclude **"non-committable" tokens (e.g., draft tokens that could be rejected)**. Therefore, we cap the number at `request.num_tokens`, ensuring only **"finalized"** tokens are cached.

投机解码的草稿 token 可能被拒、内容会变，所以用 `min(..., request.num_tokens)` 排除 —— **只登记"已定稿"的 token**。

### 3.4 `delay_cache_blocks` 印证了提前登记的前提

```python
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)     # L413-414 直接 return，不登记
```
P/D 场景 KV 要从远程收，`allocate_slots` 传 `delay_cache_blocks=load_kv_async` → 跳过登记；等远程 KV 真收完，scheduler.py:2066 再补：
```python
# Now that the blocks are ready, actually cache them.
self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
```
说明：**"提前登记"的前提是"数据这一步内保证会就位"**。一旦数据要跨机器传、这步内不一定到，就必须 delay。这反过来证明提前登记是个**有条件的优化**。

### 3.5 "连续"指**列表下标**，不是**物理块号**

极易误解的一点。"已缓存是连续前缀"说的是**下标**，物理块号本来就是乱序散落的：

```python
req_to_blocks["abc"] = [ B5,    B2,    B9,    B7,    B3 ]
                       下标0   下标1   下标2   下标3   下标4
                      tok0-15 16-31  32-47  48-63  64-79

物理块号: 5, 2, 9, 7, 3  ←  完全乱序，散落在显存各处！
```
`blocks[num_cached_blocks:num_full_blocks]` 切的是**下标区间**。比如 `blocks[3:5]` = `[B7, B3]` —— 物理上 7 号和 3 号块毫不相邻。

这正是 解析.md 里那句：**"逻辑上连续的 token，物理上可散落在显存任意位置（block id 可乱序）——这是 paged attention 的精髓"**。

### 3.6 多个 request 怎么办：各管各的

```python
req_to_blocks = {
    "A": [B5, B2, B9],      num_cached_block["A"] = 2
    "B": [B5, B7, B3, B1],  num_cached_block["B"] = 3    ← B5 是共享的（前缀命中）
}
```
- 每个 request 有**自己独立**的块列表和游标，`cache_blocks` 里 `self.req_to_blocks[request.request_id]` 按 id 取，互不干扰。
- 甚至可以**共享物理块**：A 和 B 的 prompt 前缀相同 → 两个列表下标 0 都指向 B5，`B5.ref_cnt = 2`。这正是前缀缓存跨请求复用的样子。

### 3.7 游标闭环与那个 assert

`allocate_new_computed_blocks` L204 的注释：
```python
self.num_cached_block[request_id] = len(req_blocks)
# All cached hits (including skipped nulls) are already cached; mark
# them so cache_blocks() will not try to re-cache blocks that already
# have a block_hash set.
```
前缀命中来的块**本来就已经有 hash、已经在全局表里**（不然怎么被查中的？），所以把游标直接设成 `len(req_blocks)`，让 `cache_blocks` 从它之后才开始登记。`cache_full_blocks` 里 `assert blk.block_hash is None`（L264）就是这套游标记账的守卫 —— 游标算错、把已缓存的块又登记一遍，assert 立刻炸。

最后 `cache_blocks` L274 推进游标：
```python
self.num_cached_block[request.request_id] = num_full_blocks
```

---

## 四、关键设计点速查

| 设计点 | 原因 |
|---|---|
| `allocate_new_blocks` 只管"要现算的后缀" | 已算好的前缀归 `allocate_new_computed_blocks`；两者对偶 |
| `num_tokens` 是累计目标非增量 | `num_new = 需要总块数 − 已持有`；decode 大多数步为 0 返回 `[]` |
| `num_tokens_main_model` 基类不用 | 仅为接口统一；唯一使用者是 MambaManager align 模式（投机 token 破坏 block 对齐）|
| ⑤ 重算而不复用 ② 的数 | 数不同（② 含 evictable）、④ 改过 `req_blocks`（② 的数已过期）、② 必须无副作用 |
| `touch` vs `get_new_blocks` | 前者保内容/定点摘/按需消耗；后者不管内容/LRU 弹出/**顺手驱逐原主**/每个都消耗 |
| 驱逐只删索引不擦内存 | 所以新分配的块是**脏块**，必须清零 |
| `new_block_ids` 登记 → 清零 | 防残留 NaN/旧数据污染注意力；`type() is FullAttentionSpec` 刻意收窄 |
| `take_*` 用 drain 语义 | 每块只在"刚分配未写入"那一步清零一次；重复清零会擦掉刚写的真实 KV |
| 传 block **id** 而非对象 | scheduler/worker 跨进程，`SchedulerOutput` 须可序列化；id 即 GPU 张量索引 |
| `cache_blocks` 只建索引不搬数据 | KV 一直躺在原地；"cache" = 打 hash 标签 + 插全局哈希表 |
| 只传两个游标 | "已缓存"天然是**下标**连续前缀；整除保证只登记**写满**的块 |
| 主路径在 forward **之前**登记 | block hash 只依赖 **token id**，不依赖 KV 数值 |
| `delay_cache_blocks` | 数据要跨机器传、这步内不一定就位时，必须推迟到真收完再登记 |
| "连续" = 下标连续 | 物理块号乱序散落，正是 paged attention 的精髓 |
