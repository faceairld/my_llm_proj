# find_longest_cache_hit 详解

> 记录 `single_type_kv_cache_manager.py` 中 `find_longest_cache_hit`：基类抽象声明（L309-356）+ 各子类实现（Full L421 / SlidingWindow L486 / ChunkedLocal L625 / Mamba L785 / Cross L1067）。
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 姊妹篇：[allocate_new_computed_blocks 详解](./allocate_new_computed_blocks_详解.md) · [allocate_new_blocks / cache_blocks 详解](./allocate_new_blocks与cache_blocks_详解.md)

---

## 一、定位

**它是"前缀缓存命中查找"的入口**：给定请求的 block hash 链，去全局缓存表里查"从头开始最长能白嫖多少块"。产出的 `new_computed_blocks` 就是 `allocate_new_computed_blocks` 要 touch + 挂载的那批块。

调用链：
```
scheduler.py:612  get_computed_blocks(request)
    ↓
kv_cache_manager.py:202  coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
    ↓
UnitaryKVCacheCoordinator (L349) 单 group   →  manager_instance.find_longest_cache_hit(kv_cache_group_ids=[0], ...)
HybridKVCacheCoordinator (L453) 混合模型     →  manager_cls.find_longest_cache_hit(kv_cache_group_ids=[一批], ...)
```

**声明**（L309-311）：
```python
@classmethod
@abstractmethod
def find_longest_cache_hit(cls, ...) -> tuple[list[KVCacheBlock], ...]
```
纯抽象 + **classmethod** —— docstring 明说 "Need to be **customized for each attention type**"。

### 为什么是 `@classmethod`

因为调用方是**拿类直接调**的（`kv_cache_coordinator.py:514`）：
```python
hit_blocks = manager_cls.find_longest_cache_hit(...)   # ← 类，不是实例
```
查缓存命中发生在"还没建实例状态"的阶段，coordinator 手里只有类和一堆参数。代价：**方法内拿不到 `self.xxx`**，所以 `block_pool`、`kv_cache_spec`、`block_size`、`dcp/pcp_world_size` 全靠参数传入，`block_size` 还要在函数里重算一遍（L442-444）。那个 `assert isinstance(kv_cache_spec, ...)` 也正因如此才必要 —— 参数传进来的类型没有任何机制保证。

---

## 二、docstring 翻译（基类 L324-354）

> 获取这些块中**不超过 `max_length`** 的**最长缓存命中前缀**。该前缀必须是 `kv_cache_group_ids` 里**所有 KV cache group 的公共命中前缀**。如果没有任何命中，返回空列表。
>
> 如果启用了 eagle，**丢弃最后一个匹配上的块**，强制重算这最后一块，以获得 eagle 草稿头（drafting head）所需的 hidden states。
>
> **需要为每种 attention 类型定制实现。**
>
> **Args:**
> - `block_hashes`：该请求的 block hash 列表
> - `max_length`：缓存命中前缀的最大长度
> - `kv_cache_group_ids`：这些 KV cache group 的 id
> - `block_pool` / `kv_cache_spec` / `use_eagle`
> - `alignment_tokens`：返回的缓存命中长度（token 计）**必须是这个值的整数倍**。默认应设为 `block_size`
> - `dcp_world_size` / `pcp_world_size`：解码/预填充上下文并行的 world size
>
> **Returns:**
> 为每个 group 返回一个缓存块列表，其中**被跳过的块用 null 块替代**；返回列表长度 = `len(kv_cache_group_ids)`，第 i 个元素是第 i 个 group 的缓存块列表。
> 例：block size 4、sliding window 8、单 group 时，sliding window manager 应返回形如 `([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)])`。

### 四处注解

**① "所有 group 的公共命中前缀"**：混合模型有多个 group，同一请求在**每个 group 里都要有 KV**。group A 命中 5 块、group B 只命中 3 块 → 只能取 **min = 3**，否则 group B 那些层就缺 KV。所以必须取各 group 命中的**交集前缀**（设计文档 L164："return the **intersection** of these groups"）。

**② eagle 为什么丢最后一块**：EAGLE 草稿头吃的是**大模型的 hidden state**，而缓存里**只有 KV、没有 hidden state**。一路命中到底就拿不到 hidden state 喂草稿头 → 强制丢一块重算。详见姊妹篇讨论：`pop` 不是"hash 不确定"，而是**主动放弃命中以触发重算**（知道 token id ≠ 拥有 hidden state）。
这和 `kv_cache_manager.py:195-201` 同源 —— 即使没有 eagle，也要 cap 到 `num_tokens - 1` 才能拿到 **logits** 采样。都是"**缓存里没存的东西必须现算**"。

**③ `alignment_tokens`**：这就是"命中长度一定 block 对齐"的强制器（见五章）。

**④ NULL 占位例子**：滑窗中已彻底滑出窗口的前段不返回真实块、用 NULL 占位**保持位置对齐**，正对上 `allocate_new_computed_blocks` L198 的 `req_blocks.extend([self._null_block] * num_skipped_blocks)`。

---

## 三、返回结构：为什么是"每个 group 一个列表"

```python
computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
    [] for _ in range(len(kv_cache_group_ids))
)
```
**一次调用要同时服务一批 group。** 证据（`kv_cache_coordinator.py:410-432`）：
```python
"""Groups KV cache groups by their spec type for efficient batch processing."""
for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
    manager_cls = self.single_type_managers[i].__class__
    spec = g.kv_cache_spec
    for existing_spec, group_ids, existing_cls in attention_groups:
        if existing_spec == spec:                    # ← spec【完全相同】才合并
            assert manager_cls is existing_cls
            group_ids.append(i)
```
L514 传 `kv_cache_group_ids=group_ids`（列表）+ `kv_cache_spec=spec`（**单数**）—— 单数的 spec 本身就证明**这批 group 的 spec 完全一致**。

> ⚠️ **`kv_cache_group_ids` 里全是同一类型的 group，绝不会混进 sliding window。** L505+ 的循环是"每个 spec 类型跑一轮，各用各的 manager_cls"：
> ```
> 迭代1: (FullAttentionSpec, [0,2], FullAttentionManager) → 只查 group 0、2
> 迭代2: (SlidingWindowSpec, [1,3], SlidingWindowManager) → 只查 group 1、3
> ```

### 那为什么同一类型会有多个 group？

设计文档（`docs/design/hybrid_kv_cache_manager.md` L92-95）的两条分组规则：
1. **Identical attention type inside each group**：组内类型必须相同
2. **Identical page size across groups**：**跨组页大小必须相同**（显存池只有一种 page size）

**规则2 是关键**：要求每组显存占用一致 → 每组层数相同 → **某类型层数超过组大小时会被拆成多个 group**。文档例子（10 full + 20 sw）：
```
Group 0: 10 full attention layers          ┐ 1 个 FullAttentionManager
Group 1: 10 sliding window layers (sw.0-9)  ┐
Group 2: 10 sliding window layers (sw.10-19)┘ 2 个 SlidingWindowManager  ← 同类型被拆成 2 组！
```
反过来 **20 full + 10 sw** → full 被拆成 2 组 → `FullAttentionManager.find_longest_cache_hit(kv_cache_group_ids=[0,1])`。

**多 group ≠ 多类型**，是同类型层太多被切开。

### 为什么不写 `[[]] * N`（经典 Python 陷阱）

```python
a = [[]] * 3
a[0].append(1)
print(a)    # [[1], [1], [1]]   ← 三个槽位是【同一个列表】的引用！

b = tuple([] for _ in range(3))
b[0].append(1)
print(b)    # ([1], [], [])     ← N 个【独立】列表 ✓
```
外层 **tuple**（group 数固定、不可变正好）+ 内层 **list**（要不断 append、必须可变）。

---

## 四、FullAttentionManager 实现逐行（L421-467）

```python
assert isinstance(kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec), (...)
computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))
block_size = kv_cache_spec.block_size
if dcp_world_size * pcp_world_size > 1:
    block_size *= dcp_world_size * pcp_world_size          # 逻辑大 block（同 __init__ L50-54）
max_num_blocks = max_length // block_size                   # 最多查几块

for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    else:
        break

if use_eagle and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()

while (block_size != alignment_tokens
       and len(computed_blocks[0]) * block_size % alignment_tokens != 0):
    for computed in computed_blocks:
        computed.pop()
return computed_blocks
```

> **注**：assert 放行 `ChunkedLocalAttentionSpec`，但工厂映射是 `ChunkedLocalAttentionSpec → ChunkedLocalAttentionManager`（有自己的 L625 实现，还自己 assert 只收 ChunkedLocalAttentionSpec）。**当前路由下这个 spec 走不到这里**，疑似历史遗留。

### 4.1 四个语法点

**① `itertools.islice(block_hashes, max_num_blocks)`** = 迭代器版切片 ≈ `block_hashes[:max_num_blocks]`。
完整形式 `islice(iterable, start, stop, step)`；只给一个数时是 stop。

**为什么不直接 `[:n]`**：
1. `block_hashes` 类型是 `BlockHashList`，可能是**自定义视图对象**（如 `BlockHashListWithBlockSize`），不一定支持切片语法；`islice` 只要求可迭代。
2. **惰性、不拷贝**：`[:n]` 会新建列表拷 n 个元素；`islice` 边走边取。而这里随时 `break`，后面的元素**根本不会被取出来**。

**② `:=` 海象运算符（3.8+）**：
```python
if cached_block := block_pool.get_cached_block(...):    # 赋值 + 判断一步完成
```
等价于先赋值再 `if cached_block:`。

**③ `zip(computed_blocks, cached_block)`** —— 分发：
```python
computed_blocks = ( [],    [],    [] )      ← N=3 个 group 的结果容器
cached_block    = [ blkA,  blkB,  blkC ]    ← 这【一个 hash】在 3 个 group 里各自的物理块
zip → ([], blkA), ([], blkB), ([], blkC)
→ computed_blocks = ( [blkA], [blkB], [blkC] )
```
`computed` 是内层列表的**引用**，`append` 直接改到 tuple 里那个列表上。

**④ `else: break` 与链式 hash**（注释原文）：
> block_hashes is a **chain** of block hashes. If a block hash is not in the cached_block_hash_to_id, **the following block hashes are not computed yet for sure**.

`hash_k = H(hash_{k-1}, tokens_k, extra)`：第 k 块没命中 → 没人算过这个内容组合 → 第 k+1 块的 hash 从它派生、更没人算过 → **直接停**。这就是"**命中必为从 0 开始的连续前缀，中间不可能有洞**"的根本原因。

### 4.2 `get_cached_block` —— 全或无

`block_pool.py:184-209`：
```python
def get_cached_block(self, block_hash, kv_cache_group_ids) -> list[KVCacheBlock] | None:
    """...or **None if cache miss for any group**."""
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)  # ← 内容hash + group_id
        block = self.cached_block_hash_to_block.get_one_block(block_hash_with_group_id)
        if not block:
            return None                 # ← 任一 group 没命中 → 整体判 None
        cached_blocks.append(block)
    return cached_blocks                # ← 全命中 → 返回 N 个块（每 group 一个）
```
- **同一内容 hash + 不同 group_id → 不同 key → 不同物理块**（不同层组的 KV 本就不一样）。设计文档 L162 印证："the block pool uses a dict similar to `tuple(block_hash, group_id) -> block`... the same tokens of different groups are **cached and evicted independently**"。
- **全或无**正是 docstring "common prefix hit for **all** the groups" 在 block_pool 层的落实。
- 非空 list = truthy → 命中；`None` = falsy → `else: break`。

**走一遍（N=2）**：
```
取 h0: group0 (h0,0)→B5, group1 (h0,1)→B12 → [B5,B12]  → computed=([B5],[B12])
取 h1: → [B2,B9]                                        → computed=([B5,B2],[B12,B9])
取 h2: group0 (h2,0)→B7 ✓,  group1 (h2,1)→ 没有 ✗       → None → break
返回 ([B5,B2],[B12,B9])   ← group0 明明有 h2，但 group1 没有，整体只算命中 2 块
```

### 4.3 eagle pop（L457-460）

```python
if use_eagle and computed_blocks[0]:      # ← computed_blocks[0] 当 bool：空列表=没命中
    for computed in computed_blocks:
        computed.pop()                    # 所有 group 一起 pop，保持各组长度一致
```
- `computed_blocks[0]` 这个守卫必需：空列表 `pop()` 会 IndexError。
- **为什么丢一整块而非一个 token**：① 返回单位就是块；② `allocate_slots` 要求 `num_computed_tokens` **block 对齐**（`kv_cache_manager.py:195-201` 注释："This can trigger recomputation of an **entire block**... because allocate_slots() requires num_computed_tokens to be **block-size aligned**"）。
- 代价：多重算 ≤block_size 个 token，换最后一个 token 的 hidden state。在本就要跑的 forward 里是噪声。

---

## 五、`alignment_tokens` 与那个 while（L461-466）

```python
while (block_size != alignment_tokens                              # ← 快速短路
       and len(computed_blocks[0]) * block_size % alignment_tokens != 0):
    for computed in computed_blocks:
        computed.pop()
```
`len(computed_blocks[0]) * block_size` = 命中的 token 数。**一直 pop 到它能被 `alignment_tokens` 整除**。

| Coordinator | 传入 | 效果 |
|---|---|---|
| `UnitaryKVCacheCoordinator`（单 group，L361）| `alignment_tokens=self.block_size` | `block_size != alignment_tokens` 为 **False** → **while 直接不进** |
| `HybridKVCacheCoordinator`（混合，L521）| `alignment_tokens=self.lcm_block_size` | 各 group block_size 的**最小公倍数** |

第一个条件就是注释的 **"Faster for common case"**。**Qwen2.5-0.5B（纯 full，单 group）走这条，while 永不执行。**

**为什么混合模型要 LCM**（coordinator L445-451）：
> The **LCM of the block sizes** of all attention types. The cache hit length must be a multiple of the LCM to make sure the cache hit length is a **multiple of the block size of each attention type**. Requiring this because **we don't support partial block cache hit yet**.

不同类型 block_size 可能不同，命中长度必须**同时**是每个类型 block_size 的整数倍 → 只能取 LCM：
```
group A: block_size=16    group B: block_size=64    →  lcm = 64
命中 5 块 → 5×16 = 80 token
  80 % 64 = 16 ≠ 0 → pop 一块 → 4 块 = 64 token
  64 % 64 = 0 ✓    → 停
结果 64 token：对 A 是 4 整块，对 B 正好 1 整块 —— 都是整块 ✓
```
若停在 80：对 B（bs=64）是 **1.25 块** → 部分块命中 → 不支持。

---

## 六、SlidingWindowManager 实现要点（L486-566）

与 full attention **完全不同的算法**：

```python
assert isinstance(kv_cache_spec, SlidingWindowSpec)
assert dcp_world_size == 1, "DCP not support sliding window attn now."   # 见【已知局限】文档
assert pcp_world_size == 1, "PCP not support sliding window attn now."

sliding_window_contiguous_blocks = cdiv(kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size)
if use_eagle:
    sliding_window_contiguous_blocks += 1        # eagle 要多丢一块 → 门槛提高

computed_blocks = tuple([block_pool.null_block] * max_num_blocks for _ in ...)  # ← 先【全填 null】
num_contiguous_blocks = 0
for i in range(max_num_blocks - 1, -1, -1):      # ★ 从右往左！
    if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
        ...
        computed[i] = cached                      # ← 往命中位置填真块（不是 append！）
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= sliding_window_contiguous_blocks:
            del computed[i + num_contiguous_blocks:]   # 裁掉尾巴
            match_found = True
            break
    else:
        num_contiguous_blocks = 0                 # 断了就重新数
```

**三个关键差异**：

| | FullAttentionManager | SlidingWindowManager |
|---|---|---|
| 扫描方向 | 从左往右，一 miss 就 break | **从右往左**，允许中间 miss |
| 结果构造 | `append` 到空列表 | **先全填 null，再往位置填真块** |
| 命中判据 | 连续前缀 | 攒够 `sliding_window_contiguous_blocks` 个**连续块** |

**为什么滑窗能"允许中间 miss"**：full attention 要求整条前缀都在；滑窗**只要最后一窗的 KV 在**就能正确续算，更早的块本就用不到 → 用 null 占位即可。这也是"先全填 null"的来历（对应 `解析.md` 3.5 提到的"sliding window 的 `new_computed_blocks` 前面是 null 占位"）。

### `sliding_window - 1` 为什么减 1

窗口 `[p - sliding_window + 1, p]` **包含当前 token p 自己**。但续算时 **p 的 KV 是现算的、不需从缓存取**：
```
窗口 = [p-sw+1, ......, p-1,  p]
        └─ 需从缓存拿的 ─┘  └现算┘
        = sliding_window - 1 个                (不用缓存)
```
所以缓存里必须连续存在的 token 数 = `sliding_window - 1`。这和 `get_num_skipped_tokens = max(0, num_computed_tokens - sliding_window + 1)` 是**同一个 `-(sliding_window - 1)`**。

**⚠️ 不会把 4 块算成 3 块**（`cdiv` 向上取整）：
```
B=4, sliding_window=16（=4块）:  cdiv(15, 4) = ⌈3.75⌉ = 4 块   ← 还是 4！
```
减 1 只在 `sliding_window ≡ 1 (mod block_size)` 时才少算一块：
```
B=4, sliding_window=17（=4块+1）: 不减1 → cdiv(17,4)=5；减1 → cdiv(16,4)=4  ← 省 1（该省）
```
那多出来的 1 个 token 正是"当前要现算的"，不该为它单独占一整块。

---

## 七、各子类对照

| 子类 | 行号 | 实现要点 |
|---|---|---|
| `FullAttentionManager` | L421 | 左→右，一 miss 即 break；命中 = 连续前缀 |
| `SlidingWindowManager` | L486 | 右→左，先全填 null；攒够 `sliding_window_contiguous_blocks` 连续块即命中；🔴 assert dcp/pcp==1 |
| `ChunkedLocalAttentionManager` | L625 | 按 `attention_chunk_size` 算 `local_attention_start_idx`，窗口外整块标 null，从窗口起点往后查；🔴 assert dcp/pcp==1、use_eagle==False、block_size==alignment_tokens |
| `MambaManager` | L785 | 🔴 assert dcp/pcp==1 |
| `CrossAttentionManager` | L1067 | 不支持 prefix caching |

**ChunkedLocal 的 docstring 例子**（L645-655）很直观：
```
chunk size 8, block size 4, max length 15：
  下一个 token 在第 15 位 → 8~14 在窗口内（要查），0~7 不在窗口 → 直接标记为 computed
  查完整的 block3（8~11 token），若命中返回 [null, null, block3]，否则 [null, null]
```

---

## 八、关键设计点速查

| 设计点 | 原因 |
|---|---|
| `@classmethod` + 一长串参数 | coordinator 拿**类**直接调、不需实例；拿不到 `self.xxx` 所以全靠传参；assert 因此成为必要的运行时类型守卫 |
| 返回 `tuple[list, ...]` 每 group 一个 | 一次调用服务一批**同 spec** 的 group；同内容 hash + 不同 group_id = 不同物理块 |
| `tuple([] for ...)` 而非 `[[]] * N` | 后者是 N 个**同一列表**的引用，append 会串 |
| `islice` 而非 `[:n]` | `BlockHashList` 可能是视图对象；且惰性不拷贝，`break` 后面的根本不取 |
| `get_cached_block` 全或无 | 落实"所有 group 的**公共**命中前缀"（交集） |
| 一 miss 即 `break` | 链式 hash → 后面必然也没命中 → 命中恒为连续前缀 |
| eagle 丢最后一块 | hidden state 只在 forward 时产生；缓存里只有 KV。`pop` = 主动放弃命中以触发重算 |
| `alignment_tokens` | 单 group = block_size（while 短路）；混合 = LCM，保证对每个 group 都是整块（不支持部分块命中） |
| 滑窗右→左 + null 预填 | 滑窗只需最后一窗，允许中间 miss；null 占位保持位置对齐 |
| `sliding_window - 1` | 窗口含当前 token，而它现算不需缓存；`cdiv` 向上取整，不会少算整块 |
