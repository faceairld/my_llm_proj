# KVCacheSpec 与 KV Cache 物理布局 详解

> 跨文件记录：`vllm/v1/kv_cache_interface.py`（KVCacheSpec 及子类、`page_size_bytes`、`merge`）+ worker 侧的真实显存布局（`worker/gpu/attn_utils.py`、`worker/gpu/block_table.py`、`attention/backends/flash_attn.py`）。
>
> 回答的核心问题：**block 里到底存了什么？layer 维度在哪？block_id 怎么定位到实际数据？**
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 相关：[已知局限与困境](./vLLM_KVCache_已知局限与困境.md)

---

## 一、KVCacheSpec 是什么

**KV cache 的"格式描述层 / 词汇表"** —— 静态说明书，**不含任何实际 KV 张量、也不做分配**。它只回答："这类层的 KV 每块多大、怎么算显存、能不能和别的层合并"。

```
kv_cache_interface.py  →  KVCacheSpec 及子类      ← 【格式描述层】静态规格
                              ↓ 决定
KVCacheCoordinator     →  Unitary / Hybrid        ← 按 spec 数量选协调器
                              ↓ 每 group 一个
SingleTypeKVCacheManager  →  Full/Sliding/Mamba…  ← 按 spec 类型选 manager
                              ↓
BlockPool              →  一个池子，页大小由 spec 统一
```

**继承体系**：
```
KVCacheSpec（L69）  ← 只有 block_size
├── AttentionSpec（L114）        + num_kv_heads / head_size / dtype / kv_quant_mode
│   ├── FullAttentionSpec（L148）
│   │   ├── MLAAttentionSpec（L249）        ← 子类！
│   │   └── SinkFullAttentionSpec（L383）   ← 子类！
│   ├── ChunkedLocalAttentionSpec（L288）   ┐
│   ├── SlidingWindowSpec（L307）           ├ 兄弟，不是 FullAttentionSpec 的子类
│   ├── EncoderOnlyAttentionSpec（L363）    │
│   └── CrossAttentionSpec（L370）          ┘
└── MambaSpec（L333）
```
> 这个继承关系解释了两处判断的区别：`type(spec) is FullAttentionSpec`（**精确**，排除 MLA/Sink）vs `isinstance(spec, ...)`（**认子类**）。见 [allocate_new_blocks / cache_blocks 详解](./allocate_new_blocks与cache_blocks_详解.md) 3.7 节。

### 它是不是只服务混合模型？

**不是** —— 纯 full attention 模型（如 Qwen2.5-0.5B）也用 `FullAttentionSpec`，照样需要 `block_size` 做 token↔block 换算、需要 `page_size_bytes` 算显存。

**但它的复杂度确实主要为 Hybrid Memory Allocator 而生**。设计文档（`docs/design/hybrid_kv_cache_manager.md` L47-55）点出根本目的：
> We use a **single memory pool** for all layer types... The core challenge is ensuring every layer type uses the **same page size**.

| spec 里的东西 | 纯 full 用得上 | 为混合模型服务 |
|---|---|---|
| 多个子类（Sliding/ChunkedLocal/Mamba/Cross/MLA…） | ❌ | ✅ 每种 attention 一个 spec |
| `page_size_bytes` 作为统一接口 | 用（值固定） | ✅ **核心**：强制所有类型页大小相同 |
| `merge` / `merge_window_sizes` | 几乎用不到 | ✅ 折叠同组各层 + 校验一致 |
| `copy_with_new_block_size` | ❌ | ✅ 多类型 block_size 需换算时 |

---

## 二、`page_size_bytes` —— 一个块占多少字节

**"page" 和 "block" 是同义词**（paged attention 借用操作系统"内存分页"的说法）。docstring：
> The size of a **page** with `block_size` tokens **in bytes**.

基类是 `@property` + `raise NotImplementedError`（L77-85）—— 基类不知道头数、dtype，得子类才能算。`AttentionSpec` 给出实现（L136-144）：
```python
@property
def real_page_size_bytes(self) -> int:
    return 2 * self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)
```

| 因子 | 含义 |
|---|---|
| **2** | 存 **K 和 V 两份** |
| `block_size` | 一块装这么多 token |
| `num_kv_heads` | KV 注意力头数 |
| `head_size` | 每头维度 |
| `get_dtype_size(dtype)` | 每个数几字节（fp16=2, fp8=1） |

L121-134 的 `page_size_bytes` 在此基础上加料：per-token-head 量化要额外算 scale 的空间；`page_size_padded` 允许把页撑大到某对齐值。**默认走纯公式。**

### 单位分析（易错点 ⚠️）

文档里写作 `page_size_bytes = block_size × kv_hidden_size`，其中 `kv_hidden_size` 的单位是 **字节 / token**，**不是块数**：
```
page_size_bytes  =  block_size  ×  kv_hidden_size
   [字节/块]          [token/块]      [字节/token]        ← token 约掉，得字节/块 ✓

kv_hidden_size = 2 × num_kv_heads × head_size × dtype字节
                 └──── 一个token的KV有多少【个数】────┘ × 【字节/个】
                 = 一个 token 的 KV 占多少【字节】   ← 就是 字节/token
```
文档 L25 原话："The number of **bytes** to store **one token's** KV cache for a single layer."

**代入 Qwen2.5-0.5B**（num_kv_heads=2, head_size=64, fp16, block_size=16）：
```
kv_hidden_size  = 2 × 2 × 64 × 2 = 512 字节/token
    （一个 token 的 K 有 2×64=128 个数，V 也 128 个，共 256 个 × 2字节 = 512）
page_size_bytes = 16 × 512 = 8192 字节 = 8KB / 块 / 层
```

### 🚩 文档定义 ≠ 代码定义（文档亲自标注）

设计文档 L36-42 专门提醒：
```
文档的 page size    = num_layers × block_size × kv_hidden_size    ← 含层数
代码 page_size_bytes = block_size × kv_hidden_size                ← 【不含】层数（单层的）
```
对 Qwen：代码 8KB（单层一块）；文档口径 8KB × 24 层 = **192KB**（一个 block_id 概念上横跨全部层）。读代码时按**代码定义（单层）**理解。

### 两处用途

1. **启动时算显存能放多少块**（`FullAttentionSpec.max_memory_usage_bytes` L170-178；`attn_utils.py:160`）：
   ```python
   num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
   ```
   这就是 BlockPool 的 `num_gpu_blocks` 来源。
2. **"跨 group 页大小必须相同"这条分组规则比的就是它** —— 显存池只有一种页大小，所有 group 的块必须一样大才能共用一个池子。

---

## 三、`merge` —— 把"同组各层的 spec"折叠成一个

一个 KV cache group = 很多**规格相同**的层，但这些层各有自己的 spec 对象，建 group 时要合并成代表整组的那一个。

**基类实现（L102-110）最能说明意图**：
```python
@classmethod
def merge(cls, specs: list[Self]) -> Self:
    assert all(spec == specs[0] for spec in specs[1:]), (   # ← 断言：全都一样！
        "All layers in the same KV cache group must be the same."
    )
    return copy.deepcopy(specs[0])                          # 一样 → 随便拷一个当代表
```
**核心是那个 assert**：强制校验"同组各层规格必须完全相同" —— 分组规则1（identical attention type inside each group）在代码里的落地。**`merge` 一半是"合并"，一半是"守卫"**：谁把规格不同的层错分进一组，这里当场炸。

**`FullAttentionSpec.merge` 为什么要重写（L192-218）**：full attention 有个特殊字段 `sliding_window`（注释 L149-156：混合模型禁用 hybrid allocator 时，滑窗层被当 full 处理但**记下窗口大小**）。这些层其它规格一样，但可能有的记了 window、有的没记，不能简单 `==` 判等。所以它收集所有窗口值再走 `merge_window_sizes`（L180-190）：
```python
if len(window_sizes) == 0:   return None        # 没滑窗
elif len(window_sizes) == 1: return window_sizes.pop()   # 都相同 → 合成一个
else: raise ValueError("All attention layers in the same KV cache group must have the same window size.")
```
又是一层"同组必须一致"的守卫。

**顺带 `copy_with_new_block_size`（L96-100）**：
```python
return replace(self, block_size=block_size)
```
`replace` 是 dataclass 工具：**复制一份、只改 block_size**（spec 是 `frozen=True` 不可变，只能造新的）。

---

## 四、group 是按什么划分的

设计文档 L92-95 的**两条规则**：
> 1. **Identical attention type inside each group**：组内类型必须相同
> 2. **Identical page size across groups**：跨组页大小必须相同（memory pool only has one page size）

**规则2 导致"同类型也可能被拆成多个 group"**（每组层数需相同）。文档例子（10 full + 20 sw）：
```
Group 0: 10 full attention layers          → 1 个 FullAttentionManager
Group 1: 10 sliding window layers (sw.0-9)  ┐
Group 2: 10 sliding window layers (sw.10-19)┘ → 2 个 SlidingWindowManager
```
L226 原话："use **1 `FullAttentionManager` and 2 `SlidingWindowManager`** for the 3 `KVCacheGroup`s."
L115-120 还有更极端的：某组凑不满时用 **padding layers** 补齐。

### 不同 group 之间的块能互相 prefix hit 吗？——不能，而且**本就不该**

`make_block_hash_with_group_id(block_hash, group_id)`（`kv_cache_utils.py:49-58`）把 group_id 拼进 key：group 0 的 key 是 `(h,0)`、group 2 是 `(h,2)`，**同内容也算出不同 key**。设计文档 L162 印证：
> the block pool uses a dict similar to `tuple(block_hash, group_id) -> block`... the same tokens of different groups are **cached and evicted independently**.

**这是正确性保护，不是缺陷**：Group 1 装 sw.0~sw.9 的 KV、Group 2 装 sw.10~sw.19 的 KV —— 是**模型里不同层**，同一 token 的 KV 数值根本不一样。让它们互相命中 = 拿第 0 层的 KV 冒充第 10 层，纯粹是错的。

**真正的复用发生在"同一 group 内、跨请求"之间**：
```
请求 A、B 前缀相同：
  A 在 group 1 第 0 块 → key (h0,1)  ┐ 同一个 key → B 命中 A 的块 ✓
  B 在 group 1 第 0 块 → key (h0,1)  ┘   （都是 sw.0~sw.9 层，KV 真的一样）
```
所以**拆 group 完全不损害命中率**，只是把一段 token 的 KV 按层摊进几个 group，查命中时对各 group 取交集（`get_cached_block` 的"全或无"）。

---

## 五、物理布局：block 里到底存了什么

### 5.1 `KVCacheBlock` 对象**一个 KV 都不存**

`kv_cache_utils.py:110` 全部字段：
```python
class KVCacheBlock:
    block_id: int                      # 物理块编号（索引）
    ref_cnt: int = 0                   # 引用计数
    _block_hash: ... = None            # 内容 hash
    prev_free_block / next_free_block  # 空闲链表指针
```
**没有任何 KV 张量字段。** 它是纯**元数据/记账对象**，住在 CPU 侧。真数据在 worker（GPU 侧）的 KV cache 张量里，二者靠 `block_id` 连接。

### 5.2 layer 维度在哪：是 `dict` 的 key，不是张量的轴 ★

`worker/gpu/attn_utils.py:150`：
```python
kv_caches: dict[str, torch.Tensor] = {}          # ← key 是 layer_name！
for layer_name in kv_cache_group_spec.layer_names:
    ...
    kv_caches[layer_name] = raw_tensor.permute(...)   # 每层一个独立张量
```
`attention/backends/flash_attn.py:143` 的 `get_kv_cache_shape` 返回**单层**形状：
```python
return (2, num_blocks, block_size, num_kv_heads, head_size)
#       │
#       └─ K 和 V（不是层数！）
```
真实结构：
```
kv_caches = {
    "layer.0.attn":  张量 [2, num_blocks, block_size, num_kv_heads, head_size]
    "layer.1.attn":  张量 [2, num_blocks, block_size, num_kv_heads, head_size]
    ...
    "layer.23.attn": 张量 [2, num_blocks, block_size, num_kv_heads, head_size]
}       ↑ 24 个独立张量，layer 维度体现在【有几个 key】上
```
| 维度 | 含义（Qwen: `[2, num_blocks, 16, 2, 64]`） |
|---|---|
| `2` | K 和 V |
| `num_blocks` | 第几块（← **`block_id` 索引这里**） |
| `block_size`(16) | 块内第几个 token |
| `num_kv_heads × head_size`(2×64) | **一个 token 的 K（或 V）向量** |

> 不同 backend 的排布可能不同，如 `cpu_attn.py:74` 是 `(2, num_blocks, num_kv_heads, block_size, head_size)`；`get_kv_cache_stride_order` 还会做 permute。

### 5.3 一个 token 的所有层 KV 在一个 block 里吗

**`block_id` 是所有层共享的同一个编号。**
```
token 位置 p，落在 block_id=9：
  layer 0  的张量[:, 9, p%16, :, :]  ← 第 0 层的 K/V
  layer 1  的张量[:, 9, p%16, :, :]  ← 第 1 层
  ...
  layer 23 的张量[:, 9, p%16, :, :]  ← 第 23 层
     ↑ 同一个 block_id、同一个偏移，但落在【24 个不同的张量】里
```
- **逻辑上（按 block_id）**：一个 block_id 覆盖该 token 在**所有层**的 KV → 可以说"都在一个 block 里"
- **物理上**：没有一个连续张量装下所有层；每层各存各的，只是**用同一 block_id、同一偏移**

这正是二章那个"代码 8KB（单层） vs 文档 192KB（全层）"差异的由来 —— **block_id 就是把 24 个物理张量绑在一起的那根线。**

### 5.4 `block_id` → 实际数据：slot 寻址

`worker/gpu/block_table.py:256-264` 的 slot_mapping kernel：
```python
block_indices = positions // (block_size * CP_SIZE)   # token 位置 → 块表第几项
block_offsets = positions % (block_size * CP_SIZE)    # → 块内第几个槽
block_numbers = tl.load(block_table_ptr + req_state_idx * block_table_stride + block_indices)
if CP_SIZE == 1:                                       # 常见情况（未开 CP）
    slot_ids = block_numbers * block_size + block_offsets   # ★ slot_id = block_id × block_size + 偏移
```

**完整寻址链**：
```
要取 token 位置 p、第 L 层的 K：
  block_idx = p // block_size                          （第几个逻辑块）
  block_id  = block_table[req_idx][block_idx]          （二维表查物理块号，如 9）
  offset    = p % block_size                           （块内第几个槽）
  K = kv_caches["layer.L"][0, block_id, offset, :, :]  （0=K；层由 dict key 选）
      └─ 用【dict key】选层，用【block_id/offset】选位置 ─┘
```
**选层和选位置用两套完全不同的机制** —— 这就是为什么 layer 不在张量维度里。

### 5.5 block_table 是二维的

从 kernel 的索引方式 `block_table_ptr + req_state_idx * stride + block_indices` 可见是 `block_table[req_idx][block_idx]` → **2D**，值 = 物理 block_id。而 `self.block_tables: list`（block_table.py:37）是**每个 kv_cache_group 一张**：
```
per group:  [num_reqs, max_num_blocks]   ← 2D，元素是 block_id
跨 group:   list 里 num_groups 张         ← 相当于 [num_groups, num_reqs, max_num_blocks]
```

### 5.6 CPU 元数据 vs GPU 数据（同一信息的两种形态）

| | CPU 侧（scheduler/manager） | GPU 侧（worker） |
|---|---|---|
| 块对象 | `KVCacheBlock`（元数据） | KV cache 张量（真数据） |
| 存什么 | block_id、ref_cnt、hash | 每 token 每层的 **K/V 向量** |
| 块表 | `req_to_blocks: dict[req_id → list[KVCacheBlock]]` | `block_tables: [num_reqs, max_num_blocks]` **int 张量** |
| 例 | `{"abc": [B5, B2, B9]}` | 第 abc 行 = `[5, 2, 9, ...]` |

manager 用对象记账，喂给 GPU kernel 的是**把 block_id 抽出来铺成的二维 int 张量**。

### 5.7 token_id 和 hidden state 都不在 block 里

- **token_id** 住在 CPU 侧的 `Request` / `input_batch`：① 喂 embedding 做 forward 输入；② 算 block hash（`hash_block_tokens` 用的就是 token_ids）。
  → 这解释了为什么 `cache_blocks` 能在 **forward 之前**登记：hash 只需 token_id，不需 KV。
- **hidden state** 算完即弃、从不落盘 → 这解释了 EAGLE 为什么必须丢块重算（知道 token id ≠ 拥有 hidden state）。

---

## 六、闭环：几个此前结论的物理解释

| 之前的结论 | 物理原因 |
|---|---|
| **"驱逐只删索引不擦内存"** | `_maybe_evict_cached_block` 动的是 CPU 侧哈希表 + `block_hash` 字段，**GPU 张量里的 K/V 字节没动** → 新分配的块是脏块 → 要清零 |
| **"cache 只建索引不搬数据"** | `cache_full_blocks` 给 `KVCacheBlock` 打 hash 标签、插哈希表，GPU 里的 K/V 一直躺原地 |
| **paged attention 的精髓** | slot 寻址是 `block_id × block_size + offset`，每块独立定位 → `block_id` 可以乱序、物理散落无所谓 |
| **`page_size_bytes` 是分组的核心约束** | 显存池只有一种页大小 → 所有 group 的块必须一样大 |

---

## 七、关键设计点速查

| 设计点 | 原因 |
|---|---|
| spec 是"格式描述层"，不含张量 | 静态说明书；真正干活的是 coordinator / manager / block_pool |
| `page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype字节` | 一块（单层）占多少字节；`kv_hidden_size` 单位是**字节/token** 不是块数 |
| 代码 page size 不含层数、文档含 | 代码的物理单位是"单层一块"；文档口径是"一个 block_id 横跨全部层" |
| `merge` 的核心是 assert | 强制"同组各层规格完全一致"；FullAttentionSpec 额外处理 `sliding_window` 字段 |
| group 划分两条规则 | 组内类型同 + **跨组页大小同**（后者导致同类型层太多时被拆成多组） |
| group_id 拼进 block hash | 不同 group 装**不同层**的 KV，跨 group 复用是错的 —— 这是**正确性保护**，不损害命中率 |
| `KVCacheBlock` 只有 block_id/ref_cnt/hash | 它是记账对象；数据在 GPU 张量，靠 block_id 连接 |
| layer 是 `kv_caches` 的 **dict key** | 每层一个独立张量；选层用 key、选位置用 block_id+offset，两套机制 |
| `slot_id = block_id × block_size + offset` | 每块独立定位 → 逻辑连续的 token 可以物理乱序散落 |
| block_table 是 `[num_reqs, max_num_blocks]` int 张量 | CPU 的 `req_to_blocks` 抽出 block_id 铺成，供 GPU kernel 用 |
