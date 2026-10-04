# free / remove_skipped_blocks 详解（块释放机制）

> 记录 `single_type_kv_cache_manager.py` 两个**释放**函数：`free`（L276-291）、`remove_skipped_blocks`（L358-399），外加它们依赖的 `get_num_skipped_tokens`（L401）。
>
> 二者天然成对：**都调用 `block_pool.free_blocks`，都用「倒序」编码驱逐优先级**。
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 相关：[find_longest_cache_hit 详解](./find_longest_cache_hit_详解.md) · [已知局限与困境](./vLLM_KVCache_已知局限与困境.md)

---

## 〇、两者的分工

| | `free` | `remove_skipped_blocks` |
|---|---|---|
| 何时 | **请求结束/被抢占**时 | **每次 `allocate_slots` 的第①步**（分配之前） |
| 释放什么 | 该请求的**全部**块 | 仅**滑出注意力窗口**的块 |
| 块表 | 整条 `pop` 掉 | 滑出的位置**换成 null**，表还在 |
| 对谁有意义 | 所有类型 | **仅滑窗/局部注意力**（full attention 恒 0，空转） |

共同点：**都把"驱逐优先级"编码进传给 `free_blocks` 的顺序里**（见二章）。

---

## 一、`free`（L276-291）

```python
def free(self, request_id: str) -> None:
    # Default to [] in case a request is freed (aborted) before alloc.
    req_blocks = self.req_to_blocks.pop(request_id, [])

    # Free blocks in reverse order so that the tail blocks are freed first.
    ordered_blocks = reversed(req_blocks)

    self.block_pool.free_blocks(ordered_blocks)
    self.num_cached_block.pop(request_id, None)
```

四行，每行都有讲究：

### 1.1 `req_to_blocks.pop(request_id, [])`

- **`pop` 而非读取**：块表条目直接删掉。
- **默认值 `[]`**：注释说明是防止请求**在分配之前就被 abort**（`reversed([])` 安全返回空）。

### 1.2 `reversed(req_blocks)` ★ 核心

契约在 `block_pool.free_blocks`（block_pool.py:408-410）：
> The blocks should be **ordered by their eviction priority**, where the **first block will be evicted first**.

```python
def free_blocks(self, ordered_blocks):
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n(       # ← 按给定顺序依次进队列
        [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
    )
```
free queue 是 **FIFO**：`append_n` 从尾进，`get_new_blocks` 用 `popleft_n` **从头取** → **先进队列的先被驱逐**。**传进去的顺序 = 死亡顺序。**

**为什么要尾块先死**：前缀缓存的价值不对称 ——

| | 内容 | 未来被别的请求命中的概率 |
|---|---|---|
| **块 0、1、2…（前缀开头）** | system prompt、few-shot、共享对话前缀 | **高** |
| **尾部块** | 本请求特有的生成内容 | **几乎为 0** |

而且**被 free 的块并没有被清空**，它进队列后是"**僵尸态**"：`ref_cnt == 0`、躺在队列里当驱逐候选，**但内容还在、`block_hash` 还挂在全局哈希表里** → 随时可能被未来的请求 `find_longest_cache_hit` 查中、`touch` **复活**。只有真被 `get_new_blocks` 捞走、触发 `_maybe_evict_cached_block` 才**永久死亡**。

所以要让**高价值的前缀块在僵尸态待得越久越好**，把没人要的尾块推去先送死：

```
req_to_blocks["abc"] = [ B5,     B2,     B9,     B7,     B3 ]
                       tok0-15  16-31   32-47   48-63   64-79
                       ↑最有复用价值                    ↑最没价值

reversed → [B3, B7, B9, B2, B5]  依次 append 进队列尾
free_block_queue: [ ...更老的块... , B3, B7, B9, B2, B5 ]
                                      ↑先被popleft      ↑最后才被popleft
                                    (最先永久驱逐)      (存活最久，最可能被命中复活)
```
**不 reverse 的话**，B5（前缀开头）会排最前先被驱逐 —— 把最可能被复用的块最先扔掉，命中率直接崩。

### 1.3 `num_cached_block.pop(request_id, None)`

游标也 pop 掉。这正是 `解析.md` L271 记的机制：`num_cached_block` 的**键兼作 running 标志**，free 时 pop → **被抢占的请求恢复时自动退回慢路径、重新查缓存命中**（它刚才算的 KV 可能还在僵尸态躺着，正好能命中复活）。

### 1.4 `free_blocks` 里的两个细节

- **`block.ref_cnt -= 1`，只有减到 0 才进队列**：若另一请求因前缀命中共享这块（ref_cnt=2），本请求 free 后 ref_cnt=1 → **不进队列**，对方继续正常使用。共享语义就是这么维持的。
- **`not block.is_null`**：null 占位块是全局共享的，从不进空闲队列。

---

## 二、`remove_skipped_blocks`（L358-399）

```python
def remove_skipped_blocks(self, request_id: str, total_computed_tokens: int) -> None:
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    if num_skipped_tokens <= 0:
        # full attention 永远走这里，直接返回
        return
    blocks = self.req_to_blocks[request_id]
    num_skipped_blocks = num_skipped_tokens // self.block_size
    # `num_skipped_tokens` may include tokens that haven't been allocated yet ...
    # so we must cap to the number of blocks that currently exist for this request.
    num_skipped_blocks = min(num_skipped_blocks, len(blocks))          # ★ 见 2.1
    removed_blocks: list[KVCacheBlock] = []
    for i in range(num_skipped_blocks - 1, -1, -1):                    # ★ 见 2.2
        if blocks[i] == self._null_block:
            break
        removed_blocks.append(blocks[i])
        blocks[i] = self._null_block
    self.block_pool.free_blocks(removed_blocks)
```

**调用位置**（`kv_cache_manager.py:367-375`）—— `allocate_slots` 的**第①步，在任何分配之前**：
```python
# Free the blocks that are skipped during the attention computation
# (e.g., tokens outside the sliding window).
# We can do this even if we cannot schedule this request due to insufficient free blocks.
# Should call this function before allocating new blocks to reduce the number of evicted blocks.
self.coordinator.remove_skipped_blocks(request.request_id, total_computed_tokens)
```
放最前面的理由注释也写了：**先把滑出窗口的块还回池子，这一步的分配就能直接复用它们，少驱逐几个别人的缓存。**

### 2.1 `min(num_skipped_blocks, len(blocks))` 为什么需要

注释原文：
> `num_skipped_tokens` may include tokens that **haven't been allocated yet** (e.g., when the attention window moves into the **external computed tokens** range), so we must **cap to the number of blocks that currently exist** for this request.

**根因：调用时机在分配之前**，两个量不同步：

| | 描述的时刻 |
|---|---|
| `total_computed_tokens`（参数） | **这一步结束后**的已算 token（含**尚未挂上**的 local 命中 + **尚未分配落地块**的 external） |
| `len(blocks)` | **这一步分配之前**，该请求**当前**实际持有的块 |

`num_skipped_blocks` 从**前者**推出，循环却要索引**后者**。前者可能远跑在后者前面。

**不 cap 就 IndexError**：
```
全新请求，sw=8，block_size=16：
  req_to_blocks["abc"] = []              ← 空！④ 还没跑，命中块还没挂上
  local 命中 48 + external 32 → total_computed_tokens = 80
  num_skipped_tokens = max(0, 80-8+1) = 73
  num_skipped_blocks = 73 // 16 = 4

不 cap:  for i in range(3, -1, -1): → blocks[3] → 💥 IndexError（blocks 是空列表）
cap 后:  min(4, 0) = 0 → 空循环 → 什么也不做 ✓（这个请求一块都没有，本就没块可释放）
```

**⚠️ 但不要以为 `len(blocks)` 恒为 0** —— `allocate_slots` **对 running 请求也会调**（`scheduler.py:463`），那时 `req_to_blocks` 装满了前面步骤积累的块：

| | `req_to_blocks` | `num_skipped_blocks` vs `len(blocks)` | `min` 起作用 | 函数干活 |
|---|---|---|---|---|
| **新请求**（WAITING 首次） | **空 `[]`** | 4 vs 0 → **超了** | ✅ **兜底防 IndexError** | ❌ 空转 |
| **running 请求**（decode） | **满的** | 1 vs 4 → 正常 | ❌ 不生效 | ✅ **真正回收滑出窗口的块** |

**running 那条路才是主战场** —— 滑窗"边解码边回收"天天在发生。新请求的跳过区**不归这里管**，等 ④ `allocate_new_computed_blocks` 挂命中块时会自己补 null 占位（L198）。

### 2.2 倒序循环 `range(num_skipped_blocks - 1, -1, -1)`

**语法**：`range(start, stop, step)`，step=-1 倒着走；stop=-1 是**开区间**，写 -1 才能取到下标 0。
```python
range(3, -1, -1)  →  3, 2, 1, 0
range(3,  0, -1)  →  3, 2, 1      ❌ 漏掉 0
```
起点 `num_skipped_blocks - 1` 的 off-by-one，上面那行注释专门解释了：
> Because the block starts from index 0, the num_skipped_block-th block corresponds to index **num_skipped_blocks - 1**.

**为什么倒着走——两个好处**：

**① 提前终止，每次只处理新增的**。跳过区**从前往后单调增长**，前面的块早被之前的调用 null 化了，**新滑出的在区间尾部**。倒着走先撞上新的，一碰到 null 就知道"前面全是老账" → `break`：
```
上一步结束: req_to_blocks = [null, B2, B9, B7]     （下标 0 已 null 化）
这一步 num_skipped_blocks = 2 → range(1, -1, -1) → i = 1, 0
  i=1: blocks[1]=B2 不是 null → removed=[B2], blocks[1]=null
  i=0: blocks[0]=null         → break ✓（不用再往前扫）
free_blocks([B2]) → 结果 [null, null, B9, B7]
```
**正着走** `range(0, 2)`：i=0 立刻撞 null —— `break` 是错的（B2 还没处理），`continue` 就得**每次扫完整个跳过区**。序列越长跳过区越大 → O(跳过区长度) 的重复劳动。倒序让每次只花 **O(本次新增块数)**。

**② `removed_blocks` 顺序天然符合 `free_blocks` 的契约**。倒着 append → `removed_blocks = [靠后的块, ..., 靠前的块]` → 正是"先释放的先被驱逐"想要的顺序（靠后=价值低）。**和 `free` 里 `reversed(req_blocks)` 同一用意，只是倒序遍历顺手就满足了。**

---

## 三、`get_num_skipped_tokens`（L401）—— skip 的判据

```python
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    # The default behavior is to not skip any tokens.
    return 0
```
基类**有默认实现**（返回 0），不是纯抽象 —— 所以 **full attention 恒 0，整套 skip 机制不触发**（`remove_skipped_blocks` L375 直接 return）。

`SlidingWindowManager` 重写（L607）：
```python
return max(0, num_computed_tokens - self.sliding_window + 1)
```
docstring 的图（L588-599）：
```
sliding_window=4, num_computed_tokens=7
Tokens:   [ 0  1  2  3  4  5  6  7 ]
          | ---- computed -----|
                                 ^ next token to be computed
                       |-----------| sliding window for next token
          |--skipped---|
→ get_num_skipped_tokens(7) == 4
```

**为什么参考点是"续算起点"而非完整长度**（正确性问题）：skip 判据是"从**现在要续算的第一个 token** 往回看一个窗口，再往前的谁都不用了"。而它**后面的 token 还没算**、还要回看命中前缀的尾部。用完整长度会把"马上要 prefill 的 token 仍需要"的块错误 skip 掉：
```
sw=4，prompt 20，命中前 8 → 续算从 token 8 开始
  正确: skip = max(0, 8-4+1) = 5 → 丢 token 0~4，保留 5,6,7（token 8 的窗口={5,6,7,8} 要用）
  错误: 按 20 算 → skip = 17 → 丢 0~16 → token 8 还要用 5,6,7 → 💥
```
这个 `-(sliding_window - 1)` 和 `find_longest_cache_hit` 里 `cdiv(sliding_window - 1, block_size)` 的减 1 **同源**：窗口含当前 token，而当前 token 现算、不需从缓存取。

**skip 数被用在两处**：
1. **分配时**（`get_num_blocks_to_allocate` / `allocate_new_computed_blocks`）：跳过的整块用 `_null_block` 占位，不需真实显存，从"要新申请"里扣掉
2. **运行时**（本文的 `remove_skipped_blocks`）：真把滑出窗口的块在块表换成 null 并 `free_blocks` 还给 block_pool

---

## 四、关键设计点速查

| 设计点 | 原因 |
|---|---|
| `free` 用 `reversed(req_blocks)` | free queue 是 FIFO，先进先驱逐 → 倒序让**尾块先死、前缀块留最久**（前缀最可能被未来请求命中复活） |
| 被 free 的块是"僵尸态" | `ref_cnt=0` 在队列里，但**内容和 hash 还在** → 能被 `touch` 复活；只有 `get_new_blocks` 捞走才永久死 |
| `free` 里 `pop` 掉 `num_cached_block` | 键兼作 running 标志 → 抢占恢复时自动退回慢路径、重查命中 |
| `free_blocks` 只对 ref_cnt→0 的入队 | 有别的请求共享时（前缀命中）不能还回池子 |
| `remove_skipped_blocks` 放在分配**之前** | 先把滑出窗口的块还回池子，本次分配可直接复用 → 少驱逐别人的缓存 |
| `min(num_skipped_blocks, len(blocks))` | 调用时机在分配前，`total_computed_tokens` 已含"尚未挂上的命中块/external"，`req_to_blocks` 还是旧的 → 不 cap 会 IndexError（新请求 blocks 为空） |
| 倒序循环 + 碰 null 就 break | ① 跳过区单调增长、新的在尾部 → 每次只处理新增，O(新增) 而非 O(跳过区)；② 产出顺序天然符合 `free_blocks` 的驱逐优先级契约 |
| `get_num_skipped_tokens` 基类返回 0 | full attention 从不丢块 → 整套 skip 机制对它是 no-op |
| skip 以"续算起点"为参考 | 后面的 token 还没算、仍要回看命中前缀尾部；用完整长度会误删 |
