# SlidingWindowManager 详解

> `single_type_kv_cache_manager.py` 中 `SlidingWindowManager`（L480-616）—— 滑动窗口注意力的 KV cache 管理策略。本文覆盖它**重写/新增的全部方法**。
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 相关：[find_longest_cache_hit 详解](./find_longest_cache_hit_详解.md)（全类型对比）· [free / remove_skipped_blocks 详解](./free与remove_skipped_blocks_详解.md) · [已知局限与困境](./vLLM_KVCache_已知局限与困境.md)

---

## 〇、它和 FullAttentionManager 的根本区别

滑动窗口：**每个 token 只 attend 最近 `sliding_window` 个 token**，更早的永远看不到。由此带来两个 full attention 没有的能力/负担：

| | FullAttentionManager | **SlidingWindowManager** |
|---|---|---|
| 老 token 的 KV | 留到请求结束 | **滑出窗口即可回收**（省显存的核心） |
| 命中前缀形状 | `[0, k)` 从块 0 起的连续前缀 | `[null × i, 真块 run]` —— run **漂浮在中间**，前面 null |
| 缓存命中判定 | 一 miss 即停 | 攒够 `sliding_window_contiguous_blocks` 连续块 |
| 块表里出现 null | 否 | **是**（滑出窗口的位置用 null 占位） |

**继承的方法**（没重写，直接用基类的）：`get_num_blocks_to_allocate`、`allocate_new_computed_blocks`、`allocate_new_blocks`、`take_new_block_ids`、`cache_blocks`、`free`、`remove_skipped_blocks`（基类版就靠下面的 `get_num_skipped_tokens` 驱动）。

它**重写/新增 4 个**：`__init__`、`find_longest_cache_hit`、`get_num_skipped_tokens`、`get_num_common_prefix_blocks`。

---

## 一、`__init__`（L481-483）

```python
def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
    super().__init__(kv_cache_spec, **kwargs)
    self.sliding_window = kv_cache_spec.sliding_window     # ← 把窗口大小拎出来存成成员
```
只多做一件事：从 spec 里取出 `sliding_window`（token 数，模型架构参数）存为实例属性，方便 `get_num_skipped_tokens` 用。其余全交给基类 `__init__`（三张表、block_size、null_block 等）。

> `SlidingWindowSpec`（kv_cache_interface.py:307）就一个裸字段 `sliding_window: int`。它的 `max_memory_usage_bytes`（L325-329）那个 `+1` 也同源：
> ```python
> # +1 here because the sliding window may not start from the beginning of the block.
> # block size 4, num_token 4 → 需要 [XXCD][EF] 两块存 6-token 窗口 [CDEF]
> return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes
> ```

---

## 二、`get_num_skipped_tokens`（L581-607）—— skip 判据

```python
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return max(0, num_computed_tokens - self.sliding_window + 1)
```
docstring 的图（sliding_window=4, num_computed_tokens=7）：
```
Tokens:   [ 0  1  2  3  4  5  6  7 ]
          | ---- computed -----|
                                 ^ next token to be computed
                       |-----------| sliding window for next token
          |--skipped---|
→ get_num_skipped_tokens(7) == 4
```
- 已算 token 0~6（7 个），下一个要算 token 7，其窗口 = `[4,7]` → token 0~3 永远看不到了 → skip = `max(0, 7-4+1) = 4`。
- **full attention 的基类版恒返回 0**（不丢块）；这里返回真实 skip 数。

**这个 `-（sliding_window - 1）` 和 `find_longest_cache_hit` 里 `cdiv(sliding_window - 1, block_size)` 同源**：窗口含当前 token，而当前 token 现算、不需从缓存取，所以真正"用到的历史"是 `sliding_window - 1` 个。

**谁用它**：基类的 `remove_skipped_blocks`（把滑出窗口的块换 null 并 `free_blocks`）+ `get_num_blocks_to_allocate`（跳过的整块用 null 占位、从"要新申请"里扣掉）。详见 [free / remove_skipped_blocks 详解](./free与remove_skipped_blocks_详解.md) 3 章。

> ⚠️ decode 主战场：滑窗"边解码边回收滑出窗口的老块"就是靠这个函数每步驱动 `remove_skipped_blocks`。**这才是 `sliding_window` 在 decode 时的用武之地** —— 和只在 prefill 首块跑一次的 `find_longest_cache_hit` 是两条独立的线。

---

## 三、`get_num_common_prefix_blocks`（L609-616）—— 直接返回 0

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    """
    NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
    So it's not correct to count ref_cnt like FullAttentionManager. Return
    0 here for correctness. Need to support cascade attention + sliding window in the future.
    """
    return 0
```
**不支持 cascade attention**，直接返回 0。原因：滑窗的前缀块**是 null 块**（滑出窗口的位置被 null 占位），无法像 full attention 那样用 `ref_cnt == len(req_to_blocks)` 去数公共前缀。注释明说 "Return 0 here for **correctness**" —— 保守，只丢优化不影响正确性。

> 对照：Mamba / ChunkedLocal 也 return 0（各自不同原因）。见 [get_num_common_prefix_blocks 详解](./get_num_common_prefix_blocks_详解.md)。

---

## 四、`find_longest_cache_hit`（L485-579）—— 核心，与 full attention 完全不同的算法

### 4.1 签名前的守卫

```python
assert isinstance(kv_cache_spec, SlidingWindowSpec)
assert dcp_world_size == 1, "DCP not support sliding window attn now."   # 🔴 见【已知局限】
assert pcp_world_size == 1, "PCP not support sliding window attn now."
```
`@classmethod`（拿类直接调、无 self），所以一切靠传参、assert 是必要的运行时守卫。**注意：没有 full attention 那句 `block_size *= dcp*pcp`** —— 因为这里已 assert dcp/pcp==1。

### 4.2 门槛 `sliding_window_contiguous_blocks`

```python
sliding_window_contiguous_blocks = cdiv(kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size)
if use_eagle:
    sliding_window_contiguous_blocks += 1
```

**它是什么**：判定命中要凑够的**连续块数**（门槛 `W`），单位 block；由模型的 `sliding_window`（token）推导。**不是**"最大"，是 `>=` 的下限。

**`W = cdiv(sw - 1, B)` 怎么来的**（关键推导，依赖"续算点在块边界"）：
```
续算点 p（恒块对齐，见 4.5）
token p 的窗口 = [p-sw+1, p]（含自己 sw 个）
p 自己现算 → 需从缓存拿 [p-sw+1, p-1] = sw-1 个
这段右端 = p-1 = 块边界的最后一个 token（因 p=k×B）
→ 从块边界往回数 sw-1 个 token，恰好跨 cdiv(sw-1, B) 块
```
**"右端卡块边界"是 cdiv 精确的全部依赖** —— 消灭了"最右块残缺贡献"的问题。

**⚠️ 不会把 4 块算成 3 块**（`cdiv` 向上取整）：
```
B=4, sw=16（=4块）: cdiv(15,4)=⌈3.75⌉=4    ← 还是 4！
减 1 只在 sw ≡ 1 (mod B) 时才少一块，如 sw=17: cdiv(16,4)=4（该省，那多的 1 个 token 是现算的）
```

**eagle 为什么 `+1`**：eagle 结束时要 `pop` 掉最后一块（草稿头需最后 token 的 hidden state，缓存里只有 KV）。滑窗的命中 run 漂浮在中间，pop 一块 → 续算点左移一块 → 新窗口需要**更左边那一块**，而算法原本没验证它。`+1` 提前多验证一块，保证 pop 后**仍有 W 块连续、已验证的缓存**罩住新窗口。（full attention 不需要：其命中是 `[0,k)`，pop 后仍是从 0 起的完整前缀。）
```
sw=8,B=4,原W=2。blocks: [miss,hit,hit,hit,miss,hit]（idx0-5）
  不+1（错）: run=[2,4)→pop→[null,null,blk2]，续算点左移到 token12，窗口需 blk1，但 idx1 从没查过 → 可能踩空💥
  +1（对）:  找 3 连续 → run=[1,4)→[null,blk1,blk2,blk3]→pop→[null,blk1,blk2]，token12 窗口需 blk1,blk2，都验证过 ✓
```

### 4.3 TODO：跳步优化

```python
# TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to optimize
# from O(max_num_blocks) to O(max_num_blocks / sliding_window_contiguous_blocks + ...)
```
现在逐块退，O(n)。优化思路（类似 Boyer-Moore 坏字符跳跃）：块 i 是 miss → 任何合法 run 不可能含它 → 直接把 i 跳到 `i-W` 而非 `i-1`。命中率低时约 `n/W` 次探测 → O(n/W + W)。**尚未实现**（收录在[已知局限]性能 TODO）。

### 4.4 初始化：预填 null + 按下标赋值 ★

```python
max_num_blocks = max_length // kv_cache_spec.block_size
computed_blocks = tuple(
    [block_pool.null_block] * max_num_blocks         # ← 预先【全填 null】
    for _ in range(len(kv_cache_group_ids))          # 每个 group 一个
)
```
vs full attention 的 `tuple([] for ...)` + `append`。滑窗**允许中间 miss**，结果形如 `[NULL, NULL, blk8, blk3]`，所以**预填满 null，再按下标 `computed[i] = cached` 赋值**（右→左扫，append 会反）。

**为什么这里 `[null_block] * max_num_blocks` 没有 `[[]] * N` 那个坑**：
```
[[]] * 3        → 三个槽是【同一个列表】的引用，append 会串 ❌
[null_block]*n  → 所有槽指向【同一个 null_block】，但只做 computed[i]=cached【替换】，从不原地改 null → 安全 ✓
```
判据：`[x] * n` 安全 ⟺ 只**替换**元素、不**原地修改**元素。而 null_block 本就是全局唯一共享单例（`free_blocks` 里 `not block.is_null` 等后门印证：不入队、不计额度、不登记 hash）。

### 4.5 主循环：右→左，攒够 W 连续块（L529-555）

```python
num_contiguous_blocks = 0
match_found = False
for i in range(max_num_blocks - 1, -1, -1):          # ★ 从右往左
    if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
        # LCM 对齐检查：起头的块右边缘必须对齐到 alignment_tokens（混合模型）
        if (num_contiguous_blocks == 0
                and block_size != alignment_tokens
                and (i + 1) * block_size % alignment_tokens != 0):
            continue                                  # 不从这块起头
        for computed, cached in zip(computed_blocks, cached_block):
            computed[i] = cached                      # 按下标填真块
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= sliding_window_contiguous_blocks:
            for computed in computed_blocks:
                del computed[i + num_contiguous_blocks:]   # 裁掉尾巴
            match_found = True
            break
    else:
        num_contiguous_blocks = 0                     # 断了就重新数
```
- **右→左找到第一个 W 连续 run 就 break** → 是**最右/最长**的续算点 = 能白嫖最远。
- `computed[i] = cached` 直接按位置赋值（预填 null 的好处）。
- `del computed[i+num:]` 裁掉 run 右边多余的（例：`[NULL,NULL,8,3,NULL,9] → [NULL,NULL,8,3]`，见 L548 注释）。
- L536-541 的 LCM 对齐检查：只在"开始新 run"（`num_contiguous_blocks==0`）时，要求这块右边缘对齐到 `alignment_tokens`，保证续算点对每个 group 都是整块（混合模型；单 group `block_size==alignment_tokens` 短路跳过）。

### 4.6 两个出口

**出口①（找到完整窗口，L546-553）**：`del computed[i+num:]` 裁尾 + `match_found=True` + break。

**出口②（扫完没凑够，L556-560）**：
```python
if not match_found:
    # The first `num_contiguous_blocks` is a cache hit even if
    # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
    for computed in computed_blocks:
        del computed[num_contiguous_blocks:]         # 只留前 k 个真块
```
**为什么不够 W 块也算命中**：能一路扫到 i=0 没 break，说明**块 0~k-1 全命中**（miss 会清零）。而**块 0 没有前驱** —— 从块 0 起的前缀不存在"左边缺一块"的问题，所以哪怕 k < W 也是有效前缀命中。`k=0`（块 0 都没中）则 `del [0:]` 真清空 = 无命中。

### 4.7 对齐 while + eagle pop（L561-578）

```python
# 出口②里的对齐修正（和出口①的公共逻辑分开写）
while (block_size != alignment_tokens
       and len(computed_blocks[0]) * block_size % alignment_tokens != 0):
    for computed in computed_blocks:
        computed.pop()

# eagle 真正丢最后一块（+1 是在 4.2 抬门槛，这里才 pop）
if use_eagle and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()
    # Re-align after eagle pop（pop 可能又破坏 LCM 对齐，如 Gemma4）
    while (block_size != alignment_tokens
           and len(computed_blocks[0]) * block_size % alignment_tokens != 0):
        for computed in computed_blocks:
            computed.pop()
```
- **对齐 while 和裁尾是两码事**：裁尾定"命中到哪"，对齐 while 把命中长度削到 `alignment_tokens`（LCM）整数倍（防部分块命中）。单 group 短路，**Qwen 永不执行**。
- **eagle pop** 是 4.2 那个 `+1` 的兑现：门槛抬高一块保证多验证一块，这里 pop 掉最后一块。pop 后可能再破坏对齐 → 再 re-align。

---

## 五、返回值语义：它是"块表"，不是"命中块集合" ★

返回 `[null, null, blk2, blk3]`（长度 4，非 W=2）—— 为什么长度 4？

**调用方用列表长度算命中长度**（`kv_cache_coordinator.py:365`）：
```python
return hit_blocks, len(hit_blocks[0]) * self.block_size    # 命中长度 = 列表长度 × block_size
```
```
len=4 → num_computed_tokens = 4×4 = 16 → 续算从 token 16 开始
下标 i ↔ token [i×B, (i+1)×B)：
  [ null   null  | blk2   blk3 ]
   tok0-3 tok4-7  tok8-11 tok12-15
   已滑出窗口       窗口内真 KV
```
- **列表长度 = i(null) + W(真块)**：W 由 sw 定死（=2），i 由缓存内容浮动 → **长度和 sw 没有直接关系，取决于 run 在哪**。
- **null 不是多此一举**：块表**按下标↔token 位置**对齐，删掉前面 null 会让后面全部错位、读到错的 KV；且长度编码了 `num_computed_tokens`，删了就等于宣称"这些 token 没算过" → 重算。
- **为什么用 null 而非真块占位**：null 是全局共享单例，**不占显存、不消耗空闲额度、不 touch** —— 零成本占位。窗口外的 token 用不到真 KV，用真块反而白白 `touch`、吃 `num_evictable_blocks` 额度。

---

## 六、`find_longest_cache_hit` 只在 prefill 首块跑，decode 不碰 ★

反复澄清的点：
```
scheduler.py:610  if request.num_computed_tokens == 0:   ← 唯一闸门（WAITING/prefill 首块）
                      get_computed_blocks → find_longest_cache_hit
scheduler.py:463  decode 走 allocate_slots，【不传 new_computed_blocks】→ 快路径 assert 为空
```
- **decode 时 token 落在块中间完全正常**：由 `allocate_new_blocks` + 逐槽寻址 `slot = block_id×B + offset` 处理，无任何对齐要求。
- `cdiv(sw-1, B)` 那行 **decode 时根本不执行**，所以不存在"decode 时 token 在块中间 cdiv 算不准"的问题 —— 那行只在 prefill 查缓存时跑，而缓存只存整块 → 续算点天然块对齐。

**`sliding_window` 的三条独立下游线**（别混）：
| 用途 | 用哪个量 | 何时 |
|---|---|---|
| 实际注意力计算 | `sliding_window` 直接给 kernel | **每步**（prefill+decode） |
| 回收滑出窗口的块 | `get_num_skipped_tokens`（本文三章） | **每次 allocate_slots**（decode 主战场） |
| 查前缀缓存命中 | `cdiv(sw-1, B)`（本文四章） | **仅 prefill 首块一次** |

---

## 七、关键设计点速查

| 设计点 | 原因 |
|---|---|
| `__init__` 存 `self.sliding_window` | 供 `get_num_skipped_tokens` 用 |
| `get_num_skipped_tokens = max(0, n - sw + 1)` | skip 掉滑出窗口的老 token；`-(sw-1)` 因窗口含当前 token（现算） |
| `get_num_common_prefix_blocks` 返回 0 | 前缀块是 null，无法数 ref_cnt；不支持 cascade（保守，不影响正确性） |
| `find_longest_cache_hit` 右→左 + 预填 null | 滑窗只需最后一窗、允许中间 miss；null 占位保持位置对齐 |
| `W = cdiv(sw-1, B)` 是门槛 | 覆盖续算点窗口所需的连续块数；续算点块对齐 → cdiv 精确 |
| eagle `+1` 再 pop | pop 让续算点左移一块，`+1` 提前多验证一块补上 |
| `[null_block] * n` 安全 | 只做 `computed[i]=cached` 替换、不原地改 null；null 是共享单例 |
| 出口② 不够 W 也命中 | 块 0 无前驱，从 0 起的前缀天然有效 |
| 对齐 while | 混合模型削到 LCM 整数倍防部分块命中；单 group 短路 |
| 返回列表长度 = i + W | 编码 `num_computed_tokens`；null 占位不可删（否则块表错位/重算） |
| 🔴 assert dcp/pcp == 1 | 窗口丢块 × 跨卡分片组合未实现（见已知局限） |
