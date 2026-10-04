# vLLM v1 KV Cache 已知局限、未实现项与困境

> 记录读 `vllm/v1/core/` 过程中遇到的**当前版本尚未实现、刻意保守、或存在已知取舍**的地方。每条都附代码位置，便于核对与日后回看是否已被修复。
>
> 相关：[single_type_kv_cache_manager 解析](./single_type_kv_cache_manager_解析.md) ·
> [allocate_new_computed_blocks 详解](./allocate_new_computed_blocks_详解.md) ·
> [allocate_new_blocks / cache_blocks 详解](./allocate_new_blocks与cache_blocks_详解.md)
>
> ⚠️ 代码在快速演进，行号会漂移，措辞里的 "now / yet" 说明是**阶段性**限制。

三类标记：
- 🔴 **硬性不支持**：`assert` / `raise` 直接挡住，触发就崩
- 🟡 **保守但安全**：为回避复杂情况故意"少做优化"，**不影响正确性**，只损失性能
- 🔵 **性能 TODO / 取舍**：已知可优化但暂未做

---

## 一、🔴 上下文并行（DCP / PCP）只支持纯 full attention

这是最成体系的一处限制。DCP（Decode Context Parallelism）/ PCP（Prefill Context Parallelism）把一条序列的 token 切到多卡，但**只对"保留全部 KV"的 full attention 实现了**，凡是"中途按窗口丢 KV"或状态特殊的类型全被挡住：

| 被挡的类型 | 代码位置 | 断言消息 |
|---|---|---|
| hybrid（混合模型协调器） | `kv_cache_coordinator.py:406-407` | `DCP/PCP not support hybrid attn now.` |
| sliding window | `single_type_kv_cache_manager.py:501-502` | `DCP/PCP not support sliding window attn now.` |
| chunked local attention | `single_type_kv_cache_manager.py:679-680` | `DCP/PCP not support chunked local attn now.` |
| mamba | `single_type_kv_cache_manager.py:800-801` | `DCP/PCP not support mamba now.` |

**对比**：`FullAttentionManager.find_longest_cache_hit`（L433）**没有**这个 assert，且用 `block_size *= dcp*pcp`（`__init__` L50-54）把物理块折叠成"逻辑大块"来适配 CP。MLA 走 FullAttentionManager，所以 **MLA + CP 也支持**。

**为什么难（不是原理冲突，是工程未实现——注意措辞是 "now"）**：
- CP 把 block_size 抬成横跨 N 卡的"逻辑大块"；sliding window / chunked local 的价值在于**按 token 位置整块释放**滑出窗口的块。窗口边界通常**不对齐**每张卡的物理块边界。
- CP 把序列 token 打散到多卡后，每张卡"窗口内该保留的 token"是个**跳步子集**，破坏了窗口注意力赖以简洁的"跳过的永远是从头连续前缀块"不变量（`remove_skipped_blocks`、`find_longest_cache_hit` 右到左扫描都依赖它）。
- full attention **从不中途丢块**（留到请求结束），块集合单调增长，CP 只需分片，所以没有这个矛盾。

> 结论：**"边解码边按窗口丢 KV" × "按 token 跨卡分片"** 两套记账法组合起来正确实现太琐碎，vLLM 暂时 assert 挡掉。纯 full attention（如 Qwen2.5-0.5B）不受影响。

另外 `scheduler.py:261`：`enable_return_routed_experts does not support context parallelism` —— MoE 路由专家返回也和 CP 不兼容。

---

## 二、🟡 公共前缀块统计会被"未被调度但持块"的请求污染

`get_num_common_prefix_blocks`（用于 **cascade attention**：多请求共享同一段前缀时只算一遍，省显存带宽）。

**缺陷**（`kv_cache_manager.py:484-498` docstring 亲述）：
> The number of requests with allocated KV cache is **greater than or equal to** the number of requests scheduled in the current step... This can result in an edge case where the number of common prefix blocks is **0, even though all scheduled requests share a common prefix**... Currently, this case **cannot be easily detected, so the function returns 0**.

**根因**：判据是 `block.ref_cnt == len(self.req_to_blocks)`（`single_type_kv_cache_manager.py:473`）。`len(req_to_blocks)` 数的是"**持有块的请求**"，在 continuous batching 下 **≥ "本步被调度的请求"**。那些还在跑、块没释放、但这一步没排上（token_budget 用尽等，`scheduler.py:442-458`）的"旁观者"，如果不共享前缀，就会让 `ref_cnt` 凑不齐 → 返回 0。

**为什么不修**：manager 这层只有 `req_to_blocks`，**看不见 scheduler 的调度决策**；`ref_cnt` 是全局计数、不区分本步是否被调度。要修就得把"本步调度集合"传进来、放弃 `ref_cnt == len` 这个 O(1) 妙招、改成遍历求交集，复杂度和接口都涨。

**代价**：仅**错过 cascade 优化**（本可共享前缀却没识别出来），**不影响正确性** —— 返回 0 = 走普通注意力，是 fail-safe。

---

## 三、🔴 cascade attention 的类型限制

不是所有 attention 都支持 cascade：

| 类型 | 代码位置 | 行为 |
|---|---|---|
| sliding window | `single_type_kv_cache_manager.py:609-616` | `get_num_common_prefix_blocks` 直接 `return 0`（"The prefix blocks are **null blocks** for sliding window layers. So it's not correct to count ref_cnt... Return 0 here for correctness. Need to support cascade attention + sliding window in the future."）|
| mamba | `single_type_kv_cache_manager.py:862-866` | 同上，"cascade attention is not supported by mamba" |
| chunked local attention | `single_type_kv_cache_manager.py:762-764` | 同上，"not supported by chunked local attention" |

**各自不同的原因**：
- **sliding window**：前缀位置全是 `_null_block`（全局共享单例），它的 `ref_cnt` 是被无数位置引用的无意义巨值 → `ref_cnt == len(req_to_blocks)` 会误判；更本质地，滑窗前缀没有"一段共享的真实 KV"供 cascade 复用。详见 [SlidingWindowManager 详解](./SlidingWindowManager_详解.md) 三章。
- **mamba**：SSM 状态无法像 KV 那样按前缀分解。
- **chunked local**：分块局部注意力的前缀语义同样不适配。

> 注意区分：这三类 return 0 是 🔴 **该类型不支持 cascade**（硬性）；而[二章]那个"未调度请求污染"导致的 return 0 是 🟡 **支持但没识别出来**（保守）。

---

## 四、🔴 prefix caching 的适用边界

**① cross attention 完全不支持 prefix caching**
`CrossAttentionManager.cache_blocks`（`single_type_kv_cache_manager.py:1088`）：
```python
raise NotImplementedError("CrossAttentionManager does not support caching")
```
enc-dec 的 encoder KV 对每个请求唯一，没有跨请求复用价值。

**② 不支持"部分块命中"（partial block cache hit）**
`kv_cache_coordinator.py:445-451`：
> The cache hit length must be a multiple of the **LCM of the block sizes**... Requiring this because **we don't support partial block cache hit yet**.

混合模型里不同 attention 类型 block_size 可能不同，命中长度必须同时是各类型 block_size 的整数倍 → 取最小公倍数 → 命中只能停在整块，不能停在半块（这就是 `alignment_tokens` 那个 while 循环在削的东西）。

**③ eagle + chunked local attention 的混合 KV 不支持**
`single_type_kv_cache_manager.py:676-678`：
```python
assert use_eagle is False, "Hybrid KV cache is not supported for eagle + chunked local attention."
```

**④ 可选的"完全不做 prefix caching"协调器**
`KVCacheCoordinatorNoPrefixCache`（`kv_cache_coordinator.py:256-261`）："Does not implement any features related to prefix caching." —— 这是一种**模式**（禁用前缀缓存时用），不是 bug，但记一笔：它的 `find_longest_cache_hit` 直接返回空 + 命中 0（L291-299）。

---

## 五、🟡 Mamba "align" 模式对投机解码的块数**高估**

`single_type_kv_cache_manager.py:908-909 / 953-954`（`NOTE(tdouble)`）：
> this is an **over-estimate** of how many blocks we need because num_tokens can include **draft tokens that will later be rejected**.

Mamba align 模式为了保持 block 对齐，用 `num_tokens_main_model`（不含投机）算主分配，投机部分单独按 `num_speculative_blocks` **预留**。因为草稿 token 可能被拒，这个预留是**偏多估计**（宁可多留也不让它破坏对齐）。相关注释 L901-905 / L946-950：
> if x * block_size tokens are scheduled, num_tokens is x*block_size + num_lookahead_tokens and **breaks the alignment**. We can ignore lookahead tokens because **current draft models don't have mamba layers**.

即：这套绕法能成立，还依赖一个前提——**当前的草稿模型不含 mamba 层**（lookahead token 不产生 mamba state）。若未来出现带 mamba 的草稿模型，这里要重做。

---

## 六、🔵 性能 TODO / 已知可优化

| 位置 | 内容 |
|---|---|
| `single_type_kv_cache_manager.py:516-520` | sliding window 的 `find_longest_cache_hit` 现在是 O(max_num_blocks) 线性右到左扫描；TODO：cache miss 时按 `sliding_window_contiguous_blocks` 跳步，降到 O(n / sw_blocks + sw_blocks)，对低命中率场景有利 |
| `scheduler.py:2183` | `TODO (davidb): add support for hybrid memory allocator`（某条路径尚未支持混合内存分配器）|
| `scheduler.py:2206 / 2234` | KV 加载路径 "loading **does not yet support block sharing**"（前缀共享块的加载暂不支持）|
| `scheduler.py:1902` | 某场景 "which is **not supported yet**" |
| `kv_cache_utils.py:942` | 无法统一 page size 时 `raise NotImplementedError`（不同类型层的页大小凑不齐就直接放弃）|
| `kv_cache_utils.py:1554` | 某跨 worker 情形 "This is **not supported yet**" |
| `block_pool.py:101` | `TODO(Jialin)`：`get_cached_block` 里 key 命中时 block_id 的一致性小 TODO |

---

## 七、一句话地图

- **CP（DCP/PCP）**：只跟纯 full attention（含 MLA）好使；sliding window / chunked local / mamba / 混合模型全部 🔴 挡住 —— 因为"按窗口中途丢 KV"和"按 token 跨卡分片"组合难。
- **cascade attention**：mamba、chunked local 🔴 不支持；即使支持的类型，统计也可能被"未调度但持块"的请求 🟡 保守地算成 0（安全，只丢优化）。
- **prefix caching**：cross attention 🔴 完全不支持；🔴 不支持部分块命中（要 LCM 对齐）；eagle + chunked local 🔴 不支持。
- **投机解码 × mamba align**：块数 🟡 高估，且依赖"草稿模型不含 mamba 层"的前提。
- 其余多为 🔵 性能 TODO 与边角未实现。

> 对 **Qwen2.5-0.5B（纯 full attention、单 group、未开 CP）** 而言：本文绝大多数限制都**不触发** —— 走的是 `UnitaryKVCacheCoordinator` + `FullAttentionManager` 这条最成熟、限制最少的路径。这些限制主要影响**混合模型、上下文并行、投机解码**等进阶场景。
