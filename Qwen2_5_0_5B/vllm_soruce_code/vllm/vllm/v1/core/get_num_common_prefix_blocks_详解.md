# get_num_common_prefix_blocks 详解（cascade attention）

> 记录 `single_type_kv_cache_manager.py` 的 `get_num_common_prefix_blocks`：基类抽象声明（L293-307）+ 各子类实现（Full L469 / SlidingWindow L609 / ChunkedLocal L762 / Mamba L862 / Cross L1061）。
>
> - 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)
> - 相关：[已知局限与困境](./vLLM_KVCache_已知局限与困境.md)（本函数的保守缺陷收录在那里）

---

## 一、为什么函数体只有一句话：它是 `@abstractmethod`

```python
@abstractmethod
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    """..."""
    raise NotImplementedError        # L307
```
基类**只声明接口、不实现**。Python 的 ABC 机制下，子类不实现它就**无法实例化**。真正的逻辑在各子类：

| 子类 | 行号 | 实现 |
|---|---|---|
| `FullAttentionManager` | L469 | 真正的算法（见下） |
| `SlidingWindowManager` | L609 | 滑窗版 |
| `ChunkedLocalAttentionManager` | L762 | 直接 `return 0`（"cascade attention is not supported by chunked local attention"） |
| `MambaManager` | L862 | 直接 `return 0`（"cascade attention is not supported by mamba"） |
| `CrossAttentionManager` | L1061 | enc-dec 版 |

> 对比：同在基类的 `get_num_skipped_tokens`（L401）**有默认实现**（返回 0）—— 那是"虚方法+默认行为"；本函数是**纯抽象**、必须子类实现。两种不同的设计选择。

---

## 二、它算什么：所有 running 请求**共享的前缀有多少块**

`FullAttentionManager` 的实现（L469-477）非常精巧：
```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    blocks = self.req_to_blocks[running_request_id]      # 任选一个请求的块表
    num_common_blocks = 0
    for block in blocks:                                  # 从前往后走
        if block.ref_cnt == len(self.req_to_blocks):      # ★ 所有请求都在用它
            num_common_blocks += 1
        else:
            break                                         # 一旦不共享，立刻停
    return num_common_blocks
```

### 2.1 `ref_cnt == len(self.req_to_blocks)` 是核心判据

- `len(self.req_to_blocks)` = 当前**持有块的请求总数**
- `block.ref_cnt` = **有多少个请求引用这块**
- 两者相等 ⟺ **每一个请求都在用它** ⟺ 它是公共前缀的一部分

`ref_cnt` 在这里被当成"**有多少人共享我**"的计数器直接用上 —— 不用遍历所有请求的块表做交集，**O(前缀长度)** 就搞定。

### 2.2 遇到不共享就 `break`

因为公共前缀必然是**从块 0 开始的连续段**（前缀缓存的链式 hash 保证，和"命中必为连续前缀"同理）。

### 2.3 为什么参数叫 `running_request_id` 却能"任选一个"

docstring（`kv_cache_manager.py:479`）写明 "The function **selects a running request** and iterates through its blocks"；scheduler 传的就是 `any_request_id`（L871-872）。因为**共享块在每个请求的列表里都有**，从谁开始走结果都一样。

---

## 三、干什么用的：**cascade attention**

`MambaManager` 的注释直接点名（"cascade attention is not supported by mamba"），`源码导读.md` L166 也记了。

**cascade attention 是什么**：当**所有 running 请求共享同一段前缀**（最典型：同一个 system prompt、同一批 few-shot 示例），不必让每个请求各自去读一遍那段共享 KV，而是：
1. 对**整批 query** 和**共享前缀**算**一次**注意力
2. 每个请求再各自算**自己后缀**的注意力
3. 用 log-sum-exp / online-softmax 把两部分**合并**

**为什么值得**：decode 是**显存带宽瓶颈**。
```
50 个请求共享一个 2000-token 的 system prompt
朴素做法:  每个请求各读一遍那 2000 token 的 KV → 50× 显存流量
cascade:   读一次、对整批算一次，再和各自后缀合并 → 流量降到约 1/50
```

### 下游链路

```
scheduler.py:867-872   num_common_prefix_blocks = kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
scheduler.py:921    →  SchedulerOutput.num_common_prefix_blocks   (output.py:205，list[int]，每 group 一个)
gpu_model_runner.py:3874 → 传给 attention metadata builder
gpu_model_runner.py:2412 → common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
                        → attention backend 据此决定要不要走 cascade
```
注意 coordinator 层（`kv_cache_coordinator.py:201-214`）返回的是 **`list[int]`，每个 group 一个数**。

---

## 四、🟡 一个 docstring 亲述的保守缺陷

`kv_cache_manager.py:484-498`：
> The number of requests with allocated KV cache is **greater than or equal to** the number of requests scheduled in the current step... This can result in an edge case where the number of common prefix blocks is **0, even though all scheduled requests share a common prefix**... Currently, this case **cannot be easily detected, so the function returns 0**.

**两个集合的区别**（continuous batching 下）：
```
┌─────────────────────────────────────────┐
│  持有块的请求 (req_to_blocks 里有条目)     │  ← 没结束、块没被 free
│   = len(self.req_to_blocks)             │
│  ┌───────────────────────────┐          │
│  │  本步被调度的请求            │   D ←────┼── 旁观者：还在跑，但这步 budget 不够，没排上
│  │      A   B   C            │          │
│  └───────────────────────────┘          │
└─────────────────────────────────────────┘
        调度的 ⊆ 持有块的
```
"持有块但没被调度"的来源（`scheduler.py`）：
- L385 `while req_index < len(self.running) and token_budget > 0:` —— **token_budget 用尽**，后面的 running 请求这步不算，但**块还在**
- L442-458 显式跳过：PP>1 已排完 prompt / 异步调度已达 max_model_len / encoder budget 耗尽 / encoder cache 耗尽 / mamba align 块对齐 budget 不够

**缺陷现场**：
```
req_to_blocks 里 4 个请求（都持有块）:
  A, B, C —— 本步被调度，三者共享同一个 system prompt 前缀（block X）
  D       —— 本步没排上（token_budget 用完），不共享 block X

block X 的 ref_cnt      = 3      （只有 A、B、C）
len(self.req_to_blocks) = 4      （A、B、C、D 都持有块）
3 != 4 → 第一个块就 break → 返回 0        ← 白白错过 cascade
```

**为什么不修**：manager 这层**只有 `req_to_blocks`，看不见 scheduler 的调度决策**；`ref_cnt` 是全局计数、不区分本步是否被调度。要修就得把"本步调度集合"传进来 → 放弃 `ref_cnt == len` 这个 O(1) 妙招 → 改成遍历求交集，复杂度从 O(前缀长度) 涨到 O(请求数 × 前缀长度)，还要改一堆接口。

**代价**：仅**错过优化**。返回 0 = 没有公共前缀 = 走普通注意力路径，**不影响正确性** —— 典型的 fail-safe。

> ⚠️ 注意区分本函数的两个"0"：
> - **Mamba / ChunkedLocal 返回 0** = 🔴 **该类型不支持 cascade**（硬性）
> - **本缺陷返回 0** = 🟡 **支持但没识别出来**（保守）

---

## 五、关键设计点速查

| 设计点 | 原因 |
|---|---|
| 纯 `@abstractmethod` | 各 attention 类型的"公共前缀"语义差异大，必须定制；基类只给接口 |
| `ref_cnt == len(req_to_blocks)` 判据 | 把 `ref_cnt` 当"有多少人共享我"的计数器 → O(前缀长度) 搞定，无需遍历所有请求求交集 |
| 遇到不共享就 `break` | 公共前缀必为从块 0 起的连续段（链式 hash 保证） |
| 参数"任选一个 running 请求" | 共享块在每个请求的列表里都有，从谁开始走都一样 |
| Mamba / ChunkedLocal 直接 return 0 | SSM 状态 / 分块局部注意力无法按前缀分解，不支持 cascade |
| 未调度请求污染 → 保守返回 0 | manager 看不见调度决策；返回 0 只丢优化不影响正确性 |

**对 Qwen2.5-0.5B**：纯 full attention 走 L469 的真实现。若你并发跑多个共享同一 system prompt 的请求，这里会算出正的公共前缀块数，attention backend 就可能启用 cascade 省带宽 —— 前提是没有"旁观者请求"把它污染成 0。
