# allocate_new_computed_blocks 详解

> 本文专门记录 `single_type_kv_cache_manager.py` 中 `allocate_new_computed_blocks`（L142-213）这一个函数：先给整体定位与逐行拆解，再把围绕它衍生出的一连串细节问题（local/external 来源与顺序、skip、min 削减、cdiv 相减、new_block_ids 清零、清零 vs mask）逐一辨析。
>
> 返回上层总览：[single_type_kv_cache_manager.py 详细解析](./single_type_kv_cache_manager_解析.md)

---

## 一、一句话定位

> **把"已经算好 KV 的块"挂到这个请求的 block table 上，并锁住不让它们被驱逐。**

函数名有迷惑性 —— 它**不负责给"要新算的 token"分配块**，那是 `allocate_new_blocks`（L215）干的。它只处理"已经算好的部分"。

### "computed blocks" 的两个来源

一个新请求进来，prompt 里有一段 token 的 KV **早就算好了**，不用重算，只要把对应块接过来。来源有二：

| 类型 | 含义 | 数据在哪 | 处理方式 |
|---|---|---|---|
| **local**（`new_computed_blocks`，块列表）| 前缀缓存命中：块已在 block_pool、KV 已填好 | 已在本卡 GPU KV cache | `touch` 后**直接挂载**，不分配新块 |
| **external**（`num_external_computed_tokens`，只是个数字）| KV connector 传来的（如 P/D 分离、LMCache）| 在远端/外部存储 | **分配空块**当落地容器，等 KV 搬进来 |

而"还没算、这一步要现算"的后缀 token → 归 `allocate_new_blocks` 管，**不在本函数**。这是最容易混的点。

### 调用时机

调度器决定收这个请求后，先用 `get_num_blocks_to_allocate` 查容量够不够，再调本函数**真正落地这些已算好的块**。二者是"预演/执行"的镜像。

---

## 二、逐行拆解（L142-213）

### ① 快路径：running 请求直接返回（L165-169）
```python
if request_id in self.num_cached_block:
    assert len(new_computed_blocks) == 0
    return
```
`num_cached_block` 里有它 = **正在运行的老请求**。运行中的请求不可能再产生新的前缀命中（块早分配好了），所以 `new_computed_blocks` 必为空，直接返回。

### ② 新请求，取空块表（L172-176）
```python
req_blocks = self.req_to_blocks[request_id]   # 全新请求，此时是空 []
assert len(req_blocks) == 0
num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
```

### ③ 处理"跳过的块"——仅滑窗/局部注意力有意义（L177-187）
```python
num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
num_skipped_blocks = num_skipped_tokens // self.block_size
if num_skipped_blocks > 0:
    new_computed_blocks = new_computed_blocks[num_skipped_blocks:]          # 砍掉命中块前段
    num_external_computed_tokens = min(                                     # external 也随之削减
        num_total_computed_tokens - num_skipped_tokens,
        num_external_computed_tokens,
    )
```
- **full attention**：`get_num_skipped_tokens` 返回 0（父类默认，L401），整段空操作。
- **sliding window**：窗口外的老 token 的 KV 用不到了，砍掉命中块前段（后面用 null 占位）。详见下方 **3.4、3.5 节**。

### ④ touch，锁住不被驱逐（L189-195）
```python
if self.enable_caching:
    self.block_pool.touch(new_computed_blocks)
```
`touch` 把命中块从空闲可驱逐队列拿出 / 增加引用计数，保证本请求用它期间不被抢走。

### ⑤ 组装 block table（L197-204）—— 核心
```python
req_blocks.extend([self._null_block] * num_skipped_blocks)  # 被跳过的位置用 null 占位
req_blocks.extend(new_computed_blocks)                       # 接上真实命中块
self.num_cached_block[request_id] = len(req_blocks)
```
- `_null_block` 占位是为了**位置对齐**：第 i 个逻辑块必须对应第 i 段 token；滑窗虽不需老块真实 KV，但不能让后面的块前移错位。全注意力时 `num_skipped_blocks=0`，这行不产生任何东西。
- `num_cached_block[request_id] = len(req_blocks)`：记下"前 N 块都是已缓存的（已有 block_hash）"。作用：(a) 之后 `cache_blocks()` 跳过这些块不重复算哈希；(b) 该请求转为"运行中"，下次走①快路径。

### ⑥ 给 external 分配落地块（L206-213）
```python
if num_external_computed_tokens > 0:
    allocated_blocks = self.block_pool.get_new_blocks(
        cdiv(num_total_computed_tokens, block_size) - len(req_blocks)
    )
    req_blocks.extend(allocated_blocks)
    if type(self.kv_cache_spec) is FullAttentionSpec:
        self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```
只有走 KV connector 才进这里。external 的 KV 在别处，需要空块接收。为什么这里的相减不是 0、`new_block_ids` 那行为什么要判断类型，见下方 **3.6、3.7 节**。

### 一个具体例子（全注意力，block_size=16）
```
新请求 prompt 100 token，前缀命中前 48 个（3 块）：
  new_computed_blocks=[B0,B1,B2], local=48, external=0
→ skipped=0（全注意力）
→ req_blocks: [] → extend [B0,B1,B2]
→ touch(B0,B1,B2) 锁住
→ num_cached_block[req]=3
→ external=0，跳过⑥
剩下 100-48=52 个 token 需现算 → 由后续 allocate_new_blocks 分配空块。
```

---

## 三、细节辨析（Q&A）

### 3.1 `new_computed_blocks` 只在 prefill 首次分配才可能非空（decode 恒空）

**Q：decode 每步生成一个 token，也会有 new_computed_blocks 吧？**

不会。`new_computed_blocks` 的 "computed" 不是"这一步刚算出来的"，而是"**在本请求需要它之前就已算好、躺在缓存里、被前缀匹配查中的**"—— 是 cache **hit**，不是 just computed。

调度器分两条路径（scheduler.py）：

**① WAITING 队列（新请求首次 = prefill）** — `scheduler.py:610`
```python
if request.num_computed_tokens == 0:
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)   # ← 只有这里做前缀匹配
    )
...
allocate_slots(request, num_new_tokens, new_computed_blocks=new_computed_blocks, ...)  # 传进去
```

**② running 队列（decode / 续 prefill）** — `scheduler.py:463`
```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens,
    # 根本没有 new_computed_blocks 参数 → 默认 None → 空
)
```

decode 时请求已在 running、已在 `num_cached_block` 里 → 触发快路径 `assert len(new_computed_blocks) == 0`。decode 生成的 token KV 是**现算的**，写进已分配槽位；块写满时靠 `allocate_new_blocks` 拿**空块**（叫 "new blocks"，不叫 "new computed blocks"）。

| 术语 | 含义 | 谁负责 | 何时出现 |
|---|---|---|---|
| **new_computed_blocks** | 别人/之前算好、前缀命中的块 | `allocate_new_computed_blocks` | 仅 prefill 首次调度 |
| **new_blocks** | 给"这步要现算的 token"的**空**块 | `allocate_new_blocks` | prefill 和 decode 都有 |

**精确表述**：`new_computed_blocks` 只在该请求**第一次被 `allocate_slots`（第一个 prefill chunk、`request_id` 还没进 `num_cached_block`）才可能非空**，且还得真前缀命中才非空（没命中就是空列表但仍走慢路径）。之后所有 prefill 续块和全部 decode 都走快路径、恒空。分块 prefill 的后续 chunk 也不再有。被抢占重排的请求会被重置 `num_computed_tokens=0` 并移出 `num_cached_block`，恢复时算一次"新的首次分配"，可再次命中 —— 仍是重新 prefill 的首块，不破坏规则。

---

### 3.2 local / external computed tokens 到底是什么、怎么来的

**Q：`num_local_computed_tokens + num_external_computed_tokens` 是什么？和这个"还没 prefill 的 req"什么关系？**

核心反直觉点：这个新请求**自己**一个 token 都还没算，但它 prompt 的**前一段**在**别的请求/别的节点**那里早就算过、KV 还在缓存里。内容相同 → KV 相同 → 不用重算，直接接过来。这就是 **prefix caching**。

- **`num_local_computed_tokens`**（`scheduler.py:612` → `get_computed_blocks()` → `kv_cache_manager.py:202`）：
  ```python
  computed_blocks, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(
      request.block_hashes, max_cache_hit_length
  )
  ```
  把 prompt 逐块算哈希（`request.block_hashes`），拿去全局"已缓存块哈希表"查，从头连续匹配多少块就命中多少。KV **此刻就在本卡 GPU**，`= 命中块数 × block_size`（block 对齐）。复用方式：`touch` 后直接挂，不分配新块。

- **`num_external_computed_tokens`**（`scheduler.py:617`）：
  ```python
  ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
      request, num_new_local_computed_tokens
  )
  ```
  KV connector（LMCache / P/D 分离）汇报"我还能再提供多少前缀 token 的 KV"。KV **在远端**，需要**分配空块**接收。

**为什么算 `num_total_computed_tokens`**：它 = 这个请求可以"免算"的前缀总长度，直接决定三件事：
1. **prefill 只跑剩下的后缀**（`scheduler.py:665`：`num_new_tokens = request.num_tokens - num_computed_tokens`）—— 这就是它和"还没 prefill 的 req"的关系：不是从 0 prefill 整个 prompt，而是从第 `num_total_computed_tokens` 个 token 开始。
2. 摆 block table 布局（local 挂载 → external 落地块 → 后缀空块）。
3. 滑窗下算跳过多少块。

---

### 3.3 为什么一定是 local 在前、external 在后

**Q：万一远端 prefix block 正好匹配到靠前的 block 呢？**

不会，这是**协议设计强制保证**的，两层原因：

**① 前缀缓存本质只能命中"从 token 0 开始的连续前缀"，不存在命中中间孤立块。** 因为：
- **因果注意力**：token K 的 KV 依赖它前面所有 token，要复用第 K 块必须 0…K-1 也都在块表里。
- **block hash 链式**：每块哈希 = f(自己内容, 前一块哈希)，能匹配第 K 块的前提是 0…K-1 哈希全匹配。

所以任何一方能提供的永远是"从 0 开始、无洞的连续前缀 [0, N)"。local 返回最长连续命中 `[0, L)`，`L` 就是"local 第一次断掉"的位置。

**② 查 external 时把 L 告诉 connector，让它只报"L 之后还能续多少"。** `base.py:450` 接口契约：
> Get number of new tokens that can be loaded from the external KV cache **beyond the num_computed_tokens**.
> `num_computed_tokens`: the number of **locally computed tokens**.
> the connector should only consider the **largest prefix** for which KV is available.

配合 `scheduler.py:620` 传入 `num_new_local_computed_tokens`。所以：

- 若 external 有个 < L 的靠前块 → 落在 `[0, L)`，**local 已免费提供**，`beyond L` 不计数（重叠区永远归 local，本地零成本，绝不花代价搬本地已有的）。
- 若 local 有洞、external 想补洞 → 用不了（要用第 5 块必须先有 3、4；3、4 得重算，算到 5 就顺势往下了），contract 只允许给连续前缀。

**结论**：local/external 不是可交错的散块，而是**同一条连续前缀被切成前后两段**：前半本地已有（免费复用），后半本地没有、需外部搬入。切点 = L，由 connector 接口契约强制保证，永远 local 在前、external 在后。

```
位置: [0 ...................... L-1][L ............ L+E-1]
       └──── local 命中(免费复用)───┘└── external(搬进来)──┘
```

---

### 3.4 skip 为什么按 `num_total_computed_tokens` 算

**Q：skip 的 block 应该和前缀命中块无关吧？为什么不按 req 完整总 block 算？**

**先纠正：skip 掉的块恰恰是前缀命中块的一个"前段子集"。** L182 `new_computed_blocks = new_computed_blocks[num_skipped_blocks:]` 直接从命中块列表前面砍 `num_skipped_blocks` 个 —— 被砍的就是被 skip 的块。

**为什么参考点是 `num_total_computed_tokens`（续算起点）而不是完整长度**：滑动窗口的 skip 判据是"从现在要续算的第一个 token 往回看一个窗口，再往前的 token 以后谁都不用了"。而续算的第一个 token 的位置 = `num_total_computed_tokens`。关键：它**后面的 token 还没算**，还要回看命中前缀的尾部；若按完整长度算 skip，会把这些"马上要 prefill 的 token 仍需要"的块错误 skip 掉。

**例子**（`sliding_window=4`，prompt 20 token，命中前 8 个 → `num_total=8`，续算从 token 8 起）：
```
token: 0 1 2 3 4 [5 6 7 | 8] 9 ... 19
                  └─窗口─┘   token 8 窗口={5,6,7,8}，需保留命中块的 5,6,7
get_num_skipped_tokens(8)=max(0,8-4+1)=5 → skip token 0~4，保留 5,6,7
```
**反证**：若按完整长度 20 → skip=20-4+1=17 → 会丢 token 0~16，但 token 8 还要用 5、6、7，错！因为 8~19 还没算，不能当"已过去"的参考。

对 full attention，`get_num_skipped_tokens` 恒 0，整套 skip 不触发。

---

### 3.5 `num_external = min(num_total - num_skipped, num_external)` 为什么这么写

排列顺序：local 在前 `[0, L)`，external 接在后 `[L, L+E)`（见 **3.3 节**）。skip 从最前（最老）开始砍，**先吃 local，吃完才吃到 external**。窗口很小时 skip 边界可能越过 local 切进 external。这行 min 处理 external 被 skip 波及的情况：

```python
num_external_computed_tokens = min(
    num_total_computed_tokens - num_skipped_tokens,  # 存活 token 总数
    num_external_computed_tokens,                    # 原本 external 数 E
)
```
- skip 没碰到 external（存活 ≥ E）：external 不变 = E。
- skip 切进 external（存活 < E）：存活的全是 external 尾部，external 缩成"存活数"。

**为什么要缩减**：L206 用它当门槛 `if num_external > 0` 决定要不要给 external 分配落地块 / 触发 KV 搬运。已滑出窗口、被 null 占位的 external token 不必分配落地块。若被 skip 全吃掉就变 0，门槛关闭。

**例子**（bs=1 简化）：
```
L=8, E=4, num_total=12
情况A: num_skipped=5  → 存活=7 ≥4 → external=min(7,4)=4  (只吃 local)
情况B: num_skipped=10 → 存活=2 <4 → external=min(2,4)=2  (吃光8local+2external，剩2个要搬)
情况C: num_skipped=12 → 存活=0        → external=min(0,4)=0  (全滑出，不搬不分配)
```

---

### 3.6 两个 extend 之后为什么还要 `cdiv(num_total_computed_tokens, block_size) - len(req_blocks)`，不是 0 吗？

**Q：那两个 extend 感觉加了 `num_total` 长度的块，后面相减不该是 0 吗？**

踩坑点：**`new_computed_blocks` 里只装 local 命中块，不含 external 的块。** 所以两个 extend 加进去的是 `num_local` 对应的块数，不是 `num_total`。

回顾来源：`get_computed_blocks()` 返回的是 local 的**块列表**（物理存在、KV 已填、touch 即挂）；`num_external_computed_tokens` **只是个整数**，external 的 KV 还在远端，本地一个块都没有。

- `cdiv(num_total, bs)` = 覆盖 local + external 全部所需块数。
- `len(req_blocks)` = 当前只摆了 local（+null）的块数。
- 相减 = **external 区还缺的块数** ≠ 0，分配出的是**空块**当落地容器。

**例子**（bs=16，全注意力）：
```
prompt 100, local 命中 48（3块，已存在），external 32（数字，0块），num_total=80
两个 extend 后 req_blocks=[B0,B1,B2] → len=3
allocated = cdiv(80,16) - 3 = 5 - 3 = 2  ← 不是 0！给 32 个 external token 分 2 个空块
```

**为什么不会出现"相减为 0"**：有 external 时差值 = external 块数 > 0；没 external（`num_external==0`）时外层 `if num_external > 0` 为假、整段跳过，减法根本不执行。

---

### 3.7 `new_block_ids` 的 FullAttentionSpec 判断：分配即登记清零

**Q：`if type(self.kv_cache_spec) is FullAttentionSpec: self.new_block_ids.extend(...)` 为什么加这段？**

`new_block_ids` 是"这一步新分配块 id"的增量收集器，消费链：
```
① L213 / L241  self.new_block_ids.extend(...)         # 登记（本函数 & allocate_new_blocks）
② take_new_block_ids() (L244)                          # 取走并清空
③ kv_cache_manager.py:543-547                          # 汇总所有 group manager
④ scheduler.py:908-909  new_block_ids_to_zero = ... or None
⑤ SchedulerOutput.new_block_ids_to_zero
⑥ gpu_model_runner.py:1084-1085  self._zero_block_ids(...)   # GPU 上清零
```

终点 worker 注释（`gpu_model_runner.py:1082`）：
> Zero GPU memory for freshly allocated cache blocks to prevent stale **NaN/data** from corrupting attention or SSM computation.

**为什么精确类型 `type() is` 而非 `isinstance`**：看继承（kv_cache_interface.py）
```
FullAttentionSpec(AttentionSpec)          ← L148 纯全注意力
├── MLAAttentionSpec(FullAttentionSpec)    ← L249 子类
└── SinkFullAttentionSpec(FullAttentionSpec) ← L383 子类
SlidingWindowSpec / ChunkedLocalAttentionSpec / CrossAttentionSpec(AttentionSpec)  ← 兄弟
```
`type() is FullAttentionSpec` 只认纯全注意力，**故意排除 MLA、SinkFullAttention 两个子类**（`isinstance` 会把它们算进来）。这个"分配即登记清零"机制目前只对纯全注意力启用：MLA/Sink 显存布局与 kernel 不同、不走通用清零；滑窗/局部注意力窗口外靠 null 占位处理；Mamba 有独立管理（`new_block_ids` 全文件从不为它写入）。是刻意收窄的判断。

---

### 3.8 为什么"分配出来接着写"还要先清零

**Q：block 本身要接着写入，为什么先清零？不是直接覆盖旧数据吗？**

"接着写"**只写这一步真正有 token 的槽位，不写满整个 block**。而注意力 kernel 往往按**整块**读。那些"这步没写、但被读到"的槽位，才是残留 NaN 害人的地方。你说的覆盖只覆盖写入的槽位，覆盖不到没写的部分。

- 一个 block 有 `block_size`（如 16）个槽位，但**最后一块几乎总是半满**：这步 prefill 只填到某块第 3 槽，slot 0~2 写真实 KV，slot 3~15 这步没写，仍躺着从空闲池回收来的**上个请求旧 KV / 未初始化 NaN**。
- 分页注意力 kernel 对最后这块**整块 load** 进来算，再靠 causal mask 屏蔽。理论上 slot 3~15 会被 mask 掉，但**浮点 NaN 不吃 mask**（见 **3.9 节**）。

```
新块，这步写到 slot 0~2：
  slot: [ 0  1  2 | 3  4  5 ... 15 ]
         └写入真实KV┘ └─没写，残留 NaN─┘
清零后 slot 3~15=0 → 分数有限，mask 正常屏蔽，不产 NaN；写入的 0~2 照常覆盖，无影响。
```

**时序**：清零是**分配时一次性**做的（`_zero_block_ids`），不是每次写前做。之后各步陆续往里写真实 KV，写到哪覆盖到哪；还没轮到写的尾部槽位保持 0（安全值），直到未来某步真正写它。所以不是"清零完立刻被全覆盖白清了"—— 被保护的正是"迟早要写、但可能被提前读到"的空槽。

---

### 3.9 清零 KV ≠ mask 分数；为什么用 0 而非 fp16-min

**Q：不污染 softmax 不该是取 fp16 最小值吗？softmax 减最大值后不都变负数、最大变 0 了吗？**

把两件正交的事揉混了：
- **清零的对象是 KV 向量本身**（显存里的 K、V），不是 softmax 分数（logits）。
- **"排除某个位置"是 causal mask 干的**，作用在 logits 上，和 KV 值无关。

**为什么 KV 用 0 而不是 fp16-min**：分数 = `q·k`。
| 废槽位 K 填什么 | q·k | 结果 |
|---|---|---|
| **0** | `q·0 = 0` | 有限值，安全 ✅ |
| **fp16 最小值**（巨负向量）| `q·(巨负)` = 巨大幅值 | 可能溢出 ±inf，更糟 ❌ |

清零**唯一目的**是保证 `q·k` 有限、不出 NaN/Inf，**不负责排除位置**。而且：**光清零 KV 并不能排除该槽位** —— 零 KV 分数是 0，softmax 里 `exp(0-m)` 仍是正权重，仍会被"注意到"。排除必须靠 mask，清零只是让 mask 在干净基础上工作。

**softmax 减 max 你理解对，但结论反了**：`softmax(x_i)=exp(x_i-m)/Σexp(x_j-m)`，`m=max(x)`。减 max 后每项 ≤0、最大那项变 0、`exp(0)=1` —— 这是**正常健康**的（最大 logit 权重 1，其余 (0,1)，再归一化）。被 mask 的位置 logit=`-inf`，减去有限 m 仍 `-inf`，`exp(-inf)=0`，正确排除。这套流程对**正常数**工作良好。

**NaN 的致命之处不是"变负数"，是"整行毁灭"**：若废槽位 K 是 NaN
1. logit = `q·NaN = NaN`
2. `m = max(所有 logit)`，`max(NaN, 任何数)` 在硬件上常传染成 NaN → `m=NaN`
3. 于是整行每项 `exp(x_j - NaN) = NaN` → softmax **整行**输出 NaN

**为什么 mask 挡不住 NaN**：fused kernel 的 mask 常是"加 -inf 偏置"（`NaN+(-inf)=NaN`）或"乘 0"（`NaN*0=NaN`）。**mask 能抹掉数值贡献，抹不掉 NaN 传染**。所以要从根上让 KV 不是 NaN → 分配时清零。

---

## 四、关键设计点速查

| 设计点 | 原因 |
|---|---|
| 只处理"已算好的块"，不管现算后缀 | 现算后缀归 `allocate_new_blocks`；本函数只 touch+挂命中块、给 external 分落地块 |
| local 直接挂、external 分空块 | local KV 已在本卡 GPU（复用）；external KV 在远端（需空块接收搬运）|
| local 恒在前、external 恒在后 | connector 契约"报 local 长度之后还能续多少"+ 前缀缓存连续无洞 |
| skip 按 `num_total_computed_tokens` 算 | 参考点是续算起点；用完整长度会误删续算 token 仍需的命中尾块 |
| `num_external = min(...)` | external 贴在 local 后，skip 可能切进 external，只给窗口内 external 分落地块 |
| `cdiv(num_total,bs) - len(req_blocks)` ≠ 0 | `new_computed_blocks` 只含 local 块，差值 = external 缺的块数 |
| `new_block_ids` 登记 → `new_block_ids_to_zero` | 回收来的脏块可能残留 NaN，worker 在 forward 前清零，防污染注意力 |
| `type() is FullAttentionSpec`（精确）| 只对纯全注意力启用清零登记，排除 MLA/SinkFull 子类及其它内存语义不同的类型 |
| 清零用 0 而非 fp16-min | 清零 KV≠mask；`q·0=0` 有限安全，排除靠 causal mask；NaN 会让 max 变 NaN 整行毁灭 |
