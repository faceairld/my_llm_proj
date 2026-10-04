# vllm_musa prefix-caching 命中场景下 varlen_fa_seqlen_unpad 越界写

> 首次发现：2026-05-09 14:10:45（vllm_musa 20260309）—— 当时误诊为 "broadcast 死锁"
> 复现验证：2026-05-19（vllm_musa 20260323,bug 未修复)
> **元凶定位完成：2026-05-21**(经 bisection,精确到 `ops.varlen_fa_seqlen_unpad` 越界写)
> **Path A 验证通过:2026-05-26**(不崩 + 输出语义正确,Path A 工程化可用 ✅)
> 环境：MTT S5000 × 8,MUSA SDK 4.3.5,torch_musa 2.7.1,MuDNN 3.1.5
> 容器：`gy_work` @ node165
> 严重程度:**严重**(原版 vllm_musa 一定崩;Path A workaround **已可用**)

---

## 📦 测试环境 / 版本信息(2026-05-26 实测)

| 组件 | 版本 | 备注 |
|---|---|---|
| **vllm_musa**(主体) | `0.1.dev358+gd3980eddc.d20260323` | **build 自 2026-03-23** |
| vllm(上游) | `0.9.3.dev0+ga5dd03c1e.d20260323` | 同日 build |
| torch_musa | `2.7.1` | — |
| MUSA SDK | `4.3.5` | `/usr/local/musa-4.3.5` |
| MuDNN | `3.1.5` | — |
| 容器镜像 | `sh-harbor.mthreads.com/sets/sets_vllm_musa:musa_sdk_4.3.5_torch_2.7.1_fix_ray` | sha256 `c3ae44c3e6ee`(本地拉于 2 个月前) |
| 镜像快照 | `gy_work_snapshot:vanilla`(本地 commit) | 从 gy_work 容器 commit 出来,用于起 gy_work_2 干净对照 |
| 测试机 | MTT S5000 × 8 @ `node165` | TP=8 |
| OS | Ubuntu 22.04.5 LTS(容器内) | — |
| 测试模型 | Qwen2.5-14B | `/data/SETS-2.0-test/models/Qwen2.5-14B` |

### 是不是最新?

⚠️ 这个 build 是 **2026-03-23**,距今(2026-05-27)约 **2 个月**。本机缓存里 `sets_vllm_musa` tag 下只有两个版本(都是 2 个月前的):
- `musa_sdk_4.3.5_torch_2.7.1_fix_ray`(我们用的)
- `musa_sdk_4.3.5_torch_2.7.1`(无 fix_ray)

**未确认 MTT harbor 上是否有更新的 release** —— 需要登录 harbor 查 tag。两种确认方法:

```bash
# 1. 直接拉 latest 看是否存在 / 是不是更新的版本
docker pull sh-harbor.mthreads.com/sets/sets_vllm_musa:latest

# 2. 列 harbor 上的所有 tag(需要 harbor 凭据)
curl -s -u USER:TOKEN "https://sh-harbor.mthreads.com/v2/sets/sets_vllm_musa/tags/list"
```

或者直接问 MTT 团队当前最新的 sets_vllm_musa tag。

### 升级前要考虑的事

如果有新版本,**不要贸然升**:
- 当前所有 STUB 二分定位 + Path A workaround 都是针对**这个版本的 `_kernels.so`** 测出来的
- 新版本里 `varlen_fa_seqlen_unpad` 行为可能变化 → 我们的实验结论可能失效,需要重新验证
- Path A patch 针对当前 `flash_attn.py` 写的,新版可能 diff 不一致

**建议升级流程**:
1. 先在当前版本把诊断报告 + Path A 提给 MTT,等他们给修复版
2. 在他们指定的修复 tag 上跑同样的 long_context 测试,验证 OOB 不再触发
3. 如果根因修了 → 可以撤 Path A;如果没修 → 把 Path A 重新应用到新版 `flash_attn.py`(可能需要适配)

---

## 📌 附加发现:V1 引擎在 PH1 (S5000) 上当前不可用(2026-05-27)

> **背景**:Path A 在 V0 引擎下验证通过(2026-05-26)。原本想试 V1 引擎,因为 V1 的 attention layer 已经 hardcoded 了 LMCache 集成需要的 KV connector hook,而且 V1 用 `vllm.flash_attn_varlen_func` 而不是 `varlen_fa_seqlen_pad/unpad`,理论上能完全绕开 Bug 1。但**实测发现 V1 在我们的 S5000 上压根跑不起来**。

### 验证步骤(2026-05-27)

```bash
# 起 V1 引擎,跑 long_context benchmark
CONTAINER_NAME=gy_work VLLM_USE_V1=1 \
  MODE=our SCENARIOS=long_context CONCURRENCY=1 REQUESTS=8 \
  ... bash wait_and_run.sh
```

### 遇到的两个错误(逐个解开)

**错误 1**:vllm_musa 自己代码不一致 — 已 patch 绕过

```
ValueError: On MUSA platform, V1 flash_attn block_size must be 32.
```

| 位置 | 行为 |
|---|---|
| `platforms/musa.py:170` `check_and_update_config` | PH1 上无脑 `cache_config.block_size = 64` |
| `platforms/musa.py:256-258` `get_attn_backend_cls` | V1 强制要求 `block_size == 32`,否则抛 |

⇒ PH1 + V1 必崩。这是 vllm_musa 自己的代码不一致(QY2 上 V1 有 block_size=32 的分支,PH1 上漏了)。

**我们的修法**:patch `platforms/musa.py:170-171`,让 PH1 在 V1 模式下也用 32。代码:

```python
# 原版:
if on_ph1():
    cache_config.block_size = 64

# Patch 后(CLAUDE PATCH 2026-05-27):
if on_ph1():
    if envs.VLLM_USE_V1:
        cache_config.block_size = 32   # 跟 QY2+V1 对齐
    else:
        cache_config.block_size = 64   # V0 保持原状
```

patch 脚本:`/data/my_vllm_test/patches/ph1_v1_block_size_fix.py`(idempotent + inline 注释 + 回滚说明)

**错误 2**:MTT 闭源 kernel 在 PH1 上真没实现 ⛔

patch 之后 V1 engine 成功启动,模型加载也成功。但在 `compile_or_warm_up_model` → `capture_model` → 跑 dummy forward 时:

```
RuntimeError: Worker failed with error
'flash_attn_varlen_fun not support on MUSA arch 310'
                                              ↑↑↑
                                       arch 310 = PH1 = S5000
```

### 调用栈定位

```
vllm V1 engine
  └── attention layer (vllm_musa V1 patch_attention_layer.py)
      └── vllm_musa/attention/flash_attn.py
          └── ops.flash_attn_varlen_func(q, k, v, ...)
              └── ❌ MTT C++/MUSA kernel:"not support on MUSA arch 310"
```

这次的拒绝在 `_kernels.so` 里面(闭源)。**`flash_attn_varlen_func` 这个底层 kernel 没在 S5000 (PH1, arch 310) 上实现**。我们的 Python patch 改不了这层。

### 这意味着什么

**V1 + PH1 当前完全不可用**,即使绕过 Python 层的 block_size 检查也跑不通,因为底层 kernel 真的没适配。

V1 + LMCache 整套集成方案,在 S5000 上**要等 MTT 把 `flash_attn_varlen_func` 适配到 arch 310 后**才能尝试。

### 实测验证矩阵

| 引擎 | 卡 | block_size | 状态 |
|---|---|---|---|
| V0 | QY2(S3000/S4000) | 16 默认 | ✅ 可用(workshop 验过) |
| V0 | **PH1(S5000)我们** | 64 | ✅ 可用,但有 Bug 1(需要 Path A) |
| V1 | QY2 | 32 | 推测可用(未实测) |
| V1 | **PH1(S5000)我们** | 32(patch 后) | ❌ **kernel arch 310 不支持** |

### 给 MTT 的诉求(随 Bug 1 一起提)

1. 修 `varlen_fa_seqlen_unpad`(Bug 1 主体,见后续章节)
2. **同时**:把 V1 的 `flash_attn_varlen_func` 适配到 PH1(arch 310)
3. 修 `platforms/musa.py` 里 PH1 + V1 block_size 的代码不一致 bug(我们的 patch 内联示范了改法)
4. 在 V1 + S5000 上做 long_context + prefix-caching 端到端测试,**这是 LMCache 后续集成必经的路径**

### Path 选择最终结论

| Path | 状态 |
|---|---|
| **A. Python 层 concat + SDPA(Path A)** | ✅ 已验证 + 部署中。**当前唯一可用方案** |
| B. MTT 改 C++ `varlen_fa_seqlen_unpad` | ❌ 等 MTT |
| C. vllm 上游 `context_attention_fwd`(Triton) | ❌ Triton 不可用 |
| **D. 切到 V1 引擎绕开 Bug 1** | ❌ **2026-05-27 验证:V1 在 S5000 上 kernel 不支持** |

---

## 🎉 2026-05-26 更新:Path A 验证通过 — 项目里程碑

之前(2026-05-21)我们以为 Path A "不崩但输出乱码"。**这是错的判断** —— 经 2026-05-26 一组对照实验,真相是:

| 之前(2026-05-21)以为 | 实际(2026-05-26 实验) |
|---|---|
| Path A 输出乱码 → Path A 实现错了 | Path A 实现正确 ✅ |
| benchmark long_context 是公平测试 | benchmark long_context prompt **本身有 bug**(6 倍重复同一段话 + 无意义的 "问题角度N" 后缀),让模型脱轨 |

### 验证步骤(2026-05-26 跑的 3 组对照)

| 组 | 配置 | 输出 |
|---|---|---|
| 1 | gy_work_2 vanilla + short prompt + cache ON | ✅ 大体正常,有少量 stop-token 杂质 |
| 2 | gy_work_2 vanilla + **重复的** long prompt + cache OFF | ❌ 全乱码(跨语言、code 注入、偏题) |
| 3 | gy_work_2 vanilla + chat 多轮 + cache **ON** | ❌ 3 轮后陷入 `!!!!!` 永久死循环(prefix-cache 命中真触发 OOB) |
| 4 | gy_work Path A + **真实** long prompt + cache **ON** | ✅ **8/8 全过,输出语义正确** |

### Path A 在第 4 组实验里的硬证据

第 4 组(`runs/20260526_065339_our/`)有三类客观证据证明 Path A 真生效:

1. **`enable_prefix_caching=True`**(server.log 确认 config)
2. **Path A 自带的诊断 print** 输出:
   ```
   [CODEX PATH-A-DIAG] num_prefill_tokens=55  seq_lens=[695]
                       new_lens=[55]  cached_lens=[640]  block_size=64
   ```
   `cached_lens=640` 意味着 695 个 token 里 640(92%)从 paged KV cache 命中,只需要 prefill 后面 55 个 token 的问题部分 → **prefix-cache 真的命中,Path A 真的处理这种命中场景**
3. **TTFT 对比**:
   - Request #0(第一次,文章入 cache):319.1ms
   - Request #1-7 + 第二轮全部:60-100ms
   - **TTFT 降到 1/5** = prefix-cache 加速可见

### 输出语义评估(第 4 组,16 条共抽样)

| 等级 | 数量 | 典型表现 |
|---|---|---|
| ✅ 优秀(完整准确) | 8 | "大脑哪些区域参与时间感知" 给了结构化 4 点;"延长生命体验" 给了 4 条 |
| ⚠️ 可用(答对但末尾有杂质) | 5 | 答案正确,末尾出现 "with withuser" / "UTOPIA 10.999..." / `you are a helpful assistant` 等 chat-template 泄漏 |
| ❌ 严重截断 | 3 | tokens_out=5-33,只吐几个字就停(stop token / template 误触发 EOS) |

**结论**:Path A 在 attention 数学层面**正确**(8 条优秀回答证明)。剩下的"末尾杂质 + 偶发截断"是 **stop token / chat template handling** 的独立 bug(短 prompt 也有、关 cache 也有,跟 Path A 无关 — 见 [Bug 2/3 章节](#bug-地图1-cache-崩溃-2-stop-token-3-长-prompt-数值))。

### 最终 bug 地图(1 cache 崩溃 / 2 stop-token / 3 长 prompt 数值)

```
vllm_musa + Qwen2.5-14B + TP=8 实测共有 3 类问题:

┌─────────────────────────────────────────────────────┐
│ Bug 1: prefix-cache 命中 → !!!!! 死循环 / OOB 崩溃  │
│   元凶:  varlen_fa_seqlen_unpad(C++ kernel)        │
│   触发:  sum_seq < max_seq × batch_size            │
│   修法:  Path A(Python concat + SDPA) ← 已验证 ✅ │
├─────────────────────────────────────────────────────┤
│ Bug 2: stop-token / chat template 残留              │
│   表现:  "you are a helpful assistant"、"user"     │
│           "assistant" 标签泄漏、偶尔提前截断       │
│   性质:  跟 Path A/cache/attention 都无关         │
│   现状:  存在但不阻塞,待修                        │
├─────────────────────────────────────────────────────┤
│ Bug 3: 长 prompt 数值/语义偶发不稳                  │
│   表现:  生僻陌生词("UTOPIA"、"sterol")、         │
│           偶发跨语言或末尾胡言乱语                 │
│   现状:  影响小,跟 attention 后端 SDPA 精度有关  │
└─────────────────────────────────────────────────────┘
```

### Bug 1 三种 workaround 路径对照(为什么最终选 Path A)

| 路径 | 思路 | 工程量 | 风险 | 结果 |
|---|---|---|---|---|
| **A. Python concat + SDPA** | Python 层用 `block_tables` 自己从 paged KV cache 拉出 cached K/V,padded buffer 拼上 new K/V,调原稳定的 `_scaled_dot_product_attention_flash_musa`。**完全旁路 `varlen_fa_seqlen_pad/unpad` 这对函数** | 中(~100 行 Python) | 我们自己实现的散布/取回逻辑要跟原 kernel 等价 | ✅ **已上线 + 已验证语义对** |
| **B. 改 MTT 的 C++ `varlen_fa_seqlen_unpad`** | 在 kernel 内部加输出 buffer 边界检查,或让上层调用方传 `query_lens`(真实新 token 数)和 `seq_lens`(完整长度)两个参数,unpad 用 `query_lens` 决定写多少行 | 低(几行 C++) | 需要 MTT 编译发版 | ❌ **未走**:`_kernels.so` 是闭源 MuDNN 编译产物,我们改不了。已发现 bug 但**没向 MTT 上报修复**,等后续 |
| **C. 用 vllm 主仓的 `context_attention_fwd`** | vllm 上游有专门 paged-aware prefill kernel(Triton 实现),它直接用 `block_tables` 在 kernel 内部访问 KV cache,无 OOB | 极小(直接 import) | 依赖 Triton | ❌ **失败**:MUSA 环境检测到 "2 active drivers" 自动把 `vllm.triton_utils.triton` 替换成 `TritonPlaceholder`,缺 `triton.next_power_of_2` 等必需 API,Triton kernel 起不来。已写 patch(`path_c_paged_prefill.py`)但跑不通 |

**为什么选 A**:B 要等 MTT;C 依赖 Triton 但 MUSA 把 Triton 屏蔽了;A 是唯一**我们能完全自主控制 + 不依赖外部修复**的路径。

### prefix-cache 三档命中行为 + Path A / Bug 1 触发条件

vllm 的 prefix-cache 不是"命中 / 未命中"二元的,而是按跟 cache 重合的程度分三档,每档走不同 kernel:

| 重合度 | vllm 的处理 | 是否进 `prefill_metadata` 分支 | 是否触发 Bug 1(OOB) | Path A 是否需要 |
|---|---|---|---|---|
| **0% 重合**(全新 prompt,fresh prefill) | 整段 prompt 走 prefill 路径,完整算 attention | ✅ 进 | ❌ 不(`sum_seq == max_seq × bs`,无不对称) | ❌ 不需要,gate 关 |
| **部分重合**(共享 prefix + 新尾部) | 只对"新尾部"的 token 走 prefill,KV cache 提供 prefix 部分 | ✅ 进 | ✅ **触发**(`sum_seq < max_seq × bs`,unpad 越界写) | ✅ **必须**(Path A 唯一保护的场景) |
| **100% 完全重合**(整段 prompt 已在 cache) | 跳过 prefill,直接进 decode 阶段 | ❌ **不进**(无 prefill 工作) | ❌ 不(走 decode kernel,不调 varlen_fa_*) | ❌ 不需要(Path A 看不到这种请求) |

**关键含义**:

1. **Bug 1 仅在"部分重合"时触发** —— "fresh prefill" 和 "完全命中" 都不会崩。生产环境如果保证 prompt 要么完全新(0%)要么完全旧(100%),就能避开 bug。但**多轮对话天然就是部分重合**(每轮新 user message 不同,前面历史共享),所以避不开,Path A 必须装。

2. **Path A 只覆盖一档场景**(part 命中),另外两档与 Path A 无关:
   - 0% 重合时:走原 `sdpa_attention_with_kernel_seqlen_pad` 的 fresh prefill 路径,已知稳定(B2 实验证明)
   - 100% 命中时:走 vllm_musa 的 decode kernel(`paged_attention_v2` 或同等),也已知稳定

3. **这解释了为什么 short scenario 测不出 Path A 的行为**:
   - benchmark short 8 个不同问题 → 互相无共享 prefix → Round 1 全是 0% 重合(fresh prefill)
   - Round 2 跑完全相同的 8 个问题 → 100% 完全重合 → 跳 prefill 走 decode
   - **两轮没有一条进 partial 命中**,Path A 自然不触发
   - 要测 Path A 必须用 long_context(8 条共享 ~1k token 文章 + 各自不同的问题 → partial 命中)

4. **这也解释了 2026-05-26 chat 实验里 3 轮就 `!!!!`**:
   - Turn 1:fresh prefill(0%),正常
   - Turn 2:输入 = system + user1 + assistant1 + user2,前面三段在 cache 里 → **partial 命中** → 进 prefill → 触发 unpad OOB → 污染 GPU
   - Turn 3:更深的 partial 命中 → 又 OOB → 累积污染 → 模型生成 `!!!!`

### Path A 在代码里怎么路由三档场景(只加 1 个分支)

接手者常问:**Path A 是不是要分别处理"0% / partial / 100%"三档场景?那要加 3 个分支吗?**

**答**:不需要。Path A patch **只加了 1 个分支**(partial 命中那个),另外两档场景走的都是 vllm_musa **原版就有**的代码路径。

#### 三档路由分布

```
prefill_meta = attn_metadata.prefill_metadata
  │
  ├── 是 None? → decode 分支(原版)                                ← 原版,没动
  │              ★ 100% 完全命中走这里:vllm 调度器跳过 prefill
  │
  └── 不是 None(prefill 阶段):
        │
        ├── _path_a_hit 检查(★ 我加的 gate)
        │     条件 = DECODER + kv_cache 非空 + block_tables 非空
        │            + num_prefill_tokens < sum(seq_lens)  ← 关键 gate
        │
        ├── _path_a_hit = True   → 走 Path A(Python concat + SDPA) ← ★ 我加的
        │                          ★ partial 命中走这里(bug 1 触发场景)
        │
        └── _path_a_hit = False  → 落到原 sdpa_attention_with_kernel_seqlen_pad
                                   ★ 0% 命中(fresh prefill)走这里  ← 原版,没动
```

#### 三档场景跟代码路径的对应

| 场景 | `prefill_meta` | `_path_a_hit` gate 条件 | 路由到 | 是不是新加 |
|---|---|---|---|---|
| **0% 重合**(fresh prefill) | 非 None | `num_prefill_tokens == sum(seq_lens)` → gate **不满足** | 原 `sdpa_attention_with_kernel_seqlen_pad` 的 prefill 路径 | ❌ 原版,没动 |
| **partial 重合**(命中部分) | 非 None | `num_prefill_tokens < sum(seq_lens)` → gate **满足** | **Path A**(Python concat + SDPA) | ✅ **新加** |
| **100% 重合**(完全命中) | **是 None**(调度器跳过 prefill) | (根本进不去 prefill 分支) | 原 decode 分支(用 `paged_attention_v2` 或同等 kernel) | ❌ 原版,没动 |

#### Path A 代码骨架(简化)

```python
if prefill_meta := attn_metadata.prefill_metadata:                  # 原版的 prefill 入口
    # ═══════════════════════════════════════════════
    # ↓↓↓ Path A patch 加的内容从这里开始 ↓↓↓
    # ═══════════════════════════════════════════════
    _path_a_hit = (
        self.attn_type == AttentionType.DECODER                     # 普通 decoder attention
        and kv_cache.numel() > 0                                    # KV cache 已分配
        and prefill_meta.block_tables is not None
        and prefill_meta.block_tables.numel() > 0                   # block_tables 已配
        and attn_metadata.num_prefill_tokens                        # ★ 关键 gate:
            < int(prefill_meta.seq_lens_tensor.sum().item())        #   只在 partial 命中时为 True
    )

    if _path_a_hit:
        # Python 层从 KV cache 用 block_tables 拉 cached K/V
        # concat 新 K/V 到 padded buffer
        # 调 _scaled_dot_product_attention_flash_musa (SDPA)
        # slice 出新 token 的输出
        return _pa_output                                           # ★ 早返回,不走原 fresh prefill 路径

    # ═══════════════════════════════════════════════
    # ↑↑↑ Path A patch 加的内容到这里结束 ↑↑↑
    # ═══════════════════════════════════════════════

    # ↓↓↓ 下面是 vllm_musa 原版 fresh prefill 路径,完全没动 ↓↓↓
    if self.attn_type == AttentionType.DECODER and (
            kv_cache.numel() == 0 or prefill_meta.block_tables is None
            or prefill_meta.block_tables.numel() == 0):
        # 原版 fresh prefill 调 sdpa_attention_with_kernel_seqlen_pad
        ...
```

完整代码(包括 Python concat + SDPA 那 100 行)见 `/data/my_vllm_test/patches/path_a_python_concat.py`。

#### 这么设计的好处

- **改动面最小**:只拦截 bug 路径(partial 命中),不动两个已知稳定的路径
- **失败模式安全**:如果 gate 写错条件(比如把 fresh prefill 误判进 Path A),最多输出错误,不会比原版更糟,因为 Path A 自己只用稳定的 SDPA kernel(B2 实验已验证)
- **回滚成本低**:只要把这一段删掉就能回到完全原版 vllm_musa

### 残留问题 Bug 2/3 详解(2026-05-26 讨论)

Path A 修了 Bug 1(prefix-cache OOB)后,剩两类小问题。这一节详细记录症状、根因假设、修复路径,**留给后续工作 / MTT 上报**。

#### 残留症状清单(2026-05-26 多次测试观察到)

| 症状 | 例子 | 出现场景 |
|---|---|---|
| chat template / EOS 标签泄漏 | 末尾冒 `You are a helpful assistant. 用简洁的中文回答`、`user/assistant` 标签、自演假对话 | 短 / 长 prompt 都有,生成接近 max_tokens 时尤其明显 |
| 生僻 / 错位 token | `kościelne`(波兰)、`hastalık`(土耳其)、`эффектно`(俄)、`忭`、`羲`、`漖`、`RLHFzen`、`Vas blanco`、`UTOPIA 10.999...` | 散点出现,短 / 长 / Path A 命中 / 不命中场景下都有 |
| 偶发跨语言段落 | 中文回答里突然切英文 / 俄文一段 | 长 prompt 居多 |
| 偶发早截断 | `tokens_out=5-30` 就停 | 短 prompt 也会 |

#### 这些**不是** Bug 1 / Path A 的残留(已确认)

- gy_work_2 干净原版 + 短 prompt + 关 cache 也出杂字 → 跟 Path A 无关
- gy_work_2 + 短 prompt + 关 cache 出 stop-token 泄漏 → 跟 cache 无关
- ⇒ 这是 vllm_musa 后端 / 模型 / chat template 配置层面的独立问题,跟我们前两周追的 OOB bug 是**完全平行的另一组 bug**

#### 分成两类来分析

##### 类别 1 — chat template / stop-token 残留(配置层)

**根因**:Qwen2.5 用 `<|im_start|>` / `<|im_end|>` 划分对话回合。vllm 服务可能没把 `<|im_end|>` 识别成停止信号 → 模型继续生成 → 跑出 turn 边界。

**可解决程度**:✅ 可解,纯配置问题

**修复路径**:
1. 检查 `/data/SETS-2.0-test/models/Qwen2.5-14B/tokenizer_config.json` 里 `eos_token_id`、`pad_token_id` 设置
2. 启动 vllm 时显式传 stop token:
   ```bash
   vllm serve ... --stop-token-ids 151645  # <|im_end|>
   ```
3. 或客户端调用时传:
   ```python
   client.chat.completions.create(..., extra_body={"stop_token_ids": [151645]})
   ```
4. 重测看 `user/assistant` 标签和 chat-template 泄漏是否消失

##### 类别 2 — 数值噪声导致 logit 污染(后端精度层)

**重要澄清**(2026-05-26 讨论):
- ❌ **不是 temperature 太高导致的**
- 正常推理栈(H100 + cuDNN)+ temperature=0.7-1.0 不应该产生 `kościelne`/`hastalık`/`忭`/`漖` 这种本应概率≈0 的 token
- ✅ **是 logit 分布被污染了** — 本应分配给这些 token 的概率被数值噪声推高到 sampling 能抽到的范围
- temperature 只是把后果**放大**的开关,不是根因
- ⇒ greedy decoding(temp=0)能**掩盖**症状(只取 top-1),但**不修根因**

**根因候选**(按可能性排序):

| 候选 | 解释 | 改正成本 |
|---|---|---|
| **A. MuDNN flash sdpa BF16 累积精度** | MTT 实现的 `_scaled_dot_product_attention_flash_musa` BF16 累加器精度不够,长 seq / 大 batch 时数值漂 | 🔴 高 — 闭源 kernel,我们改不了 |
| **B. RoPE 实现精度** | 比如用 BF16 算 sin/cos 而非 FP32,长 position 累积误差 | 🟡 中 — vllm_musa 里能改 |
| **C. TP=8 all-reduce BF16 累积误差** | 8 卡 reduce 用 BF16,数值漂 | 🟡 中 — 试 MCCL FP32 reduce |
| **D. CUDA Graph 捕获副作用** | 捕获时跟实际形状对不上 | 🟢 易 — `--enforce-eager` 测一下 |
| **E. KV cache BF16 精度** | KV cache 默认 BF16 存储,长 seq 累积 | 🟡 中 — 试 `--kv-cache-dtype fp16` |
| **F. 某个具体算子(layernorm / softmax / activation)精度问题** | 同 A,但在别的算子里 | 🔴 中-高 — 要逐层 diff |

**可解决程度**:⚠️ 部分可解 — D 容易测,B/C/E 中等,A/F 难

#### 诊断顺序(从便宜到贵)

| 步骤 | 操作 | 信息量 |
|---|---|---|
| 1 | 临时改 `temperature=0` 重测 long_context | 区分"污染严重(top-1 都中)"vs"只污染尾部" |
| 2 | `ENFORCE_EAGER=1` 重测 | 排除 / 确认 CUDA Graph 副作用(D) |
| 3 | `--stop-token-ids 151645` 配上 | 修类别 1(chat template 泄漏) |
| 4 | `--kv-cache-dtype fp16` 试 | 排除 / 确认 KV cache BF16 精度(E) |
| 5 | 拿同 prompt 在 H100 + 同模型 跑一遍,对比 token 概率分布 | 如果 H100 干净 → MUSA 后端有数值问题(A) |
| 6 | 把诊断对照证据交给 MTT,要求查 MuDNN flash sdpa 精度 | 终极修法 |

**先做 step 1+2+3 比较便宜**,步骤 4+5+6 是 escalation 路径。

#### 严重程度评估

| 维度 | 评估 |
|---|---|
| 是否阻塞生产 | ❌ 不阻塞 — 大部分输出仍可用,只是偶尔有杂质 |
| 是否影响 demo / 评测分数 | ⚠️ 影响 — 评测得分会比 H100 上稍低 |
| 是否影响多轮对话稳定性 | ⚠️ 偶尔影响 — 输出末尾有 template 泄漏看起来不专业 |

⇒ Bug 1 修好后,这俩是**质量改进项**,不是 blocker。生产可以先上,质量优化后续迭代。

### 当前部署状态

| 组件 | 状态 |
|---|---|
| `gy_work` 容器 | flash_attn.py 装着 Path A patch + REF-KV-DIAG 诊断 + 顶部 `import sys` 修复(后两者是历史遗留,可保留) |
| `gy_work_2` 容器 | 完全干净的原版 vllm_musa,做对照用,**长 prompt + prefix-cache 必崩** |
| `benchmark.py` | long_context scenario 已换成真实非重复中文文章 + 8 个真问题 |
| 文档 | 本文件 |

### 还剩什么没做(待办)

| 优先级 | 任务 | 说明 |
|---|---|---|
| 高 | 把 Path A patch 给 MTT,让他们修 C++ unpad 的根因 | 我们只是 workaround,根因在他们的 _kernels.so 里。送一份诊断报告 + Path A 代码作为示范 |
| 中 | 修 Bug 2(stop-token 泄漏) | 看看 Qwen2.5-14B chat template 和 vllm 的 stop_token_ids 设置是否对齐 |
| 中 | 调研 Bug 3(长 prompt 数值不稳) | 怀疑 `_scaled_dot_product_attention_flash_musa` 在长 seq 上 FP16/BF16 累积误差,或 RoPE 位置编码长 position 数值漂 |
| 低 | 把 Path A 包装成正式 patch 提交给上游(vllm_musa) | 而不是停留在我们的临时 docker exec patch |
| 低 | benchmark 加更多 scenario(varied length prompts、multi-turn) | 当前只有 short 和 long_context_real |

---

## 🚀 项目交接概览(2026-05-21 初版 → 2026-05-26 更新)

> 本节是 2026-05-21 写的初版交接信息。**2026-05-26 已有重大进展**:Path A 验证通过,Bug 1 已实质解决。新结论详见文档顶部 "🎉 2026-05-26 更新" 章节。下面的内容已就 2026-05-26 状态修正,但保留了 5/21 时的进度时间线作为历史参考。

### 一句话:这个项目是什么

诊断 **vllm_musa 0.9.3.dev0+ga5dd03c1e.d20260323**(摩尔线程 vllm 移植版)在 **`--enable-prefix-caching` 命中场景下挂掉的 bug**。

经过两周多次实验,**已 100% 锁定根因**:`vllm_musa/_kernels.so` 里的 C++ 算子 `ops.varlen_fa_seqlen_unpad` 在 prefix-cache 命中触发的 shape 不一致(`sum_seq=8, max=904`)时**越界写 output 缓冲区,污染 GPU 显存**,导致 sticky GPU error。所有之前看到的症状("broadcast 死锁"、"MuDNN AsmKernel Error"、"shm_broadcast 超时"、"Fill::Run failed"、"Permute::Run failed"、各种 illegal memory access)**都是这个根因在下游不同 op 上的次生症状**。

**Workaround 已完成**:Path A(Python concat + SDPA)2026-05-26 验证通过,既不崩 + 输出语义正确。Bug 1 实质解决,剩 Bug 2/3 是质量小毛病。

### TL;DR(给完全没接手过的人 30 秒了解)

| 维度 | 状态 |
|---|---|
| **Bug 根因** | ✅ 锁定 `ops.varlen_fa_seqlen_unpad` 越界写 |
| **复现方法** | ✅ 干净原版 + `SCENARIOS=long_context` + prefix-caching ON → 第 2 条请求开始崩 / 多轮 chat 3 轮后 `!!!!` 死循环 |
| **触发条件** | `--enable-prefix-caching=on` + **partial cache 命中**(prefix 共享 + 新尾部)+ TP ≥ 某个值。fresh prefill 和 100% 命中都不触发,见 [prefix-cache 三档命中行为](#prefix-cache-三档命中行为--path-a--bug-1-触发条件) |
| **生产 workaround(可立即用)** | **方案 A**:装 Path A patch(`docker exec gy_work python3 /data/my_vllm_test/patches/path_a_python_concat.py`),保留 prefix-caching 加速;**方案 B**:关 `--enable-prefix-caching` 失去多轮加速但稳 |
| **代码层 workaround(Path A)** | ✅ 已实现 + 2026-05-26 验证语义正确(`runs/20260526_065339_our/` 有 8/8 全过 + `cached_lens=640` 硬证据 + TTFT 319→60ms 加速) |
| **MTT 侧修法** | 给 MTT 改 C++ unpad 加边界检查;或加完整 prefix-cache 支持。待上报。 |
| **剩余 bug** | Bug 2(stop-token 泄漏,配置层可解)+ Bug 3(数值噪声 logit 污染,后端精度问题难解)。详见 [残留问题章节](#残留问题-bug-23-详解2026-05-26-讨论) |

### 文件位置速查

| 文件 | 用途 |
|---|---|
| **这个 ISSUE 文档** | `/data/my_vllm_test/ISSUE_vllm_musa_broadcast_deadlock.md` |
| **测试入口脚本** | `/data/my_vllm_test/wait_and_run.sh`(等 GPU 空 + 触发 run_all)|
| **vllm 启动脚本** | `/data/my_vllm_test/run.sh`(三种 MODE:our/workshop/minimal) |
| **完整 benchmark 编排** | `/data/my_vllm_test/run_all.sh` |
| **benchmark 客户端** | `/data/my_vllm_test/benchmark.py`(已加 sidecar `.outputs.txt` 输出) |
| **所有累计 patch 脚本** | `/data/my_vllm_test/patches/*.py`(每个 patch 一个) |
| **当前 vllm_musa 修改目标** | `/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py`(容器 `gy_work` 内) |
| **原版备份(回滚用)** | `/data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py`(host 上) |
| **diagnostic 输出和 run 日志** | `/data/my_vllm_test/runs/<timestamp>_<mode>/` |
| **诊断脚本** | `/data/my_vllm_test/concurrent_chat.py`(并发压测),`/data/my_vllm_test/show_output.py`(发请求看输出) |

### 当前文件状态(2026-05-26 最新)

容器 `gy_work` 内 `flash_attn.py` 当前装着 **Path A patch**(Python 层 prefix-cache 处理),已经过 2026-05-26 验证可用:
- prefix-cache 命中场景绕开 `varlen_fa_seqlen_pad` / `_scaled_dot_product_attention_flash_musa` 的内层 / `varlen_fa_seqlen_unpad` 这三个 C++ op
- 改用纯 Python 实现:从 `kv_cache` 用 `block_tables` 取 cached K/V,concat 上新 K/V,zero-fill 到 padded buffer,然后调 SDPA,最后 slice 出新 token 的输出
- 结果:✅ **不崩了**(8/8 OK,无 MUSA error)
- 结果:✅ **输出语义正确**(2026-05-26 用真实长文章 + 8 个真问题验证,大部分回答优秀;之前 5/21 以为"乱码"是被坏 benchmark prompt 误导)
- 剩 Bug 2/3(stop-token / 数值噪声),跟 Path A 无关,见 [残留问题章节](#残留问题-bug-23-详解2026-05-26-讨论)

`gy_work` 内还遗留两个历史 patch:
- **REF-KV-DIAG / CACHE-LAYOUT-DIAG**:Path A 开发期的诊断 patch,留着不碍事,提供 `cached_lens` 等运行时信息
- **顶部 `import sys`** 修复:REF-KV-DIAG 漏 import,补上

容器 `gy_work_2` 是**完全干净的原版 vllm_musa**(`gy_work_snapshot:vanilla` 镜像起的,仅恢复 `flash_attn.py` 为原版),做对照用,**长 prompt + prefix-cache 必崩**。

**回滚方法**:
```bash
docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py \
    gy_work:/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py
```

### 当前进度:已做 vs 待做

#### ✅ 已经做完的(全部有证据,见后续章节)

1. **5/9 首次发现** —— 观察到长 prompt+prefix-cache+TP=8 必挂,初步诊断为"broadcast 死锁"
2. **5/19 重测验证** —— 镜像升级到 20260323 仍崩,通过 8 组对照实验锁定**三个必要条件**(TP=8 + prefix-cache=on + 长 prompt)
3. **5/19 路径定位** —— py-spy 抓栈到 `flash_attn.py:388 _get_seq_len_block_table_args`(后来发现这只是 sticky error 的"信使")
4. **5/19-5/20 多轮 patch 实验** —— 修 line 388、NaN 清理、output 区分等,均失败但每次缩小诊断范围
5. **5/20 完整 checkpoint 链(FULL-CKPT)** —— 12 个 sync + try/except 点,精确到我们函数体内每一步
6. **5/20 关键发现** —— 我们函数体内所有 sync 都通过,但下一层 sdpa 入口 sync 必然失败 → bug 在函数外
7. **5/21 STUB 完全替换实验** —— 整个 sdpa 函数变 stub(直接返回 query.reshape),**不崩**,**确认 bug 100% 在函数体内**
8. **5/21 Bisection(三轮)** —— B1(只 varlen_pad)OK,B2(+SDPA)OK,B3(只 unpad)**崩**,**100% 锁定 `varlen_fa_seqlen_unpad` 为元凶**
9. **5/21 三条 workaround 路径设计**(Path A/B/C)讨论,选 Path A 实施
10. **5/21 Path A 实现** —— 不崩,初判"输出乱码"
11. **5/26 Path A 验证翻转** —— 发现 5/21 "乱码"判断是被 benchmark 坏 prompt 误导。改用真实长文章 + 真问题后,Path A **8/8 全过 + 输出语义正确 + `cached_lens=640` 硬证据**。Bug 1 实质解决。
12. **5/26 残留问题分类** —— 确认还剩 Bug 2(stop-token 泄漏)+ Bug 3(数值噪声 logit 污染),都跟 Path A 无关,是独立的小问题。

#### ⏳ 待做的(下一接手者可以从这里继续)

按优先级:

**(高)上报 MTT 修根因**:把 Path A 代码 + 这次的诊断报告(STUB 二分锁定 + 三档命中分析)发给摩尔线程,要他们改 `_kernels.so` 里 `varlen_fa_seqlen_unpad`(加输出 buffer 边界检查 / 接 KV cache 接口)。我们只是 Python 层 workaround,真根因在他们闭源 kernel 里。

**(中)修 Bug 2 — stop-token / chat template 泄漏**:
- vllm 启动加 `--stop-token-ids 151645`(Qwen2.5 的 `<|im_end|>`)
- 或客户端调用传 `extra_body={"stop_token_ids": [151645]}`
- 详见 [残留问题章节"类别 1"](#类别-1--chat-template--stop-token-残留配置层)

**(中)调研 Bug 3 — 数值噪声 logit 污染**:
- 先做 `temperature=0` 诊断,看污染程度(top-1 中没中)
- 试 `ENFORCE_EAGER=1` 排除 CUDA Graph 副作用
- 试 `--kv-cache-dtype fp16` 排除 KV cache BF16 精度
- 如果都不行,跟 H100 对比 logit 分布,把证据交给 MTT 查 MuDNN flash sdpa 精度
- 详见 [残留问题章节"类别 2"](#类别-2--数值噪声导致-logit-污染后端精度层)

**(低)Path A 工程化**:把 Path A 包装成正式 patch 提交给 vllm_musa 上游(而不是停在我们的临时 docker exec patch)

**(低)文档化**:把所有 patch 脚本整理成一个 git-format 的 patch series 文件

**(低)benchmark 扩展**:加 varied-length / multi-turn 等更多 scenario,现在只有 short + long_context

### 测试快捷指令(可立即跑)

```bash
# 在 gy_work(Path A patch 容器)跑 long_context,验证不崩 + 输出对(2026-05-26 推荐)
CONTAINER_NAME=gy_work \
  MODE=our SCENARIOS=long_context CONCURRENCY=1 REQUESTS=8 REQUEST_TIMEOUT=120 \
  TP=8 MAX_MODEL_LEN=32768 \
  MODEL_PATH=/data/SETS-2.0-test/models/Qwen2.5-14B \
  bash /data/my_vllm_test/wait_and_run.sh

# 在 gy_work_2(干净原版)复现 bug(已知会崩)
CONTAINER_NAME=gy_work_2 \
  MODE=our SCENARIOS=long_context CONCURRENCY=1 REQUESTS=8 REQUEST_TIMEOUT=120 \
  TP=8 MAX_MODEL_LEN=32768 \
  MODEL_PATH=/data/SETS-2.0-test/models/Qwen2.5-14B \
  bash /data/my_vllm_test/wait_and_run.sh

# 跑完后看输出 tokens 的语义
RUN=$(ls -td /data/my_vllm_test/runs/*/ | head -1)
cat $RUN/bench.json.outputs.txt

# 回滚 Path A 到原版(把 gy_work 还原成原版,做对照)
docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py \
    gy_work:/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py

# 验证关 prefix-caching 也是稳的(workaround B,不依赖 Path A)
CONTAINER_NAME=gy_work_2 MODE=our ENABLE_PREFIX_CACHING=0 \
  SCENARIOS=long_context CONCURRENCY=1 REQUESTS=8 \
  bash /data/my_vllm_test/wait_and_run.sh
```

### Path A patch 重新应用(如果回滚后想再尝试)

```bash
# 应用 patch
docker exec gy_work python3 /data/my_vllm_test/patches/path_a_python_concat.py

# 验证应用成功
docker exec gy_work grep -n "PATH-A 2026" /usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py
```

### 关键代码层次(完整说明在下方 5/21 章节)

```
benchmark.py 发请求
  ↓
vllm engine → Qwen2 model → attention layer
  ↓
vllm_musa/v0/flash_attn.py forward 函数
  ├── 调 reshape_and_cache_flash (line ~578) 把新 K/V 写到 KV cache
  ├── 走 prefill 分支(line ~606)
  │     ★ Path A patch 在这里加分支:
  │       检测 prefix-cache 命中 → 用 Python 实现 attention(绕过 3 个 C++ op)
  │       不命中 → 走原 sdpa_attention_with_kernel_seqlen_pad
  │     ↓ 原路径:
  │   sdpa_attention_with_kernel_seqlen_pad(line ~740,函数定义)
  │     ├── alloc q_pad/k_pad/v_pad (torch.empty)
  │     ├── alloc output (torch.empty)
  │     ├── ops.varlen_fa_seqlen_pad   ← C++ ext,已确认清白(B1)
  │     ├── torch.ops.aten._scaled_dot_product_attention_flash_musa
  │     │       ↓ (Layer 1 → 2 → 3 → 4)
  │     │   MuDNN ScaledDotProductAttention::RunFlash  ← 已确认清白(B2)
  │     └── ops.varlen_fa_seqlen_unpad  ★ ★ 元凶,B3 验证
  │           ↓ 越界写 output 缓冲区之后的相邻显存
  │           ↓ GPU context sticky error
  └── 走 decode 分支(line ~649+,用 paged attention,无问题)
```

### 一定要避免的几个坑(我们踩过)

1. **不要相信"sync 通过就没事"** —— MUSA driver/MuDNN 内部用自己的 stream 队列,`torch.musa.synchronize()` 抓不到所有错误。Sticky error 经常延迟到下游 op 才暴露
2. **不要把"NaN 在 padding 区"当成 bug 源** —— 这只是 `torch.empty` 的脏内存残留,kernel 没写到的位置自然就是脏的。zero-fill 不能解决根本问题
3. **不要相信单层 py-spy 栈** —— 它指向的位置往往是 Python 等 GPU 的"下一个 sync 点",**不是 GPU 真出错的地方**
4. **共享机器 GPU 上**:
   - **`docker exec gy_work` 内的 pkill 不要用宽 pattern**(`pkill -f 'multiprocessing.spawn'` 会误杀别人)—— 用具体 PID
   - **`FREE_THRESHOLD_GB` 至少 12**,最好 20+,因为 workshop 的测试会跟你抢 GPU
5. **改了 patch 必须 verify 它真的起作用**(我们犯过假设 patch 生效然后推导出错误结论的错)
6. **同一 deadlock 实例采样多次没意义**(死锁状态下栈不动,要换实例)

### 已知的 patch 文件清单(`/data/my_vllm_test/patches/`)

每个 patch 都是 idempotent 的 Python 脚本,跑两次会跳过:

| 脚本 | 作用 | 状态 |
|---|---|---|
| `apply_patch.py` | PATCH 1: DECODER 一律走 fast path | 历史,已被覆盖 |
| `diagnostic_patch.py` | 初版 DIAG | 历史,已被替换 |
| `full_checkpoint_chain.py` | FULL-CKPT(12 个 ckpt) | 历史 |
| `check_varlen_input.py` | CKPT-2.7 + 2.8(查 varlen 输入)| 历史 |
| `refine_ckpt6.py` | 精细化 CKPT-6(分真实区/padding区)| 历史 |
| `extend_ckpt_to_unpad.py` | 加 CKPT-8/9 包 unpad | 历史 |
| `bypass_sdpa_with_stub.py` | 整个函数 stub 化 | 历史 |
| `bisect_b1_only_varlen_pad.py` | B1:只 varlen_pad | 历史 |
| `bisect_b2_add_sdpa.py` | B2:+SDPA | 历史 |
| `bisect_b3_only_unpad.py` | B3:只 unpad,**确认元凶** | 历史 |
| `path_c_paged_prefill.py` | Path C:调 context_attention_fwd(Triton paged-aware) | ❌ 失败 — MUSA 环境 `triton.next_power_of_2` 不存在 |
| **`path_a_python_concat.py`** | **Path A:Python concat + SDPA** | ✅ **当前装版本 + 2026-05-26 验证语义正确** |

回滚顺序:这些 patch 是叠加式的,所以最快回滚是直接 docker cp 备份覆盖,而不是逐个撤。

---



## 🎯 2026-05-21 终极定位:`ops.varlen_fa_seqlen_unpad` 越界写

### 一句话结论(替代之前所有诊断)

**vllm_musa `_kernels.so` 里的 `varlen_fa_seqlen_unpad` C++ 算子,在 prefix-cache 命中场景下,会写飞 (seq_lens总和 - sum_seq) 行数据到 output 缓冲区之后的相邻显存,导致 GPU context 进入 sticky error 状态,延迟到下一次 GPU 操作时抛错。**

之前我们看到的所有症状(broadcast 死锁、MuDNNFlashSDPAFwd 失败、illegal memory access、Fill::Run 失败等)**都是这次 OOB 写之后的次生症状**,sticky error 在不同的下游 op 处冒出来,导致错误信息看起来五花八门。

### 完整调用链(用户请求 → kernel 的所有层次)

```
benchmark.py 发请求(HTTP POST → /v1/chat/completions)
  ↓
vllm API server (uvicorn 异步)
  ↓
MQLLMEngine.run_engine_loop    ← engine 主循环,每次迭代 = 一次 forward
  ↓
LLMEngine.step
  ↓
ModelExecutor.execute_model
  ↓
DriverWorker.execute_model     ← TP rank 0,带着 worker 们一起跑
  ↓
ModelRunner.execute_model
  ↓
Qwen2ForCausalLM.forward       ← 完整模型一次前向
  ↓
Qwen2Model.forward
  ↓
  for layer_idx in range(48):                     ← Qwen2.5-14B 有 48 层
    Qwen2DecoderLayer.forward:                     │
      ├── input_layernorm                          │  (RMSNorm)
      ├── self_attn (Qwen2Attention):              │
      │     ├── qkv_proj           (Linear)         │   产 Q/K/V 投影
      │     ├── rotary embedding (ROPE)             │   Q/K 位置编码
      │     ├── self.attn (Attention layer):        │
      │     │     ├── vllm/attention/layer.py       │   通用 attention 入口
      │     │     ├── vllm_musa/v0/flash_attn.py.forward
      │     │     └─→ sdpa_attention_with_kernel_seqlen_pad  ★ ★ ★ 元凶函数在这里
      │     │           ├── alloc q_pad/k_pad/v_pad (torch.empty)
      │     │           ├── alloc output (torch.empty)
      │     │           ├── ops.varlen_fa_seqlen_pad       ← 散布数据到 padded 布局
      │     │           ├── torch.ops.aten._scaled_dot_product_attention_flash_musa
      │     │           │     ↓ PyTorch dispatcher
      │     │           │     ↓
      │     │           │   torch_musa::MuDNNFlashSDPAFwd (Layer 2, SDP.cpp,开源)
      │     │           │     ↓ 调用
      │     │           │     ↓
      │     │           │   MuDNN::ScaledDotProductAttention::RunFlash (Layer 3,闭源 .so)
      │     │           │     ↓ 内部启动 ↓
      │     │           │     ↓
      │     │           │   GPU FlashAttention asm kernel (Layer 4,闭源,在 .so 二进制里)
      │     │           │     ↓ 写入 attn_out ↓
      │     │           │   attn_out 返回上来
      │     │           └── ops.varlen_fa_seqlen_unpad     ★ ★ ★ 元凶在这一行
      │     │                 ↓ 越界写到 output 缓冲区之后的相邻显存
      │     │                 ↓ GPU context 进入 sticky error
      │     │                 ↓ (我们看不见,延迟到下游 op 才显现)
      │     │
      │     └── o_proj            (Linear)         │   attention 输出投影
      ├── residual add                              │  (跨 attention 的残差)
      ├── post_attention_layernorm                  │  (RMSNorm)
      └── mlp (Qwen2MLP):                           │
            ├── gate_up_proj      (Linear)          │  FFN 上投影
            ├── SwiGLU activation                   │
            └── down_proj         (Linear)          │  FFN 下投影
  ↓
  final_norm                                        ← 模型最后一层 norm
  ↓
lm_head (vocab projection)                          ← 投影到词表概率
  ↓
Sampler                                              ← 选下一个 token
  ↓
返回给 caller / API
```

**核心点**:**一次完整 forward = 48 层 Qwen2DecoderLayer 串行执行**。每层都会调一次 `sdpa_attention_with_kernel_seqlen_pad`。所以**一次请求 prefill = 我们这个函数被调 48 次**(decode 阶段不走这个函数,走另一条 paged_attention 路径)。

TP=8 + 1 driver = 9 个进程,每个进程都跑自己那份代码,**一层 sdpa 调用就有 9 个进程独立执行**。所以每个 checkpoint 在一层里就会被打印 9 次(我们 log 里能看到的 9 条同 checkpoint 行就是这么来的)。

### 4 层抽象的详细分解(Layer 1 ~ Layer 4)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Python op 入口(PyTorch 注册)                                  │
│   torch.ops.aten._scaled_dot_product_attention_flash_musa               │
│                                                                          │
│   ▸ 这只是 PyTorch dispatcher 的一个注册名,没有"实现"                  │
│   ▸ 注册在 torch_musa/csrc/aten/ops/musa_functions.yaml:                │
│       - func: _scaled_dot_product_attention_flash_musa                  │
│         dispatch:                                                        │
│           PrivateUse1: MuDNNFlashSDPAFwd          ← 路由到 Layer 2      │
│                                                                          │
│   ▸ 我们能确切看到:✅(YAML 公开)                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ dispatcher 路由
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 2: torch_musa C++ wrapper(★开源,可看★)                          │
│   MuDNNFlashSDPAFwd in SDP.cpp                                          │
│   位置: /usr/local/lib/python3.10/dist-packages/torch_musa/csrc/        │
│         aten/ops/attention/mudnn/SDP.cpp                                │
│                                                                          │
│   ▸ 做的事:                                                              │
│      1. 校验 q/k/v 张量形状                                              │
│      2. 用 at::empty 分配 output(注意:不初始化,含残留)                │
│      3. 配置 MuDNN 的 SDPA op(SetCausal/SetEmbedDim/SetHeadsNum 等)    │
│      4. 调用 sdpa.RunFlash(handle, output, q, k, v, ...)                │
│      5. 返回 output                                                      │
│                                                                          │
│   ▸ 我们能确切看到:✅(完整 C++ 源码)                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ MuDNN API 调用
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 3: MuDNN 库 C++ API(❌闭源)                                       │
│   musa::dnn::ScaledDotProductAttention::RunFlash                        │
│   位置: /usr/local/musa-4.3.5/lib/libmudnn_*.so (二进制)                │
│   声明: /usr/local/musa-4.3.5/include/mudnn_xmma.h (★头文件公开★)      │
│                                                                          │
│   ▸ 从头文件知道存在但没有实现源码                                       │
│   ▸ 同 class 还提供其它备选 API(都没实现源码):                          │
│      - RunFlash         (定长 flash,带 padding)← 当前 SDPA 用的       │
│      - RunFlashVarlen   (变长 flash,带 cu_seqlens) ← 没用              │
│      - RunMath          (通用 GEMM,非 flash)← 没用                     │
│                                                                          │
│   ▸ 我们能确切看到:仅头文件声明,无实现                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ kernel launch
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 4: MuDNN 库内部 GPU asm kernel(❌闭源)                            │
│   FlashAttentionFwd AsmKernel(MUSA GPU 指令)                           │
│   位置: 编译进 libmudnn_*.so 二进制内                                   │
│                                                                          │
│   ▸ 5/19 当时一度怀疑这层有 bug(因为错误信息说 "FlashAttentionFwd      │
│      AsmKernel Error"),但 bisection 实验(B2 跑通)证明这层清白         │
│                                                                          │
│   ▸ 错误信息里的 "AsmKernel Error" 是 sticky error 的信使,不是凶手     │
│                                                                          │
│   ▸ 我们能确切看到:❌(机器码)                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 外层 vllm_musa Python 函数(我们做了所有 patch 的地方)

#### 关键参数符号约定

##### 三类 length 的直观图(取自 `flash_attn.py:107` 注释)

vllm 处理一条请求时是**多次迭代**(iteration)进行的,每次迭代算一段新 token,把它们的 K/V 写进 cache,下次迭代从 cache 接着算。这张图展示**同一条请求**在第 N-1 次和第 N 次迭代时的状态:

```
|---------- N-1 iteration --------|                       ← 上次迭代覆盖的范围
|---------------- N iteration ---------------------|      ← 本次迭代覆盖的范围
|- tokenA -|......................|-- newTokens ---|      ← 实际的 token 序列
|---------- context_len ----------|                       ← 本次"已在 cache 的"部分
|-------------------- seq_len ---------------------|      ← 本次"完整序列"长度
                                  |-- query_len ---|      ← 本次"新算的"部分
```

**逐行解读**:

| 这一行画的是 | 含义 |
|---|---|
| `\|--- N-1 iteration ---\|` | 上次迭代结束时,这条请求已经处理到这里。这部分 token 的 K/V **已经写进 KV cache** |
| `\|--- N iteration ---\|` | 本次迭代结束时会处理到这里(比 N-1 更长,因为又算了几个新 token) |
| `\|- tokenA -\|......\|- newTokens -\|` | 实际的 token 序列。`tokenA` 是第一个 token(占位符,代指开头的若干 token),中间 `......` 是中间的 token,`newTokens` 是本次新算的那段 |
| `\|--- context_len ---\|` | **本次不需要重新算的部分**(K/V 直接从 cache 里读)。**长度 = 上次 iteration 结束时的位置** |
| `\|--- seq_len ---\|` | 本次 attention 时 Q 要看的完整范围(从开头到 newTokens 结尾)|
| `\|-- query_len --\|` | **本次新算的 token 数**(走 Q/K/V projection + attention,然后 K/V 写进 cache 给下次用)|

| 名称 | 含义 | 简单理解 |
|---|---|---|
| **context_len** | 已经在 KV cache 里的 token 数 | "缓存里的旧账" |
| **query_len** | 本次新算的 token 数 | "本次要做的新账" |
| **seq_len** | 完整序列长度 = `context_len + query_len` | "总长度" |

不变式:`context_len + query_len == seq_len`

**对应到 prefix-cache 命中场景**:`context_len` 可能不是来自"同一条请求的 N-1 次迭代",而是**来自上一条请求(共享相同前缀)**留下的 cache。但 length 关系不变 —— 本次要算的还是只有 `query_len` 个新 token,完整范围还是 `seq_len`。

##### 批级别变量(实际函数接收的是这些,因为一次处理 batch_size 条请求)

| 符号(metadata 字段名) | 含义 | 跟单条 len 的关系 |
|---|---|---|
| `bs`(batch_size) | 这批 prefill 里有几条请求 | — |
| `seq_lens` / `seq_lens_tensor` | (bs,) 数组,**每条的完整长度** | `seq_lens[i] = seq_len_of_request_i` |
| `query_start_loc` | (bs+1,) cumsum 数组,**用来索引 query 张量切片** | `query_start_loc[i+1] - query_start_loc[i] = query_len[i]` |
| `num_prefill_tokens` | 本批新算的 token 总数 | `= sum(query_len) = query.shape[0]`,**只算新 token,已 cache 的不算** |
| `max_query_len` | 本批最长那条的 query_len | `= max(query_len[i])` |
| `max_prefill_seq_len` | 本批最长那条的完整长度 | `= max(seq_lens[i])` |

注意:函数代码里有时直接用 `sum_seq, h_q, d_q = query.shape` 提取第 0 维 —— 这个 `sum_seq` 在数值上 **= `num_prefill_tokens` = `sum(query_len)`**。下面函数代码和例子里 `sum_seq` 都是这个意思。

##### 健康场景 vs bug 场景的 length 关系

| 场景 | context_len[i] | query_len[i] | seq_lens[i] | num_prefill_tokens |
|---|---|---|---|---|
| fresh prefill(无 cache) | 0 | seq_len | seq_len | `sum(seq_lens)` |
| **partial 命中** ★ | > 0 | < seq_len | seq_len | `< sum(seq_lens)` |
| 100% 命中 | seq_len | 0 | seq_len | (不进此函数,走 decode) |

⇒ **bug 触发条件用 length 关系表达**:`num_prefill_tokens < sum(seq_lens)`(等价于"有某条请求 context_len[i] > 0")

##### 其他符号

| 符号 | 含义 | 由谁决定 |
|---|---|---|
| `h_q` / `h_kv` | Q head 数 / KV head 数(GQA 时 h_kv < h_q,被 TP 切再除) | 模型 + TP rank |
| `d_q` / `d_kv` | head 维度(每个头的 vector 长度) | 模型固定 |

#### 函数代码

```
/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py

def sdpa_attention_with_kernel_seqlen_pad(query, key, value, seq_lens, max_prefill_seq_len, is_causal):
    bs = seq_lens.shape[0] - 1                      ← batch 数
    sum_seq, h_q, d_q = query.shape                 ← sum_seq = num_prefill_tokens = sum(query_len)
    _, h_kv, d_kv = key.shape                       ← KV head 数(GQA / TP 后)

    # 分配 4 个 GPU buffer
    q_pad  = torch.empty((bs, h_q,  max_prefill_seq_len, d_q), ...)  ← 第 2 维用 max_seq
    k_pad  = torch.empty((bs, h_kv, max_prefill_seq_len, d_kv), ...)
    v_pad  = torch.empty((bs, h_kv, max_prefill_seq_len, d_kv), ...)
    output = torch.empty((sum_seq, h_q, d_q),                  ...)  ← 第 0 维用 sum_seq(小)!

    # 第 1 个 op:紧凑 → padded
    ops.varlen_fa_seqlen_pad(query, key, value,
                              q_pad, k_pad, v_pad,
                              seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    # 把 (sum_seq, h, d) 散布到 (bs, h, max_prefill_seq_len, d)
    # 每条 batch i,只填 query_len[i] 行到位置 [0:query_len[i]],剩下的 padding 区不写

    # 第 2 个 op:attention 核心计算 → 通过 Layer 1 → Layer 2 → Layer 3 → Layer 4
    attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
        q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    # 输出 attn_out shape = (bs, h_q, max_prefill_seq_len, d_q)

    # 第 3 个 op:padded → 紧凑 ★ ★ ★ 元凶在这一行 ★ ★ ★
    ops.varlen_fa_seqlen_unpad(attn_out, output,
                                seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    # 预期:从 attn_out 各 batch 取前 query_len[i] 行 → 紧凑写到 output(共 sum_seq 行)
    # 实际(buggy):**按 seq_lens[i] 而不是 query_len[i] 决定每条写多少行**
    #   → 当 num_prefill_tokens(=sum_seq) < sum(seq_lens) 时,
    #      总写入量 = sum(seq_lens) > output buffer 容量(sum_seq)
    #   → OOB 越界写

    return output.view(-1, h_q * d_q)
```

#### 具体例子(让上面函数代码具象化)

下面用一个**具体的小例子**(prefill 单 batch,seq_lens=[904],cached 896,新 8 个 token)展开,看实际数值是怎么对应到 OOB 写的。

##### 例子设定
- 1 条请求(`bs=1`)
- `context_len[0] = 896`(前 896 token 已在 KV cache)
- `query_len[0] = 8`(本次只新算尾部 8 个 token)
- `seq_len[0] = 904`(完整长度 = context + query)
- `h_q=5`(单 rank,TP=8 之前是 40)、`h_kv=1`(GQA 8:1,TP=8 后只剩 1)
- `d_q=d_kv=128`

##### 各参数实际值

| 参数 | 值 | 怎么来的 |
|---|---|---|
| `query.shape` | `(8, 5, 128)` | 第 0 维 = `sum(query_len) = 8` |
| `key.shape` / `value.shape` | `(8, 1, 128)` | 第 0 维 = `sum(query_len) = 8` |
| `sum_seq`(= `num_prefill_tokens`) | **8** | `sum(query_len)` |
| `seq_lens`(tensor) | `[0, 904]`(cumsum 形式) | 单条 batch,`seq_len[0]=904` |
| `max_prefill_seq_len` | **904** | `max(seq_lens) = 904` |
| `q_pad.shape` | `(1, 5, 904, 128)` | `(bs, h_q, max_prefill_seq_len, d_q)` |
| `k_pad.shape` / `v_pad.shape` | `(1, 1, 904, 128)` | 同上但 h_kv=1 |
| `output.shape` | **`(8, 5, 128)`** | `(sum_seq, h_q, d_q)` ← **小 buffer!** |
| `attn_out.shape` | `(1, 5, 904, 128)` | SDPA 输出 padded 大 buffer |

##### 各 op 行为

```
pad     :  query (8, ...)  →  q_pad (1, ..., 904, ...)
            [前 8 行写入有效数据,后 896 行 padding 区为 alloc 残留]

SDPA    :  q_pad (1, ..., 904, ...)  →  attn_out (1, ..., 904, ...)
            [SDPA 实际算了 max=904 长度的 attention,但前 896 K/V 全是脏数据!
             attention 数学其实已经错了,但这一步不会崩]

unpad   :  attn_out (1, ..., 904, ...)  →  output (8, ...)
            预期写 8 行(sum_seq=8)
            实际写 904 行(seq_lens[0]=904)
            → output 只能容纳 8 行 (8 * 5 * 128 = 5120 个 float)
            → 实际写 904 * 5 * 128 = 578560 个 float
            → 多写 (578560 - 5120) ≈ 573440 个 float 到 output 后面的相邻 GPU 显存
            → 污染 GPU context → 后续任何 op 都可能崩
```

##### 为什么短 prompt + prefix-cache、长 prompt + 无命中、workshop 不挂

> **本表里的 `max_seq` 是 `max_prefill_seq_len` 的简写**(就是上面"关键参数符号约定"里定义过的:`max(seq_lens[i])`,本批最长那条的完整长度)。`sum_seq` 是 `num_prefill_tokens = sum(query_len)`。下面这一列展开就是问:**这一批所有新算的 token 加起来,跟"最长那条完整长度 × 几条"哪个大**

| 场景 | sum_seq | max_seq | 关系 | 是否 OOB |
|---|---|---|---|---|
| 短 prompt(无 cache 命中) | ≈ max_seq | 同 | `sum_seq == max_seq`(query_len = seq_len) | ❌ 正好填满 |
| 长 prompt + 无 cache 命中(请求 #0) | 904 | 904 | 相等(query_len = seq_len = 904) | ❌ 正好填满 |
| 长 prompt + 部分命中(★ 我们例子) | 8 | 904 | `sum_seq < max_seq`(query_len=8 < seq_len=904) | ✅ **越界 896 行** |
| 短 prompt + 完整 cache 命中 | (不进 prefill,走 decode) | - | - | ❌ 不调用此函数 |
| workshop 关 cache | sum_seq = max_seq | 同 | 相等 | ❌ 不触发 |

这就是为什么"长 prompt + 实际命中 prefix-cache"才挂 —— 其他场景下 `sum_seq` 都不会小于 `max_seq × bs`(即上面"关键参数符号约定"里的 `num_prefill_tokens < sum(seq_lens)`),buffer 大小不对称就不存在。

> 注:单 batch(bs=1)时 `max_seq × bs == sum(seq_lens) == seq_lens[0]`,所以表里直接拿 `sum_seq vs max_seq` 比就行;多 batch 时严格说要比 `sum_seq vs sum(seq_lens)`,本质一样。

> ⚠️ 上面 8/904 只是**举例**让你直观看见数字关系。**真实场景下数字会变化**:
> - 多 batch 时,`max_prefill_seq_len = max(seq_lens)`,而 `sum_seq = sum(query_len)`
> - 例如 `bs=4`,各 batch 的新 token 数 = [8, 16, 4, 32],完整长度 = [904, 512, 256, 1200],那么 `sum_seq=60`,`max_prefill_seq_len=1200`,OOB 越界量也变化
> - **重点不是具体数字,是"sum_seq < max_seq × bs(即 unpad 误用 seq_lens 而非 query_len)" 这个不对称关系**

#### 三个 op 总结

三个 op 都是 vllm_musa C++ 扩展(`_kernels.so` 编译二进制),都没源码:

| op | 位置 | 是否真凶 |
|---|---|---|
| `ops.varlen_fa_seqlen_pad` | vllm_musa C++ ext | ❌ 清白(B1 实验) |
| `_scaled_dot_product_attention_flash_musa` | 走 Layer 1-4 链路 | ❌ 清白(B2 实验) |
| **`ops.varlen_fa_seqlen_unpad`** | vllm_musa C++ ext | ✅ **元凶(B3 实验)** |

### prefix-cache 命中如何让 sum_seq < sum(seq_lens) → 触发 OOB(⚠️ 推测,非看代码确认)

> **重要声明**:本节是**对 `varlen_fa_seqlen_unpad` 内部行为的推测**,不是基于源码的事实。`varlen_fa_seqlen_unpad` 是 `vllm_musa/_kernels.so` 里的 **C++ / MUSA 闭源 kernel**,我们看不到源码。下面的解释是**根据 STUB 二分实验结果 + 接口签名 + 函数命名 + 跟 `varlen_fa_seqlen_pad` 的对称性,反推出的最可能机制**。
>
> 实际的内部循环逻辑可能跟我们推测的不完全一样(比如可能不是逐行 copy,可能用了更复杂的 layout 重排),但**实验事实是确定的**:
> - ✅ B3 实验确认:只调用 `varlen_fa_seqlen_unpad` 就会崩 → bug 100% 在这个函数内部
> - ✅ Path A bypass 它就不崩 → 跟它的 sum_seq 和 max_seq 处理逻辑有关
> - ⚠️ 具体到底是哪一行 / 用什么循环越界写,**没看到源码无法 100% 确认**
>
> 要拿到 100% 确切答案,需要让 MTT 提供 `varlen_fa_seqlen_unpad` 源码,或者他们自己复现并定位。下面这套解释是**给 MTT 看的起点假设**。

#### 术语:什么是 OOB?

**OOB = Out-Of-Bounds**,中文叫"越界"。指 **GPU kernel 写超出了它本应写的 buffer 范围**:
- 假设给一个 `output` buffer 分配了空间能放 8 行
- kernel 写的时候认为要写 904 行
- 多写的 896 行就写到了 buffer **后面相邻的 GPU 显存里**
- 相邻显存可能正好是别的 tensor 的存储空间,被覆盖后就**污染了 GPU context**
- 下游任何 op 读到那块脏数据 → 计算出错 / 崩溃,这就是我们看到的 "sticky GPU error"

#### 推测的机制

`varlen_fa_seqlen_unpad` 的工作是把 padded 张量 `attn_out [bs, h, max_prefill_seq_len, d]` 拆回紧凑张量 `output [sum_seq, h*d]`。我们推测它内部做的事情是:

- 对每条 batch i,从 `attn_out[i, :, ?:?, :]` 取出**真实需要返回的行数**
- 把它们顺序拼到 `output` 里
- 总行数 = `sum_seq`(`output` 的第 0 维大小)

**推测的关键问题**:这个 op 内部怎么决定"每条 batch 取多少行"?

我们的猜想是:它使用 `seq_lens[i]` 作为该条 batch 取多少行的依据。理由:
- 函数接口签名里传了 `seq_lens` 这个参数
- 函数名是 `seqlen_pad/unpad`,暗示它的工作单位是 `seq_lens`
- 在原本设计里(无 prefix-cache),`query_len[i] == seq_lens[i]`,所以两种解释下行为一样

在原本设计里(无 prefix-cache),按 `seq_lens[i]` 写就是对的(因为整条请求都新做 prefill,新 token 数 = 完整长度)。所以 op 内部循环写 `seq_lens[i]` 行到 `output`,总行数刚好 `sum(seq_lens) == sum_seq`,没问题。

**但 prefix-cache 命中时**:
- 这条请求**只有新尾部 token 走 prefill**,新 token 数 `query_len[i] < seq_lens[i]`
- `sum_seq = sum(query_len) < sum(seq_lens)`
- `output` buffer 是按 `sum_seq` 分配的(小)
- 如果 op 真的还在按 `seq_lens[i]` 决定每条写多少行 → 总写入量 `sum(seq_lens) > sum_seq` → **OOB 越界**

如果这个推测对,**bug 的本质是**:这个 op 的接口设计里没有"新 token 数(query_len)"这个参数,只能用 `seq_lens` 推断。在 fresh prefill 场景下这个推断恰好成立,但 prefix-cache 命中场景下假设崩塌,op 没收到任何"该假设不再成立"的信号,继续按老规矩写就越界了。

#### MTT 验证方向

如果 MTT 看到这份报告,建议他们:
1. 读 `varlen_fa_seqlen_unpad` 源码,验证内部循环是不是真的用 `seq_lens` 决定每条 batch 的输出行数
2. 如果是:加 `query_lens` 参数,改用 `query_lens[i]` 决定写多少行
3. 如果不是(我们猜错了具体机制):看实际是哪一步越界,但**最终修法应该都是让 op 知道"新 token 数 ≠ 完整长度"**

### 那 vllm_musa V0 后端在 prefix-cache 支持上是怎么个状态?

实质上是**不完整**:
- vllm engine 把切片 + metadata 都正确传到了 backend
- 但 backend 的 prefill 路径(就是上面这个函数)**没从 KV cache 拉 cached 部分的 K/V**
- 它只把"新 8 个 token"喂给 attention,导致 attention 数学也不对(算 8 个新 Q 对 8 个新 K/V 的 attention,而不是对完整 904 K/V 的 attention)
- unpad 的越界写只是其中一个最显著的崩点;即使 unpad 修了,数学错误还在

→ 这意味着 vllm_musa V0 后端**根本没真正支持 prefix-caching 命中场景**。workshop 自己脚本里不开 `--enable-prefix-caching` 不是偶然,是因为这条路径本来就没经过端到端验证。

### 诊断方法学(可复用)

本次定位用了两类工具:

**1. 细粒度 Checkpoint 链(CKPT)**

在 `sdpa_attention_with_kernel_seqlen_pad` 函数体里**每一行 GPU 操作前后插入 `torch.musa.synchronize()` + try/except**,每个 checkpoint 带唯一编号 + 上下文 dump:

```python
[CKPT-0]  入口 sync → catch 上一层留的 sticky error
[CKPT-1]  seq_lens.cpu() 之后
[CKPT-2]  alloc q/k/v_pad 之后
[CKPT-2.7] 检查 varlen 输入是否含 NaN
[CKPT-2.8] 主动清干净 varlen 输入(用 nan_to_num_)
[CKPT-3]  varlen_fa_seqlen_pad 之后
[CKPT-3.5] 检查 q_pad/k_pad/v_pad 含 NaN/Inf 情况
[CKPT-4]  nan_to_num_ 清理 q/k/v_pad 之后
[CKPT-4.5] 验证清理生效
[CKPT-5]  SDPA flash kernel 之后
[CKPT-6]  分区检查 attn_out:真实区[0:sum_seq] vs padding 区[sum_seq:max]
[CKPT-7]  SDPA 之后 sync
[CKPT-8]  varlen_fa_seqlen_unpad 之后
[CKPT-8.5] 检查 unpad output 是否含 NaN
[CKPT-9]  函数出口最终 sync
```

**用途**:每条 GPU 语句后强制同步,出错时第一时间归到那条具体语句。但实测发现:**varlen_unpad 的 OOB 不会立即被这种 sync 抓到**(它可能用 MuDNN 内部 stream / 命令队列,默认 stream 同步抓不到),sticky error 延迟到下一个 GPU op 才暴露。

**2. STUB(桩函数)+ Bisection**

把整个 `sdpa_attention_with_kernel_seqlen_pad` 函数体**替换成一个"什么 GPU op 都不做、只返回 query.reshape"的桩**:

```python
def sdpa_attention_with_kernel_seqlen_pad(query, key, value, seq_lens, max, is_causal=True):
    sum_seq, h_q, d_q = query.shape
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()
```

实验结果:**8/8 全过,完全不崩**。证明 bug 100% 在这三个 op 之一:`varlen_fa_seqlen_pad`、`_scaled_dot_product_attention_flash_musa`、`varlen_fa_seqlen_unpad`。

然后逐个加回 op 做 bisection:

| 实验 | 配置 | 结果 | 排除 |
|---|---|---|---|
| **STUB** | 三个 op 全 skip,只返回 query.reshape | ✅ 不崩(8/8 OK) | 排除"caller 端" |
| **B1** | 只跑 varlen_fa_seqlen_pad,SDPA + unpad 跳过 | ✅ 不崩(8/8 OK) | 排除 varlen_fa_seqlen_pad |
| **B2** | varlen_pad + SDPA,unpad 跳过 | ✅ 不崩(8/8 OK) | 排除 SDPA(MuDNN flash kernel) |
| **B3** | 只跑 varlen_fa_seqlen_unpad,其余跳过 | ❌ **崩**(MUSA err 700)| **锁定 varlen_fa_seqlen_unpad** |
| 原函数 | 三个 op 全跑 | ❌ 崩(已知) | |

→ **唯一会导致崩的就是 varlen_fa_seqlen_unpad**,通过排除法 + 单独触发都验证了。

### 元凶机制详解

`varlen_fa_seqlen_unpad` 的签名:
```python
ops.varlen_fa_seqlen_unpad(
    attn_out,             # shape (bs=1, h_q=5, max=904, d_q=128) — SDPA 输出
    output,               # shape (sum_seq=8, h_q=5, d_q=128) — 紧凑目标,只 alloc 了 8 行!
    seq_lens,             # [0, 904] ← metadata 声称 batch 0 长度 904
    sum_seq=8,
    max_prefill_seq_len=904,
    d_q, h_q, bs
)
```

**问题**:
- `output` tensor 只 alloc 了 `sum_seq=8` 行
- 但 `seq_lens` 声明 batch 0 长度 `=904`
- unpad 内部很可能按 `seq_lens` 循环写 904 次到 `output`
- → **越界写 (904 - 8) = 896 行**到 `output` 缓冲区后面的相邻显存

这 896 行越界写**污染了 GPU 上某段未知的显存**(可能是别的请求的 KV cache、别的 op 的 scratch、其它 tensor 数据等等),GPU context 进入错误状态,**下一次任何 GPU 操作都会撞到 err 700 (illegal memory access)**。

由于 MUSA driver / MuDNN 内部 stream 的异步特性,sticky error **不会在 unpad 调用本身报出来**(我们的 sync 抓不到),延迟到下游某个 op 才暴露。在不同的运行里,暴露点分别在过:
- `MuDNNFlashSDPAFwd` (next attention layer)
- `Fill::Run` (next torch.zeros)
- `Permute::Run` (next .contiguous())
- `Run SDPA Flash FWD`
- 直接 raw `MUSA error: illegal memory access`

**这些都是 sticky error 在不同 op 上的"信使",真凶都是同一个上游 varlen_fa_seqlen_unpad 的 OOB 写**。

### 为什么 prefix-cache 命中时才触发

正常 prefill(无 prefix-cache 命中):
- `sum_seq = max_prefill_seq_len`(都等于完整 prompt 长度)
- output alloc `(sum_seq, h_q, d_q)`,unpad 按 `seq_lens` 写 sum_seq 行 → 不越界

prefix-cache 命中后(vllm engine 切片到只剩"新 token"):
- `sum_seq = 新 token 数(很小,如 8)`
- `max_prefill_seq_len = 完整序列长(如 904,包含已缓存的)`
- `seq_lens` 还是 `[0, 904]`
- → unpad 按 904 循环写 8 行的 output → **写飞 896 行**

### 为什么之前的所有诊断都被这一个 bug 解释了

| 历史症状 | 现在的解释 |
|---|---|
| 5/9 "TP broadcast 死锁" | varlen_unpad 越界写污染了 broadcast 用的 shm 元数据,后续 broadcast 等不到合法块 |
| 5/19 "MuDNNFlashSDPAFwd MUDNN failed" | varlen_unpad 越界写后,下次 SDPA 调用撞到坏 context |
| 5/19 "Fill::Run failed" | 同上,下次 `torch.zeros` 撞到坏 context |
| 5/19 "shm_broadcast 60s 超时" | 同上,worker 等 driver 广播,但 driver 卡在坏 GPU state |
| 5/20 "padding 区有 NaN" | torch.empty alloc 残留,不是 bug 也不是凶手 |
| 5/20 "SDPA 真实区干净,padding 区脏" | SDPA 没 bug,只写它该写的位置;脏 padding 区是 alloc 残留 |

→ **所有都源于同一个根因**:varlen_unpad 的 OOB 写。

### 修复建议

**短期 workaround(用户侧)**:关 `--enable-prefix-caching`,这是已知唯一稳定的生产方案

**正确修复(MTT 侧)** —— vllm_musa `_kernels.so` 内部的 `varlen_fa_seqlen_unpad` 实现需要修:

```cpp
// 现在(推测的 buggy 实现)
for (int b = 0; b < bs; b++) {
    int seg_len = accum_lens[b+1] - accum_lens[b];  // 904
    for (int i = 0; i < seg_len; i++) {
        output[offset + i] = ...;                    // 当 offset+i > sum_seq 时越界
    }
    offset += seg_len;
}

// 修复方案 A:限制写入计数
for (int b = 0; b < bs; b++) {
    int seg_claim = accum_lens[b+1] - accum_lens[b];
    int seg_actual = std::min(seg_claim, sum_seq - offset);  // ★ 增加上限
    for (int i = 0; i < seg_actual; i++) {
        output[offset + i] = ...;
    }
    offset += seg_actual;
}

// 修复方案 B(更彻底):上层调用方传 query_lens(真实 token 数)和 seq_lens(完整长度)两个,
// unpad 用 query_lens 决定写多少
```

**长期方案(根本性)** —— vllm_musa V0 后端的 prefill 路径**没有真正实现 prefix-cache 命中场景的 attention**:
- 它只把"新 token 的 Q/K/V"喂给 attention,没从 KV cache 拉 cached K/V
- 即使 unpad 不越界,attention 数学也是错的(cached prefix 没参与注意力)
- 需要给 backend prefill 路径加上"用 block_tables 拉 cached K/V 拼接"的完整逻辑

### 这次诊断使用的 patch 序列(给 MTT 复现用)

实验过程中累计在 `vllm_musa/v0/flash_attn.py` 写过的 patch(时间顺序):

| Patch 名 | 关键改动 | 实验结论 |
|---|---|---|
| `PATCH 1`(行 609-619)| DECODER 强制走快路径 | 不影响 bug |
| `FULL-CKPT`(12 个 ckpt)| 每个 GPU op 前后 sync + try/except | 我们函数体内全过,bug 在外 |
| `CHECK-INPUT`(CKPT-2.7/2.8)| 验证 + 清干净 varlen 输入 | 输入本来就干净,NaN 是 alloc 残留 |
| `STUB-BYPASS`(整个函数 stub)| 函数体全 skip,返回 query.reshape | **不崩** → bug 100% 在函数体内 |
| `BISECT-B1`(只跑 varlen_pad)| 其它 STUB | 不崩 → pad 清白 |
| `BISECT-B2`(+ SDPA)| 只 unpad STUB | 不崩 → SDPA 清白 |
| `BISECT-B3`(只跑 unpad)| pad+SDPA STUB | **崩** → **unpad 是元凶** |
| `path_c_paged_prefill`(调 context_attention_fwd)| 用 vllm 主仓 Triton kernel | 失败:`triton.next_power_of_2` 不存在 |
| **`path_a_python_concat`**(当前装的)| Python 实现 prefix-cache 处理 | ✅ **不崩 + 语义正确**(2026-05-26 验证) |

诊断脚本路径:`/data/my_vllm_test/patches/*.py`

### Run 目录(供后续追溯)

- bisection 实测结果(2026-05-21):
  - STUB: `runs/20260520_094607_our/` ✅
  - B1: `runs/20260520_102340_our/` ✅
  - B2: `runs/20260521_022703_our/` ✅
  - B3: `runs/20260521_023431_our/` ❌(`permute_to_contiguous MUDNN failed`,err 700)

### 还未验证的边角问题(留给 MTT)

- [ ] unpad 内部的具体 loop 逻辑是不是按 seq_lens 循环的(需要 MTT 内部看 `_kernels.so` 源码确认)
- [ ] 是否所有 TP 都会触发,还是只 TP=8(虽然原理上跟 TP 大小无关,但没测过 TP=4)
- [ ] varlen_fa_seqlen_pad 的对称越界(读越界)—— 没有触发崩,但可能产生错误数据(读到别的内存当成 query)

### 历史调查 → 修正

5/9 起当时叫"broadcast 死锁",5/19 强化为"MuDNNFlashSDPAFwd AsmKernel"的 bug,这两个**都是症状,不是根因**。今天的 bisection 把根因精确到 `varlen_fa_seqlen_unpad`,完整证据链俱备。

---

## 🔧 2026-05-21 后续:Path A workaround 详细状态 + 下一步调试线索

### Path A 设计原理(为什么这样写)

bisection 锁定 `varlen_fa_seqlen_unpad` 是元凶后,我们想做一个"绕开它"的 workaround。三条候选路径(详见上文"修复建议"):

| 路径 | 想法 | 是否可行 |
|---|---|---|
| A: Python concat + SDPA | Python 层手动从 KV cache 取 cached K/V,concat 上 new K/V,padded buffer,调 SDPA | ✅ 可做,Triton 不参与 |
| B: 修 C++ unpad | C++ 加边界检查 + 接 KV cache 接口 | ❌ MTT 才能改 |
| C: 用 vllm 主仓 paged kernel | 调 `context_attention_fwd`(Triton paged-aware) | ❌ MUSA 环境 `vllm.triton_utils.triton` 被替换成 placeholder,`triton.next_power_of_2` 等 host 工具不存在,Triton kernel 也跑不起来 |

**选了 Path A**,理由:
- 完全 Python 实现,我们能动
- 用的是 vllm_musa 已验证稳定的 `_scaled_dot_product_attention_flash_musa`(B2 实验确认了它本身没问题,问题在 unpad)
- Triton 不参与

### Path A 代码逻辑(`/data/my_vllm_test/patches/path_a_python_concat.py`)

插入到 `flash_attn.py` 的 forward 函数 prefill 分支开头,检测 prefix-cache 命中:

```python
if prefill_meta := attn_metadata.prefill_metadata:
    # ===== PATH-A: prefix-cache 命中场景走 Python concat + SDPA =====
    _path_a_hit = (
        self.attn_type == AttentionType.DECODER
        and kv_cache.numel() > 0
        and prefill_meta.block_tables is not None
        and prefill_meta.block_tables.numel() > 0
    )
    if _path_a_hit:
        # 拆 KV cache
        key_cache = kv_cache[0]   # (num_blocks, block_size, h_kv, d_kv)
        value_cache = kv_cache[1]
        block_size = key_cache.shape[1]

        # 算每个 batch 的 cached/new 长度
        block_tables = prefill_meta.block_tables           # (bs, max_blocks)
        seq_lens_tensor = attn_metadata.seq_lens_tensor    # (bs,) 完整长度
        query_start_loc = getattr(attn_metadata, 'query_start_loc', None)
        # query_start_loc 给出每个 batch 新 token 的累积起始
        # new_lens[b] = query_start[b+1] - query_start[b]
        # cached_lens[b] = seq_lens[b] - new_lens[b]

        # 准备 padded buffers
        q_pad = torch.zeros((bs, h_q,  max_full_len, d_q), ...)
        k_pad = torch.zeros((bs, h_kv, max_full_len, d_kv), ...)
        v_pad = torch.zeros((bs, h_kv, max_full_len, d_kv), ...)

        for b in range(bs):
            cached_len, new_len, full_len, new_start = ...
            # 1. 用 block_tables[b] 从 KV cache 拉 cached K/V,写到 [0:cached_len]
            if cached_len > 0:
                num_blk = (cached_len + block_size - 1) // block_size
                blk_ids = block_tables[b][:num_blk]
                cached_k = key_cache[blk_ids].reshape(-1, h_kv, d_kv)[:cached_len]
                cached_v = value_cache[blk_ids].reshape(-1, h_kv, d_kv)[:cached_len]
                k_pad[b, :, :cached_len, :] = cached_k.transpose(0, 1)
                v_pad[b, :, :cached_len, :] = cached_v.transpose(0, 1)

            # 2. 把新 Q/K/V 写到 [cached_len:full_len]
            new_q = query[new_start:new_start+new_len]   # (new_len, h_q, d_q)
            new_k = key  [new_start:new_start+new_len]
            new_v = value[new_start:new_start+new_len]
            q_pad[b, :, cached_len:full_len, :] = new_q.transpose(0, 1)
            k_pad[b, :, cached_len:full_len, :] = new_k.transpose(0, 1)
            v_pad[b, :, cached_len:full_len, :] = new_v.transpose(0, 1)

        # 调 SDPA(B2 已验证稳定)
        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=True)

        # 从 attn_out 抽取每个 batch 的新 query 输出位置 [cached_len:full_len]
        output = torch.empty((sum_new, h_q * d_q), ...)
        for b in range(bs):
            seg = attn_out[b, :, cached_len:full_len, :].transpose(0, 1).reshape(new_len, h_q * d_q)
            output[new_start:new_start+new_len] = seg

        return output
    # 不命中走原 sdpa_attention_with_kernel_seqlen_pad ...
```

### Path A 实测结果(2026-05-21 初版 → 2026-05-26 修正)

> ⚠️ **本节是 2026-05-21 写的初版,结论已经被 2026-05-26 的对照实验推翻**。
> **新结论**:Path A 实际工作正确,之前看到的"乱码"主要来自 benchmark 的坏 prompt(6 倍重复同一段话)
> 详见本文顶部 "🎉 2026-05-26 更新:Path A 验证通过" 章节
> 下面这一节保留作为历史诊断记录,但不要据此推导新动作

**运行**:`MUSA_BLOCKING=1 ... SCENARIOS=long_context bash wait_and_run.sh`

**Run 目录**:`/data/my_vllm_test/runs/20260521_075551_our/`

**当时结果**:
- ✅ **不崩**:第 1 / 2 轮都是 4/4 OK,无 MUSA error,无 sticky error
- ✅ tokens 数:每条 200 tokens out(被 max_tokens 限制)
- ✅ TTFT 145-219ms,TPOT ~17ms — 性能正常
- ❌ **输出 tokens 语义错乱**(看 `bench.json.outputs.txt` sidecar)— **后来发现是 benchmark prompt 的锅,见顶部 2026-05-26 章节**

**乱码示例(Request #0)**:
```
请总结以下文本的要点，用三句话：

Transformer 架构由 Vas blanco 等人在 2017 年提出...
```

模型在**重复输入 prompt 文本,夹杂随机 token**:
- `Vas blanco`(应该是 `Vaswani`)
- `_OUTPUT_`(没意义的占位)
- `(storage)`、`Sidebar`、`<Category>`、`Loukas`、`}}>`、`confrontation`
- `奇葩`、`PROJ`
- `פארגראление`(希伯来字符)、`afraidution strategies`、`等着`

Request #2、#3 甚至跑题到**完全无关的"情感分析"主题**,显然 attention 计算输出值偏离严重,导致 sampler 取到错误概率分布。

### 关键诊断:为什么 Request #0 也乱?

这个最值得调:**Request #0 是冷启动,没有 cache hit**,理论上 cached_len=0,Path A 的行为应该跟原 `sdpa_attention_with_kernel_seqlen_pad`(fresh prefill)等价。但 Request #0 输出也是乱的。

判定:**Path A 的"散布到 padded buffer + SDPA"步骤实际上跟原 `varlen_fa_seqlen_pad → SDPA` 不等价**,虽然形状对得上,但值不对。

### 接手者的 3 个最可能调试方向(按可行性排序)

#### 方向 1(最便宜,先试):.contiguous() 让 transpose 内存连续

`.transpose(0, 1)` 返回的张量内存不连续(只是改 stride 视图)。SDPA 可能按"以为是连续布局"的 stride 读,读到错的数据。

**改法**:在所有 `.transpose(0, 1)` 后加 `.contiguous()`:

```python
# 修改前(当前 Path A):
q_pad[b, :, cached_len:full_len, :] = new_q.transpose(0, 1)

# 修改后:
q_pad[b, :, cached_len:full_len, :] = new_q.transpose(0, 1).contiguous()
```

理由:`q_pad` 本身是 `torch.zeros` 出来的连续 tensor;但赋值时右侧 view 是非连续的,可能触发非预期行为。

#### 方向 2(中等,效果好):Request #0(无 cache hit)走原路径

让 `_path_a_hit` 不仅判断"block_tables 非空",还判断"是否真有 cached prefix":

```python
# 改 trigger 条件:只有"实际命中(cached_len>0)"才走 Path A
_path_a_hit = (
    self.attn_type == AttentionType.DECODER
    and kv_cache.numel() > 0
    and prefill_meta.block_tables is not None
    and prefill_meta.block_tables.numel() > 0
    # ★ 新加:还要看是否真的有 cached 部分
    and (attn_metadata.num_prefill_tokens < int(attn_metadata.seq_lens_tensor.sum().item()))
)
```

这样 Request #0 自动走原 `sdpa_attention_with_kernel_seqlen_pad`(对 fresh prefill 是稳定的)。Path A 只处理真正命中的请求。

预期:Request #0 立刻正确;Request #1+ 仍可能乱(就能孤立调试 Path A 的核心逻辑)。

#### 方向 3(更彻底,对账诊断):加 print 比较两个版本的 q_pad

在 Path A 代码里加 print,跟原版 `varlen_fa_seqlen_pad` 的输出对比:

```python
# 在 Path A 算完 q_pad 之后,也跑一次原版 op 对比
import torch
diag_q_pad = torch.empty((bs, h_q, max_full_len, d_q), ...)
diag_k_pad = torch.empty((bs, h_kv, max_full_len, d_kv), ...)
diag_v_pad = torch.empty((bs, h_kv, max_full_len, d_kv), ...)
ops.varlen_fa_seqlen_pad(query, key, value, diag_q_pad, diag_k_pad, diag_v_pad, ...)

# 比较真实数据区([cached_len:full_len])
diff = (q_pad[0, :, cached_len:full_len, :] - diag_q_pad[0, :, ...]).abs().max()
print(f"[DIAG] q_pad diff: {diff.item()}")
```

如果 diff 大,定位到我们 Python 散布的具体哪一步跟 C++ varlen_pad 不一致。

#### 还有些零碎假设(留作参考)

- **K/V 经过了 in-place 修改**:Qwen2 的 rotary embedding 是 in-place 改 q/k,我们的 `key`/`value` 已经过了 ROPE。从 KV cache 读出来的 cached K 也经过了 ROPE。这应该一致。但**如果 ROPE 的应用方式跟原 sdpa 路径假设的不同(比如不同的位置编码起点),就会错**。可以加诊断 print K[0:4, 0, :8] 看看实际值
- **Query 不应该 zero-fill 在 padding 区**:某些 attention 实现可能对 Q=0 的行做特殊处理(skip 或 NaN 出来)。考虑改成"用 query 已有数据的某行复制填充,然后 mask 掉"
- **GQA 头数**:Qwen2.5-14B 有 40 q_heads / 8 kv_heads(TP=8 之前)。TP=8 之后每 rank h_q=5, h_kv=1。**5 个 Q head 共享 1 个 KV head 的扩展逻辑** —— `_scaled_dot_product_attention_flash_musa` 应该自动处理,但要 verify

### 用来验证修复的最简测试

跑命令:
```bash
MUSA_BLOCKING=1 DEBUG=1 USE_CUSTOM_ALLREDUCE=1 SCENARIOS=long_context \
  BENCH_TIMEOUT=1500 OVERALL_TIMEOUT=2400 FREE_THRESHOLD_GB=20 \
  bash /data/my_vllm_test/wait_and_run.sh
```

跑完看:`$RUN/bench.json.outputs.txt`(我们在 benchmark.py 加的 sidecar)

**预期成功的输出**:
- 第 1-4 条请求都是关于"Transformer 架构"的合理中文总结
- 没有 `Vas blanco`、`_OUTPUT_`、希伯来字符等垃圾 token

**预期失败的输出**:
- 重复输入文本,夹杂随机 token,或跑题到不相关话题

### Benchmark.py 已加的 sidecar 文件

为了验证语义,我修改了 `/data/my_vllm_test/benchmark.py`:
- 在 `RequestStat` 加 `user_prompt_tail` + `output_text` 字段
- 在 `run_one` 成功后捕获用户 prompt 尾巴 + 完整输出
- 在主流程写 `bench.json` 时,同时写 sidecar `bench.json.outputs.txt`,**包含每条请求的完整模型输出文本**(便于人工验证语义)

### 关键经验教训(给接手者避坑)

(从我们这次诊断里反复栽过的)

| 教训 | 含义 |
|---|---|
| **改了 patch 必须实测验证它生效** | 我们多次假设 `nan_to_num_` 起作用就推导下一步,实际后续诊断才发现 NaN 还在 |
| **不要相信 py-spy 的 Python 栈是"卡死位置"** | 它指向的是 Python 等 GPU 的 sync 点,**真出错的 GPU op 可能在上一步**(由于 MUSA 异步)|
| **sticky GPU error 模型**:某个 op 越界 → context 损坏 → 后续 op 报"信使错误" | 我们看到的 5 种不同错误信息(broadcast/AsmKernel/Fill/Permute/raw illegal access)全是同一个 unpad 越界的信使 |
| **`torch.musa.synchronize()` 不能 100% 抓内部错误** | MuDNN/varlen 内部用自己 stream,默认 stream 同步抓不到 |
| **bisection 是终极武器** | 当所有"猜测式 patch"都失败时,**用 STUB 替换整个函数**,然后逐个 op 加回来,**这是唯一能 100% 锁定凶手的方法** |
| **共享 GPU 上慎用 pkill -f pattern** | 会误杀同机其他人(MTT 共享机器规矩)。用具体 PID |
| **回滚比改还重要** | 每次大改前 docker cp 备份,不要相信"我的改动可逆" |

### 至此项目交接信息齐全(2026-05-21 版,见 2026-05-26 更新)

~~如果你是新接手者,读完上面这一节就能开始干活了。**优先做"方向 1+2"(加 .contiguous() + 让 Request #0 走原路径)** —— 改动量最小,可能直接解决问题。~~

> 2026-05-26 更新:
> - **方向 2(让 Request #0 走原路径)实际上已经做了** — Path A patch 里加了 `num_prefill_tokens < seq_lens.sum()` gate,Request #0(fresh prefill,cached_len=0)自动走原 `sdpa_attention_with_kernel_seqlen_pad`
> - **方向 1(.contiguous())未做也未必需要** — 真实长 prompt 跑出来语义正确,说明 transpose 的非连续 view 实际没出问题(可能是 PyTorch 的 broadcast 赋值自己处理了)
> - **方向 3(对账诊断)未做** — 现在不需要了,Path A 已经验证正确

---



## ⭐ 2026-05-19 更新:最小复现条件锁定

经过一组对照实验,**三个必要条件全部锁定**(缺一不可):

```
TP=8 + prefix-caching=ON + 长 prompt(≥3k tokens) → broadcast 死锁
```

并发数**不是**必要条件 —— concurrency=1 一样能复现。

### 对照实验矩阵(2026-05-19,镜像 20260323,Qwen2.5-14B,TP=8)

| 实验 | prefix-caching | USE_CUSTOM_ALLREDUCE | prompt 长度 | concurrency | 结果 |
|---|---|---|---|---|---|
| A | **ON** | 1 | **长** (3616 in) | 1 | ❌ **死锁** (shm_broadcast 60s 超时) |
| B | OFF | 1 | 长 (3616 in) | 1 | ✅ 8/8 OK, TTFT~100ms |
| C | **ON** | 1 | **短** (137 in) | 1 | ✅ 8/8 OK, TTFT~50ms |
| D (workshop 实测) | OFF | 1 | 长 + 高并发 (256) | 256 | ✅ 512/512 OK (workshop 同机日常跑) |

**唯一的死锁组合是 A**(prefix=on **且** prompt 长)。其余三个组合任意"破"一个条件都能跑通。

### 新增证据:`shm_broadcast.py:456` 超时

20260323 版本死锁时 server.log 出现明确信号(20260309 时只有 py-spy 栈,没这条):

```
(VllmWorkerProcess pid=N) DEBUG ... [shm_broadcast.py:456] No available shared memory broadcast block found in 60 second.
```

**所有 7 个非 driver worker 同步刷出**,且每 60s 刷一次 —— 卡的不是 MCCL collective 本身,而是 vllm 上层的 `shm_broadcast`(共享内存广播路径,driver 发块给 worker)。和 5/9 那次的 `_get_driver_input_and_broadcast` 栈是同一个 broadcast 通道。

### 假设排除/确认更新

| 5/9 当时的怀疑 | 重测后结论 |
|---|---|
| ⭐⭐⭐ CUDA Graph capture shape 不匹配 | ❌ **排除**。MODE=workshop 长 prompt(实验 B/D)CUDA Graph 是开着的,完全 OK |
| ⭐⭐ prefix caching 触发 KV state 不一致 | ✅ **确认是触发因素之一**(实验 A vs B) |
| ⭐ `max-num-seqs` × paged attention 边界 | 未独立验证,但本次 `max-num-seqs=64` 一样死,跟 256 无关 |
| (新增) **prompt 长度本身** | ✅ **确认是触发因素之一**(实验 A vs C) |

### `USE_CUSTOM_ALLREDUCE=1` 不能绕过

实验 A 已经开了 `USE_CUSTOM_ALLREDUCE=1` 还是死锁。原因:这个开关**只 patch allreduce 路径**(`vllm_musa/patch/custom_allreduce/`),而我们死的是 **broadcast**,跟 allreduce 无关。
之前误以为它是 workshop 不死的原因 → 其实 workshop 不死是因为他们**关了 prefix-caching**。

### 推荐的最小绕过(优先级排序,2026-05-19 更新)

| 优先级 | 方案 | 改动 | 代价 |
|---|---|---|---|
| ⭐⭐⭐ | **关 prefix-caching** | run.sh 去掉 `--enable-prefix-caching` | 失去多轮/RAG 加速,但功能完整 |
| ⭐⭐ | 缩短最长 prompt | 业务侧切 prompt 到 <2k tokens | 仅适合短输入场景 |
| ⭐ | 降 TP 到 4 | 显存可能不够大模型 | 不适合 32B+ 模型 |

### 复现命令(2026-05-19 版,基于本仓库脚本)

```bash
# 复现死锁(MODE=our + 长 prompt):
USE_CUSTOM_ALLREDUCE=1 SCENARIOS=long_context FREE_THRESHOLD_GB=12 \
  bash /data/my_vllm_test/wait_and_run.sh

# 验证关 prefix-caching 可绕过(MODE=workshop):
MODE=workshop FREE_THRESHOLD_GB=12 \
  bash /data/my_vllm_test/wait_and_run.sh

# 验证短 prompt + prefix-caching 不死锁(MODE=our + short):
USE_CUSTOM_ALLREDUCE=1 SCENARIOS=short FREE_THRESHOLD_GB=12 \
  bash /data/my_vllm_test/wait_and_run.sh
```

run 目录(供后续追溯):
- 死锁: `/data/my_vllm_test/runs/20260519_023225_our/`
- 关 prefix 通过: `/data/my_vllm_test/runs/20260519_025648_workshop/`
- 短 prompt 通过: `/data/my_vllm_test/runs/20260519_031755_our/`

### 还未验证(留给后续)

- [ ] TP=4 / TP=2 是否也死(目前只在 TP=8 验证)
- [ ] prompt 长度阈值精确位置(只知 137 OK,3616 死,中间未测)
- [ ] 是否依赖 block-size(本次都用 64)
- [ ] 是否依赖模型架构(本次只测 Qwen2.5-14B,workshop 用 Qwen3-32B 也死过)

---

## 历史调查(2026-05-09 初次发现)

> 以下为 5/9 首次复现时的调查记录,环境为 vllm_musa 20260309 + Qwen3-8B。
> 当时锁定到"broadcast 死锁"但还没分清是 prefix-caching 还是 CUDA Graph 触发。
> 5/19 的对照实验已经排除 CUDA Graph,确认是 prefix-caching + 长 prompt 组合。

## 一句话结论

vllm_musa 在 **TP=8 + 长 prompt（≈3-4k token）+ prefix caching + CUDA Graph** 场景下，
TP worker 之间的 `torch.distributed.broadcast` 集体通信死锁。
表现为：vllm 服务收到请求后无任何响应，但 8 个 worker 进程 CPU 占满（自旋等待 MCCL 通信），GPU 利用率为 0%。

## 复现步骤

1. 启动 vllm 服务（参数见 `run.sh`，关键开关：`--tensor-parallel-size 8 --enable-prefix-caching --compilation-config '{"cudagraph_capture_sizes": [...], "simple_cuda_graph": true}'`）：
   ```bash
   cd /data/my_vllm_test
   nohup bash run.sh /data/SETS/models/qwen3-8b 8 32768 > server.log 2>&1 &
   ```
2. 跑 benchmark：
   ```bash
   docker exec gy_work python /data/my_vllm_test/benchmark.py
   ```
3. `short` 场景（输入 ~50 token，输出 100 token）跑完正常。
4. 进入 `long_context` 场景（输入 ~3-4k token，输出 200 token），第 1-2 条请求被 vllm 接收后**永不返回**。
5. 6 分钟后 vllm 服务端日志没有任何新输出，但进程仍在；客户端 benchmark 一直在等流式响应。

## 现象与证据

### 1. 时间线

| 时间 | 事件 |
|---|---|
| 14:10:43 | vllm 收到 long_context 第 1 条请求（"问题角度1"） |
| 14:10:44 | vllm 输出 metrics: `Avg prompt throughput: 297.8 tokens/s, Avg generation throughput: 621.3 tokens/s, Running: 1 reqs` |
| 14:10:45 | vllm 收到 long_context 第 2 条请求（"问题角度2"） |
| 14:10:45 ~ 14:17:19 | **vllm 6 分 34 秒无任何日志输出**（正常情况下 vllm 每 5-10 秒会自动打印 metrics 行） |

### 2. 进程状态

宿主机端 `ps -ef` 与 `/proc/<pid>/status` 显示：

| PID | 角色 | State | wchan | CPU 时间 |
|---|---|---|---|---|
| 9655 | vllm 主进程（API server） | S (sleeping) | `do_epoll_wait` | 2286 ticks（停止增长） |
| 9923 | TP Worker 0（driver worker） | **R (running)** | 0 | 67244 ticks（持续上涨） |
| 10120 | TP Worker 1 | **R (running)** | 0 | 63021 ticks（持续上涨） |
| 10121 | TP Worker 2 | **R (running)** | 0 | 63770 ticks（持续上涨） |
| 10122 | TP Worker 3 | **R (running)** | 0 | 64946 ticks（持续上涨） |
| 10123 | TP Worker 4 | **R (running)** | 0 | 63256 ticks（持续上涨） |
| 10124 | TP Worker 5 | **R (running)** | 0 | 62310 ticks（持续上涨） |
| 10125 | TP Worker 6 | **R (running)** | 0 | 63318 ticks（持续上涨） |
| 10126 | TP Worker 7 | **R (running)** | 0 | 63244 ticks（持续上涨） |

**关键观察**：所有 worker 都处于 R 状态，CPU 时间持续累积，但 6 分钟没产出任何结果。这是典型的**自旋等待死锁**。

### 3. mthreads-gmi 表现

```
ID   %GPU  Mem
0    0%    70258MiB(81920MiB)
1    0%    70111MiB(81920MiB)
...
7    0%    70105MiB(81920MiB)
```

GPU 利用率 0%，但显存仍占满。说明 worker 没在做矩阵计算，只在做**通信等待**。

### 4. Python 调用栈（py-spy 抓取所有 8 个 worker）

| Worker | PID | 调用位置 | 角色 |
|---|---|---|---|
| rank 0（driver） | 9923 | `parallel_state.py:584` `_get_driver_input_and_broadcast` | broadcast 发送者 |
| rank 1 | 10120 | `parallel_state.py:614` `_get_worker_input_from_broadcast` | 接收者 |
| rank 2 | 10121 | 同上 | 接收者 |
| rank 3 | 10122 | 同上 | 接收者 |
| rank 4 | 10123 | 同上 | 接收者 |
| rank 5 | 10124 | 同上 | 接收者 |
| rank 6 | 10125 | 同上 | 接收者 |
| rank 7 | 10126 | 同上 | 接收者 |

**关键观察：8 个 rank 全都已经进入 broadcast，没有一个掉队。**
不是"某 worker 慢了导致其他人等它"，而是 **MCCL collective 已经被 8 个 rank 同时调用，但内部就是不返回**。

driver worker (rank 0) 的完整栈：
```
broadcast (torch/distributed/distributed_c10d.py:2437)
wrapper (torch/distributed/c10d_logger.py:83)
broadcast_tensor_dict (vllm/distributed/parallel_state.py:584)
broadcast_tensor_dict (vllm/distributed/communication_op.py:41)
_get_driver_input_and_broadcast (vllm/worker/worker_base.py:352)
prepare_input (vllm/worker/worker_base.py:379)
execute_model (vllm/worker/worker_base.py:394)
_driver_execute_model (vllm/executor/mp_distributed_executor.py:145)
step (vllm/engine/llm_engine.py:1358)
engine_step (vllm/engine/multiprocessing/engine.py:235)
run_engine_loop (vllm/engine/multiprocessing/engine.py:226)
```

worker (rank 1-7) 的完整栈基本一致，只是入口换成 `_get_worker_input_from_broadcast`，接收方而已。

### 5. /metrics 接口超时

```bash
$ curl -m 3 http://127.0.0.1:8000/metrics
（3 秒超时无响应）
```

证明 vllm 引擎层完全卡住，连内部 metrics 都无法响应。

## 根因分析

### 直接原因（更新）
~~TP worker 之间 `broadcast` 操作未对齐~~ → 排除。

**实际原因：8 个 rank 已经全部到位调用 broadcast，但 MCCL 集体通信内部不返回。**
这是 MCCL 库层面的问题（参数描述符不一致 / device-side kernel 死锁 / race condition），不是上层 vllm 调度问题。

### 可能的诱发条件（需进一步定位）

按可疑度从高到低：

1. **CUDA Graph capture shape 不匹配长 prompt** ⭐⭐⭐
   - run.sh 里 `cudagraph_capture_sizes=[1,2,4,8,...,256]` 是 batch size 列表
   - 但长 prompt 的 prefill 阶段，token 数（不是 batch）才是关键 shape 维度
   - 不同 worker 上 cudagraph replay 的某次结果分支不同 → broadcast 数据 shape 不一致 → MCCL 失配

2. **prefix caching 触发不同 worker KV state 不一致** ⭐⭐
   - 前缀复用决策依赖 KV cache 命中查询
   - 如果 8 个 worker 对同一 chunk hash 的命中状态不一致（极端边界），broadcast 的 metadata 字段会发散

3. **`max-num-seqs=256` 与单条长 prompt 的 paged attention block 分配冲突** ⭐
   - 256 并发预留 + 单条 3-4k token prompt → block table 边界条件可能让某 worker 进入 swap 路径而其他没有

### 间接原因
vllm_musa 这个版本（0.9.3.dev0+ga5dd03c1e.d20260309）对长 prompt + cudagraph + prefix caching 的组合**未做端到端测试**。

## 二分定位计划

通过逐项关掉 run.sh 里的优化，确定哪个开关触发死锁。每项测试用同一条 long_context 请求验证。

| # | 修改 | 期望验证 |
|---|---|---|
| A | baseline（不改） | 复现死锁 ✅（已知） |
| B | 删 `--enable-prefix-caching` | 是否仍死锁 → 锁定 prefix cache |
| C | A + 加 `--enforce-eager`（关 CUDA Graph） | 是否仍死锁 → 锁定 cudagraph |
| D | 全关（B+C） | 应能稳定跑通 |
| E | 降 TP 到 4 张卡 | 验证是不是 8 卡才会触发 |
| F | 缩短 long_context prompt 长度（如 ~500 token） | 验证 prompt 长度阈值 |

定位出后，建议按组合记录最小复现 case 上报。

## 临时绕过方案（按代价从低到高）

| 方案 | 改动 | 代价 |
|---|---|---|
| 1. 关 CUDA Graph | run.sh 加 `--enforce-eager` 删 `--compilation-config` | 性能损失 ~10-20%，但稳定 |
| 2. 关 prefix caching | 删 `--enable-prefix-caching` | 失去多轮 / RAG 加速 |
| 3. 缩 max prompt 长度 | benchmark.py 把 `LONG_PARAGRAPH * 6` 改成 `* 2` | 测不到长 prompt 性能 |
| 4. 降 TP 到 4 | run.sh 改 `--tensor-parallel-size 4` | 显存压力大，吞吐下降 |
| 5. 改 PP=2 + TP=4 | 双机或单机切分模式 | 配置复杂 |

**建议组合**：先试 **方案 1（关 CUDA Graph）**，最便宜也最可能解决。

## 复现 / 调试命令速查

```bash
# 1. 看 vllm 服务最新日志
tail -f /data/my_vllm_test/server.log

# 2. 看 vllm 是否在响应
curl -m 3 http://127.0.0.1:8000/health

# 3. 看 worker 进程状态（R=running, S=sleeping）
docker exec gy_work bash -c "
  for pid in \$(pgrep -f 'vllm serve'; pgrep -f multiprocessing.spawn); do
    echo \"PID \$pid:\"
    cat /proc/\$pid/status | grep -E 'State|Threads'
    echo \"  wchan: \$(cat /proc/\$pid/wchan)\"
  done
"

# 4. 抓 worker Python 调用栈（关键诊断手段）
docker exec gy_work bash -c "py-spy dump --pid <PID>"

# 5. 强制清理（vllm 死锁后唯一办法）
docker exec gy_work bash -c "pkill -9 -f 'vllm serve' && pkill -9 -f multiprocessing.spawn"
sleep 3
mthreads-gmi | grep MiB    # 确认显存释放
```

## 推荐的下一步调试

```bash
# 1. 重启 vllm 时打开 MCCL debug 输出，看通信细节
docker exec -it gy_work bash
cd /data/my_vllm_test
MCCL_DEBUG=INFO MCCL_DEBUG_SUBSYS=COLL,INIT \
  bash run.sh /data/SETS/models/qwen3-8b 8 32768 2>&1 | tee server_mcclverbose.log

# 复现死锁后日志会包含 collective 的具体调用参数（buffer size、dtype、count）
# 重点看死锁前的最后几条 MCCL 日志，能定位是哪个 collective 出错
```

## 上报建议

如果确认是 vllm_musa 的 bug，按以下信息上报给 SETS / 摩尔线程：

- **复现镜像**：`sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:4.3.5_kuae2.1_hygon_ubuntu_fix_ray`
- **vllm 版本**：`0.9.3.dev0+ga5dd03c1e.d20260309`
- **vllm_musa 版本**：`0.1.dev358+gd3980ed`
- **复现条件**：上述 baseline 配置 + Qwen3-8B + ~3-4k token 输入
- **关键现象**：worker 卡在 `torch.distributed.broadcast`，不退出但不进展
- **附件**：本文档 + py-spy stack + benchmark.py 复现脚本

## 相关文件

- `/data/my_vllm_test/run.sh` —— 触发死锁的服务启动脚本
- `/data/my_vllm_test/benchmark.py` —— 触发死锁的客户端脚本（在 `long_context` 场景）
- `/data/my_vllm_test/server.log` —— 死锁前后的 vllm 服务端日志
