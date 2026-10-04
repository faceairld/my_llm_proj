# vllm_musa prefix-cache 命中 OOB 越界写 — ISSUE 包

报告时间:**2026-05-27**

## TL;DR(一句话)

`vllm_musa/_kernels.so` 里的 C++ 算子 **`ops.varlen_fa_seqlen_unpad`**,
在 `--enable-prefix-caching` 命中(`num_prefill_tokens < sum(seq_lens)`)的场景下
**越界写 output buffer**,污染 GPU 显存,导致 vllm 服务挂掉或输出全 `!` 死循环。

我们已用 STUB 二分(B1/B2/B3)100% 锁定元凶在这个函数,并在 Python 层做了 workaround(Path A)绕开它。

## 测试环境

| 组件 | 版本 |
|---|---|
| vllm_musa | `0.1.dev358+gd3980eddc.d20260323`(2026-03-23 build) |
| vllm | `0.9.3.dev0+ga5dd03c1e.d20260323` |
| torch_musa | 2.7.1 |
| MUSA SDK | 4.3.5 |
| MuDNN | 3.1.5 |
| 镜像 | `sh-harbor.mthreads.com/sets/sets_vllm_musa:musa_sdk_4.3.5_torch_2.7.1_fix_ray` |
| 硬件 | MTT S5000 × 8 |
| 模型 | Qwen2.5-14B(TP=8) |

> ⚠️ 注:我们测的镜像是 2026-03-23 的 build,距今约 2 个月。如果你们已经有更新的 vllm_musa,请先在新版上验证 bug 是否还存在 —— 内部代码可能已经变化,我们的诊断结论需要在新版重新核对。

## 包结构

```
mtt_issue_2026-05-27/
├── README.md                                       ← 你正在读
├── ISSUE_vllm_musa_broadcast_deadlock.md           ← ★ 完整诊断报告(1600+ 行)
│
├── patches/                                        ← 实验用 patch(idempotent Python 脚本)
│   ├── path_a_python_concat.py                     ← ★ 我们的 workaround,装上后不崩
│   ├── bypass_sdpa_with_stub.py                    ← STUB 实验:整个函数 stub 化,不崩 → bug 在函数体内
│   ├── bisect_b1_only_varlen_pad.py                ← 二分 B1:只 pad,不崩 → pad 清白
│   ├── bisect_b2_add_sdpa.py                       ← 二分 B2:pad + SDPA,不崩 → SDPA 清白
│   └── bisect_b3_only_unpad.py                     ← 二分 B3:只 unpad,★崩★ → unpad 是元凶
│
├── test_scripts/                                   ← 测试编排脚本
│   ├── benchmark.py                                ← OpenAI 客户端 benchmark,带 sidecar 输出
│   ├── run.sh                                      ← 启 vllm serve(MODE=our/workshop/minimal)
│   ├── run_all.sh                                  ← 编排:启 vllm + 等就绪 + 跑 benchmark + 清理
│   └── wait_and_run.sh                             ← wrapper:等 GPU 显存够再触发 run_all
│
├── reference/                                      ← 参考代码
│   ├── flash_attn_original.py                      ← vllm_musa 原版 flash_attn.py(未改)
│   └── flash_attn_with_path_a.py                   ← 装上 Path A 后的 flash_attn.py(diff 看效果)
│
└── evidence_runs/                                  ← 实测证据 run
    ├── 20260526_065339_path_a_works/               ← ★ Path A 装上后:8/8 全过 + 输出语义正确
    │   ├── meta.txt                                ← 实验配置(注意 has_path_a=yes)
    │   ├── bench.json                              ← 性能数据(TTFT 319→60ms 加速可见 cache hit)
    │   ├── bench.json.outputs.txt                  ← 完整模型输出文本(8/8 都是合理答案)
    │   ├── bench.log
    │   ├── server.log                              ← vllm server 日志(里面能看到 PATH-A 触发证据)
    │   └── flash_attn.before_run.py                ← 这次 run 用的 flash_attn 全文
    │
    └── 20260526_032833_vanilla_long_no_cache_baseline/  ← 对照:原版 + 关 cache,长 prompt 也乱码
        ├── meta.txt                                ← (注意 has_path_a=no,prefix_caching=0)
        ├── bench.json
        ├── bench.json.outputs.txt                  ← 这次输出也有乱码(证明乱码跟 Path A 无关)
        ├── server.log
        └── flash_attn.before_run.py
```

## 建议阅读顺序

1. **先读这份 README**(你在看)了解全貌
2. **读 `ISSUE_vllm_musa_broadcast_deadlock.md`** 的顶部章节:
   - `## 📦 测试环境 / 版本信息`(确认我们用的版本)
   - `## 🎉 2026-05-26 更新:Path A 验证通过` 里的 `### Bug 1 三种 workaround 路径对照`(为什么选 Path A 而不是改你们的 C++)
3. **读 `## 🎯 2026-05-21 终极定位` 章节**:
   - `### 完整调用链` —— bug 在调用栈第几层
   - `### 外层 vllm_musa Python 函数` —— 完整 length 关系讲解 + 函数代码 + 具体例子(8 valid + 896 cached)
   - `### prefix-cache 命中如何让 sum_seq < sum(seq_lens) → 触发 OOB(⚠️ 推测,非看代码确认)` —— 我们对 bug 内部机制的推测
4. **看 `patches/bisect_b3_only_unpad.py`** —— 这是 100% 锁定元凶的关键实验
5. **diff `reference/flash_attn_original.py` vs `reference/flash_attn_with_path_a.py`** —— 看 Path A 具体加了什么(只在 prefill 分支加了一个 `_path_a_hit` gate + Python 实现)
6. **看 `evidence_runs/20260526_065339_path_a_works/bench.json.outputs.txt`** —— 真实输出证据,Path A 让模型继续正常工作

## 复现步骤

### 复现 bug(原版 vllm_musa 一定崩)

```bash
# 1. 起一个干净的 vllm_musa 容器(从我们的镜像)
docker run -d -it --name repro --privileged --network host --ipc host \
  -v /data:/data \
  sh-harbor.mthreads.com/sets/sets_vllm_musa:musa_sdk_4.3.5_torch_2.7.1_fix_ray bash

# 2. 起 vllm,开 prefix-caching
docker exec -d repro bash -c "
  cd /your_path/test_scripts &&
  MODE=our bash run.sh /path/to/Qwen2.5-14B 8 32768 8001 > /tmp/vllm.log 2>&1
"

# 3. 等启动就绪后跑 benchmark,场景 long_context(8 条共享长 prompt 触发 partial 命中)
docker exec repro bash -c "
  cd /your_path/test_scripts &&
  python3 benchmark.py --port 8001 --scenarios long_context --concurrency 1 --requests 8
"

# 预期:第 2 条请求开始挂 / 崩 / 输出 !!!! 死循环
```

### 验证 Path A workaround(装上后不崩)

```bash
# 1. 应用 Path A patch
docker exec repro python3 /your_path/patches/path_a_python_concat.py

# 2. 重启 vllm + 重跑(同上)
# 预期:8/8 全过 + 输出语义正确 + bench.json.outputs.txt 里能看到合理回答
```
