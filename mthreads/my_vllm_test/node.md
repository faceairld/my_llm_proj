



❯ 这个先这样,先看看vllm_musa部署多卡分布式推理,我现在就单机可以用,有没有现成的脚本?我记得单机的分布式脚本很简单,只要设置环境变量和调用vllm的函数好像就行了                                              

● 对，单机多卡只是 tensor-parallel-size 一个参数。/data/SETS/infer/vllm-musa/ 下就有现成的，最简单的那个 run.sh 接收模型路径、TP、max_len 三个参数：

  # /data/SETS/infer/vllm-musa/run.sh 内容
  VLLM_USE_V1=0 vllm serve ${MODEL_PATH} \
      --trust-remote-code \
      --gpu-memory-utilization 0.7 \
      --served-model-name ${MODEL_NAME} \
      --max_model_len ${MAX_MODEL_LEN} \
      --block-size 64 \
      --tensor-parallel-size ${TP} \
      --pipeline-parallel-size 1 \
      --max-num-seqs 30 \
      --compilation-config '{"cudagraph_capture_sizes":[...], "simple_cuda_graph":true}'

  用法（README 里都给好了）

  cd /data/SETS/infer/vllm-musa/

  # 8 卡 TP，跑 Qwen3-32B
  nohup bash run.sh /data/playground/model/qwen/qwen3-32b 8 32768 > server.log 2>&1 &
  tail -f server.log

  # 等出现 "Uvicorn running on http://0.0.0.0:8000" 就 OK
  # 跑压测：
  nohup bash vllm-perf-test.sh /data/playground/model/qwen/qwen3-32b 8 > client.log 2>&1 &

  关键点

  - --tensor-parallel-size N：用 N 张卡做 TP 切分
  - --pipeline-parallel-size 1：单机不用 PP，固定 1
  - VLLM_USE_V1=0：vllm_musa 当前只支持 v0 调度（这是 musa 端的限制，容易踩坑）
  - MTHREADS_VISIBLE_DEVICES=0,1,2,3：指定用哪几张卡（不设就是 all）

  你最关心的 MTHREADS_VISIBLE_DEVICES

  # 只用 0-3 卡（4 张），TP=4
  MTHREADS_VISIBLE_DEVICES=0,1,2,3 bash run.sh /data/SETS/models/qwen3-8b 4 32768

  # 用全部 8 张
  MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh ... 8 32768
  # 或者干脆不设

  已有的几个变体脚本（在同目录）

  ┌───────────────────────────────────────┬───────────────────────────────┐
  │                 脚本                  │             用途              │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ run.sh                                │ 通用（带 cudagraph 编译优化） │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ run-vllm-musa.sh                      │ 极简版（无 cudagraph）        │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ run_235b.sh                           │ 跑 Qwen3-235B 的硬编码版      │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ Qwen3-14B-run.sh / Qwen3-8B-openai.sh │ 模型专用的快捷脚本            │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ vllm-perf-test.sh                     │ 压测客户端                    │
  ├───────────────────────────────────────┼───────────────────────────────┤
  │ benchmark.sh                          │ benchmark 入口                │
  └───────────────────────────────────────┴───────────────────────────────┘

  你想跑哪个模型？我帮你直接拼好命令 —— 告诉我：
  1. 模型路径（或者让我列一下 /data/playground/model/ 下有什么）
  2. 几张卡（TP 多大）
  3. 想用哪几张卡的编号（默认 all）
❯ OK,我打算用run.sh跑,这边有现成的模型吗?我看有R1的模型?还要其他的吗                                                                                                                                    

  Listed 2 directories (ctrl+o to expand)

● 模型很全。按你**单机 8 张 S5000（640G 总显存）**的可行性给你分类：

  ✅ 单机直接能跑

  ┌───────────────────────────────┬──────┬──────────────┬─────────────────────────────────────────────────┐
  │             模型              │ 大小 │   推荐 TP    │                      备注                       │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ Qwen3-8B                      │ 16G  │ 1/2/4/8 都行 │ 最快验证，先跑这个                              │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ Qwen3-14B                     │ 28G  │ 2/4/8        │                                                 │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ Qwen3-32B                     │ 62G  │ 2/4/8        │ 单机最甜区                                      │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ DeepSeek-Coder-V2-Lite        │ 30G  │ 2/4/8        │ 30B-MoE，激活 2.4B，特别快                      │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ deepseek-r1-distill-llama-70b │ 132G │ 4/8          │ R1 蒸馏到 Llama-70B，这是 R1 系列里你单机能跑的 │
  ├───────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────────────┤
  │ Qwen3-235B-A22B-FP8           │ 223G │ 8            │ FP8 量化，单机刚好（要 vllm_musa 支持 FP8）     │
  └───────────────────────────────┴──────┴──────────────┴─────────────────────────────────────────────────┘

  ⚠️  单机够呛 / 必须双机

  ┌──────────────────────────────┬──────┬─────────────────────────────────────────────────────────────────────┐
  │             模型             │ 大小 │                                备注                                 │
  ├──────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ Qwen3-235B-A22B (BF16)       │ 438G │ 8 卡装权重还行，但 KV cache 空间不够，建议 FP8 版本                 │
  ├──────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ DeepSeek-R1-0528 (满血 671B) │ 642G │ 单机装不下，必须双机 PP=2（你之前那两台机器的双机环境用的就是这个） │
  └──────────────────────────────┴──────┴─────────────────────────────────────────────────────────────────────┘

  关于 R1

  你看到的有两个：
  1. /data/SETS/models/DeepSeek-R1-0528 (642G) —— 满血 R1，单机跑不动
  2. /data/SETS/models/deepseek-r1-distill-llama-70b (132G) —— R1 蒸馏到 Llama-70B，单机能跑

  如果只是想验证 vllm_musa 跑通分布式 + 看 R1 的对话效果，蒸馏版就够用。

  推荐启动顺序

  第 1 步：Qwen3-8B TP=8（最快，2 分钟拉起）
  cd /data/SETS/infer/vllm-musa
  nohup bash run.sh /data/SETS/models/qwen3-8b 8 32768 > server.log 2>&1 &
  tail -f server.log   # 等 "Uvicorn running on" 就 OK

  第 2 步：先验证 8B 能用，再换大模型
  # Qwen3-32B TP=8
  nohup bash run.sh /data/SETS/models/qwen3-32b 8 32768 > server.log 2>&1 &

  # 或 R1-distill-70B TP=8（更接近 R1 效果）
  nohup bash run.sh /data/SETS/models/deepseek-r1-distill-llama-70b 8 32768 > server.log 2>&1 &

  要我直接帮你拉一个 Qwen3-8B 起来验证下吗？

✻ Cooked for 59s

※ recap: 目标：在 MUSA 主机上跑通 vllm_musa 单机多卡推理，并评估 LMCache 加速可行性。当前 LMCache 核心测试已通过、模型已盘点。下一步：用 `/data/SETS/infer/vllm-musa/run.sh` 拉起 Qwen3-8B TP=8 验证。
  (disable recaps in /config)

❯ 我首先有个问题,这个run.sh脚本是最基本的脚本,带不带优化策略?vllm提供的优化策略是否需要额外的设置,因为我看run.sh主要就是一些环境配置+serve函数,vllm额外的优化是否需要在这两者以外添加额外的东西         

● 好问题，确实 run.sh 是非常基础的版本，绝大多数 vllm 的优化都没开。我给你拆开看。

  run.sh 里已经在用的优化

  --block-size 64                        # KV cache 分块大小，影响显存碎片
  --tensor-parallel-size ${TP}            # 多卡并行
  --max-num-seqs 30                       # 最大并发序列数
  --compilation-config '{...}'            # CUDA Graph 捕获（vllm_musa 走 simple_cuda_graph）
  --gpu-memory-utilization 0.7            # 显存使用率（剩 30% 给临时 buffer）

  加上 vllm 默认就在用的（不用写 flag）：
  - ✅ PagedAttention：vllm 的看家本领，KV cache 按 block 管理
  - ✅ Continuous Batching：动态拼 batch（不用等齐）
  - ✅ Async 调度：调度器和算子异步重叠

  run.sh 没有开 但需要单独加 flag 的

  ┌───────────────────────┬─────────────────────────────────┬──────────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │         优化          │              flag               │                   收益                   │                       MUSA 风险                       │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Prefix Caching        │ --enable-prefix-caching         │ 同 system prompt / 多轮对话首 token 极快 │ 中（vllm_musa 改了 PagedAttention，要验证）           │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Chunked Prefill       │ --enable-chunked-prefill        │ 长 prompt 分块，prefill / decode 重叠    │ 高（V0 引擎语义和 V1 不同）                           │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ KV Cache 量化         │ --kv-cache-dtype fp8            │ KV 占用减半，并发翻倍                    │ 高（MUSA 上 FP8 KV 路径未必稳定）                     │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 权重量化推理          │ --quantization fp8 / awq / gptq │ 显存大减、解码加速                       │ 看 vllm_musa 实现，FP8 模型像 Qwen3-235B-FP8 是支持的 │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Speculative Decoding  │ --speculative-config '{...}'    │ 解码 1.5-3×                              │ 中（小草稿模型也得是 musa 版）                        │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Eager 关闭 CUDA Graph │ --enforce-eager                 │ 调试用，不优化                           │ 性能下降但稳定                                        │
  ├───────────────────────┼─────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 更大 max-num-seqs     │ --max-num-seqs 256              │ 提升吞吐                                 │ run.sh 里只 30，太保守                                │
  └───────────────────────┴─────────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────────────────────────┘

  一个重要注意点：VLLM_USE_V1=0

  run.sh 强制用了 V0 引擎。这很关键：

  ┌─────────────────┬───────────────────┬─────────────────┐
  │      特性       │ V0（run.sh 走的） │ V1（vllm 默认） │
  ├─────────────────┼───────────────────┼─────────────────┤
  │ Chunked Prefill │ 半成品            │ 默认开          │
  ├─────────────────┼───────────────────┼─────────────────┤
  │ Prefix Caching  │ 支持              │ 默认开          │
  ├─────────────────┼───────────────────┼─────────────────┤
  │ CUDA Graph      │ 手动配            │ 默认更激进      │
  ├─────────────────┼───────────────────┼─────────────────┤
  │ 调度器          │ 老                │ 重写过，更高效  │
  └─────────────────┴───────────────────┴─────────────────┘

  为什么 vllm_musa 强制 V0？ 因为 vllm_musa 0.9.3 这个版本对 V1 的兼容还没做完，V1 路径会出错或者跑不起来。所以 run.sh 才写死 VLLM_USE_V1=0。

  这意味着你用不上 V1 的默认优化（chunked prefill、prefix caching 自动开），需要的话得手动加 flag，且 V0 下这些 flag 的实现更老更慢。

  实际建议

  先用 run.sh 跑通 baseline

  不要一开始就堆优化，先确认裸跑能起来、有正常吞吐数据。

  改进版 run.sh（推荐先在 8B 上试）

  VLLM_USE_V1=0 vllm serve ${MODEL_PATH} \
      --trust-remote-code \
      --gpu-memory-utilization 0.85 \                # 0.7 太保守，提到 0.85
      --served-model-name ${MODEL_NAME} \
      --max-model-len ${MAX_MODEL_LEN} \
      --block-size 64 \
      --tensor-parallel-size ${TP} \
      --pipeline-parallel-size 1 \
      --max-num-seqs 256 \                            # 30 → 256（看显存够不够）
      --enable-prefix-caching \                       # ✨ 加上：多轮/RAG 首 token 提速
      --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,256], "simple_cuda_graph": true}'

  不建议一上来就开的

  - --enable-chunked-prefill —— V0 下兼容性差，可能崩
  - --kv-cache-dtype fp8 —— 先验证显存够用再考虑
  - --speculative-config —— 需要额外的小模型，先单模型跑通

  一个验证套路

  第一轮：纯 run.sh（baseline）→ 拿到 throughput / TTFT 基线
  第二轮：+ --enable-prefix-caching → 跑同一个 prompt 重复 100 次，看 TTFT 是否暴跌
  第三轮：+ --max-num-seqs 256 → 看吞吐有没有提升
  第四轮：+ FP8 KV cache（如果前面都稳）

  一句话：run.sh 只是"能跑"级别，开了 PagedAttention + CUDA Graph + TP，但所有上层优化（prefix cache、chunked prefill、KV 量化、SpecDec）都得自己加。而且因为 VLLM_USE_V1=0，开起来效果可能不如 NVIDIA
  那边的 vllm。

  需要我直接给你写一个加了 prefix caching 和更激进参数的 run-optimized.sh 吗？

✻ Worked for 50s

❯ 可以,你直接在data文件夹下的my_vllm_test下的run.sh上改,我复制了一份出来,然后你还要写一个py脚本让他能更方便地接受用户输入跑推理,用openai那个库我记得       
● 两个文件都写好了：

  /data/my_vllm_test/run.sh — 已优化

  主要改动：
  - ✅ 新增 --enable-prefix-caching（最大收益的优化）
  - ✅ --max-num-seqs 30 → 256（提升并发）
  - ✅ --gpu-memory-utilization 0.7 → 0.85（更多 KV 空间）
  - ✅ CUDA Graph capture 列表覆盖更宽 batch size
  - ✅ 多了第 4 个参数 port（默认 8000）
  - ✅ 注释里列了没开但可以加的优化（chunked-prefill、FP8 KV 等）

  /data/my_vllm_test/chat.py — 交互式客户端

  特性：
  - 自动列出 vllm 上的模型，默认选第一个
  - 流式输出，带 TTFT 和 token/s 统计
  - 多轮对话保留历史
  - 内置命令：/reset 清空、/save 存盘、/sys 改 system prompt、/exit 退出
  - 支持单轮非交互模式（--once "问题"）

  用法

  起服务（容器内 or 容器外都行，端口都通）：
  docker exec -it gy_work bash
  cd /data/my_vllm_test
  nohup bash run.sh /data/SETS/models/qwen3-8b 8 32768 > server.log 2>&1 &
  tail -f server.log     # 等 "Uvicorn running on http://0.0.0.0:8000"

  用 chat.py 测试（容器内）：
  # 在 gy_work 容器里跑（host 没装 openai）
  docker exec -it gy_work python /data/my_vllm_test/chat.py

  # 单轮
  docker exec -it gy_work python /data/my_vllm_test/chat.py --once "用一句话介绍你自己"

  # 改端口 / system prompt
  docker exec -it gy_work python /data/my_vllm_test/chat.py --port 8000 --system "你是一个简洁的助手"

  ▎ 顺手提醒：宿主机 python 没装 openai 库，要在 gy_work 容器里跑（容器里是 openai 1.90.0）。

  - 支持单轮非交互模式（--once "问题"）

  用法

  起服务（容器内 or 容器外都行，端口都通）：
  docker exec -it gy_work bash
  cd /data/my_vllm_test
  
  
       # 等 "Uvicorn running on http://0.0.0.0:8000"

  用 chat.py 测试（容器内）：
  # 在 gy_work 容器里跑（host 没装 openai）
  docker exec -it gy_work python /data/my_vllm_test/chat.py

  # 单轮
  docker exec -it gy_work python /data/my_vllm_test/chat.py --once "用一句话介绍你自己"

  # 改端口 / system prompt
  docker exec -it gy_work python /data/my_vllm_test/chat.py --port 8000 --system "你是一个简洁的助手"

  ▎ 顺手提醒：宿主机 python 没装 openai 库，要在 gy_work 容器里跑（容器里是 openai 1.90.0）。

  要我现在帮你拉一个 Qwen3-8B 起来验证吗？








能看到unpad这个函数的代码吗?能分析是它代码的问题还是说是那个地方设置出现了问题?函数本身是没问题的?如果能保留unpad,我是想保留的,但是为什么其他几个方案不需要pad和unpad函数,A: RunFlashVarlen和 C: 用 paged attention具体是怎么做到替换的?   

● 来逐个回答。先尝试找 unpad 的源码,再讲两条不用 pad/unpad 的路径具体怎么实现。

  Q1:能看到 unpad 的源码吗?

  之前查过,Python 层只是 thin wrapper,真正实现在 _kernels.so 编译二进制。再扫一遍各处,看有没有遗漏:

● Bash(echo "=== 找含 varlen_fa_seqlen_unpad 实现的 C++ 文件 ==="
      find /data -maxdepth 8 -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cc" -o -name "*.mu" -o -name "*.…)
  ⎿  === 找含 varlen_fa_seqlen_unpad 实现的 C++ 文件 ===

     === vllm_musa 源码备份目录结构 ===
     … +23 lines (ctrl+o to expand)

● 确认 unpad 源码看不到(只有 _kernels.cpython-...so 二进制,全盘扫无 .cpp/.cu/.mu)。但我们能从行为推测出 bug 是函数本身的契约错误。来分别回答你的三个问题。

  Q1:从行为推测 —— unpad 函数本身就有 bug,不只是调用方的锅

  unpad 的签名拿到的信息:
  attn_out  (bs, h_q, max=904, d_q)    ← 输入:padded 数据
  output    (sum_seq=8, h_q, d_q)       ← 输出:紧凑 buffer,只 alloc 了 sum_seq 行
  seq_lens  [0, 904]                    ← metadata:声明每个 batch 长度
  sum_seq   8                           ← metadata:全局总有效行数
  max_seq   904                         ← metadata:padded 维度长
  
  关键问题:函数怎么决定"每个 batch 写多少行"?推测有两种实现可能:

  ┌──────────────────────────────────────────────────┬─────────────────────────────────────┬───────────────────────────────────────────────┐
  │                     实现方式                     │                行为                 │           我们这种 mismatch 场景下            │
  ├──────────────────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
  │ (a) 用 seq_lens 决定写多少                       │ 按 seq_lens[b+1]-seq_lens[b] 循环写 │ 写 904 行,output 只能容 8 行 → 越界 896 行 ❌ │
  ├──────────────────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────┤
  │ (b) 用 min(seq_lens 长度, sum_seq - offset) 决定 │ 限制不超过 sum_seq 的剩余空间       │ 安全,只写 8 行 ✅                             │
  └──────────────────────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────────────┘

  实测它是 (a)(因为我们 B3 单独跑 unpad 就崩),说明:
  - 它内部循环用 seq_lens 决定写多少行,没用 sum_seq 做上限检查
  - sum_seq 这个参数在函数里没起到边界保护作用
  
  这是函数本身的设计 bug:
  - 函数应该信契约说"output 至少 alloc 了 sum_seq 行,我最多写 sum_seq 行,sum_seq 是上限"
  - 但实际它信了 seq_lens 是真理,在 sum_seq < sum(seq_lens) 时就越界

  当然调用方也有锅(传了 mismatch 的参数),但函数本身没做防御性检查,所以"函数本身有 bug + 调用方触发它"两个都成立。

  Q2:RunFlashVarlen 怎么不用 pad/unpad?

  核心差异:它直接吃"紧凑(packed)+ 累积偏移数组(cu_seqlens)",kernel 内部按 batch 处理但 tensor 没 padding。

  prefix-cache 命中场景下的具体用法:

  准备数据(我们自己 concat 一次):
    q = query                                          # shape (8, h_q, d) — 只新 token,紧凑
    cached_k = 从 KV cache 用 block_tables 取出       # shape (896, h_kv, d)
    cached_v = 同上                                    # shape (896, h_kv, d)
    full_k = torch.cat([cached_k, key], dim=0)         # shape (904, h_kv, d) — 紧凑
    full_v = torch.cat([cached_v, value], dim=0)       # shape (904, h_kv, d)
    cu_seqlens_q = tensor([0, 8])                      # 只有 1 个 batch,Q 长度 8
    cu_seqlens_k = tensor([0, 904])                    # K/V 长度 904(cached + new)

  直接调:
    out = MuDNN::ScaledDotProductAttention::RunFlashVarlen(
              handle,
              output,        # shape (8, h_q, d) — kernel 直接写紧凑结果
              logsumexp,
              q, full_k, full_v,
              mask, dropout_mask,
              cu_seqlens_q, cu_seqlens_k,  # ★ 关键:两个 cu_seqlens 分开传
              mem_allocator
          )

  kernel 内部:
    for batch b in 0..num_batches:
        q_range = q[cu_seqlens_q[b] : cu_seqlens_q[b+1]]   # 取这个 batch 的 query
        k_range = k[cu_seqlens_k[b] : cu_seqlens_k[b+1]]   # 取这个 batch 的 key(可能比 query 长)
        v_range = ...
        causal attention(q_range × k_range)               # 算变长 cross-attention
        写到 output[cu_seqlens_q[b] : cu_seqlens_q[b+1]]

  为什么不需要 pad:输入直接紧凑,kernel 自己用 cu_seqlens 索引,根本没有"对齐到 max_seq"的步骤。

  为什么不需要 unpad:输出直接写到紧凑 buffer,没有 padding 区要丢弃。

  多出来的步骤:concat cached_k/v + new_k/v。但这个 concat 是普通的 torch.cat,简单稳定,不需要复杂 kernel,也没有 OOB 风险。

  Q3:Paged attention 怎么不用 pad/unpad?

  核心差异:K/V 完全不材化(materialize)成一个大 tensor,kernel 直接从 KV cache 读 block。

  KV cache 在 GPU 上长这样:
    key_cache[num_blocks, block_size=64, h_kv, d]
    value_cache[num_blocks, block_size=64, h_kv, d]

    比如 num_blocks=1000(整个 cache 池能放 1000 个 block,= 64000 token 的 K/V)
    每个 block 是一段连续 64 token 的 K/V

  每个请求自己持有一张 block_tables(指针表):
    request_0.block_tables = [12, 45, 7, 89, ...]  ← 它的 token 0-63 在 block 12,token 64-127 在 block 45,...
    request_1.block_tables = [33, 88, ...]
    ...

  prefix-cache 命中场景下的具体用法:

  准备数据:
    q = query                            # shape (8, h_q, d) — 只新 token
    block_tables = batch 内每个 req 的 block 索引表  # shape (1, max_blocks_per_req)
    seq_lens = tensor([904])             # 每个请求的完整长度

    注意:K/V 不需要拉出来! 仍躺在 key_cache/value_cache 大池子里

  直接调 (类似 vllm decode 路径用的):
    paged_attention_kernel(
        output,                          # shape (8, h_q, d) — 直接写紧凑
        q,
        key_cache,                       # ← 整个 cache 池
        value_cache,                     # ← 整个 cache 池
        block_tables,                    # ★ 关键:告诉 kernel 哪些 block 属于这个 req
        seq_lens,
        ...
    )

  kernel 内部:
    for query_token in q:                # 8 个新 query
        for each cached block in block_tables of this request:
            k_block = key_cache[block_id]       # 在线从池子里取
            v_block = value_cache[block_id]
            partial_attention = q_token · k_block · v_block
            accumulate into output[query_token]

  为什么不需要 pad:K/V 直接从 cache 池按需取 block,不需要先 concat 到一个大 tensor 再 pad。

  为什么不需要 unpad:输出已经是紧凑 (sum_q, h_q, d),直接给 caller 用。

  额外优势:没有显存浪费(不用复制 cached K/V 到临时 buffer)、对超长上下文很省内存。这正是 vllm 主流一直推 paged attention 的原因。

  三条路径对比表(更精确)

  ┌──────────────────────────┬─────────────────────────────────────────────────┬───────────────────────────────────────────────────┬────────────────────────────────────────────────┐
  │                          │                A: RunFlashVarlen                │                    B: 修 unpad                    │               C: paged attention               │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ pad/unpad 状态           │ ❌ 完全不需要                                   │ ✅ 保留但修 OOB                                   │ ❌ 完全不需要                                  │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 要不要 concat cached K/V │ ✅ 要(用 torch.cat)                             │ ✅ 要(改 pad 内部做)                              │ ❌ 不要(kernel 按需读 block)                   │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 改动量                   │ 中(Python 层加 concat,换 API)                   │ 中(C++ 内部加边界检查 + Python 层接 block_tables) │ 中-大(用 triton kernel,vllm 内部跨层改)        │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ MUSA 内核成熟度          │ RunFlashVarlen 在 MuDNN 已有声明,实测稳定性未知 │ RunFlash 已稳定,只需修 unpad                      │ vllm_musa 的 paged attention 现在只用在 decode │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 代码简洁性               │ 最高                                            │ 中等                                              │ 中等                                           │
  ├──────────────────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 保留 unpad               │ ❌ 抛弃                                         │ ✅ 保留                                           │ ❌ 抛弃                                        │
  └──────────────────────────┴─────────────────────────────────────────────────┴───────────────────────────────────────────────────┴────────────────────────────────────────────────┘

  → 你想保留 unpad → 推荐 Path B。

  那 Path B 具体怎么改?

  需要两层改动:

  改动 1:vllm_musa 的 _kernels.so 里的 unpad C++ 实现(MTT 改)
  // 加边界检查
  for (int b = 0; b < bs; b++) {
      int seg_claim = accum_lens[b+1] - accum_lens[b];
      int seg_actual = std::min(seg_claim, sum_seq - offset);  // ★ 加这一行
      for (int i = 0; i < seg_actual; i++) {
          output[offset + i] = ...;
      }   
      offset += seg_actual;
  }   

  改动 2:vllm_musa 的 Python 层 pad 也要扩展(从 KV cache 拉 cached K/V)
  - 否则即使 unpad 不越界,K/V 数据还是错(只有新 8 行,没 cached 部分)
  - pad 内部需要接 block_tables + kv_cache 参数,把 cached 部分填进 k_pad/v_pad 的前 (max - sum_seq) 行
  - 这样 k_pad/v_pad 就有完整 904 行真数据
  - SDPA 输出正确的 attention,unpad 撤回正确结果

  改动 1 修崩,改动 2 修语义错误。两个都做才是完整修复。

  要不要把这三条路径的具体设计 + Path B 的 2 层改动都补到 ISSUE 文档里?
