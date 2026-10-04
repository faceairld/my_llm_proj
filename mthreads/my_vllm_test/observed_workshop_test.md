# 观察记录：vllm-workshop 串行测试

> 观察时间：2026-05-14 16:46+
> 用途：作为对照基线，方便对比我们自己的测试

## 一句话

vllm-workshop 在 node165 上反复跑 **Qwen3-32B TP=8** 单机 vllm benchmark，每轮约 25 分钟，自 4 月底起几乎每 25 分钟一次（log 目录有几百个历史日志）。所以**这台机器的 GPU 几乎一直被占用**。

## 镜像 / 容器

```yaml
# docker-compose.yml（位置：/data/SETS-2.0-test/vllm-workshop/vllm-musa-qwen3-32b/）
image: sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel
container_name: vllm-musa-qwen3-32b-test
network_mode: host        # 直接占用 host 8000 端口
privileged: true
shm_size: 500g
environment:
  - MTHREADS_VISIBLE_DEVICES=all
  - USE_PYTHON_INCLUDE_DIR=/usr/include/python3.10
  - USE_CUSTOM_ALLREDUCE=1     # ← 启用 vllm_musa 的 CustomAllreduce 路径（旁路 MCCL）
  - SETS_BASE_DIR=/data/SETS-2.0-test
volumes:
  - /data:/data
command: bash -c "cd /data/SETS-2.0-test/vllm-workshop/vllm-musa-qwen3-32b && bash run.sh"
```

**镜像内版本**（用 docker exec 抓的）：
- torch: 2.7.1
- torch_musa: 2.7.1
- vllm: 0.9.3.dev0+ga5dd03c1e.**d20260209**.empty
- vllm_musa: 0.1.dev358+gd3980ed.**d20260209**

**注意**：这个版本（20260209）比我们之前死锁那次的（20260309）还**老 1 个月**；最新的 `sets_vllm_musa:torch_2.7.1_fix_ray` 是 20260323。

## 脚本调用链

```
docker-compose-up.sh        # host 端启动入口
    ↓ docker compose up
run.sh                       # 容器内 entrypoint
    ↓ exec
run_vllm_series_test.sh      # 主循环：MODEL_TEST_LIST 里每个模型跑一遍
    ↓ for each MODEL
run_vllm_benchmark.sh        # 单模型流程：启 vllm → 等就绪 → 跑测试 → 停 vllm
    ├── 启 vllm:  run-vllm-musa-1.sh
    ├── 等就绪:   grep "Application startup complete" 在 vllm log
    ├── 跑测试:   sanity-check.sh
    └── 停 vllm:  pkill -f "vllm serve"
```

## 关键配置

### `run-vllm-musa-1.sh`（vllm serve 启动参数）

```bash
VLLM_USE_V1=0 vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --served-model-name "${MODEL_NAME}" \
    --block-size 64 \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size 1 \
    --compilation-config '{"cudagraph_capture_sizes": [1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30,32,50,64,100,128,256], "simple_cuda_graph": true}'
```

**与我们之前的对比**：
| 项 | workshop | 我们之前 |
|---|---|---|
| `VLLM_USE_V1=0` | ✅ | ✅ |
| `--gpu-memory-utilization` | 0.7 | 0.85（更激进） |
| `--tensor-parallel-size` | 8 | 8 |
| `--enable-prefix-caching` | ❌ 没开 | ✅ 开了 |
| `--max-num-seqs` | 默认（应该是 256） | 显式 256 |
| `--block-size` | 64 | 64 |
| `cudagraph_capture_sizes` | 含到 256 | 含到 256 |
| `USE_CUSTOM_ALLREDUCE` env | =1 | 未设置 |

**主要差异**：workshop **没开 prefix caching**，但**显式启用了 USE_CUSTOM_ALLREDUCE=1**（vllm_musa 自己的 fast allreduce，旁路 MCCL）。这可能是它不死锁的原因之一。

### `sanity-check.sh`（压测参数）

```bash
IO_PAIRS=("4096 1024")              # 输入 4k token，输出 1k token
CONCURRENCY_AND_PROMPTS=("256 512") # 最大并发 256，总共 512 条 prompt
DATASET_NAME="random"

# 调用
vllm bench serve \
    --model "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --num-prompts 512 \
    --max-concurrency 256 \
    --ignore-eos \
    --save-result \
    --percentile-metrics 'ttft,tpot,itl,e2el'
```

**这就是会触发我们之前 bug 的条件**（长 prompt + 高并发），但 workshop 这边没死锁，因为：
1. 没开 prefix-caching
2. 镜像版本不同（vllm_musa 20260209）
3. USE_CUSTOM_ALLREDUCE=1（数据通过 custom 路径，不走 MCCL gather）

## 模型列表

```bash
MODEL_TEST_LIST=("Qwen3-32B 8")
```
**只跑 Qwen3-32B 一个模型，TP=8**。每轮跑完释放后 sleep 15 秒。

## 测试触发频率

`/data/SETS-2.0-test/vllm-workshop/vllm-musa-qwen3-32b/log/` 下：
- 历史日志从 2026-04-24 开始
- 每个 vllm_server_Qwen3-32B_*.log 28MB（vllm 服务日志）
- 每个 vllm_test_Qwen3-32B_*.log 2KB（benchmark 结果）
- **时间戳间隔约 23 分钟**（启动 1.5 分钟 + benchmark 20 分钟 + 清理）
- 已运行 100+ 次

**所以这台机器 GPU 几乎 24/7 在跑这个测试**。除非 sleep 15 那个间隙 + 同时 wan2.2 也没跑，否则我们抢不到 GPU 长时间。

## 怎么看输出（不进容器，直接 host 端）

```bash
WORKDIR=/data/SETS-2.0-test/vllm-workshop/vllm-musa-qwen3-32b

# 1. 当前轮次的 vllm 服务日志（实时刷新）
tail -f $WORKDIR/log/vllm_server_Qwen3-32B_*.log | tail -100   # 最新一个

# 2. 当前轮次的 benchmark 结果（vllm bench serve 输出）
tail -f $WORKDIR/log/vllm_test_Qwen3-32B_*.log

# 3. 主入口日志
tail -f $WORKDIR/log/vllm-main.log

# 4. 整体状态
cat $WORKDIR/vllm-musa.status
# 输出：state=running / success / failed

# 5. 历史结果 CSV
ls $WORKDIR/result/*.csv
cat $WORKDIR/result/vllm_gpu8_Qwen3-32B_*.csv | tail
```

**简化看进度**：
```bash
# 看 vllm 是否启动完毕
grep -c 'Application startup complete' $WORKDIR/log/Qwen3-32B-tp8.log

# 看本轮 benchmark 是否在跑/跑完
tail -3 $WORKDIR/log/Qwen3-32B-tp8.log
```

## 与我们的目标对照

| 维度 | workshop | 我们 |
|---|---|---|
| 目的 | 性能监控 / 基线 | 验证新镜像是否修复 MCCL 死锁 |
| 镜像 | torch2.7.1-intel (20260209) | sets_vllm_musa:torch_2.7.1_fix_ray (20260323) ⭐ |
| 模型 | Qwen3-32B（62G） | Qwen2.5-7B（15G，避免抢资源） |
| TP | 8 | 8 |
| 关键开关 | USE_CUSTOM_ALLREDUCE=1 | 不开，强制走 MCCL（暴露 bug） |
| prefix-caching | 不开 | **开**（保留触发条件） |
| GPU util | 0.7 | 0.12（让出空间共占） |

## 风险点

1. **会和 workshop 抢同样的 8000 端口** —— 必须改成别的端口（如 8001）
2. **抢 GPU 显存** —— workshop 已经吃 70% × 80GB ≈ 56GB/卡，我们最多用 0.12-0.15
3. **如果 workshop 也跑长 prompt 时正好触发 bug**，可能它先死锁，影响我们观察

## workshop 实测一次结果（2026-05-14 16:46）

抓到一次完整运行的结果（`result/20260514_164601/`）。

### 测试条件
```
input_len=4096   output_len=1024
max_concurrency=256   num_prompts=512
model=Qwen3-32B   TP=8
```

### 结果（CSV 全文）
```
Successful_requests: 512 / 512        ← 没死锁，全部成功
Benchmark_duration_s: 302.38
Total_input_tokens: 2,094,357
Total_generated_tokens: 524,288
Request_throughput_req_s: 1.69
Output_token_throughput_tok_s: 1733.88
Total_Token_throughput_tok_s: 8660.15

Mean_TTFT_ms:   30880.84
Median_TTFT_ms: 31235.08
P99_TTFT_ms:    56474.97

Mean_TPOT_ms:    117.55
Median_TPOT_ms:  117.83
P99_TPOT_ms:     143.95

Mean_E2EL_ms:    151136.86   (≈ 2.5 分钟一条)
P99_E2EL_ms:     152656.70
```

### 关键观察

**workshop 的测试条件比我们之前死锁那次还激进**（更大模型、更高并发、更长输出），但它**没死锁，512/512 成功完成**。

| 维度 | 我们的死锁场景 | workshop（成功） |
|---|---|---|
| 输入长度 | ~3-4k | 4096 ✓ |
| 输出长度 | 200 | 1024（更大） |
| 并发数 | 16 | 256（远高） |
| 模型 | Qwen3-8B (16G) | Qwen3-32B (62G) |
| 跑了几分钟 | 几分钟内必死 | 跑完 5 分钟无问题 |

### 关键差异（嫌疑点）

| 配置 | 我们的（死锁） | workshop（OK） | 嫌疑等级 |
|---|---|---|---|
| vllm_musa | 20260309 | 20260209 | 排除（更老反而 OK） |
| **`USE_CUSTOM_ALLREDUCE`** | 不设 | **=1** | ⭐⭐⭐ |
| **`--enable-prefix-caching`** | ✅ 开 | ❌ 关 | ⭐⭐⭐ |
| `--gpu-memory-utilization` | 0.85 | 0.7 | ⭐ |
| `--max-num-seqs` | 256 | 默认 | ⭐ |

### 推论：bug 可能是绕得过去的

**`USE_CUSTOM_ALLREDUCE=1`** 这个环境变量让 vllm_musa 启用自己写的 fast path allreduce（`vllm_musa/patch/custom_allreduce/`），**完全旁路 MCCL**。

我们之前死锁现场分析（见 ISSUE_vllm_musa_broadcast_deadlock.md）：
- broadcast / gather 都在 MCCL `mcclEnqueueCheck` 之后挂死
- bug 在 MCCL 库内部

**如果走 custom_allreduce 路径绕过 MCCL，bug 可能就没了**。

但要注意：**`USE_CUSTOM_ALLREDUCE=1` 只影响 allreduce，不影响 broadcast / gather**。我们之前死锁卡的是 broadcast 和 gather，不是 allreduce。所以严格说这个开关**不能直接绕过我们的 bug**。

那 workshop 没死锁的真实原因更可能是 **prefix-caching 没开**：
- prefix-caching 涉及 KV cache 的 hash lookup、block 复用
- 可能在长 prompt 高并发时让 vllm 的内部状态进入某种"边界条件"
- 这种边界条件让 MCCL broadcast/gather 进入死锁

### 待验证的实验

| 实验 | 改 run.sh 怎么改 | 期望 |
|---|---|---|
| A. 加 `USE_CUSTOM_ALLREDUCE=1` | export env | 如果不死锁，custom 路径意外有效 |
| B. 关 `--enable-prefix-caching` | 注释那行 | 如果不死锁，prefix-caching 是触发条件 |
| C. AB 同时开 | 同时改 | 等价 workshop 配置，期望肯定不死锁 |

跑完 A、B 单独的对比，能定位 root cause 究竟是哪个。

## 历史结果 CSV 在哪

```
/data/SETS-2.0-test/vllm-workshop/vllm-musa-qwen3-32b/result/YYYYMMDD_HHMMSS/
  ├── vllm_gpu8_Qwen3-32B_<ts>_results.csv       ← 主指标
  └── vllm_gpu8_Qwen3-32B_<ts>_results/
       └── openai-infqps-concurrency256-Qwen3-32B-<ts>.json  ← vllm bench serve 原始输出
```

最近若干次结果（4 月 24 - 5 月 14）都保留，**保留 14 天**（调度器配置 `retention_days: 14`）。

## 怎么把我们的测试塞进队列

### 调度器的"真实"门槛

`prepare_queue_test.sh` 里那个 `SUPPORTED_TESTS=(vllm-musa-qwen3-32b, wan2_2)` **只是这个便捷工具的白名单**，**调度器自己根本不看它**。

调度器实际逻辑（`scheduler.py` 的 `scan_test_items`）：
```python
for item_dir in sorted(test_root.iterdir()):
    if item_dir.is_dir():
        item = TestItem.from_path(item_dir)   # ← 只看有没有 config.yaml
        if item:
            items.append(item)
```

**任何目录只要有 `config.yaml`，调度器就当成测试项加入队列**。

### 当前队列实情

```json
{
  "order": [
    "vllm-musa-qwen3-32b",          ← 主测试（每轮跑）
    "wan2_2",                        ← 视频生成
    "vllm-musa-qwen3-32b-queue-1",
    "wan2_2-queue-1",
    "vllm-musa-qwen3-32b-queue-2",   ← 但被 disabled
    "vllm-musa-qwen3-32b-queue-3"    ← 被 disabled
  ],
  "disabled": ["vllm-musa-qwen3-32b-queue-2", "vllm-musa-qwen3-32b-queue-3"]
}
```

当前 state.json:
- `last_round_duration: 1403` 秒 = **23.4 分钟**
- `last_round_completed: 4` 个测试
- `current_index: 0` 跑到第 0 个

一轮 = 串行跑 4 个测试 = 23 分钟。

### 我们插进去的时间估计

| 阶段 | 时间 | 说明 |
|---|---|---|
| 等当前轮跑完 | **5-23 分钟**（看现在跑到第几个） | 调度器不打断当前测试 |
| 我们在本轮的位置 | 默认插到末尾（位置 5） | 还要等前 4 个跑完 |
| 我们自己跑 | 1-6 分钟 | 取决于死锁 / 成功 |
| **第一次结果延迟** | **~25-40 分钟** | 最坏情况 |
| 之后每次结果 | **每 ~23 分钟一次** | 等本轮的前 4 个 + 我们 |

### 加速选项

| 办法 | 首次结果 | 每次迭代 | 工作量 | 干扰别人 |
|---|---|---|---|---|
| **A. 进队列末尾**（最稳） | 25-40 分钟 | 23 分钟 | 写 3 个文件 | 无 |
| **B. 进队列首位** | 5-10 分钟 | 23 分钟 | 写 3 个文件 + 改 queue.json 顺序 | 改了别人的设置 |
| **C. 改 queue.json 把别人 disable 一部分** | 取决于禁用了几个 | 缩短到 ~15 分钟 | 改 queue.json | 影响别人回归 |
| **D. 不进队列，人工触发** | 几分钟（等 wan2_2 完） | 几分钟 | 写 3 个文件，手动 docker compose up | 抢 GPU 时段 |
| **E. 借现有 vllm 服务（workshop 的 8000）** | 即时 | 即时 | 0 | 不影响别人，但测的是 workshop 镜像 |

### 办法 B/C 怎么改 queue.json

`queue.json` 是普通 JSON 文件，路径：
```
/data/SETS-2.0-test/sanity-check-scheduler-main/data/queue.json
```

调度器每轮开始时会重新读这个文件，所以**直接编辑生效**。

```bash
# 把我们插到 order 的第 0 位（最先跑）
vim /data/SETS-2.0-test/sanity-check-scheduler-main/data/queue.json
# 修改 "order" 数组，第一个改成我们的目录名

# 或者把别人的加进 disabled（不影响他们的测试目录文件，只是这轮跳过）
# disabled 列表里加 "vllm-musa-qwen3-32b-queue-1" 等
```

### 办法 D（推荐快速迭代用）：手动触发

如果不进队列，直接 docker compose 自己跑：
```bash
cd /data/SETS-2.0-test/vllm-workshop/<我们的目录>
bash docker-compose-up.sh up   # 触发一次跑流程
```

调度器**不会重复**跑我们（只要我们没在 order 里），但我们自己想跑就跑。

### 办法 E：借现有服务的局限

workshop 跑的 vllm 在 `localhost:8000`，可以直接用我们 `benchmark.py` 打：
```bash
python benchmark.py --port 8000 --scenarios long_context --concurrency 1 --requests 4
```
**但镜像不是最新的**（vllm_musa 20260209 vs 我们要测的 20260323），所以**只能作参考**，不能验证新版本是否修复 bug。

### 实际建议

**双轨**：
1. **写好 3 个文件放到 `vllm-workshop/gy-vllm-musa-newimg/`**：调度器自动加入，长期每 23 分钟一次（无人值守收集数据）
2. **同时手动 `docker compose up` 立刻触发一次**：现在就有结果，不用等

两条路不冲突，因为产生的 result 子目录用 timestamp 区分。
