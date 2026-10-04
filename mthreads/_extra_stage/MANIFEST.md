# 增量文件清单（相对 `/data/my_vllm_test.tar.gz`，2026-06-16 10:46 那版）

本压缩包收录的是 **当前 `/data/my_vllm_test` 文件夹比上述基线压缩包多出来的 16 个文件**（只增不删，基线里没有任何文件被删除）。
所有路径均相对于 `my_vllm_test/` 根目录，解压后可直接对应原目录结构。

打包日期：2026-06-17

---

## 一、LMCache benchmark 相关（2026-06-17 新增）

| 路径 | 说明 |
|---|---|
| `ISSUE_vllm_musa_broadcast_deadlock2.md` | 续篇主文档。记录新版 vllm_musa(V1) 上配 LMCache 的全过程：APC vs LMCache 机制、读/写路径、Step0-2 性能测试、冷热分离测试用例结构、benchmark 指标速查、多卡/多机扩展方向。承接 `ISSUE_vllm_musa_broadcast_deadlock.md` |
| `vllm_020/lmcache_cold_warm_revisit_bench.py` | 冷热分离 benchmark 脚本，用于验证 LMCache 性能收益。不随机打乱请求：cold 阶段每个 shared-prefix 灌一次 cache，warm 阶段同序复访只统计收益；前后抓 `/metrics` 区分 APC / LMCache 命中 |
| `vllm_020/__pycache__/lmcache_cold_warm_revisit_bench.cpython-310.pyc` | 上面脚本的 Python 编译缓存（自动生成，可忽略） |
| `vllm_020/start_qwen3_8b_vllm_musa.sh` | qwen3-8b vllm_musa 服务启动脚本，`MODEL_PATH` / `PORT` / `GPU_IDS` 等环境变量可配 |
| `vllm_020/sitecustomize.py.disabled` | vLLM MUSA 环境的运行时兼容补丁（**已禁用**，扩展名带 `.disabled`）。仅当 `vllm_020` 置于 PYTHONPATH 最前时生效，用于绕开 torch_musa 2.9 在 S5000 上的行为。feasibility 文档结论是不应靠手写 sitecustomize 补 API，应对齐运行时包，故保留为禁用态 |

## 二、PD 分离（prefill/decode disaggregation）相关（2026-06-16 新增）

| 路径 | 说明 |
|---|---|
| `vllm_musa_proj/auto_bench/pd_smoke_test.sh` | PD 分离冒烟测试。验证 mooncake KV 搬运在 MTT S5000 上是否可用；布局 = prefill tp4(卡0-3, producer) + decode tp4(卡4-7, consumer) + proxy。改编自 vllm_musa 官方 disaggregated_serving 示例 |
| `vllm_musa_proj/auto_bench/toy_proxy_server.py` | PD 的 prefill/decode 代理服务器（改自 vLLM 官方 example） |
| `vllm_musa_proj/auto_bench/pd_monitor.py` | PD 过载观测采样器：抓 prefill(8100)/decode(8200) 的请求队列 + KV 填充率 + 卡 util，区分「prefill 真算不过来」vs「prefill 空等交接」 |
| `vllm_musa_proj/auto_bench/pd_monitor_run.sh` | `pd_monitor.py` 的运行封装脚本 |
| `vllm_musa_proj/auto_bench/pd_probe.py` | PD 并发爬坡拆解：并发 K=1/8/32/64 各打一轮，用 TTFT − prefill 定位是 mooncake 发送池还是代理成为瓶颈 |
| `vllm_musa_proj/auto_bench/pd_probe.sh` | `pd_probe.py` 的运行封装脚本 |
| `vllm_musa_proj/auto_bench/tp_pd_params_32b_bf16.json` | Qwen3-32B-BF16 的 TP+PD 压测参数（served_name、cases 的 input/output/rate 序列等） |
| `vllm_musa_proj/auto_bench/tp_pd_params_32b_fp8.json` | Qwen3-32B-FP8 的 TP+PD 压测参数 |

## 三、auto_bench PD 改造前备份（2026-06-16）

| 路径 | 说明 |
|---|---|
| `vllm_musa_proj/auto_bench/_backup_20260616_prePD/auto_bench.py` | PD 分离改造**之前**的 `auto_bench.py` 备份 |
| `vllm_musa_proj/auto_bench/_backup_20260616_prePD/run_list.py` | 同上，`run_list.py` 改造前备份 |
| `vllm_musa_proj/auto_bench/_backup_20260616_prePD/test_list.py` | 同上，`test_list.py` 改造前备份 |

---

> 备注：Step2 的实验结果 JSON（`vllm_020/bench_results_lmcache_step2/*.json`）不在本包内——它们存在 146 服务器容器里，本地 `/data/my_vllm_test` 没有这些文件。
