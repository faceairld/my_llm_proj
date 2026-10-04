# LMCache 移植到新版 vllm_musa 0.20/V1 的可行性判断

> 日期: 2026-06-11  
> vLLM/vllm_musa 环境: `/data/my_vllm_test/vllm_020/vllm-musa`  
> 本地容器: `vllm020_test`  
> 旧 LMCache MUSA 资产: `/data/_backup_to_local/LMCache`

---

## 1. 结论

**可以移植, 而且比旧版更可行。**

原因是新版 vLLM 0.20 已经内置 V1 KV connector 框架和 `LMCacheConnectorV1`, 不再需要像旧版那样自己找 V1 engine hook。旧版失败点是 vllm_musa 当时没有可用 V1 引擎; 现在 `vllm_musa 0.1.1 + vllm 0.20.1` 是纯 V1, 正好满足 LMCache 当前主要接入方式。

但当前还不能说“已跑通”, 因为:

1. `vllm020_test` 容器里还没有安装 `lmcache`;
2. 127 远端 venv 里也没有安装 `lmcache`;
3. 本地备份的 LMCache 是 `0.3.0` MUSA port, 能被 Python 找到, 但缺运行依赖, 首个缺口是 `sortedcontainers`;
4. LMCache 真正和新版 vllm_musa 端到端联调以前, 仍需验证 KV cache layout、chunk/block 对齐、MUSA kernel ABI、prefix-cache 命中时的 load/save 行为。

推荐路线: **先用本地 `/data/_backup_to_local/LMCache` 这份 MUSA port 做本地 editable/symlink 安装, 补齐依赖后跑 import 和核心单元测试, 再做单卡小模型端到端 smoke test。**

---

## 2. 新版 vLLM 侧已经具备的 LMCache 接口

新版 vLLM 包里已经有完整 V1 KV connector 目录:

```text
third_party/vllm/vllm/distributed/kv_transfer/
├── kv_connector/
│   ├── factory.py
│   └── v1/
│       ├── base.py
│       ├── lmcache_connector.py
│       ├── lmcache_mp_connector.py
│       └── lmcache_integration/
│           ├── utils.py
│           └── vllm_v1_adapter.py
└── kv_transfer_state.py
```

`KVConnectorFactory` 已注册:

```text
LMCacheConnectorV1 -> vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector.LMCacheConnectorV1
LMCacheMPConnector -> vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector.LMCacheMPConnector
```

`CacheConfig.kv_offloading_backend == "lmcache"` 时, `VllmConfig._post_init_kv_transfer_config()` 也会自动设置:

```text
kv_connector = "LMCacheConnectorV1"
kv_role = "kv_both"
kv_connector_extra_config = {
  "lmcache.local_cpu": True,
  "lmcache.max_local_cpu_size": kv_gb_per_rank,
}
```

所以新版里有两种启用方式:

```bash
# 显式 connector
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# 或走 kv offload 高层配置
--kv-offloading-backend lmcache --kv-offloading-size <GB>
```

---

## 3. LMCache V1 在 vLLM 中的调用结构

```text
vllm serve / LLM(...)
  │
  ├─ KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
  │
  ├─ KVConnectorFactory.create_connector(...)
  │   ├─ scheduler process: LMCacheConnectorV1(role=SCHEDULER)
  │   └─ worker process:    LMCacheConnectorV1(role=WORKER)
  │
  ├─ scheduler side
  │   ├─ get_num_new_matched_tokens(request, num_computed_tokens)
  │   ├─ update_state_after_alloc(request, blocks, num_external_tokens)
  │   ├─ build_connector_meta(scheduler_output)
  │   └─ request_finished(...)
  │
  └─ worker/model-runner side
      ├─ KVConnectorModelRunnerMixin._get_kv_connector_output()
      │   ├─ bind_connector_metadata(...)
      │   ├─ start_load_kv(forward_context)
      │   └─ wait_for_save() / get_finished()
      │
      └─ attention layer decorator
          └─ maybe_transfer_kv_layer(...)
              ├─ connector.wait_for_layer_load(layer_name)
              ├─ attention forward
              └─ connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这一套正是 LMCache 依赖的 V1 暴露接口。

---

## 4. vllm_musa 与 LMCache 的关键适配点

### 4.1 设备名

vLLM 内置 adapter 里有:

```python
torch.accelerator.set_device_index(local_rank)
device = torch.device(f"cuda:{local_rank}")
```

在 `vllm020_test` 里实测:

```text
torchada_is_musa True
accelerator musa
device_count 8
torch.device("cuda:0") -> musa:0
torch.musa.is_available() -> True
```

所以这个 `cuda:` 字符串在 torchada 环境下会映射成 MUSA, 不是硬阻塞。

### 4.2 KV cache layout

vllm_musa V1 FlashAttention backend 的 KV cache shape:

```text
(2, num_blocks, block_size, num_kv_heads, head_size)
```

LMCache 的 VLLM paged connector 预期从 vLLM KV cache tensor 中发现格式:

```text
KV_2LTD / KV_MLA_FMT
slot_mapping -> block_id * block_size + offset
```

这与 MUSA FlashAttention 的非 MLA 普通 decoder 形状在概念上匹配。需要实测的是:

1. `discover_gpu_kv_format(kv_caches, EngineType.VLLM)` 是否识别 MUSA tensor layout;
2. `multi_layer_kv_transfer` MUSA kernel 是否支持新版 block_size/head_size;
3. prefix-cache 命中时 `slot_mapping` 中 `-1` 前缀是否被正确避开。

### 4.3 attention hook

容器里 runtime patch 后的 vLLM attention 文件仍有:

```text
from vllm.model_executor.layers.attention.kv_transfer_utils import maybe_transfer_kv_layer
@maybe_transfer_kv_layer
```

也就是说 vllm_musa 的 attention patch 没有破坏 LMCache 的 V1 layerwise hook。

### 4.4 cudagraph

`LMCacheConnectorV1.requires_piecewise_for_cudagraph(extra_config)` 表明:

```text
use_layerwise=True 时需要 PIECEWISE CUDA graph mode
```

在 MUSA 上建议第一阶段直接 `--enforce-eager` 或禁用复杂 cudagraph, 先验证功能正确性。等能稳定 store/load 后再打开 cudagraph 做性能验证。

---

## 5. 本地已有 LMCache MUSA port 状态

路径:

```text
/data/_backup_to_local/LMCache
```

已知信息:

| 项 | 值 |
| --- | --- |
| LMCache 版本 | `0.3.0` |
| 旧验证环境 | vllm 0.9.3 + torch_musa 2.5/2.7 系列 |
| MUSA 资产 | `csrc_musa/`, `lmcache/c_ops.so`, `lmcache/native_storage_ops.so`, `lmcache/lmcache_redis.so` |
| 旧测试结果 | memory/kernel 核心 84 个用例通过 |
| 旧阻塞 | vLLM 0.9.3 的 V1/API 不匹配, 不是 MUSA kernel 本身 |

这份 LMCache 已经做过 MUSA 替换:

```text
import torch_musa
torch.musa.Stream()
torch.musa.Event()
device.type == "musa"
VLLMPagedMemGPUConnectorV2/V3
```

临时 import 测试:

```bash
PYTHONPATH=/data/_backup_to_local/LMCache python -c 'import lmcache'
```

结果:

```text
lmcache 包能找到;
import lmcache.v1.gpu_connector 时首个缺失依赖是 sortedcontainers
```

说明当前最先要补的是 Python runtime dependencies, 不是重新 simple-porting。

---

## 6. 当前缺口

### 6.1 依赖缺失

`vllm020_test` 和 127 venv 当前都没有 `lmcache`。本地 LMCache `requirements/common.txt` 包含:

```text
aiofile
aiofiles
blake3
awscrt
cufile-python
msgspec
nixl
numba
nvtx
pyzmq
redis
safetensors
sortedcontainers
cupy-cuda12x
...
```

这些依赖不能盲目 `pip install -r`:

1. `cupy-cuda12x` / `cufile-python` 对 MUSA 环境可能无用甚至冲突;
2. `numpy<=2.2.6` 可能会破坏 vLLM 环境, 旧记录里已经遇到过;
3. 本地 torch 是 2.9.0, 不能让 pip 重新解析安装 torch。

建议先最小化安装:

```text
sortedcontainers, msgspec, redis, safetensors, aiofile/aiofiles,
blake3, pyzmq, psutil, py-cpuinfo, nvtx, numba
```

对 `cupy-cuda12x`, `cufile-python`, `nixl` 先不装, 除非要测远端 disagg/NIXL。

### 6.2 vLLM adapter 版本差异

新版 vLLM 0.20 内置了自己的 `lmcache_integration/vllm_v1_adapter.py`, 但本地 LMCache 0.3.0 也自带:

```text
lmcache.integration.vllm.vllm_v1_adapter
```

`LMCacheConnectorV1` 默认会走外部 LMCache 的最新版 adapter:

```python
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
```

也可以通过 extra_config:

```json
{"use_native": true}
```

强制走 vLLM 包内置 adapter:

```text
vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter
```

建议第一阶段用 `use_native=true`, 因为它和 vLLM 0.20 当前接口最匹配; 之后再评估 LMCache 自带 adapter 是否更新/更完整。

### 6.3 MUSA native .so ABI

LMCache 备份里已有:

```text
lmcache/c_ops.so
lmcache/native_storage_ops.so
lmcache/lmcache_redis.so
```

它们旧环境验证过, 但需要确认与 torch_musa 2.9.0 ABI 是否兼容。若 import `lmcache.c_ops` 报 ABI/符号错误, 需要在 `vllm020_test` 里重新 `FORCE_MUSA=1 pip install -e . --no-build-isolation` 编译。

---

## 7. 建议验证路线

### Phase 0: 不启动模型的 import 验证

```bash
docker exec vllm020_test bash -lc '
  export PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH
  /root/.virtualenvs/sglang-0.5.6/bin/python - <<PY
import lmcache
import lmcache.c_ops
import lmcache.native_storage_ops
import lmcache.v1.gpu_connector
import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector
print("ok")
PY'
```

目标: 补齐纯 Python 依赖并确认 `.so` ABI。

### Phase 1: LMCache 自身核心测试

先跑不依赖模型的核心测试:

```bash
MUSA_VISIBLE_DEVICES=0 python -m pytest tests/v1/test_memory_management.py -q
MUSA_VISIBLE_DEVICES=0 python -m pytest tests/v1/test_mem_kernels.py -q
```

这两个旧环境通过过, 是确认 torch2.9 ABI 的最小测试。

### Phase 2: vLLM connector 构造测试

不加载大模型, 只验证:

1. `KVConnectorFactory.get_connector_class("LMCacheConnectorV1")`;
2. `LMCacheConnectorV1(..., role=SCHEDULER/WORKER, kv_cache_config=...)`;
3. `use_native=true` 与默认外部 adapter 两条路径。

### Phase 3: 单卡小模型端到端

用小模型、短上下文、`--enforce-eager`, 端口避开本机已有服务:

```bash
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=2 \
MUSA_VISIBLE_DEVICES=0 \
vllm serve <small-model> \
  --port <free-port> \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"discard_partial_chunks":false}}'
```

通过标准:

1. 服务启动成功;
2. 两次 shared-prefix 请求都成功;
3. 日志出现 LMCache store/retrieve 或 matched tokens;
4. 输出与不开 LMCache 时语义一致。

### Phase 4: 原目标场景

在单卡通过后再验证:

1. Qwen3-32B / Qwen3-32B-FP8;
2. TP=4/TP=8;
3. prefix-cache 命中;
4. long context;
5. 如需要, 再考虑 `LMCacheMPConnector`、NIXL/disagg。

---

## 8. 风险判断

| 风险 | 等级 | 说明 |
| --- | --- | --- |
| LMCache 未安装 | 中 | 可控, 本地已有 MUSA port 源码和 .so |
| Python 依赖污染 vLLM | 中 | 不能直接 `pip install -r`; 要最小化补依赖 |
| LMCache .so 与 torch2.9 ABI 不兼容 | 中高 | 需要 import/核心测试确认; 不行就重编 |
| KV cache layout 不匹配 | 中 | vllm_musa shape 看起来兼容, 但必须实测 transfer kernel |
| cudagraph + layerwise load/save | 中 | 先 enforce eager 避开 |
| TP/多进程 | 高 | 单卡通过后再做; 旧 LMCache MP 路径也改了 MUSA, 但未在新版 vLLM 上验证 |
| 127 当前测试被影响 | 低 | 当前只读检查; 后续测试在本地 `vllm020_test` 做 |

---

## 9. 最终判断

LMCache **不是没有接口可接**, 新版 vllm_musa 的 V1 引擎正好提供了需要的 V1 KV connector hook。真正工作量在:

1. 把 `/data/_backup_to_local/LMCache` 这份 MUSA port 安装进 `vllm020_test`;
2. 补齐但不污染 vLLM 的依赖;
3. 验证 `.so` 与 torch_musa 2.9 ABI;
4. 用 `LMCacheConnectorV1 + use_native=true + enforce_eager` 先跑单卡端到端;
5. 再扩到 TP、long-context、prefix-cache 命中场景。

因此结论是: **可行, 推荐继续做; 但当前状态是“接口和代码路径可行”, 不是“功能已验证跑通”。**

---

## 10. 2026-06-16 轻量 import/test 结果

测试容器: `vllm020_test`

当前容器状态:

```text
vllm020_test   Up   registry.mthreads.com/mcconline/inference/sglang:v0.5.6.post2-ph1-4.3.5-torch2.9.0-20260403
```

### 10.1 vLLM/vllm_musa 覆盖状态

容器内 import 通过:

| 包 | 结果 | 版本 |
| --- | --- | --- |
| `vllm` | OK | `0.20.1.dev0+g88d34c640.d20260519` |
| `vllm_musa` | OK | `0.1.1` |
| `torch` | OK | `2.9.0` |
| `torch_musa` | OK | `2.9.0+0680da1` |
| `openai` | OK | `2.37.0` |
| `flash_attn_3` | OK | `0.1.4` |

`vllm_musa` import 时仍会打印一次 plugin circular import 日志, 随后 `Platform plugin musa is activated`, 与前面 `vllm --help` 观察一致。

### 10.2 LMCache import 验证

使用:

```bash
export PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH
```

通过项:

| 模块 | 结果 |
| --- | --- |
| `lmcache` | OK |
| `lmcache.c_ops` | OK |
| `lmcache.native_storage_ops` | OK |
| `lmcache.lmcache_redis` | OK |
| `lmcache.v1.gpu_connector` | OK |
| `lmcache.v1.gpu_connector.gpu_connectors` | OK |
| `lmcache.integration.vllm.utils` | OK |
| `lmcache.integration.vllm.vllm_v1_adapter` | OK |
| `vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector` | OK |

失败项:

```text
vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter
ImportError: cannot import name 'LMCacheEngineMetadata' from 'lmcache.config'
```

解释: 这是 vLLM 仓内 native adapter 需要新版 LMCache Python API, 但当前 `/data/_backup_to_local/LMCache` 是旧版 0.3.0 MUSA port。默认 `LMCacheConnectorV1` 的 `use_native=false` 路径会使用 LMCache 包自带的 `lmcache.integration.vllm.vllm_v1_adapter`, 该路径 import 已通过。

### 10.3 LMCache 轻量单元测试

补过的测试依赖:

```bash
pip install --no-deps sortedcontainers redis aiofile aiofiles nvtx caio
pip install --no-deps pytest-asyncio backports.asyncio.runner
```

基础测试命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/test_utils.py \
  tests/test_protocol.py \
  tests/test_serde.py \
  tests/v1/test_basic_check.py \
  tests/v1/test_config.py
```

结果:

```text
160 passed, 1 xpassed, 3 failed in 168.24s
```

3 个失败均为容器缺少 `pytest-asyncio` 导致的 async 测试运行器问题, 不是 LMCache 功能断言失败。补齐 `pytest-asyncio` 和 `backports.asyncio.runner` 后重跑失败项:

```bash
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_basic_check.py::TestMain::test_list_mode \
  tests/v1/test_basic_check.py::TestMain::test_unknown_mode \
  tests/v1/test_basic_check.py::TestMain::test_valid_mode_invokes_function
```

结果:

```text
3 passed in 1.20s
```

因此基础轻量测试结论按合并结果看是:

```text
163 passed, 1 xpassed
```

### 10.4 最小 MUSA allocator 测试

未跑完整 `test_memory_management.py`, 只跑了最小 inplace allocator 覆盖 Host/Pin/GPU/Mixed 四类 allocator, 避免 128MB/page allocator 和更重 kernel 测试:

```bash
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_memory_management.py::test_inplace_modification
```

结果:

```text
4 passed in 8.29s
```

### 10.5 当前结论

到轻量测试层面:

1. 新版 `vllm_musa` 覆盖环境仍可 import;
2. LMCache MUSA port 的 Python 包和三个 `.so` 可以在 torch/torch_musa 2.9 容器内加载;
3. LMCache 基础配置/协议/序列化/工具函数测试通过;
4. 最小 MUSA allocator 测试通过;
5. 还没有跑完整 LMCache kernel 测试、完整 memory management、也没有启动模型做 vLLM 端到端。

下一步建议先跑 `tests/v1/test_memory_management.py` 完整模块, 再跑 `tests/v1/test_mem_kernels.py`; 两者通过后再用 `ISSUE_vllm_musa_broadcast_deadlock.md` 里的模型负载脚本做 vLLM+LMCache 端到端验证。

---

## 11. 2026-06-16 继续测试结果

### 11.1 完整 memory management

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_memory_management.py
```

结果:

```text
47 passed in 9.58s
```

说明: pytest 摘要通过。退出阶段仍有 `MemoryObj ... garbage collected with ref_count=1` 日志噪声, 与旧测试记录一致。

### 11.2 mem kernels

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_mem_kernels.py
```

结果:

```text
19 passed, 18 failed, 1 warning in 21.20s
```

失败原因主要是当前本机 GPU 被已有 `sglang` TP=8 服务占用, 不是数值断言不一致。`mthreads-gmi` 显示每张 S5000 约 75GB 已占用, 只剩约 5-6GB:

```text
0..7 MTT S5000: ~75348-75437MiB / 81920MiB used
Processes: sglang::scheduler_TP0..TP7, each rank on对应卡约75017-75023MiB
```

典型失败:

```text
torch.OutOfMemoryError: MUSA out of memory
RuntimeError: musaHostAlloc failed: 2
```

已通过的 kernel 子项包括:

1. `test_extract_and_load_back` 的 `256/500/1024` token;
2. 多数 `test_single_layer_kernel`;
3. `test_lmcache_memcpy_async`;

失败主要集中在:

1. `8000` token 的大张量路径;
2. multi-layer kernel, 需要构造 32 层 KV cache;
3. MLA multi-layer 路径;
4. 4GB pinned CPU buffer 分配。

结论: 在当前本机已有大模型服务占卡的状态下, `test_mem_kernels.py` 不能作为最终通过/失败结论。需要等空闲卡, 或停止/迁移本机 `sglang` 服务后重跑。

### 11.3 V1 逻辑测试

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_address_manager.py \
  tests/v1/test_cache_policy.py \
  tests/v1/test_kv_layer_groups_manager.py \
  tests/v1/test_token_database.py \
  tests/v1/test_remote_metadata.py
```

结果:

```text
61 passed, 8 skipped in 11.52s
```

### 11.4 Connector/health/observability

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_connector_discovery.py \
  tests/v1/test_impl_completeness.py \
  tests/v1/test_health_monitor.py \
  tests/v1/test_health_monitor_fallback_recovery.py \
  tests/test_observability.py
```

结果:

```text
65 passed in 4.74s
```

其中有一个 async pending task 日志, 但 pytest 结果通过。

### 11.5 Manager mock 测试

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q \
  tests/v1/test_manager.py
```

结果:

```text
30 passed in 1.58s
```

### 11.6 当前汇总

截至本轮, 除去前面已经单独跑过的最小 allocator 重复项, 当前主要结果:

| 测试范围 | 结果 |
| --- | --- |
| import / `.so` 加载 | 通过 |
| 基础 utils/protocol/serde/basic/config | `163 passed, 1 xpassed` |
| 完整 memory management | `47 passed` |
| V1 address/cache policy/layer group/token db/remote metadata | `61 passed, 8 skipped` |
| connector/health/observability | `65 passed` |
| manager mock | `30 passed` |
| mem kernels | `19 passed, 18 failed` |

`mem kernels` 的失败当前判断为资源不足导致, 因为所有失败栈都指向 MUSA OOM 或 `musaHostAlloc failed`, 且本机 8 卡均被已有 `sglang` 服务占用约 75GB/卡。没有看到 kernel 输出值不一致的断言失败。

下一步建议:

1. 等本机 8 卡服务结束或找一张空闲 S5000, 重跑 `tests/v1/test_mem_kernels.py`;
2. 再跑 `tests/v1/test_cache_engine.py` 这类真正使用 MUSA paged KV 的用例;
3. 以上通过后, 再进入 vLLM+LMCache 端到端模型负载验证。
