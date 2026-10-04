# LMCache 移植到新版 vllm_musa 0.20/V1 的可行性判断

> 创建日期: 2026-06-11  
> 最近更新: 2026-06-17  
> vLLM/vllm_musa 环境: `/data/my_vllm_test/vllm_020/vllm-musa`  
> 本地容器: `vllm020_test`  
> 旧 LMCache MUSA 资产: `/data/_backup_to_local/LMCache`

本文是交接文档, 目标不是只证明“能不能跑”, 而是让后来接手的人知道:

1. 为什么新版 `vllm_musa` 比旧版更适合接 LMCache;
2. 需要迁移哪些文件、容器、Python 包和运行时库;
3. 迁移后应该按什么顺序验证, 先验证纯 vLLM, 再验证 LMCache;
4. 遇到过哪些错误, 每个错误背后的真实原因是什么;
5. 后续如果要在新机器上复现, 应该照什么步骤执行。

---

## 0. 先读这一节: 相关组件到底是什么

如果没有接触过 MUSA / vLLM / LMCache, 后面看到 `torchada`、`mate`、`LMCacheConnectorV1`、`LocalCPUBackend` 会很难判断它们各自负责什么。这一节先把名字讲清楚。

### 0.1 整体调用链

本项目的推理请求大致经过下面这条链:

```text
OpenAI API 请求
  │
  ▼
vLLM V1 engine
  │  负责调度请求、管理 KV cache、调用模型 forward
  │
  ▼
vllm_musa
  │  把 vLLM 的 GPU/attention 执行路径适配到 MUSA 设备
  │
  ├─ torch / torch_musa / torchada
  │    负责 PyTorch 层的 MUSA 设备、内存、accelerator 抽象
  │
  ├─ mate / flash_attn_3
  │    负责 MUSA 上的 FlashAttention / paged attention kernel
  │
  └─ LMCacheConnectorV1
       在 vLLM V1 的 KV connector hook 上接入 LMCache
       │
       ▼
     LMCache LocalCPUBackend
       把被复用的 KV 存到 CPU 内存, 需要时再搬回 GPU
```

所以排查时要分层:

1. `vllm_musa` 自己能不能跑 V1 engine;
2. MUSA 运行时包是不是和 `vllm_musa` 匹配;
3. LMCache Python 包和 MUSA `.so` 能不能加载;
4. vLLM V1 connector 能不能调用 LMCache;
5. benchmark 里 LMCache 是否真的 external hit。

前两层没过时, 不要先怀疑 LMCache。

### 0.2 核心名词解释

| 名词 | 它是什么 | 在本文里为什么重要 |
| --- | --- | --- |
| MUSA | 摩尔线程的 GPU 计算平台, 类似 CUDA 在 NVIDIA 生态里的位置 | 所有模型推理和 LMCache MUSA kernel 都跑在 MUSA 设备上 |
| `torch_musa` | PyTorch 的 MUSA 后端包 | 提供 `torch.musa`、MUSA tensor、MUSA memory 等能力 |
| `torchada` | MUSA 对 PyTorch accelerator 抽象的适配/patch 层 | 让 `torch.accelerator.*` 这类上层接口正确落到 `torch_musa`。版本不对会导致 vLLM V1 worker 初始化失败 |
| `mate` | MUSA attention/kernel 运行库。注意名字是 `mate`, 不是 `meta` | vLLM MUSA 的 V1 attention backend 会调用它的 FlashAttention 接口。版本不对会导致 attention 函数参数不匹配 |
| `flash_attn_3` | FlashAttention Python 接口包 | vllm_musa 的 attention backend 会经由它调用底层 attention 实现 |
| `vllm` | 推理框架主体 | 负责 OpenAI API、请求调度、KV cache、模型执行框架 |
| `vllm_musa` | vLLM 到 MUSA 的适配插件 | 把 vLLM 的设备、attention、kernel 调用适配到 MUSA |
| V1 engine | vLLM 新版执行引擎 | 当前新版 `vllm_musa` 只有 V1 路径, LMCache 也是通过 V1 KV connector 接入 |
| KV cache | Transformer 推理中的 key/value 缓存 | prefix 复用、APC、LMCache 都是在复用或搬运 KV cache |
| APC / prefix cache | vLLM 内置 GPU 前缀缓存 | KV 在 GPU 上, 命中最快; 但容量受 GPU 显存限制 |
| LMCache | 外部 KV cache 系统 | 可以把 KV 放到 CPU/磁盘/远端, 用更大容量弥补 GPU APC 容量不足 |
| `LMCacheConnectorV1` | vLLM V1 里的 LMCache connector | vLLM 通过它向 LMCache 查询/保存 KV |
| `LocalCPUBackend` | LMCache 的本地 CPU 内存后端 | 当前实际跑通的 LMCache 后端, 不是 remote/NIXL |

### 0.3 为什么 `torchada` / `mate` 会影响 LMCache 迁移

这两个包不是 LMCache 的包, 但它们决定 **纯净 `vllm_musa` baseline 能不能跑起来**。

本次迁移里一开始看起来像是 vLLM API 不兼容, 实际是:

```text
torchada 旧 -> torch.accelerator.empty_cache 没 patch 到 torch_musa -> V1 worker 初始化失败
mate 旧     -> FlashAttention 函数签名不支持 page_table       -> dummy/profile forward 失败
```

也就是说, 当时如果继续给 LMCache 或 vLLM 打补丁, 只是在错误层级修问题。正确做法是先对齐 127 上已经能跑的 MUSA 运行时包, 再验证纯 vLLM baseline。

---

## 1. 结论

**新版 `vllm_musa` 上移植 LMCache 是可行的, 并且 146 单卡 Qwen3-8B 场景已经跑通。**

原因是新版 vLLM 0.20 已经内置 V1 KV connector 框架和 `LMCacheConnectorV1`, 不再需要像旧版那样自己找 V1 engine hook。旧版失败点是 vllm_musa 当时没有可用 V1 引擎; 现在 `vllm_musa 0.1.1 + vllm 0.20.1` 是纯 V1, 正好满足 LMCache 当前主要接入方式。

当前已经完成的部分:

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| vLLM/vllm_musa import | 通过 | `vllm 0.20.1.dev0`, `vllm_musa 0.1.1`, `torch/torch_musa 2.9` |
| LMCache Python 包和 `.so` | 通过 | `lmcache`, `c_ops.so`, `native_storage_ops.so`, `lmcache_redis.so` 可加载 |
| LMCache 基础/管理类测试 | 通过 | utils/protocol/config/memory management/manager/connector 等通过 |
| LMCache MUSA mem kernels | 通过 | 165 因资源占用失败, 146 空闲卡复测 `37 passed` |
| LMCache cache engine 本地路径 | 基本通过 | CPU/local_disk/paged KV 主路径通过; remote 失败是因为没起 remote server |
| 146 纯 vLLM baseline | 通过 | 19000 `qwen3-8b-pure-fixed` 可启动并完成 chat |
| 146 LMCache 端到端 | 通过 | 19001 `qwen3-8b-lmcache` 可启动并完成 chat |
| LMCache 性能收益验证 | 通过 | 详见 `../ISSUE_vllm_musa_broadcast_deadlock2.md` 的 cold/warm 分离测试 |

仍未覆盖或不应过度外推的部分:

1. 当前端到端主验证是 **单卡 Qwen3-8B**; TP=4/TP=8、多机、多 remote backend 还需要单独验证;
2. LMCache remote backend 的测试失败不是功能结论, 因为当时没有启动 `lm://localhost:18078` remote server;
3. `64GB/80GB` LMCache CPU cache 在 146 上会触发 MUSA host allocation 失败, 当前稳定配置按 `40GB` 设计;
4. 后续如迁移到新镜像/新机器, 必须优先对齐 `torchada`、`mate` 等 MUSA 运行时包, 不能靠手写 `sitecustomize.py` 补 API。

一句话总结:

> LMCache 接入新版 `vllm_musa` 的核心难点不在 V1 hook, 而在 **运行时环境对齐** 和 **分阶段验证**。先把纯净 `vllm_musa` baseline 跑通, 再挂 LMCache; 不要一上来把 vLLM 基础环境问题误判成 LMCache 问题。

### 1.1 如何阅读这份文档

如果只是想接手继续跑实验, 建议按下面顺序看:

1. 先看 **第 0 节**: 搞清楚 `torchada`、`mate`、`vllm_musa`、`LMCacheConnectorV1` 分别在调用链哪一层;
2. 再看 **第 13 节**: 理解之前为什么会误判成 API 不兼容, 以及最终为什么是运行时包版本问题;
3. 然后看 **第 14 节**: 按 runbook 从资源检查、容器迁移、import、pure baseline、LMCache 启动逐步复现;
4. 遇到错误先查 **第 15 节故障速查表**;
5. 性能 benchmark 和 LMCache external hit 的解释看 `../ISSUE_vllm_musa_broadcast_deadlock2.md`。

不要只复制最后的启动命令。这个迁移过程最容易出错的地方是“命令看起来一样, 但 venv 里的 MUSA 运行时包不一样”。

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

### 3.1 读路径 vs 写路径: 一个最容易搞混的点

上面 `save_kv_layer`(写) 和 `get_num_new_matched_tokens`(读) 是两条**方向相反、规则不同**的路径。初次接触最常见的误解是"是不是 APC 装不下了才写 LMCache?"——**不是**。把这两条分清, 后面读 benchmark 才不会误判。

**写路径(store, cold 注入时) = 主动写穿, 不看 APC 是否已满**

- prefill 在 GPU 上算完 KV 后, worker 侧的 `save_kv_layer` 会把该层 KV 按 chunk(默认 256 token)**主动 offload 一份到 LMCache(CPU)**, 和 GPU APC 还有没有空间无关。
- 它**不是** victim / 溢出缓存(不是"等 APC 淘汰了, 再把被踢出来的塞给 LMCache")。
- 实测佐证: cold/warm 测试里, cold 阶段 GPU 只装得下约 64 个前缀、约 16 个被淘汰, 但 warm 阶段 `external_prefix_cache_hits_total = 245,760 = 80 × 3072`, **全部 80 个前缀都在 LMCache 里**。如果是"溢出才写", 那 ~64 个没被淘汰的前缀根本不会进 LMCache, warm 命中就到不了满值 → 所以一定是 cold 时把每个前缀都主动写穿进了 LMCache。
- 当前配置 `save_decode_cache=False`、`save_unfull_chunk=False`: 只存 prefill 阶段的整 chunk KV, decode 增量和不满一个 chunk 的尾巴不存。

**读路径(lookup, warm 命中时) = APC 优先, LMCache 兜底**

- scheduler 侧先算 GPU APC 本地命中了多少 token, 再调 `get_num_new_matched_tokens` 问 LMCache "在 APC 之外你还能**额外**补多少"(注意是 new)。
- 所以 APC 覆盖到的部分不会再走 LMCache(APC 在显存里、零搬运, 更快); 只有 APC 没命中的部分才由 LMCache 从 CPU 取回。
- 推论: 工作集 < GPU 容量时 APC 全包, external hit ≈ 0(这是预期, 不是 LMCache 坏了); 只有工作集超过 GPU 容量、APC 发生淘汰后, LMCache 才会现身。这也是 Step2 必须把工作集做到超容量的原因。

**一句话**: 写是"GPU 算完主动 offload, APC 和 LMCache 两层都拿到一份"; 读是"先查 APC, APC 不够才用 LMCache 兜底"。

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

### 4.5 运行时包分工: `torchada` 和 `mate` 不是可有可无的依赖

这次迁移里最容易误判的一点是: 看到 `torch.accelerator.*` 或 FlashAttention 报错时, 会下意识以为是 vLLM API 不兼容, 然后去给 Python 层打补丁。后续对比 127 可运行环境证明, 真正问题是 146 初始镜像里的 MUSA 运行时包版本不匹配。

需要分清两个包的角色:

| 包 | 作用 | 本次相关错误 |
| --- | --- | --- |
| `torchada` | MUSA 对 PyTorch accelerator 抽象的适配层。它会把一些 `torch.accelerator.*` 接口 patch 到 `torch_musa` 的实现上 | 旧版 `torchada 0.1.48` 没把 `torch.accelerator.empty_cache` patch 到 `torch_musa.core.memory`, 导致 V1 worker 初始化失败 |
| `mate` | MUSA attention/kernel 运行库, vLLM MUSA V1 attention backend 会调用这里的 FlashAttention 接口 | 旧版 `mate 0.1.3+mu436torch2.9` 的 `flash_attn_varlen_func` 不支持 `page_table` 参数, dummy/profile forward 失败 |

为什么不能靠 `sitecustomize.py` 之类的手工补丁解决:

1. `torchada` 不是只补一个函数名, 它还承担 MUSA backend 的一组注册和运行时 patch;
2. `mate` 的问题是底层 attention 函数签名不匹配, Python 层随便吞掉参数会导致后续 kernel 行为不可控;
3. 127 上同一代新版 `vllm_musa` 已经能正常跑, 说明正确方向是对齐运行时包, 不是继续堆兼容补丁。

因此后续迁移必须把 `torchada` / `mate` 视为 `vllm_musa` 运行时的一部分, 和 `torch` / `torch_musa` 一起核对。只看 `torch==2.9.0`、`torch_musa==2.9.0` 不够。

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

早期判断是“优先尝试 `use_native=true`”, 但后续 import 实测修正为:

1. 当前 `/data/_backup_to_local/LMCache` 是 LMCache 0.3.0 MUSA port;
2. vLLM 仓内 native adapter 需要更新的 LMCache Python API, 例如 `LMCacheEngineMetadata`;
3. 因此 `use_native=true` 路径会 import 失败;
4. 当前已经跑通的端到端路径是默认 `use_native=false`, 即使用 LMCache 包自带的 `lmcache.integration.vllm.vllm_v1_adapter`。

所以交接时按当前实测结论执行: **默认不传 `use_native=true`**。只有在升级 LMCache Python API 或补齐 native adapter 兼容后, 再重新评估 `use_native=true`。

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
3. 默认外部 adapter 路径是否可 import。

备注: `use_native=true` 在当前 LMCache 0.3.0 MUSA port 下已知会因为 API 不匹配失败, 不作为主路径阻塞项。

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
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
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
4. 用 `LMCacheConnectorV1` 默认外部 adapter 先跑单卡端到端;
5. 再扩到 TP、long-context、prefix-cache 命中场景。

该判断是 2026-06-11 的初始可行性判断。后续 2026-06-16/17 已经完成 146 单卡 Qwen3-8B pure baseline、LMCache 启动、LMCache external hit 和 cold/warm 性能收益验证; 当前最终状态见本文第 16 节。

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

---

## 12. 2026-06-16 迁移到 10.10.142.146 后的测试

### 12.1 146 资源状态

服务器: `10.10.142.146`, hostname `worker146`

资源检查:

| 项 | 状态 |
| --- | --- |
| GPU | 8 x MTT S5000, 初始 `0MiB/81920MiB`, 无运行进程 |
| 内存 | 2.2TiB total, 约 1.7TiB free |
| `/data` | 独立数据盘, `/dev/mapper/data--vg-data--lv`, ext4, 13T, 可用约 7.1T |
| Docker root | `/var/lib/docker`, 独立 LV, 可用约 933G |

### 12.2 Docker 迁移方式

按“打包本机 docker 再传到 146”的方式执行:

```bash
docker commit vllm020_test vllm020_lmcache_committed:20260616
docker save vllm020_lmcache_committed:20260616 | zstd -T0 -1 | \
  ssh root@10.10.142.146 'zstd -d | docker load'
```

迁移镜像:

```text
vllm020_lmcache_committed:20260616
```

146 上创建的测试容器:

```text
vllm020_lmcache_test
```

容器启动命令等价于本机测试容器, 挂载 `/data`:

```bash
docker run -d --name vllm020_lmcache_test \
  --runtime=mthreads --privileged --network host --pid host \
  --shm-size 500g --ulimit memlock=-1:-1 \
  --security-opt label=disable \
  -v /data:/data -v /data:/home/dist -v /data/models:/data/models \
  -w /data/my_vllm_test \
  vllm020_lmcache_committed:20260616 sleep infinity
```

注意: 当前本机容器里的 vLLM/vllm_musa 是 symlink 到 `/data/my_vllm_test/vllm_020`, LMCache 也是通过 `PYTHONPATH=/data/_backup_to_local/LMCache` 使用, 所以除了 `docker commit/save/load`, 还需要同步 `/data/my_vllm_test/vllm_020` 和 `/data/_backup_to_local/LMCache`。这两部分已经传到 146。

### 12.3 146 import 验证

146 容器内 import 通过:

| 模块 | 结果 |
| --- | --- |
| `vllm` | OK, `0.20.1.dev0+g88d34c640.d20260519` |
| `vllm_musa` | OK, `0.1.1` |
| `torch` | OK, `2.9.0` |
| `torch_musa` | OK, `2.9.0+0680da1` |
| `openai` | OK, `2.37.0` |
| `flash_attn_3` | OK, `0.1.4` |
| `lmcache` | OK |
| `lmcache.c_ops` | OK |
| `lmcache.native_storage_ops` | OK |
| `lmcache.lmcache_redis` | OK |
| `lmcache.integration.vllm.vllm_v1_adapter` | OK |
| `vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector` | OK |

### 12.4 mem kernels 在空闲卡上复测

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
37 passed, 1 warning in 223.64s (0:03:43)
```

结论: 165 上 `test_mem_kernels.py` 的 18 个失败确认为资源占用/OOM 导致, 不是 MUSA kernel 数值问题。146 空闲 S5000 上完整通过。

### 12.5 cache engine 测试

日志文件:

```text
/data/my_vllm_test/vllm_020/test_cache_engine_146_20260616.log
```

命令:

```bash
cd /data/_backup_to_local/LMCache
MUSA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH \
/root/.virtualenvs/sglang-0.5.6/bin/python -m pytest -q --tb=short \
  tests/v1/test_cache_engine.py
```

结果:

```text
18 failed, 36 passed, 2 skipped, 8 warnings in 945.82s (0:15:45)
```

失败项全部是 `remote` / `remote_cachegen` / 包含 `remote` 的组合:

```text
test_paged_retrieve_prefix[..., remote, ...]
test_paged_retrieve_prefix[..., remote_cachegen, ...]
test_paged_store_offset[..., remote, ...]
test_paged_hierarchy_retrieve[..., remote, ...]
test_paged_mem_leak[..., remote, ...]
```

典型失败原因:

```text
Failed to initialize/re-establish remote connection: [Errno 111] Connection refused
Connection is None in batched_contains, returning 0
TimeoutError: Operation timed out after 30 seconds.
```

解释: 测试配置使用 `remote_url='lm://localhost:18078'`, 但当前没有启动 LMCache remote server, 所以 remote 后端连接失败并超时。这不是本地 CPU/local_disk/paged KV 主路径失败。

已通过的关键本地路径包括:

1. `test_paged_same_retrieve_store`;
2. `test_paged_retrieve_prefix` 的 `cpu` / `local_disk`;
3. `test_paged_store_offset` 的 `cpu` / `local_disk`;
4. `test_paged_hierarchy_retrieve` 的 `local_cpu` / `local_disk`;
5. `test_paged_mem_leak` 的 `cpu` / `local_disk`;
6. `test_paged_mixed_retrieve`;
7. `test_paged_store_kv_tensors_mask`;
8. `test_paged_prefetch_retrieve`;
9. `test_paged_retrieve_after_eviction`;
10. `test_builder`, `test_force_store_wait`, `test_builder_destroy`, `test_builder_destroy_multiple_instances`.

### 12.6 当前判断

在 146 空闲卡上, LMCache MUSA port 的核心 kernel 和本地 cache engine 路径已经比 165 上更充分地验证:

1. `.so` ABI/import 通过;
2. memory management 通过;
3. mem kernels 全量通过;
4. cache engine 本地 CPU/local_disk/paged KV 路径通过;
5. remote 路径未通过, 当前原因是没有启动 remote server, 不是 MUSA kernel 失败。

下一步如果继续验证 remote backend, 需要先按 LMCache 测试 fixture/脚本启动 `lm://localhost:18078` 对应的 server; 如果目标是 vLLM+LMCache 本地 CPU/disk KV offload, 可以先不阻塞在 remote backend 上, 进入 vLLM 端到端小模型验证。

---

## 13. 2026-06-16 迁移后 vLLM MUSA 纯净 baseline 问题排查与修复

本节从“146 上移植后的 vLLM MUSA 环境先跑纯净小模型 baseline”开始记录。目标是先确认新版 `vllm_musa` 自身可正常启动 V1 engine, 再继续接 LMCache, 避免把基础环境问题误判为 LMCache 移植问题。

这一节是整个迁移过程里最重要的排障记录。后来证明, 当时大量启动失败 **不是 LMCache 引起的**, 而是 146 初始容器和 127 正常运行环境存在 `torchada` / `mate` 等运行时包差异。排障方法也因此发生了变化:

```text
错误做法: 看到 torch/vLLM API 报错 -> 手写 sitecustomize.py 补 API -> 继续遇到下一个底层错误
正确做法: 127 上同类新版 vllm_musa 能跑 -> 只读对比 127 与 146 -> 同步运行时包差异 -> 先跑纯净 baseline
```

给后续接手者的判断原则:

1. 只要纯净 `vllm_musa` baseline 没跑通, 就不要先怀疑 LMCache;
2. 不要关闭 V1 engine, 当前新版 vLLM 物理上就是 V1 路径;
3. 不要继续恢复 `/data/my_vllm_test/vllm_020/sitecustomize.py`, 这个方向已经被证伪;
4. 遇到 accelerator/attention 相关错误, 先核对 `torchada`、`mate`、`flash_attn_3`、`vllm_musa` 是否和可运行环境一致。

### 13.1 初始现象

在 146 容器 `vllm020_lmcache_test` 中, 使用纯净新版 vLLM MUSA 启动小模型:

```bash
unset PYTHONPATH
CUDA_VISIBLE_DEVICES=0 MUSA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_DISABLE_COMPILE_CACHE=1 \
VLLM_USE_DEEP_GEMM_E8M0=0 \
vllm serve /data/SQT-v1.0.5-test/models/qwen3-8b \
  --trust-remote-code \
  --gpu-memory-utilization 0.65 \
  --served-model-name qwen3-8b-pure-fixed \
  --block-size 64 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --port 19000 \
  --max-model-len 4096
```

最早失败点:

```text
RuntimeError: device_allocator INTERNAL ASSERT FAILED ...
Allocator for musa is not a DeviceAllocator.
```

具体栈在 vLLM V1 worker 初始化:

```text
vllm/v1/worker/gpu_worker.py:279
torch.accelerator.empty_cache()
```

当时曾临时写过 `/data/my_vllm_test/vllm_020/sitecustomize.py` 做 `torch.accelerator.*` 兼容兜底, 但该方向后来确认不合适: 127 上新版 `vllm_musa` 能正常跑 V1 推理, 不应该靠外部 `sitecustomize.py` 人工补一串 accelerator API。该文件已改名禁用:

```text
/data/my_vllm_test/vllm_020/sitecustomize.py.disabled
```

### 13.2 排查原则修正

用户指出 127 上同类新版 `vllm_musa` 测试程序正在正常运行, 因此排查方向改为:

1. 不关闭 V1 engine;
2. 不继续堆外部 API 兼容补丁;
3. 用 127 正在运行的环境作为基准, 只读对比配置和包版本;
4. 146 只修正环境差异, 先跑纯净 `vllm_musa` baseline, 再接 LMCache。

127 只读观察到的运行进程:

```text
auto_bench.py --param-file tp_pd_params_32b_bf16.json --models qwen3-32b-bf16 --pd --tp 4
vllm serve /mnt/seed17/001688/models/Qwen3-32B ... --port 8100 ... kv_producer
vllm serve /mnt/seed17/001688/models/Qwen3-32B ... --port 8200 ... kv_consumer
toy_proxy_server.py ... --port 8000
bench_serving.py ... --port 8000 ...
```

127 运行进程关键环境变量:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 / 4,5,6,7
MUSA_VISIBLE_DEVICES=0,1,2,3 / 4,5,6,7
VLLM_USE_V1=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_DISABLE_COMPILE_CACHE=1
VLLM_USE_DEEP_GEMM_E8M0=0
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:/usr/local/mtshmem/lib/:/usr/local/musa/lib:/usr/local/openmpi/lib:/usr/local/musa/mudnn/lib:
```

说明: 虽然脚本里设置 `VLLM_USE_V1=0`, 当前新版 vLLM 日志仍显示 `Initializing a V1 LLM engine`, 因此不能把关闭 V1 作为解决方案。

### 13.3 第一处根因: `torchada` 版本不同

先解释这个错误为什么和 `torchada` 有关。

vLLM V1 worker 初始化时会释放/整理设备缓存, 调用的是 PyTorch 新的 accelerator 抽象:

```python
torch.accelerator.empty_cache()
```

在 CUDA 机器上, 这个接口最终会落到 CUDA memory allocator; 在 MUSA 机器上, 它必须被 MUSA 适配层改到 `torch_musa` 的 memory 实现。这个改写动作不是 vLLM 自己做的, 而是由 MUSA 生态里的 `torchada` / `vllm_musa` import 过程完成。

正确情况下, import `vllm_musa` 后:

```text
torch.accelerator.empty_cache.__module__ == torch_musa.core.memory
```

错误情况下, 它仍然指向 PyTorch 原生的:

```text
torch.accelerator.memory
```

这时 PyTorch accelerator 层知道当前 accelerator 是 `musa`, 但它手里的 allocator 不是 MUSA DeviceAllocator, 所以报:

```text
Allocator for musa is not a DeviceAllocator
```

这就是为什么这个错误看起来像 `torch.accelerator` API 问题, 实际根因是 `torchada` 版本/加载位置不对。

最小复现:

```python
import torch, torch_musa
torch.accelerator.current_accelerator()  # musa
torch.accelerator.empty_cache()          # 146 初始失败
torch.musa.empty_cache()                 # OK
```

对比 127 与 146:

| 项 | 127 可运行环境 | 146 初始迁移环境 |
| --- | --- | --- |
| `torchada` | `0.1.56`, venv 内 | `0.1.48`, `/usr/local` |
| `import vllm_musa` 后 `torch.accelerator.empty_cache.__module__` | `torch_musa.core.memory` | `torch.accelerator.memory` |
| `torch.accelerator.empty_cache()` | OK | `Allocator for musa is not a DeviceAllocator` |

关键结论:

1. 两边 `torch` / `torch_musa` 版本号相同, 但 `torchada` 不同;
2. 127 的 `torchada 0.1.56` 会在 `import vllm_musa` 后把 `torch.accelerator.empty_cache` 正确 patch 到 `torch_musa.core.memory`;
3. 146 初始环境的 `torchada 0.1.48` 没有完成这个 patch, 所以 vLLM V1 一进 `gpu_worker.py:279` 就失败。

修复动作:

```bash
# 从 127 只读复制到 146 容器 venv
torchada
torchada-0.1.56.dist-info
```

复制后 146 最小验证:

```text
Name: torchada
Version: 0.1.56
Location: /root/.virtualenvs/sglang-0.5.6/lib/python3.10/site-packages

before torch.accelerator.memory
torchada 0.1.56 ... True
vllm_musa .../site-packages/vllm_musa/__init__.py
after torch_musa.core.memory
accelerator.empty_cache OK
```

交接结论:

- 看到 `Allocator for musa is not a DeviceAllocator` 时, 第一反应不是改 vLLM 或 LMCache;
- 先检查 `torchada` 版本和 `torch.accelerator.empty_cache.__module__`;
- 只要它没指到 `torch_musa.core.memory`, 纯 vLLM baseline 就不可信。

### 13.4 第二处根因: `mate` / flash attention 运行库不同

`torchada` 修完后, vLLM 已经能走到模型加载和 profile/dummy forward, 说明设备初始化这层过去了。新的失败发生在 attention kernel 调用层。

vLLM MUSA 的 V1 attention 大致调用链是:

```text
vllm_musa/v1/attention/backends/flash_attn.py
  -> flash_attn_3/interface.py
    -> mate.mha_interface.flash_attn_varlen_func(...)
```

`page_table` 是 paged attention / paged KV cache 路径里的关键参数。vLLM V1 传入 `page_table`, 是为了告诉 attention kernel 当前 token 对应哪些 KV cache block。新版 `vllm_musa` 期待底层 `mate` 的 `flash_attn_varlen_func` 支持这个参数。

146 初始环境的 `mate 0.1.3+mu436torch2.9` 不支持 `page_table`, 所以 Python 在调用函数时直接报:

```text
unexpected keyword argument 'page_table'
```

这不是 LMCache 的 KV layout 问题, 也不是 benchmark 问题, 而是 attention 运行库接口版本太旧。127 能跑, 正是因为 127 的 `mate 0.2.0+mu437torch2.9` 支持这个参数。

修复 `torchada` 后, 纯净 qwen3-8b baseline 已经能够:

1. 进入 V1 engine 初始化;
2. 加载模型权重;
3. 进入 dummy/profile forward。

随后出现新错误:

```text
TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'page_table'
```

栈位置:

```text
vllm_musa/v1/attention/backends/flash_attn.py
flash_attn_3/interface.py
mate.mha_interface.flash_attn_varlen_func(...)
```

对比 127 与 146:

| 项 | 127 可运行环境 | 146 初始迁移环境 |
| --- | --- | --- |
| `mate` | `0.2.0+mu437torch2.9`, venv 内 | `0.1.3+mu436torch2.9`, `/usr/local` |
| `mate.mha_interface.flash_attn_varlen_func` | 支持 `page_table` 参数 | 不支持 `page_table` 参数 |
| 结果 | 可供当前 `vllm_musa` V1 attention 调用 | dummy forward 时报 TypeError |

127 签名中包含:

```text
page_table: Optional[torch.Tensor] = None
```

146 初始签名没有该参数。

修复动作:

```bash
# 从 127 只读复制到 146 容器 venv
mate
mate-0.2.0+mu437torch2.9.dist-info
mate_flash_attention-0.1.3.dist-info
mate_deep_gemm-0.1.2.dist-info
flash_attn
flash_attn-2.6.3.dist-info
flash_attn_3
flash_attn_3-0.1.4.dist-info
flash_attn_interface.py
```

复制后 146 验证:

```text
mate 0.2.0+mu437torch2.9 /root/.virtualenvs/sglang-0.5.6/lib/python3.10/site-packages/mate/__init__.py
has_page_table True
flash_attn_3 /root/.virtualenvs/sglang-0.5.6/lib/python3.10/site-packages/flash_attn_3/interface.py
```

交接结论:

- 看到 `page_table` 参数错误, 不要去删参数或包一层 wrapper;
- 这是 `vllm_musa` 和 `mate`/`flash_attn_3` 的接口版本不匹配;
- 应该同步可运行环境里的 `mate`、`flash_attn_3` 等配套包, 然后重新跑 pure baseline。

### 13.5 修复后的纯净 vLLM MUSA baseline 结果

日志:

```text
/data/my_vllm_test/vllm_020/qwen3_8b_pure_fixed_19000.log
```

启动后关键日志:

```text
Platform plugin musa is activated
MUSA patches and custom ops registered
Initializing a V1 LLM engine ...
Starting to load model /data/SQT-v1.0.5-test/models/qwen3-8b...
Using FlashAttention version 3
Loading weights took 6.46 seconds
Model loading took 15.27 GiB memory and 11.390663 seconds
torch.compile and initial profiling/warmup run together took 35.51 s in total
Available KV cache memory: 29.76 GiB
GPU KV cache size: 216,704 tokens
```

`/v1/models` 验证通过:

```text
READY
model: qwen3-8b-pure-fixed
root: /data/SQT-v1.0.5-test/models/qwen3-8b
max_model_len: 4096
```

轻量 chat completion 验证通过:

```bash
curl http://127.0.0.1:19000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-pure-fixed","messages":[{"role":"user","content":"请用一句话说明你是谁。"}],"max_tokens":32,"temperature":0}'
```

返回了正常 `chat.completion`, `usage` 中:

```text
prompt_tokens=15
completion_tokens=32
total_tokens=47
```

### 13.6 当前结论

146 迁移后的 vLLM MUSA baseline 初始失败不是 LMCache 引入的问题, 而是迁移环境与 127 可运行环境有运行时包差异:

1. `torchada 0.1.48` 未将 `torch.accelerator.empty_cache` patch 到 `torch_musa.core.memory`, 导致 V1 worker 初始化失败;
2. `mate 0.1.3+mu436torch2.9` 的 flash attention varlen 接口不支持 `page_table`, 导致模型 dummy/profile forward 失败;
3. 同步 127 的 `torchada 0.1.56` 与 `mate 0.2.0+mu437torch2.9` 后, 146 纯净新版 `vllm_musa` 小模型 baseline 已经可以启动并完成一次 chat 推理。

后续接 LMCache 时应基于这个已经修正的 146 venv 环境继续, 不应再启用之前的 `sitecustomize.py` 兼容补丁。

### 13.7 2026-06-16 LMCache 参数化启动验证

为便于后续 benchmark 对比, 新增统一启动脚本:

```text
/data/my_vllm_test/vllm_020/start_qwen3_8b_vllm_musa.sh
```

关键参数:

```bash
ENABLE_LMCACHE=0|1
GPU_IDS=0
PORT=19000
SERVED_MODEL_NAME=qwen3-8b-pure-fixed
MODEL_PATH=/data/SQT-v1.0.5-test/models/qwen3-8b
LMCACHE_PATH=/data/_backup_to_local/LMCache
LMCACHE_LOCAL_CPU=True
LMCACHE_MAX_LOCAL_CPU_SIZE=20
LMCACHE_CHUNK_SIZE=256
```

`ENABLE_LMCACHE=0` 时不设置 `PYTHONPATH` 到 LMCache, 也不传 `--kv-transfer-config`;
`ENABLE_LMCACHE=1` 时追加:

```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

并设置:

```bash
PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH
LMCACHE_LOCAL_CPU=True
LMCACHE_MAX_LOCAL_CPU_SIZE=20
LMCACHE_CHUNK_SIZE=256
LMCACHE_USE_EXPERIMENTAL=True
```

测试命令:

```bash
cd /data/my_vllm_test/vllm_020
ENABLE_LMCACHE=1 \
GPU_IDS=1 \
PORT=19001 \
SERVED_MODEL_NAME=qwen3-8b-lmcache \
LOG_FILE=/data/my_vllm_test/vllm_020/qwen3_8b_lmcache_19001.log \
./start_qwen3_8b_vllm_musa.sh
```

验证结果:

```text
/v1/models ready
model: qwen3-8b-lmcache
root: /data/SQT-v1.0.5-test/models/qwen3-8b
max_model_len: 4096
```

轻量 chat completion 验证通过:

```bash
curl http://127.0.0.1:19001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-lmcache","messages":[{"role":"user","content":"请用一句话说明你是谁。"}],"max_tokens":32,"temperature":0}'
```

返回了正常 `chat.completion`, `usage` 中:

```text
prompt_tokens=15
completion_tokens=32
total_tokens=47
```

日志证据:

```text
Creating v1 connector with name: LMCacheConnectorV1
Initializing latest dev LMCache connector
Creating LMCacheEngine with config: {'chunk_size': 256, 'local_cpu': True, 'max_local_cpu_size': 20.0, ...}
Created backend: LocalCPUBackend (LocalCPUBackend)
LMCache initialized for role KVConnectorRole.WORKER
LMCache initialized for role KVConnectorRole.SCHEDULER
Reqid: ..., Total tokens 15, Inference Engine computed tokens: 0, LMCache hit tokens: 0, need to load: 0
```

当前结论: 在 146 的修正 venv 环境中, 新版 `vllm_musa` 挂载 LMCache 后可以正常启动到 OpenAI API ready 状态, 并能完成一次小请求。后续可以用同一个脚本通过 `ENABLE_LMCACHE=0/1` 切换纯 vLLM 与 LMCache 路径做 benchmark 对比。

---

## 14. 从零复现迁移的 runbook

这一节按“以后换一台服务器也能照做”的目标写。核心思想是: **先迁移环境, 再验证纯 vLLM, 最后接 LMCache**。不要把三件事混在一起排查。

### 14.1 迁移对象: 到底要搬哪些东西

本次迁移不是单纯 `docker save/load` 就结束, 因为容器里有若干目录通过 `/data` 挂载或 symlink 到宿主机。需要同时迁移:

| 对象 | 路径 / 名称 | 用途 |
| --- | --- | --- |
| Docker 镜像 | `vllm020_lmcache_committed:20260616` | 基础 Python/torch/vLLM 运行环境 |
| 测试容器 | `vllm020_lmcache_test` | 146 上实际运行的容器 |
| 新版 vLLM/vllm_musa 代码 | `/data/my_vllm_test/vllm_020` | vLLM 0.20 / vllm_musa 0.1.1 及脚本、日志 |
| LMCache MUSA port | `/data/_backup_to_local/LMCache` | 已 MUSA 化的 LMCache 0.3.0 源码和 `.so` |
| 模型 | `/data/SQT-v1.0.5-test/models/qwen3-8b` | Qwen3-8B baseline/LMCache 端到端验证 |
| benchmark 原始脚本 | `/data/my_vllm_test/vllm_musa_proj` | 原始 auto_bench / bench_serving 参考 |
| LMCache 结果文档 | `/data/my_vllm_test/ISSUE_vllm_musa_broadcast_deadlock2.md` | LMCache benchmark 设计和结果 |

如果只搬镜像、不搬 `/data/my_vllm_test/vllm_020` 和 `/data/_backup_to_local/LMCache`, 容器内会出现“包能 import 一部分, 但脚本/源码/LMCache 路径不存在”的半迁移状态。

### 14.2 目标机器资源检查

在目标机器上先确认资源, 这一步用于决定能不能直接跑模型, 以及 `/data` 是否是独立数据盘。

```bash
hostname
df -h /data
docker info | grep -E 'Docker Root Dir|Runtimes'
mthreads-gmi
free -h
```

本次 146 的有效状态:

| 项 | 结果 |
| --- | --- |
| 服务器 | `10.10.142.146`, hostname `worker146` |
| GPU | 8 x MTT S5000, 初始空闲 |
| 内存 | 2.2TiB total |
| `/data` | 独立 ext4 数据盘, 约 13T |
| Docker root | `/var/lib/docker`, 独立 LV |

判断规则:

1. GPU 如果已有大模型进程, 先不要跑 LMCache kernel/full model 测试;
2. `/data` 如果不是独立大盘, 不要盲目同步模型和镜像;
3. Docker 需要 `mthreads` runtime, 否则容器看不到 MUSA 设备。

### 14.3 打包并传输 Docker 镜像

源机器上:

```bash
docker commit vllm020_test vllm020_lmcache_committed:20260616
docker save vllm020_lmcache_committed:20260616 | zstd -T0 -1 | \
  ssh root@10.10.142.146 'zstd -d | docker load'
```

目标机器上确认:

```bash
docker images | grep vllm020_lmcache_committed
```

注意:

- `docker commit` 只保存容器文件系统, 不保存宿主 `/data` 的真实内容;
- 因此镜像传完后仍然要同步 `/data/my_vllm_test/vllm_020` 和 `/data/_backup_to_local/LMCache`;
- 如果源容器里某些包实际来自 venv site-packages, 需要后续 import 验证确认。

### 14.4 创建目标容器

146 上实际容器:

```bash
docker run -d --name vllm020_lmcache_test \
  --runtime=mthreads --privileged --network host --pid host \
  --shm-size 500g --ulimit memlock=-1:-1 \
  --security-opt label=disable \
  -v /data:/data -v /data:/home/dist -v /data/models:/data/models \
  -w /data/my_vllm_test \
  vllm020_lmcache_committed:20260616 sleep infinity
```

这些参数的含义:

| 参数 | 为什么需要 |
| --- | --- |
| `--runtime=mthreads` | 让容器使用 MUSA 设备运行时 |
| `--privileged` | 避免设备/驱动访问权限问题 |
| `--network host` | vLLM 端口直接暴露在宿主网络, 也便于容器内打 `127.0.0.1` |
| `--pid host` | 便于排查残留进程和 GPU 占用 |
| `--shm-size 500g` | 大模型和多进程运行时需要较大 shared memory |
| `--ulimit memlock=-1:-1` | LMCache / MUSA pinned memory 场景避免 memlock 限制 |
| `-v /data:/data` | 复用宿主数据盘里的源码、模型、结果 |

### 14.5 同步源码和 LMCache 目录

需要保证目标机器存在:

```text
/data/my_vllm_test/vllm_020
/data/_backup_to_local/LMCache
```

同步完成后进容器检查:

```bash
docker exec -it vllm020_lmcache_test bash
ls /data/my_vllm_test/vllm_020
ls /data/_backup_to_local/LMCache
```

关键文件:

| 文件 | 用途 |
| --- | --- |
| `/data/my_vllm_test/vllm_020/start_qwen3_8b_vllm_musa.sh` | 统一启动 pure/LMCache 服务 |
| `/data/my_vllm_test/vllm_020/bench_serving.py` | 146 容器内实际使用的 benchmark_serving |
| `/data/my_vllm_test/vllm_020/lmcache_cold_warm_revisit_bench.py` | 后续新增的冷/热分离 LMCache 性能验证脚本 |
| `/data/_backup_to_local/LMCache/lmcache/*.so` | LMCache MUSA native 扩展 |

### 14.6 第一层验证: import 和 `.so`

进入容器后先只做 import, 不启动模型:

```bash
export PY=/root/.virtualenvs/sglang-0.5.6/bin/python3
export PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH

$PY - <<'PY'
import torch, torch_musa
import vllm, vllm_musa
import lmcache
import lmcache.c_ops
import lmcache.native_storage_ops
import lmcache.lmcache_redis
import lmcache.v1.gpu_connector
import lmcache.integration.vllm.vllm_v1_adapter
import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector
print("import ok")
PY
```

通过标准:

1. `vllm_musa` import 后日志出现 `Platform plugin musa is activated`;
2. LMCache 三个 `.so` 都能 import;
3. 默认 LMCache 自带 adapter `lmcache.integration.vllm.vllm_v1_adapter` 能 import;
4. vLLM 仓内 native adapter 可能因为 LMCache 0.3.0 API 旧而失败, 这不是当前主路径阻塞。

### 14.7 第二层验证: LMCache 自身测试

最小建议顺序:

```bash
cd /data/_backup_to_local/LMCache
export PY=/root/.virtualenvs/sglang-0.5.6/bin/python3
export PYTHONPATH=/data/_backup_to_local/LMCache:$PYTHONPATH

MUSA_VISIBLE_DEVICES=0 $PY -m pytest -q \
  tests/test_utils.py \
  tests/test_protocol.py \
  tests/test_serde.py \
  tests/v1/test_basic_check.py \
  tests/v1/test_config.py

MUSA_VISIBLE_DEVICES=0 $PY -m pytest -q tests/v1/test_memory_management.py
MUSA_VISIBLE_DEVICES=0 $PY -m pytest -q tests/v1/test_mem_kernels.py
```

146 上已验证:

| 测试 | 结果 | 解释 |
| --- | --- | --- |
| 基础 utils/protocol/serde/basic/config | `163 passed, 1 xpassed` | Python 层基础逻辑通过 |
| memory management | `47 passed` | allocator / memory object 管理通过 |
| mem kernels | `37 passed` | 空闲 S5000 上 MUSA kernel 路径通过 |

如果 `test_mem_kernels.py` 在资源繁忙机器上失败, 先看是不是 OOM:

```text
torch.OutOfMemoryError: MUSA out of memory
musaHostAlloc failed
```

这类失败优先检查 `mthreads-gmi`, 不要直接判定 kernel 错。

### 14.8 第三层验证: 纯净 vLLM MUSA baseline

先启动不挂 LMCache 的 19000:

```bash
cd /data/my_vllm_test/vllm_020
ENABLE_LMCACHE=0 \
GPU_IDS=0 \
PORT=19000 \
SERVED_MODEL_NAME=qwen3-8b-pure-fixed \
LOG_FILE=/data/my_vllm_test/vllm_020/qwen3_8b_pure_fixed_19000.log \
./start_qwen3_8b_vllm_musa.sh
```

验证:

```bash
curl -s http://127.0.0.1:19000/v1/models
curl http://127.0.0.1:19000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-pure-fixed","messages":[{"role":"user","content":"请用一句话说明你是谁。"}],"max_tokens":32,"temperature":0}'
```

必须先看到:

```text
GPU KV cache size: 216,704 tokens
/v1/models ready
chat.completion 正常返回
```

如果 pure baseline 还没过, 不要继续挂 LMCache。

### 14.9 第四层验证: 挂 LMCache 启动

启动 19001:

```bash
cd /data/my_vllm_test/vllm_020
ENABLE_LMCACHE=1 \
GPU_IDS=1 \
PORT=19001 \
SERVED_MODEL_NAME=qwen3-8b-lmcache \
LMCACHE_MAX_LOCAL_CPU_SIZE=40 \
LOG_FILE=/data/my_vllm_test/vllm_020/qwen3_8b_lmcache_19001.log \
./start_qwen3_8b_vllm_musa.sh
```

日志里要看到:

```text
Creating v1 connector with name: LMCacheConnectorV1
Creating LMCacheEngine with config: {... 'local_cpu': True, 'max_local_cpu_size': 40.0, ...}
Created backend: LocalCPUBackend
LMCache initialized for role KVConnectorRole.WORKER
LMCache initialized for role KVConnectorRole.SCHEDULER
```

验证:

```bash
curl -s http://127.0.0.1:19001/v1/models
curl http://127.0.0.1:19001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-lmcache","messages":[{"role":"user","content":"请用一句话说明你是谁。"}],"max_tokens":32,"temperature":0}'
```

当前 146 上 `40GB` 是实测稳定值。`64GB/80GB` 会触发:

```text
LMCacheEngine marked as init failed: musaHostAlloc failed: 205
MUDNN FillOp failed during CUDA graph capture
```

因此不要按 80GB 方案直接启动, 除非先解决 MUSA host allocation 限制。

### 14.10 第五层验证: LMCache 是否真的命中

小 chat 只能证明服务可用, 不能证明 LMCache 命中。要看命中必须抓 `/metrics`:

```bash
curl -s http://127.0.0.1:19001/metrics | grep prefix_cache
```

关键指标:

| 指标 | 含义 |
| --- | --- |
| `vllm:prefix_cache_hits_total` | GPU APC 命中 token 数 |
| `vllm:prefix_cache_queries_total` | GPU APC 查询 token 数 |
| `vllm:external_prefix_cache_hits_total` | LMCache 命中 token 数 |
| `vllm:external_prefix_cache_queries_total` | LMCache 查询 token 数 |

验证性能收益时不要只跑随机 benchmark, 建议用冷/热分离脚本:

```bash
/root/.virtualenvs/sglang-0.5.6/bin/python3 \
  /data/my_vllm_test/vllm_020/lmcache_cold_warm_revisit_bench.py \
  --port 19001 \
  --served-model-name qwen3-8b-lmcache \
  --groups 80 \
  --prefix-len 3072 \
  --question-len 128 \
  --output-len 32 \
  --concurrency 4 \
  --seed 2026061702 \
  --output /data/my_vllm_test/vllm_020/bench_results_lmcache_step2/lmcache40_cold_warm_19001.json
```

pure baseline 用相同参数、相同 seed, 只改端口和 served model name:

```bash
--port 19000
--served-model-name qwen3-8b-pure-fixed
```

已验证结果详见:

```text
/data/my_vllm_test/ISSUE_vllm_musa_broadcast_deadlock2.md
```

---

## 15. 迁移故障速查表

### 15.1 `Allocator for musa is not a DeviceAllocator`

现象:

```text
RuntimeError: device_allocator INTERNAL ASSERT FAILED
Allocator for musa is not a DeviceAllocator.
vllm/v1/worker/gpu_worker.py:279
torch.accelerator.empty_cache()
```

真实原因:

```text
torchada 版本不对或没有被 vllm_musa 正确加载。
```

判断方法:

```python
import torch, torch_musa, vllm_musa
print(torch.accelerator.empty_cache.__module__)
```

正确结果:

```text
torch_musa.core.memory
```

错误结果:

```text
torch.accelerator.memory
```

修复:

1. 对齐 127 可运行环境里的 `torchada 0.1.56`;
2. 确认它位于 venv site-packages, 而不是继续用 `/usr/local` 里的旧版;
3. 不要恢复 `sitecustomize.py` 兼容补丁。

### 15.2 `flash_attn_varlen_func() got an unexpected keyword argument 'page_table'`

现象:

```text
TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'page_table'
vllm_musa/v1/attention/backends/flash_attn.py
mate.mha_interface.flash_attn_varlen_func(...)
```

真实原因:

```text
mate / flash_attn_3 运行库版本和当前 vllm_musa V1 attention backend 不匹配。
```

判断方法:

```python
import inspect
from mate.mha_interface import flash_attn_varlen_func
print(inspect.signature(flash_attn_varlen_func))
```

正确签名里应该包含:

```text
page_table: Optional[torch.Tensor] = None
```

修复:

1. 对齐 127 的 `mate 0.2.0+mu437torch2.9`;
2. 同步配套 `flash_attn`, `flash_attn_3`, `flash_attn_interface.py` 等包;
3. 再跑纯净 vLLM baseline, 不要直接挂 LMCache。

### 15.3 LMCache native adapter import 失败

现象:

```text
ImportError: cannot import name 'LMCacheEngineMetadata' from 'lmcache.config'
```

解释:

这是 vLLM 仓内 native adapter 期望更新的 LMCache Python API, 而当前 MUSA port 是 LMCache 0.3.0。默认外部 adapter 路径:

```text
lmcache.integration.vllm.vllm_v1_adapter
```

已经能 import, 当前端到端启动走的是这条路径。因此这不是当前阻塞。

处理:

1. 不要把这个错误误判为 LMCache 整体不可用;
2. 第一阶段使用默认外部 adapter;
3. 之后如需 `use_native=true`, 再升级/适配 LMCache API。

### 15.4 `test_mem_kernels.py` 大量失败

现象:

```text
torch.OutOfMemoryError: MUSA out of memory
RuntimeError: musaHostAlloc failed: 2
```

判断:

先看 `mthreads-gmi`。165 当时 8 卡都被 `sglang` TP=8 服务占用约 75GB/卡, 所以 kernel 测试失败是资源不足, 不是数值错误。

验证:

146 空闲卡重跑后:

```text
37 passed
```

处理:

1. 等空闲卡或换空闲服务器;
2. 不要在显存只剩 5GB 的情况下判断 LMCache kernel 失败;
3. 关注是否有数值 assert failure, 而不是只看 OOM。

### 15.5 cache engine remote 测试失败

现象:

```text
Failed to initialize/re-establish remote connection: [Errno 111] Connection refused
TimeoutError: Operation timed out after 30 seconds.
```

解释:

测试配置使用:

```text
remote_url='lm://localhost:18078'
```

但当前没有启动 LMCache remote server。这个失败只说明 remote backend 没服务, 不影响本地 `LocalCPUBackend` / local_disk / paged KV 主路径。

处理:

1. 如果目标是本地 CPU cache offload, 可以先忽略 remote 测试;
2. 如果要验证 remote backend, 先按 LMCache fixture/脚本启动 `lm://localhost:18078` 对应服务。

### 15.6 LMCache CPU cache 64GB/80GB 启动失败

现象:

```text
LMCacheEngine marked as init failed: musaHostAlloc failed: 205
MUDNN FillOp failed during CUDA graph capture
```

解释:

LMCache `LocalCPUBackend` 初始化大容量 CPU cache 时会申请 host/pinned memory。146 当前 MUSA runtime + 容器配置下, 64GB/80GB 申请失败。这个问题发生在 LMCache 初始化/host allocation 阶段, 不是 vLLM V1 推理框架自身坏。

处理:

1. 当前稳定配置使用 `LMCACHE_MAX_LOCAL_CPU_SIZE=40`;
2. 失败后检查是否有残留 `VLLM::EngineCore` 占 GPU;
3. 清理残留后再重启, 不要反复叠加启动进程;
4. 如果必须使用 80GB, 需要单独排查 MUSA host allocation / memlock / runtime 限制。

### 15.7 benchmark 客户端去 HuggingFace 联网

现象:

```text
HEAD https://huggingface.co/qwen3-8b-lmcache/resolve/main/tokenizer_config.json
Connection reset by peer
```

原因:

benchmark 参数把 served model name 当成了模型路径:

```text
--model qwen3-8b-lmcache
```

修复:

```bash
--model /data/SQT-v1.0.5-test/models/qwen3-8b
--served-model-name qwen3-8b-lmcache
--tokenizer /data/SQT-v1.0.5-test/models/qwen3-8b
```

含义:

- `--model`: 给 benchmark/tokenizer/config 用的本地模型路径;
- `--served-model-name`: OpenAI 请求里的 `model` 字段, 必须匹配 vLLM 服务启动名;
- `--tokenizer`: 明确指定本地 tokenizer, 避免联网。

---

## 16. 最终交接状态

截至 2026-06-17, 可交接状态如下:

| 项 | 状态 |
| --- | --- |
| 146 容器 | `vllm020_lmcache_test` |
| pure 服务 | 19000, `qwen3-8b-pure-fixed` |
| LMCache 服务 | 19001, `qwen3-8b-lmcache` |
| GPU KV cache | 216,704 tokens |
| LMCache CPU cache | 40GB 稳定; 64/80GB 失败 |
| LMCache 后端 | `LocalCPUBackend` |
| 端到端小请求 | pure/LMCache 均通过 |
| LMCache external hit | 已通过 cold/warm 分离测试验证 |
| 性能收益 | warm revisit 场景已验证, 详见 `ISSUE_vllm_musa_broadcast_deadlock2.md` |

接手后如果要继续扩展, 建议顺序:

1. 保持当前 19000/19001 单卡 Qwen3-8B 基线不变, 先复跑 cold/warm 分离脚本确认环境未变;
2. 再扩大工作集或并发, 观察 `external_prefix_cache_hits_total` 和 TTFT;
3. 再考虑 Qwen3-14B/32B 或 TP 场景;
4. 最后再研究 remote backend / NIXL / 多机, 不要把这些和本地 CPU cache 主路径混在同一轮排查。
