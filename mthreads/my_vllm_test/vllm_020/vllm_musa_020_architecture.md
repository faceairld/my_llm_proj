# vllm_musa 0.20/V1 架构与本地移植记录

> 日期: 2026-06-11  
> 本地机器: node165 `/data/my_vllm_test`  
> 远端来源: `192.168.4.127:30001`, venv `/root/.virtualenvs/sglang-0.5.6`  
> 本地容器: `vllm020_test`  
> 目标: 在不影响 127 当前压测程序的前提下, 复原新版 vllm_musa 环境, 并梳理源码结构与调用链。

---

## 1. 当前结论

本地 `vllm020_test` 已经用 165 本地 torch2.9 镜像创建, 并把 127 同版本的新版 vllm/vllm_musa 以 symlink overlay 的方式接入:

| 项 | 状态 |
| --- | --- |
| base image | `registry.mthreads.com/mcconline/inference/sglang:v0.5.6.post2-ph1-4.3.5-torch2.9.0-20260403` |
| container | `vllm020_test`, `sleep infinity`, 未启动模型服务 |
| torch / torch_musa | `2.9.0 / 2.9.0` |
| vllm | `0.20.1.dev0+g88d34c640.d20260519.empty` |
| vllm_musa | `0.1.1` |
| vLLM engine | 纯 V1, `vllm/worker/` 不存在, `vllm/v1/` 存在 |
| CLI 验证 | `vllm --help` 和 `vllm --version` 通过 |
| 模型服务验证 | 尚未启动, 为避免抢 GPU/影响测试程序 |

`vllm --help` 日志里仍会先出现一次:

```text
Failed to load plugin musa ... partially initialized module 'vllm_musa'
```

随后会打印:

```text
Platform plugin musa is activated
```

这与之前 issue 文档里记录的现象一致: 首次 eager plugin load 有 circular import 噪声, 后续 lazy 解析能激活 MUSA platform。

---

## 2. 本地移植方式

### 2.1 资产位置

| 资产 | 路径 |
| --- | --- |
| 新版源码 | `/data/my_vllm_test/vllm_020/vllm-musa/` |
| vllm 完整 Python 包 | `/data/my_vllm_test/vllm_020/vllm-musa/third_party/vllm/vllm/` |
| vllm_musa Python 包 | `/data/my_vllm_test/vllm_020/vllm-musa/vllm_musa/` |
| vllm/vllm_musa dist-info | `/data/my_vllm_test/vllm_020/_dist_info/` |
| 从 127 同步的补充依赖 | `/data/my_vllm_test/vllm_020/_remote_deps/` |
| overlay 脚本 | `/data/my_vllm_test/install_vllm020_overlay.sh` |
| 依赖 overlay 脚本 | `/data/my_vllm_test/install_remote_deps_overlay.sh` |

### 2.2 容器创建命令

```bash
docker run -d \
  --name vllm020_test \
  --runtime=mthreads \
  --privileged \
  --network host \
  --pid host \
  --shm-size 500g \
  --ulimit memlock=-1:-1 \
  --security-opt label=disable \
  -v /data:/data \
  -v /data:/home/dist \
  -v /data/models:/data/models \
  -w /data/my_vllm_test \
  registry.mthreads.com/mcconline/inference/sglang:v0.5.6.post2-ph1-4.3.5-torch2.9.0-20260403 \
  sleep infinity
```

这个容器只是 idle 常驻, 没有启动 `vllm serve`。

### 2.3 overlay 安装

```bash
docker exec vllm020_test bash /data/my_vllm_test/install_vllm020_overlay.sh
docker exec vllm020_test bash /data/my_vllm_test/install_remote_deps_overlay.sh
```

`install_vllm020_overlay.sh` 做的事:

1. 备份容器 venv 里原始 `vllm`, `vllm_musa`, `*.dist-info`;
2. `site-packages/vllm -> /data/my_vllm_test/vllm_020/vllm-musa/third_party/vllm/vllm`;
3. `site-packages/vllm_musa -> /data/my_vllm_test/vllm_020/vllm-musa/vllm_musa`;
4. dist-info 指向 `_dist_info` 中从 127 拷回的元数据。

`install_remote_deps_overlay.sh` 补齐新版 CLI 需要但 base image 缺少/过旧的依赖:

| 包 | 来源版本 | 用途 |
| --- | --- | --- |
| `openai` | `2.37.0` | vLLM OpenAI entrypoint 的 typing/protocol 依赖 |
| `model_hosting_container_standards` | `0.1.15` | vLLM sagemaker router import 依赖 |
| `jmespath` | `1.1.0` | 上一个包的 transform 依赖 |
| `supervisor` | `4.3.0` | 上一个包声明依赖 |
| `flash_attn_interface.py` | remote file | MUSA FA3 接口兼容层 |
| `flash_attn_3` | `0.1.4` | `flash_attn_varlen_func` 等 V1 attention kernel Python 入口 |

### 2.4 验证命令

```bash
docker exec vllm020_test /root/.virtualenvs/sglang-0.5.6/bin/vllm --version
docker exec vllm020_test /root/.virtualenvs/sglang-0.5.6/bin/vllm --help
docker exec vllm020_test /root/.virtualenvs/sglang-0.5.6/bin/python -c 'from importlib import metadata; import vllm, vllm_musa; print(metadata.version("vllm")); print(metadata.version("vllm_musa"))'
```

验证结果:

```text
vllm 0.20.1.dev0+g88d34c640.d20260519.empty
vllm_musa 0.1.1
torch 2.9.0
torch_musa 2.9.0
openai 2.37.0
flash_attn_3 0.1.4
has_worker False
has_v1 True
```

---

## 3. 与旧 ISSUE 的关系

旧问题 `ISSUE_vllm_musa_broadcast_deadlock.md` 里真正的根因是 V0 路径上的:

```text
vllm_musa/v0/flash_attn.py
  -> sdpa_attention_with_kernel_seqlen_pad
  -> ops.varlen_fa_seqlen_pad / ops.varlen_fa_seqlen_unpad
```

新版环境的关键变化:

1. 上游 vLLM 已经删除 V0 执行层, `vllm/worker/` 不存在;
2. `VLLM_USE_V1` 不再是有效切换开关;
3. vllm_musa 不再有 `v0/` 目录;
4. prefill/decode/prefix-cache attention 进入 V1 backend:

```text
vllm.v1 engine
  -> vllm_musa.platform.MUSAPlatform.get_attn_backend_cls()
  -> vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend
  -> FlashAttentionImpl.forward()
  -> flash_attn_varlen_func / reshape_and_cache_flash
```

所以旧 OOB 路径是被架构性绕开, 不是在原 kernel 上修好。新版仍要实测 `flash_attn_varlen_func` 在 S5000/arch310 的具体场景稳定性。

---

## 4. 新版 vllm_musa 层次结构图

### 4.1 源码目录层次

```text
vllm-musa/
├── vllm_musa/                         # MUSA out-of-tree plugin 主包
│   ├── __init__.py                    # vLLM plugin 入口: platform_plugins/general_plugins
│   ├── platform.py                    # MUSAPlatform: 设备抽象、配置修正、attention backend 选择
│   ├── worker.py                      # MTGPUWorker: V1 Worker 子类
│   ├── _custom_ops.py                 # torch.ops._C_musa_ops Python wrapper
│   ├── collect_env.py                 # 环境采集 CLI
│   │
│   ├── _inductor/
│   │   └── template_heuristics.py     # MUSA inductor template heuristic 注册
│   │
│   ├── utils/
│   │   └── environ.py                 # vllm_musa 自定义环境变量
│   │
│   ├── distributed/
│   │   ├── device_communicators/
│   │   │   └── quick_all_reduce.py    # MUSA quick all-reduce 适配
│   │   └── kv_transfer/
│   │       └── kv_connector/v1/
│   │           └── mooncake_connector.py
│   │
│   ├── kernels/
│   │   └── musa_ops.py                # vllm.ir.ops 的 MUSA provider
│   │
│   ├── model_executor/
│   │   ├── kernels/linear/scaled_mm/  # DeepGEMM / torch scaled-mm
│   │   ├── layers/
│   │   │   ├── activation.py          # SiluAndMul / swish_glu
│   │   │   ├── layernorm.py           # RMSNorm / fused_add_rms_norm
│   │   │   ├── fused_moe/             # MoE experts/router/config bridge
│   │   │   └── quantization/          # FP8 weight + activation quant
│   │   └── warmup/
│   │       └── deep_gemm_warmup.py
│   │
│   ├── patches/                       # 运行时 patch 上游 vLLM Python 源码
│   │   ├── __init__.py                # patch 扫描、字符串替换、atomic rename
│   │   └── vllm__*.patch.py           # 逐模块 patch 配置
│   │
│   └── v1/
│       ├── attention/
│       │   ├── backends/
│       │   │   ├── fa_utils.py        # flash_attn_3 接口与 KV cache update wrapper
│       │   │   ├── flash_attn.py      # V1 主 attention backend
│       │   │   ├── tree_attn.py       # tree drafting attention
│       │   │   ├── turboquant.py      # TurboQuant attention 适配
│       │   │   └── mla/
│       │   │       ├── common.py      # MLA common helper
│       │   │       └── flashmla.py    # MUSA FlashMLA backend
│       │   └── ops/
│       │       └── flashmla.py        # FlashMLA op support 检测/封装
│       └── spec_decode/
│           ├── attn_backend_array.py  # Eagle 多 step metadata array
│           ├── eagle_full_loop_runner.py
│           ├── spec_info.py           # Eagle buffer/context/result 数据结构
│           └── utils.py               # Eagle 辅助 kernel/helper
│
├── third_party/vllm/vllm/             # 与 vllm_musa 匹配的上游 vLLM 0.20.1 包
│   ├── engine/                        # V1 engine alias/入口
│   ├── entrypoints/                   # vllm CLI / OpenAI server
│   ├── v1/                            # V1 engine、worker、scheduler、attention
│   ├── model_executor/                # 上游模型执行层
│   ├── distributed/                   # 上游分布式框架
│   ├── compilation/                   # torch.compile / cudagraph
│   └── _C*.so                         # vLLM 预编译扩展
│
└── vllm/                              # 仅保存顶层 vLLM .so 备份, 不是完整包
    ├── _C.cpython-310-x86_64-linux-gnu.so
    └── _moe_C.cpython-310-x86_64-linux-gnu.so
```

### 4.2 运行时分层图

```text
┌─────────────────────────────────────────────────────────────┐
│ vllm CLI / OpenAI API server                                │
│ third_party/vllm/vllm/entrypoints/*                         │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│ vLLM V1 engine                                               │
│ vllm.v1.engine / scheduler / worker / gpu_model_runner       │
└───────────────────────────────┬─────────────────────────────┘
                                │ current_platform
┌───────────────────────────────▼─────────────────────────────┐
│ vllm_musa.platform.MUSAPlatform                             │
│ - worker_cls = vllm_musa.worker.MTGPUWorker                  │
│ - block_size / cudagraph / backend selection                 │
│ - device memory/capability/fully-connected query             │
└───────────────────────────────┬─────────────────────────────┘
                                │ register backends + custom ops
┌───────────────────────────────▼─────────────────────────────┐
│ vllm_musa V1 backend layer                                   │
│ flash_attn / flashmla / turboquant / tree_attn               │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│ vllm_musa model_executor layer                               │
│ activation / RMSNorm / MoE / FP8 / DeepGEMM                  │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│ Python wrapper + native kernels                              │
│ vllm_musa._custom_ops -> torch.ops._C_musa_ops               │
│ flash_attn_interface -> flash_attn_3.interface               │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│ torch_musa / MUSA runtime / MCCL / MTT kernel                │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 总体加载与调用结构

### 5.1 插件入口

dist-info 入口:

```ini
[vllm.platform_plugins]
musa = vllm_musa:musa_platform_plugin

[vllm.general_plugins]
musa_custom_ops = vllm_musa:register_custom_ops
```

加载链:

```text
vllm CLI / vllm serve
  -> vllm.plugins.load_plugins_by_group("vllm.platform_plugins")
  -> vllm_musa.musa_platform_plugin()
      -> torchada.is_musa_platform() or import torch_musa
      -> return "vllm_musa.platform.MUSAPlatform"
  -> current_platform = MUSAPlatform
  -> vllm 配置初始化
      -> MUSAPlatform.apply_config_platform_defaults()
      -> MUSAPlatform.check_and_update_config()
  -> vllm.plugins.load_general_plugins()
  -> vllm_musa.register_custom_ops()
      -> _register_patches()
      -> _register_ops()
      -> _register_modules()
```

### 5.2 配置阶段

`MUSAPlatform.check_and_update_config()` 是最核心的配置修正点:

```text
parallel_config.worker_cls == "auto"
  -> "vllm_musa.worker.MTGPUWorker"

cache_config.block_size is None
  -> 16

MLA / FlashMLA / sparse MLA
  -> 可能强制 block_size = 64

Qwen3 MoE FP8
  -> cap max_cudagraph_capture_size = 64

torch_musa 2.9 large cudagraph bug workaround
  -> 默认 cap max_cudagraph_capture_size = 8
  -> 可由 VLLM_MUSA_MAX_CUDAGRAPH_CAPTURE_SIZE 覆盖
```

### 5.3 V1 attention 选择链

```text
vllm.v1.attention.selector.get_attn_backend()
  -> current_platform.get_attn_backend_cls(...)
  -> MUSAPlatform.get_attn_backend_cls(...)
      -> register_attention_backends()
          FLASHMLA -> vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend
          FLASH_ATTN -> vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend
          TURBOQUANT -> vllm_musa.v1.attention.backends.turboquant.MUSATurboQuantAttentionBackend
          TREE_ATTN -> vllm_musa.v1.attention.backends.tree_attn.MUSATreeAttentionBackend
      -> validate selected backend or choose highest-priority valid backend
```

普通 decoder 模型优先级:

```text
FLASH_ATTN -> TRITON_ATTN -> TURBOQUANT
```

MLA 模型优先级:

```text
FLASHMLA -> TRITON_MLA
```

---

## 6. 关键调用结构图

### 6.1 `vllm serve` 启动调用链

```text
vllm serve ...
  │
  ├─ vllm.entrypoints.cli.main.main()
  │   └─ vllm.entrypoints.cli.serve.ServeSubcommand
  │       └─ vllm.entrypoints.openai.api_server
  │
  ├─ vllm.plugins.load_plugins_by_group("vllm.platform_plugins")
  │   └─ vllm_musa:musa_platform_plugin()
  │       ├─ import torchada early
  │       ├─ torchada.is_musa_platform() / import torch_musa
  │       └─ return "vllm_musa.platform.MUSAPlatform"
  │
  ├─ current_platform = MUSAPlatform
  │
  ├─ EngineArgs / VllmConfig 初始化
  │   ├─ MUSAPlatform.apply_config_platform_defaults()
  │   │   ├─ compilation_config.custom_ops += "all"
  │   │   └─ Qwen3Moe FP8 cudagraph capture size cap
  │   └─ MUSAPlatform.check_and_update_config()
  │       ├─ worker_cls = "vllm_musa.worker.MTGPUWorker"
  │       ├─ default block_size = 16
  │       ├─ MLA / FlashMLA block_size 修正
  │       └─ torch_musa 2.9 cudagraph 大 shape cap
  │
  ├─ vllm.plugins.load_general_plugins()
  │   └─ vllm_musa:register_custom_ops()
  │       ├─ _register_patches()
  │       │   └─ vllm_musa.patches.apply_patches()
  │       ├─ _register_ops()
  │       │   └─ import vllm_musa.model_executor
  │       └─ _register_modules()
  │           ├─ import vllm_musa.distributed
  │           ├─ import vllm_musa.utils
  │           └─ import vllm_musa.v1
  │
  └─ V1 engine start
      └─ workers = MTGPUWorker / vllm.v1.worker.gpu_worker.Worker
```

### 6.2 一次请求的 V1 推理调用链

```text
OpenAI HTTP request
  │
  └─ vLLM OpenAI server
      │
      └─ AsyncLLM / V1 engine
          │
          ├─ scheduler 组织 batch
          │
          ├─ gpu_model_runner 构造 model input + attention metadata
          │
          ├─ model forward
          │   ├─ Attention layer
          │   │   ├─ vllm.v1.attention.selector.get_attn_backend()
          │   │   │   └─ MUSAPlatform.get_attn_backend_cls()
          │   │   │       └─ MUSAFlashAttentionBackend / FlashMLA / TurboQuant
          │   │   │
          │   │   └─ FlashAttentionImpl.forward()
          │   │       ├─ reshape_and_cache_flash()
          │   │       │   ├─ musa_reshape_and_cache_flash_nhd()
          │   │       │   └─ fallback vllm._custom_ops.reshape_and_cache_flash()
          │   │       ├─ flash_attn_varlen_func()
          │   │       └─ cascade_attention()   # prefix cache 命中时可能走
          │   │
          │   ├─ RMSNorm / activation / MoE / FP8
          │   │   └─ vllm_musa.model_executor.layers.*
          │   │       └─ torch.ops._C_musa_ops.*
          │   │
          │   └─ logits processor / sampler
          │       └─ vllm_musa patches 修改部分 topk/topp/rejection sampler
          │
          └─ response output tokens
```

### 6.3 Attention backend 选择图

```text
vllm.v1.attention.selector
  │
  └─ current_platform.get_attn_backend_cls(...)
      │
      ├─ register_attention_backends()
      │   ├─ FLASH_ATTN  -> MUSAFlashAttentionBackend
      │   ├─ FLASHMLA    -> MUSAFlashMLABackend
      │   ├─ TURBOQUANT  -> MUSATurboQuantAttentionBackend
      │   └─ TREE_ATTN   -> MUSATreeAttentionBackend
      │
      ├─ if user selected backend:
      │   ├─ validate_configuration()
      │   └─ return selected_backend.get_path()
      │
      └─ else auto select:
          ├─ MLA model:
          │   └─ FLASHMLA -> TRITON_MLA
          └─ non-MLA model:
              └─ FLASH_ATTN -> TRITON_ATTN -> TURBOQUANT
```

### 6.4 Prefix-cache / cascade attention 调用图

```text
common prefix exists
  │
  └─ FlashAttentionMetadataBuilder.build(common_prefix_len, ...)
      │
      ├─ split_decodes_and_prefills()
      ├─ prefix_kv_lens = [common_prefix_len]
      ├─ suffix_kv_lens = seq_lens - common_prefix_len
      ├─ prefix_scheduler_metadata = schedule(prefix)
      ├─ scheduler_metadata = schedule(suffix)
      └─ FlashAttentionMetadata(use_cascade=True, ...)
          │
          └─ FlashAttentionImpl.forward()
              │
              ├─ use_cascade_attention(...)
              │   ├─ common_prefix_len < 256 -> False
              │   └─ tile estimate says worthwhile -> True
              │
              └─ cascade_attention()
                  ├─ prefix_output, prefix_lse =
                  │     flash_attn_varlen_func(query, prefix_kv, ...)
                  ├─ suffix_output, suffix_lse =
                  │     flash_attn_varlen_func(query, suffix_kv, ...)
                  └─ merge_attn_states(prefix, suffix)
```

### 6.5 旧 V0 bug 路径与新版 V1 路径对照

```text
旧版 vllm_musa 0.9/torch2.7 路径
  vllm_musa/v0/flash_attn.py
    -> sdpa_attention_with_kernel_seqlen_pad()
      -> varlen_fa_seqlen_pad()
      -> SDPA
      -> varlen_fa_seqlen_unpad()     # 旧 ISSUE 中定位的越界写

新版 vllm_musa 0.20/torch2.9 路径
  vllm_musa/v1/attention/backends/flash_attn.py
    -> FlashAttentionMetadataBuilder.build()
    -> FlashAttentionImpl.forward()
      -> reshape_and_cache_flash()
      -> flash_attn_varlen_func()     # flash_attn_3 / MUSA FA3 入口
      -> cascade_attention()          # prefix-cache 命中时可能走
```

---

## 7. V1 FlashAttention 调用链

核心文件:

```text
vllm_musa/v1/attention/backends/flash_attn.py
vllm_musa/v1/attention/backends/fa_utils.py
```

### 7.1 metadata build

```text
FlashAttentionMetadataBuilder.build()
  -> split_decodes_and_prefills(...)
  -> 生成 decode_query_start_loc / decode_seq_lens / decode_block_table
  -> 生成 prefill_query_start_loc / prefill_max_seq_len
  -> common_prefix_len > 0 时构造 cascade metadata:
       cu_prefix_query_lens
       prefix_kv_lens
       suffix_kv_lens
       prefix_scheduler_metadata
       scheduler_metadata
  -> 返回 FlashAttentionMetadata
```

### 7.2 forward 主路径

```text
FlashAttentionImpl.forward(query, key, value, kv_cache, attn_metadata, ...)
  -> 如有新 key/value:
       reshape_and_cache_flash(...)
          -> 优先 vllm_musa._custom_ops.musa_reshape_and_cache_flash_nhd(...)
          -> 不满足 guard 时 fallback 到 vllm._custom_ops.reshape_and_cache_flash(...)
  -> 没有 cascade:
       flash_attn_varlen_func(...)
  -> 有 common prefix / prefix cache cascade:
       cascade_attention(...)
          -> prefix_output = flash_attn_varlen_func(prefix kv)
          -> suffix_output = flash_attn_varlen_func(suffix kv)
          -> merge_attn_states(prefix_output, suffix_output)
```

### 7.3 prefix-cache 相关判断

```text
use_cascade_attention(common_prefix_len, query_lens, ...)
  -> common_prefix_len < 256 时不用 cascade
  -> 否则根据 tile 数量估算 prefix/suffix 分开算是否划算
```

这就是新版 prefix-cache 命中时的关键路径。旧版问题里的 `varlen_fa_seqlen_unpad` 不在这条链路中。

---

## 8. 逐文件说明

### 8.1 包入口与平台

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/__init__.py` | vllm_musa 包入口。定义版本、platform plugin `musa_platform_plugin()`、general plugin `register_custom_ops()`、torchada 预加载、runtime patch 注册、custom ops/module 注册。 |
| `vllm_musa/platform.py` | MUSAPlatform 实现。对接 vLLM Platform API, 设置 device 类型、worker 类、block size、cudagraph 限制、attention backend 选择、设备能力/显存/互联查询。 |
| `vllm_musa/worker.py` | `MTGPUWorker`, 继承 vLLM V1 `Worker`; 当前主要覆盖 dummy batch 执行方式。 |
| `vllm_musa/collect_env.py` | 环境采集命令, 用于 `vllm_collect_env`, 汇总 torch/torch_musa/MUSA/系统版本。 |
| `vllm_musa/_custom_ops.py` | Python wrapper, 调用 `torch.ops._C_musa_ops.*`, 包括 fused gemv/moe、fused add rms norm、MUSA reshape_and_cache_flash。 |

### 8.2 Inductor 与环境变量

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/_inductor/__init__.py` | 暴露 MUSA Inductor heuristic 注册入口。 |
| `vllm_musa/_inductor/template_heuristics.py` | 为 torch inductor 注册 MUSA template heuristic; 默认关闭, 通过 `VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1` 启用。 |
| `vllm_musa/utils/__init__.py` | utils 包占位。 |
| `vllm_musa/utils/environ.py` | vllm_musa 自定义环境变量定义, 例如 `VLLM_MUSA_RESHAPE_CACHE_FLASH` 等开关。 |

### 8.3 分布式与 KV transfer

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/distributed/__init__.py` | 注册/导入 MUSA 分布式相关模块。 |
| `vllm_musa/distributed/device_communicators/__init__.py` | device communicator 包占位。 |
| `vllm_musa/distributed/device_communicators/quick_all_reduce.py` | MUSA quick all-reduce wrapper/适配层。 |
| `vllm_musa/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` | V1 KV transfer connector, 对接 Mooncake 相关 KV 传输路径。 |

### 8.4 IR kernels

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/kernels/__init__.py` | 导入 MUSA IR provider。 |
| `vllm_musa/kernels/musa_ops.py` | 注册 `vllm.ir.ops` 的 MUSA provider, 主要用于 eager/显式优先级路径。 |

### 8.5 model_executor 层

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/model_executor/__init__.py` | 注册 MUSA model executor 层实现。 |
| `vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py` | DeepGEMM scaled-mm 实现/适配。 |
| `vllm_musa/model_executor/kernels/linear/scaled_mm/torch_scaled_mm.py` | torch scaled-mm fallback/适配实现。 |
| `vllm_musa/model_executor/layers/activation.py` | `MusaSiluAndMul`, 通过 `_musa_swish_glu` custom op 实现 activation OOT forward。 |
| `vllm_musa/model_executor/layers/layernorm.py` | `MusaRMSNorm`, 满足条件时使用 `musa_fused_add_rms_norm`, 否则 fallback。 |
| `vllm_musa/model_executor/layers/utils.py` | MUSA 层工具函数。 |
| `vllm_musa/model_executor/warmup/deep_gemm_warmup.py` | DeepGEMM warmup 入口。 |

### 8.6 MoE 与 FP8

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/model_executor/layers/fused_moe/fused_moe.py` | MUSA fused experts 主实现, 处理 FP8 scale block、quant scheme 判断、expert 执行。 |
| `vllm_musa/model_executor/layers/fused_moe/moe_config_bridge.py` | 在 torchada/vLLM MoE config 目录之间桥接配置。 |
| `vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py` | grouped top-k routing 实现。 |
| `vllm_musa/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | 未量化 MoE method 的 MUSA OOT forward/DeepGEMM 选择。 |
| `vllm_musa/model_executor/layers/quantization/fp8.py` | MUSA FP8 权重创建、尺寸 roundup、apply 逻辑。 |
| `vllm_musa/model_executor/layers/quantization/utils/fp8_utils.py` | FP8 block post-process、per-token group quant、Silu+FP8 融合路径。 |

### 8.7 Runtime patches

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/patches/__init__.py` | runtime patch 框架。扫描 `*.patch.py`, 找到目标 vLLM 模块, 进行字符串替换, 用 tempfile + atomic rename 避免多进程 import 读到半写文件。 |
| `vllm_musa/patches/README.md` | patch 机制说明。 |
| `vllm_musa/patches/vllm__compilation__backends.patch.py` | patch `vllm.compilation.backends`。 |
| `vllm_musa/patches/vllm__compilation__caching.patch.py` | patch `vllm.compilation.caching`。 |
| `vllm_musa/patches/vllm__compilation__compiler_interface.patch.py` | patch `vllm.compilation.compiler_interface`。 |
| `vllm_musa/patches/vllm__compilation__passes__pass_manager.patch.py` | patch compile pass manager。 |
| `vllm_musa/patches/vllm__compilation__piecewise_backend.patch.py` | patch piecewise backend。 |
| `vllm_musa/patches/vllm__distributed__device_communicators__all2all.patch.py` | patch all2all communicator。 |
| `vllm_musa/patches/vllm__distributed__device_communicators__cuda_communicator.patch.py` | patch CUDA communicator, 让 CUDA-like 路径适配 MUSA/MCCL。 |
| `vllm_musa/patches/vllm__distributed__device_communicators__custom_all_reduce.patch.py` | patch custom all-reduce。 |
| `vllm_musa/patches/vllm__model_executor__kernels__linear.patch.py` | patch linear kernel 选择/调用。 |
| `vllm_musa/patches/vllm__model_executor__layers__attention__attention.patch.py` | patch attention layer。 |
| `vllm_musa/patches/vllm__model_executor__layers__fused_moe__deep_gemm_moe.patch.py` | patch DeepGEMM MoE。 |
| `vllm_musa/patches/vllm__model_executor__layers__fused_moe__experts__deep_gemm_moe.patch.py` | patch experts DeepGEMM MoE。 |
| `vllm_musa/patches/vllm__model_executor__layers__quantization__fp8.patch.py` | patch FP8 quantization。 |
| `vllm_musa/patches/vllm__model_executor__layers__quantization__utils__fp8_utils.patch.py` | patch FP8 utils。 |
| `vllm_musa/patches/vllm__profiler__wrapper.patch.py` | patch profiler wrapper。 |
| `vllm_musa/patches/vllm__utils__deep_gemm.patch.py` | patch vLLM DeepGEMM utils。 |
| `vllm_musa/patches/vllm__v1__attention__backends__mla__flashmla.patch.py` | patch V1 FlashMLA backend。 |
| `vllm_musa/patches/vllm__v1__attention__ops__flashmla.patch.py` | patch V1 FlashMLA ops。 |
| `vllm_musa/patches/vllm__v1__attention__ops__triton_turboquant_decode.patch.py` | patch TurboQuant decode op。 |
| `vllm_musa/patches/vllm__v1__attention__ops__triton_unified_attention.patch.py` | patch Triton unified attention。 |
| `vllm_musa/patches/vllm__v1__sample__ops__topk_topp_sampler.patch.py` | patch top-k/top-p sampler。 |
| `vllm_musa/patches/vllm__v1__sample__ops__topk_topp_triton.patch.py` | patch top-k/top-p triton kernel。 |
| `vllm_musa/patches/vllm__v1__sample__rejection_sampler.patch.py` | patch rejection sampler。 |
| `vllm_musa/patches/vllm__v1__spec_decode__eagle.patch.py` | patch V1 Eagle spec decode。 |
| `vllm_musa/patches/vllm__v1__spec_decode__llm_base_proposer.patch.py` | patch LLM base proposer。 |
| `vllm_musa/patches/vllm__v1__worker__gpu_model_runner.patch.py` | patch V1 GPU model runner。 |
| `vllm_musa/patches/vllm__v1__worker__gpu_worker.patch.py` | patch V1 GPU worker。 |

### 8.8 V1 attention 后端

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/v1/__init__.py` | 导入 V1 attention/spec decode 相关模块, 触发 backend 注册。 |
| `vllm_musa/v1/attention/backends/__init__.py` | attention backend 包占位。 |
| `vllm_musa/v1/attention/backends/fa_utils.py` | MUSA FlashAttention helper。导入 `flash_attn_interface`, 封装 `reshape_and_cache_flash`, 声明 FA version=3, FP8/sink/MLA 支持能力。 |
| `vllm_musa/v1/attention/backends/flash_attn.py` | 主 V1 FlashAttention backend。定义 backend/metadata/builder/impl, 处理 prefill/decode/cascade/DCP/KV cache update。 |
| `vllm_musa/v1/attention/backends/tree_attn.py` | Tree attention backend, 用于 tree drafting/spec decode 场景。 |
| `vllm_musa/v1/attention/backends/turboquant.py` | TurboQuant attention backend 的 MUSA 适配, 将 varlen FA 调用路由到 MUSA `fa_utils`。 |
| `vllm_musa/v1/attention/backends/mla/common.py` | MLA common helper, 管理 FlashMLA/flash_attn_varlen_func 调用与输出布局。 |
| `vllm_musa/v1/attention/backends/mla/flashmla.py` | MUSA FlashMLA backend。 |
| `vllm_musa/v1/attention/ops/flashmla.py` | FlashMLA op availability/support 检测与调用封装。 |

### 8.9 Spec decode / Eagle

| 文件 | 功能 |
| --- | --- |
| `vllm_musa/v1/spec_decode/__init__.py` | spec decode 包入口。 |
| `vllm_musa/v1/spec_decode/attn_backend_array.py` | 为 Eagle 多步 draft 构造每 step attention metadata array。 |
| `vllm_musa/v1/spec_decode/eagle_full_loop_runner.py` | Eagle3 full-loop cudagraph runner, 尝试把多步 draft loop 捕获为单个 graph。 |
| `vllm_musa/v1/spec_decode/spec_info.py` | EagleDraftBuffers、capture context、replay result 数据结构。 |
| `vllm_musa/v1/spec_decode/utils.py` | Eagle next-token padding kernel 和 batch 变化时 token 计数更新 helper。 |

---

## 9. 需要继续验证的点

当前只验证到 CLI/import 层, 没有启动模型服务。后续如果要验证真实推理, 建议从低风险到高风险逐步做:

1. 单卡小模型 `vllm serve --help` / dry config 级别验证;
2. 单卡小模型 serve, 不开 prefix cache 压测;
3. 单卡 Qwen3-32B 或 FP8, 对齐 127 当前参数;
4. prefix-cache 命中场景复测;
5. TP>1 / TP=8 场景;
6. LMCache/V1 KV connector 集成验证。

注意: 本地容器有 host network 和 `--pid host`, 后续启动服务时要先确认端口/GPU 不与本机正在跑的 `vllm-musa-qwen3-30b-a3b-test` 冲突。

---

## 10. 安全边界

本次对 127 的操作只有:

1. `ssh` 只读查询版本/路径;
2. `scp`/`tar` 从 127 venv 只读同步 Python 包;
3. `pgrep`/`ps` 查看测试进程。

没有在 127 上执行 kill、重启、安装、改文件、启动服务或编译。

本地 `vllm020_test` 目前只是 idle 容器, 未启动 `vllm serve`, 不占用模型端口, 不主动打流。
