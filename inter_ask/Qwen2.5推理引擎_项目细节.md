# Qwen2.5-0.5B 推理引擎 + CUDA 算子：项目细节复习

> 依据：`Qwen2_5_0_5B/` 下全部源码、`测试时间记录.txt`、`vllm比对.txt`、git 历史（含未提交改动）、简历 `高杨_简历_infra.pdf` 第 2 页。整理于 2026-09-15。
> 另做了三个几秒钟的小检查（不是重跑 benchmark）：benchmark prompt 的分词长度；CUDA Graph 里的 `fill_` 会不会被固化；`CUDA_LAUNCH_BLOCKING` 有没有生效。脚本在本次会话 scratchpad 的 `qwen_probe.py`、`lb_probe.py`。

标记：✅ 代码或记录能对上 ⚠️ 与代码或记录冲突，面试容易被抓 ❓ 仓库里找不到出处 🔎 推断或估算，没实测

---

## 0. 一分钟口述版

用 PyTorch 从零搭 Qwen2.5-0.5B-Instruct。只借用 HF 的分词器和权重，模型代码自己写，fp16，静态 KV cache，跑在 RTX 3060 Laptop 上。先用 nsys + NVTX 找瓶颈，确认 batch=1 decode 时 GPU 大部分时间在等 CPU 下发 kernel。然后手写 5 个 CUDA kernel，通过 `cpp_extension` 注入：RMSNorm、decode attention、分块 decode attention（flash decoding）、prefill FlashAttention v1 和 v2。decode 路径改成静态 shape、序列长度放进 GPU buffer 之后接上 CUDA Graph，50 token 端到端生成（含一次 prefill）从 37.21 提到 70.65 tok/s。prefill 换成手写 FA v2 后记录到 82.41 tok/s。

---

## 1. 时间线

| 日期 | 事件 | 来源 |
|---|---|---|
| 简历写 2026.02–03 | | 简历 |
| 3/16 | 下载模型；`chat.py` / `benchmark.py`（HF 基线） | 文件 mtime |
| 3/18–3/19 | nsys 报告 `qwen_handwritten_v2`…`v4`、`v5_cuda_kernel`、`v6_use_attention_fun` | mtime。🔎 按文件名推测：v2–v4 做 PyTorch 层面优化，v5 上 RMSNorm kernel，v6 换 SDPA，顺序和简历一致 |
| 3/20 | `my_qwen_in_wsl_gpu.nsys-rep`（WSL 下跑） | mtime |
| 3/21 | `vllm_report.nsys-rep`、`vllm比对.txt`、`main.py` | mtime |
| 3/24 | `qwen_use_graph_report.nsys-rep`（CUDA Graph） | mtime |
| 3/25 | 首个提交 47023dd：引擎、RMSNorm、decode attention、CUDA Graph。记录文件已有 36.36 / 37.21 / 70.65 / vLLM 158.53 | git |
| 3/31 | 4fddef1：FA v1 | git |
| 4/1 | `qwen_use_prefill_flash_attention.nsys-rep` | mtime |
| 4/6 | bc63ca3：FA v2；benchmark prompt 改成 ×15；`my_qwen2.py` 加 `CUDA_LAUNCH_BLOCKING=1`；记录文件加 78.83 / 82.41 | git |
| 4/12 | 未提交：新增 `my_flash_decoding.cu`（未跟踪）；decode 按长度分派；FA v2 共享内存 +2；capture 调用改传 `past_len + 1`；flash decoding 对比数据 | mtime + git diff |

⚠️ 简历写 02–03，但 FA v2 和 flash decoding 实际是 4 月做的。被问到时间就按实际说：2 月底开始，4 月中收尾。

---

## 2. 环境、目录、怎么跑

### 2.1 环境
- GPU：NVIDIA GeForce RTX 3060 Laptop，6 GB ✅（nvidia-smi）
- Python 环境：`E:\envs\cuda_env`。提交进仓库的 pyc 是 `cpython-310`，当时用的是 Python 3.10 ✅；现在环境里是 torch 2.8.0+cu129，3–4 月时的 torch 版本没有记录 ❓
- Windows 原生运行（脚本里写的是 Windows 路径）；WSL 下也跑过一次对比（记录第 2 行）
- 算子编译：`torch.utils.cpp_extension.load` 在 import 时 JIT 编译 5 个扩展（`my_qwen2.py:17-45`），需要 MSVC + CUDA toolkit。每个 `.cu` 用 `PYBIND11_MODULE` 导出 `forward`

### 2.2 目录

| 文件 | 作用 |
|---|---|
| `my_qwen2.py` | 模型定义 + 加载 5 个算子 |
| `benchmark2.py` | 测速主脚本：预热 → 50 token 计时 → nsys 抓取窗口 |
| `main.py` | 交互式聊天：eager 模式，不用 graph，温度 0.7 采样，最多 100 token |
| `chat.py` / `benchmark.py` | HF `AutoModelForCausalLM` fp16 基线：聊天 / 打印结构 |
| `modeling_qwen2.py` | HF 官方实现，对照参考 |
| `test.py` | 草稿，跑不通（`slef`、`RoPE_function` 未定义） |
| `csrc/rmsnorm_kernel.cu` | RMSNorm |
| `csrc/my_decode_attention.cu` | normal decode attention（512 lane 的 stride 循环） |
| `csrc/my_flash_decoding.cu` | 分块 decode attention，2 个 kernel，未提交 |
| `csrc/my_flash_attention.cu` | prefill FA v1（已不用） |
| `csrc/my_flash_attention_v2.cu` | prefill FA v2（在用） |
| `测试时间记录.txt` | 吞吐原始记录，从 36.36 开始 |
| `vllm比对.txt` | nsys `cuda_gpu_kern_sum`：自己引擎 vs vLLM |
| `*.nsys-rep` ×9 | 各阶段 nsys 报告 |
| `modeel_dir/` | 权重 + 分词器。是 Instruct 版：eos 为 151645 `<|im_end|>`，带 chat template |

### 2.3 运行
```bat
cd /d E:\vscode\cuda_proj\SNN_proj1\Qwen2_5_0_5B
E:\envs\cuda_env\python.exe benchmark2.py
E:\envs\cuda_env\python.exe main.py
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -o qwen_xxx E:\envs\cuda_env\python.exe benchmark2.py
nsys stats --report cuda_gpu_kern_sum qwen_xxx.nsys-rep
```
`my_qwen2` 是相对 import，所以要在 `Qwen2_5_0_5B` 目录下运行。`main.py` 输入 quit/exit 退出。nsys 这行只抓 `cudaProfilerStart/Stop` 之间的第 3 阶段。
当时实际用的 nsys 命令没有记录 ❓。`vllm比对.txt` 里 `rms_kernel` 出现 3185 次 = 65 次前向 × 49，说明那次抓的是整个进程，三个阶段都在（见 §5.5）。

---

## 3. 模型结构

### 3.1 超参（`modeel_dir/config.json` ✅）

| 项 | 值 |
|---|---|
| hidden | 896 |
| MLP intermediate | 4864，激活 SiLU |
| 层数 | 24 |
| 注意力头 | Q 14 / KV 2（GQA，7:1），head_dim 64 |
| vocab | 151936 |
| RoPE base | 1e6 |
| RMSNorm eps | 1e-6 |
| tie_word_embeddings | true（lm_head 与 embedding 共享） |
| 权重文件 dtype | bf16；引擎里全部 fp16 |
| 结束符 | 151645 `<|im_end|>`、151643 `<|endoftext|>` |

### 3.2 参数量（手算，可能被问）
- q：896×896 + 896 = 803,712；k、v：896×128 + 128 = 114,816 各一份；o：802,816（无 bias）→ attention 共 1,836,160
- MLP：3 × 896 × 4864 = 13,074,432
- 两个 RMSNorm：1,792
- 每层 14,912,384；×24 = 357,897,216
- embedding（与 lm_head 共享）136,134,656；final norm 896
- **合计 494,032,768 ≈ 0.49B** ✅，和官方标称一致
- Qwen2 的 q/k/v 带 bias，o 不带 ✅（`my_qwen2.py:186-189`）

### 3.3 KV cache
- 每层 `k_cache`、`v_cache` 各一个 (1, 2, 512, 64) 的 fp16 buffer，预分配（`my_qwen2.py:178-184`）
- 每层 K+V = 2 × 2×512×64×2B = 262,144 B；24 层 ≈ **6.0 MiB**
- 每 token：24 层 × 2(K,V) × 2 头 × 64 × 2B = **12 KiB**。MHA（14 个 KV 头）要 84 KiB，GQA 省 7 倍
- 上限 512 写死：prompt + 生成超过 512 就越界。`main.py` 没有检查 ⚠️

### 3.4 前向数据流
```
input_ids (1, L)
 → Embedding 151936×896                                    my_qwen2.py:352, 372
 → RoPE cos/sin (1,1,L,64)：每次前向算一次，24 层共用          :81-99, 373
 → 24 × DecoderLayer                                        :303-342
      x ─→ RMSNorm(kernel) → Attention → + x
        ─→ RMSNorm(kernel) → MLP       → + x
 → final RMSNorm → 只取最后一个位置 [:, -1:] → lm_head 896→151936   :384
```
**Attention**（`my_qwen2.py:192-277`）
1. q/k/v Linear → view → transpose：q (1,14,L,64)，k/v (1,2,L,64)
2. RoPE：纯 PyTorch 实现（切两半、`cat(-x2, x1)`、`q*cos + rot*sin`，`:117-127`）。⚠️ **没有做成 kernel**
3. 写 cache：`self.k_cache[:,:,cache_position,:] = k_after_emb`。这是 index_put，下标 `cache_position` 是 GPU tensor（`:209-210`）
4. 分派
   - prefill（q 长度 > 1）：切 `k_cache[:,:,:L,:]` → `.contiguous()` → FA v2（`:244-262`）
   - decode（q 长度 = 1）：`current_seq_len > 256` 走 flash decoding，否则走 normal decode。两者都直接传整块 512 长的 cache 和 `seq_len_t`（`:264-268`）
5. (1,14,L,64) → (1,L,896) → o_proj（`:275`）

**MLP**：`down(silu(gate(x)) * up(x))`（`:293-297`）

**权重加载**（`benchmark2.py:10-38`，`main.py:54-91`）：HF key 映射到自己的 key。`input_layernorm → pre_Normal`，`post_attention_layernorm → post_Normal`，`self_attn.{q,k,v}_proj(+bias) → attention.{q,k,v}_weight`，`o_proj → o_weight`，`mlp.* → mlplayer.*`。safetensors 里没有 `lm_head.weight` 时用 `embed_tokens`。
⚠️ 注释写"零拷贝"（`benchmark2.py:9`），实际 `load_state_dict` 是 `copy_`，还顺带做了 bf16→fp16 转换，面试别说零拷贝。另外 `strict=False` 会把拼错的 key 静默吞掉。

**分词器**：用的是 HF `AutoTokenizer`，不是自己写的（简历写"输入分词器"时别让人误会）。

---

## 4. benchmark2.py 一次测速到底执行了什么

### 4.1 三个阶段
1. 预热（`:130-132`）：跑 10 步，不用 graph
2. 测速（`:141-152`）：`synchronize` → `perf_counter` → `benchmark_generate(max_seq_len=50, graph_use=True)` → `synchronize` → **tps = 50 / 耗时**
3. nsys 窗口（`:164-172`）：`cudaProfilerStart` + `emit_nvtx`，跑 5 步（用 graph）

### 4.2 prompt 长度（小检查 ✅）
- 基础句 ×1（3/25 提交版）：**45 token**
- 基础句 ×15（4/6 起）：**269 token**，50 步中 decode 的序列长度为 270–318

### 4.3 测速阶段逐步（prompt ×15）

| step | 做什么 | seq_len | 代码 |
|---|---|---|---|
| 0 | prefill 269 token，eager，走 FA v2；另外构造了一个 L×L mask，但 FA kernel 不接收它，自己做 causal | 269 | `:56-63, 78-86` |
| 1 | decode，eager | 270 | `:78-86` |
| 2 | **capture**：`torch.cuda.graph` 用静态输入录制一次前向（录制时真实执行），接着再 replay 一次 | 271 | `:67-77, 87-92` |
| 3–49 | 新 token、位置 `copy_` 进静态 tensor → `seq_len_t.fill_` → replay | 272–318 | `:87-92` |

每一步在 graph 外还有：`torch.arange`、`argmax`、`torch.cat`，以及放在 `if` 里的 `(final_data == 151643).any()`。最后这个会隐式调用 `.item()`，**每步一次 GPU→CPU 同步**（`:95-101`）。

### 4.4 口径结论 ⚠️
- 分子固定 50：50 步产出 50 个 token（prefill 那步也出一个）。EOS 只判断 151643，而 Instruct 模型的轮次结束符是 151645，对这个长 prompt 在 50 步内不会停，50 步会跑满 🔎。
- **分母里包含 prefill、一次 eager decode、一次 graph capture**。所以这是"含 prefill 的 50 token 端到端生成吞吐"，**不是纯 decode 吞吐**。脚本打印的"只算纯解码耗时"（`:138`）和简历上"只测试了 decoding 阶段"都和代码不符。
- 每个配置只跑一次、出一个数，没有重复取中位数（只有 4/12 的 flash decoding 对比测了 4–5 次）。
- 这也说明了为什么"prefill 阶段注入 FA"会改变这个吞吐数字：prefill 就在计时窗口里。

### 4.5 CUDA Graph 是怎么接进来的（核心，要能讲清）
CUDA Graph 的要求：每次 replay 的 kernel 序列、tensor 地址、launch 参数完全一样；Python 分支和 Python 标量在录制那一刻就定死。项目里为此做的改动：
1. **静态输入**：`static_input` (1,1)、`static_pos_num` (1,1)、`static_cache_pos` (1)，每步 `copy_` / `fill_` 进去（`benchmark2.py:45-47, 88-90`）✅
2. **KV cache 预分配**：固定 512 长，用 GPU tensor 作下标做 index_put，地址不变（`my_qwen2.py:181-184, 209`）✅
3. **decode 不再切片**：`k_cache[:,:,:L,:]` 的 shape 随 L 变，会被 graph 固化；改成把整块 cache 传给 kernel（`:264-268`）✅。简历上"消除 python 端的动态数据切片"说的就是这一点。prefill 仍然切片，但 prefill 不进 graph。
4. **长度放在 GPU 上**：`seq_len_t` 是 int32 buffer（`:356`）。kernel 从设备内存读：lane 0 读一次 `current_seq_len[0]`，`__syncthreads()` 之后所有 lane 共享（`my_decode_attention.cu:22-27`）✅
5. **grid/block 与 L 无关**：normal decode 的 block 固定 512；flash decoding 的 grid 固定 8 组×64 ✅，所以每步 launch 参数相同
6. RoPE 的位置走 `static_pos_num`，cos/sin 在图内计算 ✅
7. 第 0、1 步 eager，等于 capture 前预热了两步。PyTorch 文档建议在 side stream 上预热，这里是在默认 stream 上跑的

### 4.6 4/12 未提交版本里的两个 capture 问题 ⚠️
**问题 1：分派被固化。**`if(current_seq_len > 256)` 是 Python 分支，graph 只记住 capture 那一步走的分支。capture 时 seq=271 > 256，之后所有 replay 都是 flash decoding。再加上 prompt 有 269 token，eager 的第 1 步（270）也大于 256。
→ **配置 1（按长度分派）和配置 2（只用 flash decoding）执行的 kernel 完全相同。**

**问题 2：`seq_len_t` 被固化成 271。**因为分派需要 int，capture 调用从 `current_seq_len=None` 改成了 `past_len + 1`（`benchmark2.py:75`，git diff 可见）。而 `Qwen2Model.forward` 里有 `if current_seq_len is not None: self.seq_len_t.fill_(current_seq_len)`（`my_qwen2.py:370-371`），于是这个带着 Python 常量 271 的 `fill_` 被录进了图。每次 replay 都会先执行它，把 `benchmark2.py:91` 刚写入的真实长度覆盖回 271。
- 小检查验证过 ✅：图里有 `t.fill_(5)`，replay 前在图外 `t.fill_(9)`，kernel 读到的是 5；图里没有 `fill_` 时读到的是 9。
- 后果：从第 3 步起，decode attention 只看位置 0–270，当前 token 自己的 K/V 和之后生成的全都看不到，**生成的 token 是错的**。benchmark 不检查输出，所以照样出速度数字。
- 按修改时间（`benchmark2.py` 4/12 17:44 早于记录文件 18:15）🔎，4/12 那组 flash decoding 对比很可能就是在这个状态下测的。对速度影响不大（attention 长度固定为 271，而不是从 271 涨到 318），但输出正确性没有保证。
- bc63ca3 提交版里 capture 传的是 None，没有这个问题。
- 修法：capture 继续传 None；分派改成 capture 前按长度选定一个 kernel，或者不同长度区间各录一张图。

---

## 5. 优化链：改了什么、为什么快、证据

### 5.1 数字总表

| # | 改动 | tok/s | 耗时 s | 50/耗时 | 出处 |
|---|---|---|---|---|---|
| 0 | 纯 PyTorch 手写模型 | 21.36 | 2.3410 | 21.36 ✅ | ❓ 只在简历 |
| 1 | 少用 `.item()` / `repeat_interleave` | 29.01 | 1.7234 | 29.01 ✅ | ❓ 只在简历 |
| 2 | RMSNorm kernel | 34.09 | 1.4665 | 34.09 ✅ | ❓ 只在简历 |
| 3 | SDPA | 36.36 | 1.3753 | 36.36 ✅ | 记录:1 |
| 4 | 手写 decode attention | 37.21 | 1.3437 | 37.21 ✅ | 记录:4 |
| 5 | CUDA Graph | 70.65 | 0.7078 | 70.64 ✅ | 记录:5 |
| 6 | prefill FA v1 | 78.83 | 0.5626 | **88.87** ⚠️ | 记录:6 |
| 7 | prefill FA v2 | 82.41 | 0.6068 | 82.40 ✅ | 记录:7 |

- 21.36 / 29.01 / 34.09 在仓库文本文件和整个 git 历史里都搜不到（`git grep` 只命中二进制的 ncu-rep 和 pdf）❓。它们都满足 tps = 50/耗时，🔎 说明用的是同一个"分子 50"的脚本口径，这又和"只测 decoding"矛盾 ⚠️。
- 第 6 行算不对：78.83 对应的耗时应是 0.6343 s，0.5626 s 对应的是 88.87 tok/s。记录原文就是这样，至少抄错了一个。
- 第 3–5 行是 3/25 以前测的，那时 prompt ×1（45 token）✅；第 7 行在 4/6 提交，prompt ×15（269 token）✅；第 6 行不知道用的哪个 ❓。**前后 prompt 长度不同，不是同条件对比。**
- `CUDA_LAUNCH_BLOCKING=1` 和第 7 行在同一个提交里加入，并且确认会生效 ✅：环境变量在 `import torch` 之后、第一次建 tensor 之前设置。小检查里，关掉时 host 循环 0.5 ms、等同步 130 ms；打开后 host 循环 135.7 ms、等同步 0.1 ms。所以第 7 行和 4/12 全部数据都是在"每次 launch 都等 GPU 执行完"的状态下测的，第 3–5 行不是。🔎 它对 eager 的 prefill、第 1 步、capture 影响大，对 replay 影响小。
- 第 3→4（+2.3%）、6→7（+4.5%）的提升，都小于同一配置重复测量的波动（§5.9：同配置 4 次在 81.36–87.20 之间，差 7%）⚠️。

### 5.2 纯 PyTorch 基线（21.36 ❓）
- 原版 RMSNorm：`x.float() → pow(2) → mean → +eps → rsqrt → mul → mul(weight) → type_as`。注释版还留在 `my_qwen2.py:59-70` ✅
- 原版 attention：`repeat_interleave(7)` 把 2 个 KV 头复制成 14 个，再做 matmul+softmax 或 SDPA。注释版留在 `:216-259` ✅
- 痕迹：`:211` 注释掉的 `cache_position[-1].item() + 1`，每层一次 `.item()` ✅

### 5.3 少用 `.item()` / `repeat_interleave`（→ 29.01 ❓）
- `.item()`：CPU 要等 GPU 把队列里的 kernel 全部跑完才能拿到值。每层一次就是每 token 24 次同步，CPU 下发和 GPU 执行没法流水线化。
- `repeat_interleave(7, dim=1)`：每层每步把 (1,2,L,64) 的 K、V 各复制成 (1,14,L,64)，是真实的显存拷贝，还多几次 kernel launch。
- 不拷贝的替代写法（注释里都有）：`expand` 视图（`:216-217`）；把 q reshape 成 (2,7,L,64)，与 k (2,1,L,64) 广播（`:229-241`）。

### 5.4 RMSNorm kernel（→ 34.09 ❓）
- 机制：eager 版有约 8 个小 kernel，每个都要走 Python → dispatcher → launch，单次 launch 在 CPU 上要几十 µs。每 token 有 49 次 RMSNorm（24×2+1）。手写版每次只 launch 一个 kernel。
- ⚠️ "190–400 µs → 40–50 µs"是 **CPU 时间线上 NVTX range `RMSNorm` 的长度**，不是 GPU 执行时间。GPU 上 `rms_kernel` 平均只要 2407.7 ns ≈ **2.4 µs**（`vllm比对.txt:15`）。这一步省的是下发开销，不是计算。
- 量级核对 🔎：按 NVTX 数字算，每 token 应省 49×(190−50) ≈ 7 ms 到 49×(400−40) ≈ 18 ms；实测 34.47 → 29.33 ms/token，只省了 5.1 ms，比估算小。原因之一：nsys 下 NVTX range 自身带开销，WSL 下开 nsys 吞吐会掉 16%（记录:2，36.78 → 30.94），range 时间偏大。
- 追问"GPU 上才 2.4 µs，为什么还值得写 kernel"：batch=1 decode 是 CPU bound，GPU 在等 CPU 下发，减少 launch 次数比加速 kernel 更有效。CUDA Graph 能翻倍也是同一个原因。

### 5.5 SDPA（→ 36.36 ✅ 记录:1）
- decode 走 `pytorch_flash::flash_fwd_kernel`（`vllm比对.txt:7`，1488 次 = 62 次 decode 前向 × 24 层 ✅）；prefill 走 mem-efficient 的 `fmha_cutlassF`（`:24`，72 次 = 3 次 prefill × 24 ✅）。
- 这份 nsys 里有 `rms_kernel` 和 SDPA 的 flash kernel，但没有手写 attention，🔎 对应的就是第 3 步。
- **能从 kern_sum 数出每次前向下发了什么** ✅（65 次前向 = 预热 10 + 测速 50 + 抓取 5；其中 3 次 prefill、62 次 decode）：
  - Linear（decode）：`gemv2T` 49 + `gemvx` 48 + `s16816gemm` 48 + `wmma` 24 = **169 次/前向 = 7×24 + 1**（q、k、v、o、gate、up、down，外加 lm_head）
  - Linear（prefill）：`sm80_xmma` 144+216 + `ampere` 72+72 = 504 = 3 × 168；lm_head 只对最后一个位置算，走 gemv（gemv2T 的 3041 = 62×49 + 3）
  - `rms_kernel` 3185 = 65 × 49；RoPE 的 `CatArrayBatchedCopy` 3120 = 65 × 48（每层 q、k 各一次）；KV 写入 `index_elementwise_kernel` 3120 = 65 × 48；`silu` 1560 = 65 × 24
- **GPU 时间大头是 Linear**：batch=1、seq=1 时 Linear 退化成矩阵乘向量，cuBLAS 选 gemv 系 kernel，`gemv2T` 占 39.2%，而手写 RMSNorm 只占 1.9% ⚠️。被问"GPU 侧下一步优化什么"时应该答 Linear / lm_head，而不是继续抠 RMSNorm。

### 5.6 手写 decode attention（→ 37.21 ✅ 记录:4）
- 只快了 2.3%，单次测量，在噪声范围内 ⚠️。它真正的价值是给 CUDA Graph 铺路：不再 `repeat_interleave`，不再按 L 切片，长度从 GPU buffer 读（§4.5）。
- kernel 细节见 §6.2。

### 5.7 CUDA Graph（→ 70.65 ✅ 记录:5，1.90×，即简历上的"提升一倍"）
- 机制：replay 时 host 只发一次调用，省掉每 token 数百次 Python / dispatcher / launch 开销（§5.5：仅 Linear 就 169 次，加上 RMSNorm、RoPE、KV 写入、残差等）。
- 为什么能翻倍 🔎（`vllm比对.txt` 估算）：整次运行 GPU kernel 总时间 = 155.94 ms / 39.2% ≈ 398 ms，分到 65 次前向约 **6.1 ms/次**；当时每 token 墙钟 ≈ 1/36.36 = **27.5 ms**。GPU 只忙了约 22%，瓶颈在 CPU。

### 5.8 prefill FA v1 → v2（→ 78.83 ⚠️ → 82.41）
- 两次都是单次测量，第 6 行本身数字对不上，前后还换了 prompt、加了 `CUDA_LAUNCH_BLOCKING`。
- 🔎 为什么这些增量不能归功于 FA：prefill 在窗口里只发生一次，替换掉的只是 24 层 prefill attention 和 KV 的 repeat 拷贝，而 SDPA 本身就是 flash / mem-efficient kernel。70.65 → 78.83 意味着 50 token 窗口少了 73 ms（按 0.5626 s 算是 145 ms），对一次 prefill 的 attention 来说太多了。同配置重复测量的波动约 ±20 ms，capture 这类一次性开销也在窗口里，单次测量的数不足以归因。
- 算法和 kernel 设计是真实工作，面试重点讲设计（§6.4、§6.5），不要讲"FA 带来了 x%"。

### 5.9 flash decoding 对比（4/12，未提交，记录:9-19）

| 配置 | 各次 tok/s | 去最高后均值（记录口径） | 中位数 | 全部均值 | 极差 |
|---|---|---|---|---|---|
| 1 按长度分派（>256 走 flash） | 82.04, 87.20, 82.03, 81.36 | 81.81 | 82.0 | 83.2 | 5.8 |
| 2 只用 flash decoding | 83.73, 81.85, 82.25, 86.16 | 82.61 | 83.0 | 83.5 | 4.3 |
| 3 只用 normal decode | 83.26, 86.64, 84.44, 88.96 | 84.78 | 85.5 | 85.8 | 5.7 |
| 1 + 共享内存 +2 | 83.79, 89.70, 84.94, 85.52, 84.05 | 84.57 | 84.9 | 85.6 | 5.9 |

1. **配置 1 和配置 2 跑的是同一套 kernel**（§4.6 问题 1）。81.81 和 82.61 的差就是测量噪声，约 0.8 tok/s；单个配置内部的极差有 4–6 tok/s ✅。
2. "去掉最高值"只剔除一侧，会把均值系统性压低。换成中位数或全部均值，排名不变：normal (3) ≳ +2 版 (4) > flash (1≈2)。
3. 配置 3 对配置 1：配置 3 的最小值 83.26，高于配置 1 除离群点外的所有值，算是有一定证据表明 ~300 长度下 normal decode 更快。但 n=4，窗口里还有 prefill 和 capture，只能说"倾向"。
4. "+2 消除 bank conflict"同时改了 FA v2 和 flash decoding 两个文件（4/12 18:04、18:07）。84.57 比 81.81 高 3.4%，和噪声一个量级，没法单独归因 ⚠️。
5. ⚠️ 记录里写的分析"flash decoding 要启动两个 kernel，launch 开销占比明显"，**在 CUDA Graph 下站不住**：replay 时 host 侧没有逐个 kernel 的 Python/dispatcher 开销，多一个 kernel 代价很小。🔎 从代码看更直接的原因（没测）：
   - flash decoding 的 grid 固定为 8 组 × 64 = 512 个位置，超过 seq_len 的位置也照样搬进共享内存、跑完 64 维循环（只是算的是 0）；normal decode 的 stride 循环到 `current_seq_len` 就停
   - flash decoding 的 v 加权求和是 64 次循环，每次都有 shuffle 规约和 2 次 `__syncthreads()`（`my_flash_decoding.cu:119-140`），每个 block 一共 **133 次** barrier；normal decode 整个 kernel 只有 **6 次**
   - 每次调用都要 `torch::zeros` 两个中间 tensor（`:258-259`），×24 层
6. ⚠️ 结论说"~300 长度下 normal 更快"，阈值却让 >256 的长度走 flash，自相矛盾。

---

## 6. 手写算子逐个拆

### 6.1 RMSNorm — `csrc/rmsnorm_kernel.cu`
- **launch**：grid(batch×seq, 1, 1)，block(896, 1, 1)。一个 token 一个 block，896 条 lane = 28 个 warp（`:64-65`）
- **步骤**
  1. `data = x*x/896`，在 float 寄存器里算：fp16 输入先转 float 避免 x² 溢出，先除 896 最后直接得到均值（`:15-17`）
  2. warp 内 5 级 shuffle 求和（16/8/4/2/1），lane 0 拿到本 warp 的和，存进 `sum[warp]`（`__shared__ float sum[28]`）→ barrier（`:19-25`）
  3. warp 0 的前 28 条 lane 读回 28 个 warp 和（其余 lane 补 0）→ 再做 5 级规约 → 存 `sum[0]` → barrier（`:27-41`）
  4. 所有 lane 读 `sum[0]`：`out = x · rsqrt(mean + eps) · weight[threadIdx.x]`（`:46-48`）
- **代价**：1 次 launch，2 次 barrier，10 级 shuffle
- **局限**：896 写死（`:17, 60`），只适用于这个模型

### 6.2 normal decode attention — `csrc/my_decode_attention.cu`
- **launch**：grid(14 头, batch)，block(512, 1)，512 lane = 16 warp。launch 时申请共享内存 `(cache_len+16)×4` 字节：前 cache_len 个 float 存每个位置的 qk，后 16 个 float 存每个 warp 的中间规约值（`:18-20, 169-175`）
- **一个 block 对应一个 query 头，512 条 lane 对应 cache 的 512 个位置**
  1. lane 0 从设备读 seq_len → barrier 共享（`:22-27`）
  2. GQA 映射：`head_group = blockIdx.x < 7 ? 0 : 1`。q 头 0–6 用 KV 头 0，7–13 用 KV 头 1，等价于 `repeat_interleave(7)`，但不拷贝数据（`:30-35`）
  3. stride 循环 `for(pos = threadIdx.x; pos < seq_len; pos += 512)`：block 宽 512 等于 cache 长度，所以每条 lane 最多算 1 个位置。64 维点积 × 0.125（= 1/√64），存 `qk_data[pos]`，顺便求 max（`:45-57`）
  4. max 两级规约：warp 内 5 级 → 存 `soft_max[warp]` → barrier → warp 0 的前 16 条 lane 读回 → 再 4 级 → `soft_max[0]` → barrier（`:59-77`）
  5. softmax 分母 Σ exp(qk − max)（减 max 防溢出），同样两级规约（`:82-102`）
  6. 输出：只有 `threadIdx.x < 64` 的 lane 干活。lane j 串行遍历所有位置，算 Σ exp(qk_i − max) · v_i[j] / 分母，写入 `out[头, j]`（`:131-143`）
- 超出 seq_len 的 lane 不进循环：max 初值 −FLT_MAX、sum 初值 0，不影响规约 ✅
- **代价**：1 次 launch，6 次 barrier；第 6 步是 64 条 lane 各自串行循环 seq_len（~300）次
- **共享内存访问**：lane l 写 `qk_data[32w + l]`，相邻 lane 是相邻的 float，bank 各不相同，没有冲突

### 6.3 flash decoding — `csrc/my_flash_decoding.cu`（未提交；只实现了 fp16 分支，`:278-295`）
**kernel 1 `my_decode_attention`**：grid(8, 14, batch)，block(64, 1)。一个 block = (KV 组 x∈0..7, 头 y, batch z)，每组 64 个位置，64 lane = 2 warp。
1. 读 seq_len（同上，`:30-35`）
2. 搬运，64 次循环 → barrier（`:55-72`）：
   - `k_smem[位置 i][维 x] ← k_cache`
   - `v_smem[维 i][位置 x] ← v_cache`，超出 seq_len 的写 0
   - K 和 V 在共享内存里的排布是转置的：K 按 [位置][维] 存，方便 lane=位置时读整行；V 按 [维][位置] 存，方便逐维循环时读一列
3. lane = 位置：`q · k_smem[x]` × 0.125 → 组内 max（warp 内 5 级，再经 `mim_data[2]` 跨 2 个 warp 合并）→ 写 `att_group_sum_max[..,1,组]`（`:74-109`）
4. 组内 softmax 分子 exp(qk − max_组)（`:112-117`）
5. 64 次循环（每一维 i）：exp · `v_smem[i][x]` → 两级规约 → lane i 留下结果 → 写 `att_group_data[..,组,维]`（`:119-143`）
6. 组内分母 Σexp → 写 `att_group_sum_max[..,0,组]`（`:148-163`）

**kernel 2 `atten_updata`**：grid(64, 14, batch)，block(8, 1)。一个 block = (维, 头)，8 条 lane = 8 个组（`:197-230`）。
- `real_max` = 8 个组 max 中的最大值（shuffle 规约 + `__shfl_sync` 广播）
- 各组的分子、分母都乘 exp(max_组 − real_max)，换到全局基准，再跨组求和：out = Σ分子 / Σ分母
- 数学：exp(qk − max_g) · exp(max_g − real_max) = exp(qk − real_max)，结果等价于全局 softmax ✅

思路是把 FlashAttention 的"分块 softmax + 基准重缩放"用到 decode 上：先组内规约，再用 log-sum-exp 式的重缩放把各组合并。
**代价**：2 次 launch；每 block 133 次 barrier；每次调用新建 2 个 float32 中间 tensor。

### 6.4 prefill FA v1 — `csrc/my_flash_attention.cu`（已不用）
- **launch**：grid(q 组数, 14, batch)，block(64, 16)。q 每 16 个位置一组；lane x∈0..63（每行 2 个 warp），y∈0..15 是组内的行（`:229-236`）
- **共享内存**：`q_mem[16][64]`、`k_mem[64][64]`、`v_mem[64][64]`、`mim_data[16][2]`
- **一个 block 的流程**
  1. 搬 q：lane (x,y) 存 q[行 y][维 x]（`:36-39`）
  2. 对每个 KV 组 i（每组 64 个位置；只遍历因果上可能看到的组，`:48`）：
     - 搬 K/V：每条 lane 写 4 次（行 y+16k，k=0..3）→ barrier（`:50-61`）
     - **lane 身份切换**：搬运时 x 表示维度，计算时 x 表示 key 位置（64 个 key 对应 64 条 lane）。lane (x,y) 算 q[y]·k[x]。因果条件：q 位置 `blockIdx.x·16 + y` ≥ key 位置 `i·64 + x`，不满足就置 −FLT_MAX（`:68-77`）
     - max：64 条 lane 分在 2 个 warp → warp 内 5 级 → 存 `mim_data[y][warp]` → barrier → 借 y=0 那一行的 32 条 lane，两两合并 16 行 → barrier → 每行读回（`:79-103`）
     - sum：同样的两级 + 2 次 barrier（`:108-133`）
     - 64 维循环：每一维都是"规约 → 存 → barrier → 跨 warp 合并 → barrier → lane j 取回"（`:136-181`）
     - 跨组流式合并：`max = fmax(本组 max, max_c)`（`:105`）；`attn_c = attn_c·exp(max_c − max) + attn`，`sum_c` 同理，然后 `max_c = max`（`:183-185`）
  3. `out = attn_c / (sum_c + 1e-6)`（`:189-190`）
- **痛点**：一个 qk 占一条 lane，64 个 key 跨 2 个 warp，每次规约都要"存共享内存 → barrier → 合并 → barrier"。每个 KV 组 1 + 2 + 2 + 64×2 = **133 次 barrier**

### 6.5 prefill FA v2 — `csrc/my_flash_attention_v2.cu`（在用）
- **launch**：grid(q 组数, 14, batch)，block(32, 32)。q 每 32 个位置一组；32×32 = 32 个 warp，每个 warp 就是一行 q（`:205-213`）
- **共享内存**：`q_mem[32][66]`、`k_mem[64][66]`、`v_mem[64][66]`（+2 是 4/12 加的，见 §6.6）
- **相对 v1 的改动**
  1. **每条 lane 搬两份**：lane x 同时搬维度 x 和 x+32，32 条 lane 就搬完 64 维（`:39-47`）。K/V 同理，每条 lane 写 8 次：2 行 × 2 维 × K、V（`:60-82`）
  2. **单 warp 执行 QK**：lane x 同时算 q·k[x] 和 q·k[x+32]，64 个 key 在一个 warp（32 条 lane）里算完（`:92-106`）
  3. 于是 max、sum 规约都在 warp 内完成：5 级 shuffle + `__shfl_sync` 广播，不经过共享内存，也不需要 barrier 合并（`:115-134`）
  4. 64 维循环：每维算 `qk1·v[x][j] + qk2·v[x+32][j]` → 5 级规约 + 广播 → lane j%32 取回，j<32 放 `attention_data`，否则放 `attention_data2`（`:137-153`）
  5. 跨组流式合并和 v1 相同，只是有两个累加寄存器（`:155-158`）；输出时每条 lane 写 2 个维度（`:162-166`）
- **效果**：每个 KV 组的 `__syncthreads()` 从 133 次降到 1 次（只剩搬运那次）；q 组从 16 变成 32，block 数减半
- 这就是简历上的"单 warp 执行 QK，优化了规约逻辑"✅

### 6.6 bank conflict 与 +2（简历："消除了 shared mem 的 bank conflict"）
规则和 trans.cu 里推过的一样：共享内存按 4 字节单元编号，**bank = 单元号 mod 32**；同一 wavefront 里如果两条 lane 落进同一个 bank，就要排队。
half 数组 `[R][C]` 的一行是 C 个 half = 2C 字节 = C/2 个单元。

**(a) 按行寻址的读：lane x 读 `k_mem[x][j]`（QK 点积和 v 加权求和都是这种）**
- 同一 wavefront（j 固定）里，相邻 lane 恰好隔一整行
- `[64][64]`：一行 32 个单元，32 mod 32 = 0，**32 条 lane 全部落进同一个 bank**，64 次点积循环的每一步都是 32 路冲突
- `[64][66]`：一行 33 个单元，33 和 32 互质，32 条 lane 落进 32 个不同的 bank，冲突消失 ✅
- 受影响的访问：FA v2 的 `k_mem[x]`、`k_mem[x+32]`、`v_mem[x]`、`v_mem[x+32]`（`:94, 102, 139`）；flash decoding 的 `k_smem[x]`（`:77`）

**(b) 按列寻址的写和读：搬运阶段 lane x 写 `k_mem[y][x]`；flash decoding 读 `v_smem[i][x]`**
- 相邻 lane 访问相邻的两个 half，而每 2 个 half 共用一个 4B 单元，所以 32 条 lane 落进 16 个 bank，两两冲突
- 这与一行有多宽无关，**+2 解决不了** ⚠️（trans.cu 里是靠列下标 ×2 解决的）
- `q_mem[y][j]` 是整个 warp 读同一个地址，+2 对它没有影响

**(c) float32 实例化**：FA v2 的 float 分支里，`[64][66]` 一行是 66 个单元，gcd(66, 32) = 2，落进 16 个 bank、两两冲突。比 `[64][64]` 的 32 路好，但没有消除。benchmark 只用 fp16。

**面试说法**：消除的是 QK 和加权求和这两个热循环里"按行读"造成的 32 路冲突；搬运阶段的两两冲突还在，可以再用列下标 ×2 去掉。实测提升（81.81 → 84.57）和噪声同一量级，没有单独做 A/B。

---

## 7. Profiling 与 nsys

- **代码里的 NVTX range**（`my_qwen2.py`）：`top_Model`、`DecoderLayer`、`Attention_generate`、`MLP`、`RMSNorm`、`RoPEfactorgenerate`
- **benchmark2.py**：每步一个 `record_function("{phase}_Step_{i}_{Prefill/Decode}")`，外加 `Warmup_Phase`、`Throughput_Phase`、`Profile_CUDAGraph`；第 3 阶段用 `emit_nvtx()` 给每个 aten op 打 range
  - 🔎 按 PyTorch 文档，`record_function` 只有在 profiler 或 `emit_nvtx` 开启时才会产生标记。第 1、2 阶段的 step 标签在 nsys 里看不到，能看到的只有 `nvtx.range_push`。所以 `benchmark2.py:54` 的注释"record_function 替代了之前没用的 nvtx.range_push"说反了
- **nsys 报告**：v2–v6（3/18–19）、WSL（3/20）、vLLM（3/21）、graph（3/24）、prefill FA（4/1），各阶段内容见 §1
- **nsys 自身开销**：WSL 下开 nsys 36.78 → 30.94 tok/s，掉 16%（记录:2）
- **`vllm比对.txt` 里能读出的结论**：§5.5 的 kernel 计数、§5.4 的 RMSNorm GPU 耗时、§5.7 的 GPU 空闲估算

---

## 8. 和 vLLM 的对比（158.53 tok/s，记录:3）

- 测试条件：仓库里没有 vLLM 的测试脚本，prompt、输出长度、计时口径都没有记录 ❓。vLLM 只能在 Linux 上跑，🔎 推测是在 WSL 里测的。
- nsys 里 vLLM 的 kernel（`vllm比对.txt:45` 起）：
  - `triton_poi_fused_mul_silu_slice`：torch.compile / Inductor 融合的 MLP 激活
  - `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt`：融合后的 RMSNorm
  - `flash_fwd_splitkv_kernel`：paged KV 上的 FlashAttention
  - `reshape_and_cache_flash_kernel`：写 KV cache
- ⚠️ vLLM 那张表里单个 kernel 最长 47 s（`:50`），显然把编译或等待算进去了，不能拿来逐个 kernel 比较。
- 差距怎么解释（诚实版）：
  1. 口径不同，不能直接比
  2. 🔎 按 §5.7，每次 decode 前向的 GPU kernel 时间约 6 ms（SDPA 阶段的数据），纯 replay 的理论上限约 160 tok/s 量级；自己的 70–85 tok/s 窗口里还有 prefill、eager 步、capture，以及每步 graph 外的小算子和同步。所以差距里很大一部分可能来自计时窗口和 graph 外开销，而不是 kernel 本身。**没测过，不要说成结论**
  3. vLLM 做了算子融合（Inductor 融合 RMSNorm 和 SiLU×up）、用了 FlashAttention kernel，也用了 CUDA Graph

---

## 9. 面试高危点和建议说法

| # | 容易说错的话 | 事实 | 建议说法 |
|---|---|---|---|
| 1 | "手写了 RoPE 算子" | RoPE 是 PyTorch 写的（`my_qwen2.py:117-127`），nsys 里是 cat 和 elementwise | "RoPE 用 PyTorch 实现，cos/sin 每次前向算一次、24 层共用；没做成 kernel，下一步可以和 KV 写入融合" |
| 2 | "decode 吞吐 21.36 → 82.41" | 计时窗口含 prefill、eager 步和 capture | "50 token 端到端生成吞吐，含一次 prefill"；被追问就主动讲窗口里有什么、应该怎么拆 |
| 3 | 从 21.36 讲起 | 21.36 / 29.01 / 34.09 仓库里没有记录 | 主讲有记录、机制清楚的 36.36 → 37.21 → 70.65；早期数字被问到就说是同一脚本早期测的；被问"只测 decoding"就承认当时理解有误，脚本其实包含 prefill |
| 4 | 报 78.83 tok/s（0.5626 s） | 两个数互相矛盾 | 只报 tok/s，或者直接说这条记录有抄写错误 |
| 5 | "RMSNorm 从 190–400 µs 降到 40–50 µs" | 那是 CPU 侧 NVTX range，GPU kernel 本身只要 2.4 µs | "省的是 CPU 下发开销：8 次 launch 变 1 次，batch=1 时瓶颈在 CPU" |
| 6 | "FA v2 带来了 x% 提升" | 单次测量，改动混杂，小于噪声 | 讲设计：单 warp QK、barrier 从 133 次降到 1 次；收益说"没做严格隔离测试" |
| 7 | "flash decoding 慢是因为 launch 开销" | graph 下不成立；配置 1 和 2 执行的代码相同 | 用 §5.9 第 5 点的解释，并说明是读代码推断、没测 |
| 8 | "消除了 bank conflict" | 只消除了按行读的 32 路冲突，搬运阶段两两冲突还在 | 用 §6.6 的说法 |
| 9 | 阈值 256 | 结论说 ~300 长度 normal 更快，阈值却让它走 flash；graph 下分支还会被固化 | 承认阈值没调；讲清 graph 固化 Python 分支这件事，本身就是加分项 |
| 10 | "4/12 的数据" | `seq_len_t` 被固化成 271，输出不正确（§4.6） | 如果主动讲，就当成"踩过的 CUDA Graph 坑"来讲：Python 常量的 `fill_` 被录进图 |
| 11 | "怎么验证手写 kernel 算得对" | 仓库里没有和 SDPA / HF 输出逐元素对比的测试（搜过 `allclose` 等，`test.py` 跑不通）❓ | 如实说当时靠看生成文本是否通顺；应该补上与 `F.scaled_dot_product_attention` 的 `allclose` 对比和 greedy 输出 token 对齐 |
| 12 | "零拷贝权重注入" | `load_state_dict` 是拷贝 | 不要这么说 |
| 13 | `CUDA_LAUNCH_BLOCKING=1` | 4/6 起一直开着，并且确实生效 | 被问就说"调 kernel 报错时开的，测速时应该关，它会让 eager 部分变慢" |
| 14 | 项目时间 | 简历 02–03，实际做到 4 月中 | 按实际说 |
| 15 | `main.py` 能正常聊天 | 只判断 151643 停止，而 Instruct 模型的轮次结束符是 151645；超过 512 会越界 | 小问题，被问到再说 |

---

## 10. 可能的追问和要点

- **为什么 batch=1 decode 是 CPU bound？** 0.5B 模型每个 kernel 在 GPU 上只要几 µs 到几十 µs，而 CPU 下发一个 op 要几十 µs；每 token 要下发几百个 op。数据：GPU kernel 约 6 ms/前向，墙钟 27.5 ms/token（§5.7）。
- **CUDA Graph 有什么限制？** 静态 shape 和地址；Python 分支、Python 标量在录制时固化；图里不能有 host 同步；显存来自图的私有内存池。本项目里 `seq_len_t.fill_(271)` 被录进图就是活例子。
- **softmax 为什么先减 max？** exp 在 float 下超过约 709 就溢出（fp16 超过约 11 就溢出），减掉 max 后指数都 ≤ 0。分块时每组各自减组内 max，合并时乘 exp(max_g − real_max) 换回全局基准。
- **FA 为什么省显存、为什么快？** 标准 FA：分块在 SRAM 里算，不存 L×L 矩阵，用 IO 感知减少 HBM 访问。你的版本：分块 + 流式 softmax 重缩放，只有前向，也不存 L×L 矩阵。
- **GQA 省了多少？** KV 头 14 → 2，KV cache 省 7 倍（每 token 84 KiB → 12 KiB）；kernel 里用 `head_group` 映射，零拷贝。
- **单 warp QK 为什么快？** 规约需要的数据都在一个 warp 里，用 shuffle 就能做完；跨 warp 就必须"存共享内存 → barrier → 再合并"，而 barrier 要等所有 warp 对齐。
- **如果继续做，下一步是什么？**
  1. 先把测量拆开：prefill、capture、replay 分别计时；每个配置跑多次取中位数；关掉 `CUDA_LAUNCH_BLOCKING`；每步检查输出 token 是否和 HF 一致
  2. 挪掉 graph 外的每步开销：`.any()` 同步（例如隔 N 步再检查 EOS）、`cat`、`arange`
  3. GPU 侧大头是 Linear：q/k/v 合成一个 Linear，gate/up 合并（vLLM 的 QKVParallelLinear / MergedColumnParallelLinear 就是这么做的）
  4. RoPE + KV 写入融合成一个 kernel
  5. 修掉 §4.6 的两个 capture 问题
