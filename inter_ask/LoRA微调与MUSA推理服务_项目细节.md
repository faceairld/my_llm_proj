# 摩尔线程 LoRA 微调 + MUSA 推理服务 —— 项目细节核实版

> 面向 2026-09-17 蔚来一面。所有结论标注出处：
> ✅ 核实（代码/git/对话记录里有原文） ⚠️ 冲突（简历与实际对不上） ❓ 找不到出处 🔎 推断
>
> 证据来源：
> - `E:\vscode\cuda_proj\proj_mor\`（projmor 仓库 + lora/ adapter 实物 + git log）
> - `mor_train\大模型_LoRA_推理框架_课程大作业_会话详细记录.docx`（训练侧 71 轮，下称**训练记录**）
> - `mor_train\Qwen3_MUSA_serve_会话完整记录.docx`（推理侧 13 轮，下称**推理记录**）
> - 简历：`高杨简历_infra.pdf`（2026-09-14，已提交）、`高杨简历2月.pdf`（2026-02-05，细节更多）

---

## 0. 一分钟口述版（背这个）

> 这是 GPU 并行编程课程的大作业，走的是摩尔线程赛道：在一台 MTT S4000（48GB 显存 / 100GB 内存，AutoDL 上租的 MUSA 容器）上，从数据构建 → LoRA 微调 → 推理服务部署跑通全流程。交付物是一个 FastAPI 服务，评测机断网、只暴露 `/predict` 和 `/` 两个端点，build 900s、health 180s、单次 predict 360s 超时。
>
> 知识库是《Programming Massively Parallel Processors》第 3 版，我按章节把书里的概念做成单轮问答对（Alpaca 格式，在 `dataset_info.json` 里注册成 `GPU_dataset`），规模从每章 30 条起步扩到几百条。训练用 LLaMA-Factory 做 LoRA，注入 q/k/v/o/gate/up/down 七类投影。
>
> 这个项目真正花时间的不是调参，是**国产卡的算子兼容性**：torch_musa 的 SDPA 要求传进来的 scale 和它自己算的 `1/sqrt(d_k)` 逐位相等，而 transformers 用 `head_dim**-0.5` 算，head_dim=128 时差 1 个 ULP 就直接 RuntimeError；Qwen3 在 MUSA 上 DynamicCache 追加 KV 时 `torch.cat(dim=-2)` 报 `Wrong Cat dim: 2`，把 KV cache 这条路径整个堵死。我用二分对照（1 token vs 16 token、cache on/off、pipeline vs 直接 generate）把崩溃点锁死在 cache 追加那一行，先关 cache 保证服务跑通，再从模型和 cache 实现上绕开。
>
> 最后因为 8B 在这张卡上只有 7 tok/s，远超评测 360s 限时，我把基座一路换到 1.5B，并且把服务从"逐条 generate"改成"一次请求整批 forward"，最终评测速度 1022.75 tok/s。

---

## 1. 这个项目到底是什么（背景 + 约束）

| 项 | 内容 | 出处 |
|---|---|---|
| 性质 | GPU 并行编程课程大作业，摩尔线程赛道 | ✅ 训练记录第 1 轮 |
| 硬件 | MTT S4000 48GB 显存、约 100GB 内存、15 CPU、50GB 磁盘 | ✅ 训练记录 + README |
| 环境 | Driver 2.7.0 / MUSA 3.1.0 / torch 2.2.0a0+git8ac9b20 + torch_musa 1.3.0 / transformers ≥4.51 | ✅ README |
| 交付契约 | `POST /predict {"prompt": str 或 List[str]}` → `{"response": ...}`；`GET /` 健康检查。**API 契约不可改** | ✅ README |
| 硬约束 | 评测机**断网**（`TRANSFORMERS_OFFLINE=1` + 权重必须提前下到本地目录）；容器实例不能再建容器，FakeDockerfile 的 EXPOSE/CMD 不许动；**不许替换任何 torch 相关包** | ✅ README + serve.py:5 |
| 超时 | build 900s / health 180s / predict 360s | ✅ README |
| 评分 | 准确率 + tokens/s（环境里装了 `rouge-score`） | ✅ README 包列表 + 训练记录第 6 步 |

**"为什么不直接上 vLLM"**：评测机是固定 conda 环境、不许动 torch 系的包；当时 vLLM-MUSA 的 `worker.py` 里还在直接调 `torch.cuda` 而报错（✅ 2 月简历记过这条）。所以最终只能用 `transformers` 手搭服务。

---

## 2. 时间线（以 git 实物为准）

| 时间 | 事件 | 出处 |
|---|---|---|
| 2025-12-04 | 模板阶段：基座从 opt-1.3b 换到 Qwen2.5-0.5B，测断网 | ✅ git |
| 12-07 ~ 12-11 | 调通模板 `serve.py` / `download_model.py` | ✅ git |
| 12 月中 | 数据构建 + 第一次 LoRA 训练（Qwen2.5-7B-Instruct） | ✅ 训练记录第 28 轮 |
| 12-24 20:24~22:41 | 换 **Qwen3-8B** + `AIRload/my_lora`（PeftModel、fp16、`device_map="musa"`） | ✅ git + adapter_config.json |
| 12-25 15:22 | 换 **Qwen3-1.7B** + `AIRload/1_7B_lora`（master 分支） | ✅ git |
| 12-25 16:49 | **最终版（HEAD，分支 ff）**：换成已 merge 好的 `raceven/PPMP3-Qwen2.5-1.5B-Instruct`，删掉 peft / fp16 / device_map | ✅ git + download_model.py |

> ⚠️ **简历写 2026.01–02，git 实物是 2025.12**。统一口径：主体在 2025 年 12 月完成，2026 年初做收尾和报告。

---

## 3. 数据：做了什么、为什么

- **来源**：《Programming Massively Parallel Processors》第 3 版，按章节抽概念做**单轮问答对**。✅ 训练记录里第 1 章的 30 条原文都在（例：Q"CPU 与 GPU 的设计目标有何本质差异" → A"CPU 低延迟导向，资源给复杂控制/乱序/大缓存；GPU 吞吐导向，资源给算术单元，用海量线程隐藏延迟"）。
- **格式**：Alpaca 单轮（instruction / input / output），在 LLaMA-Factory 的 `data/dataset_info.json` 注册成 `GPU_dataset`。✅
- **规模演进**：每章 30 条起步 → 200+ → 目标 400~600 条。✅ 训练记录
- ⚠️ 2 月简历写"1000 条问答对"，对话记录里的规划上限是 400~600。说"几百条量级"更安全。
- **为什么自己造数据**：这个领域没有现成中文问答集；而且要的是"基座见过但不会按这个口径答"，属于格式/口径对齐任务，LoRA 性价比最高，不需要全参微调。

---

## 4. 模型选型链：为什么换了四次（这是加分项）

| 版本 | 基座 | 为什么换掉 | 出处 |
|---|---|---|---|
| v0 | Qwen-7B-Chat（老版，自带 `modeling_qwen.py`） | 建模文件里无条件访问 `past_key_values[0][0].size(-2)`，该值为 None → AttributeError；改库文件太脏 | ✅ 训练记录 |
| v1 | Qwen2.5-7B-Instruct | 48GB 跑 7B LoRA 很宽裕，不必上 Int4/Int8；实测只用 17GB | ✅ 训练记录第 28 轮 |
| v2 | Qwen3-8B-Base（本地 `lora/` 就是这版 adapter） | 推理侧 `Wrong Cat dim: 2` + **7 tok/s**，360s 限时下答不完 | ✅ 推理记录第 9 轮 |
| v3 | Qwen3-1.7B | 仍是 Qwen3 系，`cache_position[-1]` IndexError 的老问题 | ✅ git 12-25 15:22 |
| **v4 最终** | **Qwen2.5-1.5B-Instruct**（LoRA 已 merge，传到 ModelScope `raceven/PPMP3-Qwen2.5-1.5B-Instruct`） | 定型 | ✅ download_model.py |

**口径**：缩小模型不是退让，是**固定评测口径下的显式权衡**——评分 = 准确率 × 速度，8B 单流 7 tok/s 意味着一条 200 token 的回答要 28 秒，360s 只能答十几条；换 1.5B 才能开批量把吞吐拉到三位数，准确率损失用领域 LoRA 补。

---

## 5. 训练配置：每个参数是什么、为什么这么配

### 5.1 实际有记录的配置

| 参数 | 实跑第一版（✅ 训练记录第 28 轮） | 本地 adapter 实物（✅ `lora/adapter_config.json`） | 简历写的 |
|---|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | Qwen3-8B-Base | "1.5B 基座" |
| LoRA rank r | 16 | **32** | 16 |
| lora_alpha | 16 | **16** | "scaling 32" |
| lora_dropout | 0.05 | **0.1** | 0.05 |
| target_modules | — | q/k/v/o/gate/up/down 全 7 类 | "32 个可训练层" |
| 学习率 | 1e-5 | — | 1.5e-4 |
| epoch | 3 | — | — |
| batch × 梯度累加 | 2 × 8 = **16** | — | 3 × 6 = **18** |
| 精度 | bf16 | — | — |
| val_size | 0.05 | — | — |
| 显存 / loss | 17GB，loss 2.69 → 2.4 | — | — |

adapter 文件 349,243,752 B（fp32）→ **≈ 8730 万可训练参数**，即 Qwen3-8B 的约 1.1%。✅ 由文件大小直接算得。

中途还讨论过两版更激进的方案（✅ 训练记录 38/39/50 轮）：V2 = lr 2e-5 / 4 epoch / batch4×accum4 / r32-alpha32 / val 0.1；激进版 = lr 3e-5 / 6 epoch / 有效 batch 32 / r64-alpha64；以及一版 CLI 预览 lr 1e-3 / 25 epoch / warmup_stable_decay / max_grad_norm 1.5 / r32-alpha16-dropout0.1，被判定太激进，收敛到 lr 1e-4 / 8 epoch / batch4×accum8。**本地 adapter 的 r32/alpha16/dropout0.1 正对应最后这一版。**

### 5.2 每个参数怎么解释（面试官一定挑一个问）

- **LoRA 原理**：`W' = W + BA`，A∈R^{r×d} 高斯初始化、B∈R^{d×r} **零初始化**，所以训练开始时增量为 0，等价原模型；前向再乘 `scaling = alpha / r`。
- **rank r**：增量矩阵的秩上限，决定"能学多少新东西"。r 越大容量越大、小数据越容易过拟合。几百条数据取 16~32 是常规区间。
- **alpha / scaling**：`alpha/r` 是增量放大系数。**r=32、alpha=16 → scaling=0.5**，等于把 LoRA 分支的贡献砍半，是小数据上的保守做法。若按简历读成 r=16、alpha=32 → scaling=2。⚠️ 两种读法差 4 倍，自己先定死说哪一种。
- **dropout**：只作用在 LoRA 分支的输入上，防止几百条数据过拟合。
- **target_modules = all**：attention 四投影 + MLP 三投影全注入。只注入 q/v 更省，但**领域知识主要存在 FFN 里**，所以要带上 MLP。
- **有效 batch = per_device_batch × grad_accum × GPU 数**：单卡 2×8=16。梯度累加是"用时间换显存"：拆成多个 micro-step 累积梯度再更新，数学上等价大 batch，显存峰值只按 micro-batch 算。
- **为什么 lr 这么小**：LoRA 参数少、梯度方差大，目标是口径对齐而非改写基座能力；太大直接灾难性遗忘。
- **warmup / max_grad_norm / 调度器**：warmup 防训练初期大梯度冲垮已收敛的基座；max_grad_norm 做梯度裁剪；cosine 是常规衰减；`warmup_stable_decay`(WSD) 是"预热-恒定-末段快速衰减"，适合后面可能补数据接着训的场景。
- **val_size 0.05**：留 5% 看 eval_loss 是否跟 train_loss 一起降——只看 train loss 2.69→2.4 判断不了过拟合。

### 5.3 ⚠️ 简历数字的风险点（必读）

1. **"32 个可训练层"**：Qwen2.5-1.5B / Qwen2.5-7B / Qwen3-1.7B 都是 **28 层**，Qwen3-8B 是 **36 层**，没有一个 32 层（32 层的是老 Qwen-7B-Chat）。
   → 安全说法：**"注入的是每层的 7 类线性层，不是 32 个 transformer block"**。**别硬说"1.5B 有 32 层"**，一查就穿。
2. **"lr 1.5e-4，有效 batch 18"**：对话记录里没有这一版原文（记录是 1e-5 / 16，以及后来建议的 1e-4 / 32）。2 月简历写的是"3(Batch)×6(Steps)=18"，说明确实跑过这一版，只是日志没留。
   → 被追问"这组超参怎么定的"，答**过程**而不是单点："先跑 lr 1e-5 / 有效 batch 16 / 3 epoch，loss 从 2.69 只降到 2.4，判断是欠拟合——数据量小、lr 偏保守，于是把 lr 提一个量级、epoch 拉长、batch 按显存能吃下的最大值往上顶。"
3. **"1.5B 基座 + rank 16"**：本地留存 adapter 是 Qwen3-8B 的 r=32。如果被要求看文件，直说"本地留的是中间那版 8B 的 adapter，最终提交是 merge 后上传的 1.5B"。

---

## 6. 推理服务：最终 `serve.py` 逐点意图

代码：`E:\vscode\cuda_proj\proj_mor\projmor\serve.py`（227 行，12-25 16:49 定型）

| 行 | 代码 | 意图 | 风险 |
|---|---|---|---|
| 1 | `import torch_musa` | 必须先于 transformers 导入，注册 musa 后端 | — |
| 5 | `TRANSFORMERS_OFFLINE=1` | 评测机断网，禁掉一切 HF 探活（否则 build 卡超时） | ✅ |
| 13-19 | `check_internet()` | 启动时打印连通性，确认确实离线 | — |
| 49/59-60 | `./local-model` + `from_pretrained` | 权重由 `download_model.py` 预先从 ModelScope 拉到本地 | ⚠️ **没指定 dtype**，走 config 的 `torch_dtype`（Qwen2.5-Instruct 是 bf16） |
| 63 | `model.config.use_cache = False` | Qwen3-8B 上为绕开 `Wrong Cat dim: 2` 加的，换 1.5B 后没删 | ⚠️⚠️ 与简历"配置 KV Cache 加速自回归解码"**直接矛盾** |
| 66-69 | `_attn_implementation = "eager"` | 绕开 torch_musa SDPA 的显式 scale 限制 | ✅ 代价是放弃 SDPA/融合路径 |
| 72-73 | `if hasattr(config,'attention_softmax_in_fp32'): = True` | 简历所说的"FP16 溢出乱码"修复 | ⚠️⚠️ **Qwen2Config 没有这个字段**（它属于 Falcon / GPT-BigCode 系），`hasattr` 为假 → 这两行在最终模型上**不执行** |
| 76-81 | `pipeline('text-generation', device='musa')` | — | — |
| 82 | `set_seed(42)` | 结果可复现（两次跑分要一致） | ✅ |
| 21-46 | `truncate_at_keywords(["问题：","回答：","Human","Assistant"])` | 截掉模型自问自答续写的下一轮 | ✅ |
| 99-129 | `process_single_prompt`：`f"Q: {p}\nA:"`、max_new_tokens=200、`do_sample=False` | 贪心解码（确定 + 省采样开销）；200 token 卡住长尾 | ✅ |
| 131-167 | **`process_batch_prompts_native`**：一次把 N 条 prompt 喂给 pipeline，`batch_size=len(prompts)` | **最终定型路径**，`/predict:215` 调的就是它 | ✅ |
| 169-191 | `process_batch_prompts_multithread`：ThreadPoolExecutor，max 64 worker | **对照组，留在代码里但没被调用** | ✅ 正是简历"对照实验"那句的物证 |
| 212-218 | 打印 `批量处理完成，耗时 X 秒，平均每个 Y 秒` | 唯一的本地计时点 | 🔎 1022.75 tok/s 更可能来自平台返回 |

另有 `musa_sdpa_patch.py`（✅ 实物）：monkey-patch `F.scaled_dot_product_attention`，在 `device.type == "musa"` 时强制 `kwargs["scale"] = 1/sqrt(query.size(-1))`。**但最终 `serve.py` 没 import 它**——最终版靠 eager 绕开。被问就照实说"两条路我都实现了，提交版用的是 eager"。

---

## 7. 做了哪些优化 + 为什么（面试主战场）

### 7.1 原生 batching vs 多线程并发（简历重点，**必背**）

**结论**：GPU 场景下多线程并发打同一个模型 ≈ 串行，原生 batching 才是真吞吐。四条理由：

1. **执行层面**：FastAPI 同步端点跑在 anyio 线程池里，N 个线程各自调 `generate`，但共享同一个 model、同一条 MUSA stream，kernel 仍然排队；叠加 Python GIL，前后处理也并行不了。✅ 推理记录第 5 轮专门为此加了 `gen_lock` 全局锁——**要加锁本身就说明那是伪并行**。
2. **算术强度层面（最硬的说法）**：decode 是 **memory-bound** 的——每生成一个 token 要把整套权重从显存读一遍，计算却只有一次 GEMV。batch=1 时算术强度 ≈ 2 FLOP/Byte，纯啃带宽；batch=N 时权重只读一次服务 N 条，GEMV 变 GEMM，算术强度线性涨到 ≈2N，吞吐几乎线性提升直到打满算力。多线程做不到这点，因为每个线程的 batch 还是 1。
3. **状态安全**：全局共享 model + KV cache 对象，多线程交错更新会互相污染（推理记录第 1 轮就把这列为 `Wrong Cat dim` 的候选根因之一）。
4. **评测口径**：评测机一次 POST 就带一批 prompt，本来就适合整批；多线程还多出线程切换和结果归并开销。

⚠️ **诚实边界**：仓库里**没有**这次对照实验的脚本和数据（❓ 无出处）。被追问数字就说：两种实现各提交跑过评测，native 整体耗时明显更低，具体数以平台返回为准；本地只有 `/predict` 里那句耗时打印。**别编两个具体 tok/s。**

### 7.2 其它优化清单

| 优化 | 机制 | 效果 |
|---|---|---|
| 基座瘦身 8B→1.5B | 参数量降 5.3×，decode 每 token 权重读取量同比下降 | 7 tok/s → 三位数量级（方向确定 ✅，具体倍数 🔎） |
| 原生 batching | 见上，GEMV→GEMM | 最终 1022.75 tok/s（❓ 只在简历里） |
| `max_new_tokens=200` | 截断长尾生成 | 保证单次 predict 不撞 360s |
| `do_sample=False` 贪心 | 省掉采样，结果确定 | 可复现 + 略快 |
| 关键词截断后处理 | 去掉"问题：/回答：/Human/Assistant"及 `Q:` 续写 | 直接影响准确率打分（多余续写拉低 ROUGE 类指标） |
| LoRA merge 后上传 | `merge_and_unload()`，推理期无 adapter 分支 | 省掉每层两次额外小 GEMM |
| `TRANSFORMERS_OFFLINE=1` + 本地权重 | 断网下零网络探活 | 避免 build/health 超时 |
| eager attention | 绕开 torch_musa SDPA 的 scale 硬约束 | 能跑；代价是放弃融合 kernel ⚠️ |

---

## 8. 遇到的问题与解决（**这个项目最值钱的部分**）

### 8.1 torch_musa SDPA 强制显式 scale ⭐

**现象**：`RuntimeError: Now torch_musa only allows explicit value of 'scale' parameter, whose value is equal to 1/sqrt(query.size(-1))`

**触发机制（2026-09-18 逐位核实）**：transformers **并没有漏传 scale**。
- `integrations/sdpa_attention.py` 调 SDPA 时传的是 `scale=scaling`，而 Qwen2/Qwen3 里 `self.scaling = self.head_dim**-0.5`（✅ 本地 transformers 源码 `modeling_qwen2.py:197`、`modeling_qwen3.py:232`；群聊截图里被注释掉的正是 `# scale=scaling,`）。
- 两种写法在 double 下**差 1 个 ULP**：head_dim=128 时 `128**-0.5 = 0.08838834764831845`，`1/sqrt(128) = 0.08838834764831843`（位模式 `…3bcd` vs `…3bcc`）。✅ 实算
- 群里流传的修法 `musa_scale_factor = 1.0 / (query.size(-1) ** 0.5)` 与 `1/math.sqrt(d)` **逐位相同**，改完就不报了 ✅。→ 推断 torch_musa 内部是拿传入的 double 和自己算的 `1/sqrt(d)` 做**精确 `==`**。🔎（没有 torch_musa 源码，但证据链闭合）
- 哪些 head_dim 中招：64 / 80 / 256 两种写法相等；96 / 112 / 128 不等 ✅。所以模板默认的 Qwen2.5-0.5B（head_dim 64）不触发，换到 7B / 8B / 1.5B（都是 128）才炸。
- float32 下两者相等 → 比较一定发生在 double 上（aten schema 里 `float? scale` 在 C++ 里就是 `optional<double>`）。
- 讽刺点：`128**-0.5` 反而是**更准**的那个（与真值差 6.0e-18，`1/sqrt` 差 7.8e-18，因为后者经过 sqrt、除法两次舍入）——torch_musa 拒绝的是正确舍入的值。

**三种解法及取舍**：

| 解法 | 做法 | 优点 | 代价 / 风险 | 实际用了吗 |
|---|---|---|---|---|
| 改 site-packages | `sdpa_attention.py` 里改成 `scale=musa_scale_factor` | 保住 SDPA 融合 kernel | 评测机环境不可改，交上去不生效 | 群聊里同学的方案；**没有证据你本人改过** |
| 运行时 monkey-patch | `musa_sdpa_patch.py` 包一层 `F.scaled_dot_product_attention` | 随代码走，保住融合路径 | 无条件覆盖 scale：遇到非默认 scale 的模型会**静默算错**；必须在任何 `from torch.nn.functional import scaled_dot_product_attention` 之前生效 | 文件写了，但**最终 serve.py 没 import，未生效** |
| eager | 训练 `--flash_attn disabled`（合法值 `auto/disabled/sdpa/fa2`，WebUI 没暴露，只能走 CLI）；推理 `_attn_implementation="eager"` | 根本不调 SDPA，最稳 | 显式物化 `[B,H,Lq,Lk]` 分数矩阵，O(L²) 显存，matmul/softmax/matmul 多个 kernel | ✅ 训练和最终提交都用这个 |

**monkey-patch 的正确写法**：只在 `scale is None` 或 `math.isclose(scale, ref, rel_tol=1e-6)` 时替换成 `ref = 1/math.sqrt(d)`，否则报错或走 eager —— 把"位模式不一致"修掉，但不把"真的不支持的 scale"变成静默错误。

**sitecustomize 为什么不可靠**：评测 CMD 固定为 `uvicorn serve:app`（FakeDockerfile 不许改）。console script 启动时工作目录不在 `sys.path` 里，sitecustomize 在解释器初始化阶段导入；uvicorn 是之后才把 app-dir 插进 `sys.path` → 项目目录下的 `sitecustomize.py` 不会被加载，除非设 `PYTHONPATH`。所以应在 `serve.py` 第一行显式 import 补丁。

> **口径**：这个报错表面像"缺参数"，实际是 torch_musa 对 scale 做 double 精确相等检查，而 transformers 用 `head_dim**-0.5`、torch_musa 用 `1/sqrt(d)`，两种数学等价的写法在 head_dim=128 时差 1 ULP。只支持默认 scale 可以理解（🔎 很可能 kernel 内部固定了缩放），但用精确 `==` 做校验是设计缺陷，应该用相对容差。

### 8.2 Qwen3 在 MUSA 上 KV cache 追加崩溃 ⭐⭐（最能讲的一个）

**现象**：
```
modeling_qwen3.py:210  key_states, value_states = past_key_values.update(...)
cache_utils.py:119     self.keys = torch.cat([self.keys, key_states], dim=-2)
RuntimeError: Wrong Cat dim: 2
```

**含义**：`self.keys` 与新的 `key_states` 除拼接维以外还有某一维对不上——即 **prefill 阶段产生的 K/V layout 和 decode 阶段不一致**（典型是某条路径返回了转置/非连续布局），DynamicCache 用 `torch.cat` 追加时直接拒绝。

**定位方法（要展示的排障方法论）**：
1. **二分生成长度**：`max_new_tokens=1` 不炸、`=16` 炸 → 崩在 **decode 追加**，不是 prefill。
2. **二分开关**：同一条 prompt，`use_cache=True` 炸 / `False` 不炸 → 坐实是 cache 路径。
3. **绕过封装**：怀疑 pipeline 没透传 kwargs，改成直接 `model.generate(..., use_cache=...)` 复测，排除"参数根本没生效"。
4. **手写 prefill+decode 两步**：先 `model(**inputs, use_cache=True)` 拿 `past_key_values`，再只喂最后一个 token + pkv 跑一步，让它必然走 update —— 把错误钉死在那一行。
5. **给 `DynamicCache.update` 打 hook**，打印 `prev_shape / new_shape`，看到底是 num_heads、head_dim 还是 seq 维对不上。

**踩的坑（真实且好讲）**：改成 `use_cache=False` 还报同样的错。两层原因——
- (a) `serve2.py` 里 `/predict` 被**定义了两遍**，FastAPI 命中先注册的那个，于是"改了不生效"；
- (b) 只改 `generate` 传参不够：**warmup 那次 `generate` 没传**，`model.config.use_cache` 和 `model.generation_config.use_cache` 也都要关。

> 教训：`generate` 的实际取值来自 **generation_config**，不是 `model.config`；三处不一致就会出现"我明明关了"的假象。另外 uvicorn 下 `print` 会被缓冲，要 `flush=True` 或用 `uvicorn.error` logger，否则会误判"代码没跑到"。

**两条出路**：
- 治标：全局关 cache（最终 `serve.py:63` 留下的就是这行）。代价是每个 token 重算全上下文注意力——**这正是 8B 只有 7 tok/s 的主因**。
- 治本方向：`generate(cache_implementation="static")`。StaticCache 预分配定长 KV buffer，用就地写入代替追加，**根本不调用 `torch.cat`**，既绕开崩溃又保住速度。

> 这里可以和自研 Qwen2.5 引擎那个项目呼应：**我自己那套引擎的 KV cache 就是静态预分配 `(1,2,512,64)` 的，正因为形状静态才能上 CUDA Graph**。动态 cat 除了兼容性坑，还会反复分配/拷贝显存。

### 8.3 其它问题（一句话一条）

| 问题 | 解决 |
|---|---|
| 老 Qwen-7B-Chat `modeling_qwen.py` 无条件访问 `past_key_values[0][0].size(-2)`，该值为 None → AttributeError | 换基座，不碰自定义建模文件 |
| Qwen3 生成时 `cache_position[-1]` IndexError（空张量取下标） | `--use_cache False` |
| 服务器连不上 HuggingFace | 全改走 ModelScope 本地快照路径 |
| WebUI 里 `max_grad_norm` 留空 → `float('') ValueError` | 填 1.0；顺带改用 CLI，拿到 WebUI 没暴露的参数（如 `--flash_attn`） |
| `dataset_info.json` JSON 语法错误 / nano 在错误目录建文件 | 校验 JSON，确认写在 LLaMA-Factory 的 `data/` 下 |
| 加载 adapter 报 `--adapter_path` 不认识 | 正确参数是 `--adapter_name_or_path` + `--finetuning_type lora` |
| `tokenizer.json` 报 `data did not match any variant of untagged enum ModelWrapper` | **tokenizers 版本太旧读不懂新格式**：训练环境和推理环境的 transformers/tokenizers 版本不一致，统一版本即可 |
| Gitee push 403、PR 冲突 | 权限/分支处理，最终走 `ff` 分支 |
| **serve.py 里 `PeftModel.from_pretrained` 被注释掉** → 服务实际跑的是基座，LoRA 完全没生效 | 代码审查时发现；最终版直接加载已 merge 的模型，从结构上消除这个隐患 |

> 最后这条如果被问"你怎么验证 LoRA 真的生效"，就讲它：**我踩过一次"服务跑的是基座"的坑，之后固定的验证方式是——同一条领域问题，加载前后输出必须有可见差异；再把 LoRA merge 进权重上传，从根上去掉"忘了加载"的可能。**

---

## 9. 结果：哪些数字有出处

| 数字 | 出处 | 可信度 |
|---|---|---|
| Qwen3-8B 在 S4000 上 **7 tok/s**（关 cache、eager、单流、FastAPI 单并发） | ✅ 推理记录第 9 轮你自己的原话 | 高，放心讲 |
| 训练显存 **17GB**（7B + LoRA r16 + bf16 + batch2） | ✅ 训练记录第 28 轮 | 高 |
| loss **2.69 → 2.40**（3 epoch） | ✅ 训练记录第 28 轮 | 高 |
| adapter **87.3M 可训练参数**（349,243,752 B ÷ 4） | ✅ 文件实物 | 高（仅对 8B 那版） |
| **1022.75 tok/s** | ❓ 只在两份简历里，仓库/日志/对话记录都没有 | 需准备口径 |
| **准确率 0.2325** | ❓ 只在 2 月简历 | 需准备口径 |

**1022.75 tok/s 被追问怎么答**（大概率会问，7 → 1022 差 146×）：
1. **口径**：这是**整批的总吞吐**（total decode tokens ÷ wall time），不是单流 decode 速度——评测一次发一批 prompt，N 条同时解码。
2. **来源**：由评测平台返回；本地对应的是 `/predict` 里那句 `批量处理完成，耗时 X 秒` 的打印。
3. **为什么能差两个量级**：模型 8B→1.5B（权重读取量 ×1/5.3）+ 单流→整批（GEMV→GEMM，把 decode 从纯 memory-bound 往算力方向挪）+ 8B 那次是**关着 cache** 的（每 token 重算全上下文）。三项相乘到三位数量级是合理的。
4. **千万别说"我用 profiler 测的"**——没有这个证据。

---

## 10. ⚠️ 简历 vs 实际：六个点 + 应对话术

| # | 简历原文 | 实际 | 怎么答 |
|---|---|---|---|
| 1 | "配置 KV Cache 加速自回归解码" | `serve.py:63` 是 `use_cache = False` | **最危险的一条。**照实说："KV cache 这块我花的时间最多，但结论是反的——Qwen3 在 MUSA 上 DynamicCache 的 `torch.cat` 追加直接崩，我定位到那一行之后，提交版本是关掉 cache 保证稳定跑通；正确的解法是换 StaticCache 预分配。简历那句写得不准确。" **主动承认比被戳穿强十倍，后半段还显得你懂得更深。** |
| 2 | "显式启用 `attention_softmax_in_fp32` 解决 FP16 精度溢出乱码" | 代码是 `hasattr` 保护调用，**Qwen2Config 没这个字段**，分支不执行；模型按 config 走 bf16，不是 fp16 | "这个开关是早期某版基座上加的，写成了 `hasattr` 保护；换到 Qwen2.5 后这个 config 字段不存在，实际没生效。**乱码真正的根因是后处理**——模型会自问自答续写下一轮，我是用关键词截断解决的。" 补充：切到 eager 后 softmax 本来就是 fp32（transformers `eager_attention_forward` 里 `softmax(..., dtype=torch.float32)`），所以"softmax 在 fp32 里算"这件事确实发生了，只是来自 eager 路径，不是那个 config 开关。 |
| 3 | "rank 16、scaling 32、dropout 0.05、32 个可训练层，lr 1.5e-4，有效 batch 18" | 记录里是 r16/alpha16/dropout0.05、lr 1e-5、有效 batch 16；本地 adapter 是 r32/alpha16/dropout0.1；28/36 层的模型没有 32 层 | 讲**迭代过程**而非单点数字（见 5.3）；"可训练层"改口径为"每层注入 7 类线性层"。 |
| 4 | "在摩尔线程 MS4000 上完成…全流程" | ✅ **成立**。AutoDL 上租的就是 MTT S4000 MUSA 容器（`root@autodl-container-*` + `device_map="musa"` + torch_musa 报错链，训练和推理都在这台） | 正常讲。注意卡名是 **MTT S4000**（简历写的 MS4000 是笔误）。 |
| 5 | "通过对照实验确认原生 batching 优于多线程并发" | 两个函数都在代码里、只调 native ✅；但**没有实验数据留存** | 讲机制（7.1 四条），数字说"以平台两次跑分为准"。 |
| 6 | 时间 2026.01–02 | git 实物 2025-12-04 ~ 12-25 | 统一成"2025 年 12 月主体完成，年初收尾"。 |

---

## 11. 高危追问速查

1. **"LoRA 为什么省显存？省在哪？"** → 省的不是激活，是**优化器状态和梯度**。Adam 每个可训练参数要存 fp32 的 m、v（+ master weight），全参微调 7B ≈ 7B×12B = 84GB 起步；LoRA 只有 8700 万参数进优化器，≈1GB。基座权重只读、不需要梯度。
2. **"那反向传播还要不要算基座的梯度？"** → 要算**激活的梯度**（链式法则必须穿过冻结层），但不算**权重的梯度**、不存优化器状态。很多人答错这点。
3. **"merge 之后推理为什么更快？"** → `W + BA` 预先加好就是一次普通 GEMM；不 merge 的话每个注入的线性层要多两次小 GEMM（x→A→B），28~36 层 × 7 个模块 = 上百次额外 kernel launch。
4. **"decode 为什么是 memory-bound？"** → 一次 decode 只算 1 个 token：GEMV，FLOPs ≈ 2×params，访存 ≈ params×dtype_size，算术强度 ≈ 2 FLOP/Byte；而现代卡的 ridge point 在几百 FLOP/Byte，所以完全卡在带宽。提 batch 是唯一的解。
5. **"用 batch 之后延迟怎么办？"** → 吞吐和 TTFT/TPOT 是权衡；真实服务用 continuous batching（vLLM 那套）在每个 decode step 动态插入/退出请求，而不是等齐整批。我这是离线批量评测，纯吞吐导向，定长整批就够。
6. **"padding 怎么处理？"** → 批内句长不一，左 padding + attention_mask；代码里 `pad_token_id = eos_token_id`（Qwen 没独立 pad token）。⚠️ **`process_batch_prompts_native` 没有显式设 `tokenizer.padding_side='left'`** —— 被问到就照实说"这是我当时没处理好的点，decoder-only 批量生成必须左 padding，否则右 padding 会让生成从 pad 之后接续"。
7. **"这个项目和你自研 Qwen2.5 引擎什么关系？"** → 这个是"用框架把整条链路跑通、在国产卡上啃兼容性"；那个是"自己写 kernel 把单模型推到极致"。前者让我知道 KV cache / batching / cache 布局在**工程上**怎么崩，后者让我知道它们在 **kernel 层**为什么这么设计。StaticCache 那条线正好把两者串起来。

---

## 12. 一句话收尾

> 这个项目最大的收获不是微调本身，而是**在一个算子支持不完整的国产后端上，怎么系统地把崩溃二分定位到具体一行**，以及**吞吐瓶颈到底在哪一层**——是框架封装、是 attention 回退、是 cache 被迫关掉，还是模型本身太大。这三个我都实测排除过一遍。
