# PyTorch 基础复习：结合 Attention 与 CUDA 项目

整理日期：2026-09-27。

本文整理本次会话讨论过的 PyTorch 基础，围绕 [attention.py](../pytorch_pratice/attention.py) 的编写过程，以及 RMSNorm/CUDA 优化中的疑问展开。本文没有修改练习代码。

说明：本会话此前调用的默认 Python 无法导入 torch，因此示例未做 PyTorch 数值运行验证。接口细节以安装的 PyTorch 版本及后端为准；以下 SDPA 示例要求支持 `enable_gqa`。公式使用纯文本。

## 阅读导航

- 创建 Tensor、大小写、dtype/device：第 1～2 节。
- Python 类、self、Parameter、buffer：第 3～4 节。
- shape/size、reshape/transpose、Linear：第 5～7 节。
- SDPA、GQA、因果 mask、KV cache：第 8～10 节。
- 自动求导、eval 与推理模式：第 11 节。
- CPU/GPU 执行、kernel 融合、torch.compile：第 12～13 节。
- 参考 Attention、代码复盘、速查表：第 14～16 节。

## 1. torch.Tensor 与 torch.tensor

### 1.1 大写是类，小写是创建函数

`torch.Tensor` 是张量类型，常用于类型注解和类型检查；`torch.tensor(data)` 根据已有数据创建张量。

```python
import torch

x = torch.tensor([1, 2, 3])
assert isinstance(x, torch.Tensor)

def identity(x: torch.Tensor) -> torch.Tensor:
    return x
```

类型注解不会创建张量，也不会自动转换输入的 dtype/device。`x: torch.tensor` 把创建函数当成类型，不是正确注解。

### 1.2 大小写的构造参数不是同一种含义

以下假定没有修改默认浮点类型：

| 写法 | 结果 |
|---|---|
| `torch.tensor([2, 3])` | 内容为2、3，shape=[2]，通常是int64 |
| `torch.tensor(2)` | 标量2，shape=[]，通常是int64 |
| `torch.tensor(2, 3)` | 错误，不支持用这两个位置参数指定形状 |
| `torch.Tensor([2, 3])` | 内容为2.0、3.0，使用默认浮点类型 |
| `torch.Tensor(2, 3)` | shape=[2,3]，内容未初始化 |
| `torch.Tensor(2)` | shape=[2]，内容未初始化，不是标量2 |

大写的旧式构造同时有“按数据创建”和“按尺寸分配”两类用法，容易混淆。日常建议把用途写明确：

```python
values = torch.tensor([2, 3], dtype=torch.float32)
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_values = torch.randn(2, 3)
uninitialized = torch.empty(2, 3)
```

`empty` 的内容未定义，不代表随机采样，也不能假定全零。[Tensor 文档](https://docs.pytorch.org/docs/2.14/tensors.html)

### 1.3 复制现有 Tensor 不要随手再包 torch.tensor

```python
y = torch.tensor(x)   # 复制并脱离原来的自动求导关系
y = x.clone()        # 复制；在启用梯度记录且输入需要梯度时保留求导关系
y = x.detach()       # 脱离求导关系，但共享底层存储
y = x.detach().clone()  # 独立存储，且不连接原计算图
```

普通赋值 `y = x` 不复制张量，两个变量引用同一个对象。`torch.as_tensor`、`from_numpy` 等可以在合适条件下共享数据，不能把所有创建接口都看成必定复制。[torch.tensor 文档](https://docs.pytorch.org/docs/2.14/generated/torch.tensor.html)

## 2. dtype、device 与模型迁移

### 2.1 明确创建类型和设备

```python
x = torch.randn(3, 10, 896, dtype=torch.float16, device="cuda")
positions = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")
```

- 普通浮点创建默认通常是float32，不会因某个 Linear 使用FP16就自动改成FP16。
- 索引一般使用整数类型，常见 `torch.long`。
- `torch.float32` / `torch.float` 是 dtype 对象；Python 的 `float` 是另一种类型，不要把它们与 `torch.Tensor` 混淆。
- 新建张量可用 `zeros_like(x)`、`x.new_zeros(...)` 等继承已有张量的类型/设备，减少不一致。

### 2.2 Tensor.to 与 Module.to

```python
x = x.to(device="cuda", dtype=torch.float16)
model = model.to(device="cuda", dtype=torch.float16)
```

`Tensor.to` 返回对应设备/类型的 Tensor；需要转换时一般得到新对象，不要只写 `x.to(...)` 却继续使用旧 x。无需转换时可能返回自身。

`Module.to` 会递归迁移注册的参数和 buffer，更新模块并返回自身。它不会自动迁移传入 forward 的 x，也不会把所有 Python 代码改成 GPU 代码。

`model.to("cuda")` 仅迁移设备，不代表所有浮点对象都变成FP16。若 Linear 创建为FP16、buffer默认FP32，仅迁移到CUDA之后仍然类型不同。

常规 Linear/SDPA 输入、权重和相关张量需要符合操作的 device/dtype 约束；自动混合精度是另外的执行机制，不能用来解释任意不匹配都可行。

## 3. Python 类、self 与局部变量

### 3.1 Python 使用冒号和缩进

```python
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 64
```

不能写 `class Attention(nn.Module) { ... }`。前面类定义语法错，编辑器可能在后面的 self 上连带标红，不一定是 self 本身有问题。

`self` 表示当前实例，是方法显式接收的第一个参数；调用 `model(x)` 时框架会把实例关联到方法，不用自己再传 self。子模块/参数注册前要先调用父类初始化。

### 3.2 局部变量与属性

```python
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        local_x = torch.zeros(64)  # __init__ 的局部变量
        self.saved_x = torch.zeros(64)  # 对象属性

    def forward(self, x):
        return x + self.saved_x
```

forward 不能直接访问另一次函数调用的局部 local_x。局部变量退出作用域后，若张量没有其他引用，则可释放；对象属性会随对象保留，直到被替换、移除或对象不再存活。闭包、返回值、计算图等也可能保留引用，所以不能机械地说函数返回就立即释放所有局部张量。

中间激活通常用局部变量：

```python
def forward(self, x):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    # q/k/v 无需写成 self.q/self.k/self.v
```

局部 Tensor 也能参与自动求导。把激活存在 self 上不是启用求导的条件，反而可能延长激活和计算图的存活时间。

## 4. 普通属性、Parameter、buffer、子模块

带 self 只说明它是对象属性，不代表它一定成为参数或 buffer。

| 写法 | 默认出现在 parameters() | 随 Module.to 迁移 | 默认保存到 state_dict |
|---|---|---|---|
| `x = torch.zeros(...)` | 否 | 否 | 否 |
| `self.x = torch.zeros(...)` | 否 | 否 | 否 |
| `self.w = nn.Parameter(...)` | 是 | 是 | 是 |
| `self.register_buffer("x", tensor)` | 否 | 是 | 是 |
| 同上，`persistent=False` | 否 | 是 | 否 |
| `self.proj = nn.Linear(...)` | 递归包含子层参数 | 是 | 包含子层注册状态 |

这里普通 Tensor 指常规 `torch.Tensor` 属性；显式 Parameter/Buffer 类型等属于注册机制，不要与普通属性混为一谈。

```python
class StateExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64))
        self.register_buffer("scale", torch.ones(64))
        self.register_buffer(
            "k_cache", torch.zeros(3, 2, 512, 64, dtype=torch.float16),
            persistent=False,
        )
```

Parameter 默认需要梯度，但冻结参数仍然是注册参数。普通 `requires_grad=True` Tensor 不会仅凭需要梯度就自动进入 `model.parameters()`。Buffer 用于非参数状态，不是“放在GPU的特殊内存类型”。

`persistent=False` 只影响是否进入 state_dict，不影响迁移或属性访问。注册名为 `k_cache` 后，通过 `self.k_cache` 使用。[nn.Module 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html)

## 5. shape、size、索引与解包

```python
x = torch.randn(3, 14, 128, 64)

x.shape       # torch.Size([3, 14, 128, 64])
x.size()      # 同上
x.shape[1]    # 14
x.size(1)     # 14
x.shape[-1]   # 64
x.size(-1)    # 64
x.shape[-2:]  # torch.Size([128, 64])

B, H, L, D = x.shape
```

shape 是属性，size 是方法。`x.shape(1)` 会把 torch.Size 当函数调用，错误；`x.shape[]` 也不是“获取全部”的写法，直接 `x.shape` 即可。

`shape` 是各维大小，`numel()` 是元素总数，`ndim` 是维数：上例分别是 `[3,14,128,64]`、`3*14*128*64`、4。

读取普通 GPU Tensor 的 shape 通常只访问 CPU 侧张量元数据，不需要读取 GPU 数据或启动 kernel。编译中的符号维度可能表示为 SymInt，但仍不能把它等同于从显存读取张量内容。

## 6. view、reshape、transpose 与连续性

### 6.1 改形状与交换维度不同

```python
q = torch.randn(3, 128, 896)
q = q.reshape(3, 128, 14, 64)  # 拆最后一个维度
q = q.transpose(1, 2)         # [3,14,128,64]
```

reshape 重新解释逻辑元素序列的分组；transpose 交换维度对应关系。把 `[B,L,H,D]` 直接 reshape 为 `[B,H,L,D]`，一般不能替代 transpose，虽然元素数量相同。

对 `[B,H,L,D]`：

```text
维度索引： 0  1  2  3
负数索引：-4 -3 -2 -1

transpose(-3,-2)：交换 H 和 L，合并heads前需要这个
transpose(-1,-2)：交换 L 和 D，不是同一件事
```

### 6.2 view 不一定可用，reshape 可能复制

- `view` 要求现有 stride 能表示目标形状，不复制数据。
- `reshape` 能用 view 时就用；否则可能复制。
- `transpose/permute` 对普通稠密张量通常生成共享存储的视图，不立即搬运所有元素。
- `contiguous()` 在需要时生成指定连续布局的副本；已经满足时可返回自身。

不是“所有非连续张量都不能 view”，而是要满足具体 stride 兼容条件。[Tensor Views 文档](https://docs.pytorch.org/docs/2.9/tensor_view.html)

合并 attention heads 的常见写法：

```python
out = out.transpose(1, 2).reshape(B, L, H * D)
# 或：out = out.transpose(1, 2).contiguous().view(B, L, H * D)
```

### 6.3 原练习里的动态拆分写法

```python
input_dim = input_data.shape[:-1]       # (B,L)
divide_dim = (*input_dim, -1, head_dim)  # (B,L,-1,D)
q = q_proj(input_data).view(divide_dim).transpose(-2, -3)
```

`*input_dim` 是 Python 解包；`-1` 让 PyTorch根据元素总数推导该维度。Q输出896、D=64时推导出14；K/V输出128时推导出2。一次 reshape/view 最多有一个待推导的 -1。

## 7. nn.Linear 的含义与形状

```python
proj = nn.Linear(in_features=896, out_features=128, bias=True)
```

它作用于输入最后一维，前面维度保留：`[B,L,896] → [B,L,128]`。不是只能输入二维张量，也不是要分别为每个batch创建一个Linear。

```text
proj.weight：[128,896]，PyTorch存储为[out,in]
proj.bias：[128]
数学操作：y = x @ weight.T + bias
```

`nn.Linear()` 不能缺少输入/输出维度。`self.q_proj` 是整个层，`self.q_proj.weight` 才是权重，所以 q_proj 比 q_weight 更准确地表达对象含义。

普通调用写 `model(x)`，由 Module 调用机制进入 forward；直接 `model.forward(x)` 会绕过部分 Module 调用功能，例如 hooks。

Qwen2.5-0.5B 相关形状：

| 层 | 输入→输出 |
|---|---|
| Q投影 | 896→14*64=896 |
| K投影 | 896→2*64=128 |
| V投影 | 896→2*64=128 |
| O投影 | 14*64=896→896 |

O的输入按Q head数确定，不是按KV head数。精确复现Qwen时还要对齐bias、RoPE等配置；本文不是完整模型实现。

## 8. scaled_dot_product_attention：输入与参数

```python
import torch.nn.functional as F

out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

它完成的是缩放点积attention，不包含QKV投影、RoPE、KV缓存管理和O投影。内部可能根据设备、类型、形状等选择融合实现或数学实现；不能因为名字是SDPA就认定必走FlashAttention。

### 8.1 标准四维布局

```text
Q：[B,Hq,L,D]
K：[B,Hkv,S,D]
V：[B,Hkv,S,Dv]
输出：[B,Hq,L,Dv]
```

Q/K最后一维相同，K/V序列长度S相同，L可以不同于S。数学接口允许Dv不同于D，但具体融合后端可能有额外约束。不要预先把K转成 `[B,H,D,S]` 再传入，函数处理内部的转置逻辑。

`[B,L,H,D]` 直接传入可能不报错，却把head数当成序列长度，计算错误。接口还可有额外batch维，这里只讲常用四维。

### 8.2 参数含义

| 参数 | 含义 |
|---|---|
| `attn_mask` | query-key可见性或加到score上的偏置，默认无自定义mask |
| `dropout_p` | softmax后attention权重的dropout概率 |
| `is_causal` | 是否使用接口定义的因果遮罩，默认False |
| `scale` | score乘法缩放，默认1/sqrt(Q最后一维)，显式传入时使用关键字 |
| `enable_gqa` | 启用多个Q head共享KV head的对应关系，依赖版本/后端支持 |

布尔attn_mask中True表示允许，False表示屏蔽；浮点mask直接加到score上，0不改变，负无穷屏蔽。mask形状需能广播到attention权重形状，例如共享的[L,S]，或按请求区分的[B,1,L,S]。不要把[B,S]默认当作能正确广播到[B,H,L,S]的padding mask。

本接口的布尔mask语义与 `nn.MultiheadAttention` 的 `key_padding_mask` 不同。按文档约定不同时使用自定义attn_mask与is_causal=True；需要组合限制时自己构造综合mask。[SDPA 参数文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 9. GQA 与因果 mask 是两件独立的事

### 9.1 GQA 控制 head 对应关系

```text
14个Q head、2个KV head：
Q head 0～6   → KV head 0
Q head 7～13  → KV head 1
```

这由 `enable_gqa=True` 表达。要求Hq能被Hkv整除，K/V head数匹配，并满足后端支持。若已经手动按组扩展K/V head，则不必再做重复处理。

**GQA不要求额外的因果mask。** 它处理head关系，mask处理token位置的可见性。

### 9.2 完整 prefill 的 causal

当Q/K从同一位置开始、长度相同：

```text
         key0 key1 key2 key3
query0    ✓    ×    ×    ×
query1    ✓    ✓    ×    ×
query2    ✓    ✓    ✓    ×
query3    ✓    ✓    ✓    ✓
```

设 `is_causal=True` 就能表达这种规则，无需显式创建三角Tensor。没有mask参数不代表没有遮罩逻辑；只传QKV且保持默认False则没有这种因果限制。

### 9.3 单 token decode 为什么常用 False

```text
Q：当前最新1个token
KV：有效历史 + 当前token，不含未来位置和padding
```

当前Q应看到所有这些KV，因此可用is_causal=False。SDPA在L≠S时的True采用左上对齐因果偏置，L=1时会只允许第一个key，不会自动识别Q实际处于历史末尾。

### 9.4 什么时候一次会有多个新 token

普通自回归decode对每条请求一次输入1个token；多个请求同时decode是增大B，而不是增大每条请求的L。带历史的一段chunked prefill、投机验证或追加用户输入才可能让L>1。

```text
已有历史：旧0、旧1、旧2
本次输入：A、B

query A：能看旧0/旧1/旧2/A，不能看B
query B：能看旧0/旧1/旧2/A/B
```

这时不能只因Q/K长度不同就把is_causal设False。连续缓存下可按绝对位置构造：

```python
q_pos = start_pos + torch.arange(L, device=q.device)
k_pos = torch.arange(start_pos + L, device=q.device)
mask = k_pos[None, :] <= q_pos[:, None]
out = F.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, is_causal=False,
    dropout_p=0.0, enable_gqa=True,
)
```

这个mask针对连续有效缓存；padding、不同请求位置等需另行表达。上面的因果对齐规则依据同一 [SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 10. KV cache 的基础读写

### 10.1 容量、实际batch、本次长度、累计长度分开

```text
cache容量：[max_batch,Hkv,max_seq,D]
当前输入：[B,L,hidden]
本次写入位置：start_pos 到 start_pos+L-1
累计有效长度：kv_len = start_pos+L
```

连续写入示意：

```python
self.k_cache[:B, :, start_pos:kv_len, :] = k
self.v_cache[:B, :, start_pos:kv_len, :] = v
k_valid = self.k_cache[:B, :, :kv_len, :]
v_valid = self.v_cache[:B, :, :kv_len, :]
```

必须检查容量，且假定此前 `[0:start_pos]` 已正确写入，同一个batch槽仍对应同一条请求。

### 10.2 cache_pos 的长度不等于累计KV长度

```text
prefill 10个：cache_pos=[0,...,9]，kv_len=10
decode 一个：cache_pos=[10]，kv_len=11
再decode一个：cache_pos=[11]，kv_len=12
```

`cache_pos.shape[-1]` 在decode时是1，不是11或12。若用张量索引写入，cache_pos采用与本次L对应的一维整数位置；不同请求使用不同位置时，需要重新设计索引，不能照搬单个公共cache_pos。

### 10.3 cache 与模型生命周期

cache适合注册为 `persistent=False` buffer，使用与K/V一致的dtype/device。注册buffer不等于自动选对dtype。

新请求必须重置或重新定义有效长度，不能把上一请求的KV当作新请求历史。未使用位置可以保留旧物理数据，但必须严格排除在有效切片/mask外。不能让不同请求并发共享同一组可变槽位而没有管理。

这是简单推理缓存，不是vLLM的paged KV管理，也不是训练时自动求导缓存。训练与带原位更新的推理缓存要分开设计。

## 11. 自动求导、eval、no_grad 与 inference_mode

局部变量与self属性都可以参与求导，关键是张量的requires_grad和执行时是否记录梯度。

| 机制 | 作用 |
|---|---|
| `model.eval()` | 切换Dropout/BatchNorm等模块的训练/评估行为，不自动关闭梯度 |
| `torch.no_grad()` | 在作用域内不记录通常的反向图 |
| `torch.inference_mode()` | 更强的推理上下文，减少包括版本追踪等开销，也有更严格的后续autograd使用限制 |

```python
model.eval()
with torch.inference_mode():
    y = model(x)
```

`eval()`与推理上下文承担不同职责，不能互相替代。`inference_mode`也不会自动把model设置为eval，且不是所有之后要参与求导的代码都适合使用。[inference_mode 文档](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad_mode.inference_mode.html)

SDPA是函数，按显式dropout_p执行，不自动读取外层Module.training。常见写法：

```python
dropout_p = self.p if self.training else 0.0
```

单纯推理可以直接设0.0。`detach()`也不复制存储，原位修改共享数据仍可能影响另一引用，不等于安全的深拷贝。

## 12. 哪些在 CPU，哪些在 GPU

以一个GPU输入为例：

```python
N = x.shape[-1]      # 普通eager下读取主机侧形状元数据
inv_N = 1.0 / N      # 普通Python标量计算
mean_sq = x.square().sum(dim=-1, keepdim=True) * inv_N
inv_rms = torch.rsqrt(mean_sq + eps)
y = x * inv_rms
```

- `N`与`inv_N`不因为model.to("cuda")就变成GPU Tensor。
- `mean_sq`、`inv_rms`来自GPU Tensor数据，相关计算在GPU上执行；“每行只有一个值”也不意味着自动在CPU算。
- Python标量通常作为运算参数参与GPU操作，不要求先创建一个显存Tensor再单独复制。
- CPU确实需要准备参数并提交工作，但读取shape不需要GPU先返回数据；这与 `x.sum().item()` 需要把计算结果变成Python值不同，后者通常带来主机等待。

同一stream按顺序保证依赖，通常不必每一步调用CPU侧synchronize。跨stream的数据依赖则需要正确事件/等待机制。

CUDA kernel对主机通常异步提交，所以纯CPU秒表围住一个GPU调用，可能主要测到提交开销。测GPU耗时要正确使用CUDA Event或同步边界，并预热；`.item()`、打印GPU数据等可能改变时序。[CUDA 执行语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)

## 13. PyTorch kernel 发射、RMSNorm 与 torch.compile

### 13.1 不能按数学符号数 kernel

eager下拆写square、mean、rsqrt、mul等通常涉及多个操作和中间Tensor，但一个PyTorch算子可能调用一个、多个或无需GPU kernel。库算子内部也可能融合。

```python
N = x.shape[-1]
scale = 1.0 / N
```

这两行不是GPU kernel。`x * scale`却是对Tensor元素做运算，通常需要设备执行，而不是把整个x搬回CPU乘。

### 13.2 手写 RMSNorm 为什么有意义

逻辑计算是：

```text
每行平方和 → 除以N → 加eps → 倒平方根 → 缩放x → 乘weight
```

合适的融合kernel可以在一次发射中组织行归约和逐元素输出，减少中间结果写回与重复提交。一个kernel内部仍然可以有多个阶段、归约和同步；不等于算法只遍历一次，也不保证任何尺寸都只读一次x。是否把整行留在寄存器，要看行长度和资源压力。

面试中不要只说“原实现至少两个kernel、我变成一个”却没有trace。应给出具体eager表达式、实际发射情况和对照耗时；收益可能来自少launch，也可能来自少访存，两者要分开验证。

### 13.3 PyTorch 可以编译融合

```python
def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (xf * inv_rms).to(x.dtype)
    return normalized * weight

compiled_rmsnorm = torch.compile(rmsnorm)
```

这是教学表达，精度/转换位置需与目标模型对齐。compile可能融合逐元素与归约等操作，减少中间内存流量；不保证整个函数变成一个kernel。首次编译、形状变化、graph break和后端支持都会影响表现。[torch.compile 教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial)

| 优化 | 主要做什么 |
|---|---|
| 算子/kernel融合 | 在较少kernel中完成多个步骤，可能减少中间访存 |
| CUDA Graph | 记录并重放提交序列，主要减少主机提交开销；不自动把所有kernel融合 |
| torch.compile | 捕获/优化计算并生成或调用实现；某些模式还可结合CUDA Graph |

比较手写kernel时，应区分eager、compiled、库原生实现，并对齐dtype、shape、eps和计时方式。compiled结果需要预热后测稳态，不把编译时间混进单次kernel收益。[torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile)

## 14. 参考实现：完整 prefill + 单 token decode

下面是组合前述知识的独立教学示例，不替换练习文件，不是完整Qwen实现：没有RoPE、预训练权重、padding或paged cache。所有batch槽共用start_pos，调用者必须顺序续写同一批请求，不能跳过历史位置。新会话从start_pos=0开始。只用于推理，在inference_mode下调用。

```python
import torch
from torch import nn
from torch.nn import functional as F


class AttentionExample(nn.Module):
    def __init__(self, max_batch=3, max_seq=512, dtype=torch.float32):
        super().__init__()
        self.hq = 14
        self.hkv = 2
        self.d = 64
        self.hidden = self.hq * self.d
        self.max_batch = max_batch
        self.max_seq = max_seq

        self.q_proj = nn.Linear(self.hidden, self.hq * self.d, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden, self.hkv * self.d, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden, self.hkv * self.d, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden, self.hidden, bias=False, dtype=dtype)

        shape = (max_batch, self.hkv, max_seq, self.d)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, L, H = x.shape
        end_pos = start_pos + L
        if H != self.hidden or not (0 < B <= self.max_batch):
            raise ValueError("输入 hidden 或 batch 大小错误")
        if L <= 0 or start_pos < 0 or end_pos > self.max_seq:
            raise ValueError("输入长度或缓存位置超出范围")
        if start_pos > 0 and L != 1:
            raise ValueError("此示例的续写只支持单 token decode")

        q = self.q_proj(x).reshape(B, L, self.hq, self.d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.hkv, self.d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.hkv, self.d).transpose(1, 2)

        self.k_cache[:B, :, start_pos:end_pos, :] = k
        self.v_cache[:B, :, start_pos:end_pos, :] = v
        k_valid = self.k_cache[:B, :, :end_pos, :]
        v_valid = self.v_cache[:B, :, :end_pos, :]

        out = F.scaled_dot_product_attention(
            q, k_valid, v_valid,
            dropout_p=0.0,
            is_causal=(start_pos == 0),
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(B, L, self.hidden)
        return self.o_proj(out)


# 要求本地有CUDA及支持该SDPA调用的PyTorch环境。
model = AttentionExample(dtype=torch.float16).to("cuda").eval()
with torch.inference_mode():
    prompt = torch.randn(2, 10, 896, device="cuda", dtype=torch.float16)
    prefill_out = model(prompt, start_pos=0)  # [2,10,896]
    new_token_hidden = torch.randn(2, 1, 896, device="cuda", dtype=torch.float16)
    decode_out = model(new_token_hidden, start_pos=10)  # [2,1,896]
```

示例输入是随机hidden states，展示维度与缓存调用，不是完整token生成流程。代码中容量检查不等于验证历史KV已正确生成，这仍是调用约定。

## 15. attention.py 修改过程中的问题复盘

截至本文读取的文件快照，已修正：Python class语法、hidden_size按Q heads计算、O输入维度、GQA开关、输出转置、cache注册buffer。

仍需注意（文件之后可能由用户继续修改）：

| 位置/问题 | 为什么 | 应怎样处理 |
|---|---|---|
| cache用默认zeros | FP32，Linear显式FP16 | 创建时对齐dtype，或迁移模型时显式统一浮点dtype |
| cache读写使用完整batch维 | 容量3不等于实际B | 使用 `:B` |
| 注解写torch.tensor | 它是函数，不是类型 | 改torch.Tensor |
| seq_len意义 | 需要累计KV有效长度，不是当前L | 明确命名kv_len并按调用更新 |
| is_causal按Q/K长度是否相等判断 | 只覆盖约定的完整prefill+单token decode | 限制范围，或支持偏移mask |
| O投影bias=True | 基础练习可行，但不是精确Qwen配置 | 精确复现时对齐bias=False |
| 尚无RoPE | 与Qwen真实attention有差别 | 学会基础流程后再加入，并使用正确位置 |

历史上错误的 `transpose(-1,-2)` 会交换L和D；现在 `transpose(-3,-2)` 对四维输出交换H和L，方向正确。reshape能成功不代表语义正确，要检查每个维度代表什么。

## 16. 常见问题速查

| 问题 | 回答 |
|---|---|
| 什么时候写Tensor？ | 类型注解、类型判断；创建时优先用明确的工厂函数 |
| tensor(2,3)是2×3吗？ | 不是，会报错；用zeros/randn/empty |
| shape全取需要[]吗？ | 不需要，直接x.shape或x.size() |
| self意味着是训练参数吗？ | 不意味着；要看Parameter/Module/buffer注册 |
| 局部变量能求导吗？ | 能，是否记录图与self无关 |
| model.cuda会搬所有self里的Tensor吗？ | 只自动处理注册状态，普通Tensor属性不会自动迁移 |
| model.eval关闭梯度吗？ | 不关闭；推理通常另加inference_mode/no_grad |
| transpose一定搬数据吗？ | 普通稠密Tensor通常是视图；之后reshape可能复制 |
| GQA为什么要mask？ | GQA本身不要求；mask来自位置/因果/padding限制 |
| decode不是一次一个吗？ | 普通每请求一次一个；多个请求增大B而非L |
| 有cache时is_causal一定True吗？ | 不；单token应看全部有效历史，非方形左上因果会错位 |
| 只设dtype就上GPU了吗？ | 没有，dtype和device独立 |
| 每个数学运算都发一个kernel吗？ | 不是；元数据/标量、融合和库内部路径要区分 |
| compile等于CUDA Graph吗？ | 不等于，优化层次不同，可以配合 |
| 手写RMSNorm一定胜过PyTorch吗？ | 要与实际eager/compiled/库实现公平对照 |

相关专题：[Tensor Core 使用细节](Tensor_Core使用细节_cuBLASLt与WMMA.md)、[多框架多模型性能基准测试项目细节](多框架多模型性能基准测试体系搭建_项目细节.md)。
