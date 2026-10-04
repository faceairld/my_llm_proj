"""把 sdpa_attention_with_kernel_seqlen_pad 整个函数体替换成 stub
   绝对不调用任何 GPU op,只返回干净数据
   回滚: 这是大改,需要从备份恢复;函数内联 ORIGINAL_BODY 注释里有原文

   用法(容器内): python3 /data/my_vllm_test/patches/bypass_sdpa_with_stub.py

   实验目的:
     - 让 sdpa_attention 变成纯黑盒,不做任何 attention 计算
     - 返回值用 query 自己 reshape 得到 (shape 对得上 + 值合法 + 同 dtype/device)
     - 不调用 varlen_pad / SDPA / varlen_unpad —— 完全跳过整条链

   预期:
     - 程序还崩 → bug 在我们函数外(QKV proj / O proj / FFN / layernorm 等)
     - 程序不崩(但输出乱码)→ bug 就在 sdpa_attention_with_kernel_seqlen_pad 内部
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

import re

with open(SRC) as f:
    text = f.read()

# 找完整函数:从 def sdpa_attention_with_kernel_seqlen_pad 开始,到下一个 def 之前
# 用正则匹配整段
pattern = re.compile(
    r"^def sdpa_attention_with_kernel_seqlen_pad\(.*?(?=^def )",
    re.DOTALL | re.MULTILINE
)

m = pattern.search(text)
if not m:
    print("ERROR: 找不到 sdpa_attention_with_kernel_seqlen_pad 函数")
    raise SystemExit(1)

ORIGINAL_BODY = m.group(0)
print(f"找到原函数,长度 {len(ORIGINAL_BODY)} 字符,从行 {text[:m.start()].count(chr(10))+1} 开始")

if "STUB-BYPASS 2026-05-20" in text:
    print("⚠ 已经 stub 过了")
    raise SystemExit(0)

STUB = '''def sdpa_attention_with_kernel_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens,
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """STUB-BYPASS 2026-05-20: 黑盒替换,完全跳过 varlen_pad/SDPA/varlen_unpad
       目的: 排除是否本函数内部触发的 GPU bug
       回滚: 从源码备份 /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py 恢复
            或参考原版同名函数(在 ROCm 后端 vllm/attention/backends/rocm_flash_attn.py 也有类似实现)

       返回: shape (sum_seq, h_q * d_q),用 query reshape 得到 —— 值合法,无 NaN,
            同 dtype/device,数量级跟真实 attention 输出相当(因为来自 Q 投影)
    """
    import sys as _stub_sys
    if not getattr(torch, '_stub_announced', False):
        print("[STUB-BYPASS] sdpa_attention_with_kernel_seqlen_pad 被 stub,完全跳过 attention 计算",
              file=_stub_sys.stderr, flush=True)
        torch._stub_announced = True

    sum_seq, h_q, d_q = query.shape
    # 注意 query 是已经验证过的干净数据(CKPT-2.7 实测 q_in=False)
    # reshape 后形状对得上 caller 期待的 (sum_seq, h_q * d_q)
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()


'''  # 结尾两个换行,让下一个 def 隔开

new_text = pattern.sub(STUB, text)

# 备份原 function body 到一个注释里(方便回滚)
# 找一个合适位置插入注释
# 简单做法:在文件顶部加注释,但要避开 BOM 等
# 这里就不内联回滚信息了,因为函数体太大;改用外部源码备份的方式
print(f"已替换为 stub")

with open(SRC, "w") as f:
    f.write(new_text)
print(f"✓ stub 写入完成")
print(f"  回滚: docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py gy_work:{SRC}")
