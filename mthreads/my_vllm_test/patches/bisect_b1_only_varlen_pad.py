"""B1 patch: 复原 sdpa_attention_with_kernel_seqlen_pad 的原始结构,
   但把 SDPA 和 varlen_fa_seqlen_unpad 两个 op 替换成 STUB,只保留 varlen_fa_seqlen_pad

   逻辑:
     原: alloc → varlen_pad → SDPA → unpad → return output.view(...)
     B1: alloc → varlen_pad → STUB(SDPA)→ STUB(unpad)→ return query.reshape(...)

   实验目的:看 varlen_fa_seqlen_pad 是否是单独的 OOB 元凶。
     - 跑完 4/4 OK → 排除 varlen_fa_seqlen_pad
     - 跑挂(MUSA error)→ 锁定 varlen_fa_seqlen_pad 是元凶

   回滚: docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py 回去
        会同时回滚 PATCH 1 (DECODER 快路径,无害的简化)
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# 当前应该是 STUB-BYPASS 状态(整个函数被替换成 STUB)
if "STUB-BYPASS 2026-05-20" not in text:
    print("ERROR: 当前不是 STUB 状态")
    raise SystemExit(1)

if "BISECT-B1 2026-05-20" in text:
    print("⚠ 已经是 B1 状态了")
    raise SystemExit(0)

# 锁定当前 STUB 函数全文
OLD_STUB = '''def sdpa_attention_with_kernel_seqlen_pad(
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
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()'''

# 新函数:复原原函数结构,但 SDPA + unpad 被 STUB 掉
NEW_B1 = '''def sdpa_attention_with_kernel_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens,
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """BISECT-B1 2026-05-20:
       复原原函数结构(alloc + varlen_pad),但把 SDPA + varlen_unpad 替换成 STUB
       只保留 varlen_fa_seqlen_pad 这一个 op,看它单独跑会不会触发 OOB

       结果判定:
         跑通(4/4 OK)→ varlen_fa_seqlen_pad 干净,bug 在 SDPA 或 unpad(下一步 B2/B3)
         崩(MUSA error)→ varlen_fa_seqlen_pad 是元凶
    """
    import sys as _b1_sys
    if not getattr(torch, '_b1_announced', False):
        print("[BISECT-B1] STUB SDPA + STUB varlen_unpad, only varlen_fa_seqlen_pad runs",
              file=_b1_sys.stderr, flush=True)
        torch._b1_announced = True

    bs = seq_lens.shape[0] - 1
    sum_seq, h_q, d_q = query.shape
    _, h_kv, d_kv = key.shape
    device, dtype = query.device, query.dtype

    # 原函数 alloc(跟原版一致)
    q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)

    # ★ 保留(就是要测它单独的影响)
    ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)

    # ===== STUB 替换 SDPA =====
    # 原本: attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
    #           q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    # 跳过

    # ===== STUB 替换 varlen_fa_seqlen_unpad =====
    # 原本: ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    # 跳过

    # 返回 STUB:用 query reshape 得到合法的 (sum_seq, h_q * d_q)
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()'''

if OLD_STUB not in text:
    print("ERROR: 找不到当前 STUB 函数")
    raise SystemExit(1)

text = text.replace(OLD_STUB, NEW_B1)
with open(SRC, "w") as f:
    f.write(text)
print("✓ B1 patch 写入完成(varlen_fa_seqlen_pad 保留,SDPA + unpad 替换成 STUB)")
