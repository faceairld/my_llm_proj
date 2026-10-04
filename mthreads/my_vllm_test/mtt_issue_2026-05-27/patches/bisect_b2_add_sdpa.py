"""B2 patch: 在 B1 基础上,把 SDPA 加回来,只剩 unpad 还是 STUB
   实验目的: 看 SDPA 是不是 OOB 元凶
     B2 4/4 OK → SDPA 干净,元凶是 varlen_unpad(继续 B3 确认)
     B2 崩 → SDPA 就是元凶
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

if "BISECT-B1 2026-05-20" not in text:
    print("ERROR: 当前不是 B1 状态,先把 B1 装好")
    raise SystemExit(1)

if "BISECT-B2 2026-05-20" in text:
    print("⚠ 已经是 B2 了")
    raise SystemExit(0)

OLD = '''def sdpa_attention_with_kernel_seqlen_pad(
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

NEW = '''def sdpa_attention_with_kernel_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens,
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """BISECT-B2 2026-05-20:
       在 B1 基础上,把 SDPA 加回来;只剩 varlen_unpad 还是 STUB
       保留 varlen_fa_seqlen_pad + SDPA, STUB unpad

       结果判定:
         跑通(4/4 OK)→ SDPA 也干净,bug 在 varlen_fa_seqlen_unpad(继续 B3 验证)
         崩(MUSA error)→ SDPA 就是元凶
    """
    import sys as _b2_sys
    if not getattr(torch, '_b2_announced', False):
        print("[BISECT-B2] varlen_fa_seqlen_pad + SDPA enabled, varlen_unpad still STUB",
              file=_b2_sys.stderr, flush=True)
        torch._b2_announced = True

    bs = seq_lens.shape[0] - 1
    sum_seq, h_q, d_q = query.shape
    _, h_kv, d_kv = key.shape
    device, dtype = query.device, query.dtype

    q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)

    # ★ 保留
    ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)

    # ★ 加回(B2 新增 — 注意: q_pad/k_pad/v_pad 的 padding 区是 alloc 残留 NaN,跟原函数情况一致)
    attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
        q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)

    # ===== STUB 替换 varlen_fa_seqlen_unpad =====
    # 原本: ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    # 跳过

    # 返回 STUB:用 query reshape 得到合法的 (sum_seq, h_q * d_q)
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()'''

if OLD not in text:
    print("ERROR: 找不到 B1 函数")
    raise SystemExit(1)

text = text.replace(OLD, NEW)
with open(SRC, "w") as f:
    f.write(text)
print("✓ B2 patch 写入完成(varlen_pad + SDPA 都跑,unpad 仍 STUB)")
