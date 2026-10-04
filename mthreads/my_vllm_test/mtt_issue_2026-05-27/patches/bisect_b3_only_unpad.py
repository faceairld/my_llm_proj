"""B3 patch: 只跑 varlen_fa_seqlen_unpad,STUB 掉 varlen_pad 和 SDPA
   实验目的: 最终确认 unpad 单独就能触发 OOB
     - B3 跑通 → unpad 也清白,问题在三个 op 的某种组合(罕见但需重新查)
     - B3 崩 → unpad 是元凶,bisection 100% 完成

   实现细节:
     - unpad 需要 attn_out 输入(原本从 SDPA 出),我们 alloc 一个空的 torch.empty
       (shape 跟 SDPA 输出一致,内容是脏内存)
     - unpad 用这个脏 attn_out 也会触发 OOB(因为 OOB 是写飞出去的问题,跟读什么无关)
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

if "BISECT-B2 2026-05-20" not in text:
    print("ERROR: 当前不是 B2 状态")
    raise SystemExit(1)

if "BISECT-B3 2026-05-20" in text:
    print("⚠ 已经是 B3 了")
    raise SystemExit(0)

OLD = '''def sdpa_attention_with_kernel_seqlen_pad(
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

NEW = '''def sdpa_attention_with_kernel_seqlen_pad(
    query: torch.Tensor, #sum_seq, h_q, d_q
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens,
    max_prefill_seq_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """BISECT-B3 2026-05-20:
       STUB 掉 varlen_fa_seqlen_pad 和 SDPA,只跑 varlen_fa_seqlen_unpad
       最终确认 unpad 单独就能触发 OOB

       结果判定:
         跑通(4/4 OK)→ unpad 单独不挂,问题在 op 间的组合(罕见,需重审)
         崩(MUSA error)→ unpad 就是元凶,100% 锁定
    """
    import sys as _b3_sys
    if not getattr(torch, '_b3_announced', False):
        print("[BISECT-B3] STUB varlen_pad + STUB SDPA, only varlen_fa_seqlen_unpad runs",
              file=_b3_sys.stderr, flush=True)
        torch._b3_announced = True

    bs = seq_lens.shape[0] - 1
    sum_seq, h_q, d_q = query.shape
    _, h_kv, d_kv = key.shape
    device, dtype = query.device, query.dtype

    # alloc output(unpad 写入这里 —— shape (sum_seq, h_q, d_q))
    output = torch.empty((sum_seq, h_q, d_q), device=device, dtype=dtype)

    # alloc attn_out(unpad 读取这里 —— shape 同 SDPA 输出 (bs, h_q, max, d_q))
    # 内容是脏内存,但不影响 OOB 测试(OOB 是 unpad 写出去的问题,跟读什么无关)
    attn_out = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)

    # ===== STUB 替换 varlen_fa_seqlen_pad =====
    # 跳过

    # ===== STUB 替换 SDPA =====
    # 跳过

    # ★ 唯一被测试的 op
    ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)

    # 返回 STUB(不用 output,因为它里面是 unpad 写的可能含脏数据,我们想干净返回值)
    return query.reshape(sum_seq, h_q * d_q).clone().contiguous()'''

if OLD not in text:
    print("ERROR: 找不到 B2 函数")
    raise SystemExit(1)

text = text.replace(OLD, NEW)
with open(SRC, "w") as f:
    f.write(text)
print("✓ B3 patch 写入完成(只跑 varlen_fa_seqlen_unpad,其余 STUB)")
