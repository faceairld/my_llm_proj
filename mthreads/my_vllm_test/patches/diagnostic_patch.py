"""诊断 patch: 在 sdpa_attention_with_kernel_seqlen_pad 加 checkpoint
   用法(容器内): python3 /data/my_vllm_test/patches/diagnostic_patch.py
   回滚: 删 DIAG-PATCH 标记块,恢复 ORIGINAL 注释里的代码
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

# 原始的 ops.varlen_fa_seqlen_pad + torch.ops.aten._scaled_dot_product_attention_flash_musa
# 我们在三处插桩
ORIG = """    ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
        q_pad,
        k_pad,
        v_pad,
        dropout_p=0.0,
        is_causal=is_causal)"""

PATCHED = """    # ===== DIAG-PATCH 2026-05-19: 加诊断检查点 =====
    # 假设: prefix-cache 命中后,query token 数 < seq_lens 总和 → q_pad 末尾是脏内存 → kernel 挂
    # 回滚: 删 DIAG-PATCH 块,恢复 ORIGINAL 段
    # ORIGINAL:
    #     ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    #     attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
    #         q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    import sys as _diag_sys
    _diag_seq_lens_cpu = seq_lens.cpu().tolist() if hasattr(seq_lens, 'cpu') else list(seq_lens)
    _diag_total_seq = _diag_seq_lens_cpu[-1] if _diag_seq_lens_cpu else 0
    _diag_mismatch = (sum_seq != _diag_total_seq)
    if _diag_mismatch:
        print(f"[DIAG] !!! 不匹配:query 实际 token={sum_seq}, seq_lens 总和={_diag_total_seq}, "
              f"差 {_diag_total_seq - sum_seq}, max_seq={max_prefill_seq_len}, bs={bs}",
              file=_diag_sys.stderr, flush=True)

    ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)

    # Checkpoint 2:padding 之后 kernel 之前,检查 q_pad/k_pad/v_pad 有无 NaN/Inf
    _diag_q_bad = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
    _diag_k_bad = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
    _diag_v_bad = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
    if _diag_q_bad or _diag_k_bad or _diag_v_bad:
        print(f"[DIAG] !!! padding 后含 NaN/Inf:q={_diag_q_bad} k={_diag_k_bad} v={_diag_v_bad}, "
              f"shapes q_pad={tuple(q_pad.shape)} k_pad={tuple(k_pad.shape)} v_pad={tuple(v_pad.shape)}, "
              f"sum_seq={sum_seq}, total_seq={_diag_total_seq}",
              file=_diag_sys.stderr, flush=True)

    # Checkpoint 3:包住 kernel 调用,挂时 dump 一切
    try:
        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    except RuntimeError as _e:
        print(f"[DIAG] !!! kernel 抛错: {_e}", file=_diag_sys.stderr, flush=True)
        print(f"[DIAG]   shapes: q_pad={tuple(q_pad.shape)} k_pad={tuple(k_pad.shape)} v_pad={tuple(v_pad.shape)}",
              file=_diag_sys.stderr, flush=True)
        print(f"[DIAG]   sum_seq={sum_seq}, total_seq={_diag_total_seq}, max_prefill={max_prefill_seq_len}, bs={bs}",
              file=_diag_sys.stderr, flush=True)
        print(f"[DIAG]   seq_lens={_diag_seq_lens_cpu[:10]}{'...' if len(_diag_seq_lens_cpu)>10 else ''}",
              file=_diag_sys.stderr, flush=True)
        print(f"[DIAG]   dtype={q_pad.dtype}, causal={is_causal}",
              file=_diag_sys.stderr, flush=True)
        # 查看 q_pad 真实数据范围(取 cpu 样本)
        try:
            _q_min = q_pad.float().min().item()
            _q_max = q_pad.float().max().item()
            _diag_n_nan = torch.isnan(q_pad).sum().item()
            _diag_n_inf = torch.isinf(q_pad).sum().item()
            print(f"[DIAG]   q_pad: min={_q_min:.3e} max={_q_max:.3e} nan={_diag_n_nan} inf={_diag_n_inf}",
                  file=_diag_sys.stderr, flush=True)
        except Exception:
            pass
        raise
    # ===== END DIAG-PATCH ====="""

with open(SRC) as f:
    text = f.read()

if "DIAG-PATCH 2026-05-19" in text:
    print("⚠ 已经 patch 过了,跳过")
    raise SystemExit(0)

if ORIG not in text:
    print("ERROR: 找不到要替换的原始代码块")
    raise SystemExit(1)

new_text = text.replace(ORIG, PATCHED)
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ diagnostic patch 写入完成")
