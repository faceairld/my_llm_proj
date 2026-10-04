"""扩展 FULL-CKPT,把 varlen_fa_seqlen_unpad 也纳入检查链
   用法(容器内): python3 /data/my_vllm_test/patches/extend_ckpt_to_unpad.py
   回滚: 把 unpad 那行从 FULL-CKPT 块里移出来

   做的事:
     - 在 CKPT-7 之后,加 CKPT-8(unpad) + CKPT-9(.view + return 前 sync)
     - 把 ops.varlen_fa_seqlen_unpad(...) 和 return output.view(...) 也包进检查链
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# 锚点:CKPT-7 + END FULL-CKPT + unpad + return,整段替换
ANCHOR = """    # === CKPT-7: 出口 sync(确认函数体内所有 GPU op 都成功)===
    _ckpt_sync("CKPT-7 出口 (函数体内全部成功)", _ckpt_ctx)
    # ===== END FULL-CKPT =====

    ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    return output.view(-1, h_q * d_q)"""

REPLACEMENT = """    # === CKPT-7: SDPA 之后 sync(函数体上半全过)===
    _ckpt_sync("CKPT-7 SDPA 后 (上半都过)", _ckpt_ctx)

    # === CKPT-8: varlen_fa_seqlen_unpad(强嫌疑越界点 —— output 只 alloc 了 sum_seq 行)===
    try:
        ops.varlen_fa_seqlen_unpad(attn_out, output, seq_lens, sum_seq, max_prefill_seq_len, d_q, h_q, bs)
    except Exception as _e:
        print(f"[CKPT-FAIL] CKPT-8 varlen_fa_seqlen_unpad 同步抛错: {type(_e).__name__}: {_e} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        raise
    _ckpt_sync("CKPT-8 varlen_fa_seqlen_unpad", _ckpt_ctx)

    # === CKPT-8.5: 检查 unpad 后 output 是否含 NaN/Inf(诊断 + 兜底清)===
    _ckpt_o_bad = torch.isnan(output).any().item() or torch.isinf(output).any().item()
    if _ckpt_o_bad:
        print(f"[CKPT-8.5] unpad 后 output 含 NaN/Inf, 清掉 | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        output.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        _ckpt_sync("CKPT-8.5 output.nan_to_num_", _ckpt_ctx)

    # === CKPT-9: 函数出口最终 sync(整个函数体所有 GPU op 都成功)===
    _ckpt_sync("CKPT-9 最终出口 (整函数全部成功)", _ckpt_ctx)
    # ===== END FULL-CKPT =====

    return output.view(-1, h_q * d_q)"""

if ANCHOR not in text:
    print("ERROR: 找不到锚点(可能 FULL-CKPT 没装好,或者 unpad 行已经被移动)")
    raise SystemExit(1)

if "CKPT-8 varlen_fa_seqlen_unpad" in text:
    print("⚠ CKPT-8 已存在,跳过")
    raise SystemExit(0)

new_text = text.replace(ANCHOR, REPLACEMENT)
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ FULL-CKPT 已扩展,unpad 现在被 CKPT-8 包围")
