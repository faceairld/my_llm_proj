"""Zero-fill patch: 把 sdpa_attention_with_kernel_seqlen_pad 里的 torch.empty 改成 torch.zeros
   用法(容器内): python3 /data/my_vllm_test/patches/zero_fill_patch.py
   回滚: 删 ZERO-PATCH 块,恢复 ORIGINAL 注释里的 3 行 torch.empty
   原因: prefix-cache 命中时,padding 区(超出真实 token 的部分)是脏内存(NaN/Inf)
        zero-fill 让 kernel 看到全 0 而不是 NaN,数学合法,attention 不崩
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

ORIG = """    q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)"""

PATCHED = """    # ===== ZERO-PATCH 2026-05-20: padding 区清零,避免脏内存 NaN 进 kernel =====
    # 原因: prefix-cache 命中时,query 只有新 token 数,但 max_prefill_seq_len=完整序列长
    #       torch.empty 不清零 → 多出的位置是脏内存(常含 NaN/Inf)→ kernel 爆
    # 回滚: 删本块恢复 ORIGINAL
    # ORIGINAL:
    #     q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    #     k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    #     v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    q_pad = torch.zeros((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.zeros((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.zeros((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    # ===== END ZERO-PATCH ====="""

with open(SRC) as f:
    text = f.read()

if "ZERO-PATCH 2026-05-20" in text:
    print("⚠ 已经 patch 过了,跳过")
    raise SystemExit(0)

if ORIG not in text:
    print("ERROR: 找不到要替换的 torch.empty 三行")
    raise SystemExit(1)

new_text = text.replace(ORIG, PATCHED)
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ zero-fill patch 写入完成")
