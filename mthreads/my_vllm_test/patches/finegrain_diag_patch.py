"""撤掉 ZERO-PATCH + 升级 DIAG-PATCH 为细粒度同步版
   用法(容器内): python3 /data/my_vllm_test/patches/finegrain_diag_patch.py

   做的事:
     1. 把 ZERO-PATCH 块还原成 torch.empty(3 行)
     2. 把 DIAG-PATCH 块替换成"每步 synchronize() + try/except"版
     3. PATCH 1(DECODER 快路径)不动,保留

   回滚: 删 FG-DIAG 块,把 q_pad/k_pad/v_pad 三行 + varlen + sdpa 一段恢复原状即可
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# ============== 第 1 步:撤 ZERO-PATCH ==============
ZERO_PATCH_BLOCK = """    # ===== ZERO-PATCH 2026-05-20: padding 区清零,避免脏内存 NaN 进 kernel =====
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

ZERO_PATCH_RESTORE = """    q_pad = torch.empty((bs, h_q, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    k_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)
    v_pad = torch.empty((bs, h_kv, max_prefill_seq_len, d_q), device=device, dtype=dtype)"""

if ZERO_PATCH_BLOCK in text:
    text = text.replace(ZERO_PATCH_BLOCK, ZERO_PATCH_RESTORE)
    print("✓ ZERO-PATCH 已撤")
else:
    print("⚠ 找不到 ZERO-PATCH 块,跳过撤销")

# ============== 第 2 步:替换 DIAG-PATCH ==============
# 找原 DIAG-PATCH 块整段,替换成新版
import re
# 用正则匹配 DIAG-PATCH 标记之间的整段
old_diag_pattern = re.compile(
    r"    # ===== DIAG-PATCH 2026-05-19:.*?# ===== END DIAG-PATCH =====",
    re.DOTALL
)

new_diag = '''    # ===== FG-DIAG 2026-05-20: 细粒度同步 + try/except,精确定位哪个 kernel 越界 =====
    # 取代旧 DIAG-PATCH,做细粒度 synchronize 强制让 GPU 错误立刻浮现
    # 设计: 失败时直接打印"哪一步挂的"和当时数据形状;成功时静默
    # 回滚: 删本块,恢复 ORIGINAL 注释里的 2 行
    # ORIGINAL:
    #     ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    #     attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    import sys as _fg_sys
    _fg_seq_lens_cpu = seq_lens.cpu().tolist() if hasattr(seq_lens, 'cpu') else list(seq_lens)
    _fg_total_seq = _fg_seq_lens_cpu[-1] if _fg_seq_lens_cpu else 0
    _fg_dump_ctx = (f"sum_seq={sum_seq} total_seq={_fg_total_seq} max={max_prefill_seq_len} bs={bs} "
                    f"q_in={tuple(query.shape)} k_in={tuple(key.shape)} v_in={tuple(value.shape)} "
                    f"q_pad={tuple(q_pad.shape)} dtype={q_pad.dtype}")

    # Step A: q_pad/k_pad/v_pad 已经 alloc(empty),先 sync 一次确保 alloc 完成
    try:
        torch.musa.synchronize()
    except Exception as _e:
        print(f"[FG-DIAG] !!! 入口 sync 失败(说明上一个 kernel 留了 sticky error): {_e} | {_fg_dump_ctx}",
              file=_fg_sys.stderr, flush=True)
        raise

    # Step B: 调 varlen_fa_seqlen_pad
    try:
        ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
        torch.musa.synchronize()    # 强制 GPU 真完成,有错立刻浮现
    except Exception as _e:
        print(f"[FG-DIAG] !!! varlen_fa_seqlen_pad 后 sync 失败: {type(_e).__name__}: {_e} | {_fg_dump_ctx}",
              file=_fg_sys.stderr, flush=True)
        raise

    # Step C: 检查 varlen 写出的 q_pad/k_pad/v_pad 含不含 NaN/Inf(只有 mismatch 时才检查,省时)
    if sum_seq != _fg_total_seq:
        _q_bad = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
        _k_bad = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
        _v_bad = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
        if _q_bad or _k_bad or _v_bad:
            print(f"[FG-DIAG] varlen 写出 NaN/Inf: q={_q_bad} k={_k_bad} v={_v_bad} | {_fg_dump_ctx}",
                  file=_fg_sys.stderr, flush=True)

    # Step D: 调 SDPA flash kernel
    try:
        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
        torch.musa.synchronize()    # 强制 SDPA 完成,有错立刻浮现
    except Exception as _e:
        print(f"[FG-DIAG] !!! SDPA flash 后 sync 失败: {type(_e).__name__}: {_e} | {_fg_dump_ctx}",
              file=_fg_sys.stderr, flush=True)
        raise
    # ===== END FG-DIAG ====='''

m = old_diag_pattern.search(text)
if m:
    text = text[:m.start()] + new_diag + text[m.end():]
    print("✓ DIAG-PATCH 升级为 FG-DIAG")
else:
    print("⚠ 找不到旧 DIAG-PATCH 块")

# ============== 写回 ==============
with open(SRC, "w") as f:
    f.write(text)
print("✓ 写入完成")
