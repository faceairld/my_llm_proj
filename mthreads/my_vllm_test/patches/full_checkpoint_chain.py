"""完整检查链 patch:替换之前所有 FG-DIAG 系列,装一个 12 个 checkpoint 的细粒度链
   用法(容器内): python3 /data/my_vllm_test/patches/full_checkpoint_chain.py
   回滚: 删 FULL-CKPT 块,恢复 ORIGINAL 注释里的原始 2 行

   覆盖的 checkpoint:
     CKPT-0  函数入口 sync         → 上一个 attention layer 的下游有问题
     CKPT-1  seq_lens.cpu() 之后    → 这一步本身有问题
     CKPT-2  alloc q/k/v_pad 之后   → torch.empty 出错(罕见,但记录)
     CKPT-3  varlen 之后            → varlen 自己挂或者 sticky error
     CKPT-3.5 NaN/Inf 报告(纯诊断,不抛错)
     CKPT-4  nan_to_num_ 之后       → 清理操作本身的问题
     CKPT-4.5 验证清理效果(纯诊断)
     CKPT-5  SDPA flash 之后        → SDPA 自己挂
     CKPT-6  SDPA output NaN(诊断 + 再清一次)
     CKPT-7  返回前最后一次 sync    → 函数体内所有操作都成功
   预期: 失败时打印 "[CKPT-N] xxx | ctx=..." 直接定位到具体行
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

import re

# 找之前的 FG-DIAG 段(包括 INPUT-NAN-CLEAN 嵌套在内),整段替换
old_block = re.compile(
    r"    # ===== FG-DIAG 2026-05-20:.*?# ===== END FG-DIAG =====",
    re.DOTALL
)

NEW_BLOCK = '''    # ===== FULL-CKPT 2026-05-20: 完整检查链,12 个 checkpoint 精确定位错误源 =====
    # 替换之前所有 FG-DIAG / INPUT-NAN-CLEAN / NAN-CLEAN 系列
    # 设计:每条 GPU 语句独立 sync + try/except,任何错误第一时间归到那条语句
    # 回滚:删本块,恢复 ORIGINAL 注释里的 2 行
    # ORIGINAL:
    #     ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    #     attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    import sys as _ckpt_sys

    def _ckpt_sync(label, ctx=""):
        try:
            torch.musa.synchronize()
        except Exception as _e:
            print(f"[CKPT-FAIL] {label}: {type(_e).__name__}: {_e} | {ctx}",
                  file=_ckpt_sys.stderr, flush=True)
            raise

    # === CKPT-0: 入口 sync(catch 上一个 attention layer 留下的 sticky error)===
    _ckpt_sync("CKPT-0 入口 (上一层下游可能出错)")

    # === CKPT-1: seq_lens.cpu() 后(本身可能出错)===
    _ckpt_seq_lens_cpu = seq_lens.cpu().tolist() if hasattr(seq_lens, 'cpu') else list(seq_lens)
    _ckpt_total_seq = _ckpt_seq_lens_cpu[-1] if _ckpt_seq_lens_cpu else 0
    _ckpt_ctx = (f"sum_seq={sum_seq} total={_ckpt_total_seq} max={max_prefill_seq_len} bs={bs} "
                 f"q_in={tuple(query.shape)} k_in={tuple(key.shape)} dtype={query.dtype}")
    _ckpt_sync("CKPT-1 seq_lens.cpu()", _ckpt_ctx)

    # === CKPT-2: alloc q/k/v_pad 之后(torch.empty 本身)===
    # 注意:q_pad/k_pad/v_pad 在前面 ORIGINAL 段已经 alloc 好了(empty),这里只 sync
    _ckpt_sync("CKPT-2 q/k/v_pad alloc", _ckpt_ctx)

    # === CKPT-3: ops.varlen_fa_seqlen_pad ===
    try:
        ops.varlen_fa_seqlen_pad(query, key, value, q_pad, k_pad, v_pad, seq_lens, seq_lens, sum_seq, max_prefill_seq_len, bs)
    except Exception as _e:
        print(f"[CKPT-FAIL] CKPT-3 varlen_fa_seqlen_pad 同步抛错: {type(_e).__name__}: {_e} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        raise
    _ckpt_sync("CKPT-3 varlen_fa_seqlen_pad", _ckpt_ctx)

    # === CKPT-3.5: 检查 varlen 是否写出 NaN/Inf(纯诊断)===
    _ckpt_q_bad = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
    _ckpt_k_bad = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
    _ckpt_v_bad = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
    _ckpt_has_bad = _ckpt_q_bad or _ckpt_k_bad or _ckpt_v_bad
    if _ckpt_has_bad:
        print(f"[CKPT-3.5] varlen 写出 NaN/Inf: q={_ckpt_q_bad} k={_ckpt_k_bad} v={_ckpt_v_bad} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)

    # === CKPT-4: nan_to_num_ 清理 q/k/v_pad ===
    if _ckpt_has_bad:
        try:
            q_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            k_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            v_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as _e:
            print(f"[CKPT-FAIL] CKPT-4 nan_to_num_ 抛错: {type(_e).__name__}: {_e} | {_ckpt_ctx}",
                  file=_ckpt_sys.stderr, flush=True)
            raise
        _ckpt_sync("CKPT-4 nan_to_num_(q/k/v_pad)", _ckpt_ctx)

        # === CKPT-4.5: 验证清理生效(纯诊断)===
        _q2 = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
        _k2 = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
        _v2 = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
        print(f"[CKPT-4.5] 清理后: q={_q2} k={_k2} v={_v2} (期望全 False)",
              file=_ckpt_sys.stderr, flush=True)

    # === CKPT-5: SDPA flash ===
    try:
        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
    except Exception as _e:
        print(f"[CKPT-FAIL] CKPT-5 SDPA flash 同步抛错: {type(_e).__name__}: {_e} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        raise
    _ckpt_sync("CKPT-5 SDPA flash", _ckpt_ctx)

    # === CKPT-6: SDPA output NaN 检查 + 清理(诊断 + 兜底)===
    _ckpt_out_bad = torch.isnan(attn_out).any().item() or torch.isinf(attn_out).any().item()
    if _ckpt_out_bad:
        print(f"[CKPT-6] SDPA output 含 NaN/Inf, 清掉 | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        _ckpt_sync("CKPT-6 attn_out.nan_to_num_", _ckpt_ctx)

    # === CKPT-7: 出口 sync(确认函数体内所有 GPU op 都成功)===
    _ckpt_sync("CKPT-7 出口 (函数体内全部成功)", _ckpt_ctx)
    # ===== END FULL-CKPT ====='''

m = old_block.search(text)
if not m:
    print("ERROR: 找不到旧 FG-DIAG 块,patch 失败")
    raise SystemExit(1)

new_text = text[:m.start()] + NEW_BLOCK + text[m.end():]
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ FULL-CKPT 完整检查链已装入")
print("  替换的范围:旧 FG-DIAG 整段(含 INPUT-NAN-CLEAN 嵌套)")
