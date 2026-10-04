"""加 CKPT-2.7(检查 varlen 输入)+ CKPT-2.8(强制清干净 varlen 输入)
   用法(容器内): python3 /data/my_vllm_test/patches/check_varlen_input.py
   回滚: 删 CHECK-INPUT 块即可

   目的:验证 varlen 是否真的是 NaN 的源头
     - CKPT-2.7: 在 varlen 之前查 q_in/k_in/v_in 是否有 NaN
     - CKPT-2.8: 主动清干净 q_in/k_in/v_in
     - 这样后续 CKPT-3.5 看到的 q_pad NaN 状态,只能归因于 varlen 自身

   预期 log:
     - "[CKPT-2.7] varlen 输入: q_in=False ..." → 上游送来的就是干净的
     - "[CKPT-2.7] varlen 输入: q_in=True ..." → 上游就有 NaN
     - 清理后跑,CKPT-3.5 还报 NaN → varlen 自己产
     - 清理后跑,CKPT-3.5 不报 → varlen 干净,NaN 是上游来的(被 nan_to_num_ 清了)
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# 锚点: CKPT-2 alloc 之后,CKPT-3 varlen 调用之前
ANCHOR = """    # === CKPT-2: alloc q/k/v_pad 之后(torch.empty 本身)===
    # 注意:q_pad/k_pad/v_pad 在前面 ORIGINAL 段已经 alloc 好了(empty),这里只 sync
    _ckpt_sync("CKPT-2 q/k/v_pad alloc", _ckpt_ctx)

    # === CKPT-3: ops.varlen_fa_seqlen_pad ==="""

REPLACEMENT = """    # === CKPT-2: alloc q/k/v_pad 之后(torch.empty 本身)===
    # 注意:q_pad/k_pad/v_pad 在前面 ORIGINAL 段已经 alloc 好了(empty),这里只 sync
    _ckpt_sync("CKPT-2 q/k/v_pad alloc", _ckpt_ctx)

    # === CKPT-2.7: varlen 输入 (q_in/k_in/v_in) 是否含 NaN/Inf(诊断)===
    # ===== CHECK-INPUT 2026-05-20 =====
    # 回滚: 删本块即可,完全独立可移除
    _q_in_bad = torch.isnan(query).any().item() or torch.isinf(query).any().item()
    _k_in_bad = torch.isnan(key).any().item() or torch.isinf(key).any().item()
    _v_in_bad = torch.isnan(value).any().item() or torch.isinf(value).any().item()
    if sum_seq != _ckpt_total_seq:  # 仅 mismatch 时报,信号才有意义
        print(f"[CKPT-2.7] varlen 输入: q_in={_q_in_bad} k_in={_k_in_bad} v_in={_v_in_bad} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)

    # === CKPT-2.8: 主动清掉 varlen 输入的 NaN/Inf,确保只剩 varlen 本身的影响 ===
    if _q_in_bad: query.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if _k_in_bad: key.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if _v_in_bad: value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if _q_in_bad or _k_in_bad or _v_in_bad:
        _ckpt_sync("CKPT-2.8 强清 varlen 输入", _ckpt_ctx)
        # 验证清干净了
        _q2 = torch.isnan(query).any().item() or torch.isinf(query).any().item()
        _k2 = torch.isnan(key).any().item() or torch.isinf(key).any().item()
        _v2 = torch.isnan(value).any().item() or torch.isinf(value).any().item()
        print(f"[CKPT-2.8] varlen 输入清理后: q_in={_q2} k_in={_k2} v_in={_v2} (期望全 False)",
              file=_ckpt_sys.stderr, flush=True)
    # ===== END CHECK-INPUT =====

    # === CKPT-3: ops.varlen_fa_seqlen_pad ==="""

if "CHECK-INPUT 2026-05-20" in text:
    print("⚠ 已经 patch 过了,跳过")
    raise SystemExit(0)

if ANCHOR not in text:
    print("ERROR: 找不到锚点(FULL-CKPT 装好的话应该在这里)")
    raise SystemExit(1)

new_text = text.replace(ANCHOR, REPLACEMENT)
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ CHECK-INPUT 装入(CKPT-2.7 + CKPT-2.8)")
