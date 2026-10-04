"""清掉 SDPA 输入(q_pad/k_pad/v_pad)的 NaN/Inf,带前后验证
   用法(容器内): python3 /data/my_vllm_test/patches/clean_input_nan_patch.py
   回滚: 删 INPUT-NAN-CLEAN 块即可

   做的事:
     1. 撤掉之前的 NAN-CLEAN(它清的是 SDPA output,绕路了)
     2. 在 varlen 之后、SDPA 之前清 q_pad/k_pad/v_pad
     3. 清理后再检查一次 NaN/Inf 验证清理生效

   预期 log:
     - 旧: "varlen 写出 NaN/Inf: q=True k=True v=True"   (清理前,预期还在)
     - 新: "清理后 q/k/v 仍含 NaN: q=False k=False v=False"  (清理后,应该都 False)
     - 如果 8/8 通过 → NaN 是根因,这才是对的修法
     - 如果还崩 → bug 在更深处,需要继续定位
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# === 第 1 步:撤掉之前的 NAN-CLEAN(它清 attn_out,绕路了)===
OLD_NAN_CLEAN = """        torch.musa.synchronize()    # 强制 SDPA 完成,有错立刻浮现
        # ===== NAN-CLEAN 2026-05-20: SDPA output 清 NaN/Inf =====
        # 防止 SDPA 输出的 NaN 传到下游 op 触发非法访问
        # 回滚: 删这两行
        if sum_seq != _fg_total_seq:    # 仅 mismatch 时清,正常情况不动
            attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        # ===== END NAN-CLEAN ====="""
RESTORE_AFTER_SDPA = """        torch.musa.synchronize()    # 强制 SDPA 完成,有错立刻浮现"""

if OLD_NAN_CLEAN in text:
    text = text.replace(OLD_NAN_CLEAN, RESTORE_AFTER_SDPA)
    print("✓ 撤掉了之前(绕路的)NAN-CLEAN on attn_out")
else:
    print("⚠ 找不到旧 NAN-CLEAN 块(可能已经被撤了)")

# === 第 2 步:在 varlen 之后(就是 Step C 的 NaN 检查那段附近)插入清理 + 验证 ===
# 锚点: Step C 的 NaN 检查代码块
ANCHOR = """    # Step C: 检查 varlen 写出的 q_pad/k_pad/v_pad 含不含 NaN/Inf(只有 mismatch 时才检查,省时)
    if sum_seq != _fg_total_seq:
        _q_bad = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
        _k_bad = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
        _v_bad = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
        if _q_bad or _k_bad or _v_bad:
            print(f"[FG-DIAG] varlen 写出 NaN/Inf: q={_q_bad} k={_k_bad} v={_v_bad} | {_fg_dump_ctx}",
                  file=_fg_sys.stderr, flush=True)"""

REPLACEMENT = """    # Step C: 检查 varlen 写出的 q_pad/k_pad/v_pad 含不含 NaN/Inf(只有 mismatch 时才检查,省时)
    if sum_seq != _fg_total_seq:
        _q_bad = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
        _k_bad = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
        _v_bad = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
        if _q_bad or _k_bad or _v_bad:
            print(f"[FG-DIAG] 清理前: varlen 写出 NaN/Inf: q={_q_bad} k={_k_bad} v={_v_bad} | {_fg_dump_ctx}",
                  file=_fg_sys.stderr, flush=True)
            # ===== INPUT-NAN-CLEAN 2026-05-20: 在 SDPA 看到之前把 NaN/Inf 清成 0 =====
            # 回滚: 删本块的"清理" + "验证清理"两段(保留 _q_bad 的初次检查)
            q_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            k_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            v_pad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            torch.musa.synchronize()   # 确保 nan_to_num_ 完成
            # 验证清理生效
            _q_bad_after = torch.isnan(q_pad).any().item() or torch.isinf(q_pad).any().item()
            _k_bad_after = torch.isnan(k_pad).any().item() or torch.isinf(k_pad).any().item()
            _v_bad_after = torch.isnan(v_pad).any().item() or torch.isinf(v_pad).any().item()
            print(f"[FG-DIAG] 清理后: q={_q_bad_after} k={_k_bad_after} v={_v_bad_after} (期望全 False)",
                  file=_fg_sys.stderr, flush=True)
            # ===== END INPUT-NAN-CLEAN ====="""

if "INPUT-NAN-CLEAN 2026-05-20" in text:
    print("⚠ INPUT-NAN-CLEAN 已存在,跳过")
elif ANCHOR not in text:
    print("ERROR: 找不到 Step C 锚点")
    raise SystemExit(1)
else:
    text = text.replace(ANCHOR, REPLACEMENT)
    print("✓ INPUT-NAN-CLEAN(在 SDPA 之前清 q/k/v_pad + 后置验证)插入完成")

with open(SRC, "w") as f:
    f.write(text)
print("✓ 写入完成")
