"""在 SDPA output 上 nan_to_num 清 NaN/Inf,验证"NaN 下游传播"假设
   用法(容器内): python3 /data/my_vllm_test/patches/clean_nan_after_sdpa_patch.py
   回滚: 删 NAN-CLEAN 标记块即可

   原理: 已确认 varlen 后 q_pad/k_pad/v_pad 含 NaN/Inf。
        SDPA 输入有 NaN,输出几乎必然也有 NaN(softmax/matmul 传染)。
        在返回 attn_out 之前清干净,看后续 layer 还崩不崩。
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

# 找 FG-DIAG 里 SDPA 调用后那段,加入 nan_to_num_
# 锚点: SDPA 调用完后,有一个 torch.musa.synchronize() + END FG-DIAG
ANCHOR = """        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
        torch.musa.synchronize()    # 强制 SDPA 完成,有错立刻浮现"""

ADDITION = """        attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q_pad, k_pad, v_pad, dropout_p=0.0, is_causal=is_causal)
        torch.musa.synchronize()    # 强制 SDPA 完成,有错立刻浮现
        # ===== NAN-CLEAN 2026-05-20: SDPA output 清 NaN/Inf =====
        # 防止 SDPA 输出的 NaN 传到下游 op 触发非法访问
        # 回滚: 删这两行
        if sum_seq != _fg_total_seq:    # 仅 mismatch 时清,正常情况不动
            attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        # ===== END NAN-CLEAN ====="""

if "NAN-CLEAN 2026-05-20" in text:
    print("⚠ 已经 patch 过了,跳过")
    raise SystemExit(0)

if ANCHOR not in text:
    print("ERROR: 找不到 SDPA 调用锚点")
    raise SystemExit(1)

new_text = text.replace(ANCHOR, ADDITION)
with open(SRC, "w") as f:
    f.write(new_text)
print("✓ NAN-CLEAN patch 写入")
