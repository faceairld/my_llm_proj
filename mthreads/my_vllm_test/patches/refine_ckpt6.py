"""精细化 CKPT-6:把 attn_out 分两个区域查 NaN
   - 真实输出区 attn_out[..., 0:sum_seq, ...]
   - padding 区   attn_out[..., sum_seq:max, ...]
   只清"真实输出区"的 NaN(如果有),padding 区不动(反正后面要截掉)
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

OLD = """    # === CKPT-6: SDPA output NaN 检查 + 清理(诊断 + 兜底)===
    _ckpt_out_bad = torch.isnan(attn_out).any().item() or torch.isinf(attn_out).any().item()
    if _ckpt_out_bad:
        print(f"[CKPT-6] SDPA output 含 NaN/Inf, 清掉 | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        _ckpt_sync("CKPT-6 attn_out.nan_to_num_", _ckpt_ctx)"""

NEW = """    # === CKPT-6: SDPA output 分区域 NaN 检查 ★精细化★ ===
    # attn_out shape = (bs, h, max, d)
    # 真实区: attn_out[:, :, 0:sum_seq, :] —— SDPA 实际写入这里
    # padding 区: attn_out[:, :, sum_seq:max, :] —— 可能是 alloc 残留,SDPA 可能没写
    _ckpt_real_bad = torch.isnan(attn_out[:, :, :sum_seq, :]).any().item() or torch.isinf(attn_out[:, :, :sum_seq, :]).any().item()
    _ckpt_pad_bad  = torch.isnan(attn_out[:, :, sum_seq:, :]).any().item() or torch.isinf(attn_out[:, :, sum_seq:, :]).any().item()
    if _ckpt_real_bad or _ckpt_pad_bad:
        print(f"[CKPT-6] attn_out 真实区[0:{sum_seq}]={_ckpt_real_bad}  padding 区[{sum_seq}:{max_prefill_seq_len}]={_ckpt_pad_bad} | {_ckpt_ctx}",
              file=_ckpt_sys.stderr, flush=True)
        # 关键判定:
        #   real=True → SDPA 真的产了 NaN(凶手是 SDPA)
        #   real=False, pad=True → 只是 alloc 残留,SDPA 没写 padding 区,本身没事
        if _ckpt_real_bad:
            print(f"[CKPT-6] ★ SDPA 真实输出区也含 NaN,SDPA 计算异常 ★",
                  file=_ckpt_sys.stderr, flush=True)
            attn_out[:, :, :sum_seq, :].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        # padding 区即使脏也不影响最终 output(unpad 只取真实区),但为了让后续 op 安全,顺手清一下
        if _ckpt_pad_bad:
            attn_out[:, :, sum_seq:, :].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        _ckpt_sync("CKPT-6 attn_out.nan_to_num_ 分区", _ckpt_ctx)"""

if "attn_out 真实区[0:" in text:
    print("⚠ 已经 patch 过了")
    raise SystemExit(0)
if OLD not in text:
    print("ERROR: 找不到旧 CKPT-6 块")
    raise SystemExit(1)

text = text.replace(OLD, NEW)
with open(SRC, "w") as f:
    f.write(text)
print("✓ CKPT-6 精细化完成")
