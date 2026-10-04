"""一次性 patch 脚本: 修 vllm_musa/v0/flash_attn.py 的死锁
   用法(容器内): python3 /data/my_vllm_test/patches/apply_patch.py
   原理: DECODER + prefix-caching 错误地走 _get_seq_len_block_table_args 慢路径,
        慢路径在长 prompt + MUSA stream 拥堵下 388 行 torch.tensor(...) 死锁
   修法: DECODER 一律走快路径(prefill_meta.seq_start_loc),与慢路径数值等价
   回滚: 把 PATCH 块的 4 行删掉,恢复 ORIGINAL 注释里的 3 行 if 条件即可
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

ORIG = """            if self.attn_type == AttentionType.DECODER and (
                    kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):"""

PATCHED = """            # ===== PATCH 2026-05-19: 修 flash_attn.py:388 死锁 =====
            # 原因: DECODER+prefix-caching 时走 else 分支调 _get_seq_len_block_table_args,
            #       内部 torch.tensor(...,device=gpu) 在长 prompt MUSA stream 拥堵时死锁
            # 改动: DECODER 一律走快路径(prefill_meta.seq_start_loc),与慢路径数值等价
            # 回滚: 删本块,恢复下面 ORIGINAL 注释的 3 行
            # ORIGINAL:
            #     if self.attn_type == AttentionType.DECODER and (
            #             kv_cache.numel() == 0 or prefill_meta.block_tables is None
            #             or prefill_meta.block_tables.numel() == 0):
            if self.attn_type == AttentionType.DECODER:
            # ===== END PATCH ====="""

with open(SRC) as f:
    text = f.read()

if "PATCH 2026-05-19" in text:
    print("⚠ 已经 patch 过了,跳过(避免重复应用)")
    raise SystemExit(0)

if ORIG not in text:
    print("ERROR: 找不到要替换的原始 if 块,可能文件已被改动")
    raise SystemExit(1)

new_text = text.replace(ORIG, PATCHED)
assert text != new_text, "替换失败"
with open(SRC, "w") as f:
    f.write(new_text)

# 验证
with open(SRC) as f:
    final = f.read()
assert "PATCH 2026-05-19" in final, "写入后再读取找不到 patch 标记"
print("✓ patch 已写入 " + SRC)
print(f"  文件行数: {final.count(chr(10))} (原 {text.count(chr(10))}, +{final.count(chr(10))-text.count(chr(10))} 行)")
