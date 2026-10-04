"""Path C patch: 给 vllm_musa flash_attn.py 的 forward 函数加 paged-aware prefill 路径
   prefix-cache 命中场景下,绕开 sdpa_attention_with_kernel_seqlen_pad,
   改用 vllm 主仓的 context_attention_fwd (Triton paged-aware kernel)

   用法(容器内): python3 /data/my_vllm_test/patches/path_c_paged_prefill.py
   回滚: docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py 回去

   原理:
     - 检测 prefix-cache 命中(kv_cache 非空 + block_tables 非空)
     - 命中时直接调 context_attention_fwd,它自己用 block_tables 从 KV cache 读 cached K/V
     - 不命中时走原 sdpa_attention_with_kernel_seqlen_pad(fresh prefill 正常路径)

   预期:
     - 4/4 全过(不再崩)
     - 模型输出语义正确(因为 context_attention_fwd 真用了 cached K/V)
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

if "PATH-C 2026-05-21" in text:
    print("⚠ 已经 patch 过了")
    raise SystemExit(0)

# 锚点:原版 prefill 分支起点 "if prefill_meta := attn_metadata.prefill_metadata:"
# 在这一行之后立刻插入 Path C 检测和早退
ANCHOR = """        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            # normal attention and DECODER
            if self.attn_type == AttentionType.DECODER and (
                    kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):"""

REPLACEMENT = """        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-C 2026-05-21: prefix-cache 命中场景走 paged-aware prefill kernel =====
            # 原本的 sdpa_attention_with_kernel_seqlen_pad 在命中场景下不接 KV cache,
            # 导致 varlen_fa_seqlen_unpad 越界写;改用 vllm 主仓的 context_attention_fwd,
            # 它直接从 KV cache (paged storage) 用 block_tables 拉 cached K/V,无 OOB,数学正确
            _path_c_hit = (
                self.attn_type == AttentionType.DECODER
                and kv_cache.numel() > 0
                and prefill_meta.block_tables is not None
                and prefill_meta.block_tables.numel() > 0
            )
            if _path_c_hit:
                from vllm.attention.ops.prefix_prefill import context_attention_fwd
                _path_c_sum_seq, _path_c_h_q, _path_c_d_q = query.shape
                _path_c_output = torch.empty_like(query)
                _path_c_key_cache = kv_cache[0]
                _path_c_value_cache = kv_cache[1]
                # query_start_loc: cumsum of NEW token lengths (优先用 metadata 上的)
                _path_c_query_start_loc = getattr(attn_metadata, 'query_start_loc', None)
                if _path_c_query_start_loc is None:
                    # fallback: 跟 seq_start_loc 一样(无命中时 query_lens == seq_lens)
                    _path_c_query_start_loc = prefill_meta.seq_start_loc
                _path_c_max_query_len = getattr(attn_metadata, 'max_query_len', num_prefill_tokens)
                # k_scale / v_scale: 默认 1.0(我们没开 fp8 KV cache)
                _path_c_k_scale = getattr(layer, '_k_scale', None)
                if _path_c_k_scale is None:
                    _path_c_k_scale = torch.tensor(1.0, device=query.device, dtype=torch.float32)
                _path_c_v_scale = getattr(layer, '_v_scale', None)
                if _path_c_v_scale is None:
                    _path_c_v_scale = torch.tensor(1.0, device=query.device, dtype=torch.float32)
                context_attention_fwd(
                    query,
                    key,
                    value,
                    _path_c_output,
                    self.kv_cache_dtype,
                    _path_c_key_cache,
                    _path_c_value_cache,
                    prefill_meta.block_tables,
                    _path_c_query_start_loc,
                    attn_metadata.seq_lens_tensor,
                    None,                             # max_seq_len: kernel 自己算
                    _path_c_max_query_len,
                    _path_c_k_scale,
                    _path_c_v_scale,
                    None,                             # alibi_slopes: 不用
                    None,                             # sliding_window: 不用
                )
                return _path_c_output.view(-1, _path_c_h_q * _path_c_d_q)
            # ===== END PATH-C ===== 下面是原版 fresh prefill 路径(不变)
            # Prompt run.
            # normal attention and DECODER
            if self.attn_type == AttentionType.DECODER and (
                    kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):"""

if ANCHOR not in text:
    print("ERROR: 找不到锚点")
    raise SystemExit(1)

text = text.replace(ANCHOR, REPLACEMENT)
with open(SRC, "w") as f:
    f.write(text)
print("✓ Path C patch 写入完成")
