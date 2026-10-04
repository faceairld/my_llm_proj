"""Path A patch: Python 层手动从 KV cache 拉 cached K/V + concat new K/V,然后调 SDPA
   绕开 varlen_fa_seqlen_pad/unpad,完全用 PyTorch 操作

   用法(容器内): python3 /data/my_vllm_test/patches/path_a_python_concat.py
   回滚: docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/v0/flash_attn.py 回去

   原理:
     - 检测 prefix-cache 命中
     - 命中时:
       1. 从 kv_cache 用 block_tables 取出 cached K/V
       2. 用 PyTorch 写入 padded buffer:cached 放前半,new 放后半
       3. 调 SDPA(RunFlash,在 B2 实验已验证稳定)
       4. 从 attn_out 抽出对应 new query 位置的输出,packed 返回
     - 不命中时:走原 sdpa_attention_with_kernel_seqlen_pad
"""
SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC) as f:
    text = f.read()

if "PATH-A 2026-05-21" in text:
    print("⚠ 已经 patch 过了")
    raise SystemExit(0)

ANCHOR = """        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            # normal attention and DECODER
            if self.attn_type == AttentionType.DECODER and (
                    kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):"""

REPLACEMENT = """        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA =====
            # 完全绕开 varlen_fa_seqlen_pad/unpad,在 Python 层手动从 KV cache 拉 cached K/V,
            # 拼到 new K/V 前面,然后调 SDPA(RunFlash 已知稳定)
            _path_a_hit = (
                self.attn_type == AttentionType.DECODER
                and kv_cache.numel() > 0
                and prefill_meta.block_tables is not None
                and prefill_meta.block_tables.numel() > 0
            )
            if _path_a_hit:
                import sys as _pa_sys
                if not getattr(torch, '_path_a_announced', False):
                    print("[PATH-A] prefix-cache hit branch active (Python concat + SDPA)",
                          file=_pa_sys.stderr, flush=True)
                    torch._path_a_announced = True

                _pa_h_q = query.shape[1]
                _pa_h_kv = key.shape[1]
                _pa_d_q = query.shape[2]
                _pa_d_kv = key.shape[2]

                _pa_key_cache = kv_cache[0]      # shape (num_blocks, block_size, h_kv, d_kv)
                _pa_value_cache = kv_cache[1]
                _pa_block_size = _pa_key_cache.shape[1]

                _pa_block_tables = prefill_meta.block_tables   # (bs, max_blocks_per_seq)
                _pa_seq_lens_tensor = attn_metadata.seq_lens_tensor   # (bs,) 完整序列长度
                _pa_bs = _pa_block_tables.shape[0]

                # query_start_loc: cumsum of NEW token lens (优先用 attn_metadata)
                _pa_query_start_loc = getattr(attn_metadata, 'query_start_loc', None)
                if _pa_query_start_loc is None:
                    # fallback: 单 batch 情况,所有 query 都是新的
                    _pa_query_start_loc = torch.tensor([0, query.shape[0]],
                                                       device=query.device, dtype=torch.int32)

                # 拿 host-side int 用于 loop
                _pa_seq_lens_cpu = _pa_seq_lens_tensor.cpu().tolist()
                _pa_query_start_cpu = _pa_query_start_loc.cpu().tolist()
                _pa_new_lens = [_pa_query_start_cpu[b+1] - _pa_query_start_cpu[b] for b in range(_pa_bs)]
                _pa_cached_lens = [_pa_seq_lens_cpu[b] - _pa_new_lens[b] for b in range(_pa_bs)]
                _pa_max_full_len = max(_pa_seq_lens_cpu)

                # alloc padded buffers,zero-fill 让 attention 数学合法
                _pa_q_pad = torch.zeros((_pa_bs, _pa_h_q,  _pa_max_full_len, _pa_d_q),
                                        device=query.device, dtype=query.dtype)
                _pa_k_pad = torch.zeros((_pa_bs, _pa_h_kv, _pa_max_full_len, _pa_d_kv),
                                        device=query.device, dtype=query.dtype)
                _pa_v_pad = torch.zeros((_pa_bs, _pa_h_kv, _pa_max_full_len, _pa_d_kv),
                                        device=query.device, dtype=query.dtype)

                # 逐 batch 填数据
                for _pa_b in range(_pa_bs):
                    _pa_cl = _pa_cached_lens[_pa_b]
                    _pa_nl = _pa_new_lens[_pa_b]
                    _pa_fl = _pa_seq_lens_cpu[_pa_b]
                    _pa_ns = _pa_query_start_cpu[_pa_b]

                    # 1. 拉 cached K/V 到位置 [0:cached_len]
                    if _pa_cl > 0:
                        _pa_num_blk = (_pa_cl + _pa_block_size - 1) // _pa_block_size
                        _pa_blk_ids = _pa_block_tables[_pa_b][:_pa_num_blk]
                        # key_cache[blk_ids] shape: (num_blk, block_size, h_kv, d_kv)
                        _pa_cached_k = _pa_key_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        _pa_cached_v = _pa_value_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        # 转置后填入: (cl, h_kv, d) -> (h_kv, cl, d)
                        _pa_k_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_k.transpose(0, 1)
                        _pa_v_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_v.transpose(0, 1)

                    # 2. 填新 Q/K/V 到位置 [cached_len:full_len]
                    if _pa_nl > 0:
                        _pa_new_q = query[_pa_ns:_pa_ns+_pa_nl]   # (nl, h_q, d_q)
                        _pa_new_k = key  [_pa_ns:_pa_ns+_pa_nl]   # (nl, h_kv, d_kv)
                        _pa_new_v = value[_pa_ns:_pa_ns+_pa_nl]
                        _pa_q_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_q.transpose(0, 1)
                        _pa_k_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_k.transpose(0, 1)
                        _pa_v_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_v.transpose(0, 1)

                # 调 SDPA(B2 已验证稳定)
                _pa_attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa(
                    _pa_q_pad, _pa_k_pad, _pa_v_pad, dropout_p=0.0, is_causal=True)
                # attn_out shape: (bs, h_q, max_full_len, d_q)

                # 从 attn_out 抽出每个 batch 的新 query 输出位置 [cached_len:full_len]
                _pa_sum_new = query.shape[0]
                _pa_output = torch.empty((_pa_sum_new, _pa_h_q * _pa_d_q),
                                         device=query.device, dtype=query.dtype)
                for _pa_b in range(_pa_bs):
                    _pa_cl = _pa_cached_lens[_pa_b]
                    _pa_nl = _pa_new_lens[_pa_b]
                    _pa_fl = _pa_seq_lens_cpu[_pa_b]
                    _pa_ns = _pa_query_start_cpu[_pa_b]
                    if _pa_nl > 0:
                        # attn_out[b, :, cl:fl, :].transpose(0,1) -> (nl, h_q, d_q) -> reshape (nl, h_q*d_q)
                        _pa_seg = _pa_attn_out[_pa_b, :, _pa_cl:_pa_fl, :].transpose(0, 1).reshape(_pa_nl, _pa_h_q * _pa_d_q)
                        _pa_output[_pa_ns:_pa_ns+_pa_nl] = _pa_seg

                return _pa_output
            # ===== END PATH-A ===== 下面是原版 fresh prefill(无命中)路径
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
print("✓ Path A patch 写入完成")
