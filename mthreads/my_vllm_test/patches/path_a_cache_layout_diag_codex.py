"""Add Codex diagnostics for KV cache read layout.

This checks whether the Path A expression
    key_cache[block_ids].reshape(-1, h_kv, d_kv)
actually reconstructs the original key/value tensors that were just written
by reshape_and_cache_flash during fresh prefill.

No math path is changed. Insertions are marked with CODEX MOD START/END.
"""

SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC, "r") as f:
    text = f.read()

if "CACHE-LAYOUT-DIAG 2026-05-21" in text:
    print("already patched")
    raise SystemExit(0)

anchor = """        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA ====="""

insert = """        # ==== CODEX MOD START: CACHE-LAYOUT-DIAG 2026-05-21 ====
        # Verify Path A's KV-cache read layout on fresh prefill. If this diff
        # is nonzero, key_cache[block_ids].reshape(-1, h, d) is not equivalent
        # to the original key/value layout written by reshape_and_cache_flash.
        if prefill_meta := attn_metadata.prefill_metadata:
            try:
                _cld_count = getattr(torch, '_codex_cache_layout_diag_count', 0)
                _cld_fresh = (
                    key is not None and value is not None
                    and kv_cache.numel() > 0
                    and prefill_meta.block_tables is not None
                    and prefill_meta.block_tables.numel() > 0
                    and attn_metadata.num_prefill_tokens == int(prefill_meta.seq_lens_tensor.sum().item())
                )
                if _cld_fresh and _cld_count < 2:
                    import os as _cld_os
                    torch._codex_cache_layout_diag_count = _cld_count + 1
                    _cld_key_cache = kv_cache[0]
                    _cld_value_cache = kv_cache[1]
                    _cld_block_size = _cld_key_cache.shape[1]
                    _cld_seq_lens = prefill_meta.seq_lens_tensor.detach().cpu().tolist()
                    _cld_h_kv = key.shape[1]
                    _cld_d_kv = key.shape[2]
                    _cld_seq0 = int(_cld_seq_lens[0])
                    _cld_num_blk0 = (_cld_seq0 + _cld_block_size - 1) // _cld_block_size
                    _cld_blk_ids0 = prefill_meta.block_tables[0][:_cld_num_blk0]
                    _cld_k_read = _cld_key_cache[_cld_blk_ids0].reshape(-1, _cld_h_kv, _cld_d_kv)[:_cld_seq0]
                    _cld_v_read = _cld_value_cache[_cld_blk_ids0].reshape(-1, _cld_h_kv, _cld_d_kv)[:_cld_seq0]
                    _cld_k_ref = key[:_cld_seq0]
                    _cld_v_ref = value[:_cld_seq0]
                    _cld_dk = (_cld_k_read - _cld_k_ref).abs().max().item()
                    _cld_dv = (_cld_v_read - _cld_v_ref).abs().max().item()
                    print(
                        f"[CODEX CACHE-LAYOUT-DIAG pid={_cld_os.getpid()} count={_cld_count}] "
                        f"key_cache_shape={tuple(_cld_key_cache.shape)} value_cache_shape={tuple(_cld_value_cache.shape)} "
                        f"seq0={_cld_seq0} block_size={_cld_block_size} blocks0={_cld_blk_ids0[:8].detach().cpu().tolist()} "
                        f"diff_k={_cld_dk} diff_v={_cld_dv}",
                        file=sys.stderr, flush=True)
            except Exception as _cld_e:
                print(f"[CODEX CACHE-LAYOUT-DIAG ERROR] {_cld_e}",
                      file=sys.stderr, flush=True)
        # ==== CODEX MOD END: CACHE-LAYOUT-DIAG 2026-05-21 ====

        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA ====="""

if anchor not in text:
    print("ERROR: anchor not found")
    raise SystemExit(1)

text = text.replace(anchor, insert, 1)

with open(SRC, "w") as f:
    f.write(text)

print("patched cache layout diagnostics")
