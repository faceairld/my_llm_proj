"""Store fresh-prefill K/V per layer, then compare Path A cache reads.

This directly tests whether Path A's cached K/V reconstruction matches the
original K/V from the first full prefill for the same layer.

All insertions are marked in flash_attn.py with CODEX MOD START/END.
"""

SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC, "r") as f:
    text = f.read()

if "REF-KV-DIAG 2026-05-21" in text:
    print("already patched")
    raise SystemExit(0)

anchor = """        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA ====="""

insert = """        # ==== CODEX MOD START: REF-KV-DIAG 2026-05-21 store fresh K/V by layer ====
        if prefill_meta := attn_metadata.prefill_metadata:
            try:
                _rkv_fresh = (
                    key is not None and value is not None
                    and attn_metadata.num_prefill_tokens == int(prefill_meta.seq_lens_tensor.sum().item())
                    and attn_metadata.num_prefill_tokens > 0
                )
                if _rkv_fresh:
                    _rkv_refs = getattr(torch, '_codex_ref_kv_by_layer', None)
                    if _rkv_refs is None:
                        _rkv_refs = {}
                        torch._codex_ref_kv_by_layer = _rkv_refs
                    _rkv_lid = id(layer)
                    if _rkv_lid not in _rkv_refs:
                        # Keep one full fresh prefill reference per layer/process.
                        _rkv_refs[_rkv_lid] = (
                            key.detach().clone().contiguous(),
                            value.detach().clone().contiguous(),
                        )
                        _rkv_count = getattr(torch, '_codex_ref_kv_store_prints', 0)
                        if _rkv_count < 4:
                            import os as _rkv_os
                            torch._codex_ref_kv_store_prints = _rkv_count + 1
                            print(
                                f"[CODEX REF-KV-DIAG store pid={_rkv_os.getpid()} count={_rkv_count}] "
                                f"layer_id={_rkv_lid} key_shape={tuple(key.shape)} value_shape={tuple(value.shape)}",
                                file=sys.stderr, flush=True)
            except Exception as _rkv_e:
                print(f"[CODEX REF-KV-DIAG store ERROR] {_rkv_e}",
                      file=sys.stderr, flush=True)
        # ==== CODEX MOD END: REF-KV-DIAG 2026-05-21 store fresh K/V by layer ====

        if prefill_meta := attn_metadata.prefill_metadata:
            # ===== PATH-A 2026-05-21: prefix-cache 命中,Python 层 concat + SDPA ====="""

if anchor not in text:
    print("ERROR: store anchor not found")
    raise SystemExit(1)
text = text.replace(anchor, insert, 1)

anchor = """                        _pa_cached_v = _pa_value_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        # 转置后填入: (cl, h_kv, d) -> (h_kv, cl, d)"""

insert = """                        _pa_cached_v = _pa_value_cache[_pa_blk_ids].reshape(-1, _pa_h_kv, _pa_d_kv)[:_pa_cl]
                        # ==== CODEX MOD START: REF-KV-DIAG 2026-05-21 compare cached K/V to fresh reference ====
                        try:
                            _rkv_refs = getattr(torch, '_codex_ref_kv_by_layer', {})
                            _rkv_ref = _rkv_refs.get(id(layer))
                            _rkv_cmp_count = getattr(torch, '_codex_ref_kv_cmp_prints', 0)
                            if _rkv_ref is not None and _rkv_cmp_count < 8:
                                import os as _rkv_os
                                torch._codex_ref_kv_cmp_prints = _rkv_cmp_count + 1
                                _rkv_ref_k, _rkv_ref_v = _rkv_ref
                                _rkv_n = min(_pa_cl, _rkv_ref_k.shape[0])
                                _rkv_dk = (_pa_cached_k[:_rkv_n] - _rkv_ref_k[:_rkv_n]).abs().max().item()
                                _rkv_dv = (_pa_cached_v[:_rkv_n] - _rkv_ref_v[:_rkv_n]).abs().max().item()
                                print(
                                    f"[CODEX REF-KV-DIAG compare pid={_rkv_os.getpid()} count={_rkv_cmp_count}] "
                                    f"layer_id={id(layer)} cached_len={_pa_cl} n={_rkv_n} "
                                    f"diff_k={_rkv_dk} diff_v={_rkv_dv}",
                                    file=_pa_sys.stderr, flush=True)
                        except Exception as _rkv_e:
                            print(f"[CODEX REF-KV-DIAG compare ERROR] {_rkv_e}",
                                  file=_pa_sys.stderr, flush=True)
                        # ==== CODEX MOD END: REF-KV-DIAG 2026-05-21 compare cached K/V to fresh reference ====
                        # 转置后填入: (cl, h_kv, d) -> (h_kv, cl, d)"""

if anchor not in text:
    print("ERROR: compare anchor not found")
    raise SystemExit(1)
text = text.replace(anchor, insert, 1)

with open(SRC, "w") as f:
    f.write(text)

print("patched ref K/V diagnostics")
