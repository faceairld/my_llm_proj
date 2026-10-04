"""Add Codex diagnostics to Path A without changing math.

The diagnostics are intentionally rate-limited per process and marked in
flash_attn.py with CODEX MOD START/END so other agents can find them.
"""

SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC, "r") as f:
    text = f.read()

if "PATH-A-DIAG 2026-05-21" in text:
    print("already patched")
    raise SystemExit(0)

old = """                _pa_max_full_len = max(_pa_seq_lens_cpu)

                # alloc padded buffers,zero-fill 让 attention 数学合法"""
new = """                _pa_max_full_len = max(_pa_seq_lens_cpu)

                # ==== CODEX MOD START: PATH-A-DIAG 2026-05-21 metadata dump ====
                import os as _pa_os
                _pa_diag_count = getattr(torch, '_path_a_diag_count', 0)
                _pa_do_diag = _pa_diag_count < 4
                if _pa_do_diag:
                    torch._path_a_diag_count = _pa_diag_count + 1
                    _pa_bt0 = []
                    try:
                        _pa_num_blk0 = (_pa_cached_lens[0] + _pa_block_size - 1) // _pa_block_size
                        _pa_bt0 = _pa_block_tables[0][:_pa_num_blk0].detach().cpu().tolist()
                    except Exception as _pa_e:
                        _pa_bt0 = ["ERR", str(_pa_e)]
                    print(
                        f"[CODEX PATH-A-DIAG pid={_pa_os.getpid()} count={_pa_diag_count}] "
                        f"num_prefill_tokens={attn_metadata.num_prefill_tokens} "
                        f"seq_lens={_pa_seq_lens_cpu} query_start={_pa_query_start_cpu} "
                        f"new_lens={_pa_new_lens} cached_lens={_pa_cached_lens} "
                        f"block_size={_pa_block_size} block_ids0={_pa_bt0[:8]}",
                        file=_pa_sys.stderr, flush=True)
                # ==== CODEX MOD END: PATH-A-DIAG 2026-05-21 metadata dump ====

                # alloc padded buffers,zero-fill 让 attention 数学合法"""

if old not in text:
    print("ERROR: metadata anchor not found")
    raise SystemExit(1)
text = text.replace(old, new, 1)

old = """                # 调 SDPA(B2 已验证稳定)
                _pa_attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa("""
new = """                # ==== CODEX MOD START: PATH-A-DIAG 2026-05-21 compare Python scatter vs varlen_pad on new-token region ====
                if _pa_do_diag:
                    try:
                        _pa_diag_q = torch.empty_like(_pa_q_pad)
                        _pa_diag_k = torch.empty_like(_pa_k_pad)
                        _pa_diag_v = torch.empty_like(_pa_v_pad)
                        ops.varlen_fa_seqlen_pad(
                            query, key, value,
                            _pa_diag_q, _pa_diag_k, _pa_diag_v,
                            _pa_query_start_loc, _pa_query_start_loc,
                            query.shape[0], max(_pa_new_lens), _pa_bs)
                        _pa_b0 = 0
                        _pa_cl0 = _pa_cached_lens[_pa_b0]
                        _pa_nl0 = _pa_new_lens[_pa_b0]
                        _pa_py_q = _pa_q_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_q = _pa_diag_q[_pa_b0, :, :_pa_nl0, :]
                        _pa_py_k = _pa_k_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_k = _pa_diag_k[_pa_b0, :, :_pa_nl0, :]
                        _pa_py_v = _pa_v_pad[_pa_b0, :, _pa_cl0:_pa_cl0 + _pa_nl0, :]
                        _pa_cpp_v = _pa_diag_v[_pa_b0, :, :_pa_nl0, :]
                        _pa_dq = (_pa_py_q - _pa_cpp_q).abs().max().item() if _pa_nl0 > 0 else -1
                        _pa_dk = (_pa_py_k - _pa_cpp_k).abs().max().item() if _pa_nl0 > 0 else -1
                        _pa_dv = (_pa_py_v - _pa_cpp_v).abs().max().item() if _pa_nl0 > 0 else -1
                        print(
                            f"[CODEX PATH-A-DIAG scatter-diff] q={_pa_dq} k={_pa_dk} v={_pa_dv}",
                            file=_pa_sys.stderr, flush=True)
                    except Exception as _pa_e:
                        print(f"[CODEX PATH-A-DIAG scatter-diff ERROR] {_pa_e}",
                              file=_pa_sys.stderr, flush=True)
                # ==== CODEX MOD END: PATH-A-DIAG 2026-05-21 compare Python scatter vs varlen_pad on new-token region ====

                # 调 SDPA(B2 已验证稳定)
                _pa_attn_out, _, _ = torch.ops.aten._scaled_dot_product_attention_flash_musa("""

if old not in text:
    print("ERROR: SDPA anchor not found")
    raise SystemExit(1)
text = text.replace(old, new, 1)

with open(SRC, "w") as f:
    f.write(text)

print("patched Path A diagnostics")
