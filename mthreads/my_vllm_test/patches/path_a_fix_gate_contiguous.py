"""Patch Path A workaround: only handle real prefix-cache hits and force
contiguous transpose results.

This patch applies to the current Path A version in:
  /usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py

It is intentionally small:
  1. Do not enter Path A for fresh prefill where cached_len == 0.
  2. Use prefill_meta query/seq tensors rather than the outer metadata.
  3. Make transposed tensors contiguous before assignments/reshape.

All inserted/changed code is marked in flash_attn.py with:
  CODEX MOD START
  CODEX MOD END
"""

SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/v0/flash_attn.py"

with open(SRC, "r") as f:
    text = f.read()

if "PATH-A-FIX gate+contiguous 2026-05-21" in text:
    print("already patched")
    raise SystemExit(0)

repls = [
    (
        """                and prefill_meta.block_tables is not None
                and prefill_meta.block_tables.numel() > 0
            )""",
        """                and prefill_meta.block_tables is not None
                and prefill_meta.block_tables.numel() > 0
                # ==== CODEX MOD START: Path A real-hit gate 2026-05-21 ====
                # PATH-A-FIX gate+contiguous 2026-05-21:
                # Only real prefix-cache hits should use Path A. Fresh prefill
                # has num_prefill_tokens == sum(seq_lens), and the original
                # varlen_pad -> SDPA -> unpad path is already correct/stable.
                and attn_metadata.num_prefill_tokens < int(prefill_meta.seq_lens_tensor.sum().item())
                # ==== CODEX MOD END: Path A real-hit gate 2026-05-21 ====
            )""",
    ),
    (
        """                _pa_seq_lens_tensor = attn_metadata.seq_lens_tensor   # (bs,) 完整序列长度""",
        """                # ==== CODEX MOD START: use prefill metadata tensors 2026-05-21 ====
                _pa_seq_lens_tensor = prefill_meta.seq_lens_tensor   # (bs,) 完整序列长度
                # ==== CODEX MOD END: use prefill metadata tensors 2026-05-21 ====""",
    ),
    (
        """                _pa_query_start_loc = getattr(attn_metadata, 'query_start_loc', None)""",
        """                # ==== CODEX MOD START: use prefill query_start_loc 2026-05-21 ====
                _pa_query_start_loc = prefill_meta.query_start_loc
                # ==== CODEX MOD END: use prefill query_start_loc 2026-05-21 ====""",
    ),
    (
        """                        _pa_k_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_k.transpose(0, 1)
                        _pa_v_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_v.transpose(0, 1)""",
        """                        # ==== CODEX MOD START: contiguous cached K/V transpose 2026-05-21 ====
                        _pa_k_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_k.transpose(0, 1).contiguous()
                        _pa_v_pad[_pa_b, :, :_pa_cl, :] = _pa_cached_v.transpose(0, 1).contiguous()
                        # ==== CODEX MOD END: contiguous cached K/V transpose 2026-05-21 ====""",
    ),
    (
        """                        _pa_q_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_q.transpose(0, 1)
                        _pa_k_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_k.transpose(0, 1)
                        _pa_v_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_v.transpose(0, 1)""",
        """                        # ==== CODEX MOD START: contiguous new Q/K/V transpose 2026-05-21 ====
                        _pa_q_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_q.transpose(0, 1).contiguous()
                        _pa_k_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_k.transpose(0, 1).contiguous()
                        _pa_v_pad[_pa_b, :, _pa_cl:_pa_fl, :] = _pa_new_v.transpose(0, 1).contiguous()
                        # ==== CODEX MOD END: contiguous new Q/K/V transpose 2026-05-21 ====""",
    ),
    (
        """                        _pa_seg = _pa_attn_out[_pa_b, :, _pa_cl:_pa_fl, :].transpose(0, 1).reshape(_pa_nl, _pa_h_q * _pa_d_q)""",
        """                        # ==== CODEX MOD START: contiguous output slice before reshape 2026-05-21 ====
                        _pa_seg = _pa_attn_out[_pa_b, :, _pa_cl:_pa_fl, :].transpose(0, 1).contiguous().reshape(_pa_nl, _pa_h_q * _pa_d_q)
                        # ==== CODEX MOD END: contiguous output slice before reshape 2026-05-21 ====""",
    ),
]

for old, new in repls:
    if old not in text:
        print("ERROR: expected snippet not found:")
        print(old)
        raise SystemExit(1)
    text = text.replace(old, new, 1)

with open(SRC, "w") as f:
    f.write(text)

print("patched Path A gate + contiguous")
