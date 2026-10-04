# vllm_musa + LMCache Code Bundle

Created: 2026-06-18

This bundle is intended to preserve the code-side work for the new vllm_musa 0.20/V1 + LMCache MUSA integration.

## Included

- `my_vllm_test/vllm_020/vllm-musa/`
  - New vLLM 0.20 source tree under `third_party/vllm/vllm/`
  - New `vllm_musa` 0.1.1 source tree
- `my_vllm_test/vllm_020/_dist_info/`
  - vLLM/vllm_musa metadata copied from the working environment
- `my_vllm_test/vllm_020/_remote_deps/`
  - Python runtime dependencies copied from the 127 working environment
- `my_vllm_test/vllm_020/start_qwen3_8b_vllm_musa.sh`
  - Parameterized launcher for pure vLLM and LMCache-enabled vLLM
- `my_vllm_test/vllm_020/lmcache_cold_warm_revisit_bench.py`
  - Cold/warm benchmark used to prove LMCache warm-revisit benefit
- `my_vllm_test/install_vllm020_overlay.sh`
  - Overlay installer for vLLM/vllm_musa source
- `my_vllm_test/install_remote_deps_overlay.sh`
  - Overlay installer for remote Python dependencies
- `_backup_to_local/LMCache/`
  - LMCache 0.3.0 MUSA port source and native `.so` files
- Documentation:
  - `my_vllm_test/vllm_020/vllm_musa_020_architecture.md`
  - `my_vllm_test/vllm_020/lmcache_on_vllm_musa_020_feasibility.md`
  - `my_vllm_test/ISSUE_vllm_musa_broadcast_deadlock.md`
  - `my_vllm_test/ISSUE_vllm_musa_broadcast_deadlock2.md`

## Not Included

- Model weights, especially `/data/SQT-v1.0.5-test/models/qwen3-8b`
- Docker image layers
- Full benchmark result directories unless they are already under `my_vllm_test/vllm_020`
- Running container state

## Restore Notes

This is a code bundle, not a full runnable image. To restore the working runtime:

1. Use a compatible torch2.9 MUSA base image.
2. Ensure `/data/my_vllm_test/vllm_020` and `/data/_backup_to_local/LMCache` are restored to the same paths, or update scripts accordingly.
3. Run:

```bash
bash /data/my_vllm_test/install_vllm020_overlay.sh
bash /data/my_vllm_test/install_remote_deps_overlay.sh
```

4. Verify the MUSA runtime packages:

```text
torchada 0.1.56
mate 0.2.0+mu437torch2.9
flash_attn_3 0.1.4
```

5. Start pure vllm_musa first. Only after pure baseline works should LMCache be enabled.

See `my_vllm_test/vllm_020/vllm_musa_020_architecture.md` and `my_vllm_test/vllm_020/lmcache_on_vllm_musa_020_feasibility.md` for the full runbook and troubleshooting notes.
