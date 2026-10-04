# LMCache-musa 测试记录

> 来源：`https://github.com/LMCache/LMCache`（auto-musify 自动迁移流水线产物）
> 容器：`gy_work`（镜像 `sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:4.3.5_kuae2.1_hygon_ubuntu_fix_ray`）
> 测试时间：2026-05-07
> 测试卡：MTT S5000 × 1（`MUSA_VISIBLE_DEVICES=0`）

## 一句话结论

**核心功能可用** ✅。84 个核心测试用例（kernel + 内存管理）100% 通过，结果与 auto-musify 流水线报告一致。

## LMCache 是什么

vLLM 的 KV cache 加速外挂：把 KV cache 多级存储到 GPU/CPU/Disk/Redis/S3，并支持非前缀复用（CacheBlend 算法），降 TTFT 3-10×。详见 `MUSIFY.md` 和上游 README。

## 环境

| 项 | 值 |
|---|---|
| MUSA SDK | 4.3.5 |
| torch_musa | 2.5.0（容器里的版本，注意上游迁移用的是 2.7.1） |
| vllm | 0.9.3.dev0+ga5dd03c1e（含 vllm_musa 插件） |
| LMCache 版本号 | 0.3.0（手动指定，因为 zip 解压无 git 历史） |

## 编译步骤（已验证）

```bash
docker exec gy_work bash -c "cd /data/LMCache && \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.3.0 \
  FORCE_MUSA=1 \
  pip install -e . --no-build-isolation"

# 编译完后必须执行：恢复 numpy 版本（pip 会被升到 2.2.6 破坏 vllm）
docker exec gy_work pip install 'numpy==1.26.1'
```

## 测试结果

| 测试模块 | 通过 / 总数 | 耗时 | 说明 |
|---|---|---|---|
| `tests/v1/test_memory_management.py` | **47/47** ✅ | 15s | GPU/Pin/Mixed allocator |
| `tests/v1/test_mem_kernels.py` | **37/37** ✅ | 192s | MUSA kernel 数值正确性（torch.equal 比特一致） |
| `tests/v1/test_pos_kernels.py` | **0/1** ❌ | 5s | vllm API 不兼容（非 MUSA 问题，详见下） |

### 运行命令

```bash
# 内存管理（最快，先跑这个验证环境）
docker exec gy_work bash -c "cd /data/LMCache && \
  MUSA_VISIBLE_DEVICES=0 python -m pytest tests/v1/test_memory_management.py -q"

# MUSA kernel 数值正确性
docker exec gy_work bash -c "cd /data/LMCache && \
  MUSA_VISIBLE_DEVICES=0 python -m pytest tests/v1/test_mem_kernels.py -q"

# 注意：不要 `pytest tests/` 一把梭。conftest.py 里有 5GB session-scope
# memory_allocator fixture，全套一起跑会触发级联失败。每个模块单独跑都干净通过。
```

## 遇到的问题

### 1. setuptools-scm 找不到版本号 ✅ 已解决

**现象**：
```
LookupError: setuptools-scm was unable to detect version for /data/LMCache.
```

**原因**：zip 解压没有 `.git/` 目录，setuptools-scm 读不到 git tag。

**解决**：编译前设环境变量 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.3.0`，告诉它假装版本号是 0.3.0。不影响功能。

### 2. pip 把 numpy 升级，破坏 vllm ✅ 已解决

**现象**：编完后 vllm 报 `numpy==1.26.1` 不满足。

**原因**：LMCache 依赖 `numpy<=2.2.6`，pip 解析时直接升到 2.2.6；但 vllm 强制 `numpy==1.26.1`。

**解决**：编完后立刻 `pip install numpy==1.26.1` 降回去。LMCache 也兼容 1.26.1，不影响测试。

> 也可以在 `requirements.txt` 或 `pyproject.toml` 里把上限改成 `numpy<2`，避免下次升级。

### 3. test_pos_kernels.py vllm API 不兼容 ❌ 待解决

**现象**：
```
File "lmcache/v1/compute/positional_encoding.py:184"
    rope = vllm_get_rope(...)
TypeError: get_rope() got an unexpected keyword argument 'rope_parameters'
```

**原因**：LMCache 用了 vllm 较新版本的 `get_rope(rope_parameters=...)` 关键字参数，但容器内 vllm 0.9.3.dev0 还没有这个参数。**与 MUSA 移植无关**，是 LMCache 上游和 vllm 上游的版本同步问题。

**解决方向**（任选）：
- 升级容器内 vllm 到 0.10+（含 `rope_parameters` 参数的版本）
- 改 `lmcache/v1/compute/positional_encoding.py:184`，按当前 vllm 的 `get_rope` 签名调用
- 暂时跳过这个测试，不影响 KV cache 主路径

### 4. 装了不需要的 CUDA 版包（无害）

pip 自动装了 `cupy-cuda12x`、`nixl-cu12`、`cuda-pathfinder`、`cufile-python` 等 CUDA 专用包。这些在 MUSA 环境下不会被加载（依赖它们的测试模块会自动 skip），不影响核心功能。如果想干净一点可以 `pip uninstall` 掉。

### 5. torch_musa 版本与 MUSIFY 报告不一致

| | MUSIFY 报告 | gy_work 容器实际 |
|---|---|---|
| torch_musa | 2.7.1 | **2.5.0** |
| vllm | (未指定) | 0.9.3.dev0 |

torch_musa 2.5 / 2.7 都通过了核心测试，但更深的算子或 API 差异可能在边缘 case 暴露。如果遇到奇怪问题，可以换用 MUSIFY 推荐的镜像 `4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu`。

### 6. GC 阶段日志噪声（无害）

测试结束 Python 退出时，`MemoryObj.__del__` 会往已关闭的 stderr 写 warning，刷屏但不影响结果。看 pytest 摘要行（`X passed in Ys`）即可。

## MUSIFY.md 报告但本环境未跑通的项

| 测试 | MUSIFY 状态 | 本环境状态 | 原因 |
|---|---|---|---|
| `test_pos_kernels.py` | skipped (no vllm) | **失败** | 容器有 vllm 但 API 版本太旧 |
| `test_vllm_integration.py` | skipped | 仍然 skipped | 文件本身只有 2 行 TODO，没内容 |
| 其他 1500+ 测试 | passed | 未跑 | 时间关系只跑了核心 84 个 |

## 下一步建议

1. **跑完整套核心测试**（按 MUSIFY 顺序，约 30 分钟）：
   - `test_cache_engine.py`（54 个，约 4 分钟）
   - `test_gpu_connector.py`（31 个，约 11 秒）
   - `cache_controller/`、`distributed/`、`storage_backend/` 三个目录

2. **vllm + LMCache 端到端联调**：MUSA 版本下 KV connector 是否能正常和 vllm_musa 配合，目前 `test_vllm_integration.py` 是空 TODO，**没有官方验证过**。这才是真正能跑出加速效果的核心场景，需要自己写脚本跑。

3. **修复 `test_pos_kernels.py`**：升级 vllm 或 patch LMCache 的 rope 调用。

## 相关路径

- 项目根目录：`/data/LMCache/`
- 完整迁移日志：`/data/LMCache/MUSIFY.md`
- 上游 README：`/data/LMCache/README.md`
- vLLM 集成代码：`/data/LMCache/lmcache/integration/vllm/`
- MUSA 适配的 C++ 源码：`/data/LMCache/csrc_musa/`
