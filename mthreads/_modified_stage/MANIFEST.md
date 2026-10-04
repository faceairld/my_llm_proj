# 修改文件清单（相对 `/data/my_vllm_test.tar.gz`，2026-06-16 10:46 那版）

本压缩包收录的是 **当前 `/data/my_vllm_test` 文件夹里、相对上述基线压缩包「同名但内容已被修改」的 6 个文件**。
（新增文件已在另一个包 `my_vllm_test_extra_files_20260617.tar.gz` 里；基线里没有任何文件被删除。）
所有路径均相对于 `my_vllm_test/` 根目录，解压后可直接对应原目录结构。

打包日期：2026-06-17

---

## 修改文件（同名内容已变）

| 路径（相对 `my_vllm_test/`） | 说明 / 改了什么 |
|---|---|
| `vllm_020/lmcache_on_vllm_musa_020_feasibility.md` | LMCache 移植到新版 vllm_musa 的可行性文档。自基线后大幅扩充：组件调用链、移植排查（torchada/mate 运行时包对齐）、从零迁移 runbook、故障速查表、最终交接状态，以及新增的 §3.1「读路径 vs 写路径」机制说明 |
| `vllm_musa_proj/reademe.md` | vLLM 多模型压测项目主说明文档。新增/更新了 PD 分离（§7.7.2 `--pd`）、多卡 TP（§7.7.1 `--tp`）等章节 |
| `vllm_musa_proj/auto_bench/auto_bench.py` | auto_bench 压测编排主脚本。做了 PD 分离（prefill/decode disaggregation）改造（改造前版本见另一个包里的 `_backup_20260616_prePD/auto_bench.py`） |
| `vllm_musa_proj/auto_bench/run_list.py` | 服务启动编排脚本。PD 分离改造（改造前版本见 `_backup_20260616_prePD/run_list.py`） |
| `vllm_musa_proj/auto_bench/test_list.py` | 压测用例列表脚本。PD 分离改造（改造前版本见 `_backup_20260616_prePD/test_list.py`） |
| `vllm_musa_proj/auto_bench/__pycache__/run_list.cpython-310.pyc` | `run_list.py` 的 Python 编译缓存（自动生成，可忽略） |

---

> 配套：新增文件包 `my_vllm_test_extra_files_20260617.tar.gz`（16 个新增文件，含 ISSUE2 主文档、冷热分离脚本、PD 相关脚本与备份）。两个包合起来 = 相对基线的完整 delta（新增 + 修改），删除项为 0。
