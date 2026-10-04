"""CLAUDE PATCH 2026-05-27: PH1 + V1 block_size 修复

修改原因:
  vllm_musa 原版在 PH1(S5000)上把 block_size 锁成 64,但 V1 attention
  又必须 block_size=32(同文件 get_attn_backend_cls 检查),导致 PH1+V1 必崩:
    ValueError: On MUSA platform, V1 flash_attn block_size must be 32.

修改方式:
  V1 模式下用 32(跟 QY2+V1 对齐);V0 模式保持 64(已知工作)。

如何回滚:
  方法 1:把 platforms/musa.py 里 "==== CLAUDE PATCH 2026-05-27" 块整个删掉,
          留原版的 'cache_config.block_size = 64' 一行(原版已作为注释保留)
  方法 2:跑 `docker cp` 把备份覆盖回去:
          docker cp /data/_backup_to_local/vllm_musa_src/vllm_musa/platforms/musa.py \\
              gy_work:/usr/local/lib/python3.10/dist-packages/vllm_musa/platforms/musa.py
  (注意 v1 platforms/musa.py 备份位置如有不同请相应调整)

用法: docker exec gy_work python3 /data/my_vllm_test/patches/ph1_v1_block_size_fix.py
"""

SRC = "/usr/local/lib/python3.10/dist-packages/vllm_musa/platforms/musa.py"

with open(SRC) as f:
    text = f.read()

if "CLAUDE PATCH 2026-05-27" in text:
    print("⚠ 已经 patch 过了,跳过")
    raise SystemExit(0)

ANCHOR = """        if cache_config:
            if on_ph1():
                cache_config.block_size = 64"""

REPLACEMENT = """        if cache_config:
            if on_ph1():
                # ==== CLAUDE PATCH 2026-05-27 START: PH1 + V1 block_size fix ====
                # 修改者: Claude(在用户指示下)
                # 修改原因:
                #   vllm_musa 原版在 PH1(S5000)上无脑把 block_size 锁成 64,
                #   但 platforms/musa.py 的 get_attn_backend_cls 又规定 V1 attention
                #   必须 block_size=32(见同文件第 256-258 行),导致 PH1+V1 必崩:
                #     ValueError: On MUSA platform, V1 flash_attn block_size must be 32.
                #   这是 vllm_musa 自己代码两处不一致的 bug(QY2 上 V1 是 32,PH1 上漏了)。
                # 修改方式:
                #   V1 模式下用 32(跟 QY2+V1 对齐),V0 模式保持 64 不变(已知工作)。
                # 风险:
                #   PH1 上 V1 attention 在 block_size=32 上没经 MTT 正式验证,
                #   可能数值上有 bug。如果 V1 启动成功但输出乱码,大概率是这条改动。
                # 如何回滚:
                #   把下面 if/else 块整个删掉,只留下面注释里的 'cache_config.block_size = 64' 一行
                # ===============================================================
                # 原版(已注释):
                # cache_config.block_size = 64
                if envs.VLLM_USE_V1:
                    cache_config.block_size = 32
                else:
                    cache_config.block_size = 64
                # ==== CLAUDE PATCH 2026-05-27 END ===="""

if ANCHOR not in text:
    print("ERROR: 找不到锚点,文件可能跟预期不同(已经被改过或者版本不一致)")
    raise SystemExit(1)

text = text.replace(ANCHOR, REPLACEMENT)
with open(SRC, "w") as f:
    f.write(text)
print("✓ patch 已写入 platforms/musa.py")
