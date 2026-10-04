-block_pool.py
   -kv_cache_utils.py






block_pool.py解析
1.参数声明：
    Block:vllm设置的显存最小分配单位，由多个token组成
    BlockHash:由block内部的几个token数值计算出的hash值
    BlockHashList:一个BlockHash列表，或者是一个BlockHashWithBlockSize类，这个类负责在target_block_size大小超过hash_block_size时通过拼接hash_block_size的hash值组成target_block_size和hash值，主要是用在target_block_size与hash_block_size不一致的情况下。
    BlockHashWithGroupId:添加了KV cache group ID的BlockHash值，不同的kv cache group下的相同BlockHash也会有所不同
    ExernalBlockHash:对外部接口的block hash类型，与block pool内部采用的hash有所不同：
   
      三种不同的hash类型解析：
         BlockHash
            内容 hash
            类型上是 bytes
            用于表示“这个 token block 的内容指纹”

         BlockHashWithGroupId
            内部查表 key
            内容是 block_hash + group_id bytes
            用于 BlockPool 内部 prefix cache dict

         ExternalBlockHash
            对外事件格式
            类型是 bytes | int
            用于 BlockStored / BlockRemoved 等事件


      FreeKVCacheBlockQueue:维护Free未使用的block的队列，内部维护了一个所有Free block按照block id依次连接的双向链表，包括成员方法:
         popleft:pop出最左侧的一个Free block
         popleft_n:pop出左侧n个Free block
         remove:移出一个指定的block
         append:将一个block添加到双向链表尾部
         append_n:将n个block添加到双向链表尾部
         get_all_free_blocks:返回目前的Free block列表

      KVCacheBlocks:基本KVCache block类，包括一些基本参数
         block_id:每个block的id
         ref_cnt:调用次数，主要用于在book_pool中记录各个block的使用次数，用于block驱逐策略
         _block_hash:为BlockHashWithGroupId类型，仅在block被填充后有效，否则为None
         prev_free_block:指向上一个block的指针，仅在block在FreeKVCacheBlockQueue中时有效，否则为None
         next_free_block:指向下一个block的指针，仅在block在FreeKVCacheBlockQueue中时有效，否则为None
         is_null:用于标注填充block，该标注生效时，block被标注为null block，不被填充，用于在需要空block的位置
         
         成员方法:
         block_hash:返回_block_hash值
         block_hash = xx:查找当前_block_hash值，如果未被填充，则可赋值
         reset_hash:清空_block_hash值

      generate_block_hash_extra_keys:参照一些其他的外部情况，为每个block生成特定环境下的key，计算参数包括多模态特征，LoRA，cache salt，prompt embedding ，这里只计算额外的key，不考虑block本身填充的hash
      get_block_hash:从给出BlockHashWithGroupId中提取出原始的block_hash值，BlockHashWithGroupId一般由block_hash和后4byte大小的group_id组成，这里就直接返回最后四个byte前面的数值，即为block_hash
      get_group_id:从给出的BlockHashWithGroupId中恢复group_id，取后4个bytes，调用int.from_bytes方法解耦成整数
      make_block_hash_with_group_id:将block_hash和group_id拼接成BlockHashWithGroupId，其中group_id也通过to_bytes方法从整数转化为byte类型
      maybe_convert_block_hash：将内部的block_hash值转化为对外事件系统的hash表示，具体操作为，如果没启用VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES环境变量，则直接输出，如果未启用，则将block_hash解耦为int类型整数并只保留低64位数据输出。
      Request:一条推理请求在 vLLM engine 内部的运行时状态对象,包括
         1. 请求基本信息
            request_id、priority、arrival_time、client_index

         2. 生成 / pooling 参数
            sampling_params、pooling_params、max_tokens

         3. 输入输出 token
            prompt_token_ids、output_token_ids、all_token_ids

         4. KV cache / prefix cache 信息
            block_hashes、num_cached_tokens、num_computed_tokens

         5. 多模态 / LoRA / prompt embeds / cache salt
            mm_features、lora_request、prompt_embeds、cache_salt

         6. 调度状态
            WAITING、RUNNING、PREEMPTED、FINISHED...

         7. streaming 和事件
            resumable、streaming_queue、events

         Request:
            管“这条请求是什么、有哪些 token、算到哪里了、hash 是什么”

         BlockPool:
             管“KV block 怎么分配、释放、缓存、驱逐”
      其他的补充:
      hash_block_tokens:计算block_hash的函数，使用parent_block_hash先前block的hash值, curr_block_token_ids_tuple当前block的hash值, extra_keys额外key

2.包含类与方法
   BlockHashToBlockMap:内部维护一个key:BlockHashWithGroupId ,value:KVCacheBlock | dict[int, KVCacheBlock]的字典，为主要的Block的查找表，成员函数有
      get_one_block:通过BlockHashWithGroupId查找对应的KVCacheBlock对象或者对象字典（在多个block有同一个BlockHashWithGroupId的情况下）
      insert:在字典中插入一个新的block，依照是否存在相同BlockHashWithGroupId有不同处理
      pop:从block查找表中pop出一个block，通过BlockHashWithGroupId和block_id定位
      _unexpected_blocks_type:报错函数，在bug产生时启用
   BlockPool：主要负责驱逐block和cached block队列的维护，输入参数包括num_gpu_blocks总block数量，enable_caching是否启用prefix caching，hash_block_size hash值对应的block大小，一般等于block size，也可为整数倍，enable_kv_cache_events是否启用event记录，metrics_collector跟踪block情况的收集器。
      init:构建了free_block_queue队列，设置block的查找表（调用FreeKVCacheBlockQueue，BlockHashToBlockMap），设置block0为null_block用于空白填充，其他设置了enable_kv_cache_events，kv_event_queue，metrics_collector等参数
      get_cached_block:输入block_hash值和group_id，计算出block_hash_with_group_id，在block的查找表中查找对应的block返回
      cache_full_blocks:将一段block cached并添加到block查找表中，其具体流程为    
         1.先计算request中cached的block并截取需要被存储的部分，再计算request中的hash值（依照实际block_size和hash_block的大小差距区分计算）并截取需要被存储的部分，然后通过一个循环循环计算每个block_hash_with_group_id并存入block的block_hash中，再在block查找表中登记，如果启用event记录，则在对外事件的hash列表中添加该block_hash的对外事件hash值
         2.以上完成后，如果启用event记录，则对num_cached_blocks之前的block也计算对外事件的hash值，随后对每个block，调用generate_block_hash_extra_keys计算extra_keys，并存储在extra_keys_list中，最后把extra_key和其他参数打包存储到kv_event_queue中
      get_new_blocks:从free_block_queue中提取出指定个数的block，如果启用prefix cache，则先调用_maybe_evict_cached_block清空内容，然后统一做ref_cnt计数和metrics_collector的记录，如果启用了block的状态跟踪，最后返还取出的block列表
      _maybe_evict_cached_block:将block查找表中对应的block驱逐，并清空对应block的hash值，然后如果启用enable_kv_cache_events，则记录该事件
      touch:接受一组block序列，如果ref_cnt为0，则未命中过，从free_block_queue中pop出，然后将其ref_cnt加1，当启用metrics_collector时则记录该事件
      free_blocks:接收一组计划驱逐的block序列，将其ref_cnt减1，当其ref_cnt为0时驱逐回free_block_queue中但不清空其hash值
      evict_blocks:输入一组block_id序列，对齐对应block调用_maybe_evict_cached_block函数
      reset_prefix_cache:清空前缀缓存的所有block的hash值，当所有block全部被free（但没有被清空hash值）时，可以执行该函数，该函数会清空所有block hash值并重新导入空的BlockHashToBlockMap表
      get_num_free_blocks:返回free_block_queue的block数
      get_usage:返回已经被使用的block占总block的占比
      take_events:返回所用event记录




single_type_kv_cache_manager.py 解析

1. 文件定位
   在 vLLM v1 的 KV cache 调用链里属于中间层：
       KVCacheManager           ← 对外接口，scheduler 调它
           ↓
       KVCacheCoordinator       ← 管理多个 group（混合注意力模型）
           ↓
       ★ SingleTypeKVCacheManager ← 每个 "attention 类型" 一个实例（本文件）
           ↓
       BlockPool                ← 物理 block 仓库
           ↓
       KVCacheBlock（kv_cache_utils.py）

   核心定位：模型里如果只有 full attention，就只用 1 个 FullAttentionManager；像 Gemma2/Mistral 这种混合了 sliding window、Sink、Mamba、Cross attention 的模型，每种 attention 类型的 KV cache 释放/复用策略不一样，就需要为每种类型起一个 manager。本文件给出"每种 attention 类型 → 一个 KV cache 管理策略类"的完整集合。

2. 内部结构

   2.1 抽象基类 SingleTypeKVCacheManager（L28）
      所有具体策略的父类，定义统一接口。关键状态：
         req_to_blocks: dict[req_id, list[KVCacheBlock]]   每个请求占了哪些 block
         num_cached_block: dict[req_id, int]               每个请求已经被 prefix cache 缓存的 block 数
         block_pool                                         持有底层 block 池的引用
         _null_block                                        占位用的 "空 block"（用于 sliding window 跳过的位置）

      主要方法：
         get_num_blocks_to_allocate（L78）       算出请求这一步还需要新申请几个 block（scheduler 用它判断够不够）
         allocate_new_computed_blocks（L142）    把 prefix cache 命中的 block "认领" 过来
         allocate_new_blocks（L215）             从 block_pool 申请新的空白 block
         cache_blocks（L250）                    把已经写满的 block 注册到 prefix cache（哈希入表）
         free（L276）                            请求结束时释放所有 block（逆序还给 free queue）
         remove_skipped_blocks（L358）           把已经移出 attention window 的 block 换成 null block 并释放
         get_num_skipped_tokens（L401）          默认返回 0（full attention 不丢），子类按窗口策略覆盖
         find_longest_cache_hit（L311, abstract）给定 hash 链，找最长的 prefix cache 命中
         get_num_common_prefix_blocks（L294, abstract）算所有 running request 共享的前缀长度（用于 cascade attention）

   2.2 各种 attention 类型的具体子类

      FullAttentionManager（L419）
         对应 spec：FullAttentionSpec / MLAAttentionSpec
         特点：普通 full attention，永远不丢 block；find_longest_cache_hit 顺着 hash 链一直匹配

      SlidingWindowManager（L480）
         对应 spec：SlidingWindowSpec
         特点：窗口外的 block 可丢；get_num_skipped_tokens 返回窗口外的 token 数；find_longest_cache_hit 反向扫描，允许中间 miss

      ChunkedLocalAttentionManager（L619）
         对应 spec：ChunkedLocalAttentionSpec
         特点：按 chunk 切的局部 attention（Llama4 用），只在 chunk 边界处考虑命中

      MambaManager（L769）
         对应 spec：MambaSpec
         特点：Mamba 的 state cache（不是 KV），只占 1~2 个固定 block，几乎重写了所有方法

      CrossAttentionManager（L1042）
         对应 spec：CrossAttentionSpec
         特点：编码-解码模型的 encoder KV；不支持 prefix caching（每个请求的 encoder 输出独立）

      SinkFullAttentionManager（L1091）
         对应 spec：SinkFullAttentionSpec
         特点：继承 full attention，额外预留 "sink" block 永不释放（StreamingLLM 风格）

   2.3 工厂（文件末尾）
      spec_manager_map = {
          FullAttentionSpec: FullAttentionManager,
          MLAAttentionSpec: FullAttentionManager,
          SlidingWindowSpec: SlidingWindowManager,
          ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
          MambaSpec: MambaManager,
          CrossAttentionSpec: CrossAttentionManager,
          SinkFullAttentionSpec: SinkFullAttentionManager,
      }
      get_manager_for_kv_cache_spec(kv_cache_spec, **kwargs)：上层 KVCacheCoordinator 拿到模型的 kv_cache_groups 后，对每个 group 调一次这个工厂，就会得到该 group 对应的 manager。

3. 建议的阅读顺序
   1）基类 SingleTypeKVCacheManager 的 __init__、allocate_new_blocks、free、cache_blocks —— 抓住"申请/释放/缓存"三件事
   2）FullAttentionManager —— 最简单的特例，看清楚 find_longest_cache_hit 怎么逐 hash 查 block_pool.get_cached_block
   3）SlidingWindowManager —— 对比看 sliding window 怎么靠 null_block 占位、怎么实现"窗口外 block 复用"
   4）其它子类按需扩展（Qwen2.5-0.5B 是纯 full attention，通常只会触发 FullAttentionManager 这条路径）

   读完之后，再回头看 kv_cache_manager.py 和 kv_cache_coordinator.py，就能看清楚"多 group 的请求，如何并行触发多个 single-type manager 的 allocate/free"。
