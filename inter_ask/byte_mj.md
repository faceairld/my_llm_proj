简历为：E:\DOCUMENT---------------------\recently_using\简历\高杨_简历_infra.pdf

exllamav2项目：
关于kernel launch过载；是否看过融合算子内部的实现？你是怎么查找是哪个位置出现问题的？at是什么库？ gemm和gemv的区别与改进，怎么找到问题的，如何定位问题的，实际推理过程中不发射这么多kernel吧？kernel 300个算很多了，它内部没有启用pytorch的算子融合吗？还有你说的同步，他每次kernel launch应该都不会同步吧？那你说的那个同步耗时不能考虑吧？你说它是在M = 1的时候走decode路径会出现性能下降这种情况，那如果decode生成不是1呢？那这种情况怎么考虑？

vLLM MUSA 多卡长 prompt 卡死问题：
你说每次报错都不同，这是什么原因？是怎么设置的checkpoint的呢？checkpoint是怎么排查错误的？然后他这个每次新增加的需要计算的new_q_len长度一样吧？(不一样吧？)那他设置cuda_graph也不行吧？是这样吗？(我记得vllm可以通过判断不同的seq_len的计数来判断是否设置cuda_graph,这个思路可以吗？)，那这样每个长度存cuda_graph不是会很占用显存吗？(那你觉得实现思路是什么？)能不能在可以设置cuda_graph的部分设置cuda_graph，跳过需要动态设置的部分？(好像可以)你简历上写的从paged KV cache 拉取是什么意思？(就是vllm page attention的实现，他会把prefix cache hit的kv分block，然后可以通过block table查找对应的block)

个人项目：
项目经历上说插入了RMS Normal的算子，你是处于什么目的尝试插入这个算子呢，我看你说为了做kernel融合，但是相比于pytorch complier的融合，你这里是为什么这么做呢(我pytorch的实现是朴素的实现，可能没开complier的融合，然后我的RMSnormal可能就是把朴素实现里两次读取input给融合了一下)，GQA又是出于什么目的写的自己的cuda算子呢(GQA我是想用一下flash attention v2的思路去做一下)，GQA相比于MLA和MHA有什么优势呢(GQA是一组q对应一组kv，相比于MLA，kv占用的显存资源更少，性能下降不多，MLA所有q公用一组kv，性能下降很严重，所以GQA相当于做了取舍)既然你说了flash attention v2，那就简单介绍一下这个吧，v3有了解吗(好像是对于Hopper架构做了特定的优化，用TMA搬运，类似DMA的思路，绕过threads直接搬运显存到shared mem，还有一个好像是异步的计算，具体记不清了，v4没太了解过)

手撕：rmsnormal or 伪代码的 online softmax

rmsnormal是访存密集型的，有什么优化方式吗？(没答出来)用half2 half4这种读取。


建议：还可以更加熟练一些，具体的评价设计面试评价不好透露