
attention计算你用Tensor core了吗？你用SASS写的？没写，那你怎么用的Tensor core？调用的cublas？那attention计算XXX(这里面试官有个疑问我记不清了)flash attention做了什么优化，简单介绍一下flash attention的原理和优势，你没看过flash attention的论文是吗？(还说到GEMM的M和N的划分什么的,但是不太记得具体是什么了，和flash attention有些联系)，onlinesoftmax优化了什么？flash decoding为什么会有，为什么不参照flash attention思路做decoding？为什么不用之前普通的decoding算法呢？block的grid的维度设计你是怎么设计的？你的v2相比于v1的改进是什么？flash attentionv1相比于v2的改进是什么？

2.你之前是做架构比较多是吗？简单介绍一下qwen2.5的架构，（介绍完后）是不是最开始少了3个linear，分别生成qkv的？那个不是加在attention模块前面的，还有RoPE。你说你手写了RMSnorm的算子，能说一下这个的为什么做手写的替换吗？是出于什么考虑呢？cudakernel的瓶颈一般是哪两种（计算，访存），RMS是哪种类型的？我看你还写的上面的page attention，这个你能介绍一下吗？（vllm的block划分…………）你说你用过NSYS，具体是怎么用的（用NVTX打标签，执行exe的时候启用NSYS查看），那你没用NSYS看过具体一个kernel吗？（那个我是用NCU看的性能，看占用了多少thread，一个grid执行不够分多个wave）

你能介绍一下cuda一般对于来讲有哪些优化（讲到内存的连续性），内存连续性这里是怎么优化的，L1cache的连续搬运机制是什么样的？那需要threads连续访问存储吗？ （讲到shared mem用来存放需要大量重复计算的数据 ），哪些情况下用shared mem？（用GEMM的分块举例，提了一下反量化是不是可以先读取出来反量化完成后放回shared mem，再进行计算的时候读出来计算，以减少反量化的计算量），不是的，这样反量化就失去意义了，(这里面试官说把fp8的weight放到Tensor core里和激活值计算得到fp16，最后反量化到fp32什么的，具体我也记不清了，还是说是Tensor输出fp8反量化到fp16) 继续吧，(讲到shared mem的bank conflict)，介绍一下bank conflict的细节，你确定是16个bank吗？介绍一下GPU的架构，（这里挑了3060讲，SM，sub子块一个SM有四个，reg的容量，调度器，sharedmem和L1cache，内部的执行单元FP32，Tensor core，LD/ST单元，特殊功能单元，一个block最大1024个thread）介绍一下grid和block在GPU上的映射，如何执行的，SM上执行的block的数量受到什么的限制（reg，shared mem，thread数目，调度器）

写一个简单的kernel（vector加法）

反问面试情况，面试官说还行，你是不是没做过一个kernel优化到极致的工作，但是你cuda和gpu的了解还可以。
之后问了HR还是通过一面了。