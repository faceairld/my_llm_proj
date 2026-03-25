#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cfloat>


template <typename scalar_t>
__global__ void my_decode_attention(
    const scalar_t *k_cache,            //batch_size, 2, seq_len(x), head_dim
    const scalar_t *v_cache,            //batch_size, 2, seq_len(x), head_dim
    const scalar_t *q,                  //batch_size, 14, seq_len(1), head_dim
    scalar_t *out_attention,
    const int *current_seq_len,
    int cache_len
    // int repeat_times
){
    // extern __shared__ scalar_t qk_data[];
    // extern __shared__ float soft_max[16];
    extern __shared__ char shared_raw[];
    float *qk_data = (float*)shared_raw;
    float *soft_max = (float*)(shared_raw + cache_len * sizeof(float));

    __shared__ int s_seq_len; 
    if(threadIdx.x == 0){
        s_seq_len = current_seq_len[0];
    }
    __syncthreads();
    int current_seq_len_d = s_seq_len;

    int head_dim = 64;
    int q_index = blockIdx.x * 64 + blockIdx.y * 64 * 14;
    int head_group = 1;
    if (blockIdx.x < 7)
    {
        head_group = 0;
    }

    
    

   
    // int j = 0;
    float sum = 0.0;
    float softmax_sum = 0.0;
    float max_data = -FLT_MAX;
    for(int pos = threadIdx.x; pos < current_seq_len_d; pos += blockDim.x){
        sum = 0.0f;
        int k_index = pos * 64 + head_group * 64 * cache_len + blockIdx.y * 64 * 2 * cache_len;
        for(int i = 0; i < head_dim; i++){
            sum += (float)q[q_index + i] * (float)k_cache[k_index + i];
        }
        // sum = sum * rsqrtf(head_dim);
        sum = sum * 0.125f;
        max_data = fmaxf(max_data, sum);
        qk_data[pos] = sum;
        // softmax_sum += sum;
        // j++;
    }

    __syncthreads();
    for(int i = 16; i > 0; i=i>>1){
       max_data = fmaxf(__shfl_down_sync(0xffffffff, max_data, i, 32),max_data);
    }
    if(threadIdx.x % 32 == 0){
        soft_max[threadIdx.x / 32] = max_data;
    }
    __syncthreads();
    if(threadIdx.x < 16){
        max_data = soft_max[threadIdx.x];
    }
    for(int i = 8; i > 0; i=i>>1){
       max_data = fmaxf(__shfl_down_sync(0xffffffff, max_data, i, 32),max_data);
    }
    if(threadIdx.x == 0){
        soft_max[0] = max_data;
    }
    __syncthreads();
    max_data = soft_max[0];
    



    for(int pos = threadIdx.x; pos < current_seq_len_d; pos += blockDim.x){
        softmax_sum += expf(qk_data[pos] - max_data); 
    }
    for(int i = 16; i > 0; i=i>>1){
       softmax_sum += __shfl_down_sync(0xffffffff, softmax_sum, i, 32);
    }
    if(threadIdx.x % 32 == 0){
        soft_max[threadIdx.x / 32] = softmax_sum;
    }
    __syncthreads();
    if(threadIdx.x < 16){
        softmax_sum = soft_max[threadIdx.x];
    }
    for(int i = 8; i > 0; i=i>>1){
       softmax_sum += __shfl_down_sync(0xffffffff, softmax_sum, i, 32);
    }
    if(threadIdx.x == 0){
        soft_max[0] = softmax_sum;
    }
    __syncthreads();
    softmax_sum = soft_max[0];


    
    // for(int i = 16; i > 0; i=i>>1){
    //    softmax_sum += __shfl_down_sync(0xffff, softmax_sum, i, 32);
    // }

    // if(threadIdx.x % 32 == 0){
    //     soft_max[threadIdx.x/32] = softmax_sum;
    // }

    // __syncthreads();

    // if(threadIdx.x / 32 == 0 && threadIdx.x < 16){
    //     softmax_sum = soft_max[threadIdx.x];
    // }
    
    // for(int i = 8; i > 0; i=i>>1){
    //    softmax_sum += __shfl_down_sync(0xff00, softmax_sum, i, 16);
    // }
    
    // if(threadIdx.x / 32 == 0 && threadIdx.x == 0){
    //     soft_max[0] = softmax_sum;
    // }

    // __syncthreads();
    sum = 0.0;

    int v_index = threadIdx.x + head_group * 64 * cache_len + blockIdx.y * 64 * 2 * cache_len ;
    if(threadIdx.x < head_dim){
        // softmax_sum = soft_max[0];
        for(int i = 0; i < current_seq_len_d; i++){
            sum += expf(qk_data[i] - max_data) * (float)v_cache[v_index + i * head_dim];
        }
        sum = sum / softmax_sum;
    }
    
    int out_idx = threadIdx.x + blockIdx.x * 1 * 64 + blockIdx.y * 64 * 14;
    if(threadIdx.x < 64){
        out_attention[out_idx] = (scalar_t)sum;
    }
    
    
}








torch::Tensor my_decode_function(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor q,
    // int  current_seq_len
    torch::Tensor current_seq_len
){
    // auto output = torch::empty_like(q);
    int batch_size = q.size(0);
    int head_num = q.size(1);
    int cache_len = k_cache.size(-2);
    int head_dim = k_cache.size(-1);
    // int repeat_times = ceil(cache_len/512);

    dim3 grid(head_num, batch_size);
    dim3 block(512, 1); // 512 
    int shared_size = (cache_len + 16) * sizeof(float);

    auto output = torch::empty_like(q);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "my_attention_kernel",([&]{
        my_decode_attention<<<grid, block, shared_size>>>(
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            current_seq_len.data_ptr<int>(),
            cache_len
        );
    }));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &my_decode_function, "Decode Attention (CUDA)");
}