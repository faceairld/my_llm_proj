#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cfloat>


template <typename scalar_t>
__global__ void my_decode_attention(
    const scalar_t *k_cache,            //batch_size, 2, seq_len(x), head_dim
    const scalar_t *v_cache,            //batch_size, 2, seq_len(x), head_dim
    const scalar_t *q,                  //batch_size, 14, seq_len(1), head_dim
    // scalar_t *out_attention,
    const int *current_seq_len,         //当前的seq_len到多长
    int cache_len,                       //cache_len总长
    float *att_group_data,
    float *att_group_sum_max
    // int repeat_times
){
    // extern __shared__ scalar_t qk_data[];
    // extern __shared__ float soft_max[16];
    // extern __shared__ char shared_raw[];
    // float *qk_data = (float*)shared_raw;
    // float *soft_max = (float*)(shared_raw + (cache_len / 4.0f) * sizeof(float));

    __shared__ scalar_t k_smem[64][64 + 2];
    __shared__ scalar_t v_smem[64][64 + 2];
    // __shared__ scalar_t k_smem[64][64];
    // __shared__ scalar_t v_smem[64][64];
    __shared__ float mim_data[2];

    __shared__ int s_seq_len; 
    if(threadIdx.x == 0){
        s_seq_len = current_seq_len[0];
    }
    __syncthreads();
    int current_seq_len_d = s_seq_len;

    int head_dim = 64;
    int q_index = blockIdx.y * 64 + blockIdx.z * 64 * 14;
    int head_group = 1;
    if (blockIdx.y < 7)
    {
        head_group = 0;
    }

    
    

   
    // int j = 0;
    float sum = 0.0;
    float softmax_sum = 0.0;
    float max_data = -FLT_MAX;
    // for(int pos = (threadIdx.x + blockIdx.x * blockDim.x); pos < current_seq_len_d; pos += gridDim.x * blockDim.x){
    sum = 0.0f;
    for(int i = 0; i < 64; i++){
        if(blockIdx.x * 64 + i < current_seq_len_d){
            int k_index = (blockIdx.x * 64 + i) * 64 + head_group * 64 * cache_len + blockIdx.z * 64 * 2 * cache_len;
            k_smem[i][threadIdx.x] = k_cache[k_index + threadIdx.x];
        }
        else{
            k_smem[i][threadIdx.x] = static_cast<scalar_t>(0.0f); 
        }
         
        if(blockIdx.x * 64 + threadIdx.x < current_seq_len_d){
            int v_index = (blockIdx.x * 64 + threadIdx.x) * 64 + head_group * 64 * cache_len + blockIdx.z * 64 * 2 * cache_len;
            v_smem[i][threadIdx.x] = v_cache[v_index + i];
        }
        else{
            v_smem[i][threadIdx.x] = static_cast<scalar_t>(0.0f); 
        }
    }
    __syncthreads();

    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if(pos < current_seq_len_d){
        for(int i = 0; i < head_dim; i++){
            sum += static_cast<float>(q[q_index + i]) * static_cast<float>(k_smem[threadIdx.x][i]);
        }
        sum = sum * 0.125f;
        max_data = sum;
            
    }
        
        // qk_data[threadIdx.x] = sum;
        // softmax_sum += sum;
        // j++;
    
    //计算最大值max

    for(int i = 16; i > 0; i=i>>1){
       max_data = fmaxf(__shfl_down_sync(0xffffffff, max_data, i, 32),max_data);
    }
    if(threadIdx.x % 32 == 0){
        mim_data[threadIdx.x / 32] = max_data;
    }
    __syncthreads();
    if(threadIdx.x < 2){
        max_data = mim_data[threadIdx.x];
    }
    for(int i = 1; i > 0; i=i>>1){
       max_data = fmaxf(__shfl_down_sync(0xffffffff, max_data, i, 32),max_data);
    }
    if(threadIdx.x == 0){
        mim_data[0] = max_data;
        //写入max
        att_group_sum_max[blockIdx.z * 14 * 2 * 8 + blockIdx.y * 2 * 8 + 1 * gridDim.x + blockIdx.x] = max_data;
    }
    __syncthreads();
    max_data = mim_data[0];

    // 计算qk结果和v的乘积，更新sum为exp
    if(pos < current_seq_len_d){
        softmax_sum = expf(sum - max_data);
    }else
        softmax_sum = 0.0f;

    sum = softmax_sum;
    float attention;
    for(int i = 0; i < 64; i++){
        float temp = softmax_sum * static_cast<float>(v_smem[i][threadIdx.x]);
        for(int j = 16; j > 0; j >>= 1){
            temp += __shfl_down_sync(0xffff'ffff, temp, j, 32);
        }
        if(threadIdx.x % 32 == 0){
            mim_data[threadIdx.x / 32] = temp;
        }
        __syncthreads();

        if(threadIdx.x < 2){
            temp = mim_data[threadIdx.x];
        }
        temp += __shfl_down_sync(0xffff'ffff, temp, 1, 32);
        if(threadIdx.x == 0)
            mim_data[0] = temp;
        __syncthreads();

        if(threadIdx.x == i)
            attention = mim_data[0];
        
    }

    //写入qk和v的乘积
    att_group_data[blockIdx.z * 14 * 8 * 64 + blockIdx.y * 8 * 64 + blockIdx.x * 64 + threadIdx.x] = attention;


    

    for(int i = 16; i > 0; i=i>>1){
       sum += __shfl_down_sync(0xffffffff, sum, i, 32);
    }
    if(threadIdx.x % 32 == 0){
        mim_data[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if(threadIdx.x < 2){
        sum = mim_data[threadIdx.x];
    }
    for(int i = 1; i > 0; i=i>>1){
       sum += __shfl_down_sync(0xffffffff, sum, i, 32);
    }
    if(threadIdx.x == 0){
       att_group_sum_max[blockIdx.z * 14 * 2 * 8 + blockIdx.y * 2 * 8 + 0 * gridDim.x + blockIdx.x] = sum;
    }

    // int out_idx = blockIdx.y * 64 + blockIdx.z * 64 * 14;
    // if(threadIdx.x == 0){
    //     mim_data[0] = sum;
    // }
    // __syncthreads();
    // softmax_sum = soft_max[0];


    
    

    // __syncthreads();
    // sum = 0.0;

    // int v_index = threadIdx.x + head_group * 64 * cache_len + blockIdx.y * 64 * 2 * cache_len ;
    // if(threadIdx.x < head_dim){
    //     // softmax_sum = soft_max[0];
    //     for(int i = 0; i < current_seq_len_d; i++){
    //         sum += expf(qk_data[i] - max_data) * (float)v_cache[v_index + i * head_dim];
    //     }
    //     sum = sum / softmax_sum;
    // }
    
    // int out_idx = threadIdx.x + blockIdx.x * 1 * 64 + blockIdx.y * 64 * 14;
    // if(threadIdx.x < 64){
    //     out_attention[out_idx] = (scalar_t)sum;
    // }
    
    
}

template <typename scalar_t>
__global__ void atten_updata(
    float *att_group_data,
    float *att_group_sum_max,
    scalar_t *out_attention
){
    float attention;
    float sum;
    float max;
    float real_max;
    
    attention = 
        att_group_data[blockIdx.z * 14 * 8 * 64 + blockIdx.y * 8 * 64 + threadIdx.x * 64 + blockIdx.x];
    sum = att_group_sum_max[blockIdx.z * 14 * 2 * 8 + blockIdx.y * 2 * 8 + 0 * 8 + threadIdx.x];
    max = att_group_sum_max[blockIdx.z * 14 * 2 * 8 + blockIdx.y * 2 * 8 + 1 * 8 + threadIdx.x];
    real_max = max;

    for(int i = 4; i > 0; i >>=1){
        real_max = fmaxf(__shfl_down_sync(0xffff'ffff, real_max, i, 8), real_max);
    }

    real_max = __shfl_sync(0xffff'ffff, real_max, 0, 8);
    
    sum = sum * expf(max - real_max);
    attention = attention * expf(max - real_max);

    for(int i = 4; i > 0; i >>=1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 8);
        attention += __shfl_down_sync(0xffff'ffff, attention, i, 8);
    }
    
    if(threadIdx.x == 0)
        out_attention[blockIdx.z * 14 * 64 + blockIdx.y * 64 + blockIdx.x] = attention / sum;
        
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

    dim3 grid(8, head_num, batch_size);
    dim3 block(64, 1); // 512 

    dim3 grid2(64, head_num, batch_size);
    dim3 block2(8, 1);

    auto att_group_data = torch::zeros({batch_size, head_num, 8, 64}, q.options().dtype(torch::kFloat32));
    auto att_group_sum_max = torch::zeros({batch_size, head_num, 2, 8}, q.options().dtype(torch::kFloat32));
    auto output = torch::empty_like(q);
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "my_attention_kernel",([&]{
    //     my_decode_attention<<<grid, block>>>(
    //         k_cache.data_ptr<scalar_t>(),
    //         v_cache.data_ptr<scalar_t>(),
    //         q.data_ptr<scalar_t>(),
    //         // output.data_ptr<scalar_t>(),
    //         current_seq_len.data_ptr<int>(),
    //         cache_len,
    //         att_group_data.data_ptr<float>(),
    //         att_group_sum_max.data_ptr<float>()
    //     );
    //     atten_updata<<<grid2, block2>>>(
    //         att_group_data.data_ptr<float>(),
    //         att_group_sum_max.data_ptr<float>(),
    //         output.data_ptr<scalar_t>()
    //     );
    // }));
    if(q.scalar_type() == torch::kFloat16){
    my_decode_attention<<<grid, block>>>(
        k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(),
        q.data_ptr<at::Half>(),
        current_seq_len.data_ptr<int>(),
        cache_len,
        att_group_data.data_ptr<float>(),
        att_group_sum_max.data_ptr<float>()
    );
    atten_updata<<<grid2, block2>>>(
        att_group_data.data_ptr<float>(),
        att_group_sum_max.data_ptr<float>(),
        output.data_ptr<at::Half>()
    );
    }else if(q.scalar_type() == torch::kFloat32){
        // float 版本同理
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &my_decode_function, "Decode Attention (CUDA)");
}