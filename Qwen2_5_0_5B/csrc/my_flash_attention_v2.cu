#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <cfloat>


template <typename scalar_t> 
__global__ void my_flash_attention(
    scalar_t *q,
    scalar_t *k_matrix,
    scalar_t *v,
    scalar_t *atten_out,
    int q_seq_len,
    int kv_seq_len,
    int kv_group_num
){

    // int q_group_num = blockIdx.x % 4;
    int q_group_idx = blockIdx.z * gridDim.y * q_seq_len * (blockDim.x * 2) 
                        + blockIdx.y * q_seq_len * (blockDim.x * 2)
                            + blockIdx.x * blockDim.y * (blockDim.x * 2);
    int kv_head_select = blockIdx.y / 7;
    int kv_group_idx = blockIdx.z * 2 * kv_seq_len * (blockDim.x * 2)
                         + kv_head_select * kv_seq_len * (blockDim.x * 2);

    bool valid = (blockIdx.x * 32 + threadIdx.y) < q_seq_len;
    
    
    __shared__ scalar_t q_mem[32][64];
    __shared__ scalar_t k_mem[64][64];
    __shared__ scalar_t v_mem[64][64];

    // __shared__ float mim_data[16][2];

    //block dim (64, 16)
    if(valid){
        q_mem[threadIdx.y][threadIdx.x] = q[q_group_idx + threadIdx.y * (blockDim.x * 2) + threadIdx.x];
        q_mem[threadIdx.y][threadIdx.x + blockDim.x] = q[q_group_idx + threadIdx.y * (blockDim.x * 2) + blockDim.x + threadIdx.x];
    }
        
    else{
        q_mem[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0.0f);
        q_mem[threadIdx.y][threadIdx.x + blockDim.x] = static_cast<scalar_t>(0.0f);
    }
        
    
    float attention_data_c = 0.0f;
    float attention_data_c2 = 0.0f;
    float max_c = -FLT_MAX;
    float sum_c = 0.0f;
    
     
    // for(int q_group = 0; q_group < 4; q_group ++){

    int kv_loop_group = min(kv_group_num, ((blockIdx.x * 32 + 32 + 63) / 64));

    for(int i = 0; i < kv_loop_group; i++){
        for(int k = 0; k < 2; k++){
            int idx = i * 64 + k * 32 + threadIdx.y;
            if(idx < kv_seq_len){
                //K shared mem copy
                k_mem[threadIdx.y + k * blockDim.y][threadIdx.x] 
                    = k_matrix[kv_group_idx + i * 2 * blockDim.y * (blockDim.x * 2) + (threadIdx.y + k * blockDim.y) * (blockDim.x * 2) + threadIdx.x];
                k_mem[threadIdx.y + k * blockDim.y][threadIdx.x + blockDim.x] 
                    = k_matrix[kv_group_idx + i * 2 * blockDim.y * (blockDim.x * 2) + (threadIdx.y + k * blockDim.y) * (blockDim.x * 2) + threadIdx.x + blockDim.x];
                
                //V shared mem copy
                v_mem[threadIdx.y + k * blockDim.y][threadIdx.x] 
                    = v[kv_group_idx + i * 2 * blockDim.y * (blockDim.x * 2) + (threadIdx.y + k * blockDim.y) * (blockDim.x * 2) + threadIdx.x];
                v_mem[threadIdx.y + k * blockDim.y][threadIdx.x + blockDim.x] 
                    = v[kv_group_idx + i * 2 * blockDim.y * (blockDim.x * 2) + (threadIdx.y + k * blockDim.y) * (blockDim.x * 2) + threadIdx.x + blockDim.x];
            }else{
                k_mem[threadIdx.y + k * blockDim.y][threadIdx.x] = 0.0f;
                k_mem[threadIdx.y + k * blockDim.y][threadIdx.x + blockDim.x] = 0.0f;

                v_mem[threadIdx.y + k * blockDim.y][threadIdx.x] = 0.0f;
                v_mem[threadIdx.y + k * blockDim.y][threadIdx.x + blockDim.x] = 0.0f;
            }
        }
        __syncthreads();
        float qk_multi_data = 0.0f;
        float qk_multi_data2 = 0.0f;
        float attention_data = 0.0f;
        float attention_data2 = 0.0f;
        float max = -FLT_MAX;
        float sum = 0.0f;
        
       
        if(threadIdx.y + blockIdx.x * 32 >= i * 64 + threadIdx.x){
            for(int j = 0; j < 64; j++){
                qk_multi_data += static_cast<float>(q_mem[threadIdx.y][j]) * static_cast<float>(k_mem[threadIdx.x][j]);           
            }
        }else{
                qk_multi_data = -FLT_MAX;           
        }

        if(threadIdx.y + blockIdx.x * 32 >= i * 64 + threadIdx.x + blockDim.x){
            for(int j = 0; j < 64; j++){
                qk_multi_data2 += static_cast<float>(q_mem[threadIdx.y][j]) * static_cast<float>(k_mem[threadIdx.x + 32][j]);
            }
        }else{
                qk_multi_data2 = -FLT_MAX;
        }

        if(qk_multi_data != -FLT_MAX)
            qk_multi_data *= 0.125;
        
        if(qk_multi_data2 != -FLT_MAX)
            qk_multi_data2 *= 0.125;
        
        //max------------------------------------------------------- 
        max = fmaxf(qk_multi_data, qk_multi_data2);
        for(int j = 16; j > 0; j >>= 1){
            max = fmaxf(__shfl_down_sync(0xffff'ffff, max, j, 32), max);
        }

        max = __shfl_sync(0xffff'ffff, max, 0, 32);
        
        //-------------------------------------------------------
        max = fmaxf(max, max_c);
        qk_multi_data = expf(qk_multi_data - max);
        qk_multi_data2 = expf(qk_multi_data2 - max);

        //sum-------------------------------------------------------
        sum = qk_multi_data + qk_multi_data2;

        for(int j = 16; j > 0; j >>= 1){
            sum += __shfl_down_sync(0xffff'ffff, sum, j, 32);
        }

        sum = __shfl_sync(0xffff'ffff, sum, 0, 32); 
        //-------------------------------------------------------
        
        for(int j = 0; j < 64; j++){
            float atten_mim_data = 0.0f;
            atten_mim_data = qk_multi_data * v_mem[threadIdx.x][j] + qk_multi_data2 * v_mem[threadIdx.x + blockDim.x][j];
            for(int k = 16; k > 0; k >>= 1){
                atten_mim_data += __shfl_down_sync(0xffff'ffff, atten_mim_data, k, 32);
            }

            atten_mim_data = __shfl_sync(0xffffffff, atten_mim_data, 0);

            if(threadIdx.x == j % 32){
                if(j / 32 == 0)
                    attention_data = atten_mim_data;
                else
                    attention_data2 = atten_mim_data;
            }
                
        }

        attention_data_c = (attention_data_c * exp(max_c - max) + attention_data); 
        attention_data_c2 = (attention_data_c2 * exp(max_c - max) + attention_data2); 
        sum_c = (sum_c * exp(max_c - max) + sum);
        max_c = max;

    }        
    
    if(valid){
        atten_out[q_group_idx + threadIdx.y * (blockDim.x * 2) + threadIdx.x] = static_cast<scalar_t>(attention_data_c / (sum_c + 1e-6f));
        atten_out[q_group_idx + threadIdx.y * (blockDim.x * 2) + threadIdx.x + blockDim.x] = static_cast<scalar_t>(attention_data_c2 / (sum_c + 1e-6f));

    }
        
}

// }













torch:: Tensor my_flash_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
    // torch::Tensor atten_out

){
    // batch_size, head_num, seq_len, head_dim
    int batch_size = q.size(0);
    int head_num = q.size(1);
    int head_dim = q.size(-1);
    int max_data_num = batch_size * head_num * q.size(-2) * head_dim;

    int q_seq_len = q.size(-2);
    int k_seq_len = k.size(-2);
    int v_seq_len = v.size(-2);
    if(k_seq_len != v_seq_len){
        std::cout<<"dim mismatch"<<std::endl;
    }
    int kv_seq_len = k_seq_len; 
    // set q_group_seq
    int q_group_seq = 32;
    int q_group_dim = 32;
    int q_group = (q.size(2) + q_group_seq - 1) / q_group_seq;
    
    int kv_group_num = (k.size(2) + 63) / 64;


    dim3 grid(q_group, head_num, batch_size);
    dim3 block(q_group_dim, q_group_seq);

    auto out_data = torch::zeros_like(q);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "my_flash_attention",([&]{
    //     my_flash_attention<<<grid, block>>>(
    //         q.data_ptr<scalar_t>(),
    //         k.data_ptr<scalar_t>(),
    //         v.data_ptr<scalar_t>(),
    //         out_data.data_ptr<scalar_t>(),
    //         max_data_num,
    //         kv_group_num
    //     );
    // }));

    if(q.scalar_type() == torch::kFloat32){
            my_flash_attention<<<grid, block>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            out_data.data_ptr<float>(),
            q_seq_len,
            kv_seq_len,
            kv_group_num
        );
    }else if(q.scalar_type() == torch::kFloat16){
            my_flash_attention<<<grid, block>>>(
            q.data_ptr<at::Half>(),
            k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(),
            out_data.data_ptr<at::Half>(),
            q_seq_len,
            kv_seq_len,
            kv_group_num
        );
    }else{
        TORCH_CHECK(false, "Unsupported tensor dtype: only float32 and float16 are supported for my_flash_attention");
    }

    return out_data;

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &my_flash_attention, "Flash Attention (CUDA)");
}



























