#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <cfloat>


template <typename scalar_t> 
__global__ void my_flash_attention(
    scalar_t *q,
    scalar_t *k,
    scalar_t *v,
    scalar_t *atten_out,
    int data_len,
    int kv_group_num
){

    // int q_group_num = blockIdx.x % 4;
    int q_group_idx = blockIdx.z * gridDim.y * gridDim.x * blockDim.y * blockDim.x 
                        + blockIdx.y * gridDim.x * blockDim.y * blockDim.x 
                            + blockIdx.x * blockDim.y * blockDim.x;
    int kv_head_select = blockIdx.y > 7;
    int kv_group_idx = blockIdx.z * 2 * gridDim.x * blockDim.y * 4 * blockDim.x 
                         + kv_head_select * gridDim.x * blockDim.y * 4 * blockDim.x;

    if(q_group_idx + blockIdx.y * blockDim.x > data_len)
        return;
    
    __shared__ scalar_t q_mem[16][64];
    __shared__ scalar_t k_mem[64][64];
    __shared__ scalar_t v_mem[64][64];

    __shared__ float mim_data[16][2];

    //block dim (64, 16)
    q_mem[threadIdx.y][threadIdx.x] = q[q_group_idx + threadIdx.y * blockDim.x + threadIdx.x];
    
    
    float attention_data_c = 0.0f;
    float max_c = 0.0f;
    float sum_c = 0.0f;
    
     
    // for(int q_group = 0; q_group < 4; q_group ++){

    for(int i = 0; i < kv_group_num; i++){
        for(int k = 0; k <4; k++){
            k_mem[threadIdx.y + k * blockDim.y][threadIdx.x] = k[kv_group_idx + i * 4 * blockDim.y * blockDim.x + (threadIdx.y + k * blockDim.y) * blockDim.x + threadIdx.x];
            v_mem[threadIdx.y + k * blockDim.y][threadIdx.x] = v[kv_group_idx + i * 4 * blockDim.y * blockDim.x + (threadIdx.y + k * blockDim.y) * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        float qk_multi_data = 0.0f;
        float attention_data = 0.0f;
        float max = 0.0f;
        float sum = 0.0f;
        
       
        if(threadIdx.y + blockIdx.x * 16 >= i * 64 + threadIdx.x){
            for(int j = 0; j < blockDim.x; j++){
                qk_multi_data += static_cast<float>(q_mem[threadIdx.y][j]) * static_cast<float>(k_mem[threadIdx.x][j]);
            }
        }else{
                qk_multi_data = -FLT_MAX;
        }
        
        //max------------------------------------------------------- 
        max = qk_multi_data;
        for(int j = 16; j > 0; j >>= 1){
            max = fmaxf(__shfl_down_sync(0xffff'ffff, max, j, 32), max);
        }

        if(threadIdx.x % 32 == 0){
            mim_data[threadIdx.y][threadIdx.x / 32] = max;
        }
        __syncthreads();

        if(threadIdx.y == 0 && threadIdx.x < 32){
            max = mim_data[threadIdx.x / 2][threadIdx.x % 2];

            for(int j = 1; j > 0; j >>= 1){
                max = fmaxf(__shfl_down_sync(0xffff'ffff, max, j, 32), max);
            }

         
            if(threadIdx.x % 2 == 0)
                mim_data[threadIdx.x / 2][0] = max;
        }
        __syncthreads();

        max = mim_data[threadIdx.y][0];
        //-------------------------------------------------------
        max = fmaxf(max, max_c);
        qk_multi_data = expf(qk_multi_data - max);

        //sum-------------------------------------------------------
        sum = qk_multi_data;

        for(int j = 16; j > 0; j >>= 1){
            sum += __shfl_down_sync(0xffff'ffff, sum, j, 32);
        }

        if(threadIdx.x % 32 == 0){
            mim_data[threadIdx.y][threadIdx.x / 32] = sum;
        }
        __syncthreads();

        if(threadIdx.y == 0 && threadIdx.x < 32){
            sum = mim_data[threadIdx.x / 2][threadIdx.x % 2];

            for(int j = 1; j > 0; j >>= 1){
                sum += __shfl_down_sync(0xffff'ffff, sum, j, 32);
            }

           
            if(threadIdx.x % 2 == 0)
                mim_data[threadIdx.x / 2][0] = sum;
        }
        __syncthreads();

        sum = mim_data[threadIdx.y][0];
        //-------------------------------------------------------
        
        for(int j = 0; j < blockDim.x; j++){
            float atten_mim_data = 0.0f;
            atten_mim_data = qk_multi_data * v_mem[threadIdx.x][j];
            for(int k = 16; k > 0; k >>= 1){
                atten_mim_data += __shfl_down_sync(0xffff'ffff, atten_mim_data, k, 32);
            }

            if(threadIdx.x % 32 == 0){
                mim_data[threadIdx.y][threadIdx.x / 32] = atten_mim_data;
            }
            
            __syncthreads();

            // if(threadIdx.y < 2){
            //     atten_mim_data = mim_data[(threadIdx.y * 64 + threadIdx.x) / 2][(threadIdx.y * 64 + threadIdx.x) % 2];

            //     for(int k = 1; k > 0; k >>= 1){
            //         atten_mim_data += __shfl_down_sync(0xffff'ffff, atten_mim_data, k, 32);
            //     }

            //     int idx = threadIdx.y * blockDim.x + threadIdx.x;
            //     __syncthreads();
            //     if(threadIdx.x % 2 == 0)
            //         mim_data[idx / 2][0] = atten_mim_data;
            // }

            // __syncthreads();
            if(threadIdx.y == 0 && threadIdx.x < 32){
                atten_mim_data = mim_data[threadIdx.x / 2][threadIdx.x % 2];

                for(int k = 1; k > 0; k >>= 1){
                    atten_mim_data += __shfl_down_sync(0xffff'ffff, atten_mim_data, k, 32);
                }

             
                if(threadIdx.x % 2 == 0)
                    mim_data[threadIdx.x / 2][0] = atten_mim_data;
            }

            __syncthreads();

            if(threadIdx.x == j)
                attention_data = mim_data[threadIdx.y][0];


        }

        attention_data_c = (attention_data_c * exp(max_c - max) + attention_data); 
        sum_c = (sum_c * exp(max_c - max) + sum);
        max_c = max;

    }        
    
    atten_out[q_group_idx + threadIdx.y * blockDim.x + threadIdx.x] = static_cast<scalar_t>(attention_data_c / sum_c);

}

// }













torch:: Tensor my_flash_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor atten_out

){
    // batch_size, head_num, seq_len, head_dim
    int batch_size = q.size(0);
    int head_num = q.size(1);
    int head_dim = q.size(-1);
    int max_data_num = batch_size * head_num * q.size(-2);

    // set q_group_seq
    int q_group_seq = 16;
    int q_group = (q.size(2) + q_group_seq - 1) / q_group_seq;
    
    int kv_group_num = (k.size(2) + 63) / 64;
    if(kv_group_num != (v.size(2) + 63) / 64)
        std::cout << "dim mismatch" << std::endl;

    dim3 grid(q_group, head_num, batch_size);
    dim3 block(head_dim, q_group_seq);

    auto out_data = torch::empty_like(q);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "my_flash_attention",([&]{
        my_flash_attention<<<grid, block>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out_data.data_ptr<scalar_t>(),
            max_data_num,
            kv_group_num
        );
    }));

    return out_data;

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &my_flash_attention, "Flash Attention (CUDA)");
}



























