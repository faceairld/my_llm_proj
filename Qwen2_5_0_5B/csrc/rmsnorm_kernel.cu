#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_kernel(
    const scalar_t* __restrict__ input_data,
    const scalar_t* __restrict__ weight,
    float eps,
    scalar_t *out_data

){
    __shared__ float sum[28];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = threadIdx.x / 32;
    float data_in = static_cast<float>(input_data[idx]);
    float original_data = data_in;
    data_in = data_in * data_in / 896;

    for(int i = 16; i > 0 ; i = i>>1){
        data_in += __shfl_down_sync(0xffff'ffff, data_in, i, 32);
    }
    
    if(threadIdx.x % 32 == 0)
        sum[warp_idx] = data_in;
    __syncthreads();

    if(warp_idx == 0 && threadIdx.x < 28)
        data_in = sum[threadIdx.x];
    else
        data_in = 0.0;
    
    for(int i = 16; i > 0 ; i = i>>1){
        data_in += __shfl_down_sync(0xffff'ffff, data_in, i, 32);
    }
    // data_in = __shfl_sync(0xffff'ffff, data_in, 0, 32);

    // if(threadIdx.x % 32 == 0)
    //     sum[warp_idx] = data_in;
    // __syncthreads();
    if(threadIdx.x == 0)
        sum[0] = data_in;
    __syncthreads();
    // if(threadIdx.x % 32 == 0)
    //     data_in = sum[warp_idx];
    // __syncthreads();
    // data_in = __shfl_sync(0xffff'ffff, data_in, 0, 32);
    data_in = sum[0];

    out_data[idx] = static_cast<scalar_t>((original_data * rsqrtf(data_in + eps)) * static_cast<float>(weight[threadIdx.x]));

}


torch::Tensor my_rmsnorm_function (
        torch::Tensor input_data,
        torch::Tensor weight,
        float eps
){
    const int batch_size = input_data.size(0);
    const int seq_len = input_data.size(1);
    const int hidden_dim = 896;

    //grid : batch_size * seq_len, 1, 1
    //block :  hidden_dim, 1, 1
    dim3 grid(batch_size * seq_len, 1, 1);
    dim3 block(hidden_dim, 1, 1);

    auto output = torch::empty_like(input_data);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_data.scalar_type(),"my_rms_kernel", ([&]{rms_kernel<<<grid, block>>>(
        input_data.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        eps,
        output.data_ptr<scalar_t>()
        );
    }) );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &my_rmsnorm_function, "RMSNorm forward (CUDA)");
}