// #include <cuda_runtime.h>
// #include <iostream>

//  __global__  void function1(
//     float *input,
//     float *reduce_sum,
//     int n
// ){

    
//     float sum = 0.0;
//     for(int pos = 0; pos < n; pos = pos + 32 * 16){
//     int idx = threadIdx.x + blockIdx.x * 32 + pos;
//     if(idx < n){
//         sum += input[idx];
//     }
        
//     }


//     for(int i = 16; i > 0; i = i >> 1){
//        sum +=  __shfl_down_sync(0xffff'ffff, sum, i, 32);
//     }

//     if(threadIdx.x == 0)
//         input[blockIdx.x] = sum;

//     if(blockIdx.x == 0 && threadIdx.x < 16){
//         sum = input[blockIdx.x];
//     }

//     for(int i = 8; i > 0; i = i >> 1){
//        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
//     }
//     if(blockIdx.x == 0 && threadIdx.x == 0)
//         reduce_sum[0] = sum;

    
// }


// int main(){
//     float *data;
//     float *data_out;
//     int n = 8192;
//     data = (float *)malloc(n * sizeof(float));
//     data_out = (float *)malloc(sizeof(float));
//     memset(data, 0, n * sizeof(float));
//     for(int i = 0; i < 8192; i++){
//         data[i] = 1.0f;
//     }


//     float *data_d;
//     float *data_out_d;
//     cudaMalloc(&data_d, n * sizeof(float));
//     cudaMalloc(&data_out_d, 1 * sizeof(float));
//     cudaMemcpy(data_d, data, n * sizeof(float), cudaMemcpyHostToDevice);

//     dim3 grid(16,1,1);
//     dim3 block(32,1,1);

//     function1<<<grid, block>>>(data_d, data_out_d, n);
//     cudaMemcpy(data_out, data_out_d, sizeof(float), cudaMemcpyDeviceToHost);

//     std::cout << "data_out:" << *data_out << std::endl;
// }



#include <cuda_runtime.h>
#include <iostream>
#include <vector>




__global__ void function1(
    float *data_in,
    float *data_out,
    int n
){
    __shared__ float s_mem[3];
    
    float sum = 0.0f;
    for(int i = 0; i < n; i += gridDim.x * blockDim.x){
        int idx = i  + blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n)
            sum += data_in[idx];
    }
    
    for(int i = 16; i > 0; i>>=1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }
    
    int warp_idx = (threadIdx.x / 32)%3;

    if (threadIdx.x % 32 == 0)
        s_mem[warp_idx] = sum;

    __syncthreads();

    if (threadIdx.x < 3)
        sum = s_mem[threadIdx.x];
    else
        sum = 0.0f;

    for(int i = 2; i > 0; i >>=1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }
    if(threadIdx.x == 0){
        data_in[blockIdx.x] = sum;
    }

    if(threadIdx.x / 32 == 0 && threadIdx.x < 16)
        sum = data_in[threadIdx.x];

    for(int i = 8; i > 0; i>>=1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }

    if(threadIdx.x / 32 == 0 && threadIdx.x == 0)
        data_out[0] = sum;

    }

int main(int agrc, char* argv[]){

    int n = 8192;
    std::vector<float> data(n, 0.0f);
    float data_out;
    for(int i = 0; i < n; i++){
        data[i] = 1.0f;
    }

    for(int j = 0; j < 16 ; j++){
        std::cout << data.at(j) <<std::endl;
    }
        
    
    float *data_d, *data_out_d;
    cudaMalloc(&data_d, n * sizeof(float));
    cudaMalloc(&data_out_d, 1 * sizeof(float));

    cudaMemcpy(data_d, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(16, 1);
    dim3 block(96, 1);


    function1<<<grid, block>>>(
        data_d,
        data_out_d,
        n
    );
    cudaMemcpy(&data_out, data_out_d, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "data_out:"<< data_out << std::endl;
    

}







