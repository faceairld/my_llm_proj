#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <vector>
__global__ void max1(const float* input, float* output, int N) {
     float max = -FLT_MAX;
    __shared__ float s_mem[8];
    for(int i = 0; i < N; i += gridDim.x * blockDim.x){
        int idx = i + blockIdx.x * 256 + threadIdx.x;
        if(idx < N)
            max = fmaxf(max, input[idx]);
    }

    for(int i = 16; i > 0; i >>= 1){
        max = fmaxf(max,__shfl_down_sync(0xffff'ffff, max, i, 32));
    }
    if(threadIdx.x%32 == 0){
        s_mem[threadIdx.x /32] = max;
    }
    __syncthreads();
    
    if(threadIdx.x < 8){
        max = s_mem[threadIdx.x];
    }
    for(int i = 4; i > 0; i >>= 1){
        max = fmaxf(max,__shfl_down_sync(0xffff'ffff, max, i, 32));
    }
    if(threadIdx.x == 0)
        output[blockIdx.x] = max;
}

__global__ void max2(float *indata, int blocksPerGrid){
    float max = -FLT_MAX;
    __shared__ float s_mem[8];
   
    if(threadIdx.x < blocksPerGrid)
        max = indata[threadIdx.x];
    
    for(int i = 16; i > 0; i >>= 1){
        max = fmaxf(max, __shfl_down_sync(0xffff'ffff, max, i, 32));
    }

    if(threadIdx.x % 32 == 0){
        s_mem[threadIdx.x /32] = max;
    }
    __syncthreads();
    
    if(threadIdx.x < 8){
        max = s_mem[threadIdx.x];
    }

    for(int i = 4; i > 0; i >>= 1){
        max = fmaxf(max, __shfl_down_sync(0xffff'ffff, max, i, 32));
    }

    if(threadIdx.x == 0){
        indata[0] = max;
        // printf("max:%f\n",max);
    }
        

}

__global__ void sum1(const float* input, float* output, int N) {

    float sum = 0.0;
    float max = output[0];
    __shared__ float s_mem[8];
    for(int i = 0; i < N; i += gridDim.x * blockDim.x){
        int idx = i + blockIdx.x * 256 + threadIdx.x;
        if(idx < N){
            float indata = input[idx];
            sum += expf(indata - max);
            // printf("indata: %f, idx: %d, sum: %f\n",indata, idx, sum);
        }
            
    }

    for(int i = 16; i > 0; i >>= 1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }
    if(threadIdx.x%32 == 0){
        s_mem[threadIdx.x /32] = sum;
    }
    __syncthreads();
    
    if(threadIdx.x < 8){
        sum = s_mem[threadIdx.x];
    }
    for(int i = 4; i > 0; i >>= 1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }
    if(threadIdx.x == 0)
        output[blockIdx.x + 1] = sum;
}

__global__ void sum2(float *indata, int blocksPerGrid){
    float sum = 0.0f;
    __shared__ float s_mem[8];
    if(threadIdx.x < blocksPerGrid)
        sum = indata[threadIdx.x + 1];
    
    for(int i = 16; i > 0; i >>= 1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }

    if(threadIdx.x % 32 == 0){
        s_mem[threadIdx.x /32] = sum;
    }
    __syncthreads();
    
    if(threadIdx.x < 8){
        sum = s_mem[threadIdx.x];
    }

    for(int i = 4; i > 0; i >>= 1){
        sum += __shfl_down_sync(0xffff'ffff, sum, i, 32);
    }

    if(threadIdx.x == 0)
        indata[1] = sum;

}

__global__ void softmax(const float* input, float *mim_data, float* output, int N){
    float max = mim_data[0];
    float sum = mim_data[1];
    float data;

    for(int i = 0; i < N; i += gridDim.x * blockDim.x){
        int idx = i + blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N){
            data = expf(input[idx]- max)/sum;
            output[idx] = data; 
        }
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = min((N + threadsPerBlock - 1) / threadsPerBlock, 256);
    // int blocksPerGrid = 256;


    dim3 grid2(1,1,1);
    dim3 block2(256,1,1);

    float *mim_data;
    cudaMalloc(&mim_data, (blocksPerGrid + 1) * sizeof(float));


    max1<<<blocksPerGrid,threadsPerBlock>>>(input, mim_data, N);
    max2<<<grid2, block2>>>(mim_data, blocksPerGrid);
    sum1<<<blocksPerGrid,threadsPerBlock>>>(input, mim_data, N);
    sum2<<<grid2, block2>>>(mim_data, blocksPerGrid);
    softmax<<<blocksPerGrid,threadsPerBlock>>>(input, mim_data, output, N);

    cudaDeviceSynchronize();

    cudaFree(mim_data);
}

