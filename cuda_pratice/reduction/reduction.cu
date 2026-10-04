#include<vector>
#include<iostream>
using namespace std;


// #define checkCudaErrors()

// void check(cudaError_t err, const char* fun, const)


__global__ void reduction(float *nums, float *out, int total_nums){
    int thread_x = threadIdx.x + blockDim.x * blockIdx.x;
    int thread_y = threadIdx.y + blockDim.y * blockIdx.y;
    int thread_z = threadIdx.z + blockDim.z * blockIdx.z;

    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    int base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    float data;

    __shared__ float share_data[8];

    int addr = base_addr * blockDim.x * blockDim.y * blockDim.z + idx;

    data = (addr < total_nums) ? nums[addr] : 0.0f;

    for(int off = 16; off > 0; off >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, off, 32);
    }

    if(idx % 32 == 0)
        share_data[idx / 32] = data;
    __syncthreads();

    if(idx / 32 == 0)
        data = idx < 8 ? share_data[idx] : 0.0f;
    for(int off = 4; off > 0; off >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, off, 32);
    }
    if(idx == 0)
        out[base_addr] = data;
}


__global__ void reduction2(float *nums){
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    float data;
    __shared__ float temp_data[2];
    data = nums[idx];
    for(int i = 16; i > 0; i >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, i, 32);
    }

    if(idx % 32 == 0)
        temp_data[idx / 32] = data;
    __syncthreads();
    if(idx / 32 == 0){
        data = threadIdx.x < 2 ? temp_data[threadIdx.x] : 0.0f;
        data += __shfl_down_sync(0xffff'ffff, data, 1, 32);
    }
    if(idx == 0)
        nums[0] = data;
}



int main(int argc, char* argv[]){
    vector<float> num1(16384, 1.0f);
    vector<float> num2(1, 0.0f);
    float *d_num1;
    float *d_temp;
    cudaMalloc(&d_num1, 16384 * sizeof(float));
    cudaMalloc(&d_temp, 64 * sizeof(float));
    cudaMemcpy(d_num1, num1.data(), 16384 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(64, 1, 1);
    dim3 block(16, 16 ,1);

    dim3 grid2(1, 1, 1);
    dim3 block2(32, 2 ,1);

    reduction<<<grid, block>>>(d_num1, d_temp, num1.size());
    reduction2<<<grid2, block2>>>(d_temp);
    
    cudaMemcpy(num2.data(), d_temp, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "sum :" << num2[0] << endl;
}