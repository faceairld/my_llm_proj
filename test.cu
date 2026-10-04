

#define block_szie 256







__global__ void add(float * input, float * input2, float * output, int data_size){
    int idx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;

    float data1;
    for(int i = idx + base_addr * block_szie; i < data_size; i += grid_size * block_szie){
        if(i < data_size){
            data1 = input[i] + input2[i];
            output[i] = data1;
        }
    }

}














