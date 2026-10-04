#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cfloat>

using namespace std;

#define block_size 256
#define warp_size 32

__constant__ float cons_mem[2];

#define CheckcudaError(val) check(val, #val, __FILE__, __LINE__)
void check(cudaError_t err, const char * const fun, const char * const file, const int line){
    if(err != cudaSuccess){
        cerr << "cuda Error in :" << file << ":" << line << ":" << fun << endl;
        cerr << "Error :" << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

#define read_file(path) (path.ends_with(".txt") ? read_file_2(path) : read_file_1(path))
vector<float> read_file_1(const string& path){
    ifstream file(path, ios::binary | ios::ate);
    if(!file){cerr << "read file failed : " << "file open error" << ".bin" << endl; exit(1);}
    vector<float> res;
    streamsize data_size = file.tellg();
    res.resize(data_size / sizeof(float));
    file.seekg(0, ios::beg);
    if(data_size % sizeof(float) != 0){
        cerr << "read file failed : " << "size miss match" << ".bin" << endl;
        exit(1); 
    }
    if(!file.read(reinterpret_cast<char *> (res.data()), data_size)){
        cerr << "read file failed : " << "read data in error" << ".bin" << endl;
    };

    return res;
}

vector<float> read_file_2(const string& path){
    ifstream file(path);
    if(!file){cerr << "read file failed : " << "file open error" << ".txt" << endl; exit(1);}
    vector<float> res;
    float x;
    while(file >> x) res.push_back(x);
    return res;
}


#define write_file(path, data) write_file_1(path, data)
bool write_file_1(const string& path, const vector<float> &data_in){
    if(path.ends_with(".txt")){
        ofstream file(path);
        if(!file){cerr << "write file failed :" << "file open failed" << ".txt" << endl; exit(1);}
        file << setprecision(9);
        int count1 = 0;
        for(auto x : data_in){
            file << x << " ";
            count1 = (count1 + 1) % 16;
            if(count1 == 15) file << "\n";
        }
        return static_cast<bool> (file);
    }else{
        ofstream file(path, ios::binary);
        if(!file){cerr << "write file failed :" << "file open failed" << ".bin" << endl; exit(1);}
        streamsize data_size = data_in.size() * sizeof(float);
        if(!file.write(reinterpret_cast<const char *>(data_in.data()), data_size)){
            cerr << "write file failed :" << "file write failed" << ".bin" << endl;
            exit(1);
        };
        return static_cast<bool> (file);
    }
}


__global__ void softmax1(const float * __restrict__ input, float * __restrict__ output, int input_size){
    int base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;

    __shared__ float max_mem[8];
    __shared__ float sum_mem[8];
    const float4 * in4 = reinterpret_cast<const float4*>(input);
    float max = -FLT_MAX;
    float sum = 0;
    for(int i = offset + base_addr * block_size; i < input_size / 4; i += block_size * grid_size){
        float4 data = in4[i];
        float max_temp = fmaxf(fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)), max);
        sum = __expf(data.x - max_temp) + __expf(data.y - max_temp) + __expf(data.z - max_temp) + __expf(data.w - max_temp) + sum * __expf(max - max_temp);
        max = max_temp;
    }

    for(int i = 16; i > 0; i >>= 1){
        float max_temp = __shfl_down_sync(0xffff'ffff, max, i, 32);
        sum = __shfl_down_sync(0xffff'ffff, sum, i, 32) * __expf(max_temp - fmaxf(max_temp, max)) + sum * __expf(max - fmaxf(max_temp, max));
        max = fmaxf(max_temp, max);
    }
    if(offset % 32 == 0){
        max_mem[offset / 32] = max;
        sum_mem[offset / 32] = sum;
    }
    __syncthreads();

    if(offset < 32){
        max = offset < 8 ? max_mem[offset] : -FLT_MAX;
        sum = offset < 8 ? sum_mem[offset] : 0.0f;
    }

    for(int i = 4; i > 0; i >>= 1){
        float max_temp = __shfl_down_sync(0xffff'ffff, max, i, 32);
        sum = __shfl_down_sync(0xffff'ffff, sum, i, 32) * __expf(max_temp - fmaxf(max_temp, max)) + sum * __expf(max - fmaxf(max_temp, max));
        max = fmaxf(max_temp, max);
    }

    if(offset == 0){
        output[base_addr] = max;
        output[base_addr + grid_size] = sum;
    }

}

__global__ void softmax2(const float * __restrict__ input, float * __restrict__ output, const int grid1_size){
    int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    float max = -FLT_MAX;
    float sum = 0;

    extern __shared__ float mem[]; // 2 * (grid1_size + warp_size - 1) / warp_size
    
    if(offset < (grid1_size + warp_size - 1) / warp_size * warp_size){
        max = offset < grid1_size ? input[offset] : -FLT_MAX;
        sum = offset < grid1_size ? input[offset + grid1_size] : 0.0f;
    }


    for(int i = 16; i > 0; i >>= 1){
        float max_temp = __shfl_down_sync(0xffff'ffff, max, i, 32);
        sum = sum * __expf(max - fmaxf(max_temp, max)) + __shfl_down_sync(0xffff'ffff, sum, i, 32) * __expf(max_temp - fmaxf(max_temp, max));
        max = fmaxf(max, max_temp);
    }

    if(offset < (grid1_size + warp_size - 1) / warp_size * warp_size && offset % 32 == 0){
        mem[offset / 32] = max;
        mem[offset / 32 + (grid1_size + warp_size - 1) / warp_size] = sum;
    }
    __syncthreads();

    // (180 + 31) / 32 =  6 < 32
    if(offset < 32){
        max = offset < (grid1_size + warp_size - 1) / warp_size ? mem[offset] : -FLT_MAX;
        sum = offset < (grid1_size + warp_size - 1) / warp_size ? mem[offset + (grid1_size + warp_size - 1) / warp_size] : -0.0f;
    }

    for(int i = 16; i > 0; i >>= 1){
        float max_temp = __shfl_down_sync(0xffff'ffff, max, i, 32);
        sum = sum * __expf(max - fmaxf(max_temp, max)) + __shfl_down_sync(0xffff'ffff, sum, i, 32) * __expf(max_temp - fmaxf(max_temp, max));
        max = fmaxf(max, max_temp);
    }

    if(offset == 0){
        output[0] = sum;
        output[1] = max;
    }

}


__global__ void softmax3(const float * __restrict__ input, float * __restrict__ output, const int input_size){
    int base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;

    float max = cons_mem[1];
    float sum = cons_mem[0];

    const float4* in4 = reinterpret_cast<const float4*> (input);
    float4* out4 = reinterpret_cast<float4*> (output);

    for(int i = offset + base_addr * block_size; i < input_size / 4; i += grid_size * block_size){
        float4 temp = in4[i];
        temp = make_float4(__expf(temp.x - max) / sum, __expf(temp.y - max) / sum, __expf(temp.z - max) / sum, __expf(temp.w - max) / sum);
        out4[i] = temp;
    }
}



int main(int argc, char * argv[]){

    if(argc < 3){
        cerr << "must put at least 2 param : input file path and output file path"<< endl;
        exit(1);
    }
    string path_in = argv[1];
    string path_out = argv[2];

    int grid_dim1 = argc > 3 ? stoi(argv[3]) : 16; 
    int grid_dim2 = argc > 4 ? stoi(argv[4]) : 1; 
    int grid_dim3 = argc > 5 ? stoi(argv[5]) : 1;
    
    int grid1_size = grid_dim1 * grid_dim2 * grid_dim3;
    if(grid1_size > 180){
        cerr << "grid1 size too large, it should less than 180" << endl;
        exit(1);
    }

    vector<float> h_data_in = read_file(path_in);
    vector<float> h_data_out(h_data_in.size());
    vector<float> h_data_kernel2_min(2, 0.0f);
    h_data_in.resize((h_data_in.size() + 3) / 4 * 4, -FLT_MAX);
    float * d_data;
    float * d_data_2;
    float * d_data_kernel2_out;
    float * d_data_final_out;
    

    //CheckcudaError(cudaMalloc(&d_data, (h_data_in.size() + 3) / 4 * 4 * sizeof(float)));
    CheckcudaError(cudaMalloc(&d_data, h_data_in.size() * sizeof(float)));
    CheckcudaError(cudaMalloc(&d_data_2, 2 * grid1_size *  sizeof(float)));
    CheckcudaError(cudaMalloc(&d_data_kernel2_out, 2 * sizeof(float)));
    CheckcudaError(cudaMalloc(&d_data_final_out, h_data_in.size() * sizeof(float)));

    //CheckcudaError(cudaMemset(d_data, 0, (h_data_in.size() + 3) / 4 * 4 * sizeof(float)));
    //CheckcudaError(cudaMemset(d_data_2, 0, 2 * grid_size *  sizeof(float)));

    CheckcudaError(cudaMemcpy(d_data, h_data_in.data(), h_data_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(grid_dim1, grid_dim2, grid_dim3);
    dim3 block(16, 16, 1);

    // block_size * 4

    dim3 grid2(1, 1, 1);


    softmax1<<<grid, block>>>(d_data, d_data_2, static_cast<int>(h_data_in.size()));
    CheckcudaError(cudaGetLastError());

    softmax2<<<grid2, block, (sizeof(float) * 2 * ((grid1_size + warp_size - 1) / warp_size) )>>>(d_data_2, d_data_kernel2_out, grid1_size);
    CheckcudaError(cudaGetLastError());
    CheckcudaError(cudaFree(d_data_2));

    CheckcudaError(cudaMemcpy(h_data_kernel2_min.data(), d_data_kernel2_out, h_data_kernel2_min.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CheckcudaError(cudaFree(d_data_kernel2_out));
    CheckcudaError(cudaMemcpyToSymbol(cons_mem, h_data_kernel2_min.data(), h_data_kernel2_min.size() * sizeof(float)));
    
    softmax3<<<grid, block>>>(d_data, d_data_final_out, static_cast<int>(h_data_in.size()));
    CheckcudaError(cudaGetLastError());
    CheckcudaError(cudaFree(d_data));

    CheckcudaError(cudaMemcpy(h_data_out.data(), d_data_final_out, h_data_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CheckcudaError(cudaFree(d_data_final_out));
    CheckcudaError(cudaDeviceSynchronize());

    cerr << "write file status :" << boolalpha << write_file(path_out, h_data_out) << endl;

}