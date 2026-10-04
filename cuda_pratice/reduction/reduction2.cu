#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
 void check(cudaError_t error, const char* const fun, const char * const file , const int line){
    if(error != cudaSuccess){
        cerr << "CUDA error at" << file << ":" << line << endl;
        cerr << cudaGetErrorString(error) << " " << fun << endl;
        exit(1);
    }
 }



vector<float> read_param_in(const string& path){
    ifstream file(path, ios::binary | ios::ate);
    if(!file) {cerr << "file open error" << path << endl; return {};}
    streamsize byte_nums = file.tellg();
    if(byte_nums < 0 || byte_nums % sizeof(float) != 0){
        cerr << "bad file" << path << endl; return {};
    }
    file.seekg(0, ios::beg);
    vector<float> data(byte_nums / sizeof(float));
    if(!file.read(reinterpret_cast<char *>(data.data()), byte_nums)){
        cerr << "read error" << file.gcount() << "/" << byte_nums << endl;
    }
    return data;
}


vector<float> read_param_in2(const string path, size_t hint = 0){
    ifstream file(path);
    if(!file) {cerr << "file path error" << endl; return {};}
    vector<float> data;
    if(hint) data.reserve(hint);
    float x;
    while(file >> x) data.push_back(x);
    return data;
}

bool write_param_out(const string& path, vector<float> data){
    if(path.ends_with(".txt")){
        ofstream file(path);
        file << setprecision(9);
        for(auto x : data)
            file << x << "\n";
        return static_cast<bool>(file);
    }
    else{
        ofstream file(path, ios::binary);
        if(!file.write(reinterpret_cast<const char*>(data.data()), static_cast<streamsize>(data.size() * sizeof(float)))){
            cerr << "write file in error" << path << endl;
        }
        return static_cast<bool>(file);
    }
    
}


__global__ void reduction(float* input, float* output, int factor_num){
    auto idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    auto base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    auto block_size = blockDim.x * blockDim.y * blockDim.z;
    auto grid_size = gridDim.x * gridDim.y * gridDim.z;
    __shared__ float temp_mem[8];

    float data = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    for(int i = 0; i < factor_num; i += block_size * grid_size * 4){
        float4 temp = ((idx + base_addr * block_size) * 4 + i < factor_num) ? in4[idx + base_addr * block_size + i / 4] : make_float4(0.0f,0.0f,0.0f,0.0f);
        data += (temp.x + temp.y) + (temp.z + temp.w);
    }

    for(int i = 16; i > 0; i >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, i, 32);
    }

    if(idx % 32 == 0)
        temp_mem[idx / 32] = data;

    __syncthreads();

    if(idx / 32 == 0)
        data = idx < 8 ? temp_mem[idx] : 0.0f;
    
    for(int i = 4; i > 0; i >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, i, 32);
    }

    if(idx == 0)
        output[base_addr] = data;

}

__global__ void reduction2(float* input, float* output, int factor_num){
    auto block_size = blockDim.x * blockDim.y * blockDim.z;
    auto idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    float data = 0.0f;;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    __shared__ float temp_mem[8];

    for(int i = 0; i < factor_num; i += block_size * 4){
        float4 temp = i + idx * 4  < factor_num ? in4[i / 4 + idx] : make_float4(0.0f,0.0f,0.0f,0.0f);
        data += (temp.x + temp.y) + (temp.z + temp.w);
    }

    for(int i = 16; i > 0; i >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, i, 32);
    }

    if(idx % 32 == 0)
        temp_mem[idx / 32] = data;
    
    __syncthreads();

    if(idx / 32 == 0)
        data = idx < 8 ? temp_mem[idx] : 0.0f;

    for(int i = 4; i > 0; i >>= 1){
        data += __shfl_down_sync(0xffff'ffff, data, i, 32);
    }

    if(idx == 0)
        output[0] = data;
}













int main(int argc, char *argv[]){

    if(argc < 3){
        cerr << "ERROR : must enter load and output file path in argv[1] and argv[2]" << endl;
        return 1;
    }
    if(argc < 6){
        cout << "WARMING : without grid setting param" << endl;
        cout << "usage :" << argv[0] << "<grid.dim1>" << "<grid.dim2>" << "<grid.dim3>" << endl;
    }

    string path1 = argv[1];
    string path2 = argv[2];
  
    int grid_dim1 = argc > 3 ? stoi(argv[3]) : 16;
    int grid_dim2 = argc > 4 ? stoi(argv[4]) : 1;
    int grid_dim3 = argc > 5 ? stoi(argv[5]) : 1;

    int grid_size = grid_dim1 * grid_dim2 * grid_dim3;

    // int factor_num = 64 * 1024 * 1024;
    vector<float> data_out(1);
    vector<float> parm1;
    // vector<float> parm1(factor_num, 1.0f);
    
    if(path1.ends_with(".txt"))
        parm1 = read_param_in2(path1);
    else
        parm1 = read_param_in(path1);

    int factor_num = parm1.size();
    
    float *d_parm1;
    float *d_k1_output;
    float *d_k2_output;
    checkCudaErrors(cudaMalloc(&d_parm1, ((factor_num + 3) / 4 * 4) * sizeof(float)));
    checkCudaErrors(cudaMemset(d_parm1, 0, ((factor_num + 3) / 4 * 4) * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_k1_output, ((grid_size + 3) / 4 * 4)* sizeof(float)));
    checkCudaErrors(cudaMemset(d_k1_output, 0, ((grid_size + 3) / 4 * 4)* sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_k2_output, 1 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_parm1, parm1.data(), factor_num * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16, 1);
    dim3 grid(grid_dim1, grid_dim2, grid_dim3);

    dim3 grid2(1, 1, 1);

    reduction<<<grid, block>>>(d_parm1, d_k1_output, factor_num);
    checkCudaErrors(cudaGetLastError());
    reduction2<<<grid2, block>>>(d_k1_output, d_k2_output, grid_size);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(data_out.data(), d_k2_output, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    cout << "sum :" << data_out[0] << endl;
    cerr << "write out status :" << boolalpha << write_param_out(path2, data_out) << endl;

}