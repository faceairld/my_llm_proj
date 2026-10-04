#include <vector>
#include <cuda_fp16.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
  
using namespace std;

#define warp_size 32
#define block_size 256
#define tile_size (32*32)

struct martix {
    int M; int N;
    vector<half> data;
};

#define CheckcudaError(err) check(err, __FILE__, #err, __LINE__) 
void check(cudaError_t err, const char * const file, const char * const fun, const int line){
    if(err != cudaSuccess){
        cerr << "cuda error at :" << file << " : " << line << " : " << fun << endl;
        cerr << "cuda error :" << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

#define read_file(path) read_file1(path)
martix read_file1(const string &path){
    ifstream file(path, ios::binary);
    if(!file){cerr << "open file falied" << endl; exit(1);}
    martix p;
    int M; int N;
    if(!(file >> M >> N)){cerr << "read M N error" << endl; exit(1);}
    float temp;
    vector<half> data_in;
    data_in.reserve(M * N);
    while(file >> temp){
        data_in.push_back(static_cast<half>(temp));
    }
    if(data_in.size() != M * N){cerr << "read martix size error" << endl; exit(1);}
    p.M = M;
    p.N = N;
    p.data = data_in;
    return p;
}

#define write_file(path, data_out) write_file1(path, data_out)
bool write_file1(const string &path, const martix &p){
    ofstream file(path, ios::binary);
    if(!file){cerr << "create write file falied" << endl; exit(1);}
    int M = p.M;
    int N = p.N;
    file << setprecision(9);
    size_t count1 = 0;
    for(auto temp : p.data){
        file << static_cast<float>(temp) << " ";
        count1 ++;
        if(count1 % M == 0)
            file << "\n";
    }
    if(count1 != M * N){cerr << "write file size error" << endl; exit(1);}
    return static_cast<bool>(file);
}


__global__ void trans(const half* __restrict__ input, half* __restrict__ output, const int M, const int N){
        int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int block_base_x = blockIdx.x * 32;
        int block_base_y = blockIdx.y * 32;
        int base_addr = block_base_y * N + block_base_x;

        __shared__ half s_mem[32][66];
        //__shared__ half s_mem[32][34];

        for(int i = offset; i < tile_size; i += block_size){
            int P = block_base_y + i / 32; int C = block_base_x + i % 32;
            half temp = (P < M && C < N) ? input[P * N + C] : static_cast<half>(0.0f);
            s_mem[i / 32][(i % 32) * 2] = temp;
        }

        __syncthreads();

        for(int i = offset; i < tile_size; i += block_size){
            int P = block_base_x + i / 32; int C = block_base_y + i % 32;
            if(P < N && C < M)
                output[P * M + C] = s_mem[i % 32][(i / 32) * 2];
        }
}






int main(int argc, char *argv[]){
    if(argc < 3){
        cerr << "must input at least 2 parameter : input path and output path" << endl; exit(1); 
    }
    string path_in  = argv[1];
    string path_out = argv[2];

    martix p = read_file(path_in);
    int M = p.M;
    int N = p.N;
    vector<half> data_in = p.data;
    vector<half> data_out(data_in.size());

    half *d_data_in, *d_data_out;

    CheckcudaError(cudaMalloc(&d_data_in, sizeof(half) * data_in.size()));
    CheckcudaError(cudaMalloc(&d_data_out, sizeof(half) * data_in.size()));
    CheckcudaError(cudaMemcpy(d_data_in, data_in.data(), sizeof(half) * data_in.size(), cudaMemcpyHostToDevice));

    dim3 grid((N + 31) / 32, (M + 31) / 32, 1);
    dim3 block(16, 16, 1);

    trans<<<grid, block>>>(d_data_in, d_data_out, M, N);
    CheckcudaError(cudaGetLastError());
    CheckcudaError(cudaDeviceSynchronize());
    CheckcudaError(cudaFree(d_data_in));

    CheckcudaError(cudaMemcpy(data_out.data(), d_data_out, sizeof(half) * data_out.size(), cudaMemcpyDeviceToHost));
    CheckcudaError(cudaFree(d_data_out));


    martix p_out;
    p_out.M = M;
    p_out.N = N;
    p_out.data = data_out;

    cout << "write file status :" << boolalpha << write_file(path_out, p_out) << endl;

}






