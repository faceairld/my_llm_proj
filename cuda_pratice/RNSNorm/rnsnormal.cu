#include <fstream>
#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <vector>
#include <iomanip>

using namespace std;

#define block_size 256
#define warp_size 32
#define eps 1e-6f

#define CheckCudaError(err) check(err, __FILE__, #err, __LINE__)
void check(cudaError_t error, const char * const file, const char * const fun, const int line){
    if(error != cudaSuccess){
        cerr << "cuda run failed in :" << file << " : " << line << " : " << fun << endl;
        cerr << "cuda error :" << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

struct matrix{
    int M; int N;
    vector<half> data;
};

#define read_file(path) read_file1(path)
matrix read_file1(const string &path){
    ifstream file(path, ios::binary);
    if(!file){cerr << "read file failed :" << " open file failed" << endl; exit(1);};
    matrix m;
    int M; int N;
    if(!(file >> M >> N)){cerr << "read file failed :" << "read M N filed" << endl; exit(1);};
    if(N % 2 != 0){cerr << "cols number of matrix must be devide by 2" << endl; exit(1);}
    int dummy_size = (N + 7) / 8 * 8 - N;
    vector<half> in_data;
    float x;
    int count1 = 0;
    while(file >> x){
        count1 ++;
        in_data.push_back(static_cast<half>(x));
        if(count1 % N == 0){
            in_data.resize((in_data.size() + 7) / 8 * 8, 0.0f);
        }
    }
    if(count1 != M * N || in_data.size() % ((N + 7) / 8 * 8 )!= 0){cerr << "read file failed :" << "read matrix" << endl; exit(1);};
    m.data = in_data;
    m.M = M; m.N = N;
    return m;
}

#define write_file(path, m_in) write_file1(path, m_in)
bool write_file1(const string &path, const matrix m_in){
    ofstream file(path);
    if(!file){cerr << "write file failed :" << "open file failed" << endl; exit(1);}
    size_t size = 0;
    file << setprecision(9);
    for(auto x : m_in.data){
        file << static_cast<float>(x) << " ";
        size ++;
        if(size % m_in.N == 0){
            file << "\n";
        }
    }
    if(size != m_in.M * m_in.N){cerr << "write file error :" << " file size not match" << endl; exit(1);}
    return static_cast<bool>(file);

}

__global__ void RNSnorm1(const half * __restrict__ input, half * __restrict__ output, const int M, const int N){
    int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int base = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const half2* h2 = reinterpret_cast<const half2*>(input);
     half2* h2_o = reinterpret_cast< half2*>(output);
    
    __shared__ float share_mem[8];

    for(int i = blockIdx.x; i < M; i += gridDim.x){
        float sum = 0.0f;
        for(int j = offset; j < (N + 7) / 8; j += block_size){ // (N + 1) / 8
            int in_base_addr = i * ((N + 7) / 8 * 8) / 2;
            #pragma unroll
            for(int k = 0; k < 4; k ++){
                half2 temp = h2[in_base_addr + j * 4 + k];
                sum = fmaf(temp.x, temp.x, sum); 
                sum = fmaf(temp.y, temp.y, sum);
            }

        }

        for(int j = 16; j > 0; j >>= 1){
            sum += __shfl_down_sync(0xffff'ffff, sum, j, 32);
        }

        if(offset % 32 == 0)
            share_mem[offset / 32] = sum;
        
        __syncthreads();

        if(offset < 32){
            sum = offset < 8 ? share_mem[offset] : 0.0f;
        }

        for(int j = 16; j > 0; j >>= 1){
            sum += __shfl_down_sync(0xffff'ffff, sum, j, 32);
        }

        if(offset == 0)
            share_mem[0] = rsqrtf(sum / N + eps);

        __syncthreads();

        sum = share_mem[0];
        
        for(int j = offset; j < (N + 7) / 8; j += block_size){ // (N + 1) / 8
            int in_base_addr = i * ((N + 7) / 8 * 8) / 2;
            int out_base_addr = i * N / 2;
            #pragma unroll
            for(int k = 0; k < 4; k ++){
                // half2 temp =  (j * 8 + k * 2) < N ? 
                //     make_half2(static_cast<float>(h2[in_base_addr + j * 4 + k].x )* sum, static_cast<float>(h2[in_base_addr + j * 4 + k].y )* sum) :
                //         make_half2(0.0f, 0.0f);
                half2 temp1 = h2[in_base_addr + j * 4 + k];
                half2 temp = make_half2(static_cast<float>(temp1.x)* sum, static_cast<float>(temp1.y)* sum);
                if((j * 8 + k * 2) < N ){
                    h2_o[out_base_addr + j * 4 + k] = temp;
                }

            }
        }

        __syncthreads();

    }
} 

// __global__ void RNSnorm2(const float * __restrict__ input, float * __restrict__ output, const int M, const int N){
//     int base = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
//     int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;


// }



int main(int argc, char *argv[]){
    if(argc < 3){
        cerr << "must input at least 2 parameter : input path , output path" << endl;
        exit(1);
    }

    string path_in = argv[1];
    string path_out = argv[2];

    matrix p;
    p = read_file(path_in);
    int M = p.M;
    int N = p.N;
    vector<half> h_data = p.data; 
    vector<half> h_data_out(M * N);

    half* d_data1_in;
    half* d_data1_out;

    CheckCudaError(cudaMalloc(&d_data1_in, sizeof(half) * h_data.size()));
    CheckCudaError(cudaMalloc(&d_data1_out,sizeof(half) * h_data_out.size()));
    CheckCudaError(cudaMemcpy(d_data1_in, h_data.data(), sizeof(half) * h_data.size(), cudaMemcpyHostToDevice));
    
    int grid_dim1 = min(M, 180);

    dim3 grid(grid_dim1, 1, 1);
    dim3 block(16, 16, 1);

    RNSnorm1<<<grid, block>>>(d_data1_in, d_data1_out, M, N);
    CheckCudaError(cudaGetLastError());
    
    CheckCudaError(cudaMemcpy(h_data_out.data(), d_data1_out, sizeof(half) * h_data_out.size(),cudaMemcpyDeviceToHost));

    matrix p_out;
    p_out.data = h_data_out;
    p_out.M = M;
    p_out.N = N;

    cout << "write file status :" << boolalpha << write_file(path_out, p_out) << endl;

    CheckCudaError(cudaFree(d_data1_in));
    CheckCudaError(cudaFree(d_data1_out));

}