#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cuda_fp16.h>

using namespace std;
#define warp_size 32
#define block_size 256
#define BM (16 * 8)
#define BN (16 * 8)
#define BK 8

struct GemmData{
    int M = 0, N = 0, K = 0;
    float alpha = 0.0f, beta = 0.0f;
    vector<half> A, B, C;
};

#define CheckcudaError(val) check(val, __FILE__, #val, __LINE__)
void check(cudaError_t err, const char * const file, const char * const fun, const int line){
    if(err != cudaSuccess){
        cerr << "cuda Error :" << cudaGetErrorString(err) << endl;
        cerr << "cuda Error in : " << file << " : " << fun <<  " : " << line << endl;
        exit(1);
    }
}


#define read_file(path) path.ends_with(".txt") ? read_file1(path) : read_file2(path)
GemmData read_file1(const string &path){
    ifstream file(path, ios::binary);
    GemmData p;
    if(!file){cerr << "read file error :" << "open file error" << endl; exit(1);}
    if(!(file >> p.M >> p.N >> p.K >> p.alpha >> p.beta)){
        cerr << "read file error :" << "parameter input error" << endl; 
        exit(1);
    }
    auto load = [&](vector<half> & dst, size_t n, const char* name){
        float f;
        dst.resize(n);
        for(int i = 0; i < n; i++){
            if(!(file >> f)){cerr << "read file error :" << "martix data input error in martix :" << name << endl; exit(1);}
            dst[i] = static_cast<half>(f);
        }
    };
    load(p.A, size_t(p.M) * p.K, "A");
    load(p.B, size_t(p.N) * p.K, "B");
    load(p.C, size_t(p.M) * p.N, "C");

    return p;
    
}

GemmData read_file2(const string &path){
    ifstream file(path, ios::binary);
    if(!file){cerr << "read file error : " << "open file filed" << endl; exit(1);}
    GemmData p;
    int MNK[3]; float parameter[2];
    if(!file.read(reinterpret_cast<char *>(MNK), 3 * sizeof(int))){
        {cerr << "read file error : " << "read MKN filed" << endl; exit(1);};
    }
    if(!file.read(reinterpret_cast<char *>(parameter), 2 * sizeof(float))){
        {cerr << "read file error : " << "read alpha beta filed" << endl; exit(1);};
    }
    p.M = MNK[0]; p.N = MNK[1]; p.K = MNK[2];
    p.alpha = parameter[0]; p.beta = parameter[1];
    vector<half> A_m(p.M * p.K); vector<half> B_m(p.N * p.K); vector<half> C_m(p.M * p.N);

    if(!file.read(reinterpret_cast<char *>(A_m.data()), A_m.size() * sizeof(half))){
        {cerr << "read file error : " << "read A filed" << endl; exit(1);};
    }
    if(!file.read(reinterpret_cast<char *>(B_m.data()), B_m.size() * sizeof(half))){
        {cerr << "read file error : " << "read B filed" << endl; exit(1);};
    }
    if(!file.read(reinterpret_cast<char *>(C_m.data()), C_m.size() * sizeof(half))){
        {cerr << "read file error : " << "read C filed" << endl; exit(1);};
    }

    p.A = A_m; p.B = B_m; p.C = C_m;
    return p;
}
#define write_file(path, output) write_file(path, output)
bool write_file(const string &path, const vector<half> &output){
    ofstream file(path, ios::binary);
    if(!file){cerr << "write file error :" << "open file error" << endl; exit(1);}
    file << setprecision(9);
    int count1 = 0;
    for(auto c : output){
        file << static_cast<float>(c) << " ";
        count1 = (count1 + 1) % 4;
        if(count1 == 3)
        file << "\n"; 
    }
    return static_cast<bool>(file);
}



__global__ void gemm1(const half * __restrict__ input_a, const half * __restrict__ input_b, const half *__restrict__ input_c, 
                            half * __restrict__ output, const int M, const int N, const int K, const float alpha, const float beta){
    int base_addr = blockIdx.x  + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int thread_x = threadIdx.x + blockDim.x * blockIdx.x;
    int thread_y = threadIdx.y + blockDim.y * blockIdx.y;

    int row_c = blockIdx.y * BM;
    int col_c = blockIdx.x * BN;


    __shared__ half A_mem[BM * BK];
    __shared__ half B_mem[BK * BN]; // BK * BN

    float data[8][8] = {};

    // const float4 *in4 = reinterpret_cast<const float4*> (input);
    for(int j = 0; j < K; j += BK){   
        for(int i = offset; i < BM * BK ; i += block_size){
            int base_y = row_c;
            int offset_y = i / BK;
            int base_x = j ;
            int offset_x = i % BK;
            half a_data = base_y + offset_y < M  && base_x + offset_x < K? input_a[(base_y + offset_y) * K + (base_x + offset_x)] : static_cast<half>(0.0f);
            A_mem[i] = a_data;
        }

        for(int i = offset; i < BN * BK ; i += block_size){
            int base_y = col_c;
            int offset_y = i / BK;
            int base_x = j;
            int offset_x = i % BK;
            half b_data = base_y + offset_y < N && base_x + offset_x < K? input_b[(base_y + offset_y) * K + (base_x + offset_x)] : static_cast<half>(0.0f);
            B_mem[offset_x * BN + offset_y] = b_data;
        }
        
        __syncthreads();
        #pragma unroll
        for(int idx_BK = 0; idx_BK < BK; idx_BK ++){
            half A_temp[8];
            for(int i = 0; i < 8; i++){
                int x = idx_BK;
                int y = threadIdx.y * 8;
                A_temp[i] = A_mem[(y + i) * BK + x];
            }
            half B_temp[8];
            for(int i = 0; i < 8; i++){
                int y = idx_BK;
                int x = threadIdx.x * 8;
                B_temp[i] = B_mem[y * BN + x + i];
            }

            for(int x = 0; x < 8; x ++){
                for(int y = 0; y < 8; y ++){
                    data[y][x] = fmaf(A_temp[y], B_temp[x], data[y][x]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int x = 0; x < 8; x ++){
        for(int y = 0; y < 8; y ++){
            int r_y = row_c + threadIdx.y * 8 + y;
            int r_x = col_c +  + threadIdx.x * 8 + x;
            if(r_y < M && r_x < N){
                data[y][x] = data[y][x] * alpha + static_cast<float>(input_c[r_y * N + r_x]) * beta;
                output[r_y * N + r_x] = static_cast<half>(data[y][x]);
            }
        }
    }
}






int main (int argc, char *argv[]){
    if(argc < 3){
        cerr << "you must input at least input path and output path" << endl;
        exit(1);
    }
    string path_in = argv[1];
    string path_out = argv[2];

    // int grid_dim1 = argc > 3 ? stoi(argv[3]) : 4;
    // int grid_dim2 = argc > 4 ? stoi(argv[4]) : 4;
    // int grid_dim3 = argc > 5 ? stoi(argv[5]) : 1;

    

    GemmData p = read_file(path_in);
    int M = p.M;
    int N = p.N;
    int K = p.K;
    float alpha = p.alpha;
    float beta = p.beta;
    vector<half> A = p.A;
    vector<half> B = p.B;
    vector<half> C = p.C;

    vector<half> out_data(C.size(), 0.0f);

    half *d_A, *d_B, *d_C, *d_out;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    dim3 block(16, 16, 1);

    CheckcudaError(cudaMalloc(&d_A, sizeof(half) * A.size()));
    CheckcudaError(cudaMalloc(&d_B, sizeof(half) * B.size()));
    CheckcudaError(cudaMalloc(&d_C, sizeof(half) * C.size()));
    CheckcudaError(cudaMalloc(&d_out, sizeof(half) * out_data.size()));
    
    CheckcudaError(cudaMemcpy(d_A, A.data(), sizeof(half) * A.size(), cudaMemcpyHostToDevice));
    CheckcudaError(cudaMemcpy(d_B, B.data(), sizeof(half) * B.size(), cudaMemcpyHostToDevice));
    CheckcudaError(cudaMemcpy(d_C, C.data(), sizeof(half) * C.size(), cudaMemcpyHostToDevice));

    gemm1<<<grid, block>>>(d_A, d_B, d_C, d_out, M, N, K, alpha, beta);
    CheckcudaError(cudaGetLastError());

    CheckcudaError(cudaMemcpy(out_data.data(), d_out, sizeof(half) * out_data.size(), cudaMemcpyDeviceToHost));
    cout << "write status :" << boolalpha << write_file(path_out, out_data) << endl;

}







