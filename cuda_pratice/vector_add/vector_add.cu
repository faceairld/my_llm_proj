#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;


#define checkCudaError(val) check(val, __FILE__, #val, __LINE__)
void check(cudaError_t err, const char * const file, const char * const fun, const int line){
    if(err != cudaSuccess){
        cerr << "cuda error at" << file << ":" << line << endl;
        cerr <<  cudaGetErrorString(err) << " " << fun << endl;
        exit(1);
    }
}

#define read_data(path) path.ends_with(".txt") ? read_data_in2(path) : read_data_in(path)
vector<float> read_data_in(const string& path){
    ifstream file(path, ios::binary | ios::ate);
    streamsize len = file.tellg();
    if(len < 0 || len % sizeof(float) != 0){
        cerr << "file size not match"<< endl;
        exit(1);
    }
    file.seekg(0,ios::beg);
    vector<float> data(len / sizeof(float));
    if(!file.read(reinterpret_cast<char *>(data.data()), len)){
        cerr << "file read error" << endl;
        exit(1);
    }
    return data;

}

vector<float> read_data_in2(const string& path){
    ifstream file(path);
    if(!file){cerr << "open file failed" << endl; exit(1);}
    vector<float> res;
    float x;
    while(file >> x){
        res.push_back(x);
    }
    return res;
}

#define write_data(path, data_in) write_data_out(data_in, path)
bool write_data_out(const vector<float> &data_in, const string &path){
    if(path.ends_with(".txt")){
        ofstream file(path);
        if(!file){cerr << "open file failed" << endl; exit(1);}
        file << setprecision(9);
        for(auto c : data_in){
            file << c << "\n";
        }
        return static_cast<bool> (file);
    }else
    {ofstream file(path, ios::binary);
    streamsize len = data_in.size() * sizeof(float);
    if(!file.write(reinterpret_cast<const char*>(data_in.data()), len)){
        cerr << "file write error in" << endl;
        exit(1);
    }
    return static_cast<bool> (file);
    }
}


__global__ void add1(const float * __restrict__ input2, const float * __restrict__ input, float * __restrict__ output, int input_nums){
    int base_addr = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;
    const float4 * in4 = reinterpret_cast<const float4 *> (input);
    const float4 * in4_2 = reinterpret_cast<const float4 *> (input2);
    float4 * out = reinterpret_cast<float4 *> (output);
    float4 data_in;
    float4 data_in2;
    for(int i = 0; i < input_nums; i += grid_size * block_size * 4){
        data_in = ((base_addr * block_size + idx) * 4 + i < input_nums) ? in4[i / 4 + base_addr * block_size + idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        data_in2 = ((base_addr * block_size + idx) * 4 + i < input_nums) ? in4_2[i / 4 + base_addr * block_size + idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        data_in.x += data_in2.x;
        data_in.y += data_in2.y;
        data_in.z += data_in2.z;
        data_in.w += data_in2.w;
        if((base_addr * block_size + idx) * 4 + i < input_nums) out[i / 4 + base_addr * block_size + idx] = data_in;
    }
}



int main(int argc, char* argv[]){
    if(argc < 4){
        cerr << "parameter list : input_file_path1, input_file_path2, output_file_path(.txt or .bin)"<< "\n" <<
                "alter : grid_dim1  grid_dim2   grid_dim3"<<"\n"<<
                "you nead put at least 3 prarmters" << endl;
        exit(1);
    }
    string path_in = argv[1];
    string path_in2 = argv[2];
    string path_out = argv[3];

    int grid_dim1 = (argc > 4) ? stoi(argv[4]) : 16;
    int grid_dim2 = (argc > 5) ? stoi(argv[5]) : 1;
    int grid_dim3 = (argc > 6) ? stoi(argv[6]) : 1;

    vector<float>h_data1 = read_data(path_in);
    vector<float>h_data2 = read_data(path_in2);
    if(h_data1.size() != h_data2.size())
        cerr << "the len of two vector are not same" << endl; 
    vector<float>h_dout(h_data1.size(), 0.0f);
    float * d_data1;
    float * d_data2;
    float * d_dout;
    

    checkCudaError(cudaMalloc(&d_data1, (h_data1.size() + 3) / 4 * 4 * sizeof(float)));
    checkCudaError(cudaMalloc(&d_data2, (h_data2.size() + 3) / 4 * 4 * sizeof(float)));
    checkCudaError(cudaMalloc(&d_dout, (h_dout.size() + 3) / 4 * 4 * sizeof(float)));

    checkCudaError(cudaMemset(d_data1, 0, (h_data1.size() + 3) / 4 * 4 * sizeof(float)));
    checkCudaError(cudaMemset(d_data2, 0, (h_data2.size() + 3) / 4 * 4 * sizeof(float)));
    checkCudaError(cudaMemset(d_dout, 0, (h_dout.size() + 3) / 4 * 4 * sizeof(float)));

    checkCudaError(cudaMemcpy(d_data1, h_data1.data(), h_data1.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_data2, h_data2.data(), h_data2.size() * sizeof(float), cudaMemcpyHostToDevice));

   
    dim3 grid1(grid_dim1, grid_dim2, grid_dim3);
    dim3 block1(16, 16 ,1);

    add1<<<grid1, block1>>>(d_data1, d_data2, d_dout, h_data1.size());
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(h_dout.data(), d_dout, h_dout.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "write out status :"<< boolalpha << write_data(path_out, h_dout) << endl;
    


}