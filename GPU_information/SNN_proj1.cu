#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <map>

#include <cstdio>
#include <cstdlib>
#include <cmath> 
#include <thread>

#ifdef _MSC_VER
#include <stdlib.h>
#define __builtin_bswap32 _byteswap_ulong
#endif

#define Tile_size_in 20
#define Tile_size_out 16
#define warp_size 32
#define blocksize 256 //16*16
#define conv1_channel_num 6
#define conv2_channel_num 16
#define total_loop_times 8

#define test_images_num 10048




// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
//std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
//    std::ifstream file(path, std::ios::binary);
//    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
//    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
//    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
//    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
//    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
//    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
//    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
//    std::vector<unsigned char> buffer(num_rows * num_cols);
//    for (int i = 0; i < num_images; ++i) {
//        file.read((char*)buffer.data(), buffer.size());
//        for (size_t j = 0; j < buffer.size(); ++j) {
//            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
//        }
//    }
//    return images;
//}

static  int col;
static  int row;

std::vector<float> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    col = num_cols;
    row = num_rows;
    std::vector<float> images(num_images * num_rows * num_cols);
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i * buffer.size() + j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

//    bool read_mnist_images(const std::string& path, float * pinned_mem_ptr) {
//    std::ifstream file(path, std::ios::binary);
//    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return false;}
//    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
//    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
//    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
//    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
//    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
//    col = num_cols;
//    row = num_rows;
//    //std::vector<float> images(num_images * num_rows * num_cols);
//    std::vector<unsigned char> buffer(num_rows * num_cols);
//    for (int i = 0; i < num_images; ++i) {
//        file.read((char*)buffer.data(), buffer.size());
//        for (size_t j = 0; j < buffer.size(); ++j) {
//            pinned_mem_ptr[i * buffer.size() + j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
//        }
//    }
//    return true;
//}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

std::vector<float> read_param_w1(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params(150); float param;
    int i = 0;
    while (file >> param) {
        int out_c = i / (5 * 5);
        int y = (i % (5 * 5)) / 5;
        int x = i % 5;       
        params[out_c + x * 6 + y * 30] = param;
        i ++ ;
    }
    return params;
}

std::vector<float> read_param_fc_w1(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params(30720); float param;
    int i = 0;
    while (file >> param) {
        int count = i % 256;
        int nums = i / 256;
        params[count * 120 + nums] = param;
        i ++ ;
    }
    return params;
}

std::vector<float> read_param_fc_w2(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params(10080); float param;
    int i = 0;
    while (file >> param) {
        int count = i % 120;
        int nums = i / 120;
        params[count * 84 + nums] = param;
        i ++ ;
    }
    return params;
}

std::vector<float> read_param_fc_w3(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params(840); float param;
    int i = 0;
    while (file >> param) {
        int count = i % 84;
        int nums = i / 84;
        params[count * 10 + nums] = param;
        i ++ ;
    }
    return params;
}


// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================
    __constant__ float conv1_core[150];
    __constant__ float conv1_bias[6];
    __constant__ float conv2_core[2400];
    __constant__ float conv2_bias[16];
    //__constant__ float fc3_core[84][10];
    //__constant__ float fc3_bias[10];
    
    
    //block: 16*16 = 256
    //grid : 2*2*64
    //use constant memory
    __global__ void conv1_kernel(/*float *d_conv1_w,float *d_conv1_b,*/float* images_t,
                                    /*float* conv_core,float* conv_bias,*/float* conv1_sum_out,
                                        float*conv1_ReLU_out, int Loop_times){
        int thread_x = blockDim.x*blockIdx.x+threadIdx.x;
        int thread_y = blockDim.y*blockIdx.y+threadIdx.y;

        int idx = threadIdx.y*blockDim.x+threadIdx.x;
        int images_num = blockIdx.z;

        __shared__  float Tile_block2[Tile_size_in][Tile_size_in]; // for picture pixel
       
        
        for(int i = idx; i < Tile_size_in *Tile_size_in;i+= blocksize){
            int in_row = i%Tile_size_in;
            int in_col = i/Tile_size_in;

            int globel_pix_x = blockIdx.x*blockDim.x + in_row - 2;
            int globel_pix_y = blockIdx.y*blockDim.y + in_col - 2;
    
            if(globel_pix_x>=0 && globel_pix_x<28 && globel_pix_y>=0 && globel_pix_y<28){
                Tile_block2[in_col][in_row] = images_t[globel_pix_y*28+globel_pix_x + images_num * 784];
            }else{
                Tile_block2[in_col][in_row] = 0.0f;
            }
        
        }
        __syncthreads();
        
        float sum[6] = {0.0};
        float out[6] = {0.0};

        if(Loop_times > 0){
            for(int i = 0; i < 6 ; i++){
                int idx0 = images_num * conv1_channel_num * 24 * 24 + i * 24 * 24 + thread_y * 24 + thread_x;
                sum[i] = conv1_sum_out[idx0];
            }
        }

        for(int i = 0;i < 25; i++){
            int row = i % 5;
            int col = i / 5;
            float pixel = Tile_block2[threadIdx.y + col][threadIdx.x + row];
            for(int j = 0; j < 6; j++){
                sum[j] +=  pixel * conv1_core[j + 6 * row + 5 * 6 * col];
            }
            
        }
        for(int i = 0; i < 6; i++){
            sum[i] += conv1_bias[i];
            out[i] = sum[i] >= 1.0 ? 1.0 : 0.0;
            sum[i] = sum[i] * (1.0 - out[i]);
        }
   
        if(thread_x < 24 && thread_y < 24){
            for(int i = 0;i < 6; i++){
                conv1_sum_out[images_num * conv1_channel_num * 24 * 24 + i * 24 * 24 + thread_y * 24 + thread_x] = sum[i];
                conv1_ReLU_out[images_num * conv1_channel_num * 24 * 24 + i * 24 * 24 + thread_y * 24 + thread_x] = out[i];
            }
        }

        
    };

    //block: 12 * 12 * 6
    //grid: 1*1*64
    __global__ void Maxpool1(float *din , float *Maxpool_out){
        int images_num = blockIdx.z;
        int images_idx_x = threadIdx.x * 2;
        int images_idx_y = threadIdx.y * 2;
        int channel = threadIdx.z;
        

        float p1 =  din[images_num * 24*24 * conv1_channel_num + channel * 24*24 + images_idx_x + images_idx_y * 24];
        float p2 =  din[images_num * 24*24 * conv1_channel_num + channel * 24*24 + images_idx_x + (images_idx_y + 1) * 24];
        float p3 =  din[images_num * 24*24 * conv1_channel_num + channel * 24*24 + images_idx_x + 1 + images_idx_y * 24];
        float p4 =  din[images_num * 24*24 * conv1_channel_num + channel * 24*24 + images_idx_x + 1 + (images_idx_y + 1) * 24];

        float max_pixel = fmaxf(fmaxf(p1,p2),fmaxf(p3,p4));
        

        Maxpool_out[images_num * 12 * 12 * conv1_channel_num + channel * 12 * 12 + threadIdx.x + threadIdx.y * 12] = max_pixel;

    }

    

    // block : 8 * 8 * 16
    // grid : 1 * 1 * 64
    // use constant memory
    __global__ void conv2_kernel(/*float *d_conv2_w,float *d_conv2_b,*/float* conv1_o,
                                    /*float* conv_core,float* conv_bias,*/float* conv2_sum_out,
                                        float*conv2_ReLU_out,int Loop_times){
        int thread_x = blockDim.x*blockIdx.x+threadIdx.x;
        int thread_y = blockDim.y*blockIdx.y+threadIdx.y;
        int thread_z = threadIdx.z;

        int idx = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
        int images_num = blockIdx.z;
        int global_pixel = images_num * 12 * 12 * 6 + idx;

        int local_addr = blockIdx.z * conv2_channel_num * 8 * 8 + threadIdx.z * 8 * 8 + threadIdx.y * 8 + threadIdx.x; 

        __shared__   float Tile_block2[6][12][12];

        int z = idx / (12 * 12);
        int m = idx % (12 * 12);
        int y = m / 12;
        int x = m % 12;

        if (idx < 12 * 12 * 6)
            Tile_block2[z][y][x] = conv1_o[global_pixel];
            

        __syncthreads();
        
        float sum = {0.0};
        float out = {0.0};
        
        if(Loop_times > 0){
                sum = conv2_sum_out[local_addr];    
        }


        for(int k = 0; k < 6; k++){
            for(int i = 0; i < 5 * 5 ; i++){
                //int num = i / (5 * 5);
                int row = i % 5;
                int col = i / 5;   
                sum += Tile_block2[k][thread_y + col][thread_x + row] * conv2_core[thread_z * 25 * 6 + k * 25 + col * 5 + row];
            }    
        }

        sum += conv2_bias[thread_z];
        out = sum >= 1.0 ? 1.0 : 0.0;
        sum = sum * (1.0 - out);

        
        conv2_sum_out[local_addr] = sum;
        conv2_ReLU_out[local_addr] = out;
        
    };

    //block: 4*4*16
    //grid: 1*1*64
    __global__ void Maxpool2(float *din , float *Maxpool_out){
        int images_num = blockIdx.z;
        int images_idx_x = threadIdx.x * 2;
        int images_idx_y = threadIdx.y * 2;
        int channel = threadIdx.z;

        float p1 =  din[images_num * 8*8 * conv2_channel_num + channel * 8*8 + images_idx_x + images_idx_y * 8];
        float p2 =  din[images_num * 8*8 * conv2_channel_num + channel * 8*8 + images_idx_x + (images_idx_y + 1) * 8];
        float p3 =  din[images_num * 8*8 * conv2_channel_num + channel * 8*8 + images_idx_x + 1 + images_idx_y * 8];
        float p4 =  din[images_num * 8*8 * conv2_channel_num + channel * 8*8 + images_idx_x + 1 + (images_idx_y + 1) * 8];

        float max_pixel = fmaxf(fmaxf(p1,p2),fmaxf(p3,p4));

        Maxpool_out[images_num * 4 * 4 * conv2_channel_num + channel * 4 * 4 + threadIdx.x + threadIdx.y * 4] = max_pixel;

    }
    
    //block: 128*1*1
    //grid: 1*1*64

    __global__ void fc1(float *din, float *conv1_core,float* bias, float* fc1_sum, float* fc1_out, int Loop_times){
        int idx = threadIdx.x;
        if(idx >= 120) return;

        int images_num = blockIdx.z;
        int signal_imag = 16 * 4 * 4;
        int weight_idx = idx * signal_imag;
        int local_addr = 120 * images_num + idx;
        
        float sum = 0.0;
        float out = 0.0;
        if(Loop_times > 0){
            sum = fc1_sum[local_addr];
        }

    
        for(int i = 0; i < signal_imag; i ++){
            sum += din[images_num * signal_imag + i] * conv1_core[i * 120 + idx]; 
        }
            //fc1_out[images_num * 120 + idx] = sum;
    
        sum += bias[idx];

        out = sum >= 1.0 ? 1.0 : 0.0;
        sum = sum * (1.0 - out);
        fc1_sum[local_addr] = sum;
        fc1_out[local_addr] = out;
    }


    //block 96 * 2 * 1
    //grid 1 * 1 * 32
    __global__ void fc2(float *din, float *conv2_core,float* bias, float* fc2_sum, float* fc2_out, int Loop_times){
        int thread_x = threadIdx.x;
        int thread_y = threadIdx.y;
        //int idx = thread_x + thread_y * 96;
        if(thread_x >= 84)
            return ;

        int images_group = blockIdx.z;
        int images_nums = threadIdx.y;
        int signal_imag = 120;
        int local_addr = 2 * 84 * images_group + thread_y * 84 + thread_x;
        
        float sum = {0.0};
        float out = {0.0};
        if(Loop_times > 0){
            sum = fc2_sum[local_addr];
        }

        for(int i = 0; i < signal_imag; i ++){
            sum += din[images_group * signal_imag * 2 + thread_y * signal_imag + i] * conv2_core[i * 84 + thread_x]; 
        }
            //fc1_out[images_num * 120 + idx] = sum;
    
        sum += bias[thread_x];

        out = sum >= 1.0 ? 1.0 : 0.0;
        sum = sum * (1.0 - out);
        fc2_sum[local_addr] = sum;
        fc2_out[local_addr] = out;
    }


    //block 96 * 1 * 1 
    //grid 1 * 1 * 8
    //use constant memory
    __global__ void fc3(float *din, float *conv3_core,float* bias, float* fc3_sum, float* fc3_out, int Loop_times){
        if(threadIdx.x >= 80)
            return ;
        int signal_imag = 84;
        int image_num = threadIdx.x / 10;
        int conv_core_num = threadIdx.x % 10;
        
        int images_group = blockIdx.z;
        int local_addr = 8 * 10 * images_group + image_num * 10 + conv_core_num;
        
        float sum = {0.0};
        float out = {0.0};
        if(Loop_times > 0){
            sum = fc3_sum[local_addr];
        }

        for(int i = 0; i < signal_imag; i ++){
            sum += din[images_group * signal_imag * 8 + image_num * signal_imag + i] * conv3_core[i * 10 + conv_core_num]; 
        }
            //fc1_out[images_num * 120 + idx] = sum;
    
        sum += bias[conv_core_num];

        //out = sum >= 1.0 ? 1.0 : 0.0;
        //sum = sum * (1.0 - out);
        //fc3_sum[local_addr] = sum;
        if(Loop_times == 0){
            fc3_out[local_addr] = sum;
        }else{
            fc3_out[local_addr] += sum;
        }
        
    }



    
    
    
   enum images_batch_size {I64_perb = 64,I32_perb = 32,I16_perb = 16};
//   const float* images_set(const std::vector<float> &images, images_batch_size images_batch){
//       static const float* cur_image_set = images.data();
//       if(cur_image_set >= images.data() + images.size())
//           return nullptr;
//       const float* nex_image_set = cur_image_set;
//       cur_image_set += std::min(static_cast<ptrdiff_t>(images_batch * col * row), images.data() + images.size() - nex_image_set);
//       return nex_image_set;   
//   }
//   
    float* labels_set(float* labels, images_batch_size images_batch, int Loop_times){
       float* cur_image_set = labels + Loop_times * images_batch * 10;
       if(cur_image_set >= labels + test_images_num * 10)
           return nullptr;
       return cur_image_set;   
   }

  const float* images_set(const std::vector<float> &images, images_batch_size images_batch, int Loop_times){
       const float* cur_image_set = images.data() + Loop_times * images_batch * col * row;
       if(cur_image_set >= images.data() + images.size())
           return nullptr;
       return cur_image_set;   
   }

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" <<"<images_batch_size>"<< std::endl;
        return 1;
    }
	std::string dir = argv[1];

    std::map<std::string, images_batch_size> input_change{
        {"I64_perb",I64_perb},
        {"I32_perb",I32_perb},
        {"I16_perb",I16_perb}
    };
	images_batch_size b_size = input_change[argv[2]];
    // Load test data

    //float *images;
    
    //auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto images = read_mnist_images(dir + /*"/../../.." */+ "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + /*"/../../.." */+ "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    //size_t images_byte = images.size() * sizeof(float); 
    //memset(images.data() + images.size(),0,(test_images_num - images.size())*sizeof(float));
    images.resize(test_images_num * col * row, 0.0f);
    size_t images_byte = test_images_num * 28 * 28 * sizeof(float);

    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param_w1(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");

    auto fc1_w = read_param_fc_w1(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param_fc_w2(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param_fc_w3(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    //173KB - 10KB = 163KB
    //checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float))); //5*5*6
    //checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float))); //6
    //checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float))); //5*5*6*16
    //checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float))); //16
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float))); //4*4*16*120
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float))); //120
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float))); //120*84
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float))); //84
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float))); //84*10
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float))); //10

    // --- 2. Copy constant parameters from host to device ---
    //checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================
    
    float *d_mem_A, *d_mem_B;
    float *conv1_sum_out, *conv1_ReLU_out, *Maxpool1_out , *conv2_sum_out, *conv2_ReLU_out, *Maxpool2_out;
    float *fc1_sum, *fc1_out, *fc2_sum, *fc2_out, *fc3_sum, *fc3_out;
    float *final_data;
    //2520KB
    checkCudaErrors(cudaMalloc(&conv1_sum_out, conv1_channel_num * b_size * 24*24 * sizeof(float))); //6*64*24*24
    checkCudaErrors(cudaMalloc(&conv1_ReLU_out, conv1_channel_num * b_size * 24*24 * sizeof(float))); //6*64*24*24
    checkCudaErrors(cudaMalloc(&Maxpool1_out, conv1_channel_num * b_size * 12*12* sizeof(float))); //6*64*12*12
    checkCudaErrors(cudaMalloc(&conv2_sum_out,conv2_channel_num * b_size * 8*8 * sizeof(float)));//16*64*8*8
    checkCudaErrors(cudaMalloc(&conv2_ReLU_out,conv2_channel_num * b_size * 8*8 * sizeof(float)));//16*64*8*8
    checkCudaErrors(cudaMalloc(&Maxpool2_out, conv2_channel_num * b_size * 4*4 * sizeof(float)));//16*64*4*4
    
    checkCudaErrors(cudaMalloc(&fc1_out, b_size * 120 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc1_sum, b_size * 120 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc2_out, b_size * 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc2_sum, b_size * 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc3_out, b_size * 10 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc3_sum, b_size * 10 * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&final_data, test_images_num * 10 * sizeof(float)));

    float *d_mem_A_2, *d_mem_B_2;
    float *conv1_sum_out_2, *conv1_ReLU_out_2, *Maxpool1_out_2 , *conv2_sum_out_2, *conv2_ReLU_out_2, *Maxpool2_out_2;
    float *fc1_sum_2, *fc1_out_2, *fc2_sum_2, *fc2_out_2, *fc3_sum_2, *fc3_out_2;
    //2520KB
    checkCudaErrors(cudaMalloc(&conv1_sum_out_2, conv1_channel_num * b_size * 24*24 * sizeof(float))); //6*64*24*24
    checkCudaErrors(cudaMalloc(&conv1_ReLU_out_2, conv1_channel_num * b_size * 24*24 * sizeof(float))); //6*64*24*24
    checkCudaErrors(cudaMalloc(&Maxpool1_out_2, conv1_channel_num * b_size * 12*12* sizeof(float))); //6*64*12*12
    checkCudaErrors(cudaMalloc(&conv2_sum_out_2,conv2_channel_num * b_size * 8*8 * sizeof(float)));//16*64*8*8
    checkCudaErrors(cudaMalloc(&conv2_ReLU_out_2,conv2_channel_num * b_size * 8*8 * sizeof(float)));//16*64*8*8
    checkCudaErrors(cudaMalloc(&Maxpool2_out_2, conv2_channel_num * b_size * 4*4 * sizeof(float)));//16*64*4*4
    
    checkCudaErrors(cudaMalloc(&fc1_out_2, b_size * 120 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc1_sum_2, b_size * 120 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc2_out_2, b_size * 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc2_sum_2, b_size * 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc3_out_2, b_size * 10 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fc3_sum_2, b_size * 10 * sizeof(float)));


    //checkCudaErrors(cudaMallocHost(&mem_A, (b_size * col * row) * sizeof(float)));
    //checkCudaErrors(cudaMallocHost(&mem_B, (b_size * col * row) * sizeof(float)));
    //392KB

    checkCudaErrors(cudaHostRegister(images.data(),images_byte,cudaHostRegisterDefault));
    checkCudaErrors(cudaMalloc(&d_mem_A,(b_size * col * row) * sizeof(float) ));//64*28*28
    checkCudaErrors(cudaMalloc(&d_mem_B,(b_size * col * row) * sizeof(float) ));//64*28*28
    //checkCudaErrors(cudaMemcpy(d_mem_A,mem_A,(b_size * col * row) * sizeof(float),cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_mem_B,mem_B,(b_size * col * row) * sizeof(float),cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpyToSymbol(conv1_core, conv1_w.data(), 150*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(conv1_bias, conv1_b.data(), 6*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(conv2_core, conv2_w.data(), 2400*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(conv2_bias, conv2_b.data(), 16*sizeof(float)));
    //checkCudaErrors(cudaMemcpyToSymbol(fc3_core, fc3_w.data(), 840 * sizeof(float)));
    //checkCudaErrors(cudaMemcpyToSymbol(fc3_bias, fc3_b.data(), 10 * sizeof(float)));
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    dim3 conv1_block(16,16,1), conv1_grid(2,2,64), Maxpool1_block(12,12,6), Maxpool1_grid(1,1,64);
    dim3 conv2_block(8,8,16), conv2_grid(1,1,64), Maxpool2_block(4,4,16), Maxpool2_grid(1,1,64);
    dim3 fc1_block(128,1,1), fc1_grid(1,1,64);
    dim3 fc2_block(96,2,1), fc2_grid(1,1,32);
    dim3 fc3_block(96,1,1), fc3_grid(1,1,8);

//    for(int images_group = 0; images_group < test_images_num / b_size; images_group ++){
//        const float* now_p = images_set(images, b_size, images_group);
//        if(images_group % 2)
//        {
//            checkCudaErrors(cudaMemcpyAsync(d_mem_A,now_p,(b_size * col * row) * sizeof(float),cudaMemcpyHostToDevice,stream1));
//
//            for(int Loop_times = 0; Loop_times < total_loop_times; Loop_times ++){
//
//                conv1_kernel<<<conv1_block,conv1_grid,20 * 20 * 4,stream1>>>(d_mem_A, conv1_sum_out, conv1_ReLU_out, Loop_times);
//                Maxpool1<<<Maxpool1_block,Maxpool1_grid,0,stream1>>>(conv1_ReLU_out, Maxpool1_out);
//
//                conv2_kernel<<<conv2_block,conv2_grid,6 * 12 * 12 * 4,stream1>>>(Maxpool1_out, conv2_sum_out, conv2_ReLU_out, Loop_times);
//                Maxpool2<<<Maxpool2_block,Maxpool2_grid,0,stream1>>>(conv2_ReLU_out, Maxpool2_out);
//
//                fc1<<<fc1_block,fc1_grid,0,stream1>>>(Maxpool2_out, d_fc1_w, d_fc1_b, fc1_sum, fc1_out, Loop_times);
//                fc2<<<fc2_block,fc2_grid,0,stream1>>>(fc1_out, d_fc2_w, d_fc2_b, fc2_sum, fc2_out, Loop_times);
//                fc3<<<fc3_block,fc3_grid,0,stream1>>>(fc2_out, d_fc3_w, d_fc3_b, fc3_sum, fc3_out, Loop_times);
//
//            }
//        }else{
//            checkCudaErrors(cudaMemcpyAsync(d_mem_B,now_p,(b_size * col * row) * sizeof(float),cudaMemcpyHostToDevice,stream2));
//
//            for(int Loop_times = 0; Loop_times < total_loop_times; Loop_times ++){
//
//                conv1_kernel<<<conv1_block,conv1_grid,20 * 20 * 4,stream2>>>(d_mem_B, conv1_sum_out_2, conv1_ReLU_out_2, Loop_times);
//                Maxpool1<<<Maxpool1_block,Maxpool1_grid,0,stream2>>>(conv1_ReLU_out_2, Maxpool1_out_2);
//
//                conv2_kernel<<<conv2_block,conv2_grid,6 * 12 * 12 * 4,stream2>>>(Maxpool1_out_2, conv2_sum_out_2, conv2_ReLU_out_2, Loop_times);
//                Maxpool2<<<Maxpool2_block,Maxpool2_grid,0,stream2>>>(conv2_ReLU_out_2, Maxpool2_out_2);
//
//                fc1<<<fc1_block,fc1_grid,0,stream2>>>(Maxpool2_out_2, d_fc1_w, d_fc1_b, fc1_sum_2, fc1_out_2, Loop_times);
//                fc2<<<fc2_block,fc2_grid,0,stream2>>>(fc1_out_2, d_fc2_w, d_fc2_b, fc2_sum_2, fc2_out_2, Loop_times);
//                fc3<<<fc3_block,fc3_grid,0,stream2>>>(fc2_out_2, d_fc3_w, d_fc3_b, fc3_sum_2, fc3_out_2, Loop_times);
//
//            }
//    
//        }
//}

for(int images_group = 0; images_group < test_images_num / b_size; images_group ++){
        const float* now_p = images_set(images, b_size, images_group);       
        checkCudaErrors(cudaMemcpyAsync(d_mem_A,now_p,(b_size * col * row) * sizeof(float),cudaMemcpyHostToDevice,stream1));
            for(int Loop_times = 0; Loop_times < total_loop_times; Loop_times ++){

                conv1_kernel<<<conv1_block,conv1_grid,20 * 20 * 4,stream1>>>(d_mem_A, conv1_sum_out, conv1_ReLU_out, Loop_times);
                Maxpool1<<<Maxpool1_block,Maxpool1_grid,0,stream1>>>(conv1_ReLU_out, Maxpool1_out);

                conv2_kernel<<<conv2_block,conv2_grid,6 * 12 * 12 * 4,stream1>>>(Maxpool1_out, conv2_sum_out, conv2_ReLU_out, Loop_times);
                Maxpool2<<<Maxpool2_block,Maxpool2_grid,0,stream1>>>(conv2_ReLU_out, Maxpool2_out);

                fc1<<<fc1_block,fc1_grid,0,stream1>>>(Maxpool2_out, d_fc1_w, d_fc1_b, fc1_sum, fc1_out, Loop_times);
                fc2<<<fc2_block,fc2_grid,0,stream1>>>(fc1_out, d_fc2_w, d_fc2_b, fc2_sum, fc2_out, Loop_times);
                fc3<<<fc3_block,fc3_grid,0,stream1>>>(fc2_out, d_fc3_w, d_fc3_b, fc3_sum, fc3_out, Loop_times);

            } 
        checkCudaErrors(cudaMemcpyAsync(labels_set(final_data, I64_perb, images_group), fc3_out, b_size * 10 * sizeof(float), cudaMemcpyDeviceToHost,stream1));
}


    

  
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    //checkCudaErrors(cudaFree(d_conv1_w));
    //checkCudaErrors(cudaFree(d_conv1_b));
    //checkCudaErrors(cudaFree(d_conv2_w));
    //checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    int predictions[10000] = {0};
    for(int i = 0; i < 10000; i++){
        float max_data = final_data[i * 10];
        int best_class = 0;
        for(int idx = 1; idx < 10; idx ++){
            float data = final_data[i * 10 + idx];
            if(max_data < data){
                max_data = data;
                best_class = idx;
            }
        }
        predictions[i] = best_class;
    }
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================