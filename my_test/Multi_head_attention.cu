#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
// Q, K, V, output are device pointers

__global__ void soft_max(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h, int dk){

                        int col = blockIdx.x * blockDim.x + threadIdx.x;
                        int row = blockIdx.y * blockDim.y + threadIdx.y;

                        if(col >= d_model || row >= N)
                          return;

                        int head_num = col / dk;
                        int head_inter_dim = col % dk;

                        float max = -FLT_MAX;
          
                        for(int i = 0; i < N; i ++){
                          float data = 0.0f;
                          for(int j = 0; j < dk; j++){
                          data += Q[row * d_model + head_num * dk + j]
                                    * K[i * d_model + head_num * dk + j];
                          }
                          max = fmaxf(max, data);
                        }

                        float sum = 0.0f;
                        for(int i = 0; i < N; i ++){
                          float data = 0.0f;
                          for(int j = 0; j < dk; j++){
                          data += Q[row * d_model + head_num * dk + j] 
                                    * K[i * d_model + head_num * dk + j];
                          }
                          sum += expf((data - max)/sqrtf(dk));
                        }

                        float outdata = 0.0f;
                        
                        for(int i = 0; i < N; i ++){
                          float data = 0.0f;
                          float softmax_data = 0.0f;

                          for(int j = 0; j < dk; j++){
                          data += Q[row * d_model + head_num * dk + j] 
                                    * K[i * d_model + head_num * dk + j];
                          }
                          softmax_data = expf((data - max)/sqrtf(dk))/sum;
                          outdata += softmax_data * V[col + i * d_model];
                          
                        }
                          output[row * d_model + col] = outdata;
                      }




extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
                        
                        dim3 block(16,16,1);
                        dim3 grid(((d_model + 15) / 16), ((N + 15) / 16), 1);
                        float dk = d_model / h;
                        soft_max<<<grid, block>>>(Q, K, V, output, N, d_model, h, dk);

                      }