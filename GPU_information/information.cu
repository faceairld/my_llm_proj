#include <iostream>
#include <cuda_runtime.h>
#include <windows.h>

int main() {
    SetConsoleOutputCP(CP_UTF8);
    cudaDeviceProp prop;
    int deviceId = 0; // 默认查第 0 张显卡
    cudaGetDeviceProperties(&prop, deviceId);

    std::cout << "========== 终极硬件侦察报告 ==========\n";
    std::cout << "显卡型号: " << prop.name << "\n";
    std::cout << "计算能力 (Compute Capability): " << prop.major << "." << prop.minor << "\n";
    
    // 🌟 最核心的兵力分配：
    std::cout << "流多处理器数量 (SM Count): " << prop.multiProcessorCount << "\n";
    std::cout << "每个 SM 最大线程数: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "每个 Block 最大线程数: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "每个 Warp 的线程数: " << prop.warpSize << "\n";
    
    // 🌟 存储器情报：
    std::cout << "全局显存大小 (Global Memory): " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "每个 SM 共享内存上限 (Shared Mem per SM): " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "每个 Block 共享内存上限 (Shared Mem per Block): " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "每个 Block 寄存器上限 (Registers per Block): " << prop.regsPerBlock << "\n";
    std::cout << "每个 SM 最大 Block 数: " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "======================================\n";

    return 0;
}