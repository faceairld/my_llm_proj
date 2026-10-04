#pragma once

#ifdef USE_ROCM
////////////////////////////////////////
// For compatibility with CUDA and ROCm
////////////////////////////////////////
  #include <hip/hip_runtime_api.h>

extern "C" {
  #ifndef MUSA_SUCCESS
    #define MUSA_SUCCESS hipSuccess
  #endif  // MUSA_SUCCESS

// https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
typedef unsigned long long MUdevice;
typedef hipDeviceptr_t MUdeviceptr;
typedef hipError_t MUresult;
typedef hipCtx_t MUcontext;
typedef hipStream_t MUstream;
typedef hipMemGenericAllocationHandle_t MUmemGenericAllocationHandle;
typedef hipMemAllocationGranularity_flags CUmemAllocationGranularity_flags;
typedef hipMemAllocationProp MUmemAllocationProp;
typedef hipMemAccessDesc MUmemAccessDesc;

  #define MU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
  #define MU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
  #define MU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
  #define MU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
  #define MU_MEM_ALLOCATION_COMP_NONE 0x0

// Error Handling
// https://docs.nvidia.com/cuda/archive/11.4.4/cuda-driver-api/group__CUDA__ERROR.html
MUresult muGetErrorString(MUresult hipError, const char** pStr) {
  *pStr = hipGetErrorString(hipError);
  return MUSA_SUCCESS;
}

// Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
MUresult muCtxGetCurrent(MUcontext* ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxGetCurrent(ctx);
}

MUresult muCtxSetCurrent(MUcontext ctx) {
  // This API is deprecated on the AMD platform, only for equivalent cuCtx
  // driver API on the NVIDIA platform.
  return hipCtxSetCurrent(ctx);
}

// Primary Context Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html
MUresult muDevicePrimaryCtxRetain(MUcontext* ctx, MUdevice dev) {
  return hipDevicePrimaryCtxRetain(ctx, dev);
}

// Virtual Memory Management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
MUresult muMemAddressFree(MUdeviceptr ptr, size_t size) {
  return hipMemAddressFree(ptr, size);
}

MUresult muMemAddressReserve(MUdeviceptr* ptr, size_t size, size_t alignment,
                             MUdeviceptr addr, unsigned long long flags) {
  return hipMemAddressReserve(ptr, size, alignment, addr, flags);
}

MUresult muMemCreate(MUmemGenericAllocationHandle* handle, size_t size,
                     const MUmemAllocationProp* prop,
                     unsigned long long flags) {
  return hipMemCreate(handle, size, prop, flags);
}

MUresult muMemGetAllocationGranularity(
    size_t* granularity, const MUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  return hipMemGetAllocationGranularity(granularity, prop, option);
}

MUresult muMemMap(MUdeviceptr dptr, size_t size, size_t offset,
                  MUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  return hipMemMap(dptr, size, offset, handle, flags);
}

MUresult muMemRelease(MUmemGenericAllocationHandle handle) {
  return hipMemRelease(handle);
}

MUresult muMemSetAccess(MUdeviceptr ptr, size_t size,
                        const MUmemAccessDesc* desc, size_t count) {
  return hipMemSetAccess(ptr, size, desc, count);
}

MUresult muMemUnmap(MUdeviceptr ptr, size_t size) {
  return hipMemUnmap(ptr, size);
}
}  // extern "C"

#else
////////////////////////////////////////
// Import CUDA headers for NVIDIA GPUs
////////////////////////////////////////
  #include <musa_runtime_api.h>
  #include <musa.h>
#endif
