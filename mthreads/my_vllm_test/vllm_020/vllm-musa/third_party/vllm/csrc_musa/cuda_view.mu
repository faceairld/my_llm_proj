#include <torch/all.h>
#include <torch/musa.h>
#include <musa_runtime.h>

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::empty(cpu_tensor.sizes(),
                        cpu_tensor.options().device(torch::kPrivateUse1));
  }

  if (cpu_tensor.is_pinned()) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    void* device_ptr = nullptr;
    musaError_t err = musaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == musaSuccess,
                "musaHostGetDevicePointer failed: ", musaGetErrorString(err));

    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {},  // keep cpu tensor alive
        cpu_tensor.options().device(torch::kPrivateUse1));
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  size_t nbytes = contiguous_cpu.nbytes();

  void* host_ptr = nullptr;
  musaError_t err = musaHostAlloc(&host_ptr, nbytes, musaHostAllocMapped);
  if (err != musaSuccess) {
    AT_ERROR("musaHostAlloc failed: ", musaGetErrorString(err));
  }

  err = musaMemcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes,
                   musaMemcpyDefault);
  if (err != musaSuccess) {
    musaFreeHost(host_ptr);
    AT_ERROR("musaMemcpy failed: ", musaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = musaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != musaSuccess) {
    musaFreeHost(host_ptr);
    AT_ERROR("musaHostGetDevicePointer failed: ", musaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { musaFreeHost(host_ptr); };

  return torch::from_blob(device_ptr, contiguous_cpu.sizes(),
                          contiguous_cpu.strides(), deleter,
                          contiguous_cpu.options().device(torch::kPrivateUse1));
}