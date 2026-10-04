#include <musa_runtime.h>
#include <stdexcept>
#include "utils.h"

std::string get_gpu_pci_bus_id(int device) {
  char pciBusId[13];  // 13 bytes per CUDA doc
  musaError_t err = musaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
  if (err != musaSuccess) {
    throw std::runtime_error(std::string("musaDeviceGetPCIBusId failed: ") +
                             musaGetErrorString(err));
  }
  return std::string(pciBusId);
}
