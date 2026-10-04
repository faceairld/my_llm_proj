#include "mutlass_extensions/common.hpp"

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  musaDeviceGetAttribute(&major_capability, musaDevAttrComputeCapabilityMajor,
                         0);
  musaDeviceGetAttribute(&minor_capability, musaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}