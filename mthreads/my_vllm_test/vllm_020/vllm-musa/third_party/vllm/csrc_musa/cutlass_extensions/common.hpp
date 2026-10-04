#pragma once

#include "mutlass/mutlass.h"
#include <climits>
#include "musa_runtime.h"
#include <cstdio>
#include <cstdlib>

#include <torch/headeronly/util/shim_utils.h>

/**
 * Helper function for checking MUTLASS errors
 */
#define MUTLASS_CHECK(status)                           \
  {                                                     \
    mutlass::Status error = status;                     \
    STD_TORCH_CHECK(error == mutlass::Status::kSuccess, \
                    mutlassGetStatusString(error));     \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  musaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         musaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

/**
 * A wrapper for a kernel that is used to guard against compilation on
 * architectures that will never use the kernel. The purpose of this is to
 * reduce the size of the compiled binary.
 * __MUSA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
 * into code that will be executed on the device where it is defined.
 */

template <typename Kernel>
struct enable_sm75_to_sm80 : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ >= 750 && __MUSA_ARCH__ < 800
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[75, 80).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ >= 220 && __MUSA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[80, 89).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ >= 890 && __MUSA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[89, 90).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm >= 90.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm90_only : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ == 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm90.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm100f_only : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ == 1000 || __MUSA_ARCH__ == 1030
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm100f.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm100a_only : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ == 1000
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm100a.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm120_only : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__
  #if __MUSA_ARCH__ == 1200
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm120.\n");
    asm("trap;");
  #endif
#endif
  }
};

// SM12x family includes SM120 (RTX 5090) and SM121 (DGX Spark GB10)
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  MUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __MUSA_ARCH__ && (__MUSA_ARCH__ >= 1200 && __MUSA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
