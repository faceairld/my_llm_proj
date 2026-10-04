#include <musa_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include "mem_alloc.h"

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  musaError_t err = musaHostAlloc(&ptr, size, flags);
  if (err != musaSuccess) {
    throw std::runtime_error("musaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  musaError_t err = musaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != musaSuccess) {
    throw std::runtime_error("musaFreeHost failed: " + std::to_string(err));
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = reinterpret_cast<void*>(alloc_numa_ptr(size, node));

  musaError_t st = musaHostRegister(ptr, size, 0);
  if (st != musaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("musaHostRegister failed: ") +
                             musaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  musaError_t st = musaHostUnregister(p);
  if (st != musaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("musaHostUnregister failed: ") +
                             musaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open failed: ") +
                             strerror(errno));

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed: ") + strerror(err));
  }

  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  first_touch(ptr, size);

  musaError_t st = musaHostRegister(ptr, size, 0);
  if (st != musaSuccess) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("musaHostRegister failed: ") +
                             musaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name) {
  void* p = reinterpret_cast<void*>(ptr);
  musaError_t st = musaHostUnregister(p);
  if (st != musaSuccess) {
    munmap(p, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("musaHostUnregister failed: ") +
                             musaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  shm_unlink(shm_name.c_str());
}
