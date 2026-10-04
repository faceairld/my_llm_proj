// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// bf16 x bf16 -> fp32 router GEMM via cuBLAS.
// Uses CUBLAS_COMPUTE_32F so bf16 operands accumulate into fp32,
// matching TRT-LLM's cuBLAS fallback behaviour in dsv3RouterGemmOp.

#include <torch/all.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <mublas.h>

// cuBLAS column-major math for row-major PyTorch tensors:
//   weight[N,K]_row  lda=K  -> cuBLAS sees (K,N) col-major; CUBLAS_OP_T ->
//   (N,K) input[M,K]_row   ldb=K  -> cuBLAS sees (K,M) col-major; CUBLAS_OP_N
//   -> (K,M) out[M,N]_row     ldc=N  -> cuBLAS sees (N,M) col-major (written as
//   output^T)
// cuBLAS: C(N,M) = weight(N,K) @ input(K,M)  =>  C^T = output[M,N]
// params: m=N, n=M, k=K, lda=K (weight), ldb=K (input), ldc=N (output)

torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: input must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: weight must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "router_gemm_bf16_fp32: input and weight must be 2-D");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "router_gemm_bf16_fp32: inner dimensions must match");

  int64_t const M = input.size(0);
  int64_t const N = weight.size(0);
  int64_t const K = input.size(1);

  auto out = torch::empty({M, N}, input.options().dtype(torch::kFloat32));

  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  TORCH_MUSABLAS_CHECK(
      mublasSetStream(handle, at::musa::getCurrentMUSAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;

  TORCH_MUSABLAS_CHECK(mublasGemmEx(
      handle, MUBLAS_OP_T, MUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, weight.data_ptr(),
      MUSA_R_16BF, static_cast<int>(K), input.data_ptr(), MUSA_R_16BF,
      static_cast<int>(K), &beta, out.data_ptr(), MUSA_R_32F,
      static_cast<int>(N), MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT));

  return out;
}
