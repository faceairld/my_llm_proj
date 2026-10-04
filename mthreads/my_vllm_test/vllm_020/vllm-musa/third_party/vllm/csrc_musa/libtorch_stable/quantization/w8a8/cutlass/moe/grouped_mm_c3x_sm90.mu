#include <musaTypedefs.h>

#include "libtorch_stable/torch_utils.h"
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "mutlass/mutlass.h"
#include "grouped_mm_c3x.muh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (16, inf)
  static_assert(std::is_same<InType, mutlass::float_e4m3_t>());
  using KernelSchedule =
      mutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      mutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using ArchTag = mutlass::arch::Sm90;

  using Cutlass3xGemm =
      mutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M4 {
  // M in [1, 4]
  static_assert(std::is_same<InType, mutlass::float_e4m3_t>());
  using KernelSchedule =
      mutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      mutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = mutlass::arch::Sm90;

  using Cutlass3xGemm =
      mutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M64 {
  // M in (4, 64]
  static_assert(std::is_same<InType, mutlass::float_e4m3_t>());
  using KernelSchedule =
      mutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      mutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_256>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using ArchTag = mutlass::arch::Sm90;

  using Cutlass3xGemm =
      mutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_K8192 {
  // K in [8192, inf)
  static_assert(std::is_same<InType, mutlass::float_e4m3_t>());
  using KernelSchedule =
      mutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      mutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;
  using ArchTag = mutlass::arch::Sm90;

  using Cutlass3xGemm =
      mutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_N8192 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, mutlass::float_e4m3_t>());
  using KernelSchedule =
      mutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      mutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;
  using ArchTag = mutlass::arch::Sm90;

  using Cutlass3xGemm =
      mutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_mutlass_moe_mm_sm90(torch::stable::Tensor& out_tensors,
                             torch::stable::Tensor const& a_tensors,
                             torch::stable::Tensor const& b_tensors,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             torch::stable::Tensor const& expert_offsets,
                             torch::stable::Tensor const& problem_sizes,
                             torch::stable::Tensor const& a_strides,
                             torch::stable::Tensor const& b_strides,
                             torch::stable::Tensor const& c_strides,
                             bool per_act_token, bool per_out_ch) {
  STD_TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  STD_TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  STD_TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  STD_TORCH_CHECK(
      a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "A tensors must be of type float8_e4m3fn.");
  STD_TORCH_CHECK(
      b_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "B tensors must be of type float8_e4m3fn.");

  using Cutlass3xGemmN8192 = typename sm90_fp8_config_N8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmK8192 = typename sm90_fp8_config_K8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM4 = typename sm90_fp8_config_M4<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM64 = typename sm90_fp8_config_M64<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  // Use swap_ab for M <= 64 by default to reduce padding
  if (m <= 4) {
    mutlass_group_gemm_caller<Cutlass3xGemmM4>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (m <= 64) {
    mutlass_group_gemm_caller<Cutlass3xGemmM64>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (n >= 8192) {
    mutlass_group_gemm_caller<Cutlass3xGemmN8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (k >= 8192) {
    mutlass_group_gemm_caller<Cutlass3xGemmK8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    mutlass_group_gemm_caller<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

void dispatch_moe_mm_sm90(torch::stable::Tensor& out_tensors,
                          torch::stable::Tensor const& a_tensors,
                          torch::stable::Tensor const& b_tensors,
                          torch::stable::Tensor const& a_scales,
                          torch::stable::Tensor const& b_scales,
                          torch::stable::Tensor const& expert_offsets,
                          torch::stable::Tensor const& problem_sizes,
                          torch::stable::Tensor const& a_strides,
                          torch::stable::Tensor const& b_strides,
                          torch::stable::Tensor const& c_strides,
                          bool per_act_token, bool per_out_ch) {
  if (out_tensors.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    run_mutlass_moe_mm_sm90<mutlass::float_e4m3_t, mutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    run_mutlass_moe_mm_sm90<mutlass::float_e4m3_t, mutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

}  // namespace

void mutlass_moe_mm_sm90(torch::stable::Tensor& out_tensors,
                         torch::stable::Tensor const& a_tensors,
                         torch::stable::Tensor const& b_tensors,
                         torch::stable::Tensor const& a_scales,
                         torch::stable::Tensor const& b_scales,
                         torch::stable::Tensor const& expert_offsets,
                         torch::stable::Tensor const& problem_sizes,
                         torch::stable::Tensor const& a_strides,
                         torch::stable::Tensor const& b_strides,
                         torch::stable::Tensor const& c_strides,
                         bool per_act_token, bool per_out_ch) {
  dispatch_moe_mm_sm90(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                       expert_offsets, problem_sizes, a_strides, b_strides,
                       c_strides, per_act_token, per_out_ch);
}
