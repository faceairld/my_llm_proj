#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_int8_dispatch.muh"
#include "mutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

void mutlass_scaled_mm_sm90_int8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    STD_TORCH_CHECK(bias->scalar_type() == out.scalar_type(),
                    "currently bias dtype must match output dtype ",
                    out.scalar_type());
    return mutlass_scaled_mm_sm90_int8_epilogue<c3x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return mutlass_scaled_mm_sm90_int8_epilogue<c3x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
