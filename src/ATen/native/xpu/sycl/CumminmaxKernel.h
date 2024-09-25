#pragma once

#include <ATen/core/TensorBase.h>

namespace at::native::xpu {

TORCH_XPU_API void launch_cummax_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim);

TORCH_XPU_API void launch_cummin_kernel(
    const TensorBase& self,
    const TensorBase& values,
    const TensorBase& indices,
    int64_t dim);

} // namespace at::native::xpu
