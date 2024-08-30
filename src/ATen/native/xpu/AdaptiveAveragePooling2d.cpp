#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <aten/src/ATen/ops/mean_ops.h>
#endif

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

namespace at {

Tensor XPUNativeFunctions::_adaptive_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  TensorArg grad_output_arg{grad_output, "grad_output", 1},
      input_arg{input, "input", 2};

  native::adaptive_pool_empty_output_check(
      grad_output, "adaptive_avg_pool2d_backward");

  checkAllSameGPU(__func__, {grad_output_arg, input_arg});

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  globalContext().alertNotDeterministic("_adaptive_avg_pool2d_backward");

  Tensor grad_input;
  if (input.numel() != 0) {
    native::xpu::adaptive_avg_pool2d_backward_kernel(
        grad_input, grad_output, input);
  } else {
    grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return grad_input;
}

Tensor& XPUNativeFunctions::adaptive_avg_pool2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = input.dim();
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  for (const auto i : {-2, -1}) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i + ndim,
        " being "
        "empty");
  }

  native::xpu::adaptive_avg_pool2d_kernel(output, input, output_size);
  return output;
}

Tensor XPUNativeFunctions::_adaptive_avg_pool2d(
    at::Tensor const& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out(input, output_size, output);
  return output;
}

} // namespace at
