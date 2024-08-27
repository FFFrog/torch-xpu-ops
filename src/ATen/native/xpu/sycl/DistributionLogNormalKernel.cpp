#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void log_normal_kernel(
    TensorIteratorBase& iter,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::log_normal_kernel(iter, mean, std, generator);
}

} // namespace at::native::xpu
