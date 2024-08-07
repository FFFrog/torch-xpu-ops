#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void addcmul_kernel(TensorIteratorBase& iter, const Scalar& value);

void addcdiv_kernel(TensorIteratorBase& iter, const Scalar& value);

void mse_backward_kernel(TensorIterator& iter, const Scalar& value);

void smooth_l1_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double beta);

void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta);

} // namespace at::native::xpu
