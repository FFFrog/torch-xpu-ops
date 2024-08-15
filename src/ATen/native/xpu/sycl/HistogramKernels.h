#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges);

void histogramdd_linear_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges,
    const std::pair<std::vector<double>, std::vector<double>>& outer_bin_edges,
    bool local_search);

void histogramdd_infer_bin_edges_from_input_kernel(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges);

} // namespace at::native::xpu