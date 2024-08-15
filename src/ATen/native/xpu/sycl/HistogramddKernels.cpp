#include <algorithm>
#include <cstdint>
#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/linspace.h>
#endif

namespace at::native::xpu {

template <typename scalar_t>
struct HistogramddKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_global_id();
    if (wi_id >= input_size_) {
      return;
    }
    int64_t ele_idx = wi_id;
    int64_t hist_idx = 0;
    for (int dim = 0; dim < input_dim_; ++dim) {
      auto elem = input_data[ele_idx][dim];
      const scalar_t* bin_edges = bin_edges_list_[dim];
      auto bin_edges_size = num_bin_edges_[dim];
      if (!(elem >= bin_edges[0] && elem <= bin_edges[bin_edges_size - 1])) {
        return;
      }
      auto bin_idx =
          std::upper_bound(bin_edges, bin_edges + bin_edges_size, elem) -
          bin_edges - 1;
      // Unlike other bins, the rightmost bin includes its right boundary
      if (bin_idx == (bin_edges_size - 1)) {
        bin_idx -= 1;
      }
      hist_idx += bin_idx * hist_strides_[dim];
    }

    scalar_t value = (scalar_t)1;
    if (use_weight_) {
      auto weight_data = weight_;
      value = weight_data[ele_idx];
    }
    atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), value);
  }

  HistogramddKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* const* bin_edges_list,
      scalar_t* hist,
      const int64_t* hist_strides,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      bool use_weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges)
      : input_(input),
        bin_edges_list_(bin_edges_list),
        hist_(hist),
        hist_strides_(hist_strides),
        weight_(weight),
        use_weight_(use_weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* const* bin_edges_list_;
  scalar_t* hist_;
  const int64_t* hist_strides_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  bool use_weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
};

template <typename scalar_t>
void histogramdd_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* const* bin_edges_list,
    scalar_t* hist,
    const int64_t* hist_strides,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    bool use_weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges) {
  HistogramddKernelFunctor<scalar_t> kfn(
      input,
      bin_edges_list,
      hist,
      hist_strides,
      weight,
      use_weight,
      input_size,
      input_dim,
      num_bin_edges);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg = (input_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct HistogramddLinearKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_global_id();
    if (wi_id >= input_size_) {
      return;
    }
    int64_t ele_idx = wi_id;
    int64_t hist_idx = 0;
    for (int dim = 0; dim < input_dim_; ++dim) {
      auto i_value = input_data[ele_idx][dim];
      auto leftmost_edge = leftmost_edges_[dim];
      auto rightmost_edge = rightmost_edges_[dim];
      auto bin_size = num_bin_edges_[dim] - 1;
      if (!(i_value >= leftmost_edge && i_value <= rightmost_edge)) {
        return;
      }
      int64_t bin_idx =
          (int64_t)(((i_value - leftmost_edge)) * bin_size / (rightmost_edge - leftmost_edge));
      if (bin_idx == bin_size) {
        bin_idx -= 1;
      }
      hist_idx += bin_idx * hist_strides_[dim];
    }

    scalar_t value = (scalar_t)1;
    if (use_weight_) {
      auto weight_data = weight_;
      value = weight_data[wi_id];
    }
    atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), value);
  }

  HistogramddLinearKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* const* bin_edges_list,
      scalar_t* hist,
      const int64_t* hist_strides,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      bool use_weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges,
      const scalar_t* leftmost_edges,
      const scalar_t* rightmost_edges)
      : input_(input),
        bin_edges_list_(bin_edges_list),
        hist_(hist),
        hist_strides_(hist_strides),
        weight_(weight),
        use_weight_(use_weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges),
        leftmost_edges_(leftmost_edges),
        rightmost_edges_(rightmost_edges) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* const* bin_edges_list_;
  scalar_t* hist_;
  const int64_t* hist_strides_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  bool use_weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
  const scalar_t* leftmost_edges_;
  const scalar_t* rightmost_edges_;
};

template <typename scalar_t>
void histogramdd_linear_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* const* bin_edges_list,
    scalar_t* hist,
    const int64_t* hist_strides,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    bool use_weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges,
    const scalar_t* leftmost_edges,
    const scalar_t* rightmost_edges) {
  HistogramddLinearKernelFunctor<scalar_t> kfn(
      input,
      bin_edges_list,
      hist,
      hist_strides,
      weight,
      use_weight,
      input_size,
      input_dim,
      num_bin_edges,
      leftmost_edges,
      rightmost_edges);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg = (input_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges) {
  globalContext().alertNotDeterministic("histogramdd_kernel_xpu");
  hist.fill_(0);

  const int64_t M = self.size(0);
  const int64_t N = self.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_xpu", [&]() {
        const auto self_accessor = self.packed_accessor64<const scalar_t, 2>();
        const auto weight_accessor = weight.has_value()
            ? weight.value().packed_accessor64<const scalar_t, 1>()
            : PackedTensorAccessor64<const scalar_t, 1>(
                  nullptr, self.sizes().data(), self.strides().data());

        Tensor hist_strides_xpu = at::tensor(
            hist.strides(),
            self.options()
                .dtype(c10::kLong)
                .memory_format(at::MemoryFormat::Contiguous));
        Tensor bin_edges_contig_ptr = at::empty(
            {N},
            self.options()
                .dtype(c10::kUInt64)
                .device(at::kCPU)
                .memory_format(at::MemoryFormat::Contiguous));
        Tensor num_bin_edges = at::empty(
            {N},
            self.options()
                .dtype(c10::kLong)
                .device(at::kCPU)
                .memory_format(at::MemoryFormat::Contiguous));

        auto bin_edges_contig_ptr_accessor =
            bin_edges_contig_ptr.packed_accessor64<uint64_t, 1>();
        auto num_bin_edges_accessor =
            num_bin_edges.packed_accessor64<int64_t, 1>();

        for (const auto dim : c10::irange(N)) {
          const scalar_t* data_ptr = bin_edges[dim].const_data_ptr<scalar_t>();
          bin_edges_contig_ptr_accessor[dim] =
              reinterpret_cast<uint64_t>(data_ptr);
          num_bin_edges_accessor[dim] = bin_edges[dim].numel();
        }

        const Tensor bin_edges_contig_ptr_xpu =
            bin_edges_contig_ptr.to(self.device());
        const Tensor num_bin_edges_xpu = num_bin_edges.to(self.device());

        histogramdd_template<scalar_t>(
            self_accessor,
            reinterpret_cast<const scalar_t* const*>(
                bin_edges_contig_ptr_xpu.const_data_ptr()),
            hist.data_ptr<scalar_t>(),
            hist_strides_xpu.const_data_ptr<int64_t>(),
            weight_accessor,
            weight.has_value(),
            M,
            N,
            num_bin_edges_xpu.const_data_ptr<int64_t>());
      });

  /* Divides each bin's value by the total count/weight in all bins,
   * and by the bin's volume.
   */
  if (density) {
    const auto hist_sum = hist.sum().item();
    hist.div_(hist_sum);

    /* For each dimension, divides each bin's value
     * by the bin's length in that dimension.
     */
    for (const auto dim : c10::irange(N)) {
      const auto bin_lengths = bin_edges[dim].diff();

      // Used to reshape bin_lengths to align with the corresponding dimension
      // of hist.
      std::vector<int64_t> shape(N, 1);
      shape[dim] = bin_lengths.numel();

      hist.div_(bin_lengths.reshape(shape));
    }
  }
}

void histogramdd_linear_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges,
    const std::pair<std::vector<double>, std::vector<double>>& outer_bin_edges,
    bool local_search) {
  globalContext().alertNotDeterministic("histogramdd_linear_kernel_xpu");
  hist.fill_(0);

  TORCH_INTERNAL_ASSERT(self.dim() == 2);

  const int64_t M = self.size(0);
  const int64_t N = self.size(1);
  if (weight.has_value()) {
    TORCH_INTERNAL_ASSERT(
        weight.value().dim() == 1 && weight.value().numel() == M);
  }

  const int64_t D = self.size(1);
  TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
  for (const auto dim : c10::irange(D)) {
    TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
    TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
  }

  if (D == 0) {
    // hist is an empty tensor in this case; nothing to do here
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_linear_xpu", [&]() {
        const auto self_accessor = self.packed_accessor64<const scalar_t, 2>();
        const auto weight_accessor = weight.has_value()
            ? weight.value().packed_accessor64<const scalar_t, 1>()
            : PackedTensorAccessor64<const scalar_t, 1>(
                  nullptr, self.sizes().data(), self.strides().data());

        Tensor hist_strides_xpu = at::tensor(
            hist.strides(),
            self.options()
                .dtype(c10::kLong)
                .memory_format(at::MemoryFormat::Contiguous));
        Tensor leftmost_edges_xpu = at::tensor(
            outer_bin_edges.first,
            self.options().memory_format(at::MemoryFormat::Contiguous));
        Tensor rightmost_edges_xpu = at::tensor(
            outer_bin_edges.second,
            self.options().memory_format(at::MemoryFormat::Contiguous));

        Tensor bin_edges_contig_ptr = at::empty(
            {N},
            self.options()
                .dtype(c10::kUInt64)
                .device(at::kCPU)
                .memory_format(at::MemoryFormat::Contiguous));
        Tensor num_bin_edges = at::empty(
            {D},
            self.options()
                .dtype(c10::kLong)
                .device(at::kCPU)
                .memory_format(at::MemoryFormat::Contiguous));
        auto bin_edges_contig_ptr_accessor =
            bin_edges_contig_ptr.packed_accessor64<uint64_t, 1>();
        auto num_bin_edges_accessor =
            num_bin_edges.packed_accessor64<int64_t, 1>();
        for (const auto dim : c10::irange(D)) {
          const scalar_t* data_ptr = bin_edges[dim].const_data_ptr<scalar_t>();
          bin_edges_contig_ptr_accessor[dim] =
              reinterpret_cast<uint64_t>(data_ptr);
          num_bin_edges_accessor[dim] = bin_edges[dim].numel();
        }
        const Tensor bin_edges_contig_ptr_xpu =
            bin_edges_contig_ptr.to(self.device());
        const Tensor num_bin_edges_xpu = num_bin_edges.to(self.device());

        histogramdd_linear_template<scalar_t>(
            self_accessor,
            reinterpret_cast<const scalar_t* const*>(
                bin_edges_contig_ptr_xpu.const_data_ptr()),
            hist.data_ptr<scalar_t>(),
            hist_strides_xpu.const_data_ptr<int64_t>(),
            weight_accessor,
            weight.has_value(),
            M,
            N,
            num_bin_edges_xpu.const_data_ptr<int64_t>(),
            leftmost_edges_xpu.const_data_ptr<scalar_t>(),
            rightmost_edges_xpu.const_data_ptr<scalar_t>());
      });

  /* Divides each bin's value by the total count/weight in all bins,
   * and by the bin's volume.
   */
  if (density) {
    const auto hist_sum = hist.sum().item();
    hist.div_(hist_sum);

    /* For each dimension, divides each bin's value
     * by the bin's length in that dimension.
     */
    for (const auto dim : c10::irange(N)) {
      const auto bin_lengths = bin_edges[dim].diff();

      // Used to reshape bin_lengths to align with the corresponding dimension
      // of hist.
      std::vector<int64_t> shape(N, 1);
      shape[dim] = bin_lengths.numel();

      hist.div_(bin_lengths.reshape(shape));
    }
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
