#include <cstdint>
#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
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
    int64_t wi_id = item_id.get_global_id();
    if (wi_id >= input_size_) {
      return;
    }
    int64_t ele_idx = wi_id;
    int64_t hist_idx = 0;
    int64_t hist_stride = 1;
    bool skip_elt = false;
    scalar_t weight_value = weight_ ? weight_[ele_idx] : (scalar_t)1;
    for (int dim = 0; !skip_elt && dim < input_dim_; ++dim) {
      auto elem = input_[ele_idx * input_dim_ + dim];
      const scalar_t* bin_edges = bin_edges_list_[dim];
      auto bin_edges_size = num_bin_edges_[dim];
      bool skip_elt_dim = true;
      for (auto bin_idx = 0; bin_idx < bin_edges_size - 1; ++bin_idx) {
        // [left, right) execept for last bin: [left, right]
        if ((bin_idx == (bin_edges_size - 2) &&
             elem == bin_edges[bin_idx + 1]) ||
            (elem >= bin_edges[bin_idx] && elem < bin_edges[bin_idx + 1])) {
          hist_idx = (hist_idx * hist_stride) + bin_idx;
          skip_elt_dim = false;
          break;
        }
      }
      hist_stride = num_bin_edges_[dim] - 1;
      skip_elt |= skip_elt_dim;
    }
    if (!skip_elt) {
      atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), weight_value);
    }
  }

  HistogramddKernelFunctor(
      const scalar_t* input,
      const scalar_t* const* bin_edges_list,
      scalar_t* hist,
      const scalar_t* weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges)
      : input_(input),
        bin_edges_list_(bin_edges_list),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges) {}

 private:
  const scalar_t* input_;
  const scalar_t* const* bin_edges_list_;
  scalar_t* hist_;
  const scalar_t* weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
};

template <typename scalar_t>
void histogramdd_template(
    const scalar_t* input,
    const scalar_t* const* bin_edges_list,
    scalar_t* hist,
    const scalar_t* weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges) {
  HistogramddKernelFunctor<scalar_t> kfn(
      input,
      bin_edges_list,
      hist,
      weight,
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
    int64_t wi_id = item_id.get_global_id();
    if (wi_id >= input_size_) {
      return;
    }
    int64_t ele_idx = wi_id;
    int64_t hist_idx = 0;
    int64_t hist_stride = 1;
    bool skip_elt = false;
    scalar_t weight_value = weight_ ? weight_[ele_idx] : (scalar_t)1;
    for (int dim = 0; !skip_elt && dim < input_dim_; ++dim) {
      auto i_value = input_[ele_idx * input_dim_ + dim];
      auto leftmost_edge = leftmost_edges_[dim];
      auto rightmost_edge = rightmost_edges_[dim];
      auto bin_size = num_bin_edges_[dim] - 1;
      if (i_value >= leftmost_edge && i_value <= rightmost_edge) {
        int64_t bin_idx =
            (int64_t)(((i_value - leftmost_edge)) * bin_size / (rightmost_edge - leftmost_edge));
        if (bin_idx == bin_size) {
          bin_idx -= 1;
        }
        hist_idx = (hist_idx * hist_stride) + bin_idx;
      } else {
        skip_elt = true;
        break;
      }
      hist_stride = bin_size;
    }
    if (!skip_elt) {
      atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), weight_value);
    }
  }

  HistogramddLinearKernelFunctor(
      const scalar_t* input,
      scalar_t* hist,
      const scalar_t* weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges,
      const scalar_t* leftmost_edges,
      const scalar_t* rightmost_edges)
      : input_(input),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges),
        leftmost_edges_(leftmost_edges),
        rightmost_edges_(rightmost_edges) {}

 private:
  const scalar_t* input_;
  scalar_t* hist_;
  const scalar_t* weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
  const scalar_t* leftmost_edges_;
  const scalar_t* rightmost_edges_;
};

template <typename scalar_t>
void histogramdd_linear_template(
    const scalar_t* input,
    scalar_t* hist,
    const scalar_t* weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges,
    const scalar_t* leftmost_edges,
    const scalar_t* rightmost_edges) {
  HistogramddLinearKernelFunctor<scalar_t> kfn(
      input,
      hist,
      weight,
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
        Tensor bin_edges_contig_ptr =
            at::empty({N}, self.options().dtype(c10::kUInt64).device(at::kCPU));
        Tensor num_bin_edges = at::empty(
            bin_edges_contig_ptr.sizes(),
            self.options().dtype(c10::kLong).device(at::kCPU));

        auto bin_edges_contig_ptr_accessor =
            bin_edges_contig_ptr.generic_packed_accessor<uint64_t, 1>();
        auto num_bin_edges_accessor =
            num_bin_edges.generic_packed_accessor<int64_t, 1>();

        for (const auto dim : c10::irange(N)) {
          const scalar_t* data_ptr = bin_edges[dim].const_data_ptr<scalar_t>();
          bin_edges_contig_ptr_accessor[dim] =
              reinterpret_cast<uint64_t>(data_ptr);
          num_bin_edges_accessor[dim] = bin_edges[dim].numel();
        }

        const Tensor bin_edges_contig_ptr_xpu =
            bin_edges_contig_ptr.to(self.device()).contiguous();
        const Tensor num_bin_edges_xpu =
            num_bin_edges.to(self.device()).contiguous();

        histogramdd_template<scalar_t>(
            self.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t* const*>(
                bin_edges_contig_ptr_xpu.const_data_ptr()),
            hist.data_ptr<scalar_t>(),
            weight.has_value() ? weight->data_ptr<scalar_t>() : nullptr,
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
    const std::pair<std::vector<double>, std::vector<double>>&
        outer_bin_edges) {
  // histogramdd_kernel(self, weight, density, hist, bin_edges);
  // return;
  globalContext().alertNotDeterministic("histogramdd_linear_kernel_xpu");
  hist.fill_(0);

  const int64_t M = self.size(0);
  const int64_t N = self.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_linear_xpu", [&]() {
        const int64_t D = self.size(1);
        if (D == 0) {
          // hist is an empty tensor in this case; nothing to do here
          return;
        }

        Tensor num_bin_edges =
            at::empty({D}, self.options().dtype(c10::kLong).device(at::kCPU));
        Tensor leftmost_edges = at::empty({D}, self.options().device(at::kCPU));
        Tensor rightmost_edges =
            at::empty({D}, self.options().device(at::kCPU));

        auto num_bin_edges_accessor =
            num_bin_edges.generic_packed_accessor<int64_t, 1>();
        auto leftmost_edges_accessor =
            leftmost_edges.generic_packed_accessor<scalar_t, 1>();
        auto rightmost_edges_accessor =
            rightmost_edges.generic_packed_accessor<scalar_t, 1>();

        for (const auto dim : c10::irange(D)) {
          num_bin_edges_accessor[dim] = bin_edges[dim].numel();
          leftmost_edges_accessor[dim] = outer_bin_edges.first[dim];
          rightmost_edges_accessor[dim] = outer_bin_edges.second[dim];
        }

        const Tensor num_bin_edges_xpu =
            num_bin_edges.to(self.device()).contiguous();
        const Tensor leftmost_edges_xpu =
            leftmost_edges.to(self.device()).contiguous();
        const Tensor rightmost_edges_xpu =
            rightmost_edges.to(self.device()).contiguous();

        histogramdd_linear_template<scalar_t>(
            self.data_ptr<scalar_t>(),
            hist.data_ptr<scalar_t>(),
            weight.has_value() ? weight->data_ptr<scalar_t>() : nullptr,
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

template <typename scalar_t>
void histogramdd_infer_bin_edges_from_input_kernel_template(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  // Calls aminmax on input with dim=0, reducing all but the innermost dimension
  // of input.
  auto [min, max] = at::aminmax(input, 0);

  Tensor min_cpu = min.cpu();
  Tensor max_cpu = max.cpu();

  TORCH_INTERNAL_ASSERT(min_cpu.is_contiguous() && max_cpu.is_contiguous());

  const scalar_t* min_data = min_cpu.const_data_ptr<scalar_t>();
  std::copy(min_data, min_data + N, leftmost_edges.begin());

  const scalar_t* max_data = max_cpu.const_data_ptr<scalar_t>();
  std::copy(max_data, max_data + N, rightmost_edges.begin());
}

void histogramdd_infer_bin_edges_from_input_kernel(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "histogramdd_infer_bin_edges_from_input_xpu", [&]() {
        histogramdd_infer_bin_edges_from_input_kernel_template<scalar_t>(
            input, N, leftmost_edges, rightmost_edges);
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop