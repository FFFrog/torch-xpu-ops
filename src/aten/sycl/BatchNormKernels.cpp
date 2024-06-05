#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUContext.h>
#include <aten/Resize.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/Reduce.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

#define SIMD32 32
#define SIMD16 16

// ========================== batch_norm utils ==========================

ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

// A transform is mixed type if the parameters are higher precision than the
// input
template <typename... Args>
bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return (
      (parameter_type != ScalarType::Undefined) &&
      (parameter_type != input.scalar_type()));
}

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return (
      self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      self.is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
      (self.is_contiguous() && self.strides()[1] == 1));
}

enum class Impl {
  Contiguous,
  ChannelsLast,
  General,
};

inline Impl batch_norm_choose_impl(const Tensor& self) {
  if (!canUse32BitIndexMath(self)) {
    return Impl::General;
  }

  if (self.is_contiguous()) {
    return self.strides()[1] == 1 ? Impl::ChannelsLast : Impl::Contiguous;
  }

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Impl::ChannelsLast;
  }

  return Impl::General;
}

template <
    typename scalar_t,
    int64_t dim,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>
get_packed_accessor(const Tensor& t, c10::string_view var_name) {
  constexpr auto expect_type = c10::CppTypeToScalarType<
      typename std::remove_const<scalar_t>::type>::value;
  const auto actual_type = t.scalar_type();
  TORCH_CHECK(
      actual_type == expect_type,
      "Expected ",
      var_name,
      " to have type ",
      expect_type,
      " but got ",
      actual_type);
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template <
    typename scalar_t,
    int64_t dim,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>
packed_accessor_or_dummy(const Tensor& t, c10::string_view var_name) {
  if (!t.defined()) {
    const std::array<index_t, dim> zeros{{0}};
    return GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(
        nullptr, zeros.data(), zeros.data());
  }
  return get_packed_accessor<scalar_t, dim, PtrTraits, index_t>(t, var_name);
}

struct InvStd {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    T invstd = 0.0f;
    if (var != static_cast<T>(0.0f) || epsilon != static_cast<T>(0.0f)) {
      invstd = static_cast<T>(1.0f) / std::sqrt(var + static_cast<T>(epsilon));
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    return var;
  }
};

static int get_num_threads(int nElem, int max_size) {
  int threadSizes[5] = {32, 64, 128, 256, max_size};
  for (int i = 0; i < 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return max_size;
}

int get_prefer_wg_size(unsigned int nHw, int simd) {
  if (nHw < simd)
    return simd;
  auto size_problem = get_num_threads(nHw, simd * simd);
  auto wg_size = syclMaxWorkGroupSize();
  return std::min(int64_t(size_problem), wg_size);
}

int get_prefer_simd(int numPlane, int nHw) {
  // decide SIMD: SIMD32 or SIMD16

  auto dev_id = at::xpu::getDeviceIndexOfCurrentQueue();

  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto sub_group_size = dev_prop->sub_group_sizes;
  int simd = sub_group_size[1];
  if (simd <= SIMD16)
    return simd;

  // if max supported simd >16
  if (nHw <= SIMD16)
    return SIMD16;
  if (simd >= SIMD32 && nHw <= SIMD32)
    return SIMD32;

  int64_t target_tile_size = syclMaxWorkItemsPerTile(dev_id);
  // for work group barrier perf
  int64_t wg_size = syclMaxWorkItemsPerEU(dev_id);
  if (simd == SIMD32) {
    // when setting wg_size 256 can achieve high occupancy, use SIMD16
    if (wg_size * numPlane >= target_tile_size)
      return SIMD16;
    // for latency case
    if (nHw <= 1024 && numPlane > 128 && SIMD16 * SIMD16 >= wg_size) {
      return SIMD16;
    }
  }
  return simd;
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  Float2() {}

  Float2(scalar_t v1, scalar_t v2)
      : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  Float2(int v)
      : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }

  friend Float2 operator+(Float2 a, const Float2& b) {
    a += b;
    return a;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  GradOp(accscalar_t m, const PTA& i, const PTA& g)
      : mean(m), input(i), grad_output(g) {}
  inline Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename item_t,
    typename local_shared_t>
static inline void group_reduce(
    item_t item,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared_t& local_data,
    reduce_op bin_op) {
  auto sg = item.get_sub_group();
  uint32_t lane_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
  }
  if (sub_group_num == 1) {
    if (lane_id == 0) {
      local_data[0] = val;
    }
    item.barrier(sycl_local_fence);
    val = local_data[0];

    return;
  }

  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  if (lane_id == 0) {
    local_data[sg_id] = val;
  }
  item.barrier(sycl_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (sg_id == 0) {
    val = init;
    if (lane_id < sub_group_num) {
      val = accscalar_t(local_data[lane_id]);
    }
    for (int i = lane_id + SIMD; i < sub_group_num; i += SIMD) {
      val = bin_op(val, static_cast<accscalar_t>(local_data[i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final
    // result
    if (lane_id == 0) {
      local_data[0] = val;
    }
  }

  item.barrier(sycl_local_fence);
  val = local_data[0];
}

template <
    int SIMD,
    typename scalar_t,
    typename item_t,
    typename Op,
    typename PTA,
    typename local_shared_t>
scalar_t plane_reduce(
    item_t item,
    Op grad_op,
    PTA tensor,
    int plane,
    int sub_group_num,
    const local_shared_t& shared) {
  // first the reductions each thread does separately
  scalar_t sum_value = static_cast<scalar_t>(0);
  for (int batch = item.get_local_id(0); batch < tensor.size(0);
       batch += item.get_local_range(0)) {
    for (int x = item.get_local_id(1); x < tensor.size(2);
         x += item.get_local_range(1)) {
      auto res = grad_op(batch, plane, x);

      sum_value += res;
    }
  }
  group_reduce<SIMD, scalar_t>(
      item,
      sub_group_num,
      sum_value,
      scalar_t(0),
      shared,
      [](scalar_t a, scalar_t b) { return a + b; });
  if (item.get_local_linear_id() == 0) {
    shared[0] = sum_value;
  }
  item.barrier(sycl_local_fence);
  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t>
int inline get_nhwc_suggest_vec_size(
    const Tensor input,
    int reduction_size,
    int channels) {
  if (!batch_norm_use_channels_last_kernels(input))
    return 1;
  // no need to vectorize if channels < 16
  if (channels < 16)
    return 1;
  // if small reduction size, make no vectorization for higher occupancy
  if (reduction_size < 8 * syclMaxWorkGroupSize())
    return 1;

  // just to load/store data
  auto func = [](scalar_t a) { return a + static_cast<scalar_t>(1.0f); };
  at::detail::Array<char*, 1> data;
  data[0] = (char*)input.data_ptr();

  int vec_size = memory::can_vectorize_up_to<decltype(func)>(data);

  // for resnet50 shape, bf16 type, vec 4 have better performance
  if (vec_size == 8 && reduction_size == 256 * 56 * 56 &&
      (channels == 128 || channels == 256))
    return 4;

  if (channels % vec_size != 0)
    return 1;
  return vec_size;
}

inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

std::tuple<sycl::range<2>, sycl::range<2>> flexible_launch_configs(
    const int reduction,
    const int stride,
    const bool coop_flag = false,
    const int loops_per_item = 1) {
  int wg_size = syclMaxWorkItemsPerEU();
  int group_x = std::min(last_pow2(stride), 32);
  int group_y =
      std::min(last_pow2(div_up(reduction, loops_per_item)), wg_size / group_x);
  if (group_x * group_y != wg_size) {
    group_x = std::min(last_pow2(stride), wg_size / group_y);
  }

  int grid_x = div_up(stride, group_x);
  //  int grid_y = std::min(div_up(reduction, group_y * loops_per_item), 1024);
  int grid_y = std::min(
      div_up(reduction, group_y * loops_per_item),
      int(syclMaxWorkItemsPerTile()) / (grid_x * group_x) / (group_y));
  grid_y = std::max(grid_y, 1);

  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  sycl::range<2> local_range(group_y, group_x);
  sycl::range<2> global_range(grid_y * group_y, grid_x * group_x);

  return std::make_tuple(global_range, local_range);
}

// ========================== batch_norm_stats ==========================

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
struct BatchNormCollectStatisticsKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    int plane = item.get_group(0);
    int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    auto sg_lid = sg.get_local_linear_id();
    auto sg_id = sg.get_group_linear_id();

    // Compute the mean and variance across (batch, x/y/z)
    // this uses the Welford (in the for loop)/parallel algorithm (to sum
    // across the group)
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    // and the parallel algorithm on the same page.
    // We use two shuffles to reduce across the entire group.
    // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a
    // description.

    // first the reductions each thread does separately
    stat_accscalar_t avg = 0;
    stat_accscalar_t var_n = 0;
    int n = 0;
    for (int batch = item.get_local_id(0); batch < N_;
         batch += item.get_local_range(0)) {
      for (int x = item.get_local_id(1); x < Hw_;
           x += item.get_local_range(1)) {
        stat_accscalar_t v = input_[batch][plane][x];
        stat_accscalar_t d1 = v - avg;
        n++;
        avg += d1 / n;
        var_n += d1 * (v - avg);
      }
    }

    // first warpSum to get one value per thread to
    // one value per warp
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
      int o_n = sg.shuffle_xor(n, i);
      stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
      var_n += sg.shuffle_xor(var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // this writes each warps item into shared memory

    if (sg_lid == 0) {
      shared_n_[sg_id] = n;
      shared_avg_var_[sg_id * 2] = avg;
      shared_avg_var_[sg_id * 2 + 1] = var_n;
    }
    item.barrier(sycl_local_fence);
    // now have a second warpSum to reduce the intermediate values
    // from shared memory to a single number. The very first
    // thread writes it to shared memory.

    if (tid < sg_num_) {
      n = shared_n_[tid];
      avg = shared_avg_var_[2 * tid];
      var_n = shared_avg_var_[2 * tid + 1];
    } else {
      n = 0;
      avg = stat_accscalar_t(0);
      var_n = stat_accscalar_t(0);
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
      int o_n = sg.shuffle_xor(n, i);
      stat_accscalar_t factor = 1.0f / fmaxf(1.0f, n + o_n);
      var_n += sg.shuffle_xor(var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // Save the mean, variance, and moving averages
    auto save_mean = save_mean_;
    auto save_transformed_var = save_transformed_var_;
    if (tid == 0) {
      if (save_mean_.data() != NULL) {
        save_mean[plane] = avg;
      }
      if (save_transformed_var_.data() != NULL) {
        save_transformed_var[plane] =
            VarTransform{}(var_n / (N_ * Hw_), epsilon_);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_n_ = sycl_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{(size_t)sg_num_}, cgh);
    shared_avg_var_ = sycl_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{(size_t)sg_num_ * 2 + 2}, cgh);
  }

  BatchNormCollectStatisticsKernelFunctor(
      int N,
      int numPlane,
      int Hw,
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          RestrictPtrTraits,
          index_t> input,
      const stat_accscalar_t epsilon,
      const stat_accscalar_t momentum,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          RestrictPtrTraits,
          index_t> save_mean,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          RestrictPtrTraits,
          index_t> save_transformed_var,
      int64_t sg_num,
      int batch_stride)
      : N_(N),
        numPlane_(numPlane),
        Hw_(Hw),
        input_(input),
        epsilon_(epsilon),
        momentum_(momentum),
        save_mean_(save_mean),
        save_transformed_var_(save_transformed_var),
        sg_num_(sg_num),
        batch_stride_(batch_stride) {}

 private:
  int N_;
  int numPlane_;
  int Hw_;
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      RestrictPtrTraits,
      index_t>
      input_;
  const stat_accscalar_t epsilon_;
  const stat_accscalar_t momentum_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
      save_mean_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
      save_transformed_var_;
  int64_t sg_num_;
  int batch_stride_;
  sycl_local_acc_t<stat_accscalar_t, 1> shared_n_;
  sycl_local_acc_t<stat_accscalar_t, 1> shared_avg_var_;
};

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_collect_statistics_kernel(
    const GenericPackedTensorAccessor<
        const input_scalar_t,
        3,
        RestrictPtrTraits,
        index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
        save_mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>
        save_transformed_var) {
  auto& queue = getCurrentSYCLQueue();
  auto N = input.size(0);
  auto numPlane = input.size(1);
  auto Hw = input.size(2);
  int64_t wg_size = get_prefer_wg_size(N * Hw, SIMD);
  int64_t work_group_size_x = get_num_threads(Hw, wg_size);
  int64_t work_group_size_y = std::max(int64_t(1), wg_size / work_group_size_x);
  work_group_size_y = std::min(int64_t(N), work_group_size_y);
  int64_t sg_num = work_group_size_x * work_group_size_y / SIMD;
  auto batch_stride = numPlane * Hw;
  auto caller = BatchNormCollectStatisticsKernelFunctor<
      SIMD,
      VarTransform,
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t>(
      N,
      numPlane,
      Hw,
      input,
      epsilon,
      momentum,
      save_mean,
      save_transformed_var,
      sg_num,
      batch_stride);
  sycl_kernel_submit(
      sycl::range<2>(
          (size_t)numPlane * work_group_size_y, (size_t)work_group_size_x),
      sycl::range<2>((size_t)work_group_size_y, (size_t)work_group_size_x),
      queue,
      caller);
}

template <typename scalar_t, typename index_t, typename VarTransform>
void batch_norm_stats_template(
    const Tensor& out_mean,
    const Tensor& out_invstd,
    const Tensor& input_,
    double epsilon) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions

  int N = input_reshaped.size(0);
  int C = input_reshaped.size(1);
  int Hw = input_reshaped.size(2);
  int simd = get_prefer_simd(C, N * Hw);

  at::native::resize_output(out_mean, {n_input});
  at::native::resize_output(out_invstd, {n_input});

  auto input =
      get_packed_accessor<const scalar_t, 3, RestrictPtrTraits, index_t>(
          input_reshaped, "input");

  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  auto mean =
      packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(
          out_mean, "out_mean");
  auto invstd =
      packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(
          out_invstd, "out_invstd");

  if (simd == SIMD32) {
    batch_norm_collect_statistics_kernel<
        SIMD32,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(input, epsilon, 0.0, mean, invstd);
  } else {
    batch_norm_collect_statistics_kernel<
        SIMD16,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(input, epsilon, 0.0, mean, invstd);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    typename vec_y,
    int vec_size,
    bool two_pass_reduce>
struct BatchNormReduceSumChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    // int plane = item.get_group(0);
    // int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    int thread_idx_y = item.get_local_id(0);
    // int thread_idx_x = item.get_local_id(1);
    int group_idx_y = item.get_group(0);
    // int group_idx_x = item.get_group(1);

    int address_base = m_offset * stride_ + c_offset_base;
    int inner_loop_stride = global_range_y_;
    int address_increment = inner_loop_stride * stride_;

    accscalar_t x_sum[vec_size] = {0.0f};
    accscalar_t x_sq_sum[vec_size] = {0.0f};
    // thread reduction
    for (int i = 0; i < loop_count_; i++) {
      vec_t x_math_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        auto c_offset = c_offset_base + j;

        if (c_offset < stride_ && m_offset < reduction_size_) {
          // scalar_t arr = input_ptr_[address_base + j];
          auto x_math = x_math_vec[j];
          x_sum[j] += x_math;
          x_sq_sum[j] += x_math * x_math;
        }
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      vec_y value;
      value[0] = x_sum[j];
      value[1] = x_sq_sum[j];

      value = group_y_reduce(
          item, shared_, value, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });

      x_sum[j] = value[0];
      x_sq_sum[j] = value[1];

      item.barrier(sycl_local_fence);
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      auto c_offset = c_offset_base + j;
      // global_reduciton
      if (thread_idx_y == 0 && c_offset < stride_) {
        if constexpr (two_pass_reduce) {
          // write to temp[c][group_idx_y]
          // int offset = c_offset * group_num_y_ + group_idx_y;
          temp_sum_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sum[j];
          temp_sum_sq_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sq_sum[j];
        } else {
          out_mean_ptr_[c_offset] = x_sum[j];
          out_invstd_ptr_[c_offset] = x_sq_sum[j];
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<vec_y, 1>(sycl::range<1>{(size_t)wg_size_}, cgh);
  }

  BatchNormReduceSumChannelsLastKernelFunctor(
      const int reduction_size,
      const int stride,
      int global_range_y,
      int local_range_y,
      int group_num_x,
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      scalar_t* input_ptr,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr,
      int loop_count)
      : reduction_size_(reduction_size),
        stride_(stride),
        global_range_y_(global_range_y),
        local_range_y_(local_range_y),
        group_num_x_(group_num_x),
        group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        input_ptr_(input_ptr),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr),
        loop_count_(loop_count) {}

 private:
  const int reduction_size_;
  const int stride_;
  int global_range_y_;
  int local_range_y_;
  int group_num_x_;
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  scalar_t* input_ptr_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
  int loop_count_;
  sycl_local_acc_t<vec_y, 1> shared_;
};

template <typename accscalar_t>
struct BatchNormReduceSumChannelsLastTwoPassKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto local_id = item.get_local_linear_id();
    // auto global_id = item.get_global_linear_id();
    auto c_offset = item.get_group_linear_id();

    accscalar_t temp_sum_val = 0.0f;
    accscalar_t temp_sum_sq_val = 0.0f;
    for (int i = local_id; i < group_num_y_; i += wg_size_) {
      int offset = c_offset * group_num_y_ + i;
      temp_sum_val += temp_sum_ptr_[offset];
      temp_sum_sq_val += temp_sum_sq_ptr_[offset];
    }
    auto total_sum = sycl::reduce_over_group(
        item.get_group(), temp_sum_val, sycl::plus<accscalar_t>());
    auto total_sum_sq = sycl::reduce_over_group(
        item.get_group(), temp_sum_sq_val, sycl::plus<accscalar_t>());
    if (local_id == 0) {
      out_mean_ptr_[c_offset] = total_sum;
      out_invstd_ptr_[c_offset] = total_sum_sq;
    }
  }
  BatchNormReduceSumChannelsLastTwoPassKernelFunctor(
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr)
      : group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr) {}

 private:
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
};

// sum x and x^2 in channels
template <
    typename scalar_t,
    typename accscalar_t,
    int vec_size,
    bool two_pass_reduce>
void batch_norm_reduce_sum_channels_last_kernel(
    const Tensor input,
    Tensor& out_mean,
    Tensor& out_invstd,
    const int reduction_size,
    const int stride) {
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size, true);
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();
  auto global_range_y = global_range[0];
  auto local_range_y = local_range[0];
  int group_num_x = global_range[1] / local_range[1];
  int group_num_y = global_range[0] / local_range[0];
  Tensor temp_sum, temp_sum_sq;
  accscalar_t* temp_sum_ptr = nullptr;
  accscalar_t* temp_sum_sq_ptr = nullptr;
  if constexpr (two_pass_reduce) {
    out_mean.zero_();
    out_invstd.zero_();
    temp_sum = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_sq = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_ptr = temp_sum.data_ptr<accscalar_t>();
    temp_sum_sq_ptr = temp_sum_sq.data_ptr<accscalar_t>();
  }
  int wg_size = local_range[0] * local_range[1];

  auto input_ptr = input.data_ptr<scalar_t>();
  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();

  int loop_count = 1 + (reduction_size - 1) / (global_range_y);
  using vec_y = at::detail::Array<accscalar_t, 2>;

  auto caller = BatchNormReduceSumChannelsLastKernelFunctor<
      scalar_t,
      accscalar_t,
      vec_t,
      vec_y,
      vec_size,
      two_pass_reduce>(
      reduction_size,
      stride,
      global_range_y,
      local_range_y,
      group_num_x,
      group_num_y,
      temp_sum_ptr,
      temp_sum_sq_ptr,
      wg_size,
      input_ptr,
      out_mean_ptr,
      out_invstd_ptr,
      loop_count);
  sycl_kernel_submit(global_range, local_range, queue, caller);

  // reduce temp sum
  if constexpr (two_pass_reduce) {
    int wg_size = std::min(group_num_y, int(syclMaxWorkItemsPerEU()));
    auto caller =
        BatchNormReduceSumChannelsLastTwoPassKernelFunctor<accscalar_t>(
            group_num_y,
            temp_sum_ptr,
            temp_sum_sq_ptr,
            wg_size,
            out_mean_ptr,
            out_invstd_ptr);
    sycl_kernel_submit(
        (size_t)stride * wg_size, (size_t)wg_size, queue, caller);
  }
}

template <typename VarTransform, typename scalar_t, typename stat_accscalar_t>
struct BatchNormUpdateMeanVarKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto c_offset = item.get_global_linear_id();
    if (c_offset < channel_num_) {
      scalar_t mean = mean_[c_offset] * factor_;

      mean_[c_offset] = mean;
      var_[c_offset] =
          VarTransform{}(var_[c_offset] * factor_ - mean * mean, epsilon_);
    }
  }
  BatchNormUpdateMeanVarKernelFunctor(
      scalar_t* mean,
      scalar_t* var,
      int channel_num,
      scalar_t factor,
      stat_accscalar_t epsilon)
      : mean_(mean),
        var_(var),
        channel_num_(channel_num),
        factor_(factor),
        epsilon_(epsilon) {}

 private:
  scalar_t* mean_;
  scalar_t* var_;
  int channel_num_;
  scalar_t factor_;
  stat_accscalar_t epsilon_;
};

template <typename VarTransform, typename scalar_t, typename stat_accscalar_t>
void batch_norm_update_mean_var_kernel(
    scalar_t* mean_,
    scalar_t* var_,
    int channel_num,
    scalar_t factor,
    stat_accscalar_t epsilon) {
  auto& queue = getCurrentSYCLQueue();
  int64_t wg_size = std::min(
      int64_t(channel_num),
      syclMaxWorkItemsPerEU()); // for work group barrier

  sycl::range<1> local_range(wg_size);
  sycl::range<1> global_range((channel_num + wg_size - 1) / wg_size * wg_size);

  auto caller = BatchNormUpdateMeanVarKernelFunctor<
      VarTransform,
      scalar_t,
      stat_accscalar_t>(mean_, var_, channel_num, factor, epsilon);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename scalar_t, typename VarTransform>
void batch_norm_stats_channels_last_template(
    Tensor& out_mean,
    Tensor& out_invstd,
    const Tensor& input,
    double epsilon) {
  using accscalar_t = acc_type<scalar_t, true>;

  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::native::resize_output(out_mean, {stride});
  at::native::resize_output(out_invstd, {stride});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  int suggest_vec_size =
      get_nhwc_suggest_vec_size<scalar_t>(input, reduction_size, stride);

#define DISPATCH_REDUCE_2_PASS_IMPL(vec_size)                       \
  {                                                                 \
    batch_norm_reduce_sum_channels_last_kernel<                     \
        scalar_t,                                                   \
        accscalar_t,                                                \
        vec_size,                                                   \
        true>(input, out_mean, out_invstd, reduction_size, stride); \
  }

#define DISPATCH_REDUCE_IMPL(vec_size)                               \
  {                                                                  \
    batch_norm_reduce_sum_channels_last_kernel<                      \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        vec_size,                                                    \
        false>(input, out_mean, out_invstd, reduction_size, stride); \
  }
  sycl::range<2> global_range(1, 1), local_range(1, 1);

  switch (suggest_vec_size) {
    case 8: {
      constexpr int vec_size = 8;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
      break;
    }
    case 4: {
      constexpr int vec_size = 4;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
      break;
    }
    default: {
      constexpr int vec_size = 1;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
    }
  }

  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();
  const auto factor = static_cast<accscalar_t>(1.0f / reduction_size);
  batch_norm_update_mean_var_kernel<VarTransform>(
      out_mean_ptr, out_invstd_ptr, stride, factor, epsilon);
#undef DISPATCH_REDUCE_2_PASS_IMPL
#undef DISPATCH_REDUCE_IMPL
}

std::tuple<Tensor, Tensor> batch_norm_stats_kernel(
    const Tensor& self,
    double epsilon) {
  auto options =
      self.options().dtype(at::toAccumulateType(self.scalar_type(), true));
  auto n_channels = self.size(1);
  auto save_mean = at::empty({n_channels}, options);
  auto save_invstd = at::empty({n_channels}, options);

  bool use_channels_last_kernel = batch_norm_use_channels_last_kernels(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_stats_xpu",
      [&] {
        if (canUse32BitIndexMath(self)) {
          if (use_channels_last_kernel) {
            batch_norm_stats_channels_last_template<scalar_t, InvStd>(
                save_mean, save_invstd, self, epsilon);
          } else {
            batch_norm_stats_template<scalar_t, int32_t, InvStd>(
                save_mean, save_invstd, self, epsilon);
          }
        } else {
          batch_norm_stats_template<scalar_t, int64_t, InvStd>(
              save_mean, save_invstd, self, epsilon);
        }
      });
  return std::tuple<Tensor, Tensor>(save_mean, save_invstd);
}

// ========================== batch_norm_elemt ==========================

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
struct BatchNormTransformInputKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto group_idx_x = item.get_group().get_group_id(1);
    index_t plane = group_idx_x;

    if (plane >= numPlane_) {
      return;
    }

    stat_accscalar_t gamma = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta = bias_.size(0) > 0
        ? static_cast<stat_accscalar_t>(bias_[plane])
        : static_cast<stat_accscalar_t>(0);

    stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
    stat_accscalar_t invstd;
    if constexpr (train) {
      invstd = var_or_invstd_[plane];
    } else {
      invstd =
          static_cast<stat_accscalar_t>(1) /
          std::sqrt(
              static_cast<stat_accscalar_t>(var_or_invstd_[plane]) + epsilon_);
    }

    index_t bstep = item.get_global_range(0);
    for (index_t batch = item.get_global_id(0); batch < bs_; batch += bstep) {
      auto o = output_[batch][plane];
      auto i = input_[batch][plane];
      for (index_t feature = item.get_local_id(1); feature < fs_;
           feature += item.get_local_range(1)) {
        o[feature] = static_cast<input_scalar_t>(
            gamma * (i[feature] - mean) * invstd + beta);
      }
    }
  }

  BatchNormTransformInputKernelFunctor(
      stat_accscalar_t epsilon,
      int numPlane,
      int64_t target_tile_size,
      int64_t wg_size,
      int bs,
      int fs,
      int weight_size,
      int bias_size,
      int tf,
      int tb,
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          RestrictPtrTraits,
          index_t> input,
      GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
          output,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          RestrictPtrTraits,
          index_t> bias,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> mean,
      const GenericPackedTensorAccessor<
          typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::
              type,
          1,
          RestrictPtrTraits,
          index_t> var_or_invstd)
      : epsilon_(epsilon),
        numPlane_(numPlane),
        target_tile_size_(target_tile_size),
        wg_size_(wg_size),
        bs_(bs),
        fs_(fs),
        weight_size_(weight_size),
        bias_size_(bias_size),
        tf_(tf),
        tb_(tb),
        input_(input),
        output_(output),
        weight_(weight),
        bias_(bias),
        mean_(mean),
        var_or_invstd_(var_or_invstd) {}

 private:
  stat_accscalar_t epsilon_;
  int numPlane_;
  int64_t target_tile_size_;
  int64_t wg_size_;
  int bs_;
  int fs_;
  int weight_size_;
  int bias_size_;
  int tf_;
  int tb_;
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      RestrictPtrTraits,
      index_t>
      input_;
  GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t>
      output_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      weight_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>
      bias_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      mean_;
  const GenericPackedTensorAccessor<
      typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type,
      1,
      RestrictPtrTraits,
      index_t>
      var_or_invstd_;
};

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
void batch_norm_transform_input_kernel(
    const Tensor input,
    Tensor& output,
    const Tensor& mean_,
    const Tensor& var_or_invstd,
    const Tensor& weight,
    const Tensor& bias,
    stat_accscalar_t epsilon) {
  auto& queue = getCurrentSYCLQueue();
  int numPlane = input.size(1);
  int64_t target_tile_size = syclMaxWorkItemsPerTile();
  int64_t wg_size = syclMaxWorkItemsPerEU(); // for work group barrier
  if (wg_size * numPlane < target_tile_size) {
    wg_size = syclMaxWorkGroupSize(); // for higher occupancy
  }

  int bs = input.size(0);
  int fs = input.size(2);
  int weight_size = weight.size(0);
  int bias_size = bias.size(0);

  int tf = get_num_threads(fs, wg_size);
  int tb = std::max<int>(wg_size / tf, 1);
  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range((bs + tb - 1) / tb * tb, numPlane * tf);

  auto input_pa =
      get_packed_accessor<const input_scalar_t, 3, RestrictPtrTraits, index_t>(
          input, "input");
  auto output_pa =
      get_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>(
          output, "output");
  auto weight_pa = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>(weight, "weight");
  auto bias_pa = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      RestrictPtrTraits,
      index_t>(bias, "bias");
  auto mean_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(
          mean_, "mean");
  auto invstd_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(
          var_or_invstd, "invstd");

  auto caller = BatchNormTransformInputKernelFunctor<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      train,
      index_t>(
      epsilon,
      numPlane,
      target_tile_size,
      wg_size,
      bs,
      fs,
      weight_size,
      bias_size,
      tf,
      tb,
      input_pa,
      output_pa,
      weight_pa,
      bias_pa,
      mean_pa,
      invstd_pa);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_template(
    const Tensor& output_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& mean_,
    const Tensor& invstd_) {
  using stat_accscalar_t = acc_type<stat_scalar_t, true>;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  // NOTE: We use transform_input_kernel in training mode, which ignores
  // epsilon
  const double dummy_epsilon = 1e-5;

  batch_norm_transform_input_kernel<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      true,
      index_t>(
      input_reshaped,
      output_reshaped,
      mean_,
      invstd_,
      weight_,
      bias_,
      dummy_epsilon);
}

template <typename scalar_t, typename acc_t>
struct BatchNormElementwiseLoopsFunctor {
  scalar_t operator()(
      scalar_t input,
      acc_t weight,
      acc_t bias,
      acc_t mean,
      acc_t invstd) const {
    return ((input - mean) * invstd) * weight + bias;
  }
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size,
    typename vec_t,
    typename vec_s_t>
struct BatchNormTransformInputChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // auto group_idx_x = item.get_group().get_group_id(1);

    // int inner_loop_stride = item.get_global_range(0);
    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    if (c_offset_base >= stride_ || m_offset >= reduction_size_) {
      return;
    }

    vec_s_t m_c = *(reinterpret_cast<vec_s_t*>(mean_ptr_ + c_offset_base));
    vec_s_t inv_vec =
        *(reinterpret_cast<vec_s_t*>(inv_std_ptr_ + c_offset_base));
    vec_s_t w_c;
    vec_s_t s_c;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      if (weight_ptr_ != nullptr) {
        w_c[j] = static_cast<accscalar_t>(weight_ptr_[c_offset_base + j]) *
            inv_vec[j];
      } else {
        w_c[j] = (inv_vec[j]);
      }
      if (shift_ptr_ != nullptr) {
        s_c[j] = shift_ptr_[c_offset_base + j];
      } else {
        s_c[j] = static_cast<accscalar_t>(0.0f);
      }
    }

    int address_base = m_offset * stride_ + c_offset_base;
    int address_increment = item.get_global_range(0) * stride_;

    vec_t output_vec;
    for (; address_base < total_num_; address_base += address_increment) {
      vec_t x_math_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        // auto c_offset = c_offset_base + j;

        output_vec[j] =
            w_c[j] * (static_cast<accscalar_t>(x_math_vec[j]) - m_c[j]) +
            s_c[j];
      }
      *(reinterpret_cast<vec_t*>(output_ptr_ + address_base)) = output_vec;
    }
  }
  BatchNormTransformInputChannelsLastKernelFunctor(
      scalar_t* input_ptr,
      const scalar_t* z_ptr,
      accscalar_t* mean_ptr,
      accscalar_t* inv_std_ptr,
      const layerscalar_t* weight_ptr,
      const layerscalar_t* shift_ptr,
      scalar_t* output_ptr,
      const int reduction_size,
      const int stride,
      const bool fuse_relu,
      int64_t total_num)
      : input_ptr_(input_ptr),
        z_ptr_(z_ptr),
        mean_ptr_(mean_ptr),
        inv_std_ptr_(inv_std_ptr),
        weight_ptr_(weight_ptr),
        shift_ptr_(shift_ptr),
        output_ptr_(output_ptr),
        reduction_size_(reduction_size),
        stride_(stride),
        fuse_relu_(fuse_relu),
        total_num_(total_num) {}

 private:
  scalar_t* input_ptr_;
  const scalar_t* z_ptr_;
  accscalar_t* mean_ptr_;
  accscalar_t* inv_std_ptr_;
  const layerscalar_t* weight_ptr_;
  const layerscalar_t* shift_ptr_;
  scalar_t* output_ptr_;
  const int reduction_size_;
  const int stride_;
  const bool fuse_relu_;
  int64_t total_num_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size>
void batch_norm_transform_input_channels_last_kernel(
    scalar_t* input_ptr,
    const scalar_t* z_ptr,
    accscalar_t* mean_ptr,
    accscalar_t* inv_std_ptr,
    const layerscalar_t* weight_ptr,
    const layerscalar_t* shift_ptr,
    scalar_t* output_ptr,
    const int reduction_size,
    const int stride,
    const bool fuse_relu) {
  // tensor dimension (m,c)
  // loop along m dimension
  int64_t total_num = reduction_size * stride;
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  using vec_s_t = memory::aligned_vector<accscalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size);

  auto caller = BatchNormTransformInputChannelsLastKernelFunctor<
      scalar_t,
      accscalar_t,
      layerscalar_t,
      vec_size,
      vec_t,
      vec_s_t>(
      input_ptr,
      z_ptr,
      mean_ptr,
      inv_std_ptr,
      weight_ptr,
      shift_ptr,
      output_ptr,
      reduction_size,
      stride,
      fuse_relu,
      total_num);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

void batch_norm_elemt_channels_last_template(
    const Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& shift, // bias of BN
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::optional<at::Tensor>& z = c10::nullopt, // bias after BN
    const bool fuse_relu = false) {
  const auto second_dtype = weight.defined()
      ? weight.scalar_type()
      : (shift.defined() ? shift.scalar_type() : input.scalar_type());
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

#define DISPATCH_TRANSFORM_INPUT_IMPL(vec_size)                   \
  {                                                               \
    batch_norm_transform_input_channels_last_kernel<              \
        scalar_t,                                                 \
        accscalar_t,                                              \
        scalar_t,                                                 \
        vec_size>(                                                \
        input.data_ptr<scalar_t>(),                               \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr, \
        mean.data_ptr<accscalar_t>(),                             \
        inv_std.data_ptr<accscalar_t>(),                          \
        weight.defined() ? weight.data_ptr<scalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<scalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                              \
        reduction_size,                                           \
        stride,                                                   \
        fuse_relu);                                               \
  }

#define DISPATCH_TRANSFORM_ACC_INPUT_IMPL(vec_size)                  \
  {                                                                  \
    batch_norm_transform_input_channels_last_kernel<                 \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        accscalar_t,                                                 \
        vec_size>(                                                   \
        input.data_ptr<scalar_t>(),                                  \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr,    \
        mean.data_ptr<accscalar_t>(),                                \
        inv_std.data_ptr<accscalar_t>(),                             \
        weight.defined() ? weight.data_ptr<accscalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<accscalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                                 \
        reduction_size,                                              \
        stride,                                                      \
        fuse_relu);                                                  \
  }

  if (input.scalar_type() != second_dtype) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(1);
          }
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_forward: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_INPUT_IMPL(1);
          }
        });
  }
#undef DISPATCH_TRANSFORM_INPUT_IMPL
#undef DISPATCH_TRANSFORM_ACC_INPUT_IMPL
}

void batch_norm_elemt_kernel(
    Tensor& out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const Tensor& mean_,
    const Tensor& invstd_) {
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      c10::MaybeOwned<Tensor> weight =
          at::borrow_from_optional_tensor(weight_opt);
      c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
      at::native::resize_output(out, self.sizes());
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using accscalar_t = acc_type<scalar_t, true>;
            const bool mixed_type = is_mixed_type(self, *weight, *bias);
            if (mixed_type) {
              batch_norm_elemt_template<scalar_t, accscalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            } else {
              batch_norm_elemt_template<scalar_t, scalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            }
          });
      return;
    }
    case Impl::ChannelsLast: {
      auto weight = at::borrow_from_optional_tensor(weight_opt);
      auto bias = at::borrow_from_optional_tensor(bias_opt);

      if (resize_output_check(out, self.sizes())) {
        resize_impl_xpu_(
            out.unsafeGetTensorImpl(), self.sizes(), self.strides());
      }
      if ((out.strides() == self.strides()) &&
          (!weight->defined() || weight->is_contiguous()) &&
          (!bias->defined() || bias->is_contiguous()) &&
          (!mean_.defined() || mean_.is_contiguous()) &&
          (!invstd_.defined() || invstd_.is_contiguous())) {
        batch_norm_elemt_channels_last_template(
            out, self, *weight, *bias, mean_, invstd_);
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      // Helper to convert 1d tensors to an nd tensor that broadcasts with
      // input All elements go into the channel dimension
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };

      auto weight = weight_opt.has_value() && weight_opt->defined()
          ? as_nd(*weight_opt)
          : at::scalar_tensor(1, mean_.options());
      auto bias = bias_opt.has_value() && bias_opt->defined()
          ? as_nd(*bias_opt)
          : at::scalar_tensor(0, mean_.options());
      auto mean = as_nd(mean_);
      auto invstd = as_nd(invstd_);

      auto iter = TensorIteratorConfig()
                      .add_output(out)
                      .add_input(self)
                      .add_input(weight)
                      .add_input(bias)
                      .add_input(mean)
                      .add_input(invstd)
                      .check_all_same_dtype(false)
                      .promote_inputs_to_common_dtype(false)
                      .build();

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using acc_t = acc_type<scalar_t, true>;
            auto f = BatchNormElementwiseLoopsFunctor<scalar_t, acc_t>();
            gpu_kernel(iter, f);
          });
      return;
    }
  }
}

// ====================== batch_norm_backward_reduce ======================

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
struct BatchNormBackwardReduceKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    index_t plane = item.get_group(0);
    auto lidx = item.get_local_id(1);
    // auto lidy = item.get_local_id(0);

    if (plane >= numPlane_) {
      return;
    }

    stat_accscalar_t r_mean = mean_[plane];
    stat_accscalar_t factor = invstd_[plane];
    GradOp<
        input_scalar_t,
        stat_accscalar_t,
        GenericPackedTensorAccessor<
            input_scalar_t,
            3,
            DefaultPtrTraits,
            index_t>>
        g(r_mean, input_, grad_output_);
    auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
        item, g, grad_output_, plane, sg_num_, local_sum_);

    if (lidx == 0) {
      if (grad_weight_.size(0) > 0) {
        auto grad_weight = grad_weight_;
        grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
      }
      if (grad_bias_.size(0) > 0) {
        auto grad_bias = grad_bias_;
        grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
      }
      if (sum_dy_.size(0) > 0) {
        auto sum_dy = sum_dy_;
        sum_dy[plane] = static_cast<stat_accscalar_t>(res.v1);
      }
      if (sum_dy_xmu_.size(0) > 0) {
        auto sum_dy_xmu = sum_dy_xmu_;
        sum_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum_ = sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1>(
        sycl::range<1>{(size_t)tx_ * ty_}, cgh);
  }

  BatchNormBackwardReduceKernelFunctor(
      int numPlane,
      int i_batch_size,
      int i_feature_size,
      int o_batch_size,
      int o_feature_size,
      int64_t wg_size,
      int tx,
      int ty,
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> input,
      const GenericPackedTensorAccessor<
          input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> grad_output,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> mean,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> invstd,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy,
      GenericPackedTensorAccessor<
          stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> sum_dy_xmu,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_weight,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_bias,
      int sg_num)
      : numPlane_(numPlane),
        i_batch_size_(i_batch_size),
        i_feature_size_(i_feature_size),
        o_batch_size_(o_batch_size),
        o_feature_size_(o_feature_size),
        wg_size_(wg_size),
        tx_(tx),
        ty_(ty),
        input_(input),
        grad_output_(grad_output),
        mean_(mean),
        invstd_(invstd),
        sum_dy_(sum_dy),
        sum_dy_xmu_(sum_dy_xmu),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        sg_num_(sg_num) {}

 private:
  int numPlane_;
  int i_batch_size_;
  int i_feature_size_;
  int o_batch_size_;
  int o_feature_size_;
  int64_t wg_size_;
  int tx_;
  int ty_;
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      input_;
  const GenericPackedTensorAccessor<
      input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      grad_output_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      mean_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      invstd_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      sum_dy_;
  GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t>
      sum_dy_xmu_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_weight_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_bias_;
  int sg_num_;
  sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1> local_sum_;
};

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_backward_reduce_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& mean,
    const Tensor& invstd,
    Tensor& sum_dy,
    Tensor& sum_dy_xmu,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  int numPlane = input.size(1);
  auto i_batch_size = input.size(0);
  auto i_feature_size = input.size(2);
  auto o_batch_size = grad_output.size(0);
  auto o_feature_size = grad_output.size(2);

  auto& queue = getCurrentSYCLQueue();
  int64_t wg_size = get_prefer_wg_size(
      i_batch_size * i_feature_size, SIMD); // for higher occupancy

  int tx = get_num_threads(i_feature_size, wg_size);
  int ty = std::min(int64_t(last_pow2(i_batch_size)), wg_size / tx);
  ty = std::max(1, ty);
  sycl::range<2> local_range(ty, tx);
  sycl::range<2> global_range(numPlane * ty, tx);

  auto input_pa =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input, "input");
  auto grad_output_pa =
      get_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output, "grad_output");
  auto grad_weight_pa =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_weight, "grad_weight");
  auto grad_bias_pa =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_bias, "grad_bias");
  auto mean_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          mean, "mean");
  auto invstd_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          invstd, "invstd");
  auto sum_dy_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy, "sum_dy");
  auto sum_dy_xmu_pa =
      packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(
          sum_dy_xmu, "sum_dy_xmu");

  int sg_num = tx * ty / SIMD;
  auto caller = BatchNormBackwardReduceKernelFunctor<
      SIMD,
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t>(
      numPlane,
      i_batch_size,
      i_feature_size,
      o_batch_size,
      o_feature_size,
      wg_size,
      tx,
      ty,
      input_pa,
      grad_output_pa,
      mean_pa,
      invstd_pa,
      sum_dy_pa,
      sum_dy_xmu_pa,
      grad_weight_pa,
      grad_bias_pa,
      sg_num);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

// supports CF and CL
template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const bool input_g,
    const bool weight_g,
    const bool bias_g) {
  using stat_accscalar_t = acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto new_shape = {
      input_.size(0),
      input_.size(1),
      input_.numel() / input_.size(0) / input_.size(1)};
  auto new_stride = {input_.stride(0), input_.stride(1), input_.stride(-1)};

  auto input_reshaped = at::as_strided(input_, new_shape, new_stride);
  auto grad_output_reshaped = at::as_strided(grad_out_, new_shape, new_stride);

  if (input_g) {
    sum_dy_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    sum_dy_xmu_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (weight_g) {
    grad_weight_ = at::empty({n_input}, weight_.options());
  }
  if (bias_g) {
    grad_bias_ = at::empty({n_input}, weight_.options());
  }

  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));
  if (simd == SIMD32) {
    batch_norm_backward_reduce_kernel<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        mean_,
        invstd_,
        sum_dy_,
        sum_dy_xmu_,
        grad_weight_,
        grad_bias_);
  } else {
    batch_norm_backward_reduce_kernel<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        stat_accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        mean_,
        invstd_,
        sum_dy_,
        sum_dy_xmu_,
        grad_weight_,
        grad_bias_);
  }
  return std::make_tuple(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

template <
    int vec_size,
    bool two_pass_reduce,
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    typename vec_t,
    typename vec_y>
struct BatchNormBackwardReduceChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    // hide latency with concurrency
    accscalar_t sum_dy[vec_size];
    accscalar_t sum_dy_xmu[vec_size];

#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      sum_dy[i] = accscalar_t(0.0f);
      sum_dy_xmu[i] = accscalar_t(0.0f);
    }
    // tensor dimension (m,c)

    // loop along m dimension
    int inner_loop_stride = item.get_global_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;
    // auto lidx = item.get_local_id(1);
    auto lidy = item.get_local_id(0);
    auto group_idx_y = item.get_group().get_group_id(0);
    // auto group_idx_x = item.get_group().get_group_id(1);

    int loop_count = 1 + (reduction_size_ - 1) / (inner_loop_stride);
    int address_base = m_offset * stride_ + c_offset_base;
    int address_increment = inner_loop_stride * stride_;

    accscalar_t r_mean[vec_size] = {0.0f};
    accscalar_t factor[vec_size] = {1.0f};
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      if (c_offset_base + j < stride_) {
        r_mean[j] = mean_ptr_[c_offset_base + j];
        factor[j] = inv_std_[c_offset_base + j];
      }
    }
    for (int i = 0; i < loop_count; i++) {
      accscalar_t x_input[vec_size];
      accscalar_t x_grad_output[vec_size];
      vec_t input_vec;
      vec_t grad_out_vec;
      if ((c_offset_base) < stride_ && m_offset < reduction_size_) {
        input_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
        grad_out_vec = *(reinterpret_cast<vec_t*>(grad_output_ + address_base));
      }

      // load multiple data in
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        if (c_offset_base < stride_ && m_offset < reduction_size_) {
          x_input[j] = input_vec[j];
          x_grad_output[j] = grad_out_vec[j];
        } else {
          x_input[j] = accscalar_t(0);
          x_grad_output[j] = accscalar_t(0);
        }
      }

      // calculate sum_dy / sum_dy_xmu
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        sum_dy[j] += x_grad_output[j];
        sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean[j]);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      vec_y value;
      value[0] = sum_dy[j];
      value[1] = sum_dy_xmu[j];

      value = group_y_reduce(
          item, shared_, value, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });
      sum_dy[j] = value[0];
      sum_dy_xmu[j] = value[1];

      item.barrier(sycl_local_fence);
    }

    if constexpr (two_pass_reduce) {
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        int c_offset = c_offset_base + j;
        if (lidy == 0 && (c_offset) < stride_) {
          // write to temp[c][group_idx_y]
          int offset = c_offset * group_num_y_ + group_idx_y;
          temp_sum_dy_ptr_[offset] = sum_dy[j];
          temp_sum_dy_xmu_ptr_[offset] = sum_dy_xmu[j];
        }
      }

    } else {
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        int c_offset = c_offset_base + j;
        if (lidy == 0 && c_offset < stride_) {
          if (grad_bias_ != nullptr) {
            grad_bias_[c_offset] = static_cast<layerscalar_t>(sum_dy[j]);
          }
          if (grad_weight_ != nullptr) {
            grad_weight_[c_offset] =
                static_cast<layerscalar_t>(sum_dy_xmu[j] * factor[j]);
          }
          // mean_dy[c_offset] = sum_dy_th / reduction_size_;
          // mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size_;
          sum_dy_o_[c_offset] = sum_dy[j];
          sum_dy_xmu_o_[c_offset] = sum_dy_xmu[j];
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<vec_y, 1>(sycl::range<1>{(size_t)wg_size_}, cgh);
  }

  BatchNormBackwardReduceChannelsLastKernelFunctor(
      scalar_t* grad_output,
      accscalar_t* inv_std,
      layerscalar_t* grad_weight,
      layerscalar_t* grad_bias,
      const int reduction_size,
      const int stride,
      int loop_count,
      int group_num_y,
      int group_num_x,
      int global_range_y,
      int local_range_y,
      int global_range_x,
      int local_range_x,
      int wg_size,
      accscalar_t* sum_dy_o,
      accscalar_t* sum_dy_xmu_o,
      accscalar_t* temp_sum_dy_ptr,
      accscalar_t* temp_sum_dy_xmu_ptr,
      int total_count,
      scalar_t* input_ptr,
      accscalar_t* mean_ptr)
      : grad_output_(grad_output),
        inv_std_(inv_std),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        reduction_size_(reduction_size),
        stride_(stride),
        loop_count_(loop_count),
        group_num_y_(group_num_y),
        group_num_x_(group_num_x),
        global_range_y_(global_range_y),
        local_range_y_(local_range_y),
        global_range_x_(global_range_x),
        local_range_x_(local_range_x),
        wg_size_(wg_size),
        sum_dy_o_(sum_dy_o),
        sum_dy_xmu_o_(sum_dy_xmu_o),
        temp_sum_dy_ptr_(temp_sum_dy_ptr),
        temp_sum_dy_xmu_ptr_(temp_sum_dy_xmu_ptr),
        total_count_(total_count),
        input_ptr_(input_ptr),
        mean_ptr_(mean_ptr) {}

 private:
  scalar_t* grad_output_;
  accscalar_t* inv_std_;
  layerscalar_t* grad_weight_;
  layerscalar_t* grad_bias_;
  const int reduction_size_;
  const int stride_;
  int loop_count_;
  int group_num_y_;
  int group_num_x_;
  int global_range_y_;
  int local_range_y_;
  int global_range_x_;
  int local_range_x_;
  int wg_size_;
  accscalar_t* sum_dy_o_;
  accscalar_t* sum_dy_xmu_o_;
  accscalar_t* temp_sum_dy_ptr_;
  accscalar_t* temp_sum_dy_xmu_ptr_;
  int total_count_;
  scalar_t* input_ptr_;
  accscalar_t* mean_ptr_;
  sycl_local_acc_t<vec_y, 1> shared_;
};

template <typename accscalar_t, typename layerscalar_t>
struct BatchNormBackwardReduceChannelsLastTwoPassKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto local_id = item.get_local_linear_id();
    // auto global_id = item.get_global_linear_id();
    auto c_offset = item.get_group_linear_id();
    accscalar_t r_mean = 0.0f;
    accscalar_t factor = 1.0f;
    if (c_offset < stride_) {
      r_mean = mean_ptr_[c_offset];
      factor = inv_std_[c_offset];
    }

    accscalar_t temp_sum_dy = 0.0f;
    accscalar_t temp_sum_dy_xmu = 0.0f;
    for (int i = local_id; i < group_num_y_; i += wg_size_) {
      int offset = c_offset * group_num_y_ + i;
      temp_sum_dy += temp_sum_dy_ptr_[offset];
      temp_sum_dy_xmu += temp_sum_dy_xmu_ptr_[offset];
    }
    auto total_sum_dy = sycl::reduce_over_group(
        item.get_group(), temp_sum_dy, sycl::plus<accscalar_t>());
    auto total_sum_dy_xmu = sycl::reduce_over_group(
        item.get_group(), temp_sum_dy_xmu, sycl::plus<accscalar_t>());
    if (local_id == 0) {
      sum_dy_o_[c_offset] = total_sum_dy;
      sum_dy_xmu_o_[c_offset] = total_sum_dy_xmu;
      if (grad_bias_ != nullptr) {
        grad_bias_[c_offset] = static_cast<layerscalar_t>(total_sum_dy);
      }
      if (grad_weight_ != nullptr) {
        grad_weight_[c_offset] =
            static_cast<layerscalar_t>(total_sum_dy_xmu * factor);
      }
    }
  }
  BatchNormBackwardReduceChannelsLastTwoPassKernelFunctor(
      accscalar_t* inv_std,
      layerscalar_t* grad_weight,
      layerscalar_t* grad_bias,
      const int stride,
      int group_num_y,
      int wg_size,
      accscalar_t* sum_dy_o,
      accscalar_t* sum_dy_xmu_o,
      accscalar_t* temp_sum_dy_ptr,
      accscalar_t* temp_sum_dy_xmu_ptr,
      accscalar_t* mean_ptr)
      : inv_std_(inv_std),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        stride_(stride),
        group_num_y_(group_num_y),
        wg_size_(wg_size),
        sum_dy_o_(sum_dy_o),
        sum_dy_xmu_o_(sum_dy_xmu_o),
        temp_sum_dy_ptr_(temp_sum_dy_ptr),
        temp_sum_dy_xmu_ptr_(temp_sum_dy_xmu_ptr),
        mean_ptr_(mean_ptr) {}

 private:
  accscalar_t* inv_std_;
  layerscalar_t* grad_weight_;
  layerscalar_t* grad_bias_;
  const int stride_;
  int group_num_y_;
  int wg_size_;
  accscalar_t* sum_dy_o_;
  accscalar_t* sum_dy_xmu_o_;
  accscalar_t* temp_sum_dy_ptr_;
  accscalar_t* temp_sum_dy_xmu_ptr_;
  accscalar_t* mean_ptr_;
};

template <
    int vec_size,
    bool two_pass_reduce,
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t>
void batch_norm_backward_reduce_channels_last_kernel(
    const Tensor& input,
    scalar_t* grad_output,
    const Tensor& mean,
    accscalar_t* inv_std,
    Tensor& sum_dy,
    Tensor& sum_dy_xmu,
    layerscalar_t* grad_weight,
    layerscalar_t* grad_bias,
    const int reduction_size,
    const int stride) {
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size);
  int loop_count = 1 + (reduction_size - 1) / (global_range[0]);
  int group_num_y = global_range[0] / local_range[0];
  int group_num_x = global_range[1] / local_range[1];
  int global_range_y = global_range[0];
  int local_range_y = local_range[0];
  int global_range_x = global_range[1];
  int local_range_x = local_range[1];
  auto wg_size = local_range[0] * local_range[1];
  accscalar_t* sum_dy_o = sum_dy.data_ptr<accscalar_t>();
  accscalar_t* sum_dy_xmu_o = sum_dy_xmu.data_ptr<accscalar_t>();
  Tensor temp_sum_dy;
  Tensor temp_sum_dy_xmu;
  accscalar_t* temp_sum_dy_ptr;
  accscalar_t* temp_sum_dy_xmu_ptr;

  if constexpr (two_pass_reduce) {
    temp_sum_dy = at::empty({group_num_y * stride}, sum_dy.options());
    temp_sum_dy_xmu = at::empty({group_num_y * stride}, sum_dy_xmu.options());
    temp_sum_dy_ptr = temp_sum_dy.data_ptr<accscalar_t>();
    temp_sum_dy_xmu_ptr = temp_sum_dy_xmu.data_ptr<accscalar_t>();
  }
  auto& queue = getCurrentSYCLQueue();
  int total_count = stride * reduction_size;

  scalar_t* input_ptr = input.data_ptr<scalar_t>();
  accscalar_t* mean_ptr = mean.data_ptr<accscalar_t>();
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  using vec_y = at::detail::Array<accscalar_t, 2>;

  auto caller = BatchNormBackwardReduceChannelsLastKernelFunctor<
      vec_size,
      two_pass_reduce,
      scalar_t,
      accscalar_t,
      layerscalar_t,
      vec_t,
      vec_y>(
      grad_output,
      inv_std,
      grad_weight,
      grad_bias,
      reduction_size,
      stride,
      loop_count,
      group_num_y,
      group_num_x,
      global_range_y,
      local_range_y,
      global_range_x,
      local_range_x,
      wg_size,
      sum_dy_o,
      sum_dy_xmu_o,
      temp_sum_dy_ptr,
      temp_sum_dy_xmu_ptr,
      total_count,
      input_ptr,
      mean_ptr);

  sycl_kernel_submit(global_range, local_range, queue, caller);

  // reduce temp sum
  if constexpr (two_pass_reduce) {
    int wg_size = std::min(group_num_y, int(syclMaxWorkItemsPerEU()));
    auto caller = BatchNormBackwardReduceChannelsLastTwoPassKernelFunctor<
        accscalar_t,
        layerscalar_t>(
        inv_std,
        grad_weight,
        grad_bias,
        stride,
        group_num_y,
        wg_size,
        sum_dy_o,
        sum_dy_xmu_o,
        temp_sum_dy_ptr,
        temp_sum_dy_xmu_ptr,
        mean_ptr);
    sycl_kernel_submit(stride * wg_size, wg_size, queue, caller);
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
batch_norm_backward_reduce_channels_last_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const bool input_g,
    const bool weight_g,
    const bool bias_g) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::Tensor sumn_dy = at::zeros({stride}, mean.options());
  at::Tensor sum_dy_xmu = at::zeros({stride}, mean.options());

  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight.defined()) {
    grad_weight = at::zeros({stride}, weight.options());
    grad_bias = at::zeros({stride}, weight.options());
  } else {
    // because I cannot return an uninitialized at::Tensor
    grad_weight = at::empty({0}, mean.options());
    grad_bias = at::empty({0}, mean.options());
  }

#define DISPATCH_BWD_REDUCE_ACC_IMPL(vec_size, two_pass)               \
  batch_norm_backward_reduce_channels_last_kernel<vec_size, two_pass>( \
      input,                                                           \
      grad_output.data_ptr<scalar_t>(),                                \
      mean,                                                            \
      inv_std.data_ptr<accscalar_t>(),                                 \
      sumn_dy,                                                         \
      sum_dy_xmu,                                                      \
      grad_weight.data_ptr<accscalar_t>(),                             \
      grad_bias.data_ptr<accscalar_t>(),                               \
      reduction_size,                                                  \
      stride);

#define DISPATCH_BWD_REDUCE_IMPL(vec_size, two_pass)                   \
  batch_norm_backward_reduce_channels_last_kernel<vec_size, two_pass>( \
      input,                                                           \
      grad_output.data_ptr<scalar_t>(),                                \
      mean,                                                            \
      inv_std.data_ptr<accscalar_t>(),                                 \
      sumn_dy,                                                         \
      sum_dy_xmu,                                                      \
      weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr,   \
      weight.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr,     \
      reduction_size,                                                  \
      stride);

  sycl::range<2> global_range(1, 1), local_range(1, 1);
  if (weight.defined() && input.scalar_type() != weight.scalar_type()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);

          switch (suggest_vec_size) {
            case 8: {
              constexpr int vec_size = 8;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_ACC_IMPL(8, true);
              } else {
                DISPATCH_BWD_REDUCE_ACC_IMPL(8, false);
              }
              break;
            }
            case 4: {
              constexpr int vec_size = 4;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_ACC_IMPL(4, true);
              } else {
                DISPATCH_BWD_REDUCE_ACC_IMPL(4, false);
              }
              break;
            }
            default: {
              constexpr int vec_size = 1;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_ACC_IMPL(1, true);
              } else {
                DISPATCH_BWD_REDUCE_ACC_IMPL(1, false);
              }
            }
          }
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_backward_reduce: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_reduce",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              constexpr int vec_size = 8;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_IMPL(8, true);
              } else {
                DISPATCH_BWD_REDUCE_IMPL(8, false);
              }
              break;
            }
            case 4: {
              constexpr int vec_size = 4;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_IMPL(4, true);
              } else {
                DISPATCH_BWD_REDUCE_IMPL(4, false);
              }
              break;
            }
            default: {
              constexpr int vec_size = 1;
              std::tie(global_range, local_range) = flexible_launch_configs(
                  reduction_size, stride, false, vec_size);
              int group_num_y = global_range[0] / local_range[0];
              if (group_num_y > 1) {
                DISPATCH_BWD_REDUCE_IMPL(1, true);
              } else {
                DISPATCH_BWD_REDUCE_IMPL(1, false);
              }
            }
          }
        });
  }
  return std::make_tuple(sumn_dy, sum_dy_xmu, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (canUse32BitIndexMath(grad_output) &&
      batch_norm_use_channels_last_kernels(grad_output) &&
      batch_norm_use_channels_last_kernels(input) &&
      (!weight.defined() || weight.is_contiguous()) && mean.is_contiguous() &&
      invstd.is_contiguous()) {
    return batch_norm_backward_reduce_channels_last_template(
        grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
  }
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_output.scalar_type(),
      "batch_norm_backward_reduce",
      [&] {
        auto mean_st = mean.dtype();
        auto invstd_st = invstd.dtype();
        TORCH_CHECK(
            mean_st == invstd_st,
            "mean and invstd need to have the same data types");
        const bool mixed_type = is_mixed_type(input, weight);
        using accscalar_t = acc_type<scalar_t, true>;

        if (canUse32BitIndexMath(grad_output)) {
          if (mixed_type) {
            return batch_norm_backward_reduce_template<
                scalar_t,
                accscalar_t,
                int32_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          } else {
            return batch_norm_backward_reduce_template<
                scalar_t,
                scalar_t,
                int32_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          }
        } else {
          if (mixed_type) {
            return batch_norm_backward_reduce_template<
                scalar_t,
                accscalar_t,
                int64_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          } else {
            return batch_norm_backward_reduce_template<
                scalar_t,
                scalar_t,
                int64_t>(
                grad_output,
                input,
                mean,
                invstd,
                weight,
                input_g,
                weight_g,
                bias_g);
          }
        }
      });
}

// ====================== batch_norm_backward_elemt ======================

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t,
    bool has_count = false>
struct BatchNormBackwardElemtKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    index_t plane = item.get_group(1);

    if (plane >= numPlane_) {
      return;
    }

    // Use float to calculate to avoid double issues in ATSM
    auto norm_fct = static_cast<stat_accscalar_t>(
        static_cast<float>(1.0) / reduction_size_);
    if constexpr (has_count) {
      int64_t total_numel = 0;
      for (int i = 0; i < world_size_; i++) {
        total_numel += numel_[i];
      }
      norm_fct =
          static_cast<stat_accscalar_t>(static_cast<float>(1.0) / total_numel);
    }

    stat_accscalar_t m_c = mean_ptr_[plane];
    stat_accscalar_t m_dy_c = sum_dy_ptr_[plane] * norm_fct;
    stat_accscalar_t factor_1_c = invstd_ptr_[plane];
    stat_accscalar_t factor_2_c = weight_ptr_ != nullptr
        ? static_cast<stat_accscalar_t>(weight_ptr_[plane])
        : stat_accscalar_t(1);
    factor_2_c *= factor_1_c;
    factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu_ptr_[plane] * norm_fct;

    index_t bstep = global_range_y_;
    for (index_t batch = item.get_global_id(0); batch < N_; batch += bstep) {
      auto g_i_offset = batch * gi_batch_stride_ + plane * gi_Hw_;
      auto g_o_offset = batch * go_batch_stride_ + plane * go_Hw_;
      auto i_offset = batch * i_batch_stride_ + plane * Hw_;
      for (index_t feature = item.get_local_id(1); feature < Hw_;
           feature += local_range_x_) {
        grad_input_ptr_[g_i_offset + feature] = static_cast<input_scalar_t>(
            (grad_output_ptr_[g_o_offset + feature] - m_dy_c -
             (input_ptr_[i_offset + feature] - m_c) * factor_1_c) *
            factor_2_c);
      }
    }
  }
  BatchNormBackwardElemtKernelFunctor(
      const int* numel,
      const int world_size,
      int Hw,
      int N,
      int numPlane,
      int weight_size,
      int64_t target_tile_size,
      int64_t wg_size,
      int i_batch_stride,
      int gi_batch_stride,
      int go_batch_stride,
      int gi_Hw,
      int go_Hw,
      int tf,
      int tb,
      int global_range_y,
      int local_range_x,
      input_scalar_t* input_ptr,
      input_scalar_t* grad_output_ptr,
      input_scalar_t* grad_input_ptr,
      stat_accscalar_t* mean_ptr,
      stat_accscalar_t* invstd_ptr,
      stat_scalar_t* weight_ptr,
      stat_accscalar_t* sum_dy_ptr,
      stat_accscalar_t* sum_dy_xmu_ptr,
      int n_input,
      int reduction_size)
      : numel_(numel),
        world_size_(world_size),
        Hw_(Hw),
        N_(N),
        numPlane_(numPlane),
        weight_size_(weight_size),
        target_tile_size_(target_tile_size),
        wg_size_(wg_size),
        i_batch_stride_(i_batch_stride),
        gi_batch_stride_(gi_batch_stride),
        go_batch_stride_(go_batch_stride),
        gi_Hw_(gi_Hw),
        go_Hw_(go_Hw),
        tf_(tf),
        tb_(tb),
        global_range_y_(global_range_y),
        local_range_x_(local_range_x),
        input_ptr_(input_ptr),
        grad_output_ptr_(grad_output_ptr),
        grad_input_ptr_(grad_input_ptr),
        mean_ptr_(mean_ptr),
        invstd_ptr_(invstd_ptr),
        weight_ptr_(weight_ptr),
        sum_dy_ptr_(sum_dy_ptr),
        sum_dy_xmu_ptr_(sum_dy_xmu_ptr),
        n_input_(n_input),
        reduction_size_(reduction_size) {}

 private:
  const int* numel_;
  const int world_size_;
  int Hw_;
  int N_;
  int numPlane_;
  int weight_size_;
  int64_t target_tile_size_;
  int64_t wg_size_;
  int i_batch_stride_;
  int gi_batch_stride_;
  int go_batch_stride_;
  int gi_Hw_;
  int go_Hw_;
  int tf_;
  int tb_;
  int global_range_y_;
  int local_range_x_;
  input_scalar_t* input_ptr_;
  input_scalar_t* grad_output_ptr_;
  input_scalar_t* grad_input_ptr_;
  stat_accscalar_t* mean_ptr_;
  stat_accscalar_t* invstd_ptr_;
  stat_scalar_t* weight_ptr_;
  stat_accscalar_t* sum_dy_ptr_;
  stat_accscalar_t* sum_dy_xmu_ptr_;
  int n_input_;
  int reduction_size_;
};

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t,
    bool has_count = false>
void batch_norm_backward_elemt_kernel_impl(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const int* numel,
    const int world_size,
    Tensor& grad_input) {
  auto Hw = input.size(2);
  auto N = input.size(0);
  auto numPlane = input.size(1);
  int weight_size = weight.size(0);

  auto& queue = getCurrentSYCLQueue();
  int64_t target_tile_size = syclMaxWorkItemsPerTile();
  int64_t wg_size = syclMaxWorkItemsPerEU(); // for work group barrier
  if (wg_size * numPlane < target_tile_size) {
    wg_size = syclMaxWorkGroupSize(); // for higher occupancy
  }
  auto i_batch_stride = input.size(1) * input.size(2);
  auto gi_batch_stride = grad_input.size(1) * grad_input.size(2);
  auto go_batch_stride = grad_output.size(1) * grad_output.size(2);
  auto gi_Hw = grad_input.size(2);
  auto go_Hw = grad_output.size(2);

  int tf = get_num_threads(Hw, wg_size);
  int tb = std::max<int>(wg_size / tf, 1);
  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range((N + tb - 1) / tb * tb, tf * numPlane);

  int global_range_y = global_range[0];
  int local_range_x = local_range[1];
  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto grad_output_ptr = grad_output.data_ptr<input_scalar_t>();
  auto grad_input_ptr = grad_input.data_ptr<input_scalar_t>();

  auto mean_ptr = mean.data_ptr<stat_accscalar_t>();
  auto invstd_ptr = invstd.data_ptr<stat_accscalar_t>();
  auto weight_ptr =
      weight.defined() ? weight.data_ptr<stat_scalar_t>() : nullptr;
  auto sum_dy_ptr = sum_dy.data_ptr<stat_accscalar_t>();
  auto sum_dy_xmu_ptr = sum_dy_xmu.data_ptr<stat_accscalar_t>();

  auto n_input = input.size(1);
  auto reduction_size = input.numel() / n_input;

  auto caller = BatchNormBackwardElemtKernelFunctor<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t,
      has_count>(
      numel,
      world_size,
      Hw,
      N,
      numPlane,
      weight_size,
      target_tile_size,
      wg_size,
      i_batch_stride,
      gi_batch_stride,
      go_batch_stride,
      gi_Hw,
      go_Hw,
      tf,
      tb,
      global_range_y,
      local_range_x,
      input_ptr,
      grad_output_ptr,
      grad_input_ptr,
      mean_ptr,
      invstd_ptr,
      weight_ptr,
      sum_dy_ptr,
      sum_dy_xmu_ptr,
      n_input,
      reduction_size);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    typename vec_t,
    typename vec_s_t,
    int vec_size = 1,
    bool has_count = false>
struct BatchNormBackwardElemtChannelsLastKernelImplFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // tensor dimension (m,c)
    // loop along m dimension
    int inner_loop_stride = item.get_global_range(0);

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    if (c_offset_base >= stride_ || m_offset >= reduction_size_) {
      return;
    }

    vec_s_t m_c = *(reinterpret_cast<vec_s_t*>(mean_ + c_offset_base));
    vec_s_t m_dy_c = *(reinterpret_cast<vec_s_t*>(sum_dy_ + c_offset_base));
    vec_s_t sum_dy_xmu_vec =
        *(reinterpret_cast<vec_s_t*>(sum_dyxmu_ + c_offset_base));
    vec_s_t factor_1_c =
        *(reinterpret_cast<vec_s_t*>(inv_std_ + c_offset_base));
    vec_s_t factor_2_c;

    // Use float to calculate to avoid double issues in ATSM
    auto norm_fct =
        static_cast<accscalar_t>(static_cast<float>(1.0) / reduction_size_);
    if constexpr (has_count) {
      int64_t total_numel = 0;
      for (int i = 0; i < world_size_; i++) {
        total_numel += numel_[i];
      }
      norm_fct =
          static_cast<accscalar_t>(static_cast<float>(1.0) / total_numel);
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      if (weight_ != nullptr) {
        factor_2_c[j] = static_cast<accscalar_t>(weight_[c_offset_base + j]) *
            factor_1_c[j];
      } else {
        factor_2_c[j] = accscalar_t(1.0f);
      }
      m_dy_c[j] = m_dy_c[j] * norm_fct;
      factor_1_c[j] =
          factor_1_c[j] * factor_1_c[j] * sum_dy_xmu_vec[j] * norm_fct;
    }

    int address_base = m_offset * stride_ + c_offset_base;
    int address_increment = item.get_global_range(0) * stride_;

    for (int m_offset_loop = item.get_global_id(0);
         m_offset_loop < reduction_size_;
         m_offset_loop += inner_loop_stride) {
      vec_t input_vec = *(reinterpret_cast<vec_t*>(input_ + address_base));
      vec_t grad_output_vec =
          *(reinterpret_cast<vec_t*>(grad_output_ + address_base));
      vec_t output_vec;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        output_vec[j] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(grad_output_vec[j]) - m_dy_c[j] -
             (static_cast<accscalar_t>(input_vec[j]) - m_c[j]) *
                 factor_1_c[j]) *
            factor_2_c[j]);
      }
      *(reinterpret_cast<vec_t*>(grad_input_ + address_base)) = output_vec;
      address_base += address_increment;
    }
  }
  BatchNormBackwardElemtChannelsLastKernelImplFunctor(
      scalar_t* grad_output,
      scalar_t* input,
      accscalar_t* mean,
      accscalar_t* inv_std,
      const layerscalar_t* weight,
      accscalar_t* sum_dy,
      accscalar_t* sum_dyxmu,
      const int* numel,
      scalar_t* grad_input,
      const int world_size,
      const int reduction_size,
      const int stride)
      : grad_output_(grad_output),
        input_(input),
        mean_(mean),
        inv_std_(inv_std),
        weight_(weight),
        sum_dy_(sum_dy),
        sum_dyxmu_(sum_dyxmu),
        numel_(numel),
        grad_input_(grad_input),
        world_size_(world_size),
        reduction_size_(reduction_size),
        stride_(stride) {}

 private:
  scalar_t* grad_output_;
  scalar_t* input_;
  accscalar_t* mean_;
  accscalar_t* inv_std_;
  const layerscalar_t* weight_;
  accscalar_t* sum_dy_;
  accscalar_t* sum_dyxmu_;
  const int* numel_;
  scalar_t* grad_input_;
  const int world_size_;
  const int reduction_size_;
  const int stride_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size = 1,
    bool has_count = false>
void batch_norm_backward_elemt_channels_last_kernel_impl(
    scalar_t* grad_output,
    scalar_t* input,
    accscalar_t* mean,
    accscalar_t* inv_std,
    const layerscalar_t* weight,
    accscalar_t* sum_dy,
    accscalar_t* sum_dy_xmu,
    const int* numel,
    scalar_t* grad_input,
    const int world_size,
    const int reduction_size,
    const int stride) {
  auto& queue = getCurrentSYCLQueue();
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size);
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  using vec_s_t = memory::aligned_vector<accscalar_t, vec_size>;
  auto caller = BatchNormBackwardElemtChannelsLastKernelImplFunctor<
      scalar_t,
      accscalar_t,
      layerscalar_t,
      vec_t,
      vec_s_t,
      vec_size,
      has_count>(
      grad_output,
      input,
      mean,
      inv_std,
      weight,
      sum_dy,
      sum_dy_xmu,
      numel,
      grad_input,
      world_size,
      reduction_size,
      stride);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

at::Tensor batch_norm_backward_elemt_channels_last_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  // Input is guarunteed to be channels-last compatible
  at::Tensor grad_input = at::empty_like(input);

  if (weight.defined() && weight.scalar_type() != input.scalar_type()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "batchnorm_backward_element",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          batch_norm_backward_elemt_channels_last_kernel_impl<
              scalar_t,
              accscalar_t,
              accscalar_t,
              1,
              true>(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.data_ptr<accscalar_t>(),
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              count.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              count.numel(),
              reduction_size,
              stride);
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_backward_element: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "batchnorm_backward_element",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          batch_norm_backward_elemt_channels_last_kernel_impl<
              scalar_t,
              accscalar_t,
              scalar_t,
              1,
              true>(
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              mean.data_ptr<accscalar_t>(),
              inv_std.data_ptr<accscalar_t>(),
              weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
              sum_dy.data_ptr<accscalar_t>(),
              sum_dy_xmu.data_ptr<accscalar_t>(),
              count.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              count.numel(),
              reduction_size,
              stride);
        });
  }

  return grad_input;
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& weight_,
    const Tensor& sum_dy_,
    const Tensor& sum_dy_xmu_,
    const Tensor& count) {
  using stat_accscalar_t = acc_type<stat_scalar_t, true>;
  // int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped =
      at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  batch_norm_backward_elemt_kernel_impl<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t,
      true>(
      input_reshaped,
      grad_output_reshaped,
      mean_,
      invstd_,
      weight_,
      sum_dy_,
      sum_dy_xmu_,
      count.data_ptr<int>(),
      count.numel(),
      grad_input_reshaped);

  return grad_input_reshaped.view(input_.sizes());
}

Tensor batch_norm_backward_elemt_kernel(
    const Tensor& self,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const Tensor& count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (canUse32BitIndexMath(self) &&
      batch_norm_use_channels_last_kernels(self) &&
      batch_norm_use_channels_last_kernels(input)) {
    return batch_norm_backward_elemt_channels_last_template(
        self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_backward_elemt",
      [&] {
        auto mean_st = mean.dtype();
        auto invstd_st = invstd.dtype();
        TORCH_CHECK(
            mean_st == invstd_st,
            "mean and invstd need to have the same data types");
        bool is_half_float =
            std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
        bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value &&
            mean_st == at::kFloat;
        using accscalar_t = acc_type<scalar_t, true>;
        if (canUse32BitIndexMath(self)) {
          if (is_half_float || is_bfloat16_float) {
            return batch_norm_backward_elemt_template<
                scalar_t,
                accscalar_t,
                int32_t>(
                self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
          } else {
            return batch_norm_backward_elemt_template<
                scalar_t,
                scalar_t,
                int32_t>(
                self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
          }
        } else {
          if (is_half_float || is_bfloat16_float) {
            return batch_norm_backward_elemt_template<
                scalar_t,
                accscalar_t,
                int64_t>(
                self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
          } else {
            return batch_norm_backward_elemt_template<
                scalar_t,
                scalar_t,
                int64_t>(
                self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
          }
        }
      });
}

// ====================== batch_norm_update_stats ======================

template <typename scalar_t, typename acc_t>
struct BatchNormUpdateStatsFunctor {
  std::tuple<scalar_t, scalar_t> operator()(
      acc_t mean,
      acc_t var,
      scalar_t running_mean,
      scalar_t running_var) const {
    const auto unbiased_var = var * bessel_correction_factor;
    return std::tuple<scalar_t, scalar_t>{
        mean * momentum + (1 - momentum) * running_mean,
        unbiased_var * momentum + (1 - momentum) * running_var,
    };
  }

  BatchNormUpdateStatsFunctor(
      const acc_t bessel_correction_factor,
      const acc_t momentum)
      : bessel_correction_factor(bessel_correction_factor),
        momentum(momentum) {}

 private:
  const acc_t bessel_correction_factor;
  const acc_t momentum;
};

void batch_norm_update_stats(
    const Tensor& save_mean,
    const Tensor& save_var,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum_,
    int64_t N) {
  auto iter = TensorIteratorConfig()
                  .add_output(running_mean)
                  .add_output(running_var)
                  .add_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats_xpu",
      [&] {
        using acc_t = acc_type<scalar_t, true>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto momentum = static_cast<acc_t>(momentum_);
        BatchNormUpdateStatsFunctor<scalar_t, acc_t> f(
            bessel_correction_factor, momentum);
        gpu_kernel_multiple_outputs(iter, f);
      });
}

void batch_norm_mean_var(
    const Tensor& self,
    Tensor& save_mean,
    Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is ignored.
  const double dummy_epsilon = 1e-5;
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_xpu", [&] {
            batch_norm_stats_template<scalar_t, int32_t, Var>(
                save_mean, save_var, self, dummy_epsilon);
          });
      return;
    }
    case Impl::ChannelsLast: {
      if ((!save_mean.defined() || save_mean.is_contiguous()) &&
          (!save_var.defined() || save_var.is_contiguous())) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_xpu", [&] {
              batch_norm_stats_channels_last_template<scalar_t, Var>(
                  save_mean, save_var, self, dummy_epsilon);
            });
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector reduce_dims(ndim - 1);
      reduce_dims[0] = 0;
      for (int64_t i = 2; i < ndim; ++i) {
        reduce_dims[i - 1] = i;
      }

      // For some reason this isn't an actual operator but it exists anyway...
      var_mean_out(
          save_var,
          save_mean,
          self,
          /*dims=*/reduce_dims,
          /*unbiased=*/false,
          /*keepdim=*/false);
      return;
    }
  }
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum) {
  c10::MaybeOwned<Tensor> running_mean =
      at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var =
      at::borrow_from_optional_tensor(running_var_opt);

  const int64_t n_input = self.size(1);
  TORCH_CHECK(
      self.numel() != 0,
      "input tensor must have at least one element, but got input_sizes = ",
      self.sizes());

  auto options =
      self.options().dtype(at::toAccumulateType(self.scalar_type(), true));

  auto save_mean = at::empty({n_input}, options);
  auto save_var = at::empty({n_input}, options);

  batch_norm_mean_var(self, save_mean, save_var);
  TORCH_CHECK(running_mean->defined() == running_var->defined());
  if (running_mean->defined()) {
    const int64_t N = self.numel() / save_mean.numel();
    batch_norm_update_stats(
        save_mean, save_var, *running_mean, *running_var, momentum, N);
  }
  return std::tuple<Tensor, Tensor>(save_mean, save_var);
}

// ====================== native_batch_norm ======================

template <typename scalar_t, typename acc_t>
struct BatchNormUpdateStatsAndInvertFunctor {
  std::tuple<scalar_t, scalar_t, acc_t> operator()(
      acc_t mean,
      acc_t var,
      scalar_t running_mean,
      scalar_t running_var) const {
    const auto unbiased_var = var * bessel_correction_factor_;
    return std::tuple<scalar_t, scalar_t, acc_t>{
        mean * momentum_ + (1 - momentum_) * running_mean,
        unbiased_var * momentum_ + (1 - momentum_) * running_var,
        c10::xpu::compat::rsqrt(var + eps_)};
  }

  BatchNormUpdateStatsAndInvertFunctor(
      const acc_t bessel_correction_factor,
      const acc_t eps,
      const acc_t momentum)
      : bessel_correction_factor_(bessel_correction_factor),
        eps_(eps),
        momentum_(momentum) {}

 private:
  const acc_t bessel_correction_factor_;
  const acc_t eps_;
  const acc_t momentum_;
};

void batch_norm_update_stats_and_invert(
    const Tensor& save_mean,
    const Tensor& save_var,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum_,
    double epsilon,
    int64_t N) {
  auto iter = TensorIteratorConfig()
                  .add_output(running_mean)
                  .add_output(running_var)
                  .add_output(save_var)
                  .add_const_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats_and_invert_xpu",
      [&] {
        using acc_t = acc_type<scalar_t, true>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto eps = static_cast<acc_t>(epsilon);
        const auto momentum = static_cast<acc_t>(momentum_);
        BatchNormUpdateStatsAndInvertFunctor<scalar_t, acc_t> f(
            bessel_correction_factor, eps, momentum);
        gpu_kernel_multiple_outputs(iter, f);
      });
}

template <typename scalar_t, typename acc_t>
struct BatchNormCalcInvstdFunctor {
  acc_t operator()(scalar_t var) const {
    return c10::xpu::compat::rsqrt(var + eps_);
  }

  BatchNormCalcInvstdFunctor(acc_t eps) : eps_(eps) {}

 private:
  acc_t eps_;
};

void batch_norm_calc_invstd(
    const Tensor& out_invstd,
    const Tensor& running_var,
    double epsilon) {
  auto iter = TensorIteratorConfig()
                  .add_output(out_invstd)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      running_var.scalar_type(),
      "batch_norm_invert_std_xpu",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        auto eps = static_cast<acc_t>(epsilon);
        BatchNormCalcInvstdFunctor<scalar_t, acc_t> f(eps);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename acc_t>
struct BatchNormElementwiseFunctor {
  scalar_t operator()(
      scalar_t input,
      acc_t weight,
      acc_t bias,
      acc_t mean,
      acc_t invstd) const {
    return ((input - mean) * invstd) * weight + bias;
  }
};

void batch_norm_elementwise(
    const Tensor& out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const Tensor& mean_,
    const Tensor& invstd_) {
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      c10::MaybeOwned<Tensor> weight =
          at::borrow_from_optional_tensor(weight_opt);
      c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
      resize_output(out, self.sizes());
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;
            const bool mixed_type = is_mixed_type(self, *weight, *bias);
            if (mixed_type) {
              batch_norm_elemt_template<scalar_t, accscalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            } else {
              batch_norm_elemt_template<scalar_t, scalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            }
          });
      return;
    }
    case Impl::ChannelsLast: {
      auto weight = at::borrow_from_optional_tensor(weight_opt);
      auto bias = at::borrow_from_optional_tensor(bias_opt);

      if (resize_output_check(out, self.sizes())) {
        resize_impl_xpu_(
            out.unsafeGetTensorImpl(), self.sizes(), self.strides());
      }
      if ((out.strides() == self.strides()) &&
          (!weight->defined() || weight->is_contiguous()) &&
          (!bias->defined() || bias->is_contiguous()) &&
          (!mean_.defined() || mean_.is_contiguous()) &&
          (!invstd_.defined() || invstd_.is_contiguous())) {
        batch_norm_elemt_channels_last_template(
            out, self, *weight, *bias, mean_, invstd_);
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      // Helper to convert 1d tensors to an nd tensor that broadcasts with input
      // All elements go into the channel dimension
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };

      auto weight = weight_opt.has_value() && weight_opt->defined()
          ? as_nd(*weight_opt)
          : at::scalar_tensor(1, mean_.options());
      auto bias = bias_opt.has_value() && bias_opt->defined()
          ? as_nd(*bias_opt)
          : at::scalar_tensor(0, mean_.options());
      auto mean = as_nd(mean_);
      auto invstd = as_nd(invstd_);

      auto iter = TensorIteratorConfig()
                      .add_output(out)
                      .add_input(self)
                      .add_input(weight)
                      .add_input(bias)
                      .add_input(mean)
                      .add_input(invstd)
                      .check_all_same_dtype(false)
                      .promote_inputs_to_common_dtype(false)
                      .build();

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using acc_t = at::acc_type<scalar_t, true>;
            BatchNormElementwiseFunctor<scalar_t, acc_t> f;
            gpu_kernel(iter, f);
          });
      return;
    }
  }
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_out_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd) {
  const bool has_running_mean =
      (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var =
      (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);

  if (train) {
    batch_norm_mean_var(self, save_mean, save_invstd);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      batch_norm_update_stats_and_invert(
          save_mean,
          save_invstd,
          *running_mean_opt,
          *running_var_opt,
          momentum,
          epsilon,
          N);
    } else {
      batch_norm_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    TORCH_CHECK(has_running_mean);
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
    batch_norm_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }

  batch_norm_elementwise(
      output, self, weight_opt, bias_opt, save_mean, save_invstd);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

// ====================== native_batch_norm_bw ======================

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t,
    typename accscalar_t>
struct BatchNormBackwardKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    index_t plane = item.get_group(0);
    auto lix = item.get_local_id(1);
    auto liy = item.get_local_id(0);
    auto local_range_y = item.get_local_range(0);
    auto local_range_x = item.get_local_range(1);

    stat_accscalar_t mean, invstd;
    if (train_) {
      mean = save_mean_[plane];
      invstd = save_invstd_[plane];
    } else {
      mean = static_cast<stat_accscalar_t>(running_mean_[plane]);
      invstd =
          static_cast<stat_accscalar_t>(1) /
          std::sqrt(
              static_cast<stat_accscalar_t>(running_var_[plane]) + epsilon_);
    }

    stat_accscalar_t weight_val = weight_.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight_[plane])
        : stat_accscalar_t(1);
    stat_accscalar_t norm = stat_accscalar_t(1) / numel_;

    // Compute two values across (batch, x/y/z) in one pass:
    // 1. Sum(grad_output)
    // 2. DotProduct(input - mean, grad_output)
    GradOp<
        input_scalar_t,
        stat_accscalar_t,
        GenericPackedTensorAccessor<
            const input_scalar_t,
            3,
            DefaultPtrTraits,
            index_t>>
        g(mean, input_, grad_output_);
    auto res = plane_reduce<SIMD, Float2<input_scalar_t, stat_accscalar_t>>(
        item, g, grad_output_, plane, sg_num_, local_sum_);

    stat_accscalar_t grad_output_sum = res.v1;
    stat_accscalar_t dot_p = res.v2;

    stat_accscalar_t grad_mean = grad_output_sum * norm;
    stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
    stat_accscalar_t grad_scale = invstd * weight_val;

    auto grad_input = grad_input_;

    if (grad_input_.data() != nullptr) {
      for (int batch = liy; batch < N_; batch += local_range_y) {
        for (int x = lix; x < Hw_; x += local_range_x) {
          input_scalar_t go = grad_output_[batch][plane][x];
          if (train_) {
            stat_accscalar_t inp = input_[batch][plane][x];
            stat_accscalar_t proj = (inp - mean) * proj_scale;
            grad_input[batch][plane][x] = static_cast<input_scalar_t>(
                (go - proj - grad_mean) * grad_scale);
          } else {
            grad_input[batch][plane][x] =
                static_cast<input_scalar_t>(go * grad_scale);
          }
        }
      }
    }

    if (grad_weight_.size(0) > 0) {
      if (lix == 0) {
        auto grad_weight = grad_weight_;
        grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
      }
    }

    if (grad_bias_.size(0) > 0) {
      if (lix == 0) {
        auto grad_bias = grad_bias_;
        grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum_ = sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1>(
        sycl::range<1>{(size_t)work_group_size_x_ * work_group_size_y_}, cgh);
  }

  BatchNormBackwardKernelFunctor(
      bool train,
      stat_accscalar_t epsilon,
      int N,
      int numPlane,
      int Hw,
      index_t numel,
      int64_t wg_size,
      int64_t work_group_size_x,
      int64_t work_group_size_y,
      int sg_num,
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> input,
      const GenericPackedTensorAccessor<
          const input_scalar_t,
          3,
          DefaultPtrTraits,
          index_t> grad_output,
      GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
          grad_input,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_weight,
      GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
          grad_bias,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> weight,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> running_mean,
      const GenericPackedTensorAccessor<
          const stat_scalar_t,
          1,
          DefaultPtrTraits,
          index_t> running_var,
      const GenericPackedTensorAccessor<
          const stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> save_mean,
      const GenericPackedTensorAccessor<
          const stat_accscalar_t,
          1,
          DefaultPtrTraits,
          index_t> save_invstd)
      : train_(train),
        epsilon_(epsilon),
        N_(N),
        numPlane_(numPlane),
        Hw_(Hw),
        numel_(numel),
        wg_size_(wg_size),
        work_group_size_x_(work_group_size_x),
        work_group_size_y_(work_group_size_y),
        sg_num_(sg_num),
        input_(input),
        grad_output_(grad_output),
        grad_input_(grad_input),
        grad_weight_(grad_weight),
        grad_bias_(grad_bias),
        weight_(weight),
        running_mean_(running_mean),
        running_var_(running_var),
        save_mean_(save_mean),
        save_invstd_(save_invstd) {}

 private:
  bool train_;
  stat_accscalar_t epsilon_;
  int N_;
  int numPlane_;
  int Hw_;
  index_t numel_;
  int64_t wg_size_;
  int64_t work_group_size_x_;
  int64_t work_group_size_y_;
  int sg_num_;
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      input_;
  const GenericPackedTensorAccessor<
      const input_scalar_t,
      3,
      DefaultPtrTraits,
      index_t>
      grad_output_;
  GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>
      grad_input_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_weight_;
  GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t>
      grad_bias_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      weight_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      running_mean_;
  const GenericPackedTensorAccessor<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      running_var_;
  const GenericPackedTensorAccessor<
      const stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      save_mean_;
  const GenericPackedTensorAccessor<
      const stat_accscalar_t,
      1,
      DefaultPtrTraits,
      index_t>
      save_invstd_;
  sycl_local_acc_t<Float2<input_scalar_t, stat_accscalar_t>, 1> local_sum_;
};

template <
    int SIMD,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_backward_kernel_impl(
    const Tensor& input,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor save_invstd,
    bool train,
    stat_accscalar_t epsilon) {
  using accscalar_t = acc_type<stat_scalar_t, true>;
  auto& queue = getCurrentSYCLQueue();
  auto N = grad_output.size(0);
  auto numPlane = grad_output.size(1);
  auto Hw = grad_output.size(2);
  index_t numel = grad_output.size(0) * grad_output.size(2);

  int64_t wg_size = get_prefer_wg_size(N * Hw, SIMD);

  int64_t work_group_size_x = get_num_threads(Hw, wg_size);
  int64_t work_group_size_y = std::max(int64_t(1), wg_size / work_group_size_x);
  int sg_num = work_group_size_x * work_group_size_y / SIMD;

  auto input_pa =
      get_packed_accessor<const input_scalar_t, 3, DefaultPtrTraits, index_t>(
          input, "input");
  auto grad_output_pa =
      get_packed_accessor<const input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_output, "grad_output");
  auto grad_input_pa =
      packed_accessor_or_dummy<input_scalar_t, 3, DefaultPtrTraits, index_t>(
          grad_input, "grad_input");
  auto weight_pa = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(weight, "weight");
  auto grad_weight_pa =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_weight, "grad_weight");
  auto grad_bias_pa =
      packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(
          grad_bias, "grad_bias");
  auto running_mean_pa = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(running_mean, "running_mean");
  auto running_var_pa = packed_accessor_or_dummy<
      const stat_scalar_t,
      1,
      DefaultPtrTraits,
      index_t>(running_var, "running_var");
  auto save_mean_pa =
      packed_accessor_or_dummy<const accscalar_t, 1, DefaultPtrTraits, index_t>(
          save_mean, "save_mean");
  auto save_invstd_pa =
      packed_accessor_or_dummy<const accscalar_t, 1, DefaultPtrTraits, index_t>(
          save_invstd, "save_invstd");

  auto caller = BatchNormBackwardKernelFunctor<
      SIMD,
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      index_t,
      accscalar_t>(
      train,
      epsilon,
      N,
      numPlane,
      Hw,
      numel,
      wg_size,
      work_group_size_x,
      work_group_size_y,
      sg_num,
      input_pa,
      grad_output_pa,
      grad_input_pa,
      grad_weight_pa,
      grad_bias_pa,
      weight_pa,
      running_mean_pa,
      running_var_pa,
      save_mean_pa,
      save_invstd_pa);

  auto global_range =
      sycl::range<2>(numPlane * work_group_size_y, work_group_size_x);
  auto local_range = sycl::range<2>(work_group_size_y, work_group_size_x);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_template(
    const Tensor& grad_out_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    const Tensor& save_mean_,
    const Tensor& save_invstd_,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  using accscalar_t = acc_type<stat_scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  int simd = get_prefer_simd(
      input_reshaped.size(1), input_reshaped.size(0) * input_reshaped.size(1));
  if (simd == SIMD32) {
    batch_norm_backward_kernel_impl<
        SIMD32,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        grad_input_reshaped,
        grad_weight_,
        grad_bias_,
        weight_,
        running_mean_,
        running_var_,
        save_mean_,
        save_invstd_,
        train,
        epsilon);
  } else {
    batch_norm_backward_kernel_impl<
        SIMD16,
        input_scalar_t,
        stat_scalar_t,
        accscalar_t,
        index_t>(
        input_reshaped,
        grad_output_reshaped,
        grad_input_reshaped,
        grad_weight_,
        grad_bias_,
        weight_,
        running_mean_,
        running_var_,
        save_mean_,
        save_invstd_,
        train,
        epsilon);
  }
  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> save_mean =
      at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd =
      at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean =
      at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var =
      at::borrow_from_optional_tensor(running_var_opt);

  const bool needs_reduction =
      train || grad_input_mask[1] || grad_input_mask[2];

  // Fused reduction & elementwise kernel
  if (needs_reduction && grad_input_mask[0] &&
      !batch_norm_use_channels_last_kernels(input) &&
      canUse32BitIndexMath(input) && canUse32BitIndexMath(grad_out)) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batch_norm_backward_xpu", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          const bool mixed_type =
              is_mixed_type(input, *weight, *running_mean, *running_var);
          if (mixed_type) {
            return batch_norm_backward_template<scalar_t, accscalar_t, int32_t>(
                grad_out,
                input,
                *weight,
                *running_mean,
                *running_var,
                *save_mean,
                *save_invstd,
                train,
                epsilon,
                grad_input_mask);
          } else {
            return batch_norm_backward_template<scalar_t, scalar_t, int32_t>(
                grad_out,
                input,
                *weight,
                *running_mean,
                *running_var,
                *save_mean,
                *save_invstd,
                train,
                epsilon,
                grad_input_mask);
          }
        });
  }

  const auto acc_type = at::toAccumulateType(input.scalar_type(), true);
  Tensor mean;
  TORCH_INTERNAL_ASSERT(
      save_mean->defined(), "save_mean should always be defined\n");
  if (save_mean->numel() != 0) {
    mean = *save_mean;
  } else if (needs_reduction) {
    TORCH_CHECK(!train && running_mean->defined());
    mean = (running_mean->scalar_type() == acc_type)
        ? *running_mean
        : running_mean->to(acc_type);
  }

  Tensor invstd;
  TORCH_INTERNAL_ASSERT(
      save_invstd->defined(), "save_invstd should always be defined\n");
  if (save_invstd->numel() != 0) {
    invstd = *save_invstd;
  } else {
    TORCH_CHECK(!train && running_var->defined());
    auto n_channels = input.sizes()[1];
    invstd = at::empty({n_channels}, input.options().dtype(acc_type));
    batch_norm_calc_invstd(invstd, *running_var, epsilon);
  }

  Tensor sum_dy, sum_dy_xmu, grad_weight, grad_bias;
  if (needs_reduction) {
    std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias) =
        batch_norm_backward_reduce(
            grad_out,
            input,
            mean,
            invstd,
            *weight,
            grad_input_mask[0],
            grad_input_mask[1],
            grad_input_mask[2]);
  }

  Tensor grad_input;
  if (grad_input_mask[0]) {
    if (train) {
      // NOTE: sum_dy and sum_dy_xmy are defined, as train implies
      // needs_reduction grad_input = batch_norm_elementwise_backward_train(
      //     grad_out, input, mean, invstd, *weight, sum_dy, sum_dy_xmu);
    } else {
      // grad_input = batch_norm_elementwise_backward_eval(
      //     grad_out, input, invstd, *weight);
    }
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace xpu
} // namespace native
} // namespace at