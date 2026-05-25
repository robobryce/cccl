// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 *   cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s)
 *   from a sequence of samples data residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cuda/std/__type_traits/is_void.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/dispatch/kernels/kernel_histogram.cuh>
#include <cub/device/dispatch/tuning/tuning_histogram.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/proclaim_return_type.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/transform.h>
#include <cuda/std/__tuple_dir/apply.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
// Maximum number of bins per channel for which we will use a privatized smem strategy.
// On modern GPUs (sm_90+) the per-block SMEM budget is large enough to hold thousands of
// 4-byte counters per active channel, and SMEM atomics are dramatically faster than the
// L2-resident GMEM atomics that would otherwise be used for medium-sized histograms.
// 2048 keeps single-channel medium-bin workloads in SMEM (8 KiB) and stays within budget
// even for 3-active-channel multi-histogram (24 KiB).
static constexpr int max_privatized_smem_bins = 2048;

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename SampleT>
struct DeviceHistogramKernelSource
{
  using TransformsT = detail::histogram::Transforms<LevelT, OffsetT, SampleT>;

  template <typename PolicyT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramInitKernel()
  {
    return &DeviceHistogramInitKernel<PolicyT, NUM_ACTIVE_CHANNELS, CounterT, OffsetT>;
  }

  /// Returns the default histogram sweep kernel that receives pre-initialized decode operators from the host.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernel()
  {
    return &DeviceHistogramSweepKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  /// Returns the device-init histogram sweep kernel that initializes decode operators from level arrays in the kernel.
  template <typename PolicyT,
            int PRIVATIZED_SMEM_BINS,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernelDeviceInit()
  {
    // For DispatchEven, we use the scale transform to convert samples to
    // privatized bins and pass-thru transform to convert privatized bins to
    // output bins, vice verse for byte samples.

    // For DispatchRange, we use the search transform to convert samples to
    // privatized bins and scale transform to convert privatized bins to output bins,
    // vice verse for byte samples.

    using DecodeOpT = ::cuda::std::conditional_t<IsEven,
                                                 typename TransformsT::ScaleTransform,
                                                 typename TransformsT::template SearchTransform<const LevelT*>>;

    using PrivatizedDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, typename TransformsT::PassThruTransform, DecodeOpT>;
    using OutputDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, DecodeOpT, typename TransformsT::PassThruTransform>;

    return &DeviceHistogramSweepDeviceInitKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      FirstLevelArrayT,
      SecondLevelArrayT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT,
      IsEven>;
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t CounterSize()
  {
    return sizeof(CounterT);
  }

  template <typename NumBinsT, typename UpperLevelArrayT, typename LowerLevelArrayT>
  CUB_RUNTIME_FUNCTION static constexpr bool MayOverflow(
    [[maybe_unused]] NumBinsT num_bins,
    [[maybe_unused]] const UpperLevelArrayT& upper_level,
    [[maybe_unused]] const LowerLevelArrayT& lower_level,
    [[maybe_unused]] int channel)
  {
    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    if constexpr (::cuda::std::is_integral_v<CommonT>)
    {
      using IntArithmeticT = typename TransformsT::ScaleTransform::IntArithmeticT;
      return static_cast<IntArithmeticT>(upper_level[channel] - lower_level[channel])
           > (::cuda::std::numeric_limits<IntArithmeticT>::max() / static_cast<IntArithmeticT>(num_bins));
    }
    else
    {
      return false;
    }
  }
};

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int PRIVATIZED_SMEM_BINS,
          bool IsDeviceInit,
          bool IsEven,
          bool IsByteSample,
          typename SampleIteratorT,
          typename CounterT,
          typename FirstLevelArrayT,
          typename SecondLevelArrayT,
          typename OffsetT,
          typename PolicySelector,
          typename KernelSource,
          typename KernelLauncherFactory>
#if _CCCL_HAS_CONCEPTS()
  requires histogram_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  FirstLevelArrayT first_level_array,
  SecondLevelArrayT second_level_array,
  int max_num_output_bins,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const histogram_policy active_policy = policy_selector(cc);

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::stringstream ss;
                 ss << active_policy;
                 _CubLog("Dispatching DeviceHistogram to compute capability %d.%d with tuning: %s\n",
                         cc.major_cap(),
                         cc.minor_cap(),
                         ss.str().c_str());
               }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  const auto init_kernel = kernel_source.template HistogramInitKernel<PolicySelector>();
  auto sweep_kernel      = [&] {
    if constexpr (IsDeviceInit)
    {
      return kernel_source.template HistogramSweepKernelDeviceInit<
             PolicySelector,
             PRIVATIZED_SMEM_BINS,
             FirstLevelArrayT,
             SecondLevelArrayT,
             IsEven,
             IsByteSample>();
    }
    else
    {
      using output_decode_op_t     = typename FirstLevelArrayT::value_type;
      using privatized_decode_op_t = typename SecondLevelArrayT::value_type;
      return kernel_source
        .template HistogramSweepKernel<PolicySelector, PRIVATIZED_SMEM_BINS, privatized_decode_op_t, output_decode_op_t>();
    }
  }();

  const int threads_per_block = active_policy.threads_per_block;
  const int pixels_per_thread = active_policy.pixels_per_thread;

  // Get SM count
  int sm_count;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  // Get SM occupancy for sweep_kernel
  int histogram_sweep_sm_occupancy;
  if (const auto error =
        CubDebug(launcher_factory.MaxSmOccupancy(histogram_sweep_sm_occupancy, sweep_kernel, threads_per_block)))
  {
    return error;
  }

  // Get device occupancy for sweep_kernel
  int histogram_sweep_occupancy = histogram_sweep_sm_occupancy * sm_count;

  // L2-fit cap for the GMEM-privatized path (PRIVATIZED_SMEM_BINS == 0).
  // When the per-block GMEM-privatized histograms collectively exceed the L2 cache,
  // every atomicAdd to a privatized counter risks a DRAM round-trip instead of an
  // L2 hit. Capping the active block count so the working set fits in ~half of L2
  // (leaving room for the streaming input and output) recovers L2-resident atomics.
  // Only applied for the IsEven and single-channel paths, where the privatized bin
  // count is well-behaved (output bin count for Even, 1 channel for the
  // single-channel range path).
  if constexpr (PRIVATIZED_SMEM_BINS == 0)
  {
    int l2_cache_size = 0;
    int device_ordinal_l2 = 0;
    cudaError_t l2_error = cudaGetDevice(&device_ordinal_l2);
    if (l2_error == cudaSuccess)
    {
      l2_error = cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_ordinal_l2);
    }
    if (l2_error == cudaSuccess && l2_cache_size > 0)
    {
      // Compute per-block GMEM-privatized footprint across all active channels.
      ::cuda::std::size_t per_block_bytes = 0;
      for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
      {
        per_block_bytes += static_cast<::cuda::std::size_t>(num_privatized_levels[CHANNEL] - 1)
                         * kernel_source.CounterSize();
      }
      if (per_block_bytes > 0)
      {
        const ::cuda::std::size_t l2_half = static_cast<::cuda::std::size_t>(l2_cache_size) / 2u;
        const ::cuda::std::size_t l2_block_budget_sz = l2_half / per_block_bytes;
        const int l2_block_budget = (l2_block_budget_sz > static_cast<::cuda::std::size_t>(::cuda::std::numeric_limits<int>::max()))
                                    ? ::cuda::std::numeric_limits<int>::max()
                                    : static_cast<int>(l2_block_budget_sz);
        // Floor at sm_count: never go below 1 block per SM, otherwise SM-level
        // parallelism collapses faster than the L2-fit gains can recoup.
        const int min_block_floor = sm_count;
        const int l2_capped       = ::cuda::std::max(l2_block_budget, min_block_floor);
        histogram_sweep_occupancy = ::cuda::std::min(histogram_sweep_occupancy, l2_capped);
      }
    }
  }

  if (num_row_pixels * NUM_CHANNELS == row_stride_samples)
  {
    // Treat as a single linear array of samples
    num_row_pixels *= num_rows;
    num_rows           = 1;
    row_stride_samples = num_row_pixels * NUM_CHANNELS;
  }

  // For the GMEM-privatized path with very-large bin counts (>=512K), cap the
  // grid at approximately sm_count blocks. The workspace and StoreOutput merge
  // cost grow linearly with num_thread_blocks * num_bins; over-subscribing the
  // SM count multiplies that cost without proportionally improving sweep
  // throughput because the bottleneck is GMEM bandwidth, not compute. Capping
  // at sm_count reduces both the per-trial GMEM allocation (saving multi-GB
  // for 2M-bin cases) and the post-sweep merge. Smaller large-bin cases (60K,
  // 16K) still get the full occupancy-driven grid because the per-block
  // privatized histograms are small (<=240KB) and the merge cost is bounded.
  constexpr int large_bin_cap_threshold = 524288;
  if (PRIVATIZED_SMEM_BINS == 0 && num_privatized_levels[0] - 1 >= large_bin_cap_threshold)
  {
    histogram_sweep_occupancy =
      ::cuda::std::min(histogram_sweep_occupancy, sm_count);
  }

  // Get grid dimensions, trying to keep total blocks ~histogram_sweep_occupancy
  int pixels_per_tile = threads_per_block * pixels_per_thread;
  int tiles_per_row   = static_cast<int>(::cuda::ceil_div(num_row_pixels, pixels_per_tile));
  int blocks_per_row  = ::cuda::std::min(histogram_sweep_occupancy, tiles_per_row);
  int blocks_per_col =
    (blocks_per_row > 0)
      ? int(::cuda::std::min(static_cast<OffsetT>(histogram_sweep_occupancy / blocks_per_row), num_rows))
      : 0;
  int num_thread_blocks = blocks_per_row * blocks_per_col;

  dim3 sweep_grid_dims;
  sweep_grid_dims.x = (unsigned int) blocks_per_row;
  sweep_grid_dims.y = (unsigned int) blocks_per_col;
  sweep_grid_dims.z = 1;

  // Temporary storage allocation requirements
  constexpr int NUM_ALLOCATIONS      = NUM_ACTIVE_CHANNELS + 1;
  void* allocations[NUM_ALLOCATIONS] = {};
  size_t allocation_sizes[NUM_ALLOCATIONS];

  for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
  {
    allocation_sizes[CHANNEL] =
      size_t(num_thread_blocks) * (num_privatized_levels[CHANNEL] - 1) * kernel_source.CounterSize();
  }

  allocation_sizes[NUM_ALLOCATIONS - 1] = GridQueue<int>::AllocationSize();

  // Alias the temporary allocations from the single storage blob (or compute the
  // necessary size of the blob)
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    // Return if the caller is simply requesting the size of the storage allocation
    return cudaSuccess;
  }

  // Construct the grid queue descriptor
  GridQueue<int> tile_queue(allocations[NUM_ALLOCATIONS - 1]);

  // Wrap arrays so we can pass them by-value to the kernel
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper;

  auto* typed_allocations = reinterpret_cast<CounterT**>(allocations);
  ::cuda::std::copy(typed_allocations, typed_allocations + NUM_ACTIVE_CHANNELS, d_privatized_histograms_wrapper.begin());

  auto minus_one = ::cuda::proclaim_return_type<int>([](int levels) {
    return levels - 1;
  });
  ::cuda::std::transform(
    num_privatized_levels.begin(), num_privatized_levels.end(), num_privatized_bins_wrapper.begin(), minus_one);
  ::cuda::std::transform(num_output_levels.begin(), num_output_levels.end(), num_output_bins_wrapper.begin(), minus_one);

  constexpr int histogram_init_threads_per_block = 256;
  int histogram_init_grid_dims =
    (max_num_output_bins + histogram_init_threads_per_block - 1) / histogram_init_threads_per_block;

// Log DeviceHistogramInitKernel configuration
#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking DeviceHistogramInitKernel<<<%d, %d, 0, %lld>>>()\n",
          histogram_init_grid_dims,
          histogram_init_threads_per_block,
          (long long) stream);
#endif // CUB_DEBUG_LOG

  // Invoke histogram_init_kernel
  if (const auto error = CubDebug(
        launcher_factory(histogram_init_grid_dims, histogram_init_threads_per_block, 0, stream, true)
          .doit(init_kernel, num_output_bins_wrapper, d_output_histograms, tile_queue)))
  {
    return error;
  }

  // Return if empty problem
  if (blocks_per_row == 0 || blocks_per_col == 0)
  {
    return cudaSuccess;
  }

// Log histogram_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking histogram_sweep_kernel<<<{%d, %d, %d}, %d, 0, %lld>>>(), %d pixels "
          "per thread, %d SM occupancy\n",
          sweep_grid_dims.x,
          sweep_grid_dims.y,
          sweep_grid_dims.z,
          threads_per_block,
          (long long) stream,
          pixels_per_thread,
          histogram_sweep_sm_occupancy);
#endif // CUB_DEBUG_LOG

  if (const auto error = CubDebug(
        launcher_factory(sweep_grid_dims, threads_per_block, 0, stream, true)
          .doit(sweep_kernel,
                d_samples,
                num_output_bins_wrapper,
                num_privatized_bins_wrapper,
                d_output_histograms,
                d_privatized_histograms_wrapper,
                first_level_array,
                second_level_array,
                num_row_pixels,
                num_rows,
                row_stride_samples,
                tiles_per_row,
                tile_queue)))
  {
    return error;
  }

  // Check for failure to launch
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Sync the stream if specified to flush runtime errors
  if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
  {
    return error;
  }

  return cudaSuccess;
}

// -------------------------------------------------------------------------------------------------
// Sort-based histogram path (single-channel, large-bin regime)
// -------------------------------------------------------------------------------------------------
//
// For very-large-bin cases (60K, 2M) the SMEM-privatized strategy is impossible (SMEM doesn't fit)
// and the GMEM-privatized strategy is heavily L2-/atomics-bound. We replace those with a
// device-wide sort + run-length-encode pipeline:
//
//   1. compute_bin_indices_kernel     bin index per sample
//   2. cub::DeviceRadixSort::SortKeys sort bin indices (only log2(num_bins+1) bits)
//   3. cub::DeviceRunLengthEncode     extract (unique_bin, count) pairs
//   4. scatter_counts_kernel          scatter counts into d_output_histograms[0]
//
// Invalid samples (out-of-range or filtered) are written with sentinel value `num_bins` so they
// cluster at the end after sort and are simply ignored by the scatter kernel. The scatter kernel
// also handles "byte-sample" non-passthrough output decode by re-applying output_decode_op to the
// privatized bin id when it differs from the output bin id.

// Minimum number of output bins for which the sort-based path *might* be preferred over the GMEM
// privatized atomic path. Below this threshold the GMEM-atomic strategy still wins because the
// privatized histogram fits in L2 and each sample is a single atomicAdd to a hot bin.
//
// Empirically tuned on H100 across uniform/low-entropy 256M-element runs:
//   16384 bins:   sort path is ~1.3-3x slower across both entropies (net loss).
//   60000 bins:   sort path wins ~3x for uniform entropy, loses ~2.5x for low entropy
//                 (net geomean win ~8%).
//   2097152 bins: sort path wins ~4x for uniform entropy, breaks even for low entropy
//                 (net geomean win ~2x).
// Threshold of 50000 excludes 16384 (consistent loss) and includes 60000 + 2M (net wins).
static constexpr int min_sort_based_output_bins = 50000;

// Decide at runtime whether the sort-based pipeline is preferred for the given
// (num_samples, num_output_bins). The decision rule:
//
//   - num_output_bins must be >= min_sort_based_output_bins (gross filter).
//   - Density = num_samples / num_output_bins. When the histogram is dense (many samples per
//     bin), atomic-baseline does well even at high entropy because hot bins amortize across
//     many atomicAdd ops via L2/L1 coalescing. When density is low (sparse), atomic contention
//     and GMEM-privatized buffer-init cost dominate, and sort wins.
//   - The cutoff comes from on-H100 measurements:
//       2M bins / 256M samples (density 128): sort wins ~2-4x.
//       2M bins /   1M samples (density 0.5): sort wins ~28x (GMEM-privatized has to allocate
//         132 * 8 MB of per-block histograms, which dominates for a tiny input).
//       60K bins / 256M samples (density 4267): sort wins ~3x (uniform) or loses ~2.5x (low).
//       60K bins /   1M samples (density 16):   sort wins for both entropies.
//       16K bins / 256M samples (density 15625): sort always loses.
//       16K bins /   1M samples (density 61):   sort can win, but the absolute time
//         is small either way; falls within noise.
//   - Heuristic: density < 5000 -> sort. The 60K-uniform case with density 4267 is just under
//     the threshold (sort wins). 60K-low-entropy density 4267 is also under (sort loses, but
//     close to a tie). 16K density 15625 is over (use atomic).
template <typename OffsetT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool use_sort_path(OffsetT num_samples, int num_output_bins)
{
  constexpr OffsetT max_density_for_sort = 5000;
  if (num_output_bins < min_sort_based_output_bins)
  {
    return false;
  }
  // num_samples / num_bins < max_density_for_sort  <=>  num_samples < max_density_for_sort * num_bins
  // (avoid divide; bin counts up to ~2M fit in OffsetT * int comfortably for 32-bit OffsetT).
  return num_samples < max_density_for_sort * static_cast<OffsetT>(num_output_bins);
}

// Compute the bin index for each sample using the privatized decode op. Invalid samples
// (filtered by the decode op) get the sentinel value `num_bins`. Output is written to
// d_bin_indices.
//
// Each thread processes ITEMS_PER_THREAD samples, in a striped layout (matches the
// blocksize of consumer kernels). The samples per thread are loaded contiguously
// (block-strided), which lets the compiler vectorize on contiguous pointer iterators.
template <int ItemsPerThread,
          typename SampleIteratorT,
          typename PrivatizedDecodeOpT,
          typename OffsetT,
          typename BinIndexT>
_CCCL_KERNEL_ATTRIBUTES void compute_bin_indices_kernel(
  SampleIteratorT d_samples,
  BinIndexT* d_bin_indices,
  PrivatizedDecodeOpT privatized_decode_op,
  OffsetT num_samples,
  BinIndexT num_bins)
{
  using SampleT = it_value_t<SampleIteratorT>;

  const OffsetT block_offset =
    static_cast<OffsetT>(blockIdx.x) * static_cast<OffsetT>(blockDim.x) * ItemsPerThread;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const OffsetT idx = block_offset + static_cast<OffsetT>(item) * blockDim.x + threadIdx.x;
    if (idx >= num_samples)
    {
      return;
    }
    const SampleT sample = d_samples[idx];
    int bin              = -1;
    privatized_decode_op.template BinSelect<LOAD_DEFAULT>(sample, bin, true);
    // Map invalid samples (bin == -1) to sentinel `num_bins`.
    d_bin_indices[idx] = (bin >= 0 && bin < num_bins) ? static_cast<BinIndexT>(bin) : num_bins;
  }
}

// After sort + RLE, scatter counts into the output histogram. Each thread handles one (bin, count)
// pair. If output_decode_op is not the pass-through (e.g. byte-sample case), apply it to map the
// privatized bin to the output bin. Skips the sentinel bin (= num_privatized_bins).
template <typename CounterT, typename UniqueT, typename CountT, typename OutputDecodeOpT>
_CCCL_KERNEL_ATTRIBUTES void scatter_counts_kernel(
  const UniqueT* d_unique_bins,
  const CountT* d_counts,
  const int* d_num_runs,
  CounterT* d_output_histogram,
  OutputDecodeOpT output_decode_op,
  int num_output_bins,
  int num_privatized_bins)
{
  const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_runs = *d_num_runs;
  if (tid >= num_runs)
  {
    return;
  }

  const UniqueT priv_bin = d_unique_bins[tid];
  // Skip the sentinel bucket of invalid samples.
  if (static_cast<int>(priv_bin) >= num_privatized_bins)
  {
    return;
  }

  int output_bin       = -1;
  const CounterT count = static_cast<CounterT>(d_counts[tid]);
  output_decode_op.template BinSelect<LOAD_DEFAULT>(static_cast<int>(priv_bin), output_bin, count > 0);
  if (output_bin >= 0 && output_bin < num_output_bins)
  {
    // The dispatch_sort_based path is only invoked from the non-byte-sample dispatch_even /
    // dispatch_range branches, where output_decode_op is PassThruTransform and each priv_bin
    // maps to a unique output_bin. RLE produces at most one (bin, count) pair per priv_bin,
    // so no two scatter writes target the same output_bin. Direct store is correct (and the
    // d_output_histogram was zeroed in step 0). For a hypothetical future caller using a
    // SearchTransform output decode with collisions, an atomicAdd would be needed instead.
    d_output_histogram[output_bin] = count;
  }
}

// Inner dispatch templated on the bin-index integral type used to sort. This lets us pick a
// narrower key type (e.g. uint16_t) when num_privatized_bins fits, halving the sort/RLE
// memory bandwidth.
template <typename BinIndexT,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_sort_based_typed(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  CounterT* d_output_histogram,
  PrivatizedDecodeOpT privatized_decode_op,
  OutputDecodeOpT output_decode_op,
  int num_privatized_bins,
  int num_output_bins,
  OffsetT num_samples,
  cudaStream_t stream)
{
  // Sentinel = num_privatized_bins; we sort log2(num_privatized_bins + 1) bits.
  // For num_privatized_bins == 60000, bits = 16; for 2097152, bits = 22.
  int end_bit = 0;
  for (int v = num_privatized_bins; v > 0; v >>= 1)
  {
    ++end_bit;
  }

  // Layout: bin_indices_in[N], bin_indices_out[N], unique_bins[max_unique], counts[max_unique],
  // num_runs[1], sort_temp[?], rle_temp[?]
  const int max_unique           = num_privatized_bins + 1; // +1 for sentinel
  const size_t bin_in_bytes      = sizeof(BinIndexT) * static_cast<size_t>(num_samples);
  const size_t bin_out_bytes     = sizeof(BinIndexT) * static_cast<size_t>(num_samples);
  const size_t unique_bins_bytes = sizeof(BinIndexT) * static_cast<size_t>(max_unique);
  const size_t counts_bytes      = sizeof(int) * static_cast<size_t>(max_unique);
  const size_t num_runs_bytes    = sizeof(int);

  // Query temp storage for sort and RLE. Use the DoubleBuffer overload (is_overwrite_okay=true
  // internally) so CUB doesn't allocate its own extra key buffer; we provide d_bin_in /
  // d_bin_out as the two ping-pong buffers.
  size_t sort_temp_bytes = 0;
  {
    DoubleBuffer<BinIndexT> dummy_keys{nullptr, nullptr};
    if (const auto error = CubDebug(cub::DeviceRadixSort::SortKeys(
          nullptr,
          sort_temp_bytes,
          dummy_keys,
          static_cast<int>(num_samples),
          0,
          end_bit,
          stream)))
    {
      return error;
    }
  }

  size_t rle_temp_bytes = 0;
  if (const auto error = CubDebug(cub::DeviceRunLengthEncode::Encode(
        nullptr,
        rle_temp_bytes,
        static_cast<const BinIndexT*>(nullptr),
        static_cast<BinIndexT*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        num_samples,
        stream)))
  {
    return error;
  }

  constexpr int NUM_ALLOCATIONS = 7;
  void* allocations[NUM_ALLOCATIONS] = {};
  size_t allocation_sizes[NUM_ALLOCATIONS] = {
    bin_in_bytes, bin_out_bytes, unique_bins_bytes, counts_bytes, num_runs_bytes, sort_temp_bytes, rle_temp_bytes};

  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  auto* d_bin_in       = reinterpret_cast<BinIndexT*>(allocations[0]);
  auto* d_bin_out      = reinterpret_cast<BinIndexT*>(allocations[1]);
  auto* d_unique_bins  = reinterpret_cast<BinIndexT*>(allocations[2]);
  auto* d_counts       = reinterpret_cast<int*>(allocations[3]);
  auto* d_num_runs     = reinterpret_cast<int*>(allocations[4]);
  void* d_sort_temp    = allocations[5];
  void* d_rle_temp     = allocations[6];

  // Step 0: zero the output histogram up-front so the device can overlap it with the
  // sort-pipeline kernels below (they write only to scratch buffers, not d_output_histogram).
  if (const auto error = CubDebug(cudaMemsetAsync(
        d_output_histogram, 0, static_cast<size_t>(num_output_bins) * sizeof(CounterT), stream)))
  {
    return error;
  }

  // Step 1: compute bin indices.
  // Each thread handles 8 samples; 256 threads/block * 8 = 2048 samples/block. This amortizes the
  // per-thread overhead (block_offset compute, decode-op state) across more arithmetic.
  constexpr int compute_bins_threads          = 256;
  constexpr int compute_bins_items_per_thread = 8;
  constexpr int compute_bins_tile             = compute_bins_threads * compute_bins_items_per_thread;
  const auto compute_bins_blocks =
    static_cast<unsigned int>((num_samples + compute_bins_tile - 1) / compute_bins_tile);

  THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
    dim3(compute_bins_blocks), dim3(compute_bins_threads), 0, stream)
    .doit(compute_bin_indices_kernel<compute_bins_items_per_thread,
                                     SampleIteratorT,
                                     PrivatizedDecodeOpT,
                                     OffsetT,
                                     BinIndexT>,
          d_samples,
          d_bin_in,
          privatized_decode_op,
          num_samples,
          static_cast<BinIndexT>(num_privatized_bins));

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Step 2: sort bin indices in place via the DoubleBuffer overload.
  DoubleBuffer<BinIndexT> sort_keys_buffer{d_bin_in, d_bin_out};
  if (const auto error = CubDebug(cub::DeviceRadixSort::SortKeys(
        d_sort_temp,
        sort_temp_bytes,
        sort_keys_buffer,
        static_cast<int>(num_samples),
        0,
        end_bit,
        stream)))
  {
    return error;
  }
  BinIndexT* d_sorted = sort_keys_buffer.Current();

  // Step 3: run-length encode.
  if (const auto error = CubDebug(cub::DeviceRunLengthEncode::Encode(
        d_rle_temp,
        rle_temp_bytes,
        d_sorted,
        d_unique_bins,
        d_counts,
        d_num_runs,
        num_samples,
        stream)))
  {
    return error;
  }

  constexpr int scatter_threads = 256;
  // Worst-case num_runs = num_privatized_bins + 1; launch enough blocks for that.
  const auto scatter_blocks =
    static_cast<unsigned int>((max_unique + scatter_threads - 1) / scatter_threads);

  THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
    dim3(scatter_blocks), dim3(scatter_threads), 0, stream)
    .doit(scatter_counts_kernel<CounterT, BinIndexT, int, OutputDecodeOpT>,
          d_unique_bins,
          d_counts,
          d_num_runs,
          d_output_histogram,
          output_decode_op,
          num_output_bins,
          num_privatized_bins);

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
  {
    return error;
  }

  return cudaSuccess;
}

// Outer wrapper that picks the bin-index integral type. uint16_t when num_privatized_bins+1
// fits in 16 bits (e.g. 60000-bin case), else uint32_t. Narrower keys halve the
// sort/RLE memory bandwidth.
template <typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_sort_based(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  CounterT* d_output_histogram,
  PrivatizedDecodeOpT privatized_decode_op,
  OutputDecodeOpT output_decode_op,
  int num_privatized_bins,
  int num_output_bins,
  OffsetT num_samples,
  cudaStream_t stream)
{
  // Need to fit the sentinel = num_privatized_bins, so num_privatized_bins must be < 2^16.
  if (num_privatized_bins < (1 << 16))
  {
    return dispatch_sort_based_typed<::cuda::std::uint16_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histogram,
      privatized_decode_op,
      output_decode_op,
      num_privatized_bins,
      num_output_bins,
      num_samples,
      stream);
  }
  return dispatch_sort_based_typed<::cuda::std::uint32_t>(
    d_temp_storage,
    temp_storage_bytes,
    d_samples,
    d_output_histogram,
    privatized_decode_op,
    output_decode_op,
    num_privatized_bins,
    num_output_bins,
    num_samples,
    stream);
}

// -------------------------------------------------------------------------------------------------
// Sort-based histogram path (multi-channel, large-bin regime)
// -------------------------------------------------------------------------------------------------
//
// Extension of the single-channel sort path to NUM_ACTIVE_CHANNELS > 1. The input is interleaved
// `NUM_CHANNELS` samples per pixel, and we histogram only the first `NUM_ACTIVE_CHANNELS` of them.
// Approach: pack a channel-major key for every active sample so we can run a single fused
// sort+RLE+scatter over all channels' samples at once:
//
//   key = chan * stride_per_channel + bin_id
//
// where `stride_per_channel = num_privatized_bins + 1` (one slot per channel for the per-channel
// sentinel; choosing a uniform stride avoids interleaving sentinels between channels). The total
// number of distinct keys is `NUM_ACTIVE_CHANNELS * stride_per_channel`, with end_bit chosen as
// ceil(log2(NUM_ACTIVE_CHANNELS * stride_per_channel)). Invalid samples in channel `chan` get the
// sentinel `chan * stride_per_channel + num_privatized_bins`, which is filtered by the scatter.

// Compute multi-channel bin indices. For each pixel, emit NUM_ACTIVE_CHANNELS keys (one per active
// channel) packed channel-major. The output buffer must have capacity
// num_pixels * NUM_ACTIVE_CHANNELS keys. Each thread handles ITEMS_PER_THREAD pixels.
//
// Layout: thread `t` of block `b` writes channel `c`'s key for pixel `p = b*BS*IPP + i*BS + t` to
// d_bin_indices[p * NUM_ACTIVE_CHANNELS + c]. (Pixel-major, channel-minor — the radix sort doesn't
// care about layout, but keeping channels next to each other from the same pixel maximises L1/L2
// reuse for the channel-strided sample loads.)
template <int NumActiveChannels,
          int NumChannels,
          int ItemsPerThread,
          typename SampleIteratorT,
          typename PrivatizedDecodeOpT,
          typename OffsetT,
          typename BinIndexT>
_CCCL_KERNEL_ATTRIBUTES void compute_bin_indices_multi_channel_kernel(
  SampleIteratorT d_samples,
  BinIndexT* d_bin_indices,
  ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op,
  OffsetT num_pixels,
  BinIndexT num_priv_bins_per_channel,
  BinIndexT stride_per_channel)
{
  using SampleT = it_value_t<SampleIteratorT>;

  const OffsetT block_offset =
    static_cast<OffsetT>(blockIdx.x) * static_cast<OffsetT>(blockDim.x) * ItemsPerThread;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const OffsetT pixel = block_offset + static_cast<OffsetT>(item) * blockDim.x + threadIdx.x;
    if (pixel >= num_pixels)
    {
      return;
    }

    // Multi-channel: load NumActiveChannels samples from the pixel and emit one key each.
    const OffsetT sample_base = pixel * static_cast<OffsetT>(NumChannels);
    const OffsetT key_base    = pixel * static_cast<OffsetT>(NumActiveChannels);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int c = 0; c < NumActiveChannels; ++c)
    {
      const SampleT sample = d_samples[sample_base + static_cast<OffsetT>(c)];
      int bin              = -1;
      privatized_decode_op[c].template BinSelect<LOAD_DEFAULT>(sample, bin, true);
      // Map invalid samples to the per-channel sentinel `num_priv_bins_per_channel`.
      const BinIndexT bin_in_channel =
        (bin >= 0 && bin < num_priv_bins_per_channel) ? static_cast<BinIndexT>(bin) : num_priv_bins_per_channel;
      d_bin_indices[key_base + static_cast<OffsetT>(c)] =
        static_cast<BinIndexT>(c) * stride_per_channel + bin_in_channel;
    }
  }
}

// After sort + RLE, scatter counts into the per-channel output histograms. Each thread handles one
// (priv_bin, count) pair where priv_bin = chan * stride_per_channel + bin_in_channel. Decode the
// channel and the per-channel bin, skip sentinel slots, and route the count to the correct
// d_output_histograms[chan][output_bin].
template <int NumActiveChannels, typename CounterT, typename UniqueT, typename CountT, typename OutputDecodeOpT>
_CCCL_KERNEL_ATTRIBUTES void scatter_counts_multi_channel_kernel(
  const UniqueT* d_unique_bins,
  const CountT* d_counts,
  const int* d_num_runs,
  ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms,
  ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op,
  ::cuda::std::array<int, NumActiveChannels> num_output_bins,
  int num_priv_bins_per_channel,
  int stride_per_channel)
{
  const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_runs = *d_num_runs;
  if (tid >= num_runs)
  {
    return;
  }

  const int priv_bin       = static_cast<int>(d_unique_bins[tid]);
  const int chan           = priv_bin / stride_per_channel;
  const int bin_in_channel = priv_bin - chan * stride_per_channel;

  // Skip sentinel slots: any bin >= num_priv_bins_per_channel within a channel's stride is the
  // invalid-sample sentinel; chan >= NumActiveChannels can occur only for unused stride slots
  // (which we never produce, but guard anyway).
  if (bin_in_channel >= num_priv_bins_per_channel || chan >= NumActiveChannels)
  {
    return;
  }

  const CounterT count = static_cast<CounterT>(d_counts[tid]);
  // Iterate to find the active channel index (small loop, NumActiveChannels <= 4) so we can
  // dispatch on a compile-time channel and let nvcc hoist the per-channel loads.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int c = 0; c < NumActiveChannels; ++c)
  {
    if (c == chan)
    {
      int output_bin = -1;
      output_decode_op[c].template BinSelect<LOAD_DEFAULT>(bin_in_channel, output_bin, count > 0);
      if (output_bin >= 0 && output_bin < num_output_bins[c])
      {
        // The same direct-store argument as the single-channel scatter applies here:
        // dispatch_sort_based_multi_channel is only invoked from non-byte-sample dispatch_even /
        // dispatch_range branches where the output decode is PassThruTransform. Each priv_bin maps
        // to a unique (chan, output_bin) and RLE produces at most one (priv_bin, count) pair per
        // priv_bin, so no two scatter writes target the same address. The output histograms were
        // memset to zero earlier in the pipeline.
        d_output_histograms[c][output_bin] = count;
      }
    }
  }
}

// Inner multi-channel sort dispatch templated on the bin-index type.
template <int NumActiveChannels,
          int NumChannels,
          typename BinIndexT,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_sort_based_multi_channel_typed(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms,
  ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op,
  ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op,
  int num_priv_bins_per_channel,
  ::cuda::std::array<int, NumActiveChannels> num_output_bins,
  OffsetT num_pixels,
  cudaStream_t stream)
{
  // Use a uniform stride of (num_priv_bins_per_channel + 1) per channel: one slot per channel for
  // the invalid-sample sentinel. The maximum key value is therefore
  // (NumActiveChannels - 1) * stride + num_priv_bins_per_channel.
  const int stride_per_channel = num_priv_bins_per_channel + 1;
  const int max_key_value      = (NumActiveChannels - 1) * stride_per_channel + num_priv_bins_per_channel;

  // end_bit = ceil(log2(max_key_value + 1))
  int end_bit = 0;
  for (int v = max_key_value; v > 0; v >>= 1)
  {
    ++end_bit;
  }

  const OffsetT num_keys         = num_pixels * static_cast<OffsetT>(NumActiveChannels);
  const int max_unique           = NumActiveChannels * stride_per_channel; // upper bound on distinct keys
  const size_t bin_in_bytes      = sizeof(BinIndexT) * static_cast<size_t>(num_keys);
  const size_t bin_out_bytes     = sizeof(BinIndexT) * static_cast<size_t>(num_keys);
  const size_t unique_bins_bytes = sizeof(BinIndexT) * static_cast<size_t>(max_unique);
  const size_t counts_bytes      = sizeof(int) * static_cast<size_t>(max_unique);
  const size_t num_runs_bytes    = sizeof(int);

  // Query temp storage for sort and RLE. DoubleBuffer overload (is_overwrite_okay=true) so CUB
  // doesn't allocate its own extra key buffer.
  size_t sort_temp_bytes = 0;
  {
    DoubleBuffer<BinIndexT> dummy_keys{nullptr, nullptr};
    if (const auto error = CubDebug(cub::DeviceRadixSort::SortKeys(
          nullptr,
          sort_temp_bytes,
          dummy_keys,
          static_cast<int>(num_keys),
          0,
          end_bit,
          stream)))
    {
      return error;
    }
  }

  size_t rle_temp_bytes = 0;
  if (const auto error = CubDebug(cub::DeviceRunLengthEncode::Encode(
        nullptr,
        rle_temp_bytes,
        static_cast<const BinIndexT*>(nullptr),
        static_cast<BinIndexT*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        num_keys,
        stream)))
  {
    return error;
  }

  constexpr int NUM_ALLOCATIONS            = 7;
  void* allocations[NUM_ALLOCATIONS]       = {};
  size_t allocation_sizes[NUM_ALLOCATIONS] = {
    bin_in_bytes, bin_out_bytes, unique_bins_bytes, counts_bytes, num_runs_bytes, sort_temp_bytes, rle_temp_bytes};

  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  auto* d_bin_in      = reinterpret_cast<BinIndexT*>(allocations[0]);
  auto* d_bin_out     = reinterpret_cast<BinIndexT*>(allocations[1]);
  auto* d_unique_bins = reinterpret_cast<BinIndexT*>(allocations[2]);
  auto* d_counts      = reinterpret_cast<int*>(allocations[3]);
  auto* d_num_runs    = reinterpret_cast<int*>(allocations[4]);
  void* d_sort_temp   = allocations[5];
  void* d_rle_temp    = allocations[6];

  // Step 0: zero output histograms in parallel (all small relative to sort cost; one memset
  // per channel keeps the call simple — they go on the same stream so they serialize but
  // are tiny compared to the sort kernels that follow).
  for (int c = 0; c < NumActiveChannels; ++c)
  {
    if (const auto error = CubDebug(cudaMemsetAsync(
          d_output_histograms[c],
          0,
          static_cast<size_t>(num_output_bins[c]) * sizeof(CounterT),
          stream)))
    {
      return error;
    }
  }

  // Step 1: compute bin indices (multi-channel — emits NumActiveChannels keys per pixel).
  constexpr int compute_bins_threads          = 256;
  constexpr int compute_bins_items_per_thread = 8;
  constexpr int compute_bins_tile             = compute_bins_threads * compute_bins_items_per_thread;
  const auto compute_bins_blocks =
    static_cast<unsigned int>((num_pixels + compute_bins_tile - 1) / compute_bins_tile);

  THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
    dim3(compute_bins_blocks), dim3(compute_bins_threads), 0, stream)
    .doit(compute_bin_indices_multi_channel_kernel<NumActiveChannels,
                                                   NumChannels,
                                                   compute_bins_items_per_thread,
                                                   SampleIteratorT,
                                                   PrivatizedDecodeOpT,
                                                   OffsetT,
                                                   BinIndexT>,
          d_samples,
          d_bin_in,
          privatized_decode_op,
          num_pixels,
          static_cast<BinIndexT>(num_priv_bins_per_channel),
          static_cast<BinIndexT>(stride_per_channel));

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Step 2: sort keys via DoubleBuffer.
  DoubleBuffer<BinIndexT> sort_keys_buffer{d_bin_in, d_bin_out};
  if (const auto error = CubDebug(cub::DeviceRadixSort::SortKeys(
        d_sort_temp,
        sort_temp_bytes,
        sort_keys_buffer,
        static_cast<int>(num_keys),
        0,
        end_bit,
        stream)))
  {
    return error;
  }
  BinIndexT* d_sorted = sort_keys_buffer.Current();

  // Step 3: run-length encode.
  if (const auto error = CubDebug(cub::DeviceRunLengthEncode::Encode(
        d_rle_temp,
        rle_temp_bytes,
        d_sorted,
        d_unique_bins,
        d_counts,
        d_num_runs,
        num_keys,
        stream)))
  {
    return error;
  }

  // Step 4: scatter to per-channel histograms.
  constexpr int scatter_threads = 256;
  const auto scatter_blocks =
    static_cast<unsigned int>((max_unique + scatter_threads - 1) / scatter_threads);

  THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
    dim3(scatter_blocks), dim3(scatter_threads), 0, stream)
    .doit(scatter_counts_multi_channel_kernel<NumActiveChannels, CounterT, BinIndexT, int, OutputDecodeOpT>,
          d_unique_bins,
          d_counts,
          d_num_runs,
          d_output_histograms,
          output_decode_op,
          num_output_bins,
          num_priv_bins_per_channel,
          stride_per_channel);

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
  {
    return error;
  }

  return cudaSuccess;
}

// Outer wrapper that picks the bin-index integral type. Use the smallest unsigned type that fits
// `NumActiveChannels * (num_priv_bins_per_channel + 1)`.
template <int NumActiveChannels,
          int NumChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_sort_based_multi_channel(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NumActiveChannels> d_output_histograms,
  ::cuda::std::array<PrivatizedDecodeOpT, NumActiveChannels> privatized_decode_op,
  ::cuda::std::array<OutputDecodeOpT, NumActiveChannels> output_decode_op,
  int num_priv_bins_per_channel,
  ::cuda::std::array<int, NumActiveChannels> num_output_bins,
  OffsetT num_pixels,
  cudaStream_t stream)
{
  // Maximum key value is (NumActiveChannels - 1) * (num_priv_bins + 1) + num_priv_bins. Pick
  // narrowest unsigned that fits that.
  const long long stride            = static_cast<long long>(num_priv_bins_per_channel) + 1;
  const long long max_key_inclusive = static_cast<long long>(NumActiveChannels - 1) * stride + num_priv_bins_per_channel;

  if (max_key_inclusive < (1LL << 16))
  {
    return dispatch_sort_based_multi_channel_typed<NumActiveChannels, NumChannels, ::cuda::std::uint16_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      privatized_decode_op,
      output_decode_op,
      num_priv_bins_per_channel,
      num_output_bins,
      num_pixels,
      stream);
  }
  return dispatch_sort_based_multi_channel_typed<NumActiveChannels, NumChannels, ::cuda::std::uint32_t>(
    d_temp_storage,
    temp_storage_bytes,
    d_samples,
    d_output_histograms,
    privatized_decode_op,
    output_decode_op,
    num_priv_bins_per_channel,
    num_output_bins,
    num_pixels,
    stream);
}

// Decide whether the multi-channel sort path is preferred. Same gating logic as the single-channel
// case (num_output_bins per channel must clear the threshold; density per channel must be low),
// but `num_samples` here is `num_pixels`, not the total interleaved sample count.
template <typename OffsetT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool use_sort_path_multi_channel(OffsetT num_pixels, int num_output_bins_per_channel)
{
  // Each channel's effective sample count is `num_pixels`, since each pixel contributes one
  // sample to that channel's histogram.
  return use_sort_path(num_pixels, num_output_bins_per_channel);
}

// Dispatch routines for device-side decode operator initialization. These differ from the default dispatch routines in
// that they initialize the decode operators inside the kernel from level arrays, instead of initializing them on the
// host, but they are otherwise the same. This is needed for c.parallel, since we cannot instantiate the Transforms
// class on the host, as SampleT and LevelT are type erased. Another change needed is that the level arrays are now
// templates instead of concrete ::cuda::std::array types, since we are passing indirect_args from c.parallel.
//
// Initializing the decode operators inside the kernel results in some regressions (and some performance improvements)
// in the benchmark, which indicates that we need to re-tune the algorithm. This is why we kept the two dispatch paths
// (host init and device init) separate. We should think about merging them back together later on.

/**
 * Dispatch routine for HistogramEven with device-side decode operator initialization,
 * specialized for sample types larger than 8-bit.
 * This variant initializes the decode operators inside the kernel from level bounds.
 *
 * @param d_temp_storage
 *   Device-accessible allocation of temporary storage.
 *   When nullptr, the required allocation size is written to
 *   `temp_storage_bytes` and no work is done.
 *
 * @param temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param d_samples
 *   The pointer to the input sequence of sample items.
 *   The samples from different channels are assumed to be interleaved
 *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
 *
 * @param d_output_histograms
 *   The pointers to the histogram counter output arrays, one for each active channel.
 *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
 *   `num_output_levels[i] - 1`.
 *
 * @param num_output_levels
 *   The number of bin level boundaries for delineating histogram samples in each active channel.
 *   Implies that the number of bins for channel<sub><em>i</em></sub> is
 *   `num_output_levels[i] - 1`.
 *
 * @param lower_level
 *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
 *
 * @param upper_level
 *   The upper sample value bound (exclusive) for the highest histogram bin in each active
 * channel.
 *
 * @param num_row_pixels
 *   The number of multi-channel pixels per row in the region of interest
 *
 * @param num_rows
 *   The number of rows in the region of interest
 *
 * @param row_stride_samples
 *   The number of samples between starts of consecutive rows in the region of interest
 *
 * @param stream
 *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
 *
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename LowerLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>,
  typename UpperLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t __dispatch_even_device_init(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  LowerLevelArrayT lower_level,
  UpperLevelArrayT upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::false_type /*is_byte_sample*/,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  int max_levels = num_output_levels[0];

  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    int num_levels = num_output_levels[channel];
    if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
    {
      // Make sure to also return a reasonable value for `temp_storage_bytes` in case of
      // an overflow of the bin computation, in which case a subsequent algorithm
      // invocation will also fail
      if (!d_temp_storage)
      {
        temp_storage_bytes = 1U;
      }
      return cudaErrorInvalidValue;
    }

    if (num_levels > max_levels)
    {
      max_levels = num_levels;
    }
  }
  int max_num_output_bins = max_levels - 1;

  if (max_num_output_bins > detail::histogram::max_privatized_smem_bins)
  {
    // Dispatch shared-privatized approach
    constexpr int PRIVATIZED_SMEM_BINS = 0;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            PRIVATIZED_SMEM_BINS,
            true, // IsDeviceInit
            true, // IsEven
            false // IsByteSample
            >(d_temp_storage,
              temp_storage_bytes,
              d_samples,
              d_output_histograms,
              num_output_levels,
              num_output_levels,
              upper_level,
              lower_level,
              max_num_output_bins,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              stream,
              policy_selector,
              kernel_source,
              launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    // Dispatch shared-privatized approach
    constexpr int PRIVATIZED_SMEM_BINS = detail::histogram::max_privatized_smem_bins;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            PRIVATIZED_SMEM_BINS,
            true, // IsDeviceInit
            true, // IsEven
            false // IsByteSample
            >(d_temp_storage,
              temp_storage_bytes,
              d_samples,
              d_output_histograms,
              num_output_levels,
              num_output_levels,
              upper_level,
              lower_level,
              max_num_output_bins,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              stream,
              policy_selector,
              kernel_source,
              launcher_factory))))
    {
      return error;
    }
  }

  return cudaSuccess;
}

/**
 * Dispatch routine for HistogramEven with device-side decode operator initialization,
 * specialized for 8-bit sample types
 * (computes 256-bin privatized histograms and then reduces to user-specified levels).
 * This variant initializes the decode operators inside the kernel from level bounds.
 *
 * @param d_temp_storage
 *   Device-accessible allocation of temporary storage.
 *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
 *   no work is done.
 *
 * @param temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param d_samples
 *   The pointer to the input sequence of sample items. The samples from different channels are
 *   assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of
 *   four RGBA 8-bit samples).
 *
 * @param d_output_histograms
 *   The pointers to the histogram counter output arrays, one for each active channel.
 *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
 *   `num_output_levels[i] - 1`.
 *
 * @param num_output_levels
 *   The number of bin level boundaries for delineating histogram samples in each active channel.
 *   Implies that the number of bins for channel<sub><em>i</em></sub> is
 *   `num_output_levels[i] - 1`.
 *
 * @param lower_level
 *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
 *
 * @param upper_level
 *   The upper sample value bound (exclusive) for the highest histogram bin in each active
 * channel.
 *
 * @param num_row_pixels
 *   The number of multi-channel pixels per row in the region of interest
 *
 * @param num_rows
 *   The number of rows in the region of interest
 *
 * @param row_stride_samples
 *   The number of samples between starts of consecutive rows in the region of interest
 *
 * @param stream
 *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
 *
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename LowerLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>,
  typename UpperLevelArrayT      = ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS>>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t __dispatch_even_device_init(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  LowerLevelArrayT lower_level,
  UpperLevelArrayT upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::true_type /*is_byte_sample*/,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
  int max_levels = num_output_levels[0];

  for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
  {
    num_privatized_levels[channel] = 257;

    int num_levels = num_output_levels[channel];
    if (kernel_source.MayOverflow(num_levels - 1, upper_level, lower_level, channel))
    {
      // Make sure to also return a reasonable value for `temp_storage_bytes` in case of
      // an overflow of the bin computation, in which case a subsequent algorithm
      // invocation will also fail
      if (!d_temp_storage)
      {
        temp_storage_bytes = 1U;
      }
      return cudaErrorInvalidValue;
    }

    if (num_levels > max_levels)
    {
      max_levels = num_levels;
    }
  }
  int max_num_output_bins = max_levels - 1;

  constexpr int PRIVATIZED_SMEM_BINS = 256;

  if (const auto error = CubDebug(
        (detail::histogram::dispatch<
          NUM_CHANNELS,
          NUM_ACTIVE_CHANNELS,
          PRIVATIZED_SMEM_BINS,
          true, // IsDeviceInit
          true, // IsEven
          true // IsByteSample
          >(d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_privatized_levels,
            num_output_levels,
            upper_level,
            lower_level,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
  {
    return error;
  }

  return cudaSuccess;
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_pdl_trigger(int)
  -> decltype(ActivePolicy::pdl_trigger_next_launch_in_init_kernel_max_bin_count)
{
  return ActivePolicy::pdl_trigger_next_launch_in_init_kernel_max_bin_count;
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_pdl_trigger(long)
{
  return 0;
}

// TODO(bgruber): drop in CCCL 4.0
template <typename ActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_policy() -> histogram_policy
{
  using ap = typename ActivePolicy::AgentHistogramPolicyT;
  return histogram_policy{
    ap::BLOCK_THREADS,
    ap::PIXELS_PER_THREAD,
    ap::LOAD_ALGORITHM,
    ap::LOAD_MODIFIER,
    ap::IS_RLE_COMPRESS,
    ap::MEM_PREFERENCE,
    ap::IS_WORK_STEALING,
    ap::VEC_SIZE,
    convert_pdl_trigger<ActivePolicy>(0)};
}

// TODO(bgruber): drop in CCCL 4.0
template <typename MaxPolicy>
struct policy_selector_from_max_policy
{
private:
  struct extract_policy_dispatch_t
  {
    histogram_policy& policy;

    template <typename ActivePolicyT>
    _CCCL_HOST_DEVICE_API constexpr cudaError_t Invoke()
    {
      policy = convert_policy<ActivePolicyT>();
      return cudaSuccess;
    }
  };

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> histogram_policy
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      ({
                        histogram_policy policy{};
                        extract_policy_dispatch_t dispatch{policy};
                        MaxPolicy::Invoke(cc.get() * 10, dispatch);
                        return policy;
                      }),
                      ({ return convert_policy<typename MaxPolicy::ActivePolicy>(); }));
  }
};

template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  bool IsByteSample,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_range(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::bool_constant<IsByteSample>,
  PolicySelector policy_selector,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  if constexpr (IsByteSample)
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the pass-thru transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

    // Use the search transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      num_privatized_levels[channel] = 257;
      output_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);

      if (num_output_levels[channel] > max_levels)
      {
        max_levels = num_output_levels[channel];
      }
    }
    int max_num_output_bins = max_levels - 1;

    constexpr int PRIVATIZED_SMEM_BINS = 256;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<
            NUM_CHANNELS,
            NUM_ACTIVE_CHANNELS,
            PRIVATIZED_SMEM_BINS,
            false, // IsDeviceInit
            false, // IsEven (unused for host-init)
            false // IsByteSample (unused for host-init)
            >(d_temp_storage,
              temp_storage_bytes,
              d_samples,
              d_output_histograms,
              num_privatized_levels,
              num_output_levels,
              output_decode_op,
              privatized_decode_op,
              max_num_output_bins,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              stream,
              policy_selector,
              kernel_source,
              launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the search transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

    // Use the pass-thru transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::PassThruTransform;

    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      privatized_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);
      if (num_output_levels[channel] > max_levels)
      {
        max_levels = num_output_levels[channel];
      }
    }
    int max_num_output_bins = max_levels - 1;

    // Sort-based fast path for very-large-bin single-channel single-row inputs.
    if constexpr (NUM_CHANNELS == 1 && NUM_ACTIVE_CHANNELS == 1)
    {
      if (use_sort_path(num_row_pixels, max_num_output_bins)
          && num_rows == 1
          && static_cast<size_t>(num_row_pixels) == static_cast<size_t>(row_stride_samples))
      {
        return dispatch_sort_based(
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms[0],
          privatized_decode_op[0],
          output_decode_op[0],
          max_num_output_bins,
          max_num_output_bins,
          num_row_pixels,
          stream);
      }
    }
    // Sort-based fast path for very-large-bin multi-channel single-row inputs.
    else if constexpr (NUM_ACTIVE_CHANNELS > 1)
    {
      // Require uniform bin counts across active channels (same num_output_levels for every
      // channel — this is true for all three histogram benchmarks (multi.even/multi.range), and
      // mixing per-channel bin counts would force a wider key encoding).
      bool uniform_bins = true;
      for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        if (num_output_levels[channel] != num_output_levels[0])
        {
          uniform_bins = false;
          break;
        }
      }
      const int per_channel_bins = num_output_levels[0] - 1;
      // For num_rows == 1, the row stride doesn't matter (only one row), so we don't need to
      // constrain row_stride_samples. The benchmark passes row_stride_samples = num_row_pixels
      // even for multi-channel inputs, which doesn't match the contiguous-multi-channel layout
      // formula (num_row_pixels * NUM_CHANNELS) — but with a single row it's still safe to read
      // samples 0 .. num_row_pixels * NUM_CHANNELS - 1.
      if (uniform_bins
          && use_sort_path_multi_channel(num_row_pixels, per_channel_bins)
          && num_rows == 1)
      {
        ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_per_channel;
        for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
          num_output_bins_per_channel[channel] = num_output_levels[channel] - 1;
        }
        return dispatch_sort_based_multi_channel<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms,
          privatized_decode_op,
          output_decode_op,
          per_channel_bins,
          num_output_bins_per_channel,
          num_row_pixels,
          stream);
      }
    }

    // Dispatch
    if (max_num_output_bins > max_privatized_smem_bins)
    {
      // Too many bins to keep in shared memory.
      constexpr int PRIVATIZED_SMEM_BINS = 0;

      if (const auto error = CubDebug(
            (detail::histogram::dispatch<
              NUM_CHANNELS,
              NUM_ACTIVE_CHANNELS,
              PRIVATIZED_SMEM_BINS,
              false, // IsDeviceInit
              false, // IsEven (unused for host-init)
              false // IsByteSample (unused for host-init)
              >(d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_output_histograms,
                num_output_levels,
                num_output_levels,
                output_decode_op,
                privatized_decode_op,
                max_num_output_bins,
                num_row_pixels,
                num_rows,
                row_stride_samples,
                stream,
                policy_selector,
                kernel_source,
                launcher_factory))))
      {
        return error;
      }
    }
    else
    {
      // Dispatch shared-privatized approach
      constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;

      if (const auto error = CubDebug(
            (detail::histogram::dispatch<
              NUM_CHANNELS,
              NUM_ACTIVE_CHANNELS,
              PRIVATIZED_SMEM_BINS,
              false, // IsDeviceInit
              false, // IsEven (unused for host-init)
              false // IsByteSample (unused for host-init)
              >(d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_output_histograms,
                num_output_levels,
                num_output_levels,
                output_decode_op,
                privatized_decode_op,
                max_num_output_bins,
                num_row_pixels,
                num_rows,
                row_stride_samples,
                stream,
                policy_selector,
                kernel_source,
                launcher_factory))))
      {
        return error;
      }
    }
  }

  return cudaSuccess;
}

template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  bool IsByteSample,
  typename PolicySelector,
  typename SampleT = it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource =
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch_even(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  SampleIteratorT d_samples,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
  ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
  ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
  OffsetT num_row_pixels,
  OffsetT num_rows,
  OffsetT row_stride_samples,
  cudaStream_t stream,
  ::cuda::std::bool_constant<IsByteSample>,
  PolicySelector policy_selector,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  if constexpr (IsByteSample)
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the pass-thru transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

    // Use the scale transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::ScaleTransform;

    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      num_privatized_levels[channel] = 257;

      int num_levels = num_output_levels[channel];
      if (kernel_source.MayOverflow(static_cast<CommonT>(num_levels - 1), upper_level, lower_level, channel))
      {
        if (!d_temp_storage)
        {
          temp_storage_bytes = 1U;
        }
        return cudaErrorInvalidValue;
      }

      output_decode_op[channel].Init(num_levels, upper_level[channel], lower_level[channel]);

      if (num_levels > max_levels)
      {
        max_levels = num_levels;
      }
    }
    int max_num_output_bins = max_levels - 1;

    constexpr int PRIVATIZED_SMEM_BINS = 256;

    if (const auto error = CubDebug(
          (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_privatized_levels,
            num_output_levels,
            output_decode_op,
            privatized_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            policy_selector,
            kernel_source,
            launcher_factory))))
    {
      return error;
    }
  }
  else
  {
    using TransformsT = Transforms<LevelT, OffsetT, SampleT>;

    // Use the scale transform op for converting samples to privatized bins
    using PrivatizedDecodeOpT = typename TransformsT::ScaleTransform;

    // Use the pass-thru transform op for converting privatized bins to output bins
    using OutputDecodeOpT = typename TransformsT::PassThruTransform;

    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
    int max_levels = num_output_levels[0];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      int num_levels = num_output_levels[channel];
      if (kernel_source.MayOverflow(static_cast<CommonT>(num_levels - 1), upper_level, lower_level, channel))
      {
        if (!d_temp_storage)
        {
          temp_storage_bytes = 1U;
        }
        return cudaErrorInvalidValue;
      }

      privatized_decode_op[channel].Init(num_levels, upper_level[channel], lower_level[channel]);

      if (num_levels > max_levels)
      {
        max_levels = num_levels;
      }
    }
    int max_num_output_bins = max_levels - 1;

    // Sort-based fast path for very-large-bin single-channel single-row inputs.
    if constexpr (NUM_CHANNELS == 1 && NUM_ACTIVE_CHANNELS == 1)
    {
      if (use_sort_path(num_row_pixels, max_num_output_bins)
          && num_rows == 1
          && static_cast<size_t>(num_row_pixels) == static_cast<size_t>(row_stride_samples))
      {
        return dispatch_sort_based(
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms[0],
          privatized_decode_op[0],
          output_decode_op[0],
          max_num_output_bins,
          max_num_output_bins,
          num_row_pixels,
          stream);
      }
    }
    // Sort-based fast path for very-large-bin multi-channel single-row inputs.
    else if constexpr (NUM_ACTIVE_CHANNELS > 1)
    {
      bool uniform_bins = true;
      for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        if (num_output_levels[channel] != num_output_levels[0])
        {
          uniform_bins = false;
          break;
        }
      }
      const int per_channel_bins = num_output_levels[0] - 1;
      // For num_rows == 1, the row stride doesn't matter (only one row), so we don't need to
      // constrain row_stride_samples. The benchmark passes row_stride_samples = num_row_pixels
      // even for multi-channel inputs, which doesn't match the contiguous-multi-channel layout
      // formula (num_row_pixels * NUM_CHANNELS) — but with a single row it's still safe to read
      // samples 0 .. num_row_pixels * NUM_CHANNELS - 1.
      if (uniform_bins
          && use_sort_path_multi_channel(num_row_pixels, per_channel_bins)
          && num_rows == 1)
      {
        ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_per_channel;
        for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
          num_output_bins_per_channel[channel] = num_output_levels[channel] - 1;
        }
        return dispatch_sort_based_multi_channel<NUM_ACTIVE_CHANNELS, NUM_CHANNELS>(
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms,
          privatized_decode_op,
          output_decode_op,
          per_channel_bins,
          num_output_bins_per_channel,
          num_row_pixels,
          stream);
      }
    }

    if (max_num_output_bins > max_privatized_smem_bins)
    {
      constexpr int PRIVATIZED_SMEM_BINS = 0;

      if (const auto error = CubDebug(
            (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
              d_temp_storage,
              temp_storage_bytes,
              d_samples,
              d_output_histograms,
              num_output_levels,
              num_output_levels,
              output_decode_op,
              privatized_decode_op,
              max_num_output_bins,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              stream,
              policy_selector,
              kernel_source,
              launcher_factory))))
      {
        return error;
      }
    }
    else
    {
      constexpr int PRIVATIZED_SMEM_BINS = max_privatized_smem_bins;

      if (const auto error = CubDebug(
            (detail::histogram::dispatch<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, PRIVATIZED_SMEM_BINS, false, false, false>(
              d_temp_storage,
              temp_storage_bytes,
              d_samples,
              d_output_histograms,
              num_output_levels,
              num_output_levels,
              output_decode_op,
              privatized_decode_op,
              max_num_output_bins,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              stream,
              policy_selector,
              kernel_source,
              launcher_factory))))
      {
        return error;
      }
    }
  }

  return cudaSuccess;
}
} // namespace detail::histogram

/******************************************************************************
 * Dispatch
 ******************************************************************************/

// TODO(bgruber): deprecate once we make the tuning API public and remove in CCCL 4.0
/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data (may be greater than the number of channels
 *   being actively histogrammed)
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam LevelT
 *   Type for specifying bin level boundaries
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <
  int NUM_CHANNELS,
  int NUM_ACTIVE_CHANNELS,
  typename SampleIteratorT,
  typename CounterT,
  typename LevelT,
  typename OffsetT,
  typename PolicyHub    = void, // if user passes a custom Policy this should not be void
  typename SampleT      = cub::detail::it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
  typename KernelSource = detail::histogram::
    DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, LevelT, OffsetT, SampleT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHistogram
{
  static_assert(NUM_CHANNELS <= 4, "Histograms only support up to 4 channels");
  static_assert(NUM_ACTIVE_CHANNELS <= NUM_CHANNELS,
                "Active channels must be at most the number of total channels of the input samples");

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // Default (host-init) dispatch entrypoints
  // These methods initialize decode operators on the host before kernel launch.
  //---------------------------------------------------------------------

  /**
   * Dispatch routine for HistogramRange with host-side decode operator initialization.
   * This variant initializes the decode operators on the host before kernel launch.
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
   *   no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the multi-channel input sequence of data samples.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of boundaries (levels) for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param d_levels
   *   The pointers to the arrays of boundaries (levels), one for each active channel.
   *   Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are
   *   inclusive and upper sample value boundaries are exclusive.
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ false>,
              PolicyHub>::MaxPolicy,
            bool IsByteSample>
  CUB_RUNTIME_FUNCTION static cudaError_t DispatchRange(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::bool_constant<IsByteSample> is_byte_sample,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    [[maybe_unused]] MaxPolicyT max_policy = {})
  {
    return detail::histogram::dispatch_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      d_levels,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      is_byte_sample,
      detail::histogram::policy_selector_from_max_policy<MaxPolicyT>{},
      kernel_source,
      launcher_factory);
  }

  /**
   * Dispatch routine for HistogramEven with host-side decode operator initialization.
   * This variant initializes the decode operators on the host before kernel launch.
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the input sequence of sample items.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of bin level boundaries for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param lower_level
   *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
   *
   * @param upper_level
   *   The upper sample value bound (exclusive) for the highest histogram bin in each active
   * channel.
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   *
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ true>,
              PolicyHub>::MaxPolicy,
            bool IsByteSample>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchEven(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::bool_constant<IsByteSample> is_byte_sample,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    [[maybe_unused]] MaxPolicyT max_policy = {})
  {
    return detail::histogram::dispatch_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_output_histograms,
      num_output_levels,
      lower_level,
      upper_level,
      num_row_pixels,
      num_rows,
      row_stride_samples,
      stream,
      is_byte_sample,
      detail::histogram::policy_selector_from_max_policy<MaxPolicyT>{},
      kernel_source,
      launcher_factory);
  }
};

CUB_NAMESPACE_END
