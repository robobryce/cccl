// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
// ---------------------------------------------------------------------------
// Device-tuning constants for the high-bin direct-atomic cache.
//
// These are the hardware-shaped knobs for the per-block SMEM cache that the
// direct-atomic-to-output kernels use to absorb cross-block contention. They
// live here, in the tuning header, so the dispatch sizing logic and the kernel
// bodies read ONE definition instead of repeating the literals (the count-replica
// factor in particular was previously duplicated across the host sizer and both
// direct-atomic kernels and had to be kept in sync by hand). Defaults are tuned
// for SM90/SM100 (B200) and can be revisited per architecture.
struct cache_tuning
{
  // Count-array replica factor for multi-active-channel direct-atomic caches.
  // The per-slot count array is split into this many warp-strided replicas so
  // cross-warp atomicAdd_block traffic on a hot slot is de-serialised; single
  // channel uses 1 (no replication, byte-identical to an unreplicated cache).
  // Kept compile-time (a runtime replica factor regressed the register-pinned
  // multi-channel kernel), so callers select via `replicas(num_active_channels)`.
  static constexpr int multi_channel_count_replicas = 4;

  _CCCL_HOST_DEVICE static constexpr int replicas(int num_active_channels)
  {
    return num_active_channels > 1 ? multi_channel_count_replicas : 1;
  }

  // Per-channel SMEM cache slot floor (power of two). The dispatch grows the
  // slot count above this only while it stays free of occupancy cost. Single
  // channel affords a larger floor than multi (which pays per active channel).
  static constexpr int slots_floor_single_channel = 4096;
  static constexpr int slots_floor_multi_channel  = 1024;

  _CCCL_HOST_DEVICE static constexpr int slots_floor(int num_active_channels)
  {
    return num_active_channels == 1 ? slots_floor_single_channel : slots_floor_multi_channel;
  }

  // Per-CTA opt-in dynamic-SMEM budget for the cache. The dispatch queries the
  // device's cudaDevAttrMaxSharedMemoryPerBlockOptin (B200/SM100 ~228 KiB) and
  // reserves `smem_reserve_bytes` for static/driver use; if the query fails it
  // falls back to `smem_fallback_bytes`.
  static constexpr int smem_fallback_bytes = 96 * 1024;
  static constexpr int smem_reserve_bytes  = 4096;

  // Per-CTA dynamic-SMEM BYTE budget for the whole-histogram-on-chip privatized
  // kernel (the "256 < bins <= cap" dyn-SMEM tier). This is the hardware-shaped
  // knob: the maximum bin count for which the selector keeps the histogram on chip
  // is DERIVED from this budget and the per-bin counter width at runtime
  // (`max_dynamic_smem_bins(...)` below), not frozen as a bin count. A bin count is
  // the wrong unit for a hardware property -- it silently assumes a 4-byte counter
  // and one active channel. The byte budget is the real per-arch resource; bins
  // follow from it once the counter width and channel count are known.
  //
  // Default 232448 B == the B200/SM100 cudaDevAttrMaxSharedMemoryPerBlockOptin
  // (227 KiB). At launch the dispatch clamps this to the device's actual opt-in max
  // (so a smaller-SMEM arch is respected) and subtracts `smem_reserve_bytes`. For a
  // 4-byte counter, single channel, that yields ~57344 on-chip bins on B200 -- so
  // 32768 (and below) stay on chip, while 65536 (256 KiB > budget) correctly routes
  // to the high-bin path. Revisit per architecture alongside the cache budget above.
  static constexpr int max_dynamic_smem_bytes = 232448;

  // Per-CTA dynamic-SMEM BYTE budget for the single-channel hybrid kernel's PRIMARY
  // (on-chip) bin range. Like max_dynamic_smem_bytes, this is the hardware-shaped knob:
  // the hybrid's primary split (how many low bins stay in dyn-SMEM) is DERIVED from
  // this budget and the per-bin counter width at runtime (`hybrid_smem_split_bins(...)`
  // below), not frozen as a bin count. The old frozen `49152` baked in a 4-byte counter
  // (49152 * 4 B = 192 KiB); with an 8-byte counter that same bin count needs 384 KiB
  // and exceeds the per-CTA opt-in cap, so the hybrid launch failed (cudaErrorNotSupported)
  // and 64-bit-counter histograms at the hybrid tiers crashed. Deriving the split in bytes
  // keeps a (smaller) working hybrid for wide counters instead of forfeiting it.
  // 192 KiB leaves headroom under the B200 227 KiB opt-in cap for static/driver SMEM.
  static constexpr int hybrid_smem_split_bytes = 192 * 1024;
};

// Max PRIMARY bins the single-channel hybrid kernel can stage in dyn-SMEM for a given
// counter width, derived from cache_tuning::hybrid_smem_split_bytes (clamped to the
// device opt-in). Mirrors max_dynamic_smem_bins. The caller further clamps to
// max_num_output_bins - 1 (the split must leave a non-empty GMEM secondary tail).
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
hybrid_smem_split_bins(int counter_bytes, int num_active_channels, int device_optin_smem_bytes = 0)
{
  const int tuned_bytes   = cache_tuning::hybrid_smem_split_bytes;
  const int reserve_bytes = cache_tuning::smem_reserve_bytes;
  const int hw_bytes      = device_optin_smem_bytes > 0 ? device_optin_smem_bytes : tuned_bytes;
  const int capped_bytes  = hw_bytes < tuned_bytes ? hw_bytes : tuned_bytes;
  const int budget        = capped_bytes - reserve_bytes;
  const int per_bin       = counter_bytes * num_active_channels;
  return (budget > 0 && per_bin > 0) ? budget / per_bin : 0;
}

// Maximum number of bins the whole-histogram-on-chip privatized kernel can hold per
// channel for a given per-bin counter width, derived from `cache_tuning`'s byte
// budget. This is the runtime routing cap the selector uses to decide whether a
// histogram stays on chip; it replaces the old frozen `max_dynamic_smem_bins`
// integer (which baked in a 4-byte counter and one channel).
//
// `device_optin_smem_bytes` is the device's queried cudaDevAttrMaxSharedMemoryPerBlockOptin
// (0 if the query failed -> fall back to the tuned budget). The on-chip budget is the
// smaller of the tuned byte budget and what the hardware actually offers, minus the
// static/driver reserve; the bin cap is that budget divided by the per-CTA footprint
// of one bin across all active channels (counter_bytes * num_active_channels).
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int
max_dynamic_smem_bins(int counter_bytes, int num_active_channels, int device_optin_smem_bytes = 0)
{
  // Copy the static constexpr members into locals (by value) before doing any
  // arithmetic: passing a `static constexpr` member by reference (e.g. to
  // ::cuda::std::min) ODR-uses it, which requires an out-of-line definition and
  // otherwise fails to link / compile in device code. Plain integer ops are by-value.
  const int tuned_bytes   = cache_tuning::max_dynamic_smem_bytes;
  const int reserve_bytes = cache_tuning::smem_reserve_bytes;
  const int hw_bytes      = device_optin_smem_bytes > 0 ? device_optin_smem_bytes : tuned_bytes;
  const int capped_bytes  = hw_bytes < tuned_bytes ? hw_bytes : tuned_bytes;
  const int budget        = capped_bytes - reserve_bytes;
  const int per_bin       = counter_bytes * num_active_channels;
  return (budget > 0 && per_bin > 0) ? budget / per_bin : 0;
}

// TODO(bgruber): drop in CCCL 4.0
enum class primitive_sample
{
  no,
  yes
};

// TODO(bgruber): drop in CCCL 4.0
enum class sample_size
{
  _1,
  _2,
  _4,
  _8,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
enum class counter_size
{
  _4,
  unknown
};

// TODO(bgruber): drop in CCCL 4.0
template <class T>
_CCCL_HOST_DEVICE_API constexpr primitive_sample is_primitive_sample()
{
  return is_primitive<T>::value ? primitive_sample::yes : primitive_sample::no;
}

// TODO(bgruber): drop in CCCL 4.0
template <class CounterT>
_CCCL_HOST_DEVICE_API constexpr counter_size classify_counter_size()
{
  return sizeof(CounterT) == 4 ? counter_size::_4 : counter_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT>
_CCCL_HOST_DEVICE_API constexpr sample_size classify_sample_size()
{
  return sizeof(SampleT) == 1   ? sample_size::_1
       : sizeof(SampleT) == 2   ? sample_size::_2
       : sizeof(SampleT) == 4   ? sample_size::_4
       : sizeof(SampleT) == 8   ? sample_size::_8
                                : sample_size::unknown;
}

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm90_tuning;

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  static constexpr int threads = 768;
  static constexpr int items   = 12;

  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = false;
  static constexpr bool work_stealing = false;
};

template <class SampleT>
struct sm90_tuning<SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_2>
{
  static constexpr int threads = 960;
  static constexpr int items   = 10;

  static constexpr CacheLoadModifier load_modifier               = LOAD_DEFAULT;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr bool rle_compress  = true;
  static constexpr bool work_stealing = false;
};

// TODO(bgruber): drop in CCCL 4.0
template <bool IsEven,
          class SampleT,
          int NumChannels,
          int NumActiveChannels,
          counter_size CounterSize,
          primitive_sample PrimitiveSample = is_primitive_sample<SampleT>(),
          sample_size SampleSize           = classify_sample_size<SampleT>()>
struct sm100_tuning;

// even
template <class SampleT>
struct sm100_tuning<true, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 928;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_CA;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// sample_size 2/4/8: no SM100 specialization beat the inherited SM90 tuning

// range
template <class SampleT>
struct sm100_tuning<false, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_1>
{
  // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 448;
  static constexpr bool rle_compress                             = false;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_DIRECT;
  static constexpr int vec_size                                  = 1 << 2;
};

// SM100 sample_size 4 (I32) single-channel non-byte tuning. The default Policy500 fallback
// {384 threads, t_scale(16)=16 ipt} is suboptimal for the dyn-SMEM 16384-bin tier where
// SMEM atomicAdd_block contention dominates. Use 768 threads / 12 ipt (matching the SM90
// sample_size=1 shape) to spread atomic contention across more concurrent issues per CTA,
// with LOAD_LDG (streaming) loads.
template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_4>
{
  static constexpr int items                                     = 12;
  static constexpr int threads                                   = 768;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_VECTORIZE;
  static constexpr int vec_size                                  = 1 << 2;
};

// SM100 sample_size 8 (F64) single-channel non-byte tuning. F64 has half the throughput per
// byte and the dyn-SMEM 16384 tier is already bandwidth-saturated, so aim for a balanced
// {threads, ipt} that does not regress the lower-bin tiers. 512 threads / 8 ipt trades a small
// even-path cost for a larger range-path gain over the wider 768-thread shape.
template <bool IsEven, class SampleT>
struct sm100_tuning<IsEven, SampleT, 1, 1, counter_size::_4, primitive_sample::yes, sample_size::_8>
{
  static constexpr int items                                     = 8;
  static constexpr int threads                                   = 512;
  static constexpr bool rle_compress                             = true;
  static constexpr bool work_stealing                            = false;
  static constexpr BlockHistogramMemoryPreference mem_preference = SMEM;
  static constexpr CacheLoadModifier load_modifier               = LOAD_LDG;
  static constexpr BlockLoadAlgorithm load_algorithm             = BLOCK_LOAD_VECTORIZE;
  static constexpr int vec_size                                  = 1 << 2;
};

// multi.even and multi.range: no SM100 specialization beat the inherited SM90 tuning

// TODO(bgruber): drop in CCCL 4.0
template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_hub
{
  // TODO(bgruber): move inside t_scale in C++14
  static constexpr int v_scale = (sizeof(SampleT) + sizeof(int) - 1) / sizeof(int);

  _CCCL_HOST_DEVICE_API static constexpr int t_scale(int nominalItemsPerThread)
  {
    return (::cuda::std::max) (nominalItemsPerThread / NumActiveChannels / v_scale, 1);
  }

  // SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    // TODO This might be worth it to separate usual histogram and the multi one
    using AgentHistogramPolicyT =
      AgentHistogramPolicy<384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false>;
  };

  // SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy500
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy500::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm90_tuning<SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(int)
      -> AgentHistogramPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              Tuning::rle_compress,
                              Tuning::mem_preference,
                              Tuning::work_stealing,
                              Tuning::vec_size>;

    template <typename Tuning>
    _CCCL_HOST_DEVICE_API static auto select_agent_policy(long) -> typename Policy900::AgentHistogramPolicyT;

    using AgentHistogramPolicyT =
      decltype(select_agent_policy<
               sm100_tuning<IsEven, SampleT, NumChannels, NumActiveChannels, histogram::classify_counter_size<CounterT>()>>(
        0));

    static constexpr int pdl_trigger_next_launch_in_init_kernel_max_bin_count = 2048;
  };

  using MaxPolicy = Policy1000;
};

struct histogram_policy
{
  int threads_per_block;
  int pixels_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool rle_compress;
  BlockHistogramMemoryPreference mem_preference;
  bool work_stealing;
  int vec_size;
  int pdl_trigger_next_launch_in_init_kernel_max_bin_count;
  // Thread count for the high-bin DIRECT-ATOMIC kernels (the cuckoo and
  // single-probe persistent kernels that atomic straight to the output rather
  // than running the SMEM-privatized sweep). 0 means "inherit
  // `threads_per_block`" (the default for every policy that does not override
  // it). These kernels distribute work via a pure grid-stride loop, so any
  // block size is correct; decoupling their thread count from the SMEM-priv
  // sweep's lets a policy give the latency-/SMEM-throughput-bound direct-atomic
  // path its own launch shape (e.g. a smaller block so more, smaller blocks are
  // co-resident, spreading the dynamic-SMEM cache and shortening each block's
  // atomic dependency chain) without disturbing the sweep tiers that share the
  // same policy.
  int direct_atomic_threads_per_block = 0;

  // Whether the GMEM-privatized / direct-atomic paths coalesce a warp's same-bin
  // lanes into one atomic (the dual of `rle_compress`; see warp_coalesce_atomic).
  // On by default -- the contention win on the hot-bin direct-atomic path;
  // exposed here so the behavior is controlled in one place. Trailing + defaulted
  // so positional policy braces above (which stop at direct_atomic_threads) are
  // unaffected.
  //
  // The default honors a build-time override, `CUB_HISTO_FORCE_WARP_COALESCE`
  // (1 = force on, 0 = force off), so a study/sweep can build a coalesce-off
  // variant with `-DCUB_HISTO_FORCE_WARP_COALESCE=0` without editing source --
  // mirroring the TUNE_* benchmark macros. Coalescing merges a warp's same-bin
  // lanes into one atomic: a large win on low-entropy / colliding inputs but pure
  // `__match_any_sync` overhead on high-entropy / scattered inputs (where it
  // merges nothing yet still pays the warp-collective latency). Unset (the normal
  // build) leaves it on.
#ifdef CUB_HISTO_FORCE_WARP_COALESCE
  bool warp_coalesce = (CUB_HISTO_FORCE_WARP_COALESCE != 0);
#else
  bool warp_coalesce = true;
#endif

  // Resolved direct-atomic thread count: the override when set, else the sweep
  // thread count. Used by both the host dispatch launch and the direct-atomic
  // kernels' __launch_bounds__.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int direct_atomic_threads() const
  {
    return direct_atomic_threads_per_block != 0 ? direct_atomic_threads_per_block : threads_per_block;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const histogram_policy& lhs, const histogram_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.pixels_per_thread == rhs.pixels_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.rle_compress == rhs.rle_compress && lhs.mem_preference == rhs.mem_preference
        && lhs.work_stealing == rhs.work_stealing && lhs.vec_size == rhs.vec_size
        && lhs.pdl_trigger_next_launch_in_init_kernel_max_bin_count
             == rhs.pdl_trigger_next_launch_in_init_kernel_max_bin_count
        && lhs.direct_atomic_threads_per_block == rhs.direct_atomic_threads_per_block
        && lhs.warp_coalesce == rhs.warp_coalesce;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const histogram_policy& lhs, const histogram_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const histogram_policy& p)
  {
    return os
        << "histogram_policy { .threads_per_block = " << p.threads_per_block
        << ", .pixels_per_thread = " << p.pixels_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rle_compress = " << p.rle_compress
        << ", .mem_preference = " << p.mem_preference << ", .work_stealing = " << p.work_stealing
        << ", .vec_size = " << p.vec_size << ", .pdl_trigger_next_launch_in_init_kernel_max_bin_count = "
        << p.pdl_trigger_next_launch_in_init_kernel_max_bin_count
        << ", .direct_atomic_threads_per_block = " << p.direct_atomic_threads_per_block
        << ", .warp_coalesce = " << p.warp_coalesce << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept histogram_policy_selector = policy_selector<T, histogram_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  bool sample_is_primitive;
  int sample_size;
  int counter_size;
  int sample_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int t_scale(int nominal_items_per_thread) const
  {
    const int sample_scale = (sample_size_bytes + int{sizeof(int)} - 1) / int{sizeof(int)};
    return (::cuda::std::max) (nominal_items_per_thread / num_active_channels / sample_scale, 1);
  }

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> histogram_policy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive && sample_size == 1)
      {
        if (is_even)
        {
          // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2 1.033332  0.940517  1.031835  1.195876
          return histogram_policy{928, 12, BLOCK_LOAD_DIRECT, LOAD_CA, false, SMEM, false, 1 << 2, 2048};
        }
        else
        {
          // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2 1.078987  0.985542  1.085118  1.175637
          return histogram_policy{448, 12, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 1 << 2, 2048};
        }
      }

      // SM100 single-channel NON-byte tuning (I32 sample_size==4, F64 sample_size==8).
      //
      // Before this arm, single-channel I32/F64 fell all the way through to the SM50
      // Policy500 fallback {384 threads, t_scale(16) ipt, rle=true, LDG} -- a launch
      // shape never tuned for SM100. (The byte-sample arm above only matches
      // sample_size==1, and the SM90 arm below only matches sample_size 1/2; the
      // sm100_tuning sample_size _4/_8 structs are read ONLY by
      // policy_selector_from_max_policy, which DeviceHistogram's env API does not use --
      // the benchmark drives policy_selector_from_types, i.e. THIS struct.)
      //
      // This non-byte single-channel policy drives the SMEM-priv sweep AND the high-bin
      // direct-atomic (cuckoo / single-probe) kernels. A wider launch than 384 threads
      // gives those kernels more resident warps to hide SMEM-atomic and SearchTransform
      // latency, mirroring the multi-channel SM100 arm below. EVEN and RANGE are
      // decoupled per transform (RANGE's per-sample binary search is latency-heavier).
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive
          && (sample_size == 4 || sample_size == 8))
      {
        if (is_even)
        {
          // EVEN: 768 threads. I32 -> 12 ipt, F64 -> 6 ipt (t_scale(12)).
          return histogram_policy{768, t_scale(12), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 1 << 2, 2048};
        }
        else
        {
          // RANGE: 768 threads for the SMEM-priv sweep tiers (bins 64/2000/16384).
          //
          // direct_atomic_threads_per_block=512: the high-bin direct-atomic
          // (cuckoo / single-probe) single-channel RANGE cells run a SEPARATE
          // kernel from the SMEM-priv sweep, and atomic straight to the output via
          // a pure grid-stride loop, so their block size is decoupled from the
          // sweep without affecting correctness. They are latency-bound on the
          // SMEM-cache atomic dependency at exhausted issue throughput, so a
          // narrower 512-thread block (vs the 768-thread sweep) admits more
          // co-resident blocks per SM -- each with its own dynamic-SMEM cache
          // partition and a shorter per-block CAS/atomicAdd_block dependency chain
          // -- which hides that latency. Going narrower than 512 starves the issue
          // pipeline; 768 caps occupancy. This differs from the multi-channel
          // RANGE path (which wants an even narrower block) because single-channel
          // has one SearchTransform and a larger cache, making it more
          // throughput-bound, so the per-transform decouple is per-channel-count.
          //
          // The sweep tiers keep the 768-thread shape (SMEM-priv occupancy-bound,
          // not direct-atomic latency-bound). EVEN inherits its sweep thread count
          // (override stays 0): its cheap ScaleTransform makes the high-bin
          // direct-atomic cells throughput-bound, where the wider block is fine.
          return histogram_policy{768, t_scale(12), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 1 << 2, 2048, 512};
        }
      }

      // sample_size 2: no SM100 specialization beat the inherited SM90 tuning

      // SM100 multi-channel (num_channels >= 2) tuning, decoupled per transform.
      // Previously every multi-channel configuration fell through to the SM50
      // Policy500 fallback {384 threads, t_scale(16) ipt, rle=true} -- a launch
      // shape never tuned for SM90/SM100. That shape is genuinely strong for the
      // EVEN path (cheap ScaleTransform classify => the SMEM-priv tiers are
      // contention-bound on shared-memory atomics, where intra-thread RLE
      // compression of same-bin runs plus a modest 384-thread launch minimise
      // atomic pressure), so EVEN keeps the fallback shape verbatim. The RANGE
      // path is different: its per-sample SearchTransform binary search over the
      // level boundaries is latency-heavy and paid per active channel, so the
      // sweep is classify-bound rather than atomic-bound and a wider launch hides
      // that latency. We therefore give RANGE a wider 768-thread shape while
      // keeping rle=true (free when runs are absent, e.g. uniform entropy).
      if (num_channels >= 2 && counter_size == 4 && sample_is_primitive)
      {
        if (!is_even)
        {
          // RANGE: 1024 threads. On the SMEM-priv mid-bin sweep cells 768 and 1024
          // are within noise, but this policy ALSO drives the high-bin
          // direct-atomic (cuckoo/single-probe) kernels, which atomic directly to
          // the output and are GMEM-atomic / classify-latency bound rather than
          // SMEM-priv occupancy bound; the wider 1024-thread launch gives those
          // kernels more resident warps to hide that latency (matching EVEN's
          // 1024 pick). rle=true is free. Keep LOAD_LDG: LOAD_CA (cache-all)
          // thrashes L1/L2 caching the SearchTransform level-array + sample loads
          // across the active channels, while LDG's streaming loads avoid the
          // eviction churn.
          //
          // direct_atomic_threads_per_block stays 0 (the direct-atomic kernels
          // inherit 1024): a narrower direct-atomic block was found antagonistic
          // with the multi-channel count-replica footprint -- the larger per-slot
          // count plus a narrow launch together starve the cache -- so the
          // multi-channel direct-atomic kernels keep the 1024-thread shape and the
          // count-replica de-serialization instead.
          return histogram_policy{1024, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 4, 0, 0};
        }
        else
        {
          // EVEN: 1024 threads. At t_scale(16) the per-thread accumulate holds
          // samples[pixels_per_thread][NumChannels] + bins[pixels_per_thread],
          // which compiles to enough registers to pin the contention-bound
          // SMEM-priv even sweep to 1 block/SM -- leaving no second block to hide
          // the shared-memory atomicAdd latency it is dominated by.
          //
          // Halve the nominal items (t_scale(8): 2 pixels/thread for I32, 1 for
          // F64). That shrinks the live samples/bins arrays enough for ptxas to
          // hold the kernel in <= 32 registers WITHOUT spilling (forcing 32 regs
          // at t_scale(16) instead spills and regresses), which together with the
          // DeviceHistogramSmemPrivatizedKernel __launch_bounds__ min-blocks=2 hint admits
          // a 2nd resident CTA on the register-limited low-bin even tiers (where
          // registers, not SMEM, gate occupancy). rle=true is load-bearing
          // (dropping the same-bin RLE coalescing collapses the multi-channel even
          // path). LOAD_CA matches the single-channel SM100 even tuning.
          return histogram_policy{1024, t_scale(8), BLOCK_LOAD_DIRECT, LOAD_CA, true, SMEM, false, 4, 0};
        }
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (num_channels == 1 && num_active_channels == 1 && counter_size == 4 && sample_is_primitive)
      {
        if (sample_size == 1)
        {
          return histogram_policy{768, 12, BLOCK_LOAD_DIRECT, LOAD_LDG, false, SMEM, false, 1 << 2, 2048};
        }
        else if (sample_size == 2)
        {
          return histogram_policy{960, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, true, SMEM, false, 1 << 2, 2048};
        }
      }
    }

    // fallback from SM50
    return histogram_policy{384, t_scale(16), BLOCK_LOAD_DIRECT, LOAD_LDG, true, SMEM, false, 4, 0};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(histogram_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class SampleT, class CounterT, int NumChannels, int NumActiveChannels, bool IsEven>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> histogram_policy
  {
    constexpr auto policies = policy_selector{
      is_primitive_v<SampleT>,
      int{sizeof(SampleT)},
      int{sizeof(CounterT)},
      int{sizeof(SampleT)},
      NumChannels,
      NumActiveChannels,
      IsEven};
    return policies(cc);
  }
};
} // namespace detail::histogram

CUB_NAMESPACE_END
